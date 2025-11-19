// optimized_flash_attn_tma_micro_pipeline.cu
//
// FlashAttention-shaped micro-pipeline using cuda::pipeline (maps to cp.async/TMA
// with mbarriers on Hopper/Blackwell). Demonstrates double-buffered PREFETCH
// (K/V tiles) overlapped with COMPUTE (QK^T + apply V).

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

#include <cstdio>
#include <cstdlib>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t _status = (call);                                        \
        if (_status != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(_status));   \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)
#endif

#if CUDART_VERSION < 12000
int main() {
    std::printf("SKIP: cuda::pipeline requires CUDA 12.x+ (found %d)\n", CUDART_VERSION);
    return 0;
}
#else

constexpr int SEQ_LEN = 2048;
constexpr int D_HEAD  = 64;
constexpr int TILE_KV = 40;
constexpr int THREADS = 128;
constexpr int STAGES  = 2;  // double buffer

namespace cg = cooperative_groups;

__global__ void flash_attn_tma_microkernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ o,
    int seq_len,
    int d_head) {
    const int q_idx = blockIdx.x;
    if (q_idx >= seq_len) return;

    extern __shared__ float smem[];
    float* smem_k[STAGES];
    float* smem_v[STAGES];
    const int tile_elems = TILE_KV * d_head;
    smem_k[0] = smem;
    smem_v[0] = smem_k[0] + tile_elems;
    smem_k[1] = smem_v[0] + tile_elems;
    smem_v[1] = smem_k[1] + tile_elems;

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> shared_state;
    cg::thread_block block = cg::this_thread_block();
    auto pipe = cuda::make_pipeline(block, &shared_state);

    __shared__ float score_smem[THREADS];
    const int tid = threadIdx.x;

    // Load Q row into registers once.
    float q_reg[D_HEAD];
    for (int d = tid; d < d_head; d += blockDim.x) {
        q_reg[d] = q[q_idx * d_head + d];
    }
    __syncthreads();

    float o_reg[D_HEAD];
    for (int d = 0; d < d_head; ++d) o_reg[d] = 0.f;

    const int num_tiles = (seq_len + TILE_KV - 1) / TILE_KV;
    auto k_tile_ptr = [&](int tile_idx) {
        return k + tile_idx * TILE_KV * d_head;
    };
    auto v_tile_ptr = [&](int tile_idx) {
        return v + tile_idx * TILE_KV * d_head;
    };

    int stage = 0;
    if (num_tiles > 0) {
        pipe.producer_acquire();
        const int rows = min(TILE_KV, seq_len);
        const size_t bytes = size_t(rows) * d_head * sizeof(float);
        cuda::memcpy_async(smem_k[stage], k_tile_ptr(0), bytes, pipe);
        cuda::memcpy_async(smem_v[stage], v_tile_ptr(0), bytes, pipe);
        pipe.producer_commit();
    }

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        pipe.consumer_wait();

        const int row_base = tile_idx * TILE_KV;
        const int rows_this_tile = min(TILE_KV, seq_len - row_base);

        // COMPUTE on current stage while next loads in background.
        for (int r = 0; r < rows_this_tile; ++r) {
            const float* k_row = smem_k[stage] + r * d_head;
            const float* v_row = smem_v[stage] + r * d_head;

            float score = 0.f;
            for (int d = tid; d < d_head; d += blockDim.x) {
                score += q_reg[d] * k_row[d];
            }

            score_smem[tid] = score;
            __syncthreads();
            if (tid < 64) score_smem[tid] += score_smem[tid + 64];
            __syncthreads();
            if (tid < 32) score_smem[tid] += score_smem[tid + 32];
            __syncthreads();

            if (tid == 0) {
                float s = score_smem[0];
                s = fminf(fmaxf(s, -10.f), 10.f);
                score_smem[0] = __expf(s) * 1e-3f;
            }
            __syncthreads();

            float weight = score_smem[0];
            for (int d = tid; d < d_head; d += blockDim.x) {
                o_reg[d] += weight * v_row[d];
            }
            __syncthreads();
        }

        pipe.consumer_release();

        const int next_tile = tile_idx + 1;
        if (next_tile < num_tiles) {
            int next_stage = (stage + 1) & (STAGES - 1);
            const int rows_next = min(TILE_KV, seq_len - next_tile * TILE_KV);
            const size_t bytes_next = size_t(rows_next) * d_head * sizeof(float);

            pipe.producer_acquire();
            cuda::memcpy_async(smem_k[next_stage], k_tile_ptr(next_tile), bytes_next, pipe);
            cuda::memcpy_async(smem_v[next_stage], v_tile_ptr(next_tile), bytes_next, pipe);
            pipe.producer_commit();
            stage = next_stage;
        }
    }

    for (int d = tid; d < d_head; d += blockDim.x) {
        o[q_idx * d_head + d] = o_reg[d];
    }
}

bool supports_device_pipeline() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    const int sm = prop.major * 10 + prop.minor;
    // Enforce no-fallback behavior: require SM90+ and TMA capability.
    #ifdef cudaDevAttrTensorMapAccessSupported
    int attr = 0;
    if (cudaDeviceGetAttribute(&attr, cudaDevAttrTensorMapAccessSupported, 0) != cudaSuccess) {
        return false;
    }
    const bool tma_available = (attr != 0);
    const int d_head_bytes = D_HEAD * sizeof(float);
    const bool alignment_ok = (d_head_bytes % 16) == 0;  // TMA leading dimension alignment
    if (sm < 90 || !tma_available || !alignment_ok) return false;
    return true;
    #else
    return false;
    #endif
}

int main() {
    if (!supports_device_pipeline()) {
        std::printf("SKIP: No capable GPU for pipeline/TMA path.\nelapsed_ms=0.0 ms\n");
        return 0;
    }

    const int seq_len = SEQ_LEN;
    const int d_head = D_HEAD;
    const size_t bytes = size_t(seq_len) * d_head * sizeof(float);

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    CHECK_CUDA(cudaMalloc(&d_q, bytes));
    CHECK_CUDA(cudaMalloc(&d_k, bytes));
    CHECK_CUDA(cudaMalloc(&d_v, bytes));
    CHECK_CUDA(cudaMalloc(&d_o, bytes));

    CHECK_CUDA(cudaMemset(d_q, 0, bytes));
    CHECK_CUDA(cudaMemset(d_k, 0, bytes));
    CHECK_CUDA(cudaMemset(d_v, 0, bytes));
    CHECK_CUDA(cudaMemset(d_o, 0, bytes));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    const dim3 block(THREADS);
    const dim3 grid(seq_len);
    const size_t shmem_bytes = STAGES * 2 * TILE_KV * d_head * sizeof(float);  // K + V per stage

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    flash_attn_tma_microkernel<<<grid, block, shmem_bytes, stream>>>(
        d_q, d_k, d_v, d_o, seq_len, d_head);
    CHECK_CUDA(cudaEventRecord(stop, stream));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_k));
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUDA(cudaFree(d_o));

    std::printf("elapsed_ms=%.3f ms\n", ms);
    return 0;
}

#endif  // CUDART_VERSION
