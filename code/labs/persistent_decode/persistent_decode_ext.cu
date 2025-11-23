#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <algorithm>
#include <stdexcept>

namespace {

__device__ inline float dot_tile_fallback(const float* q, const float* k, int head_dim) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        acc += q[d] * k[d];
    }
    smem[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    return smem[0];
}

__global__ void persistent_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int batch,
    int seq_len,
    int head_dim
) {
    constexpr int MAX_HEAD_DIM = 128;
    const int seq_id = blockIdx.x;
    if (seq_id >= batch) {
        return;
    }

    // shared layout: K0|K1|V0|V1|reduce
    extern __shared__ float smem_f[];
    float* smem_k0 = smem_f;
    float* smem_k1 = smem_k0 + MAX_HEAD_DIM;
    float* smem_v0 = smem_k1 + MAX_HEAD_DIM;
    float* smem_v1 = smem_v0 + MAX_HEAD_DIM;
    float* red = smem_v1 + MAX_HEAD_DIM;

    for (int t = 0; t < seq_len; ++t) {
        const float* q_ptr = q + (seq_id * seq_len + t) * head_dim;
        const float* k_ptr = k + (seq_id * seq_len + t) * head_dim;
        const float* v_ptr = v + (seq_id * seq_len + t) * head_dim;
        float* out_ptr = out + (seq_id * seq_len + t) * head_dim;

        // fallback dot + scale
        float dot = dot_tile_fallback(q_ptr, k_ptr, head_dim);
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            out_ptr[d] = v_ptr[d] * dot;
        }
        __syncthreads();
        // stash V into shared for possible future cp.async re-enable
        if (head_dim <= MAX_HEAD_DIM) {
            float* v_smem_curr = (t & 1) ? smem_v1 : smem_v0;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                v_smem_curr[d] = v_ptr[d];
            }
        }
        __syncthreads();
    }
}

} // namespace

void persistent_decode_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out, int blocks) {
    if (!q.is_cuda()) {
        throw std::runtime_error("q must be CUDA");
    }
    if (q.scalar_type() != torch::kFloat) {
        throw std::runtime_error("q must be float32");
    }
    if (!(q.sizes() == k.sizes() && q.sizes() == v.sizes())) {
        throw std::runtime_error("q/k/v shapes must match");
    }
    if (out.sizes() != q.sizes()) {
        throw std::runtime_error("out shape mismatch");
    }
    if (q.size(2) > 128) {
        throw std::runtime_error("head_dim exceeds MAX_HEAD_DIM=128");
    }

    const int batch = static_cast<int>(q.size(0));
    const int seq_len = static_cast<int>(q.size(1));
    const int head_dim = static_cast<int>(q.size(2));
    const int threads = 64;

    constexpr int MAX_HEAD_DIM = 128;
    const size_t smem_bytes = (4 * MAX_HEAD_DIM + threads) * sizeof(float);

    c10::cuda::CUDAGuard guard(q.get_device());
    cudaDeviceProp prop{};
    auto err = cudaGetDeviceProperties(&prop, q.get_device());
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaGetDeviceProperties failed");
    }
    if (prop.major < 8) {
        throw std::runtime_error("persistent_decode requires SM80+");
    }

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    const int blocks_per_batch = std::min(blocks, batch);
    persistent_decode_kernel<<<blocks_per_batch, threads, smem_bytes, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        seq_len,
        head_dim);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("persistent_decode_kernel launch failed");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("persistent_decode", &persistent_decode_cuda, "Persistent decode (CUDA)");
}
