#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../common/headers/profiling_helpers.cuh"

namespace ilp_low_occ_vec4 {

__global__ void independent_ops_kernel(float* __restrict__ output,
                                       const float* __restrict__ input,
                                       int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        float val = input[i];
        float val2 = val * 2.0f;
        float val3 = val + 1.0f;
        float val4 = val * 3.0f;
        float val5 = val - 5.0f;
        output[i] = val2 + val3 + val4 + val5;
    }
}

__global__ void unrolled_ilp_kernel(float* __restrict__ output,
                                    const float* __restrict__ input,
                                    int N) {
    constexpr int VEC_WIDTH = 4;
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = vec_idx; idx * VEC_WIDTH < N; idx += stride) {
        int base_idx = idx * VEC_WIDTH;
        if (base_idx + (VEC_WIDTH - 1) < N) {
            const float4* input4 = reinterpret_cast<const float4*>(input);
            float4 vals = input4[idx];
            float4 res;
            res.x = vals.x * 2.0f + 1.0f;
            res.y = vals.y * 3.0f - 5.0f;
            res.z = vals.z * 4.0f + 2.0f;
            res.w = vals.w * 5.0f - 3.0f;
            float4* output4 = reinterpret_cast<float4*>(output);
            output4[idx] = res;
        } else {
            for (int i = 0; i < VEC_WIDTH && base_idx + i < N; ++i) {
                float val = input[base_idx + i];
                switch (i) {
                    case 0: output[base_idx + i] = val * 2.0f + 1.0f; break;
                    case 1: output[base_idx + i] = val * 3.0f - 5.0f; break;
                    case 2: output[base_idx + i] = val * 4.0f + 2.0f; break;
                    default: output[base_idx + i] = val * 5.0f - 3.0f; break;
                }
            }
        }
    }
}

float measure_kernel(
    void (*kernel)(float*, const float*, int),
    float* d_output,
    const float* d_input,
    int N,
    int threads_per_block,
    int blocks,
    cudaStream_t stream,
    const char* name
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    if (blocks <= 0) {
        blocks = (N + threads_per_block - 1) / threads_per_block;
    }
    float best_ms = FLT_MAX;
    const int iterations = 200;
    for (int rep = 0; rep < 3; ++rep) {
        kernel<<<blocks, threads_per_block, 0, stream>>>(d_output, d_input, N);
        cudaStreamSynchronize(stream);
        cudaEventRecord(start, stream);
        {
            PROFILE_KERNEL_LAUNCH(name);
            for (int i = 0; i < iterations; ++i) {
                kernel<<<blocks, threads_per_block, 0, stream>>>(d_output, d_input, N);
            }
        }
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        best_ms = std::min(best_ms, ms / iterations);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return best_ms;
}

inline int run_ilp_low_occupancy_vec4(const char* title, int max_active_blocks_override) {
    const int N = 10'000'000;
    const int threads_per_block = 256;
    int total_blocks = (N + threads_per_block - 1) / threads_per_block;
    int active_blocks = total_blocks;
    if (max_active_blocks_override > 0) {
        active_blocks = std::min(total_blocks, max_active_blocks_override);
    }

    printf("========================================\n");
    printf("%s\n", title);
    printf("========================================\n");
    printf("Problem size: %d elements\n", N);
    printf("Threads per block: %d\n", threads_per_block);
    if (max_active_blocks_override > 0) {
        printf("Active blocks per kernel: %d (capped)\n", active_blocks);
    } else {
        printf("Active blocks per kernel: %d (full occupancy)\n", active_blocks);
    }
    printf("Iterations per measurement: 200 (best-of-3)\n\n");

    float* h_input = nullptr;
    float* h_output_indep = nullptr;
    float* h_output_unrolled = nullptr;
    cudaMallocHost(&h_input, N * sizeof(float));
    cudaMallocHost(&h_output_indep, N * sizeof(float));
    cudaMallocHost(&h_output_unrolled, N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaMallocAsync(&d_input, N * sizeof(float), stream);
    cudaMallocAsync(&d_output, N * sizeof(float), stream);
    {
        PROFILE_MEMORY_COPY("H2D copy");
        cudaMemcpyAsync(d_input, h_input, N * sizeof(float),
                       cudaMemcpyHostToDevice, stream);
    }
    cudaStreamSynchronize(stream);

    printf("1. Independent operations (grid-stride scalar):\n");
    float indep_time = measure_kernel(
        independent_ops_kernel, d_output, d_input, N,
        threads_per_block, active_blocks, stream, "independent_ops");
    printf("   Time: %.3f ms\n", indep_time);
    cudaMemcpyAsync(h_output_indep, d_output, N * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    printf("\n2. Vectorized ILP (float4 loads/stores):\n");
    float unrolled_time = measure_kernel(
        unrolled_ilp_kernel, d_output, d_input, N,
        threads_per_block, active_blocks, stream, "unrolled_ilp");
    printf("   Time: %.3f ms\n", unrolled_time);
    cudaMemcpyAsync(h_output_unrolled, d_output, N * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    bool indep_correct = true;
    bool unrolled_correct = true;
    for (int i = 0; i < N && i < 1000; ++i) {
        float expected_indep = h_input[i] * 7.0f - 4.0f;
        if (fabsf(h_output_indep[i] - expected_indep) > 1e-5f) {
            indep_correct = false;
            break;
        }
        if (i < N - 3) {
            float expected_unrolled = 0.0f;
            switch (i % 4) {
                case 0: expected_unrolled = h_input[i] * 2.0f + 1.0f; break;
                case 1: expected_unrolled = h_input[i] * 3.0f - 5.0f; break;
                case 2: expected_unrolled = h_input[i] * 4.0f + 2.0f; break;
                case 3: expected_unrolled = h_input[i] * 5.0f - 3.0f; break;
            }
            if (fabsf(h_output_unrolled[i] - expected_unrolled) > 1e-5f) {
                unrolled_correct = false;
                break;
            }
        }
    }

    printf("\n========================================\n");
    printf("Results:\n");
    printf("  Independent:     %s\n", indep_correct ? "✓ Correct" : "✗ Incorrect");
    printf("  Vectorized ILP:  %s\n", unrolled_correct ? "✓ Correct" : "✗ Incorrect");
    if (indep_time > 0 && unrolled_time > 0) {
        printf("\n  Vectorized ILP:  %.2fx speedup vs scalar\n", indep_time / unrolled_time);
    }
    printf("\nKey insight: float4 vectorization and low occupancy caps increase per-thread ILP.\n");
    printf("========================================\n");

    cudaFreeAsync(d_output, stream);
    cudaFreeAsync(d_input, stream);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_output_unrolled);
    cudaFreeHost(h_output_indep);
    cudaFreeHost(h_input);

    return (indep_correct && unrolled_correct) ? 0 : 1;
}

}  // namespace ilp_low_occ_vec4

inline int run_ilp_low_occupancy_vec4(const char* title, int max_active_blocks_override) {
    return ilp_low_occ_vec4::run_ilp_low_occupancy_vec4(title, max_active_blocks_override);
}
