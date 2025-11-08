// ilp_example.cu
// Demonstrates Instruction-Level Parallelism (ILP) with independent operations
//
// Key concepts:
// - ILP: Execute multiple independent instructions simultaneously
// - Independent operations can be executed in parallel by the scheduler
// - Loop unrolling can expose more ILP opportunities
// - ILP helps hide instruction latency and improve throughput

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "../common/headers/profiling_helpers.cuh"

//------------------------------------------------------
// Baseline: Sequential operations (low ILP)
// Each operation depends on the previous one
__global__ void sequential_ops_kernel(float* output, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Sequential operations - each depends on previous
        float val = input[idx];
        val = val * 2.0f;      // Op 1
        val = val + 1.0f;      // Op 2 (depends on Op 1)
        val = val * 3.0f;      // Op 3 (depends on Op 2)
        val = val - 5.0f;      // Op 4 (depends on Op 3)
        output[idx] = val;
    }
}

//------------------------------------------------------
// Optimized: Independent operations (high ILP)
// Multiple independent operations can execute in parallel
__global__ void independent_ops_kernel(float* output, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Independent operations - can execute in parallel
        float val = input[idx];
        float val2 = val * 2.0f;   // Independent operation 1
        float val3 = val + 1.0f;   // Independent operation 2 (independent of val2)
        float val4 = val * 3.0f;   // Independent operation 3 (independent of val2, val3)
        float val5 = val - 5.0f;   // Independent operation 4 (independent of others)
        
        // Combine independent results
        output[idx] = val2 + val3 + val4 + val5;
    }
}

//------------------------------------------------------
// Further optimized: Loop unrolling to expose more ILP
// Process multiple elements per thread with independent operations
__global__ void unrolled_ilp_kernel(float* output, const float* input, int N) {
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Process 4 elements per thread with independent operations
    if (base_idx + 3 < N) {
        // Load 4 elements independently
        float val0 = input[base_idx];
        float val1 = input[base_idx + 1];
        float val2 = input[base_idx + 2];
        float val3 = input[base_idx + 3];
        
        // Process all 4 independently (exposes ILP)
        float res0 = val0 * 2.0f + 1.0f;
        float res1 = val1 * 3.0f - 5.0f;
        float res2 = val2 * 4.0f + 2.0f;
        float res3 = val3 * 5.0f - 3.0f;
        
        // Store results
        output[base_idx] = res0;
        output[base_idx + 1] = res1;
        output[base_idx + 2] = res2;
        output[base_idx + 3] = res3;
    } else {
        // Handle remainder elements
        for (int i = 0; i < 4 && base_idx + i < N; ++i) {
            float val = input[base_idx + i];
            output[base_idx + i] = val * 2.0f + 1.0f;
        }
    }
}

//------------------------------------------------------
// Helper to measure kernel time
float measure_kernel(
    void (*kernel)(float*, const float*, int),
    float* d_output,
    const float* d_input,
    int N,
    int threads_per_block,
    cudaStream_t stream,
    const char* name
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    
    // Warmup
    kernel<<<blocks, threads_per_block, 0, stream>>>(d_output, d_input, N);
    cudaStreamSynchronize(stream);
    
    // Measure (multiple iterations for better accuracy)
    const int iterations = 100;
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
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / iterations;  // Average time per iteration
}

int main() {
    const int N = 10'000'000;
    const int threads_per_block = 256;
    
    printf("========================================\n");
    printf("Instruction-Level Parallelism (ILP) Example\n");
    printf("========================================\n");
    printf("Problem size: %d elements\n", N);
    printf("Threads per block: %d\n", threads_per_block);
    printf("Iterations per measurement: 100\n\n");
    
    // Allocate host memory
    float* h_input = nullptr;
    float* h_output_seq = nullptr;
    float* h_output_indep = nullptr;
    float* h_output_unrolled = nullptr;
    
    cudaMallocHost(&h_input, N * sizeof(float));
    cudaMallocHost(&h_output_seq, N * sizeof(float));
    cudaMallocHost(&h_output_indep, N * sizeof(float));
    cudaMallocHost(&h_output_unrolled, N * sizeof(float));
    
    // Initialize input
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }
    
    // Allocate device memory
    float* d_input = nullptr;
    float* d_output = nullptr;
    
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    
    cudaMallocAsync(&d_input, N * sizeof(float), stream);
    cudaMallocAsync(&d_output, N * sizeof(float), stream);
    
    // Copy input to device
    {
        PROFILE_MEMORY_COPY("H2D copy");
        cudaMemcpyAsync(d_input, h_input, N * sizeof(float),
                       cudaMemcpyHostToDevice, stream);
    }
    cudaStreamSynchronize(stream);
    
    // Test sequential operations
    printf("1. Sequential operations (low ILP):\n");
    float seq_time = measure_kernel(
        sequential_ops_kernel, d_output, d_input, N,
        threads_per_block, stream, "sequential_ops");
    printf("   Time: %.3f ms\n", seq_time);
    
    cudaMemcpyAsync(h_output_seq, d_output, N * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Test independent operations
    printf("\n2. Independent operations (high ILP):\n");
    float indep_time = measure_kernel(
        independent_ops_kernel, d_output, d_input, N,
        threads_per_block, stream, "independent_ops");
    printf("   Time: %.3f ms\n", indep_time);
    
    cudaMemcpyAsync(h_output_indep, d_output, N * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Test unrolled ILP
    printf("\n3. Unrolled ILP (4 elements per thread):\n");
    // Adjust thread count for unrolled version
    int unrolled_threads = threads_per_block;
    int unrolled_blocks = ((N + 3) / 4 + unrolled_threads - 1) / unrolled_threads;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    unrolled_ilp_kernel<<<unrolled_blocks, unrolled_threads, 0, stream>>>(
        d_output, d_input, N);
    cudaStreamSynchronize(stream);
    
    // Measure
    const int iterations = 100;
    cudaEventRecord(start, stream);
    {
        PROFILE_KERNEL_LAUNCH("unrolled_ilp");
        for (int i = 0; i < iterations; ++i) {
            unrolled_ilp_kernel<<<unrolled_blocks, unrolled_threads, 0, stream>>>(
                d_output, d_input, N);
        }
    }
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    
    float unrolled_ms;
    cudaEventElapsedTime(&unrolled_ms, start, stop);
    float unrolled_time = unrolled_ms / iterations;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("   Time: %.3f ms\n", unrolled_time);
    
    cudaMemcpyAsync(h_output_unrolled, d_output, N * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Verify correctness (simplified check)
    bool seq_correct = true;
    bool indep_correct = true;
    bool unrolled_correct = true;
    
    // Check a sample of results with more lenient tolerance
    for (int i = 0; i < N && i < 1000; ++i) {
        float expected_seq = ((h_input[i] * 2.0f) + 1.0f) * 3.0f - 5.0f;
        if (fabsf(h_output_seq[i] - expected_seq) > 1e-3f) {  // More lenient tolerance
            seq_correct = false;
            break;
        }
        
        // Independent ops: val*2 + val+1 + val*3 + val-5 = 2*val + val+1 + 3*val + val-5 = 7*val - 4
        float expected_indep = h_input[i] * 7.0f - 4.0f;
        if (fabsf(h_output_indep[i] - expected_indep) > 1e-3f) {  // More lenient tolerance
            indep_correct = false;
            break;
        }
        
        // Unrolled: different operations for different elements
        // res0 = val0 * 2.0f + 1.0f
        // res1 = val1 * 3.0f - 5.0f
        // res2 = val2 * 4.0f + 2.0f
        // res3 = val3 * 5.0f - 3.0f
        if (i < N - 3 && (i % 4) == 0) {
            float expected_unrolled = h_input[i] * 2.0f + 1.0f;
            if (fabsf(h_output_unrolled[i] - expected_unrolled) > 1e-3f) {  // More lenient tolerance
                unrolled_correct = false;
                break;
            }
        } else if (i < N - 3 && (i % 4) == 1) {
            float expected_unrolled = h_input[i] * 3.0f - 5.0f;
            if (fabsf(h_output_unrolled[i] - expected_unrolled) > 1e-3f) {
                unrolled_correct = false;
                break;
            }
        } else if (i < N - 3 && (i % 4) == 2) {
            float expected_unrolled = h_input[i] * 4.0f + 2.0f;
            if (fabsf(h_output_unrolled[i] - expected_unrolled) > 1e-3f) {
                unrolled_correct = false;
                break;
            }
        } else if (i < N - 3 && (i % 4) == 3) {
            float expected_unrolled = h_input[i] * 5.0f - 3.0f;
            if (fabsf(h_output_unrolled[i] - expected_unrolled) > 1e-3f) {
                unrolled_correct = false;
                break;
            }
        }
    }
    
    printf("\n========================================\n");
    printf("Results:\n");
    printf("  Sequential:      %s\n", seq_correct ? "✓ Correct" : "✗ Incorrect");
    printf("  Independent:     %s\n", indep_correct ? "✓ Correct" : "✗ Incorrect");
    printf("  Unrolled:        %s\n", unrolled_correct ? "✓ Correct" : "✗ Incorrect");
    
    if (seq_time > 0 && indep_time > 0) {
        float speedup_indep = seq_time / indep_time;
        printf("\n  Independent ops: %.2fx speedup vs sequential\n", speedup_indep);
    }
    
    if (seq_time > 0 && unrolled_time > 0) {
        float speedup_unrolled = seq_time / unrolled_time;
        printf("  Unrolled ILP:    %.2fx speedup vs sequential\n", speedup_unrolled);
    }
    
    printf("\nKey insight: Independent operations can execute in parallel,\n");
    printf("hiding instruction latency and improving throughput.\n");
    printf("Loop unrolling exposes more ILP opportunities.\n");
    printf("========================================\n");
    
    // Cleanup
    cudaFreeAsync(d_output, stream);
    cudaFreeAsync(d_input, stream);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_output_unrolled);
    cudaFreeHost(h_output_indep);
    cudaFreeHost(h_output_seq);
    cudaFreeHost(h_input);
    
    return (seq_correct && indep_correct && unrolled_correct) ? 0 : 1;
}

