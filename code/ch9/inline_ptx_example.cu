// Architecture-specific optimizations for CUDA 13.0
// Targets Blackwell B200/B300 (sm_100)
// inline_ptx_example.cu
// Example demonstrating inline PTX for micro-optimizations

#include <cuda_runtime.h>
#include <stdio.h>

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
#include "../common/cuda13_teasers.cuh"
#endif

// Example kernel using inline PTX for prefetching
__global__ void PrefetchExample(const float *in, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Manually prefetch the next cache line (128B) of in[] into L2:
        if (idx + 32 < N) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"(in + idx + 32));
        }
        
        float x = in[idx];
        
        // Do some work here before using in[idx+32] to give time for prefetch
        float result = x;
        for (int i = 0; i < 10; ++i) {
            result = result * 1.1f + 0.1f;
        }
        
        out[idx] = result;
    }
}

// Example using inline PTX for cache control
__global__ void CacheControlExample(const float *in, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float val;
        // Load with cache global (.cg) modifier - cache in L2, bypass L1
        asm("ld.global.cg.f32 %0, [%1];" : "=f"(val) : "l"(in + idx));
        
        // Process the value
        val = val * val + 1.0f;
        
        out[idx] = val;
    }
}

// Example using inline PTX to read special registers
__global__ void SpecialRegisterExample(int *smid_output, int *lane_output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Get SM ID using inline PTX
    unsigned int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    
    // Get lane ID (warp-local thread index)
    unsigned int laneid;
    asm("mov.u32 %0, %laneid;" : "=r"(laneid));
    
    smid_output[idx] = smid;
    lane_output[idx] = laneid;
}

// Example demonstrating manual instruction scheduling with PTX
__global__ void InstructionSchedulingExample(const float *a, const float *b, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N && idx + 1 < N) {
        float val1, val2;
        
        // Manual scheduling: issue both loads back-to-back to overlap latencies
        asm("ld.global.f32 %0, [%1];" : "=f"(val1) : "l"(a + idx));
        asm("ld.global.f32 %0, [%1];" : "=f"(val2) : "l"(b + idx + 1));
        
        // Now compute on both values
        float result = val1 * val2 + val1 + val2;
        
        out[idx] = result;
    }
}

int main() {
    const int N = 1024 * 1024;
    size_t bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_in = new float[N];
    float *h_out = new float[N];
    int *h_smid = new int[N];
    int *h_lane = new int[N];
    
    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_in[i] = float(i % 1000) / 1000.0f;
    }
    
    // Allocate device memory
    float *d_in, *d_out, *d_b;
    int *d_smid, *d_lane;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_smid, N * sizeof(int));
    cudaMalloc(&d_lane, N * sizeof(int));
    
    // Copy input to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_in, bytes, cudaMemcpyHostToDevice);
    
    // Launch parameters
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    printf("=== Inline PTX Examples ===\n");
    
    // Test prefetch example
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    PrefetchExample<<<blocks, threads>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Prefetch example: %.2f ms\n", ms);
    
    // Test cache control example
    cudaEventRecord(start);
    CacheControlExample<<<blocks, threads>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&ms, start, stop);
    printf("Cache control example: %.2f ms\n", ms);
    
    // Test special register example
    SpecialRegisterExample<<<blocks, threads>>>(d_smid, d_lane);
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_smid, d_smid, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lane, d_lane, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Special registers example:\n");
    printf("  Thread 0: SM ID = %d, Lane ID = %d\n", h_smid[0], h_lane[0]);
    printf("  Thread 32: SM ID = %d, Lane ID = %d\n", h_smid[32], h_lane[32]);
    printf("  Thread 64: SM ID = %d, Lane ID = %d\n", h_smid[64], h_lane[64]);
    
    // Test instruction scheduling example
    cudaEventRecord(start);
    InstructionSchedulingExample<<<blocks, threads>>>(d_in, d_b, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&ms, start, stop);
    printf("Instruction scheduling example: %.2f ms\n", ms);
    
    // Copy final result
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("Final result[0]: %.3f\n", h_out[0]);
    
    printf("\n=== PTX Optimization Notes ===\n");
    printf("1. prefetch.global.L2 - Manually prefetch data into L2 cache\n");
    printf("2. ld.global.cg - Load with cache global hint (L2 only, bypass L1)\n");
    printf("3. %%smid, %%laneid - Special registers for SM and lane identification\n");
    printf("4. Manual scheduling - Issue independent loads back-to-back for ILP\n");
    
    printf("\nTo analyze with Nsight Compute:\n");
    printf("ncu --section MemoryWorkloadAnalysis --section WarpStateStats ./inline_ptx_example\n");
    
    // Cleanup
    delete[] h_in;
    delete[] h_out;
    delete[] h_smid;
    delete[] h_lane;
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_b);
    cudaFree(d_smid);
    cudaFree(d_lane);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
    // CUDA 13 teasers for Blackwell readers.
    cuda13_teasers::stream_ordered_teaser();
    cuda13_teasers::tma_teaser();
#else
    printf("CUDA 13 teasers require CUDA 13 headers; see later chapters for full demos.\n");
#endif

    return 0;
}

// CUDA 13.0 Stream-ordered Memory Allocation Example
__global__ void stream_ordered_memory_example() {
    // Example of stream-ordered memory allocation
    // This is a placeholder for actual implementation
    // See ch11/stream_ordered_allocator.cu for a full cudaMallocAsync demo.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Your kernel code here
}

// CUDA 13.0 TMA (Tensor Memory Accelerator) Example
__global__ void tma_example() {
    // Example of TMA usage for Blackwell B200/B300
    // This is a placeholder for actual implementation
    // See ch7/async_prefetch_tma.cu or ch10/tma_2d_pipeline_blackwell.cu for real TMA usage.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Your TMA code here
}
