// Architecture-specific optimizations for CUDA 13.0
// Targets Blackwell B200/B300 (sm_100)
// launch_bounds_example.cu
// Example demonstrating __launch_bounds__ for occupancy tuning

#include <cuda_runtime.h>
#include <stdio.h>

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
#include "../common/cuda13_teasers.cuh"
#endif

// Kernel with launch bounds annotation
__global__ __launch_bounds__(256, 8)
void myKernel(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Some computation that uses registers
        float temp1 = input[idx];
        float temp2 = temp1 * temp1;
        float temp3 = temp2 + temp1;
        float temp4 = temp3 * 2.0f;
        output[idx] = temp4;
    }
}

// Alternative kernel without launch bounds for comparison
__global__ void myKernelNoLB(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Same computation
        float temp1 = input[idx];
        float temp2 = temp1 * temp1;
        float temp3 = temp2 + temp1;
        float temp4 = temp3 * 2.0f;
        output[idx] = temp4;
    }
}

int main() {
    const int N = 1024 * 1024;
    
    float *h_input, *h_output1, *h_output2;
    float *d_input, *d_output1, *d_output2;
    
    // Allocate host memory
    h_input = new float[N];
    h_output1 = new float[N];
    h_output2 = new float[N];
    
    // Initialize input
    for (int i = 0; i < N; ++i) {
        h_input[i] = float(i % 1000) / 1000.0f;
    }
    
    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output1, N * sizeof(float));
    cudaMalloc(&d_output2, N * sizeof(float));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch parameters
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Time both kernels
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Test kernel with launch bounds
    cudaEventRecord(start);
    myKernel<<<blocks, threads>>>(d_input, d_output1, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms1;
    cudaEventElapsedTime(&ms1, start, stop);
    
    // Test kernel without launch bounds
    cudaEventRecord(start);
    myKernelNoLB<<<blocks, threads>>>(d_input, d_output2, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms2;
    cudaEventElapsedTime(&ms2, start, stop);
    
    // Copy results back
    cudaMemcpy(h_output1, d_output1, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output2, d_output2, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Kernel with launch bounds: %.2f ms\n", ms1);
    printf("Kernel without launch bounds: %.2f ms\n", ms2);
    printf("First result (with LB): %.3f\n", h_output1[0]);
    printf("First result (without LB): %.3f\n", h_output2[0]);
    
    // Cleanup
    delete[] h_input;
    delete[] h_output1;
    delete[] h_output2;
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
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
