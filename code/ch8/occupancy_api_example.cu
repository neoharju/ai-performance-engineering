// Architecture-specific optimizations for CUDA 13.0
// Targets Blackwell B200/B300 (sm_100)
#include <cuda_runtime.h>
#include <iostream>

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
#include "../common/cuda13_teasers.cuh"
#endif

__global__ void exampleKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Example computation with some register usage
        float a = input[idx];
        float b = a * a;
        float c = b + a;
        float d = c * 2.0f;
        output[idx] = sqrtf(d);
    }
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);
    
    // Use CUDA Occupancy API to find optimal block size
    int minGridSize = 0, bestBlockSize = 0;
    
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &bestBlockSize,
        exampleKernel,
        /* dynamicSmemBytes = */ 0,
        /* blockSizeLimit = */ 0
    );
    
    printf("Recommended block size: %d\n", bestBlockSize);
    printf("Minimum grid size for max occupancy: %d\n", minGridSize);
    
    // Calculate actual grid size needed for our data
    int actualGridSize = (N + bestBlockSize - 1) / bestBlockSize;
    printf("Actual grid size needed: %d\n", actualGridSize);
    
    // Calculate theoretical occupancy
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks,
        exampleKernel,
        bestBlockSize,
        0  // dynamic shared memory size
    );
    
    // Get device properties to calculate occupancy percentage
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    float maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / 32.0f;
    float warpsPerBlock = bestBlockSize / 32.0f;
    float activeWarpsPerSM = maxActiveBlocks * warpsPerBlock;
    float theoreticalOccupancy = activeWarpsPerSM / maxWarpsPerSM;
    
    printf("Max warps per SM: %.0f\n", maxWarpsPerSM);
    printf("Warps per block: %.0f\n", warpsPerBlock);
    printf("Max active blocks per SM: %d\n", maxActiveBlocks);
    printf("Active warps per SM: %.0f\n", activeWarpsPerSM);
    printf("Theoretical occupancy: %.1f%%\n", theoreticalOccupancy * 100);
    
    // Allocate memory and run the kernel
    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    float* d_input = nullptr;
    float* d_output = nullptr;
    
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Initialize input
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i % 1000);
    }
    
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel with optimal configuration
    exampleKernel<<<actualGridSize, bestBlockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Verify result
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Simple verification
    bool correct = true;
    for (int i = 0; i < 10; ++i) {
        float expected = sqrtf((h_input[i] * h_input[i] + h_input[i]) * 2.0f);
        if (fabs(h_output[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    printf("Results %s\n", correct ? "CORRECT" : "INCORRECT");
    
    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
    // CUDA 13 teasers for Blackwell readers.
    cuda13_teasers::stream_ordered_teaser();
    cuda13_teasers::tma_teaser();
#else
    std::cout << "CUDA 13 teasers require CUDA 13 headers; see later chapters for full demos.\n";
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
