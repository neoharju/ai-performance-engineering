// Architecture-specific optimizations for CUDA 13.0
// Targets Blackwell B200/B300 (sm_100)
// 2d_kernel.cu
// 2D kernel example for processing images/matrices

#include <cuda_runtime.h>
#include <iostream>

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
#include "../common/cuda13_teasers.cuh"
#endif

//-------------------------------------------------------
// Kernel: my2DKernel running on the device (GPU)
// - input : device pointer to float array of size width×height
// - width : number of columns
// - height : number of rows
//-------------------------------------------------------
__global__ void my2DKernel(float* input, int width, int height) {
    // Compute 2D thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only process valid pixels
    if (x < width && y < height) {
        int idx = y * width + x;
        input[idx] *= 2.0f;
    }
}

int main() {
    // Image dimensions
    const int width = 1024;
    const int height = 1024;
    const int N = width * height;
    
    // 1) Allocate and initialize host image
    float* h_image = nullptr;
    cudaMallocHost(&h_image, N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_image[i] = 1.0f; // e.g., initialize all pixels to 1.0f
    }
    
    // 2) Allocate device image and copy data to device
    float* d_image = nullptr;
    cudaMalloc(&d_image, N * sizeof(float));
    cudaMemcpy(d_image, h_image, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 3) Configure and launch the 2D kernel
    dim3 threadsPerBlock2D(16, 16); // 256 threads per block
    dim3 blocksPerGrid2D(
        (width + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
        (height + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y
    );
    
    // 4) launch the kernel
    my2DKernel<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_image, width, height);
    
    // 5) wait for kernel to finish
    cudaDeviceSynchronize();
    
    // 6) Copy results back to host
    cudaMemcpy(h_image, d_image, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 7) Verify a sample element
    std::cout << "h_image[0] = " << h_image[0] << std::endl;
    
    // 8) Cleanup
    cudaFree(d_image);
    cudaFreeHost(h_image);
    
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
