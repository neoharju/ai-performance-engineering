// Architecture-specific optimizations for CUDA 13.0
// Targets Blackwell B200/B300 (sm_100)
#include <cuda_runtime.h>
#include <iostream>

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
#include "../common/cuda13_teasers.cuh"
#endif

// Original loop - limited ILP
__global__ void originalLoop(const float* A, const float* w, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float sum = 0.0f;
        for (int k = 0; k < 4; ++k) {
            float a = A[idx * 4 + k];
            sum += a * w[k]; // dependent on load a
        }
        out[idx] = sum;
    }
}

// Unrolled loop - better ILP
__global__ void unrolledLoop(const float* A, const float* w, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Load all elements first (independent loads)
        float a0 = A[idx * 4 + 0];
        float a1 = A[idx * 4 + 1];
        float a2 = A[idx * 4 + 2];
        float a3 = A[idx * 4 + 3];
        
        // Independent multiply operations
        float sum0 = a0 * w[0];
        float sum1 = a1 * w[1];
        float sum2 = a2 * w[2];
        float sum3 = a3 * w[3];
        
        // Final reduction
        float sum = sum0 + sum1 + sum2 + sum3;
        out[idx] = sum;
    }
}

// Multiple accumulators for even better ILP
__global__ void multipleAccumulators(const float* A, const float* w, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Use multiple accumulators to increase parallelism
        float sum_even = 0.0f;
        float sum_odd = 0.0f;
        
        // Process 8 elements with 2 accumulators
        for (int k = 0; k < 8; k += 2) {
            if (idx * 8 + k + 1 < N * 8) {
                float a_even = A[idx * 8 + k];
                float a_odd = A[idx * 8 + k + 1];
                
                sum_even += a_even * w[k % 4];
                sum_odd += a_odd * w[(k + 1) % 4];
            }
        }
        
        out[idx] = sum_even + sum_odd;
    }
}

// Pragma unroll example
__global__ void pragmaUnrolled(const float* A, const float* w, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float sum = 0.0f;
        
        #pragma unroll 4
        for (int k = 0; k < 4; ++k) {
            float a = A[idx * 4 + k];
            sum += a * w[k];
        }
        
        out[idx] = sum;
    }
}

int main() {
    const int N = 1 << 18;  // 256K elements (each processes 4-8 sub-elements)
    const int total_elements = N * 8;  // For multiple accumulators version
    const size_t bytes_A = total_elements * sizeof(float);
    const size_t bytes_out = N * sizeof(float);
    const size_t bytes_w = 4 * sizeof(float);
    
    // Allocate host memory
    float* h_A = (float*)malloc(bytes_A);
    float* h_w = (float*)malloc(bytes_w);
    float* h_out = (float*)malloc(bytes_out);
    
    // Initialize data
    for (int i = 0; i < total_elements; ++i) {
        h_A[i] = static_cast<float>(i % 100) / 100.0f;
    }
    
    for (int i = 0; i < 4; ++i) {
        h_w[i] = static_cast<float>(i + 1) / 4.0f;
    }
    
    // Allocate device memory
    float* d_A = nullptr;
    float* d_w = nullptr;
    float* d_out = nullptr;
    
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_w, bytes_w);
    cudaMalloc(&d_out, bytes_out);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, bytes_w, cudaMemcpyHostToDevice);
    
    // Launch configuration
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    
    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Test original loop
    cudaEventRecord(start);
    originalLoop<<<gridSize, blockSize>>>(d_A, d_w, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_original;
    cudaEventElapsedTime(&time_original, start, stop);
    
    // Test unrolled loop
    cudaEventRecord(start);
    unrolledLoop<<<gridSize, blockSize>>>(d_A, d_w, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_unrolled;
    cudaEventElapsedTime(&time_unrolled, start, stop);
    
    // Test multiple accumulators
    cudaEventRecord(start);
    multipleAccumulators<<<gridSize, blockSize>>>(d_A, d_w, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_multiple;
    cudaEventElapsedTime(&time_multiple, start, stop);
    
    // Test pragma unrolled
    cudaEventRecord(start);
    pragmaUnrolled<<<gridSize, blockSize>>>(d_A, d_w, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_pragma;
    cudaEventElapsedTime(&time_pragma, start, stop);
    
    printf("Original loop time: %.3f ms\n", time_original);
    printf("Unrolled loop time: %.3f ms\n", time_unrolled);
    printf("Multiple accumulators time: %.3f ms\n", time_multiple);
    printf("Pragma unrolled time: %.3f ms\n", time_pragma);
    
    printf("Unroll speedup: %.2fx\n", time_original / time_unrolled);
    printf("Multiple acc speedup: %.2fx\n", time_original / time_multiple);
    printf("Pragma speedup: %.2fx\n", time_original / time_pragma);
    
    printf("\nProfile with Nsight Compute to see ILP improvements:\n");
    printf("ncu --metrics smsp__issue_efficiency.avg,smsp__inst_executed.avg.per_cycle ./loop_unrolling\n");
    
    // Simple verification
    cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost);
    printf("\nFirst few results: %.6f, %.6f, %.6f\n", h_out[0], h_out[1], h_out[2]);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_A);
    free(h_w);
    free(h_out);
    cudaFree(d_A);
    cudaFree(d_w);
    cudaFree(d_out);
    
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
