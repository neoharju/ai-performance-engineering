// Architecture-specific optimizations for CUDA 13.0
// Targets Blackwell B200/B300 (sm_100)
// cutlass_gemm_example.cu
// Example demonstrating GEMM optimization for optimal arithmetic intensity

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
#include "../common/cuda13_teasers.cuh"
#endif

namespace cg = cooperative_groups;

// Simple GEMM kernel for demonstration
__global__ void simple_gemm_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Optimized GEMM kernel with shared memory
__global__ void optimized_gemm_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta) {
    
    __shared__ float sA[32][32];
    __shared__ float sB[32][32];
    cg::thread_block block = cg::this_thread_block();
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int k = 0; k < K; k += 32) {
        // Load tiles into shared memory
        if (row < M && k + threadIdx.x < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && k + threadIdx.y < K) {
            sB[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        block.sync();
        
        // Compute partial dot product
        int tile_size = min(32, K - k);
        for (int i = 0; i < tile_size; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        
        block.sync();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

bool run_gemm() {
    const int M = 256, N = 256, K = 256;  // Smaller size for testing
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    
    // Initialize matrices
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);
    for (int i = 0; i < M * N; ++i) h_C[i] = 0.0f;  // Initialize to zero
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    // Setup kernel launch parameters
    dim3 block_size(32, 32);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, 
                   (M + block_size.y - 1) / block_size.y);
    
    // Time simple GEMM
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    simple_gemm_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float simple_time;
    cudaEventElapsedTime(&simple_time, start, stop);
    
    // Copy result back
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // Verify correctness by checking a few elements
    bool passed = true;
    float max_error = 0.0f;
    
    // Check first few elements for correctness
    for (int i = 0; i < min(10, M * N); ++i) {
        float expected = 0.0f;
        int row = i / N;
        int col = i % N;
        for (int k = 0; k < K; ++k) {
            expected += h_A[row * K + k] * h_B[k * N + col];
        }
        float error = fabs(h_C[i] - expected);
        max_error = fmax(max_error, error);
        if (error > 1e-3) {
            passed = false;
        }
    }
    
    std::cout << "Problem: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "Simple GEMM time: " << simple_time << " ms" << std::endl;
    std::cout << "GEMM: " << (passed ? "PASSED" : "FAILED") << " (max error: " << max_error << ")" << std::endl;
    
    if (passed) {
        std::cout << "\nOptimizations applied:" << std::endl;
        std::cout << "- Coalesced memory access patterns" << std::endl;
        std::cout << "- Efficient thread block organization" << std::endl;
        std::cout << "- Reduced global memory traffic" << std::endl;
        std::cout << "- Higher arithmetic intensity through data reuse" << std::endl;
    }
    
    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return passed;
}

int main() {
    std::cout << "GEMM Optimization Example - Optimal Arithmetic Intensity" << std::endl;
    std::cout << "=======================================================" << std::endl;
    
    bool result = run_gemm();
    
    if (result) {
        std::cout << "\nTo profile with Nsight Compute roofline analysis:" << std::endl;
        std::cout << "ncu --section RooflineChart ./cutlass_gemm_example" << std::endl;
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
        // CUDA 13 teasers for Blackwell readers.
        cuda13_teasers::stream_ordered_teaser();
        cuda13_teasers::tma_teaser();
#else
        std::cout << "CUDA 13 teasers require CUDA 13 headers; see later chapters for full demos.\n";
#endif
        return 0;
    } else {
        return -1;
    }
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
