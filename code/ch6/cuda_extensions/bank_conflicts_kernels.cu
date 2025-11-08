// bank_conflicts_kernels.cu - CUDA kernels for shared memory bank conflicts benchmarks
// Can be loaded as PyTorch CUDA extension

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "profiling_helpers.cuh"

#define SHARED_SIZE 1024

// Baseline: Bank conflicts occur when threads access same bank
__global__ void bank_conflicts_kernel(float* output, const float* input, int N) {
    __shared__ float shared_data[SHARED_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (tid < SHARED_SIZE && idx < N) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();
    
    // Access pattern that causes bank conflicts
    int stride = 8;
    if (tid < SHARED_SIZE / stride && idx < N) {
        float val = shared_data[tid * stride];
        output[idx] = val * 2.0f;
    }
}

// Optimized: Eliminate bank conflicts using padding
__global__ void bank_conflicts_padded_kernel(float* output, const float* input, int N) {
    __shared__ float shared_data[SHARED_SIZE + 1];  // Padding eliminates conflicts
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (tid < SHARED_SIZE && idx < N) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();
    
    // Same access pattern, but padding eliminates bank conflicts
    int stride = 8;
    if (tid < SHARED_SIZE / stride && idx < N) {
        float val = shared_data[tid * stride];
        output[idx] = val * 2.0f;
    }
}

void bank_conflicts(torch::Tensor output, torch::Tensor input) {
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    
    int N = input.size(0);
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    
    // Use default stream (nullptr)
    cudaStream_t stream = nullptr;
    
    {
        PROFILE_KERNEL_LAUNCH("bank_conflicts");
        bank_conflicts_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            N
        );
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
        }
        // Synchronize to catch kernel execution errors
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel execution failed: ", cudaGetErrorString(err));
        }
    }
}

void bank_conflicts_padded(torch::Tensor output, torch::Tensor input) {
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    
    int N = input.size(0);
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    
    // Use default stream (nullptr)
    cudaStream_t stream = nullptr;
    
    {
        PROFILE_KERNEL_LAUNCH("bank_conflicts_padded");
        bank_conflicts_padded_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            N
        );
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
        }
        // Synchronize to catch kernel execution errors
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel execution failed: ", cudaGetErrorString(err));
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bank_conflicts", &bank_conflicts, "Bank conflicts kernel (baseline)");
    m.def("bank_conflicts_padded", &bank_conflicts_padded, "Bank conflicts kernel with padding (optimized)");
}

