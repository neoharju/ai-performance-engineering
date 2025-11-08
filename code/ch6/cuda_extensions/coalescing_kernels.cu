// coalescing_kernels.cu - CUDA kernels for memory coalescing benchmarks
// Can be loaded as PyTorch CUDA extension

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "profiling_helpers.cuh"

// Uncoalesced memory access pattern
__global__ void uncoalesced_copy_kernel(float* output, const float* input, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int access_idx = idx * stride;  // Non-consecutive access
    
    if (access_idx < N) {
        output[access_idx] = input[access_idx];
    }
}

// Coalesced memory access pattern
__global__ void coalesced_copy_kernel(float* output, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        output[idx] = input[idx];  // Consecutive access
    }
}

// Python-callable wrapper for uncoalesced copy
void uncoalesced_copy(torch::Tensor output, torch::Tensor input, int stride) {
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    
    int N = input.size(0);
    int threads_per_block = 256;
    // Number of threads needed: (N + stride - 1) / stride
    int num_threads = (N + stride - 1) / stride;
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
    
    // Use default stream (nullptr)
    cudaStream_t stream = nullptr;
    
    {
        PROFILE_KERNEL_LAUNCH("uncoalesced_copy");
        uncoalesced_copy_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            N,
            stride
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

// Python-callable wrapper for coalesced copy
void coalesced_copy(torch::Tensor output, torch::Tensor input) {
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
        PROFILE_KERNEL_LAUNCH("coalesced_copy");
        coalesced_copy_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
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
    m.def("uncoalesced_copy", &uncoalesced_copy, "Uncoalesced memory copy kernel");
    m.def("coalesced_copy", &coalesced_copy, "Coalesced memory copy kernel");
}

