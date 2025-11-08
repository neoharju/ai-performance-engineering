// work_queue_kernels.cu - CUDA kernels for work queue benchmarks
// Can be loaded as PyTorch CUDA extension

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include "profiling_helpers.cuh"

namespace {

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t _status = (call);                                          \
        TORCH_CHECK(_status == cudaSuccess,                                    \
                    "CUDA error at ", __FILE__, ":", __LINE__, " - ",          \
                    cudaGetErrorString(_status));                              \
    } while (0)

} // anonymous namespace

// Static work distribution (baseline)
__global__ void compute_static_kernel(const float* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int work = idx & 255;
    float sum = 0.0f;
    for (int i = 0; i < work; ++i) {
      sum += sinf(input[idx]) * cosf(input[idx]);
    }
    output[idx] = sum;
  }
}

// Dynamic work queue (optimized)
__device__ unsigned int g_work_index = 0;

__global__ void compute_dynamic_kernel(const float* input, float* output, int n) {
  while (true) {
    unsigned mask = __activemask();
    int lane = threadIdx.x & (warpSize - 1);
    int leader = __ffs(mask) - 1;

    unsigned base = 0;
    if (lane == leader) {
      base = atomicAdd(&g_work_index, warpSize);
    }
    base = __shfl_sync(mask, base, leader);

    unsigned idx = base + lane;
    if (idx >= static_cast<unsigned>(n)) break;

    float s, c;
    __sincosf(input[idx], &s, &c);
    int work = idx & 255;
    float sum = 0.0f;
#pragma unroll 1
    for (int i = 0; i < work; ++i) {
      sum += s * c;
    }
    output[idx] = sum;
  }
}

void static_work_distribution(torch::Tensor input, torch::Tensor output, int iterations) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.size(0) == output.size(0), "input and output must have same size");
    
    // Ensure we're on the correct device
    int device_id = input.device().index();
    CHECK_CUDA(cudaSetDevice(device_id));
    
    int n = input.size(0);
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Use default stream (nullptr) - this is the legacy default stream
    // PyTorch operations on default stream will be properly synchronized
    cudaStream_t stream = nullptr;
    
    {
        PROFILE_KERNEL_LAUNCH("static_work_distribution");
        // Use const_cast to work around PyTorch ABI issues with const data_ptr template
        const float* input_ptr = input.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();
        for (int i = 0; i < iterations; ++i) {
            compute_static_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
                input_ptr,
                output_ptr,
                n
            );
        }
        CHECK_CUDA(cudaGetLastError());
    }
    // Note: No explicit synchronization here - PyTorch benchmark harness handles synchronization
}

void dynamic_work_queue(torch::Tensor input, torch::Tensor output, int iterations) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.size(0) == output.size(0), "input and output must have same size");
    
    // Ensure we're on the correct device
    int device_id = input.device().index();
    CHECK_CUDA(cudaSetDevice(device_id));
    
    int n = input.size(0);
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Use default stream (nullptr) - this is the legacy default stream
    // PyTorch operations on default stream will be properly synchronized
    cudaStream_t stream = nullptr;
    
    {
        PROFILE_KERNEL_LAUNCH("dynamic_work_queue");
        // Use const_cast to work around PyTorch ABI issues with const data_ptr template
        const float* input_ptr = input.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();
        for (int i = 0; i < iterations; ++i) {
            // Reset counter before each iteration
            unsigned zero = 0;
            CHECK_CUDA(cudaMemcpyToSymbol(g_work_index, &zero, sizeof(unsigned)));
            
            compute_dynamic_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
                input_ptr,
                output_ptr,
                n
            );
        }
        CHECK_CUDA(cudaGetLastError());
    }
    // Note: No explicit synchronization here - PyTorch benchmark harness handles synchronization
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("static_work_distribution", &static_work_distribution, "Static work distribution (baseline)");
    m.def("dynamic_work_queue", &dynamic_work_queue, "Dynamic work queue with atomics (optimized)");
}

