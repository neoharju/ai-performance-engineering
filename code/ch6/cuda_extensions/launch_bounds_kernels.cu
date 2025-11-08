// launch_bounds_kernels.cu - CUDA kernels for launch bounds benchmarks
// Can be loaded as PyTorch CUDA extension

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t _status = (call);                                          \
        TORCH_CHECK(_status == cudaSuccess,                                    \
                    "CUDA error at ", __FILE__, ":", __LINE__, " - ",          \
                    cudaGetErrorString(_status));                              \
    } while (0)

} // anonymous namespace

// Kernel without launch bounds (baseline)
__global__ void kernel_no_launch_bounds(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Some computation that uses registers
        float temp1 = input[idx];
        float temp2 = temp1 * temp1;
        float temp3 = temp2 + temp1;
        float temp4 = temp3 * 2.0f;
        output[idx] = temp4;
    }
}

// Kernel with launch bounds annotation (optimized)
__global__ __launch_bounds__(256, 8)
void kernel_with_launch_bounds(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Some computation that uses registers
        float temp1 = input[idx];
        float temp2 = temp1 * temp1;
        float temp3 = temp2 + temp1;
        float temp4 = temp3 * 2.0f;
        output[idx] = temp4;
    }
}

void launch_bounds_baseline(torch::Tensor input, torch::Tensor output, int iterations) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.size(0) == output.size(0), "input and output must have same size");
    
    int n = input.size(0);
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Use PyTorch's current CUDA stream for consistency
    c10::cuda::CUDAGuard guard(input.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input.device().index());
    cudaStream_t stream_handle = stream.stream();
    
    for (int i = 0; i < iterations; ++i) {
        kernel_no_launch_bounds<<<num_blocks, threads_per_block, 0, stream_handle>>>(
            const_cast<float*>(input.data_ptr<float>()),
            output.data_ptr<float>(),
            n
        );
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream_handle));
}

void launch_bounds_optimized(torch::Tensor input, torch::Tensor output, int iterations) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.size(0) == output.size(0), "input and output must have same size");
    
    int n = input.size(0);
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Use PyTorch's current CUDA stream for consistency
    c10::cuda::CUDAGuard guard(input.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input.device().index());
    cudaStream_t stream_handle = stream.stream();
    
    for (int i = 0; i < iterations; ++i) {
        kernel_with_launch_bounds<<<num_blocks, threads_per_block, 0, stream_handle>>>(
            const_cast<float*>(input.data_ptr<float>()),
            output.data_ptr<float>(),
            n
        );
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream_handle));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_bounds_baseline", &launch_bounds_baseline, "Kernel without launch bounds (baseline)");
    m.def("launch_bounds_optimized", &launch_bounds_optimized, "Kernel with launch bounds (optimized)");
}

