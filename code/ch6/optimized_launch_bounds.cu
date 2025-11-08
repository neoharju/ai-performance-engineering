// optimized_launch_bounds.cu -- kernel with launch bounds annotation (optimized).

#include <cuda_runtime.h>
#include <cstdlib>
#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(status));                                     \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

// Kernel with launch bounds annotation (optimized)
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

int main() {
    const int N = 1024 * 1024;
    
    float *h_input, *h_output;
    CUDA_CHECK(cudaMallocHost(&h_input, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_output, N * sizeof(float)));
    
    // Initialize input
    for (int i = 0; i < N; ++i) {
        h_input[i] = float(i % 1000) / 1000.0f;
    }
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_input, N * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_output, N * sizeof(float), stream));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, N * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    
    // Launch parameters
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Time kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    cudaEventRecord(start, stream);
    myKernel<<<blocks, threads, 0, stream>>>(d_input, d_output, N);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    // Copy results back
    CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, N * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    printf("Kernel with launch bounds: %.2f ms\n", ms);
    printf("First result: %.3f\n", h_output[0]);
    
    // Cleanup
    CUDA_CHECK(cudaFreeAsync(d_input, stream));
    CUDA_CHECK(cudaFreeAsync(d_output, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    
    return 0;
}
