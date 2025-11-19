// occupancy_tuning.cu -- simple __launch_bounds__ illustration with tunable launch config.

#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

constexpr int N = 1 << 20;

__global__ __launch_bounds__(256, 4)
void kernel(const float* in, float* out, int n, int smem_bytes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ unsigned char smem[];
  // Touch shared memory when requested so occupancy accounts for it.
  if (smem_bytes > 0 && threadIdx.x == 0) {
    smem[0] = 0;
  }
  if (idx < n) {
    float val = in[idx];
    out[idx] = sqrtf(val * val + 1.0f);
  }
}

int main(int argc, char** argv) {
  int block_size = 256;
  int smem_bytes = 0;
  // Basic flag parsing for --block-size and --smem-bytes.
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
      block_size = std::max(1, atoi(argv[++i]));
    } else if (strcmp(argv[i], "--smem-bytes") == 0 && i + 1 < argc) {
      smem_bytes = std::max(0, atoi(argv[++i]));
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      printf("Usage: %s [--block-size N] [--smem-bytes N]\n", argv[0]);
      return 0;
    }
  }
  if (block_size <= 0 || block_size > 1024) {
    fprintf(stderr, "Invalid block size %d (must be 1..1024)\n", block_size);
    return 1;
  }

  float *h_in, *h_out;
  cudaMallocHost(&h_in, N * sizeof(float));
  cudaMallocHost(&h_out, N * sizeof(float));
  for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

  float *d_in, *d_out;
  cudaMalloc(&d_in, N * sizeof(float));
  cudaMalloc(&d_out, N * sizeof(float));
  cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block(block_size);
  dim3 grid((N + block.x - 1) / block.x);
  kernel<<<grid, block, smem_bytes>>>(d_in, d_out, N, smem_bytes);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
  printf("out[0]=%.1f (block=%d, smem=%d bytes)\n", h_out[0], block_size, smem_bytes);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFreeHost(h_in);
  cudaFreeHost(h_out);
  return 0;
}
