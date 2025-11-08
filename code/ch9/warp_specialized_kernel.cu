#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
            exit(1); \
        } \
    } while(0)

constexpr int WARP_SIZE = 32;
constexpr int TILE_SIZE = 256;
constexpr int PIPELINE_DEPTH = 2;

// Warp specialization kernel: Producer/Consumer pattern
// Warp 0: Producer (loads data)
// Warps 1-6: Compute (process data)
// Warp 7: Consumer (stores results)
__global__ void warp_specialized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_elements
) {
    cg::thread_block block = cg::this_thread_block();
    
    // Shared memory for double buffering
    extern __shared__ float smem[];
    float* stage_buffer = smem;
    
    // CUDA Pipeline API for producer/consumer synchronization
    using pipe_state = cuda::pipeline_shared_state<cuda::thread_scope_block, PIPELINE_DEPTH>;
    __shared__ alignas(pipe_state) unsigned char state_storage[sizeof(pipe_state)];
    auto* state = reinterpret_cast<pipe_state*>(state_storage);
    auto pipe = cuda::make_pipeline(block, state);
    
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    
    // Warp roles (8 warps per block: 1 producer, 6 compute, 1 consumer)
    constexpr int PRODUCER_WARP = 0;
    constexpr int CONSUMER_WARP = 7;
    constexpr int COMPUTE_WARPS_START = 1;
    constexpr int COMPUTE_WARPS_END = 6;
    
    bool is_producer = (warp_id == PRODUCER_WARP);
    bool is_consumer = (warp_id == CONSUMER_WARP);
    bool is_compute = (warp_id >= COMPUTE_WARPS_START && warp_id <= COMPUTE_WARPS_END);
    
    int num_tiles = (n_elements + TILE_SIZE - 1) / TILE_SIZE;
    
    // PRODUCER WARP: Load tiles into shared memory
    if (is_producer) {
        for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
            int buf_idx = tile % PIPELINE_DEPTH;
            int tile_start = tile * TILE_SIZE;
            int tile_end = min(tile_start + TILE_SIZE, n_elements);
            
            pipe.producer_acquire();
            
            // Load tile into shared memory
            for (int idx = lane; idx < TILE_SIZE; idx += WARP_SIZE) {
                int global_idx = tile_start + idx;
                if (global_idx < n_elements) {
                    stage_buffer[buf_idx * TILE_SIZE + idx] = input[global_idx];
                }
            }
            
            pipe.producer_commit();
        }
    }
    
    // COMPUTE WARPS: Process tiles from shared memory
    if (is_compute) {
        int compute_warp_id = warp_id - COMPUTE_WARPS_START;
        int num_compute_warps = COMPUTE_WARPS_END - COMPUTE_WARPS_START + 1;
        
        for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
            int buf_idx = tile % PIPELINE_DEPTH;
            
            pipe.consumer_wait();
            block.sync();
            
            // Each compute warp processes a slice
            int elems_per_warp = TILE_SIZE / num_compute_warps;
            int start_idx = compute_warp_id * elems_per_warp;
            
            for (int idx = start_idx + lane; idx < start_idx + elems_per_warp; idx += WARP_SIZE) {
                if (idx < TILE_SIZE) {
                    float val = stage_buffer[buf_idx * TILE_SIZE + idx];
                    // Compute: ReLU activation
                    val = fmaxf(val, 0.0f);
                    stage_buffer[buf_idx * TILE_SIZE + idx] = val;
                }
            }
            
            // Only last compute warp releases
            if (compute_warp_id == num_compute_warps - 1) {
                pipe.consumer_release();
            }
            
            block.sync();
        }
    }
    
    // CONSUMER WARP: Store tiles from shared memory
    if (is_consumer) {
        for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
            int buf_idx = tile % PIPELINE_DEPTH;
            int tile_start = tile * TILE_SIZE;
            int tile_end = min(tile_start + TILE_SIZE, n_elements);
            
            pipe.consumer_wait();
            block.sync();
            
            // Store tile from shared memory
            for (int idx = lane; idx < TILE_SIZE; idx += WARP_SIZE) {
                int global_idx = tile_start + idx;
                if (global_idx < n_elements) {
                    float val = stage_buffer[buf_idx * TILE_SIZE + idx];
                    // Consumer: Apply transformation
                    output[global_idx] = val * 0.5f;
                }
            }
            
            pipe.consumer_release();
            block.sync();
        }
    }
}

torch::Tensor warp_specialized_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    
    int n_elements = input.numel();
    auto output = torch::empty_like(input);
    
    // Launch configuration
    dim3 block(256);  // 8 warps * 32 threads
    dim3 grid((n_elements + TILE_SIZE - 1) / TILE_SIZE);
    int shared_mem_size = PIPELINE_DEPTH * TILE_SIZE * sizeof(float);
    
    warp_specialized_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n_elements
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("warp_specialized_forward", &warp_specialized_forward, "Warp specialized forward pass");
}

