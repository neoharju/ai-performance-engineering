// optimized_ilp_low_occupancy_vec4.cu
// Variant that caps active blocks to highlight ILP speedups further

#include "optimized_ilp_low_occupancy_vec4_impl.cuh"

int main() {
    constexpr int kLowOccupancyCap = 32;
    return run_ilp_low_occupancy_vec4(
        "Vectorized ILP (float4, 32 active blocks)",
        kLowOccupancyCap);
}
