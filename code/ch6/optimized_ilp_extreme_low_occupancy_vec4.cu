// optimized_ilp_extreme_low_occupancy_vec4.cu
// Most aggressive variant: float4 vectorization with only 16 active blocks

#include "optimized_ilp_low_occupancy_vec4_impl.cuh"

int main() {
    constexpr int kExtremeCap = 16;
    return run_ilp_low_occupancy_vec4(
        "Vectorized ILP (float4, 16 active blocks)",
        kExtremeCap);
}
