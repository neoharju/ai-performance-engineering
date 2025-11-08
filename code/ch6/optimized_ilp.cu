// optimized_ilp.cu
// Default ILP sample (vectorized float4 path with mild occupancy cap)

#include "optimized_ilp_low_occupancy_vec4_impl.cuh"

int main() {
    constexpr int kDefaultCap = 48;
    return run_ilp_low_occupancy_vec4(
        "Optimized ILP (vectorized, 48 active blocks)",
        kDefaultCap);
}
