// Optimized double buffering binary.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "double_buffering_common.cuh"

using namespace ch8;

namespace {

struct RunConfig {
    int elements = kDoubleBufferBlock * kDoubleBufferTile * 8192;
    int warmup = 5;
    int iterations = 80;
};

int clamp_elements(int value) {
    const int minimum = kDoubleBufferBlock * kValuesPerThread;
    return std::max(value, minimum);
}

int parse_int_arg(const std::string& arg, const char* prefix) {
    const auto prefix_len = std::strlen(prefix);
    return std::stoi(arg.substr(prefix_len));
}

RunConfig parse_run_config(int argc, char** argv) {
    RunConfig cfg{};
    bool explicit_elements = false;

    if (const char* env_elems = std::getenv("CH8_DOUBLE_BUFFER_ELEMENTS")) {
        cfg.elements = clamp_elements(std::max(1, std::atoi(env_elems)));
        explicit_elements = true;
    }
    if (const char* env_iters = std::getenv("CH8_DOUBLE_BUFFER_ITERATIONS")) {
        cfg.iterations = std::max(1, std::atoi(env_iters));
    }
    if (const char* env_warmup = std::getenv("CH8_DOUBLE_BUFFER_WARMUP")) {
        cfg.warmup = std::max(0, std::atoi(env_warmup));
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind("--elements=", 0) == 0) {
            cfg.elements = clamp_elements(parse_int_arg(arg, "--elements="));
            explicit_elements = true;
        } else if (arg.rfind("--iterations=", 0) == 0) {
            cfg.iterations = std::max(1, parse_int_arg(arg, "--iterations="));
        } else if (arg.rfind("--warmup=", 0) == 0) {
            cfg.warmup = std::max(0, parse_int_arg(arg, "--warmup="));
        } else if (arg == "--profile-lite") {
            // Profiling-only preset: reduce the grid so Nsight Compute can finish on GB10.
            if (!explicit_elements) {
                cfg.elements = clamp_elements(kDoubleBufferBlock * kDoubleBufferTile * 1024);
            }
            cfg.iterations = std::min(cfg.iterations, 20);
            cfg.warmup = std::min(cfg.warmup, 2);
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: optimized_double_buffering_pipelined_sm121 [options]\n"
                << "Options:\n"
                << "  --elements=<int>      Total elements to process (default "
                << kDoubleBufferBlock * kDoubleBufferTile * 8192 << ")\n"
                << "  --iterations=<int>    Timed iterations (default 80)\n"
                << "  --warmup=<int>        Warmup launches (default 5)\n"
                << "  --profile-lite        Convenient preset for Nsight (smaller input)\n"
                << "  -h, --help            Show this message\n";
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::exit(EXIT_FAILURE);
        }
    }

    cfg.elements = clamp_elements(cfg.elements);
    cfg.iterations = std::max(cfg.iterations, 1);
    cfg.warmup = std::max(cfg.warmup, 0);
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    RunConfig cfg = parse_run_config(argc, argv);
    std::cout << "Elements: " << cfg.elements << ", warmup: " << cfg.warmup
              << ", iterations: " << cfg.iterations << "\n";

    std::vector<float> host(cfg.elements);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : host) {
        v = dist(gen);
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, cfg.elements * sizeof(float));
    cudaMalloc(&d_output, cfg.elements * sizeof(float));
    cudaMemcpy(d_input, host.data(), cfg.elements * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const size_t shared_bytes = kDoubleBufferBlock * kValuesPerThread * kPipelineStages * sizeof(float);

    for (int i = 0; i < cfg.warmup; ++i) {
        double_buffer_optimized_kernel<<<double_buffer_grid(cfg.elements), kDoubleBufferBlock, shared_bytes>>>(
            d_input, d_output, cfg.elements);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < cfg.iterations; ++i) {
        double_buffer_optimized_kernel<<<double_buffer_grid(cfg.elements), kDoubleBufferBlock, shared_bytes>>>(
            d_input, d_output, cfg.elements);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    std::cout << "Optimized double buffering: " << (total_ms / cfg.iterations) << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
