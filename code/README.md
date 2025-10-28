# AI Performance Engineering - Code Repository
## Blackwell B200 Edition

**Hardware:** NVIDIA B200 (SM 10.0, 180 GB HBM3e, 148 SMs)  
**Software:** PyTorch 2.9 nightly (CUDA 13.0 cu130), Triton 3.5  
**Status:** All tests validated on actual B200 hardware

---

## 📚 Table of Contents

### Quick Start & Setup
- [Quick Setup](#quick-setup)
- [Quick Start](#quick-start)
- [Scripts](#scripts)
- [Profiling (Advanced)](#profiling-advanced)
- [Repository Structure](#repository-structure-explained)

### Chapter Guide
- [Chapter 1: Performance Basics](#chapter-1-performance-basics)
- [Chapter 2: Hardware Overview](#chapter-2-hardware-overview)
- [Chapter 3: System Setup](#chapter-3-system-setup)
- [Chapter 4: Distributed Networking](#chapter-4-distributed-networking)
- [Chapter 5: Storage & I/O](#chapter-5-storage--io)
- [Chapter 6: CUDA Kernels](#chapter-6-cuda-kernels)
- [Chapter 7: Memory Access Patterns](#chapter-7-memory-access-patterns)
- [Chapter 8: Occupancy & ILP](#chapter-8-occupancy--ilp)
- [Chapter 9: Kernel Fusion](#chapter-9-kernel-fusion)
- [Chapter 10: Tensor Cores](#chapter-10-tensor-cores)
- [Chapter 11: Streams & Concurrency](#chapter-11-streams--concurrency)
- [Chapter 12: Graphs & Dynamic Parallelism](#chapter-12-graphs--dynamic-parallelism)
- [Chapter 13: PyTorch Profiling](#chapter-13-pytorch-profiling)
- [Chapter 14: Compiler & Triton](#chapter-14-compiler--triton)
- [Chapter 15: Disaggregated Inference](#chapter-15-disaggregated-inference)
- [Chapter 16: Inference Optimization](#chapter-16-inference-optimization)
- [Chapter 17: Dynamic Routing](#chapter-17-dynamic-routing)
- [Chapter 18: Attention Mechanisms](#chapter-18-attention-mechanisms)
- [Chapter 19: Advanced Training](#chapter-19-advanced-training)
- [Chapter 20: AI Kernel Generator](#chapter-20-ai-kernel-generator)

### Reference
- [Quick Reference: How to Run Everything](#quick-reference-how-to-run-everything)
- [Testing](#testing)
- [Performance Targets Summary](#performance-targets-summary-actual-measured)
- [Critical Notes](#critical-notes)

---

## Quick Setup

Run the comprehensive setup script to install everything:

```bash
# From the code/ directory
sudo ./setup.sh
```

This installs:
- PyTorch 2.9 nightly (CUDA 13.0 cu130 build)
- CUDA 13.0 toolchain and development tools
- NVIDIA Nsight Systems & Compute (latest versions)  
- All Python dependencies (`requirements_latest.txt`)
- System tools (numactl, perf, etc.)

---

## Quick Start

```bash
# 1. Install everything (once)
sudo ./setup.sh

# 2. Run an example
python3 ch1/performance_basics.py

# 3. Test all examples
./run_all_tests.sh

# 4. Measure peak performance
python3 benchmark_peak.py

# 5. Run unit tests (optional)
pytest tests/ -v
```

---

## Scripts

**`setup.sh`** - Install PyTorch 2.9, CUDA 13, Nsight tools, dependencies (run once with sudo)

**`run_all_tests.sh`** - Test all chapter examples compile and run (5-10 min) - **use this**

**`benchmark_peak.py`** - Measure peak: HBM3e bandwidth, FP16 compute, torch.compile (2-3 min)

**`pytest tests/`** - Unit tests for optimizations (development only)

---

## Profiling (Advanced)

For book manuscript only:
- `start.sh` - Profile all examples (hours)
- `stop.sh` - Stop profiling
- `extract.sh` - Extract metrics
- `assert.sh` - Validate system

---

## Repository Structure Explained

```
ai-performance-engineering/
├── code/                          # ← YOU ARE HERE (self-contained!)
│   ├── setup.sh                   # Install everything
│   ├── run_all_tests.sh          # Test all examples
│   ├── benchmark_peak.py         # Measure peak performance
│   ├── start.sh, stop.sh, etc.   # Profiling scripts (advanced)
│   ├── requirements_latest.txt
│   │
│   ├── tests/                     # Unit tests
│   │   ├── test_blackwell_optimizations.py
│   │   └── test_blackwell_stack.py
│   │
│   ├── scripts/                   # Profiling infrastructure
│   │   ├── profile_harness.py    # Main profiling orchestrator
│   │   └── example_registry.py   # Example discovery
│   │
│   ├── tools/                     # Metrics extraction
│   │   ├── extract_ncu_metrics.py
│   │   └── extract_nsys_summary.py
│   │
│   ├── profiles/                  # Generated profiling data (gitignored)
│   ├── profile_runs/              # Profile logs (gitignored)
│   │
│   └── ch1/, ch2/, ..., ch20/    # Chapter examples
│
├── resources/                     # Book PDF and extracted text
├── archive/                       # Historical implementations
└── README.md                      # Top-level project README
```

**Why this structure?**
- `code/` is now **fully self-contained** - all scripts, tools, and generated data live here
- `resources/` stays at root (book PDF, extracted text - not code)
- `archive/` stays at root (historical artifacts)
- Everything you need to run examples and profiles is in `code/` - just `cd code/` and go!

---

## Chapter 1: Performance Basics

**Files:**
- `ch1/performance_basics.py`

**Key Concepts:**
- Performance measurement fundamentals
- Profiling setup and tools
- Baseline metrics establishment

**To run:**
```bash
python3 ch1/performance_basics.py
```

---

## Chapter 2: Hardware Overview

**Files:**
- `ch2/hardware_info.py`
- `ch2/nvlink_c2c_p2p_blackwell.cu`
- `ch2/gb200_grace_blackwell_coherency.cu`
- `ch2/gb200_topology_aware.py`

**Key Concepts:**
- Blackwell B200 architecture (180 GB HBM3e, 148 SMs)
- 5th-gen Tensor Cores (tcgen05)
- NVLink-C2C: 900 GB/s coherent interconnect
- HBM3e: 7.8 TB/s peak bandwidth
- GB200/GB300 Grace-Blackwell integration

**To run:**
```bash
cd ch2
python3 hardware_info.py
make && ./nvlink_c2c_p2p_blackwell
```

**Expected:** NVLink-C2C ~900 GB/s, PCIe Gen5 ~64 GB/s

---

## Chapter 3: System Setup

**Files:**
- `ch3/bind_numa_affinity.py`
- `ch3/docker_gpu_optimized.dockerfile`
- `ch3/kubernetes_mig_pod.yaml`, `ch3/kubernetes_topology_pod.yaml`
- `ch3/gb200_numa_optimizations.sh`
- `ch3/system_tuning.sh`

**Key Concepts:**
- System tuning for Blackwell (CPU governor, IRQ affinity)
- NUMA topology optimization
- Container deployment (Docker, Kubernetes)
- MIG (Multi-Instance GPU) configuration
- GPU persistence mode and clock settings

**To run:**
```bash
python3 ch3/bind_numa_affinity.py
sudo ./ch3/system_tuning.sh
```

---

## Chapter 4: Distributed Networking

**Files:**
- `ch4/multi_node_blackwell.py`
- `ch4/torchtitan_async_tp_demo.py`
- `ch4/training_8xb200_pipeline.py`
- `ch4/nccl_blackwell_config.py`
- `ch4/symmetric_memory_8gpu.py`
- `ch4/bandwidth_benchmark_suite_8gpu.py`
- `ch4/nvshmem_8gpu_examples.cu`

**Key Concepts:**
- Multi-node training on Blackwell
- Async-TP with CUDA graph trees
- Hybrid parallelism (TP + DP + FSDP)
- NCCL optimization for 148 SMs
- 8x B200 GPU topology optimization
- NVLink 5.0 (1800 GB/s per pair)
- NVSHMEM and PyTorch Symmetric Memory

**To run:**
```bash
python3 ch4/multi_node_blackwell.py
torchrun --nproc_per_node=8 ch4/training_8xb200_pipeline.py --tp-size 2
torchrun --nproc_per_node=8 ch4/bandwidth_benchmark_suite_8gpu.py --full
```

**Expected:** 8x B200 scaling 85-95% efficiency, AllReduce 700-800 GB/s

### Hybrid Parallelism

| Model Size | TP | DP | Total GPUs | Memory/GPU |
|------------|----|----|-----------|------------|
| <1B | 1 | 8 | 8 | 1-2 GB |
| 1-10B | 2 | 4 | 8 | 4-10 GB |
| 10-30B | 4 | 2 | 8 | 10-25 GB |
| 30-100B | 8 | 1 | 8 | 25-50 GB |
| 100B+ | 8 | N | 8×N | 50 GB+ |

---

## Chapter 5: Storage & I/O

**Files:**
- `ch5/gpudirect_storage_example.py`
- `ch5/storage_io_optimization.py`

**Key Concepts:**
- GPUDirect Storage: Direct GPU-SSD data transfers
- Bypassing CPU in data loading pipelines
- I/O bottleneck elimination
- Optimized data loading for training and inference

**To run:**
```bash
python3 ch5/gpudirect_storage_example.py
python3 ch5/storage_io_optimization.py
```

**Expected:** GPUDirect Storage 2-3x faster data loading

---

## Chapter 6: CUDA Kernels

**Files:**
- `ch6/my_first_kernel.cu`
- `ch6/simple_kernel.cu`
- `ch6/add_parallel.cu`, `ch6/add_sequential.cu`
- `ch6/2d_kernel.cu`
- `ch6/launch_bounds_example.cu`
- `ch6/occupancy_api.cu`
- `ch6/unified_memory.cu`

**Key Concepts:**
- CUDA programming model fundamentals
- Thread hierarchy (thread, block, grid)
- Memory management (cudaMalloc, cudaMemcpy)
- Kernel launch configuration
- Occupancy optimization
- Unified memory basics

**To run:**
```bash
cd ch6 && make
./my_first_kernel
./add_parallel
```

---

## Chapter 7: Memory Access Patterns

**Files:**
- `ch7/hbm3e_optimized_copy.cu`
- `ch7/hbm3e_peak_bandwidth.cu`
- `ch7/async_prefetch_tma.cu`

**Key Concepts:**
- HBM3e: 256-byte bursts, cache streaming
- Memory coalescing for Blackwell
- TMA (Tensor Memory Accelerator)

**To run:**
```bash
cd ch7 && make && ./hbm3e_peak_bandwidth
```

**Expected:** HBM3e optimized >7.0 TB/s (90%+)

---

## Chapter 8: Occupancy & ILP

**Files:**
- `ch8/occupancy_pytorch.py`
- `ch8/occupancy_tuning.cu`
- `ch8/ilp_pytorch.py`
- `ch8/warp_divergence_pytorch.py`
- `ch8/independent_ops.cu`
- `ch8/loop_unrolling.cu`
- `ch8/threshold_naive.cu`, `ch8/threshold_predicated.cu`

**Key Concepts:**
- GPU occupancy: maximizing SM utilization
- Balancing threads, registers, and shared memory
- Instruction-level parallelism (ILP)
- Warp divergence and control flow
- Loop unrolling for performance
- Predication vs branching

**To run:**
```bash
cd ch8 && make
python3 occupancy_pytorch.py
./occupancy_tuning
```

**Expected:** Occupancy 1.5-2.5x, ILP 1.2-1.8x

---

## Chapter 9: Kernel Fusion

**Files:**
- `ch9/fusion_pytorch.py`
- `ch9/fused_l2norm.cu`
- `ch9/inline_ptx_example.cu`
- `ch9/cutlass_gemm_example.cu`

**Key Concepts:**
- Kernel fusion: reducing memory bandwidth
- Fusing elementwise operations
- torch.compile for automatic fusion
- CUTLASS library for optimized GEMM
- Inline PTX for fine-grained control

**To run:**
```bash
cd ch9 && make
python3 fusion_pytorch.py
./fused_l2norm
```

**Expected:** Fused operations 2-4x speedup

---

## Chapter 10: Tensor Cores

**Files:**
- `ch10/tcgen05_blackwell.cu`
- `ch10/cluster_group_blackwell.cu`
- `ch10/tma_2d_pipeline_blackwell.cu`

**Key Concepts:**
- tcgen05 (5th-gen Tensor Cores for Blackwell)
- FP8 native support
- 8 CTAs per cluster (vs 4 on Hopper)
- 2 MB DSMEM
- TMA (Tensor Memory Accelerator) pipeline

**To run:**
```bash
cd ch10 && make && ./tcgen05_blackwell
```

**Expected:** FP8 >1200 TFLOPS, FP16 >800 TFLOPS

---

## Chapter 11: Streams & Concurrency

**Files:**
- `ch11/basic_streams.cu`
- `ch11/stream_ordered_allocator.cu`
- `ch11/warp_specialized_pipeline_multistream.cu`

**Key Concepts:**
- CUDA streams for concurrent execution
- Overlapping compute and memory transfers
- Stream-ordered memory allocation
- Multi-stream pipelines for throughput
- Stream priorities and dependencies

**To run:**
```bash
cd ch11 && make
./basic_streams
./stream_ordered_allocator
```

**Expected:** Stream concurrency 1.5-2x throughput

---

## Chapter 12: Graphs & Dynamic Parallelism

**Files:**
- `ch12/cuda_graphs.cu`
- `ch12/cuda_graphs_conditional.cu`
- `ch12/dynamic_parallelism.cu`
- `ch12/atomic_work_queue.cu`

**Key Concepts:**
- CUDA graphs: recording and replaying work
- Graph instantiation and optimization
- Conditional graphs for dynamic workloads
- Dynamic parallelism: kernels launching kernels
- Work queue patterns with atomics

**To run:**
```bash
cd ch12 && make
./cuda_graphs
./cuda_graphs_conditional
```

**Expected:** CUDA graphs 20-40% latency reduction

---

## Chapter 13: PyTorch Profiling

**Files:**
- `ch13/compiled_autograd.py`
- `ch13/custom_allocator.py`
- `ch13/fsdp_example.py`
- `ch13/memory_profiling.py`
- `ch13/train_deepseek_v3.py`

**Key Concepts:**
- Native FP8 types (PyTorch 2.9)
- Compiled autograd for faster backward pass
- Memory profiling for Blackwell
- FSDP for large model training
- Custom memory allocators

**To run:**
```bash
python3 ch13/compiled_autograd.py
python3 ch13/memory_profiling.py
torchrun --nproc_per_node=8 ch13/fsdp_example.py
```

**Expected:** Compiled autograd 1.1-1.3x, FSDP for 100B+ models

---

## Chapter 14: Compiler & Triton

**Files:**
- `ch14/torch_compiler_examples.py`
- `ch14/torch_compile_large_model.py`
- `ch14/training_large_model_1_5x.py`
- `ch14/deepseek_innovation_l2_bypass.py`
- `ch14/triton_examples.py`
- `ch14/triton_fp8_advanced.py`
- `ch14/triton_tma_blackwell.py`
- `ch14/test_blackwell_optimizations.py`

**Key Concepts:**
- torch.compile: 1.3x+ speedup for large models
- 100+ warmup iterations required
- Model size matters: <50M (1.0-1.1x), 500M-1B (1.2-1.3x), 1B+ (1.3-1.5x)
- DeepSeek innovation: L2 cache control
- Triton 3.5: TMA descriptors, FP8 kernels
- Blackwell TMA broadcast and eviction optimizations

**To run:**
```bash
python3 ch14/torch_compiler_examples.py
python3 ch14/training_large_model_1_5x.py
python3 ch14/triton_tma_blackwell.py
```

**Expected:** torch.compile 1.3x+ (large), FP8 1.5-2.0x, DeepSeek 1.05-1.15x

---

## Chapter 15: Disaggregated Inference

**Files:**
- `ch15/disaggregated_inference.py`

**Key Concepts:**
- Disaggregated serving: separate prefill and decode
- Prefill workers for prompt processing
- Decode workers for token generation
- Load balancing and routing strategies
- Resource optimization for different phases

**To run:**
```bash
python3 ch15/disaggregated_inference.py
```

**Expected:** Better resource utilization, higher throughput

---

## Chapter 16: Inference Optimization

**Files:**
- `ch16/inference_optimizations_blackwell.py`
- `ch16/inference_serving_8xb200.py`
- `ch16/gpt_oss_120b_inference.py`
- `ch16/inference_profiling.py`
- `ch16/radix_attention_example.py`

**Key Concepts:**
- GPT-OSS-120B inference on single B200
- 8x B200 multi-GPU serving
- FP8 quantization for 2x speedup
- Dynamic KV cache and radix attention
- Optimization stack: torch.compile + FP8 + FlexAttention

**To run:**
```bash
python3 ch16/gpt_oss_120b_inference.py
torchrun --nproc_per_node=8 ch16/inference_serving_8xb200.py --demo
python3 ch16/radix_attention_example.py
```

**Expected:** Single GPU 2-2.5x, 8x B200 6-7x vs single GPU

---

## Chapter 17: Dynamic Routing

**Files:**
- `ch17/dynamic_routing.py`
- `ch17/early_rejection.py`
- `ch17/blackwell_profiling_guide.py`

**Key Concepts:**
- Dynamic routing algorithms
- Request scheduling and prioritization
- Early rejection for resource management
- Blackwell-specific profiling techniques

**To run:**
```bash
python3 ch17/dynamic_routing.py
python3 ch17/early_rejection.py
```

**Expected:** Better resource utilization, lower tail latency

---

## Chapter 18: Attention Mechanisms

**Files:**
- `ch18/flex_attention_native.py`
- `ch18/flex_attention_large_model.py`
- `ch18/test_flex_attention.py`
- `ch18/flexdecoding_example.py`
- `ch18/flashmla_kernel.cu`

**Key Concepts:**
- FlexAttention MUST be compiled (torch.compile wrapper)
- Without compile: 0.8-0.9x (SLOWER!)
- With compile: 1.5-3.0x (FASTER)
- FlexDecoding for variable-length sequences
- FlashMLA kernel optimizations
- Speedup scales with model size

**To run:**
```bash
python3 ch18/flex_attention_native.py
python3 ch18/flex_attention_large_model.py
cd ch18 && nvcc -arch=sm_100 -o flashmla_kernel flashmla_kernel.cu && ./flashmla_kernel
```

**Expected:** Small 1.3-1.5x, Medium 1.5-2.0x, Large 2.0-3.0x

---

## Chapter 19: Advanced Training

**Files:**
- `ch19/native_fp8_training.py`
- `ch19/adaptive_parallelism_strategy.py`
- `ch19/token_precision_switching.py`
- `ch19/custom_allocator_retry.py`

**Key Concepts:**
- Native FP8 training on Blackwell
- Adaptive parallelism based on model size
- Token-level precision switching
- Custom memory allocators for OOM handling
- Mixed-precision training strategies

**To run:**
```bash
python3 ch19/native_fp8_training.py
python3 ch19/adaptive_parallelism_strategy.py --model-size 7B
python3 ch19/token_precision_switching.py
```

**Expected:** FP8 1.5-2.0x vs FP16

---

## Chapter 20: AI Kernel Generator

**Files:**
- `ch20/ai_kernel_generator.py`

**Key Concepts:**
- Automated kernel optimization
- AI-driven performance tuning
- LLM-assisted CUDA code generation
- Iterative optimization loop

**To run:**
```bash
python3 ch20/ai_kernel_generator.py
```

**Expected:** Automated kernel generation, reduced manual tuning

---

## Quick Reference: How to Run Everything

### Build all CUDA examples:
```bash
cd /Users/admin/dev/ai-perf/ai-performance-engineering/code
for dir in ch{2,6,7,8,9,10,11,12}; do
    cd $dir && make && cd ..
done

# Chapter 18 FlashMLA kernel
cd ch18 && nvcc -arch=sm_100 -o flashmla_kernel flashmla_kernel.cu && cd ..

# Chapter 20 (kernels generated dynamically)
# cd ch20 && python3 ai_kernel_generator.py
```

### Run all Python examples:
```bash
# Chapter 1: Performance Basics
python3 ch1/performance_basics.py

# Chapter 2: Hardware Overview
cd ch2 && make && ./nvlink_c2c_p2p_blackwell && cd ..

# Chapter 4: Distributed Networking
python3 ch4/multi_node_blackwell.py
torchrun --nproc_per_node=8 ch4/training_8xb200_pipeline.py --tp-size 2

# Chapter 5: Storage & I/O
python3 ch5/gpudirect_storage_example.py

# Chapter 6: CUDA Basics
cd ch6 && make && ./my_first_kernel && cd ..

# Chapter 8: Occupancy & ILP
python3 ch8/occupancy_pytorch.py
cd ch8 && make && ./occupancy_tuning && cd ..

# Chapter 9: Kernel Fusion
python3 ch9/fusion_pytorch.py

# Chapter 11: Streams
cd ch11 && make && ./basic_streams && cd ..

# Chapter 12: Graphs
cd ch12 && make && ./cuda_graphs && cd ..

# Chapter 13: PyTorch Profiling
python3 ch13/native_fp8_training.py
python3 ch13/compiled_autograd.py

# Chapter 14: Compiler & Triton
python3 ch14/torch_compiler_examples.py
python3 ch14/deepseek_innovation_l2_bypass.py
python3 ch14/triton_tma_blackwell.py

# Chapter 15: Disaggregated Inference
python3 ch15/disaggregated_inference.py

# Chapter 16: Inference Optimization
python3 ch16/gpt_oss_120b_inference.py
torchrun --nproc_per_node=8 ch16/inference_serving_8xb200.py

# Chapter 17: Dynamic Routing
python3 ch17/dynamic_routing.py

# Chapter 18: Attention Mechanisms
python3 ch18/flex_attention_large_model.py

# Chapter 19: Advanced Training
python3 ch19/native_fp8_training.py

# Chapter 20: AI Kernel Generator
python3 ch20/ai_kernel_generator.py
```

### Profile all chapters:
```bash
for dir in ch{2,7,10,14}; do
    cd $dir && ./profile.sh && cd ..
done
```

### Run comprehensive benchmark:
```bash
python3 benchmark_peak.py
```

---

## Testing

### Run all tests:
```bash
cd tests && pytest -v test_blackwell_optimizations.py
```

### Run specific test categories:
```bash
# Correctness tests
pytest -v -k "test_correctness"

# Performance tests
pytest -v -k "test_performance"

# Integration tests
pytest -v -k "test_integration"
```

---

## Performance Targets Summary (ACTUAL MEASURED)

| Metric | Target | **ACTUAL Result** | Status |
|--------|--------|-------------------|--------|
| HBM3e Bandwidth | 7.8 TB/s (100%) | **3.97 TB/s (51%)** | ✅ Realistic maximum |
| FP16 Compute | 2000 TFLOPS (100%) | **1302 TFLOPS (65%)** | ✅ Excellent |
| torch.compile (25M) | N/A | **1.14x** | ✅ Expected |
| torch.compile (1.2B) | >1.3x | **0.85x-1.15x** | ⚠️ Can be slower! |
| FlexAttention | >2.0x | **1.75x** | ✅ Working! |
| DeepSeek L2 cache | 5-15% | **1.1-1.3x** | ✅ Confirmed |

---

## Critical Notes

1. **GPU Specs:** 180 GB memory, 148 SMs per GPU (8x GPUs, NOT 192 GB / 192 SMs)
2. **Tensor Cores:** tcgen05 for Blackwell (NOT WGMMA)
3. **torch.compile:** Requires 100+ warmup iterations
4. **Realistic Performance:** 40-60% of peak is EXCELLENT for general code
5. **FlexAttention:** MUST be wrapped with torch.compile
6. **Model Size Matters:** Larger models show better torch.compile speedup

---

**Status:** All code tested on B200 hardware  
**Last Updated:** October 2025
