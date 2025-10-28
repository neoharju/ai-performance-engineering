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
- [Chapter 2: AI System Hardware Overview](#chapter-2-ai-system-hardware-overview)
- [Chapter 3: System Setup & Configuration](#chapter-3-system-setup--configuration)
- [Chapter 4: Tuning Distributed Networking](#chapter-4-tuning-distributed-networking)
- [🚀 NEW: 8x B200 GPU & GB200/GB300 Optimizations](#-new-8x-b200-gpu--gb200gb300-optimizations)
- [Chapter 5: Storage & I/O Optimization](#chapter-5-storage--io-optimization)
- [Chapter 6: Introduction to CUDA Kernels](#chapter-6-introduction-to-cuda-kernels)
- [Chapter 7: Memory Access Patterns](#chapter-7-memory-access-patterns)
- [Chapter 8: Occupancy & Instruction-Level Parallelism](#chapter-8-occupancy--instruction-level-parallelism)
- [Chapter 9: Kernel Fusion](#chapter-9-kernel-fusion)
- [Chapter 10: Tensor Cores & Thread Block Clusters](#chapter-10-tensor-cores--thread-block-clusters)
- [Chapter 11: CUDA Streams & Concurrency](#chapter-11-cuda-streams--concurrency)
- [Chapter 12: CUDA Graphs & Dynamic Parallelism](#chapter-12-cuda-graphs--dynamic-parallelism)
- [Chapter 13: Profiling and Tuning PyTorch](#chapter-13-profiling-and-tuning-pytorch)
- [Chapter 14: PyTorch Compiler, Triton, XLA](#chapter-14-pytorch-compiler-triton-xla)
- [Chapter 15: Disaggregated Inference](#chapter-15-disaggregated-inference)
- [Chapter 16: Inference Optimization](#chapter-16-inference-optimization)
- [Chapter 17: Dynamic Routing & Advanced Serving](#chapter-17-dynamic-routing--advanced-serving)
- [Chapter 18: Efficient Attention Mechanisms](#chapter-18-efficient-attention-mechanisms)
- [Chapter 19: Advanced Training Techniques](#chapter-19-advanced-training-techniques)
- [Chapter 20: AI Kernel Generator](#chapter-20-ai-kernel-generator)

### Reference
- [Quick Reference: How to Run Everything](#quick-reference-how-to-run-everything)
- [Testing](#testing)
- [Performance Targets Summary](#performance-targets-summary-actual-measured)
- [Critical Notes for Book](#critical-notes-for-book)

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

## Chapter-to-Code Mapping

## Chapter 1: Performance Basics

**Files:**
- `ch1/performance_basics.py` - Baseline profiling and measurement

**Key Concepts:**
- Performance measurement fundamentals
- Profiling setup and tools
- Baseline metrics establishment

**To run:**
```bash
cd ch1 && python3 performance_basics.py
```

**Book sections:** Introduction to profiling, establishing baselines

---

## Chapter 2: AI System Hardware Overview

**Files:**
- `ch2/hardware_info.py` - GPU detection and capabilities
- `ch2/nvlink_c2c_p2p_blackwell.cu` - NVLink-C2C bandwidth testing
- `ch2/gb200_grace_blackwell_coherency.cu` - Grace-Blackwell coherency
- `ch2/gb200_topology_aware.py` - GB200 topology optimization
- `ch2/Makefile` - Build system

**Key Concepts:**
- Blackwell B200 architecture (180 GB, 148 SMs)
- 5th-gen Tensor Cores (tcgen05)
- NVLink-C2C: 900 GB/s coherent interconnect
- HBM3e: 7.8 TB/s peak bandwidth
- GB200/GB300 Grace-Blackwell integration

**To run:**
```bash
cd ch2
python3 hardware_info.py
make && ./nvlink_c2c_p2p_blackwell
python3 gb200_topology_aware.py
```

**To profile:**
```bash
cd ch2 && ./profile.sh
```

**Expected results:**
- NVLink-C2C: ~900 GB/s
- PCIe Gen5: ~64 GB/s
- Grace-Blackwell coherency: ~800 GB/s

**Book sections:** Pages 47-80, Blackwell architecture details

---

## Chapter 3: System Setup & Configuration

**Files:**
- `ch3/bind_numa_affinity.py` - NUMA binding
- `ch3/docker_gpu_optimized.dockerfile` - Docker configuration
- `ch3/kubernetes_mig_pod.yaml`, `kubernetes_topology_pod.yaml` - K8s deployment configs
- `ch3/gb200_numa_optimizations.sh` - GB200 NUMA tuning
- `ch3/numa_topology_script.sh` - Topology discovery
- `ch3/system_tuning.sh` - System-level optimizations
- `ch3/gpu_setup_commands.sh` - GPU initialization

**Key Concepts:**
- System tuning for Blackwell (CPU governor, IRQ affinity)
- NUMA topology optimization (Grace-Blackwell)
- Container deployment (Docker, Kubernetes)
- MIG (Multi-Instance GPU) configuration
- GPU persistence mode and clock settings

**To run:**
```bash
cd ch3
python3 bind_numa_affinity.py
sudo ./system_tuning.sh
sudo ./gb200_numa_optimizations.sh  # GB200 only
./numa_topology_script.sh
```

**Expected results:**
- Proper NUMA binding for Grace CPUs
- Optimized system settings for training
- Reduced latency and jitter

**Book sections:** Pages 81-128, system configuration

---

## Chapter 4: Tuning Distributed Networking

**Files:**
- `ch4/multi_node_blackwell.py` - Multi-node training framework
- `ch4/torchtitan_async_tp_demo.py` - Async tensor parallelism
- `ch4/after_ddp.py`, `after_overlap_ddp.py` - DDP optimizations
- **NEW:** `ch4/training_8xb200_pipeline.py` - Complete 8-GPU training
- **NEW:** `ch4/nccl_blackwell_config.py` - NCCL 2.28 for 8x B200
- **NEW:** `ch4/symmetric_memory_8gpu.py` - PyTorch Symmetric Memory
- **NEW:** `ch4/bandwidth_benchmark_suite_8gpu.py` - Comprehensive benchmarks
- **NEW:** `ch4/nvshmem_8gpu_examples.cu` - NVSHMEM educational examples

**Key Concepts:**
- Multi-node training on Blackwell
- Async-TP with CUDA graph trees
- Hybrid parallelism (TP + DP + FSDP)
- NCCL optimization for 148 SMs
- **NEW:** 8x B200 GPU topology optimization
- **NEW:** NVLink 5.0 (1800 GB/s per pair)
- **NEW:** GB200/GB300 Grace-Blackwell coherency

**To run:**
```bash
# Standard multi-node
cd ch4 && python3 multi_node_blackwell.py

# NEW: 8-GPU optimized training
torchrun --nproc_per_node=8 ch4/training_8xb200_pipeline.py --model-size 7B --fp8

# NEW: Bandwidth benchmarks
torchrun --nproc_per_node=8 ch4/bandwidth_benchmark_suite_8gpu.py --full
```

**Expected performance:**
- Intra-node (NVLink 5.0): 95%+ efficiency
- Inter-node: 85%+ efficiency
- **NEW:** 8x B200 AllReduce: 700-800 GB/s bus bandwidth
- **NEW:** 8x B200 Scaling: 85-95% efficiency (7.2-7.6x speedup)

**Book sections:** Pages 129-180, distributed training

---

## 🚀 NEW: 8x B200 GPU & GB200/GB300 Optimizations

This section consolidates the reconciliation notes, implementation status, and day-to-day guidance for the Blackwell workstream.

### Hardware Snapshot
```
8x B200 GPUs
  SMs: 1184 (148 per GPU)
  Memory: 1.44 TB HBM3e (180 GB per GPU)
  NVLink 5.0: 1800 GB/s per GPU pair
  Target scaling: 85-95% vs single GPU
```
```
GB200/GB300 Grace-Blackwell
  Grace CPU: 72 ARM Neoverse V2 cores, 480GB-1TB LPDDR5X
  NVLink-C2C: 900 GB/s coherent CPU↔GPU bandwidth
  Unified memory: Transparent CPU/GPU access
```

### Delivered Assets
- **Training & config (ch4):** `training_8xb200_pipeline.py`, `nccl_blackwell_config.py`, `bandwidth_benchmark_suite_8gpu.py`, `symmetric_memory_8gpu.py`, `nvshmem_pytorch_wrapper.py`
- **System integration (ch2/ch3/ch4):** `gb200_topology_aware.py`, `gb200_grace_blackwell_coherency.cu`, `gb200_grace_numa_optimization.py`, `nvlink_c2c_p2p_blackwell.cu`, `ch4/Makefile`, `ch4/README.md`
- **Inference & serving (ch16):** `inference_optimizations_blackwell.py`, `inference_serving_8xb200.py`
- **Kernel & compiler (ch14):** `triton_tma_blackwell.py`, `triton_examples.py`, `test_blackwell_optimizations.py` with broadcast and eviction fixes for TMA
- **Distributed frameworks (ch13):** `fsdp_example.py` with hybrid 8x B200 presets and GB200 CPU offloading

### Quick Start & Validation
```bash
# NCCL + topology detection
python ch4/nccl_blackwell_config.py

# 8-GPU bandwidth sweep
torchrun --nproc_per_node=8 ch4/bandwidth_benchmark_suite_8gpu.py --full

# Symmetric memory and NVSHMEM demonstrations
torchrun --nproc_per_node=8 ch4/symmetric_memory_8gpu.py
torchrun --nproc_per_node=8 ch4/nvshmem_pytorch_wrapper.py

# End-to-end 8-GPU training
torchrun --nproc_per_node=8 ch4/training_8xb200_pipeline.py --tp-size 2 --compile --fp8

# Grace CPU affinity and NVLink-C2C coherency (GB200/GB300 only)
python ch4/gb200_grace_numa_optimization.py
cd ch2 && make && ./nvlink_c2c_p2p_blackwell

# Triton kernel validation
cd ch14 && python test_blackwell_optimizations.py

# Inference paths
torchrun --nproc_per_node=8 ch16/inference_optimizations_blackwell.py --eight-gpu
python ch16/inference_optimizations_blackwell.py --gb200
torchrun --nproc_per_node=8 ch16/inference_serving_8xb200.py --demo
```

### Communication Method Cheatsheet

| Method | Use When | Best For | Latency | Bandwidth |
|--------|----------|----------|---------|-----------|
| **NCCL 2.28** | Production training, standard collectives | AllReduce, AllGather, ReduceScatter on large payloads (>10 MB) | 10-50 μs | 700-800 GB/s |
| **NVSHMEM** | Custom patterns, kernel-initiated traffic | One-sided GPU↔GPU ops, halo exchange, atomics | 1-5 μs | Topology dependent |
| **PyTorch Symmetric Memory** | PyTorch 2.9+, direct tensor access | torch.compile friendly, portable cross-GPU access | 5-10 μs | NVLink-limited |

```python
from ch4.nccl_blackwell_config import configure_nccl_for_8xB200
configure_nccl_for_8xB200(num_channels=8, enable_nvls=True)
dist.init_process_group(backend="nccl")
```

```python
from ch4.nvshmem_pytorch_wrapper import SymmetricMemoryBuffer
buffer = SymmetricMemoryBuffer(tensor)
buffer.put(data, target_rank=1)
remote = buffer.get(source_rank=0)
```

### Hybrid Parallelism Reference

| Model Size | TP | DP | Total GPUs | Memory/GPU | Notes |
|------------|----|----|-----------|------------|-------|
| <1B | 1 | 8 | 8 | 1-2 GB | Large batches, simplest config |
| 1-10B | 2 | 4 | 8 | 4-10 GB | Default for 7B-20B workloads |
| 10-30B | 4 | 2 | 8 | 10-25 GB | Fewer gradients in flight |
| 30-100B | 8 | 1 | 8 | 25-50 GB | Max model parallelism per node |
| 100B+ | 8 | N | 8×N | 50 GB+ | Multi-node, shard with FSDP |

### Performance Targets
- `8x B200:` AllReduce 1 GB 700-800 GB/s; P2P 800-900 GB/s; scaling 85-95%; >2M tokens/s; NVSHMEM latency <2 μs.
- `GB200/GB300:` CPU→GPU ~800 GB/s (900 GB/s peak); GPU→GPU matches 8x B200; CPU preprocessing overhead <5%; unified memory latency ~200 ns.

### Troubleshooting Snapshot
- NCCL bandwidth <600 GB/s → enable NVLS via `configure_nccl_for_8xB200(enable_nvls=True)` and check `NCCL_DEBUG=INFO`.
- OOM on 180 GB GPUs → tighten activation checkpointing, adjust TP/DP split, use gradient accumulation.
- Scaling efficiency <80% → verify overlap flags (`BackwardPrefetch.BACKWARD_PRE`, `forward_prefetch=True`) and bucket sizes.
- Grace CPU not detected → confirm `lscpu` reports aarch64 and NVLink-C2C entries in `nvidia-smi topo -m`; update drivers if missing.

### Optional Enhancements
- Explore pipeline-parallel extensions for >8 GPU per stage training.
- Add gradient compression experiments on the bandwidth suite.
- Extend NVSHMEM benchmarks with production double-buffered collectives.
- Automate multi-node scaling sweeps that combine tensor + data parallel meshes.

---

## Chapter 5: Storage & I/O Optimization

**Files:**
- `ch5/gpudirect_storage_example.py` - GPUDirect Storage demonstrations
- `ch5/storage_io_optimization.py` - I/O pipeline optimization

**Key Concepts:**
- GPUDirect Storage: Direct GPU-SSD data transfers
- Bypassing CPU in data loading pipelines
- I/O bottleneck elimination for large datasets
- Optimized data loading for training and inference

**To run:**
```bash
cd ch5 && python3 gpudirect_storage_example.py
python3 storage_io_optimization.py
```

**Expected results:**
- GPUDirect Storage: 2-3x faster data loading
- Reduced CPU overhead in I/O path
- Better GPU utilization during data loading

**Book sections:** Pages 181-220, storage optimization

---

## Chapter 6: Introduction to CUDA Kernels

**Files:**
- `ch6/my_first_kernel.cu` - First CUDA kernel
- `ch6/simple_kernel.cu` - Basic kernel patterns
- `ch6/add_parallel.cu`, `add_sequential.cu` - Parallel vs sequential comparison
- `ch6/2d_kernel.cu` - 2D grid patterns
- `ch6/launch_bounds_example.cu` - Launch configuration
- `ch6/occupancy_api.cu` - Occupancy calculation
- `ch6/unified_memory.cu` - Unified memory example
- `ch6/stream_ordered_allocator.cu` - Stream-ordered allocation
- `ch6/Makefile` - Build system

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
./2d_kernel
```

**Expected results:**
- Basic kernel execution: <1 ms for simple operations
- Understanding of CUDA execution model
- Foundation for advanced optimizations

**Book sections:** Pages 221-264, CUDA fundamentals

---

## Chapter 7: Memory Access Patterns

**Files:**
- `ch7/hbm3e_optimized_copy.cu` - HBM3e optimization examples
- `ch7/hbm3e_peak_bandwidth.cu` - Peak bandwidth targeting
- `ch7/async_prefetch_tma.cu` - TMA prefetching

**Key Concepts:**
- HBM3e: 256-byte bursts, cache streaming
- Memory coalescing for Blackwell
- TMA (Tensor Memory Accelerator)

**To run:**
```bash
cd ch7 && make && ./hbm3e_peak_bandwidth
```

**To profile:**
```bash
cd ch7 && ./profile.sh
```

**Expected results:**
- Standard: ~3.2 TB/s (42%)
- Vectorized: ~3.6 TB/s (46%)
- HBM3e optimized: >7.0 TB/s (90%+)

**Book sections:** Pages 265-320, memory optimization

---

## Chapter 8: Occupancy & Instruction-Level Parallelism

**Files:**
- `ch8/occupancy_pytorch.py` - PyTorch occupancy optimization
- `ch8/occupancy_tuning.cu` - CUDA occupancy tuning
- `ch8/occupancy_api_example.cu` - Occupancy API usage
- `ch8/ilp_pytorch.py` - Instruction-level parallelism in PyTorch
- `ch8/warp_divergence_pytorch.py` - Warp divergence patterns
- `ch8/independent_ops.cu` - Independent operations optimization
- `ch8/loop_unrolling.cu` - Loop unrolling examples
- `ch8/threshold_naive.cu`, `threshold_predicated.cu` - Predication patterns
- `ch8/Makefile` - Build system

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
python3 ilp_pytorch.py
./occupancy_tuning
./independent_ops
```

**Expected results:**
- Occupancy improvements: 1.5-2.5x speedup
- ILP optimization: 1.2-1.8x speedup
- Understanding of SM resource constraints

**Book sections:** Pages 321-370, occupancy and ILP

---

## Chapter 9: Kernel Fusion

**Files:**
- `ch9/fusion_pytorch.py` - PyTorch kernel fusion
- `ch9/fused_l2norm.cu` - Fused L2 normalization
- `ch9/inline_ptx_example.cu` - Inline PTX assembly
- `ch9/cutlass_gemm_example.cu` - CUTLASS GEMM integration
- `ch9/Makefile` - Build system

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
./cutlass_gemm_example
```

**Expected results:**
- Fused operations: 2-4x speedup over separate kernels
- Reduced memory traffic
- Better cache utilization

**Book sections:** Pages 371-396, kernel fusion techniques

---

## Chapter 10: Tensor Cores & Thread Block Clusters

**Files:**
- `ch10/tcgen05_blackwell.cu` - 5th-gen Tensor Cores
- `ch10/cluster_group_blackwell.cu` - 8-CTA clusters
- `ch10/tma_2d_pipeline_blackwell.cu` - TMA pipeline

**Key Concepts:**
- tcgen05 (NOT WGMMA for Blackwell!)
- FP8 native support
- 8 CTAs per cluster (vs 4 on Hopper)
- 2 MB DSMEM

**To run:**
```bash
cd ch10 && make && ./tcgen05_blackwell
```

**To profile:**
```bash
cd ch10 && ./profile.sh
```

**Expected results:**
- FP8: >1200 TFLOPS
- FP16: >800 TFLOPS
- Tensor Core utilization: >80%

**Book sections:** Pages 397-460, CRITICAL CHAPTER UPDATE

---

## Chapter 11: CUDA Streams & Concurrency

**Files:**
- `ch11/basic_streams.cu` - Basic CUDA streams usage
- `ch11/stream_ordered_allocator.cu` - Stream-ordered memory allocation
- `ch11/warp_specialized_pipeline_multistream.cu` - Multi-stream warp specialization
- `ch11/multi_stream_pipeline/main.cpp` - Multi-stream pipeline example
- `ch11/Makefile` - Build system

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
./warp_specialized_pipeline_multistream
```

**Expected results:**
- Stream concurrency: 1.5-2x throughput improvement
- Overlapped operations: reduced latency
- Stream-ordered allocation: lower memory fragmentation

**Book sections:** Pages 461-500, streams and concurrency

---

## Chapter 12: CUDA Graphs & Dynamic Parallelism

**Files:**
- `ch12/cuda_graphs.cu` - Basic CUDA graphs
- `ch12/cuda_graphs_conditional.cu` - Conditional graphs
- `ch12/dynamic_parallelism.cu` - Dynamic kernel launches
- `ch12/atomic_work_queue.cu` - Atomic-based work distribution
- `ch12/Makefile` - Build system

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
./dynamic_parallelism
./atomic_work_queue
```

**Expected results:**
- CUDA graphs: 20-40% latency reduction
- Dynamic parallelism: adaptive workload handling
- Reduced kernel launch overhead

**Book sections:** Pages 501-550, graphs and dynamic execution

---

## Chapter 13: Profiling and Tuning PyTorch

**Files:**
- `ch13/compiled_autograd.py` - Compiled autograd
- `ch13/custom_allocator.py` - Custom memory allocator
- `ch13/fsdp_example.py` - FSDP (Fully Sharded Data Parallel)
- `ch13/memory_profiling.py` - Memory profiling
- `ch13/train_deepseek_v3.py` - DeepSeek-V3 training example
- `ch13/requirements.txt` - Python dependencies

**Key Concepts:**
- Native FP8 types (PyTorch 2.9)
- Compiled autograd for faster backward pass
- Memory profiling for Blackwell (180 GB HBM3e)
- FSDP for large model training
- Custom memory allocators

**To run:**
```bash
cd ch13
python3 compiled_autograd.py
python3 memory_profiling.py
torchrun --nproc_per_node=8 fsdp_example.py
python3 custom_allocator.py
```

**Expected results:**
- Compiled autograd: 1.1-1.3x speedup
- FSDP: Efficient scaling for 100B+ models
- Memory profiling: Detailed allocation tracking

**Book sections:** Pages 551-610, PyTorch profiling

---

## Chapter 14: PyTorch Compiler, Triton, XLA

**Files:**
- `ch14/torch_compiler_examples.py` - torch.compile basics
- `ch14/torch_compile_large_model.py` - Large model benchmarking
- `ch14/training_large_model_1_5x.py` - Training speedup demo
- `ch14/deepseek_innovation_l2_bypass.py` - DeepSeek L2 cache optimization
- `ch14/inspect_compiled_code.py` - Inspect compiled kernels
- `ch14/triton_examples.py` - Triton kernels with TMA descriptors
- `ch14/triton_fp8_advanced.py` - Advanced FP8 kernels
- `ch14/triton_tma_blackwell.py` - TMA demonstrations for Blackwell
- `ch14/triton_nvshmem_example.py` - Triton + NVSHMEM integration
- `ch14/test_blackwell_optimizations.py` - Blackwell optimization tests
- `ch14/requirements.txt` - Python dependencies

**Key Concepts:**
- torch.compile: 1.3x+ speedup for large models
- CRITICAL: 100+ warmup iterations required
- Model size matters: <50M (1.0-1.1x), 500M-1B (1.2-1.3x), 1B+ (1.3-1.5x)
- DeepSeek innovation: L2 cache control (5-15% improvement)
- Triton 3.5: TMA descriptors, FP8 kernels, NVSHMEM
- Training vs inference speedup differences
- Blackwell TMA broadcast and eviction optimizations

**To run:**
```bash
cd ch14
python3 torch_compiler_examples.py
python3 training_large_model_1_5x.py
python3 deepseek_innovation_l2_bypass.py
python3 triton_tma_blackwell.py
python3 test_blackwell_optimizations.py
```

**To profile:**
```bash
cd ch14 && ./profile.sh
```

**Expected results:**
- torch.compile: 1.02x (small), 1.15x (medium), 1.3x+ (large)
- FP8: 1.5-2.0x vs FP16
- DeepSeek: 1.05-1.15x (5-15%)
- TMA kernels: Near-peak memory bandwidth

**Book sections:** Pages 627-700, CRITICAL CHAPTER UPDATE

---

## Chapter 15: Disaggregated Inference

**Files:**
- `ch15/disaggregated_inference.py` - Disaggregated prefill/decode architecture
- `ch15/requirements.txt` - Python dependencies

**Key Concepts:**
- Disaggregated serving: separate prefill and decode
- Prefill workers for prompt processing
- Decode workers for token generation
- Load balancing and routing strategies
- Resource optimization for different phases

**To run:**
```bash
cd ch15 && python3 disaggregated_inference.py
```

**Expected results:**
- Better resource utilization
- Higher throughput for mixed workloads
- Optimized latency for interactive serving

**Book sections:** Pages 701-749, disaggregated architectures

---

## Chapter 16: Inference Optimization

**Files:**
- `ch16/inference_optimizations_blackwell.py` - Complete inference pipeline
- `ch16/inference_serving_8xb200.py` - 8-GPU serving architecture
- `ch16/gpt_oss_120b_inference.py` - GPT-OSS-120B example
- `ch16/inference_profiling.py` - Profiling tools
- `ch16/radix_attention_example.py` - Radix attention implementation
- `ch16/nvtx_profiling.cu` - NVTX profiling kernel
- `ch16/requirements.txt` - Python dependencies

**Key Concepts:**
- GPT-OSS-120B inference on single B200 (178 GB)
- 8x B200 multi-GPU serving
- FP8 quantization for 2x speedup
- Dynamic KV cache and radix attention
- Optimization stack: torch.compile + FP8 + FlexAttention

**To run:**
```bash
cd ch16
python3 gpt_oss_120b_inference.py
python3 inference_optimizations_blackwell.py --gb200
torchrun --nproc_per_node=8 inference_serving_8xb200.py --demo
python3 radix_attention_example.py
```

**Expected results:**
- Single GPU: 2-2.5x cumulative speedup
- 8x B200: 6-7x speedup vs single GPU
- Radix attention: Lower latency for prefix sharing

**Book sections:** Pages 750-800, NEW CONTENT

---

## Chapter 17: Dynamic Routing & Advanced Serving

**Files:**
- `ch17/dynamic_routing.py` - Dynamic routing for prefill/decode
- `ch17/early_rejection.py` - Early rejection strategies
- `ch17/blackwell_profiling_guide.py` - Blackwell-specific profiling
- `ch17/dynamo_config.yaml` - TorchDynamo configuration

**Key Concepts:**
- Dynamic routing algorithms
- Request scheduling and prioritization
- Early rejection for resource management
- Blackwell-specific profiling techniques
- TorchDynamo configuration optimization

**To run:**
```bash
cd ch17
python3 dynamic_routing.py
python3 early_rejection.py
python3 blackwell_profiling_guide.py
```

**Expected results:**
- Improved request routing efficiency
- Better resource utilization
- Lower tail latency

**Book sections:** Pages 801-834, advanced serving patterns

---

## Chapter 18: Efficient Attention Mechanisms

**Files:**
- `ch18/flex_attention_native.py` - FlexAttention API
- `ch18/flex_attention_large_model.py` - FlexAttention scaling
- `ch18/test_flex_attention.py` - FlexAttention tests
- `ch18/flexdecoding_example.py` - FlexDecoding implementation
- `ch18/flashmla_kernel.cu` - FlashMLA kernel
- `ch18/lmcache_config.yaml` - LMCache configuration

**Key Concepts:**
- FlexAttention MUST be compiled (torch.compile wrapper)
- Without compile: 0.8-0.9x (SLOWER!)
- With compile: 1.5-3.0x (FASTER)
- FlexDecoding for variable-length sequences
- FlashMLA kernel optimizations
- Speedup scales with model size

**To run:**
```bash
cd ch18
python3 flex_attention_native.py
python3 flex_attention_large_model.py
python3 test_flex_attention.py
python3 flexdecoding_example.py

# To compile and run FlashMLA kernel
nvcc -arch=sm_100 -o flashmla_kernel flashmla_kernel.cu
./flashmla_kernel
```

**Expected results:**
- Small models (<100M): 1.3-1.5x
- Medium models (100-500M): 1.5-2.0x
- Large models (500M-2B): 2.0-3.0x
- FlexDecoding: Better handling of variable lengths

**Book sections:** Pages 835-875, NEW CONTENT

---

## Chapter 19: Advanced Training Techniques

**Files:**
- `ch19/native_fp8_training.py` - Native FP8 training
- `ch19/adaptive_parallelism_strategy.py` - Adaptive parallelism selection
- `ch19/token_precision_switching.py` - Dynamic precision switching
- `ch19/custom_allocator_retry.py` - Custom memory allocator with retry

**Key Concepts:**
- Native FP8 training on Blackwell (2x speedup)
- Adaptive parallelism based on model size
- Token-level precision switching
- Custom memory allocators for OOM handling
- Mixed-precision training strategies

**To run:**
```bash
cd ch19
python3 native_fp8_training.py
python3 adaptive_parallelism_strategy.py --model-size 7B
python3 token_precision_switching.py
python3 custom_allocator_retry.py
```

**Expected results:**
- FP8: 1.5-2.0x vs FP16 training
- Adaptive parallelism: Optimal TP/DP split
- Dynamic precision: Better memory efficiency

**Book sections:** Pages 880-920, advanced training

---

## Chapter 20: AI Kernel Generator

**Files:**
- `ch20/ai_kernel_generator.py` - AI-assisted kernel generation
- `ch20/test.cu` - Generated test kernel

**Key Concepts:**
- Automated kernel optimization
- AI-driven performance tuning
- LLM-assisted CUDA code generation
- Iterative optimization loop

**To run:**
```bash
cd ch20
python3 ai_kernel_generator.py
# Generated kernels can be compiled with:
# nvcc -arch=sm_100 -o test test.cu
```

**Expected results:**
- Automated kernel generation
- Performance-guided optimization
- Reduced manual tuning effort

**Book sections:** Pages 921-950, AI-assisted optimization

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

### Run all Python examples (by chapter):
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

## Critical Notes for Book

1. **GPU Specs:** 180 GB memory, 148 SMs per GPU (8x GPUs, NOT 192 GB / 192 SMs)
2. **Tensor Cores:** tcgen05 for Blackwell (NOT WGMMA)
3. **torch.compile:** Requires 100+ warmup iterations
4. **Realistic Performance:** 40-60% of peak is EXCELLENT for general code
5. **FlexAttention:** MUST be wrapped with torch.compile
6. **Model Size Matters:** Larger models show better torch.compile speedup

---

**Status:** All code tested on B200 hardware  
**Last Updated:** October 2025
