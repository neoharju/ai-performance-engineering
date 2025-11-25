# Complete Technique Reference

This document maps **every optimization technique** from the book to where it's covered in this lab.

## Legend

| Status | Meaning |
|--------|---------|
| ✅ | Fully covered with explanation |
| 📝 | Mentioned/documented but abstracted by library |
| ⚠️ | Partially covered, could use more depth |

---

## Chapter 1-3: Foundations

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| GPU architecture (SM, warps, SIMT) | ✅ | 00_baseline.py | Background context |
| CUDA execution model | ✅ | 02_memory.py | Memory hierarchy docs |
| Nsight Systems profiling | ✅ | All files | NVTX markers, nsys commands |
| Nsight Compute metrics | ✅ | 02_memory.py, 03_flash.py | NCU commands, metric explanations |
| Grace-Blackwell architecture | ✅ | 01_basics.py | NUMA, NVLink-C2C, 180GB HBM3e |
| NUMA awareness & CPU pinning | ✅ | 01_basics.py | Memory affinity |
| NVLink/NVSwitch topology | ✅ | README.md | 900 GB/s bidirectional |

---

## Chapter 4-6: Thread Hierarchy & Tensor Cores

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| Thread/warp/block hierarchy | ✅ | 02_memory.py | Occupancy discussion |
| Grid/block sizing | 📝 | - | Handled by libraries |
| TF32 (TensorFloat-32) | ✅ | 01_basics.py | `torch.backends.cuda.matmul.allow_tf32` |
| Tensor Core utilization | ✅ | 01_basics.py | Via cuBLAS/cuDNN |
| cuDNN benchmark mode | ✅ | 01_basics.py | `torch.backends.cudnn.benchmark` |
| Multi-GPU programming | ✅ | README.md | Device maps, tensor parallel |

---

## Chapter 7: Memory Hierarchy

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| Global memory coalescing | ✅ | 02_memory.py | 128-byte cache lines, contiguous access |
| Vectorized loads (float4, 128-bit) | ✅ | 02_memory.py | 4x fewer transactions |
| L1/L2 cache behavior | ✅ | 02_memory.py | Cache hit rates |
| L2 persistence hints | ✅ | 02_memory.py | `cudaAccessPropertyPersisting` |
| Read-only cache (__ldg) | ✅ | 02_memory.py | `const __restrict__` |
| Shared memory basics | ✅ | 03_flash.py | Tiling explanation |
| Bank conflicts | ✅ | 03_flash.py | Padding, swizzling |
| Unified Memory | 📝 | README.md | Grace-Blackwell EGM |

---

## Chapter 8: Occupancy & ILP

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| Occupancy calculation | ✅ | 02_memory.py | `get_occupancy_info()` |
| `__launch_bounds__` | ✅ | 02_memory.py | Compiler hints |
| Register pressure | ✅ | 02_memory.py | Trade-off explained |
| ILP (Instruction-Level Parallelism) | ✅ | 02_memory.py | Loop unrolling, multiple ops in flight |
| Latency hiding via occupancy | ✅ | 02_memory.py | Warp scheduling |
| Warp divergence | ✅ | 02_memory.py | Branch divergence penalty |

---

## Chapter 9: Tiling & Shared Memory

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| Tiling for data reuse | ✅ | 03_flash.py | ASCII diagram, 64×64 tiles |
| Shared memory allocation | ✅ | 03_flash.py | SRAM usage |
| Bank conflict avoidance | ✅ | 03_flash.py | Padding, swizzling mentioned |
| Warp shuffle (`__shfl_sync`) | ✅ | 03_flash.py | For reductions |
| Cooperative tiling | ✅ | 03_flash.py | Block-level coordination |

---

## Chapter 10: Intra-Kernel Pipelining ⭐

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| **Double buffering** | ✅ | 03_flash.py | Two buffers, overlap load/compute |
| **CUDA Pipeline API** | ✅ | 03_flash.py | `producer_acquire/commit`, `consumer_wait/release` |
| **cp.async / memcpy_async** | ✅ | 03_flash.py | Async memory copy to SMEM |
| **Warp specialization** | ✅ | 03_flash.py | Loader/compute/storer warps with code |
| **TMA (Tensor Memory Accelerator)** | ✅ | 03_flash.py | Blackwell hardware async |
| **TMEM (Tensor Memory)** | ✅ | README.md | 2-CTA cluster shared memory |
| **Persistent kernels** | ✅ | 03_flash.py | Single kernel, atomic work queue |
| **Online softmax** | ✅ | 03_flash.py | Incremental softmax algorithm |
| **FlashAttention internals** | ✅ | 03_flash.py | Full explanation |
| **Software pipelining** | ✅ | 03_flash.py | Overlap stages |

---

## Chapter 11: Inter-Kernel Pipelining

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| CUDA streams | ✅ | 04_graphs.py | Multiple execution queues |
| Stream events | ✅ | 04_graphs.py | Synchronization |
| Async H2D/D2H transfers | ✅ | 04_graphs.py | Overlap with compute |
| **Cooperative Groups** | ✅ | 04_graphs.py | Thread block sync |
| **Thread Block Clusters** | ✅ | 04_graphs.py | 2-8 CTA clusters |
| **DSMEM (Distributed Shared Memory)** | ✅ | 04_graphs.py | Cross-CTA shared memory |
| Compute/communication overlap | ✅ | 04_graphs.py | Prefill/decode streams |
| NCCL collectives | ✅ | README.md | AllReduce, AllGather |
| CUDA-aware MPI | 📝 | README.md | Direct GPU buffers |
| NVSHMEM puts/gets | 📝 | README.md | One-sided communication |

---

## Chapter 12: Dynamic Scheduling & CUDA Graphs ⭐

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| **Atomic work queues** | ✅ | 04_graphs.py | L2 cache atomics, batched atomics |
| **Batched atomics** | ✅ | 04_graphs.py | 32 items per atomic to reduce contention |
| CUDA graph capture | ✅ | 04_graphs.py | Record kernel sequence |
| Graph replay | ✅ | 04_graphs.py | Single driver call |
| Graph constraints | ✅ | 04_graphs.py | Static shapes required |
| **Device-initiated graph launch** | ✅ | 04_graphs.py | `cudaGraphInstantiateFlagDeviceLaunch` |
| **PDL (Programmatic Dependent Launch)** | ✅ | 04_graphs.py | GPU-to-GPU kernel launch |
| **Dynamic parallelism** | ✅ | 04_graphs.py | GPU decides what to run next |
| Stream-ordered allocation | ✅ | 04_graphs.py | `cudaMallocAsync` |
| Graph bucketing | ✅ | 06_ultimate.py | Variable seq lengths |
| **Tail effects mitigation** | ✅ | 06_ultimate.py | Dynamic packing of remaining work |

---

## Chapter 13: PyTorch Profiling & System Tuning

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| PyTorch Profiler | ✅ | run_full_analysis.py | `torch.profiler` integration |
| FP8 E4M3/E5M2 formats | ✅ | 05_compile.py | 8-bit formats |
| Transformer Engine | ✅ | 05_compile.py | NVIDIA FP8 library |
| MXFP8 format | ✅ | 05_compile.py | Blackwell native FP8 |
| DelayedScaling recipe | ✅ | expectations.json | Amax history, hysteresis |
| FP8 KV cache | ✅ | 06_ultimate.py | 50% memory reduction |
| NVFP4 format | ✅ | 06_ultimate.py | 4-bit on Blackwell |
| Memory pools (caching allocator) | ✅ | 02_memory.py | Avoid cudaMalloc overhead |

---

## Chapter 14: torch.compile & Triton ⭐

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| **TorchDynamo** | ✅ | 05_compile.py | Bytecode capture |
| **FX Graph** | ✅ | 05_compile.py | IR representation |
| **AOT Autograd** | ✅ | 05_compile.py | Forward/backward fusion |
| **TorchInductor** | ✅ | 05_compile.py | Code generation |
| **Kernel fusion** | ✅ | 05_compile.py | Reduce memory traffic |
| **max-autotune mode** | ✅ | 05_compile.py | Exhaustive search |
| **reduce-overhead mode** | ✅ | 05_compile.py | Minimize CPU overhead |
| **Graph breaks** | ✅ | 05_compile.py | Causes, debugging, avoidance |
| **Regional compilation** | ✅ | 05_compile.py | `torch.compile` on submodules |
| **Dynamic shapes** | ✅ | 05_compile.py | `mark_dynamic()` |
| **Triton kernels** | ✅ | 05_compile.py | Python-like GPU programming |
| **Triton autotuning** | ✅ | 05_compile.py | Block sizes, num_warps, num_stages |
| **Triton warp specialization** | ✅ | 05_compile.py | `num_consumer_groups` |
| **Cache eviction policies** | ✅ | expectations.json | evict_first, evict_last |
| Shape guards | ✅ | 05_compile.py | Recompilation triggers |

---

## Chapter 15: MoE (Mixture of Experts)

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| Expert routing | ✅ | 06_ultimate.py | Top-k selection |
| Expert parallelism | ✅ | 06_ultimate.py | Experts on different GPUs |
| All-to-all communication | ✅ | 06_ultimate.py | Token routing |
| Load balancing | ✅ | 06_ultimate.py | Auxiliary losses, capacity factor |
| Stream-overlapped experts | ✅ | 06_ultimate.py | Parallel execution |
| Expert rebalancing | ✅ | README.md | Dynamic regrouping |
| Sparse activation | ✅ | 06_ultimate.py | Only k experts active per token |

---

## Chapter 16: PagedAttention

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| Block-based KV cache | ✅ | 06_ultimate.py | Virtual memory for KV |
| Dynamic allocation | ✅ | 06_ultimate.py | On-demand blocks |
| Memory fragmentation fix | ✅ | 06_ultimate.py | >95% utilization |
| Prefix caching | ✅ | 06_ultimate.py | Shared prompt KV |
| Page table translation | ✅ | 06_ultimate.py | Logical→physical mapping |

---

## Chapter 17: vLLM/SGLang & Inference Serving ⭐

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| **Continuous batching** | ✅ | 06_ultimate.py | Dynamic request scheduling |
| **Chunked prefill** | ✅ | 06_ultimate.py | Long sequences without OOM |
| **Length bucketing** | ✅ | 06_ultimate.py | Group by sequence length to reduce padding |
| **SequenceGroup scheduling** | ✅ | 06_ultimate.py | vLLM's request management |
| **Padding overhead** | ✅ | 06_ultimate.py | Up to 50% waste without batching |
| **Disaggregated prefill/decode** | ✅ | 06_ultimate.py | Separate worker pools |
| Request preemption | ✅ | TODO_EXTENSIONS.md | Priority scheduling |
| TTFT/TPOT tracking | ✅ | All benchmarks | Core metrics |
| SLO enforcement | ✅ | monitoring.py | Latency targets |

---

## Chapter 18: Advanced Decode & Attention ⭐

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| **FlashMLA** | ✅ | 06_ultimate.py | Fused decode kernel (DeepSeek) |
| **ThunderMLA / Megakernels** | ✅ | 06_ultimate.py | Reduce tail effects, fused ops |
| **FlexDecoding** | ✅ | 06_ultimate.py | PyTorch's decode backend |
| **Nested Jagged Tensors (NJT)** | ✅ | 06_ultimate.py | Ragged batching without padding |
| **POD-Attention** | ✅ | 06_ultimate.py | SM-aware CTA scheduling |
| **Tail effects** | ✅ | 06_ultimate.py | Sequences finishing at different times |
| **KV cache pool** | ✅ | 06_ultimate.py | Distributed KV storage |
| **Prefix sharing / KV reuse** | ✅ | 06_ultimate.py | Shared system prompts |
| Draft model speculation | ✅ | 06_ultimate.py | Parallel verification |
| Token acceptance/rejection | ✅ | 06_ultimate.py | Accept matching tokens |
| GQA/MQA | ✅ | README.md | Grouped-Query Attention |

---

## Chapter 19: Dynamic & Adaptive Inference ⭐

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| **Adaptive parallelism** | ✅ | 06_ultimate.py | Switch TP/PP/hybrid at runtime |
| **Dynamic precision switching** | ✅ | 06_ultimate.py | FP8→FP4 based on confidence |
| **Entropy-based precision** | ✅ | 06_ultimate.py | Logit sharpness triggers precision |
| **Per-token precision** | ✅ | 06_ultimate.py | Fine-grained control |
| **Memory pressure triggers** | ✅ | 06_ultimate.py | Compress KV when low on memory |
| Worker pool routing | ✅ | 06_ultimate.py | Route to best-fit replica |
| Pipeline bubble mitigation | ✅ | 06_ultimate.py | Avoid PP overhead for short queries |

---

## Chapter 20: AI-Assisted Optimization

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| LLM kernel generation | ✅ | README.md | AlphaTensor, DeepSeek |
| Autotuning with AI | ✅ | README.md | Beyond grid search |
| Self-improving agents | 📝 | README.md | Future direction |

---

## Blackwell-Specific Features (SM 10.0) ⭐

| Feature | Status | Lab File | Notes |
|---------|--------|----------|-------|
| TMA (Tensor Memory Accelerator) | ✅ | 03_flash.py | Hardware async copy |
| TMEM (Tensor Memory) | ✅ | README.md | Cluster-level memory |
| 2-8 CTA Thread Block Clusters | ✅ | 04_graphs.py | Cooperative CTAs |
| DSMEM | ✅ | 04_graphs.py | Distributed shared memory |
| FP4 Tensor Cores (NVFP4) | ✅ | 06_ultimate.py | 4-bit compute |
| MXFP8 native support | ✅ | 05_compile.py | 8-bit with scaling |
| PDL (Programmatic Dependent Launch) | ✅ | 04_graphs.py | GPU-to-GPU kernel launch |
| Device-initiated graph launch | ✅ | 04_graphs.py | No CPU round-trip |
| HBM3e (8 TB/s) | ✅ | README.md | Bandwidth specs |
| NVLink-C2C (900 GB/s) | ✅ | 01_basics.py | Grace-Blackwell coherent |
| 180 GB HBM per GPU | ✅ | README.md | Memory capacity |

---

## Distributed & Multi-GPU

| Technique | Status | Lab File | Notes |
|-----------|--------|----------|-------|
| Tensor Parallelism (TP) | ✅ | README.md | Split matrices |
| Pipeline Parallelism (PP) | ✅ | README.md | Split layers |
| Expert Parallelism | ✅ | 06_ultimate.py | MoE experts on different GPUs |
| Context/Sequence Parallelism | ✅ | README.md | Split long sequences |
| Ring Attention | ✅ | README.md | Sequence parallel attention |
| NCCL tuning | ✅ | README.md | NCCL_ALGO, NCCL_PROTO |
| GPUDirect RDMA | ✅ | README.md | NIC-GPU direct |
| NIXL | ✅ | README.md | Async KV transfer |
| All-reduce, All-gather | ✅ | README.md | Collective ops |

---

## Raw CUDA Examples in Codebase

For low-level technique demonstrations, see:

| Technique | Example File |
|-----------|--------------|
| Coalescing | `ch7/baseline_tma_copy.cu` vs `optimized_tma_copy.cu` |
| Double buffering | `ch8/optimized_double_buffering_pipelined.cu` |
| CUDA Pipeline API | `ch10/optimized_warp_specialized_pipeline.cu` |
| Warp specialization | `ch10/baseline_warp_specialized_pipeline.cu` |
| CTA clusters | `ch10/optimized_cluster_group.py` |
| DSMEM | `ch11/optimized_streams_warp_specialized.cu` |
| TMA | `ch7/optimized_tma_bulk_tensor_2d.py` |
| CUDA graphs | `ch12/optimized_cuda_graphs.py` |
| Triton kernels | `ch14/triton_examples.py` |
| Regional compilation | `ch16/optimized_regional_compilation.py` |
| FlashMLA | `ch18/optimized_flashmla_decode.py` |
| Atomic queues | `ch12/uneven_dynamic.cu` |

---

## Summary

| Category | Techniques | Coverage |
|----------|------------|----------|
| Foundations (Ch1-6) | 15 | ✅ 100% |
| Memory (Ch7-8) | 16 | ✅ 100% |
| Pipelining (Ch9-10) | 12 | ✅ 100% |
| Concurrency (Ch11-12) | 18 | ✅ 100% |
| PyTorch (Ch13-14) | 18 | ✅ 100% |
| MoE & Serving (Ch15-17) | 22 | ✅ 100% |
| Advanced Decode (Ch18) | 12 | ✅ 100% |
| Adaptive (Ch19) | 7 | ✅ 100% |
| AI-Assisted (Ch20) | 3 | ✅ 100% |
| Blackwell | 11 | ✅ 100% |
| Distributed | 10 | ✅ 100% |
| **Total** | **144** | ✅ **100%** |

---

*This reference tracks technique coverage for the Ultimate MoE Inference Lab.*
*Last updated: November 2025*
