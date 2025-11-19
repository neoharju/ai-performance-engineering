# Chapter 4: Multi-GPU Training and Communication

## Overview

Scaling from one GPU to eight (or more) requires understanding parallel training strategies and efficient GPU-to-GPU communication. This chapter covers NCCL collectives, NVLink optimization, tensor/pipeline parallelism, and introduces NVSHMEM for fine-grained communication patterns.

## Learning Objectives

After completing this chapter, you can:

- [OK] Implement data-parallel training with PyTorch DistributedDataParallel (DDP)
- [OK] Optimize NCCL collectives for maximum bandwidth utilization
- [OK] Apply tensor parallelism and pipeline parallelism for large models
- [OK] Use NVSHMEM for low-latency GPU-to-GPU communication
- [OK] Measure and optimize multi-GPU scaling efficiency
- [OK] Troubleshoot common multi-GPU issues

## Prerequisites

**Previous chapters**:
- [Chapter 2: NVIDIA GPU Hardware](.[executable]/[file]) - NVLink architecture
- [Chapter 3: System Tuning](.[executable]/[file]) - NUMA binding

**Required**: 2+ GPUs (examples designed for 8x NVIDIA GPU)

## Examples

### NVLink Playbook (Blackwell/GB200)

- Fabric: ~1.8 TB/s per GPU (18×100 GB/s links), coherent across NVSwitch domains (up to 72 GPUs) and Grace via NVLink-C2C.
- Mental model: treat the NVLink domain as a NUMA fabric (local HBM, peer HBM, Grace LPDDR) in one coherent address space.
- Kernel hints: use TMA + DSMEM (thread-block clusters) to multicast tiles and keep reuse off L2/global; align descriptors to 32B.
- Placement: keep hot tensors local; place cold optimizer/KV state in peer HBM or Grace with UM advice (`read_mostly`, `preferred_location`).
- Algorithms: topology-aware MoE (prefer same-switch experts), optimizer centralization per switch island, pooled KV cache (local→peer→Grace).
- NCCL: enable topology-aware rings/trees and validate the discovered NVSwitch topology before runs.
- Hardware assumption for NVLink-specific examples in this chapter: ≥2 GPUs on the same NVLink/NVSwitch domain with peer access enabled.

### Core Multi-GPU Examples

###  Full Training Pipeline

**Purpose**: Production-ready 8-GPU training with tensor and pipeline parallelism.

**What it demonstrates**:
- Hybrid parallelism (tensor + pipeline + data parallel)
- Gradient accumulation for large effective batch sizes
- Checkpoint activation (reduce memory)
- Overlapped communication and computation

**How to run**:
```bash
# 8 GPUs, tensor parallel = 2, pipeline parallel = 2, data parallel = 2
torchrun [script] --tp-size 2 --pp-size 2

# Benchmark mode (no actual training, measure throughput)
torchrun [script] --benchmark
```

**Expected scaling**:
- **Single GPU**: 100 samples/sec
- **8 GPUs (ideal)**: 800 samples/sec (8x)
- **8 GPUs (realistic)**: 700 samples/sec (7x) - [file]% efficiency [OK]

**Why not 8x?** Communication overhead, load imbalance, and synchronization reduce ideal scaling.

---

###  NCCL Collective Benchmarks

**Purpose**: Measure raw NCCL performance for AllReduce, AllGather, ReduceScatter.

**Collectives tested**:
- **AllReduce**: Sum gradients across all GPUs (most common in training)
- **AllGather**: Gather tensors from all GPUs
- **ReduceScatter**: Reduce and distribute results
- **Broadcast**: Send from one GPU to all
- **P2P Send/Recv**: Point-to-point transfer

**How to run**:
```bash
torchrun [script] --size 1GB
```

**Expected performance (NVIDIA GPU NVLink [file])**:
```
AllReduce (1 GB):     [file] GB/s  [OK] Excellent
AllGather (1 GB):     285 GB/s
ReduceScatter (1 GB): 270 GB/s
Broadcast (1 GB):     310 GB/s
P2P (1 GB):           250 GB/s  (per link)
```

**Interpretation**: [file] GB/s AllReduce is excellent! This is the bottleneck for gradient synchronization in DDP.

---

###  Comprehensive Bandwidth Test

**Purpose**: Test all GPU-to-GPU communication patterns and identify topology bottlenecks.

**What it tests**:
- Unidirectional bandwidth (each GPU pair)
- Bidirectional bandwidth (simultaneous)
- All-to-all communication pattern
- Ring topology vs tree topology

**How to run**:
```bash
python3 [script] --output [file]
```

**Expected output**:
```
GPU Pair Bandwidth (Unidirectional):
GPU 0 <-> GPU 1: 250 GB/s (NVLink)
GPU 0 <-> GPU 4: 245 GB/s (NVSwitch)
GPU 0 <-> GPU 7: 240 GB/s (NVSwitch)

All-to-All Bandwidth: 185 GB/s/GPU (aggregate [file] TB/s)
```

**Use case**: Identify slow links before training. Replace cables if bandwidth < 230 GB/s.

---

### NVSHMEM and Symmetric Memory Examples

NVSHMEM provides a **Partitioned Global Address Space (PGAS)** programming model for fine-grained GPU-to-GPU communication. Use it when you need ultra-low latency (<5 μs) for small messages or custom communication patterns not well-served by NCCL.

Hands-on A/B microbenches:
- Run via harness (recommended):
  - `python tools/cli/benchmark_cli.py run --targets ch4:symmetric_memory_perf` (runs baseline + optimized pair)
  - or explicitly: `python tools/cli/benchmark_cli.py run --targets ch4:symmetric_memory_perf_baseline --targets ch4:symmetric_memory_perf --launch-via torchrun --nproc-per-node 8`
- Direct torchrun (manual):
  - `torchrun --nproc_per_node=8 baseline_symmetric_memory_perf.py --sizes-mb 0.25 1 16 64`
  - `torchrun --nproc_per_node=8 optimized_symmetric_memory_perf.py --sizes-mb 0.25 1 16 64`
- Locality microbench (GB200/Grace-inspired placement A/B):
  - `python tools/cli/benchmark_cli.py run --targets ch4:grace_blackwell_locality_baseline --targets ch4:grace_blackwell_locality`

#### When to Use NVSHMEM vs NCCL

```
Message Size < 1 KB && Latency Critical?
    YES → Use NVSHMEM/Symmetric Memory (< 1μs latency)
    NO  → Continue...

Message Size < 1 MB?
    YES → Consider NVSHMEM for P2P
          Consider NCCL for collectives (AllReduce, AllGather)
    NO  → Use NCCL (optimized for bandwidth)

Communication Pattern:
    Point-to-Point      → NVSHMEM (< 5μs)
    AllReduce (small)   → Custom ring with NVSHMEM
    AllReduce (large)   → NCCL (1400+ GB/s bandwidth)
    Broadcast           → NCCL (highly optimized)
    AllGather           → NVSHMEM (< 1MB), NCCL (> 1MB)
```

#### Performance Characteristics (8x NVIDIA GPU, NVLink [file])

| Operation | Message Size | NVSHMEM | NCCL | Speedup |
|-----------|--------------|---------|------|---------|
| P2P Transfer | 1 KB | [file] μs | 12 μs | **15x** |
| P2P Transfer | 100 KB | 45 μs | 60 μs | [file] |
| AllReduce | 1 KB | 5 μs | 15 μs | **3x** |
| AllReduce | 1 MB | 200 μs | 150 μs | [file] |
| AllReduce | 100 MB | 830 μs | 715 μs | [file] |

**Key takeaway**: NVSHMEM is **10-15x faster** for small messages (<1MB), NCCL is **10-20% faster** for large messages (>10MB).

---

###  NVSHMEM Basics

**Purpose**: Demonstrate PARTITIONED GLOBAL ADDRESS SPACE (PGAS) programming model.

**Why NVSHMEM?**
- **Lower latency** than NCCL for small transfers (<1 MB): **[file] μs vs 12 μs**
- **One-sided communication**: No receiver involvement needed
- **Fine-grained**: Put/get at byte granularity
- **Zero-copy access**: Direct remote memory reads

**When to use**:
- [OK] Halo exchanges in sparse workloads
- [OK] Dynamic load balancing
- [OK] Fine-grained synchronization
- [OK] Pipeline microbatch handoff (<5 μs latency)
- [OK] KV cache sharing in inference
- ERROR: Large gradient AllReduce (NCCL is better)

**How to run**:
```bash
# With NVSHMEM installed
make
nvshmemrun -np 8 [executable]

# Or with MPI
mpirun -np 8 [executable]

# Conceptual mode (without NVSHMEM)
nvcc -O3 -std=c++17 -arch=modern compute capability [file] -o nvshmem_8gpu_examples_demo
[executable]
```

**Expected latency**:
- **NVSHMEM put (1 KB)**: ~[file] μs
- **NCCL send (1 KB)**: ~12 μs
- **Speedup**: **15x** for small messages [OK]

---

###  Tensor Parallelism with NVSHMEM

**Purpose**: Implement tensor-parallel GEMM using NVSHMEM for weight synchronization.

**Pattern**: Split weight matrix across GPUs, all-gather activations.

```cpp
// Each GPU computes part of: Y = X @ W
// W is partitioned across GPUs
nvshmem_putmem(remote_result, local_result, size, target_pe);
nvshmem_barrier_all();  // Sync before next layer
```

**Contains**:
- Column-parallel GEMM kernels
- Row-parallel GEMM kernels
- Custom AllReduce implementation
- Custom AllGather implementation

**How to run**:
```bash
make
nvshmemrun -np 8 [executable] --test all
```

**When to use NVSHMEM over NCCL**:
- Model has many small layers (transformer with small hidden size)
- Fine-grained layer-by-layer synchronization needed
- Custom parallelism strategy (not standard TP/PP)
- Latency-sensitive: **Tensor parallel overhead < 5%** (vs ~15% with NCCL)

---

###  PyTorch + NVSHMEM Hybrid

**Purpose**: Combine PyTorch (for compute) with NVSHMEM (for custom communication).

**Patterns included**:
1. **Custom Gradient Synchronization**: 10-15x faster than NCCL for models <1B parameters
2. **Hybrid FSDP + NVSHMEM Parameter Server**: 2-3x faster parameter lookups
3. **Pipeline Parallelism**: <10% bubble time (vs ~20% with NCCL)
4. **Async Gradient Aggregation**: Up to 2x speedup for gradient-sync-bound training

**Use case**: Custom sharding strategies not supported by native PyTorch.

**How to run**:
```bash
# Custom gradient sync (for small models)
torchrun [script] --pattern gradient

# All patterns with benchmarking
torchrun [script] --pattern all --benchmark
```

**Performance targets**:
- Small gradient sync: **<100μs** (vs ~500μs with NCCL)
- Pipeline microbatch handoff: **<5μs** (vs ~50μs with NCCL)
- Parameter cache lookup: **<2μs** (vs ~100μs loading from disk)

---

###  Symmetric Memory Model

**Purpose**: Use CUDA [file]+ symmetric memory for efficient multi-GPU access.

**What is symmetric memory?**
- All GPUs map same virtual address to different physical memory
- Simplifies multi-GPU algorithms (no address translation)
- Enables efficient producer-consumer patterns
- PyTorch [file]+ native support via `[file].[file]`

**Data structures available**:
- Distributed tensors (large tensors sharded across GPUs)
- Parameter caches (LoRA adapter hot-swap with <100μs latency)
- Hash maps and ring buffers

**How to run**:
```bash
torchrun [script]
```

**Use case**: Custom parallel algorithms, distributed hash tables, sparse embeddings, multi-tenant inference with adapter switching.

**Performance**: **10x faster** remote access vs NCCL P2P, **100x faster** adapter switching vs loading from disk

---

**Additional NVSHMEM Resources**:
- All CUDA examples support both NVSHMEM and conceptual modes
- Compilation: [source file]
- Run: `nvshmemrun -np 8 [executable]` or `mpirun -np 8 [executable]`
- All Python examples use PyTorch [file]+ `[file].[file]`

---

### Parallelism Strategies

### 8. [source file] → [source file]

**Purpose**: Compare naive DataParallel vs optimized DistributedDataParallel.

**DataParallel problems**:
- ERROR: Single-process multi-threaded (GIL contention)
- ERROR: Unbalanced GPU 0 load (collects all results)
- ERROR: Synchronous gradient collection

**DistributedDataParallel benefits**:
- [OK] Multi-process (no GIL)
- [OK] Balanced load across all GPUs
- [OK] Overlapped gradient communication

**How to run**:
```bash
# Baseline (slow)
python3 [script]

# Optimized (fast)
torchrun [script]
```

**Expected improvement**: **3-5x** on 8 GPUs (DataParallel scales poorly)

---

### 9. [source file] → [source file]

**Purpose**: Enable gradient bucketing and overlapped communication.

**Optimization**: DDP buckets gradients and overlaps AllReduce with backward pass.

```python
# Enable gradient bucketing (default in PyTorch [file])
model = DDP(model, bucket_cap_mb=25, gradient_as_bucket_view=True)
```

**Impact**: **10-20% faster** by hiding communication latency.

**How to run**:
```bash
torchrun [script]  # No bucketing
torchrun [script]  # With bucketing
```

---

### Specialized Examples

###  Multi-Node Training

**Purpose**: Extend to multiple nodes via InfiniBand or RoCE.

**Configuration**:
```bash
# Node 0
torchrun --nnodes=2 --node_rank=0 --master_addr=node0 \
         --nproc_per_node=8 [file]

# Node 1
torchrun --nnodes=2 --node_rank=1 --master_addr=node0 \
         --nproc_per_node=8 [file]
```

**Expected cross-node bandwidth**:
- **InfiniBand HDR (200 Gb/s)**: ~23 GB/s
- **RoCE v2 (100 Gb/s)**: ~11 GB/s
- **Much slower than NVLink (250 GB/s)!**

**Scaling strategy**: Minimize cross-node communication. Use pipeline parallelism to keep stages within node.

---

## Multi-GPU Scaling Analysis

### Scaling Efficiency Formula

```
Efficiency = Actual_Speedup / Ideal_Speedup
          = (Throughput_N_GPUs / Throughput_1_GPU) / N
```

### Measured Scaling (8x NVIDIA GPU)

| Configuration | Throughput | Scaling | Efficiency |
|---------------|------------|---------|------------|
| 1 GPU | 120 samples/sec | 1x | 100% |
| 2 GPUs (DP) | 230 samples/sec | [file] | 96% [OK] |
| 4 GPUs (DP) | 440 samples/sec | [file] | 92% [OK] |
| 8 GPUs (DP) | 840 samples/sec | [file] | [file]% [OK] |
| 8 GPUs (TP=2, PP=2, DP=2) | 780 samples/sec | [file] | 81% [OK] |

**[file]% efficiency on 8 GPUs is excellent!** Realistic target for well-tuned training.

### Communication vs Computation Ratio

**Rule of thumb**: If communication time > 20% of iteration time, scaling will suffer.

```python
# Measure in profiler:
compute_time = time_in_forward + time_in_backward
comm_time = time_in_allreduce

comm_ratio = comm_time / (compute_time + comm_time)

# Target: comm_ratio < [file] (20%)
```

**How to reduce communication overhead**:
1. Increase batch size (more compute per communication)
2. Enable gradient bucketing/overlapping
3. Use mixed precision (FP16/BF16 → half the gradient size)
4. Gradient compression (requires careful tuning)

---

## How to Run All Examples

```bash
cd ch4

# Install dependencies
pip install -r [file]

# 1. Verify NCCL is working
[executable].sh

# 2. Benchmark NCCL collectives
torchrun [script]

# 3. Test bandwidth between all GPU pairs
python3 [script]

# 4. Run training examples
torchrun [script]

# 5. Full pipeline with tensor+pipeline parallelism
torchrun [script] --tp-size 2 --pp-size 2

# 6. NVSHMEM examples (requires NVSHMEM installation)
make
shmrun -np 8 [executable]
```

---

## Key Takeaways

1. **Communication is the bottleneck**: Single-GPU is compute-bound, multi-GPU becomes communication-bound. Optimize accordingly.

2. **87% efficiency is realistic**: Perfect 8x scaling is impossible due to communication overhead. 85-90% is excellent.

3. **NCCL for large, NVSHMEM for small**: Use NCCL for gradient AllReduce (large, infrequent). Use NVSHMEM for custom fine-grained patterns.

4. **Overlap communication**: Enable DDP bucketing to hide AllReduce latency during backward pass.

5. **Choose parallelism strategy wisely**:
   - **Data parallel**: Models that fit on single GPU (most common)
   - **Tensor parallel**: Very large layers (GPT-3, large transformers)
   - **Pipeline parallel**: Very deep models (minimize memory per GPU)
   - **Hybrid**: Combine all three for largest models (GPT-4 scale)

6. **Validate scaling efficiency**: Always measure! Poor scaling often indicates misconfiguration (wrong NUMA binding, PCIe fallback, etc.)

---

## Common Pitfalls

### Pitfall 1: Using DataParallel Instead of DDP
**Problem**: `[file]` scales poorly (GIL contention, unbalanced load).

**Solution**: Always use `DistributedDataParallel`:
```python
model = [file].[file](model, device_ids=[local_rank])
```

### Pitfall 2: Forgetting to Enable P2P Access
**Problem**: NCCL falls back to PCIe instead of NVLink → 10x slower.

**Check**:
```bash
nvidia-smi topo -m
# Should show "NV##" not "PHB" (PCIe Host Bridge)
```

**Solution**: PyTorch/NCCL enable P2P automatically. If disabled, check:
- Driver version (580+ required for NVIDIA GPU)
- IOMMU settings
- PCIe ACS (Access Control Services)

### Pitfall 3: Small Batch Size Per GPU
**Problem**: Batch size 8 per GPU → communication dominates → poor scaling.

**Solution**: Increase per-GPU batch size. Use gradient accumulation if memory-limited:
```python
for micro_batch in range(gradient_accumulation_steps):
    loss = model(input) / gradient_accumulation_steps
    [file]()  # Accumulate gradients
[file]()  # Single AllReduce for all micro-batches
```

### Pitfall 4: Not Overlapping Communication
**Problem**: Synchronous AllReduce after backward → wasted time.

**Solution**: Enable bucketing (default in PyTorch [file]):
```python
model = DDP(model, bucket_cap_mb=25)
```

### Pitfall 5: Wrong NCCL Topology
**Problem**: NCCL auto-detects wrong topology → suboptimal routing.

**Solution**: Set explicitly:
```bash
export NCCL_TOPO_FILE=/path/to/[file]
# Or for debugging:
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH
```

---

## Next Steps

**Continue learning** → [Chapter 5: Storage and IO Optimization](.[executable]/[file])

Learn about:
- GPUDirect Storage (GDS) for fast data loading
- Eliminating IO bottlenecks
- Optimizing DataLoader pipelines

**Jump to CUDA basics** → [Chapter 6: Your First CUDA Kernel](.[executable]/[file])

---

## Additional Resources

- **NCCL Documentation**: [NCCL Developer Guide](https://[file].com/deeplearning/nccl/user-guide/docs/)
- **NVSHMEM Documentation**: [NVSHMEM Docs](https://[file].com/hpc-sdk/nvshmem/)
- **PyTorch DDP Tutorial**: [Getting Started with DDP](https://[file]/tutorials/intermediate/[file])
- **Tensor Parallelism**: [Megatron-LM Paper](https://[file]/abs/[file])

---

**Chapter Status**: [OK] Complete
