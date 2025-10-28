# Chapter 4: Multi-GPU Communication and Distributed Training

Comprehensive examples for 8x B200 GPU configurations and GB200/GB300 Grace-Blackwell superchips.

## Overview

This chapter covers:
- NCCL 2.28 optimizations for 8x B200
- NVSHMEM for direct GPU-GPU communication
- PyTorch Symmetric Memory (PyTorch 2.9+)
- GB200/GB300 CPU-GPU coherency
- Complete training pipelines with hybrid parallelism

## Quick Start

### 8-GPU Training (Recommended)

```bash
# Complete training pipeline with all optimizations
torchrun --nproc_per_node=8 training_8xb200_pipeline.py \
    --model-size 7B --batch-size 4 --compile --fp8

# NCCL benchmark to validate setup
torchrun --nproc_per_node=8 nccl_benchmark.py

# Bandwidth benchmark suite
torchrun --nproc_per_node=8 bandwidth_benchmark_suite_8gpu.py --full
```

### GB200/GB300 Superchip

```bash
# Test CPU-GPU coherency (900 GB/s via NVLink-C2C)
cd ../ch2
./gb200_coherency

# Topology-aware placement
python gb200_topology_aware.py

# NUMA optimizations (requires root)
sudo ../ch3/gb200_numa_optimizations.sh --apply
```

## File Organization

### Python Examples

| File | Description | GPUs | Notes |
|------|-------------|------|-------|
| `nccl_blackwell_config.py` | NCCL 2.28 configuration | Any | Automatic 8-GPU detection |
| `nccl_benchmark.py` | NCCL performance testing | 2+ | Collectives benchmark |
| `training_8xb200_pipeline.py` | Complete training example | 8 | TP+DP+FP8+compile |
| `symmetric_memory_8gpu.py` | PyTorch Symmetric Memory | 2-8 | Direct GPU-GPU access |
| `bandwidth_benchmark_suite_8gpu.py` | Comprehensive benchmarks | 8 | P2P matrix, collectives |
| `multi_node_blackwell.py` | Multi-node training | 16+ | Hybrid parallelism |

### CUDA/NVSHMEM Examples

| File | Description | Requires | Compile |
|------|-------------|----------|---------|
| `nvshmem_8gpu_examples.cu` | Educational NVSHMEM | NVSHMEM 3.4+ | `make nvshmem` |

### Support Files

- `Makefile` - Build CUDA examples
- `README.md` - This file

## Decision Tree: Which Communication Method?

### Use NCCL when:
- ✅ Standard collectives (AllReduce, AllGather, ReduceScatter)
- ✅ Large messages (>1MB)
- ✅ Production training (heavily optimized)
- ✅ Multi-node communication
- ✅ Automatic topology detection

**Example:**
```python
from ch4.nccl_blackwell_config import configure_nccl_for_8xB200
configure_nccl_for_8xB200(num_channels=8)
dist.init_process_group(backend='nccl')
dist.all_reduce(tensor)  # Optimized for 8 GPUs
```

### Use NVSHMEM when:
- ✅ Custom communication patterns
- ✅ Kernel-initiated communication (no CPU)
- ✅ Ultra-low latency (<1 μs for small messages)
- ✅ Direct buffer access from kernels
- ⚠️ Requires NVSHMEM 3.4+ installation

**Example:**
```cuda
// In CUDA kernel
nvshmem_float_put(dest, source, count, target_pe);
```

### Use PyTorch Symmetric Memory when:
- ✅ PyTorch 2.9+ with custom kernels
- ✅ Direct cross-GPU memory access
- ✅ Integration with torch.compile
- ✅ Portable across systems
- ⚠️ May not be available in all PyTorch builds

**Example:**
```python
sym_mem = torch.distributed.nn.SymmetricMemory(tensor, group=dist.group.WORLD)
remote_buffer = sym_mem.get_buffer(src_rank=0)  # Direct access
```

## 8x B200 Best Practices

### Hardware Configuration

```
8x Blackwell B200 GPUs:
  ├─ Total SMs: 1184 (148 per GPU)
  ├─ Total Memory: 1.44 TB HBM3e (180 GB per GPU)
  ├─ Aggregate Bandwidth: 62.4 TB/s (7.8 TB/s per GPU)
  ├─ NVLink 5.0: 1800 GB/s bidirectional per GPU pair
  └─ Power: 11.2 kW total (1400W per GPU)

Topology Options:
  ├─ NVSwitch: All-to-all connectivity (best for collectives)
  └─ Direct NVLink: Pairwise connections (good for ring patterns)
```

### NCCL Configuration

```python
# Automatic configuration based on GPU count
from ch4.nccl_blackwell_config import configure_nccl_for_8xB200

# For 8 GPUs
configure_nccl_for_8xB200(
    num_channels=8,        # 4, 8, or 16 based on model size
    enable_nvls=True,      # NVLink Sharp for 8-GPU
    enable_tce=True,       # Tensor Core Engine
    enable_nvlink_c2c=True # For GB200/GB300
)
```

**Channel Tuning Guide:**
- Small models (<1B params): `num_channels=4`
- Medium models (1-10B): `num_channels=8` (default)
- Large models (>10B): `num_channels=16`

### Hybrid Parallelism

```python
# Recommended configurations for 8 GPUs:

# Config 1: Balanced (most models)
TP = 2  # Tensor Parallel (intra-node via NVLink)
DP = 4  # Data Parallel (inter-node or intra-node)
# → Good for: 7B-13B models

# Config 2: Maximum TP (very large models)
TP = 4
DP = 2
# → Good for: 30B+ models, limited batch size

# Config 3: Maximum DP (small models, large batches)
TP = 1
DP = 8
# → Good for: <3B models with large batches
```

**Implementation:**
```python
device_mesh = init_device_mesh(
    "cuda",
    mesh_shape=(DP, TP),
    mesh_dim_names=("dp", "tp")
)
```

### Performance Targets

| Metric | Target | Validation |
|--------|--------|------------|
| AllReduce 1GB | 700-800 GB/s bus BW | `nccl_benchmark.py` |
| P2P bandwidth | 800-900 GB/s per pair | `bandwidth_benchmark_suite_8gpu.py` |
| Small message latency | <2 μs | `symmetric_memory_8gpu.py` |
| Scaling efficiency | 85-95% vs 1 GPU | `training_8xb200_pipeline.py` |
| End-to-end speedup | 7.2-7.6x | Training benchmark |

## GB200/GB300 Specific Features

### Grace-Blackwell Architecture

```
GB200/GB300 Superchip:
  CPU: Grace (72 ARM Neoverse V2 cores)
       ├─ 480GB-1TB LPDDR5X memory
       └─ 144 threads (SMT-2)
  
  GPU: 1-8x Blackwell B200
       └─ 180 GB HBM3e per GPU
  
  Interconnect: NVLink-C2C (Chip-to-Chip)
       ├─ 900 GB/s CPU↔GPU bandwidth
       ├─ Coherent memory access
       └─ Zero-copy CPU-GPU transfers
```

### Optimal Use Cases

1. **Hybrid CPU-GPU Training**
   ```python
   # Parameters on GPU (hot path)
   params = model.parameters()  # → GPU
   
   # Optimizer states on CPU (via NVLink-C2C)
   optimizer_states = allocate_tensor_with_numa_hint(
       shape=(param_count,),
       on_cpu_memory=True  # 900 GB/s access from GPU
   )
   ```
   **Benefit:** Save 2x parameter size in GPU memory

2. **Large KV Cache Inference**
   ```python
   # Model on GPU, KV cache on CPU
   kv_cache_cpu = torch.empty(
       (batch, num_layers, 2, num_heads, seq_len, head_dim),
       device='cpu',
       pin_memory=True
   )
   # Access at ~800 GB/s via NVLink-C2C
   ```
   **Benefit:** 480GB-1TB CPU memory for cache

3. **CPU Preprocessing Pipeline**
   ```
   CPU: Data loading → Tokenization → Batching
    ↓ (900 GB/s NVLink-C2C)
   GPU: Training → Gradient computation
    ↓ (async, overlapped)
   CPU: Optimizer updates (optional)
   ```
   **Benefit:** <5% CPU overhead vs PCIe

### Configuration

```python
from ch4.nccl_blackwell_config import configure_nccl_for_gb200_gb300

# Automatic Grace detection and optimization
configure_nccl_for_gb200_gb300(verbose=True)
```

## Troubleshooting

### Issue: Low NCCL Bandwidth (<500 GB/s)

**Check:**
1. NVLink topology: `nvidia-smi topo -m`
2. NCCL configuration: `echo $NCCL_*`
3. CPU frequency scaling: `cpupower frequency-info`

**Fix:**
```bash
# Re-configure NCCL
python -c "from ch4.nccl_blackwell_config import configure_nccl_for_8xB200; configure_nccl_for_8xB200(verbose=True)"

# Check topology
cd ../ch2
python gb200_topology_aware.py
```

### Issue: Out of Memory on 8 GPUs

**Solutions:**
1. **Increase TP, decrease DP:**
   ```python
   # Change from TP=2, DP=4 to TP=4, DP=2
   device_mesh = init_device_mesh("cuda", (2, 4), ("dp", "tp"))
   ```

2. **Use CPU offloading (GB200/GB300):**
   ```python
   # Offload optimizer states to CPU memory
   optimizer_states_cpu = True  # Save 2x param size
   ```

3. **Enable gradient checkpointing:**
   ```python
   model = torch.utils.checkpoint.checkpoint_sequential(
       model, segments=4
   )
   ```

### Issue: Slow Compilation (torch.compile)

**Expected behavior:** First run takes 5-10 minutes
**Workaround:** Use cached compilation

```bash
# Set cache directory
export TORCHINDUCTOR_CACHE_DIR=/tmp/torch_compile_cache

# First run: slow (compiles)
torchrun --nproc_per_node=8 training_8xb200_pipeline.py --compile

# Subsequent runs: fast (uses cache)
```

## Performance Profiling

### NCCL Profiling

```bash
# Enable NCCL debug output
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

# Run with profiling
torchrun --nproc_per_node=8 your_script.py 2>&1 | tee nccl_profile.log
```

### Nsight Systems

```bash
# Profile 8-GPU training
nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --numa-node-affinity=true \
    --output=8gpu_profile.nsys-rep \
    torchrun --nproc_per_node=8 training_8xb200_pipeline.py

# View in Nsight Systems UI
nsight-sys 8gpu_profile.nsys-rep
```

### Bandwidth Monitoring

```bash
# Continuous monitoring during training
watch -n 1 nvidia-smi dmon -s u

# Detailed P2P bandwidth matrix
torchrun --nproc_per_node=8 bandwidth_benchmark_suite_8gpu.py --full --output results.json
```

## Advanced Topics

### Custom Communication Patterns

See `nvshmem_8gpu_examples.cu` for:
- Ring exchange (7 steps for 8 GPUs)
- Butterfly/hypercube (3 steps for 8 GPUs)
- Custom reductions

### Multi-Node Training

For 2+ nodes with 8 GPUs each:

```bash
# Node 0 (master)
torchrun --nnodes=2 --nproc_per_node=8 \
         --node_rank=0 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_ADDR:29500 \
         multi_node_blackwell.py

# Node 1
torchrun --nnodes=2 --nproc_per_node=8 \
         --node_rank=1 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_ADDR:29500 \
         multi_node_blackwell.py
```

## References

- NCCL 2.28 Release Notes: https://docs.nvidia.com/deeplearning/nccl/
- NVSHMEM Documentation: https://docs.nvidia.com/hpc-sdk/nvshmem/
- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- Blackwell Architecture: https://www.nvidia.com/en-us/data-center/blackwell/

## Summary

### Key Takeaways

1. **8x B200 Configuration:**
   - 1184 SMs, 1.44 TB memory, 62.4 TB/s bandwidth
   - Target: 700-800 GB/s AllReduce, 85-95% scaling efficiency
   - Use NCCL 2.28 with NVLS for optimal performance

2. **GB200/GB300 Superchip:**
   - 900 GB/s CPU-GPU via NVLink-C2C
   - Unified memory for seamless CPU-GPU access
   - Ideal for hybrid workloads and large memory requirements

3. **Communication Choice:**
   - NCCL: Production training, standard collectives
   - NVSHMEM: Custom patterns, kernel-initiated
   - Symmetric Memory: PyTorch integration, portable

4. **Hybrid Parallelism:**
   - TP=2, DP=4: Balanced for most 7B-13B models
   - Adjust based on model size and memory constraints
   - Test with `training_8xb200_pipeline.py`

---

**Status:** All examples tested on 8x B200 hardware (October 2025)

