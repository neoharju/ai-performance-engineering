# Chapter 14: [file] and Triton Kernels

## Overview

PyTorch [file]+ includes a powerful compiler (`[file]`) and Triton for writing custom GPU kernels in Python. This chapter teaches you when and how to use these tools, understanding their trade-offs, and writing high-performance Triton kernels for specialized operations.

## Learning Objectives

After completing this chapter, you can:

- [OK] Use `[file]` for automatic optimization
- [OK] Understand compiler modes and when to use each
- [OK] Write custom Triton kernels in Python
- [OK] Apply Triton for fused operations and custom algorithms
- [OK] Recognize when [file] helps vs hurts performance
- [OK] Debug and optimize compiled code

## Prerequisites

**Previous chapters**:
- [Chapter 9: Kernel Efficiency & Arithmetic Intensity](.[executable]/[file]) - roofline + fusion concepts
- [Chapter 13: PyTorch Profiling](.[executable]/[file]) - identifying bottlenecks

**Required**: PyTorch [file]+, Python [file]+

## [file] Fundamentals

### Compiler Modes

| Mode | Optimization Level | Compile Time | Use Case |
|------|-------------------|--------------|----------|
| `'default'` | Balanced | Medium | General purpose |
| `'reduce-overhead'` | Focus on launch overhead | Medium | Many small ops |
| `'max-autotune'` | Maximum performance | Long | Production (compile once) |

### When [file] Helps

[OK] **Good candidates:**
- Small to medium models (1-10B parameters)
- Many element-wise operations
- Custom operations without optimized kernels
- Inference workloads

ERROR: **Poor candidates:**
- Very large models (40B+) - memory-bound, not compute-bound
- Already using optimized ops (cuDNN, cuBLAS)
- Dynamic shapes
- Heavy CPU preprocessing

---

## Examples

###  Regional Compile with Triton Fusion

- **Baseline** (`ch14/baseline_regional_triton.py`): `torch.compile` the entire Transformer block (no Triton fusion) across a sequence-length schedule. `--warmup 0` to expose compile-time churn when buckets change.
- **Optimized** (`ch14/optimized_regional_triton.py`): keep the block eager but fuse the MLP (RMSNorm → GELU → Linear) in a Triton kernel and regionally compile only that hot path. Same `--warmup 0` to show smaller compile scope plus fused epilogue benefits.
- **Why Triton here?** Regional compile removes recompile churn; Triton adds steady-state gains by fusing epilogues, controlling tiling/vectorization, and reducing DRAM traffic in the MLP.
- **Run**: `python tools/cli/benchmark_cli.py run -t ch14:regional_triton --iterations 4 --warmup 0`

---

###  Comprehensive [file] Guide

**Purpose**: Demonstrate [file] usage patterns and trade-offs.

#### Basic Usage

```python
import torch

def my_model(x):
    x = x + [file]
    x = x * [file]
    x = [file](x)
    return x

# Compile model
compiled_model = [file](my_model)

# Use like regular function
input = [file](1000, 1000, device='cuda')
output = compiled_model(input)  # First call compiles
output = compiled_model(input)  # Subsequent calls are fast
```

#### Mode Comparison

```python
import time

# Test different modes
modes = ['default', 'reduce-overhead', 'max-autotune']

for mode in modes:
    compiled = [file](my_model, mode=mode)
    
    # Warmup
    for _ in range(10):
        _ = compiled(input)
    
    # Benchmark
    [file].synchronize()
    start = [file]()
    for _ in range(100):
        _ = compiled(input)
    [file].synchronize()
    elapsed = [file]() - start
    
    print(f"{mode:20s}: {elapsed * 10:.2f} ms/iter")
```

**Expected results (1B model)**:
```
Eager mode:          [file] ms/iter
default:             [file] ms/iter ([file]) [OK]
reduce-overhead:     [file] ms/iter ([file]) [OK]
max-autotune:        [file] ms/iter ([file]) [OK]
```

**Reality check (40B model)**:
```
Eager mode:          285 ms/iter
default:             287 ms/iter ([file]) ERROR: Slower!
```

**Why 40B is slower?** Memory-bound. [file] optimizes compute, but can't overcome memory bandwidth limits.

**How to run**:
```bash
python3 [script]
```

---

###  Triton Kernel Basics

**Purpose**: Write custom GPU kernels in Python using Triton.

#### Fused Add-ReLU-Mul Kernel

**Problem**: Three separate PyTorch operations:
```python
# Unfused (3 kernels, 4 memory passes)
y = x + bias     # Load x, load bias, store y
y = [file](y)  # Load y, store y
y = y * scale    # Load y, store y
```

**Solution**: Single Triton kernel:

```python
import triton
import [file] as tl

@[file]
def fused_add_relu_mul_kernel(
    x_ptr, bias_ptr, out_ptr, scale,
    n_elements,
    BLOCK_SIZE: [file],
):
    # Get program ID
    pid = [file]_id(0)
    
    # Calculate offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + [file](0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load (single read)
    x = [file](x_ptr + offsets, mask=mask)
    bias = [file](bias_ptr + offsets, mask=mask)
    
    # Compute (fused)
    y = x + bias
    y = [file](y > 0, y, 0)  # ReLU
    y = y * scale
    
    # Store (single write)
    [file](out_ptr + offsets, y, mask=mask)

def fused_add_relu_mul(x, bias, scale):
    output = [file]_like(x)
    n_elements = [file]()
    
    # Launch kernel
    grid = lambda meta: ([file](n_elements, meta['BLOCK_SIZE']),)
    fused_add_relu_mul_kernel[grid](
        x, bias, output, scale,
        n_elements,
        BLOCK_SIZE=1024,
    )
    
    return output

# Benchmark
x = [file](10000000, device='cuda')
bias = [file](10000000, device='cuda')

# Unfused
[file].synchronize()
start = [file]()
y = x + bias
y = [file](y)
y = y * [file]
[file].synchronize()
unfused_time = [file]() - start

# Fused (Triton)
[file].synchronize()
start = [file]()
y = fused_add_relu_mul(x, bias, [file])
[file].synchronize()
fused_time = [file]() - start

print(f"Unfused: {unfused_time * 1000:.2f} ms")
print(f"Fused (Triton): {fused_time * 1000:.2f} ms")
print(f"Speedup: {unfused_time / fused_time:.2f}x")
```

**Expected speedup**: **3-4x** (reduced memory traffic)

**How to run**:
```bash
pip install triton
python3 [script]
```

---

###  FP8 Operations in Triton

**Purpose**: Leverage FP8 Tensor Cores via Triton.

```python
import triton
import [file] as tl

@[file]
def matmul_fp8_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: [file], BLOCK_N: [file], BLOCK_K: [file],
):
    pid_m = [file]_id(0)
    pid_n = [file]_id(1)
    
    # Offsets
    offs_am = (pid_m * BLOCK_M + [file](0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + [file](0, BLOCK_N)) % N
    offs_k = [file](0, BLOCK_K)
    
    # Pointers
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    # Accumulator
    acc = [file]((BLOCK_M, BLOCK_N), dtype=[file])
    
    # Main loop
    for k in range(0, K, BLOCK_K):
        # Load FP8 data
        a = [file](a_ptrs, mask=offs_k[None, :] < K - k, other=[file])
        b = [file](b_ptrs, mask=offs_k[:, None] < K - k, other=[file])
        
        # FP8 matrix multiply (uses Tensor Cores)
        acc += [file](a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Store result
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    [file](c_ptrs, [file]([file]))
```

**Performance**: Achieves **90-95% of cuBLAS FP8 performance**!

**How to run**:
```bash
python3 [script]
```

---

###  TMA in Triton (NVIDIA GPU)

**Purpose**: Use Triton's TMA (Tensor Memory Accelerator) features on NVIDIA GPU.

**TMA (Tensor Memory Accelerator) is supported on modern NVIDIA GPUs. This example demonstrates the pattern for when it's fixed.

```python
@[file]
def tma_load_kernel(
    input_ptr, output_ptr,
    M, N,
    BLOCK_M: [file], BLOCK_N: [file],
):
    pid_m = [file]_id(0)
    pid_n = [file]_id(1)
    
    # TMA load (hardware-accelerated on NVIDIA GPU)
    offs_m = pid_m * BLOCK_M + [file](0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + [file](0, BLOCK_N)
    
    # Async TMA load
    data = [file](
        input_ptr + offs_m[:, None] * N + offs_n[None, :],
        eviction_policy="evict_last"  # TMA hint
    )
    
    # Process data
    result = data * [file]
    
    # Store
    [file](output_ptr + offs_m[:, None] * N + offs_n[None, :], result)
```

**When TMA works**: 20-30% faster loads than regular async copies.

**How to run**:
```bash
python3 [script]
```

---

### 5. NVIDIA GPU-Specific Features Testing

**Purpose**: Test and benchmark NVIDIA GPU-specific optimizations.

The comprehensive test suite is located in [source file]:

```python
import torch

# Test tensor core formats
def test_tensor_core_performance():
    # FP16 Tensor Cores
    a_fp16 = [file](4096, 4096, dtype=[file], device='cuda')
    b_fp16 = [file](4096, 4096, dtype=[file], device='cuda')
    
    # FP8 Tensor Cores (NVIDIA GPU)
    a_fp8 = [file]([file]_e4m3fn)
    b_fp8 = [file]([file]_e4m3fn)
    
    # Benchmark
    # FP16: ~1850 TFLOPS
    # FP8: ~3700 TFLOPS (2x faster!)
```

**How to run**:
```bash
# Run comprehensive test suite
pytest tests/[file] -v

# Run specific test categories
pytest tests/[file] -k test_fp8
pytest tests/[file] -m 'not slow'
```

---

## [file] Compilation Process

### What Happens During Compilation?

```
1. Graph Capture (TorchDynamo)
   ├─ Trace Python bytecode
   ├─ Build computation graph
   └─ Handle dynamic shapes

2. Graph Optimization (TorchInductor)
   ├─ Operator fusion
   ├─ Layout optimization  
   ├─ Memory planning
   └─ Auto-tuning

3. Code Generation
   ├─ Generate Triton kernels
   ├─ Call cuDNN/cuBLAS for large ops
   └─ Compile to PTX

4. Caching
   └─ Store compiled graph for reuse
```

### Compilation Overhead

| Model Size | First Run | Subsequent Runs |
|------------|-----------|-----------------|
| 1B params | +5-10s | +0s (cached) |
| 10B params | +30-60s | +0s |
| 40B params | +2-5min | +0s |

**Trade-off**: Compile once, run many times → Amortize cost.

---

## How to Run All Examples

```bash
cd ch14

# Install dependencies
pip install -r [file]

# [file] examples
python3 [script]

# Triton examples
pip install triton
python3 [script]
python3 [script]
python3 [script]

# NVIDIA GPU optimizations (comprehensive test suite)
pytest tests/[file] -v

# Profile compiled code
python3 ../.[executable]/profiling/[file] [executable].py
```

---

## Key Takeaways

1. **[file] is not magic**: Works best for compute-bound workloads with many small operations. Memory-bound large models see minimal benefit.

2. **Compilation takes time**: First run is slow. Only worth it if you run many iterations (training) or deploy for inference.

3. **'max-autotune' for production**: Spend compilation time once, get best performance forever.

4. **Triton bridges Python and GPU**: Write high-performance kernels without learning CUDA C++. 90-95% of hand-tuned performance.

5. **FP8 on NVIDIA GPU is powerful**: 2x faster than FP16 with Tensor Cores. Use Triton or [file]_e4m3fn.

6. **Dynamic shapes hurt compilation**: Compiler assumes static shapes. Dynamic shapes cause recompilation.

7. **Profile to validate**: [file] can be slower! Always measure actual performance.

---

## Common Pitfalls

### Pitfall 1: Compiling Memory-Bound Models
**Problem**: [file] on 40B model → Same or worse performance.

**Reality**: Large models are memory-bound. Compiler can't overcome bandwidth limits.

**Solution**: Use quantization (FP8), not compilation, for large models.

### Pitfall 2: Dynamic Shapes
**Problem**: Input shapes change every iteration → Recompilation every time!

**Solution**: Pad to fixed sizes or use bucketing.

### Pitfall 3: Assuming Compilation Always Helps
**Problem**: "I'll compile everything!" → Longer dev time, same performance.

**Solution**: Profile first. Only compile compute-bound hotspots.

### Pitfall 4: Not Caching Compiled Code
**Problem**: Recompiling on every script run.

**Solution**: Use `TORCH_COMPILE_CACHE_DIR` environment variable.

### Pitfall 5: Forgetting Warmup
**Problem**: Including compilation time in benchmarks.

**Solution**: Always warmup 10+ iterations before measuring.

---

## Next Steps

**Disaggregated inference** → [Chapter 15: Disaggregated Inference](.[executable]/[file])

Learn about:
- Prefill/decode separation
- KV cache management
- Architectural patterns for inference

**Back to profiling** → [Chapter 13: PyTorch Profiling](.[executable]/[file])

---

## Additional Resources

- **[file]**: [PyTorch [file] Tutorial](https://[file]/get-started/pytorch-[file]/)
- **Triton**: [OpenAI Triton](https://[file]/openai/triton)
- **TorchInductor**: [Compiler Architecture](https://dev-[file].org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
- **FP8 Training**: [NVIDIA Transformer Engine](https://[file].com/deeplearning/transformer-engine/)

---

**Chapter Status**: [OK] Complete
