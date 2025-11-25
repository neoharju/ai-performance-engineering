# Ultimate MoE Inference Lab

**The mother of all labs** - a comprehensive end-to-end benchmark demonstrating **every optimization technique** from the AI Performance Engineering book, applied to real MoE model inference on NVIDIA Blackwell B200 GPUs.

## Quick Start

```bash
# Run the complete analysis pipeline (one command!)
python run_full_analysis.py

# Quick mode (faster, fewer iterations)
python run_full_analysis.py --quick

# With full profiling (nsys + ncu + HTA traces)
python run_full_analysis.py --profile all

# View the report
cat artifacts/ultimate_moe_inference/REPORT.md
```

---

## 🎯 Complete Technique Coverage

This lab demonstrates **ALL** optimization techniques from Chapters 1-20:

### Layer-by-Layer Breakdown

| Layer | Chapters | Speedup | Key Techniques |
|-------|----------|---------|----------------|
| **Baseline** | - | 1.0x | No optimizations (eager mode) |
| **Layer 1** | Ch1-6 | ~1.1x | NVTX, NUMA binding, TF32, cuDNN |
| **Layer 2** | Ch7-8 | ~1.3x | Memory coalescing, occupancy, ILP |
| **Layer 3** | Ch9-10 | ~2.0x | Tiling, double buffering, TMA, FlashAttention |
| **Layer 4** | Ch11-12 | ~2.5x | CUDA streams, CUDA graphs |
| **Layer 5** | Ch13-14 | ~3.5x | FP8, torch.compile, Triton |
| **Layer 6** | Ch15-20 | ~5.0x+ | MoE, PagedAttention, speculative decode |

---

## 📚 Detailed Technique Mapping

### Layer 1: Basics (Ch1-6)
| Chapter | Techniques | How Applied |
|---------|------------|-------------|
| Ch1 | GPU architecture intro | Understanding of SM, warps, memory hierarchy |
| Ch2 | Nsight profiling | NVTX markers for timeline visualization |
| Ch3 | Grace-Blackwell | NUMA binding, NVLink-C2C awareness |
| Ch4 | Thread hierarchy | Block/grid sizing via library defaults |
| Ch5 | Kernel design | Leveraged via FlashAttention, cuBLAS |
| Ch6 | Tensor Cores | **TF32 enable**, cuBLAS tensor core selection |

### Layer 2: Memory (Ch7-8)
| Chapter | Techniques | How Applied |
|---------|------------|-------------|
| Ch7 | **Coalesced access** | Contiguous tensor layout, 128B cache lines |
| Ch7 | **Vectorized loads** | float4/int4 via cuBLAS, 16B per thread |
| Ch7 | **L2 persistence** | Cache hints on Ampere+ |
| Ch7 | **Read-only cache** | `__ldg()`, `const __restrict__` in kernels |
| Ch7 | **Shared memory** | Data reuse in attention, GEMM tiles |
| Ch8 | **Occupancy tuning** | `__launch_bounds__`, cuDNN autotuning |
| Ch8 | **Register pressure** | Compiler hints, kernel selection |
| Ch8 | **ILP** | Loop unrolling, instruction scheduling |
| Ch8 | **Latency hiding** | Multiple independent ops in flight |

### Layer 3: Pipelining (Ch9-10)
| Chapter | Techniques | How Applied |
|---------|------------|-------------|
| Ch9 | **Tiling** | GEMM tiles, attention block processing |
| Ch9 | **Bank conflict avoidance** | Padding, swizzling in shared memory |
| Ch9 | **Warp shuffle** | `__shfl_sync()` for reductions |
| Ch10 | **Double buffering** | Overlap load/compute with 2 buffers |
| Ch10 | **CUDA Pipeline API** | `cuda::memcpy_async`, producer/consumer |
| Ch10 | **TMA** | Tensor Memory Accelerator (Blackwell) |
| Ch10 | **TMEM** | Tensor Memory for 2-CTA clusters |
| Ch10 | **Warp specialization** | Loader/compute/storer warp roles |
| Ch10 | **FlashAttention** | O(n) memory via tiled online softmax |
| Ch10 | **Persistent kernels** | Single kernel for full workload |

### Layer 4: Concurrency (Ch11-12)
| Chapter | Techniques | How Applied |
|---------|------------|-------------|
| Ch11 | **CUDA streams** | Prefill/decode overlap, multi-stream |
| Ch11 | **Stream events** | Synchronization between streams |
| Ch11 | **Async H2D/D2H** | Overlap data transfer with compute |
| Ch11 | **Cooperative Groups** | Thread block clusters (Blackwell) |
| Ch11 | **DSMEM** | Distributed shared memory across CTAs |
| Ch12 | **CUDA graphs** | Capture decode loop, eliminate launch overhead |
| Ch12 | **Graph bucketing** | Multiple graphs for variable seq lengths |
| Ch12 | **Stream-ordered alloc** | `cudaMallocAsync` in graphs |

### Layer 5: PyTorch (Ch13-14)
| Chapter | Techniques | How Applied |
|---------|------------|-------------|
| Ch13 | **FP8 (E4M3/E5M2)** | Transformer Engine, dynamic scaling |
| Ch13 | **DelayedScaling** | Amax history, hysteresis |
| Ch13 | **FP8 KV cache** | 50% memory reduction |
| Ch13 | **MXFP4/NVFP4** | 4-bit inference (Blackwell) |
| Ch14 | **torch.compile** | TorchInductor, kernel fusion |
| Ch14 | **max-autotune** | Exhaustive kernel search |
| Ch14 | **Triton kernels** | Custom fused operations |
| Ch14 | **Triton autotuning** | Block size, num_warps, num_stages |

### Layer 6: Advanced Inference (Ch15-20)
| Chapter | Techniques | How Applied |
|---------|------------|-------------|
| Ch15 | **MoE routing** | Expert parallelism, load balancing |
| Ch15 | **Expert overlap** | Stream-parallel expert execution |
| Ch16 | **PagedAttention** | Block-based KV cache management |
| Ch16 | **Prefix caching** | Reuse KV for shared prompts |
| Ch17 | **Continuous batching** | Dynamic request scheduling |
| Ch17 | **Chunked prefill** | Long sequence without OOM |
| Ch18 | **Speculative decode** | Draft model parallel verification |
| Ch18 | **FlashMLA** | Multi-head Latent Attention |
| Ch18 | **GQA/MQA** | Grouped/Multi-query attention |
| Ch19 | **Dynamic precision** | FP8→FP4 based on confidence |
| Ch19 | **NVFP4 training** | 4-bit with block scaling |
| Ch20 | **AI kernel gen** | LLM-assisted optimization |

---

## 🔬 Blackwell-Specific Features (SM 10.0+)

| Feature | Description | Where Used |
|---------|-------------|------------|
| **TMA** | Tensor Memory Accelerator - hardware async copy | Layer 3 (pipelining) |
| **TMEM** | Tensor Memory for cluster-level shared data | Layer 3 (2-CTA clusters) |
| **2-CTA Clusters** | Thread block clusters share DSMEM | Layer 4 (cooperative groups) |
| **DSMEM** | Distributed Shared Memory across CTAs | Layer 4 (attention tiles) |
| **FP4 Tensor Cores** | 4-bit matrix multiply | Layer 5 (NVFP4) |
| **HBM3e** | 8 TB/s bandwidth | All layers benefit |
| **NVLink-C2C** | CPU-GPU coherent memory | Layer 1 (Grace-Blackwell) |

---

## 📊 Models

| Model | Parameters | Active | Use Case |
|-------|------------|--------|----------|
| [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | 21B | 3.6B | Single GPU |
| [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | 117B | 5.1B | Multi-GPU |

---

## 🚀 Running with `benchmark_cli` Harness

### Basic Benchmark Run

```bash
# Run all benchmarks in this lab
python tools/cli/benchmark_cli.py run --targets labs/ultimate_moe_inference

# Run specific optimization layer
python tools/cli/benchmark_cli.py run --targets labs/ultimate_moe_inference:layer2_memory
python tools/cli/benchmark_cli.py run --targets labs/ultimate_moe_inference:layer3_pipelining

# List available targets
python tools/cli/benchmark_cli.py list-targets --chapter labs/ultimate_moe_inference
```

### With Profiling

```bash
# Minimal profiling (nsys basic)
python tools/cli/benchmark_cli.py run --targets labs/ultimate_moe_inference --profile minimal

# Deep dive profiling (nsys + ncu + roofline + differential analysis)
python tools/cli/benchmark_cli.py run --targets labs/ultimate_moe_inference --profile deep_dive

# Roofline-focused analysis
python tools/cli/benchmark_cli.py run --targets labs/ultimate_moe_inference --profile roofline
```

### With LLM Analysis

```bash
# Get AI-powered optimization suggestions
export ANTHROPIC_API_KEY=your_key
python tools/cli/benchmark_cli.py run --targets labs/ultimate_moe_inference --profile deep_dive --llm-analysis
```

---

## 🔍 Profiling & Analysis

### Nsight Systems (Timeline)

Visualize kernel execution, memory transfers, and NVTX regions:

```bash
# Generate trace
nsys profile -o ultimate_trace python optimized_ultimate_inference.py

# View in GUI
nsys-ui ultimate_trace.nsys-rep
```

**What to look for:**
- NVTX regions show layer boundaries
- Kernel launch gaps (→ use CUDA graphs)
- Memory copy overlap with compute (→ double buffering)

### Nsight Compute (Kernel Analysis)

Detailed kernel metrics:

```bash
# Generate report
ncu --set full -o ultimate_ncu python optimized_ultimate_inference.py

# View in GUI
ncu-ui ultimate_ncu.ncu-rep
```

**Key metrics:**
- **Occupancy**: Target >50% for memory-bound, >70% for compute-bound
- **Memory throughput**: % of peak HBM bandwidth
- **Tensor Core utilization**: % of peak TFLOPS
- **L2 hit rate**: Higher = better cache reuse

### Roofline Analysis

Determine if memory or compute bound:

```bash
python scaling_study/roofline_analyzer.py --benchmark optimized_ultimate_inference
```

**Interpretation:**
- Below ridge point (AI < 500 FLOP/Byte) → **Memory bound** → Focus on Layer 2-3
- Above ridge point → **Compute bound** → Focus on Layer 5-6

### PyTorch Profiler + HTA

```bash
# Run with HTA traces
python run_full_analysis.py --profile hta

# Analyze with HTA
pip install HolisticTraceAnalysis
python -c "
from hta.trace_analysis import TraceAnalysis
analyzer = TraceAnalysis(trace_dir='artifacts/ultimate_moe_inference/profiling/')
print(analyzer.get_gpu_kernel_breakdown())
print(analyzer.get_temporal_breakdown())
"
```

---

## 📁 Directory Structure

```
ultimate_moe_inference/
├── run_full_analysis.py              # One-command full analysis
├── baseline_ultimate_inference.py    # Baseline (no optimizations)
├── optimized_layer1_basics.py        # Ch1-6: TF32, NUMA, cuDNN
├── optimized_layer2_memory.py        # Ch7-8: Coalescing, occupancy, ILP
├── optimized_layer3_pipelining.py    # Ch9-10: Tiling, TMA, FlashAttention
├── optimized_layer4_graphs.py        # Ch11-12: Streams, CUDA graphs
├── optimized_layer5_compile.py       # Ch13-14: FP8, torch.compile
├── optimized_ultimate_inference.py   # ALL techniques
├── expectations.json                 # Performance baselines
│
├── optimization_layers/              # Reusable optimization code
│   ├── layer_01_basics.py            # TF32, NUMA, NVTX
│   ├── layer_02_memory.py            # Coalescing, occupancy, ILP
│   ├── layer_03_pipelining.py        # Double buffering, TMA, FlashAttention
│   ├── layer_04_concurrency.py       # Streams, CUDA graphs, clusters
│   ├── layer_05_pytorch.py           # FP8, torch.compile, Triton
│   └── layer_06_advanced.py          # MoE, PagedAttention, speculative
│
├── components/                       # Core inference components
│   ├── monitoring.py                 # TTFT/TPOT/TPS metrics
│   ├── torch_profiler.py             # HTA integration
│   ├── kv_cache_manager.py           # PagedAttention
│   ├── speculative_decoder.py        # Draft model speculation
│   ├── ring_attention.py             # Distributed attention
│   ├── flash_mla.py                  # Multi-head Latent Attention
│   └── gqa_optimizer.py              # GQA/MQA optimization
│
├── scaling_study/                    # Analysis tools
│   ├── roofline_analyzer.py          # Memory vs compute bound
│   ├── run_scaling_study.py          # 1→8 GPU scaling
│   ├── compare_optimizations.py      # Layer comparison
│   └── precision_sweep.py            # MXFP4 vs FP8 vs BF16
│
├── vllm_variant/                     # vLLM comparison
└── tests/                            # Unit tests
```

---

## 📈 Expected Results (B200)

| Configuration | Baseline (tok/s) | Optimized (tok/s) | Speedup |
|---------------|------------------|-------------------|---------|
| 1x B200 (gpt-oss-20b) | ~500 | ~2000+ | **4x+** |
| 8x B200 (gpt-oss-120b) | ~200 | ~1500+ | **7x+** |

### Speedup Breakdown by Layer

| Optimization | Technique | Typical Speedup |
|--------------|-----------|-----------------|
| TF32 + cuDNN | Layer 1 | 1.05-1.1x |
| Memory coalescing | Layer 2 | 1.1-1.2x |
| FlashAttention | Layer 3 | 1.5-2.0x |
| CUDA graphs | Layer 4 | 1.2-1.5x |
| torch.compile | Layer 5 | 1.2-1.4x |
| Speculative decode | Layer 6 | 1.5-2.0x |
| FP8 KV cache | Layer 5-6 | 1.1-1.2x |
| **Cumulative** | All | **5-7x** |

---

## 🧪 Scaling Studies

### GPU Scaling (1 → 8 GPUs)

```bash
python scaling_study/run_scaling_study.py --model 120b
```

### Batch Size Sweep

```bash
python scaling_study/batch_size_sweep.py --batch-sizes 1 2 4 8 16 32
```

### Precision Comparison

```bash
python scaling_study/precision_sweep.py
```

---

## 📦 Requirements

```bash
# Core
pip install torch>=2.4.0 transformers>=4.45.0 accelerate>=0.34.0

# Optimizations
pip install transformer-engine>=1.8 triton>=3.0.0

# Profiling
pip install nvidia-ml-py pynvml

# Analysis
pip install matplotlib seaborn pandas HolisticTraceAnalysis

# Optional: vLLM comparison
pip install vllm>=0.6.0
```

---

## 📖 Further Reading

For deep understanding of each technique, refer to the book chapters:

| Chapters | Topics |
|----------|--------|
| Ch1-6 | GPU Architecture, Profiling, Grace-Blackwell |
| Ch7-8 | Memory Hierarchy, Occupancy, ILP |
| Ch9-10 | Tiling, Pipelining, TMA, FlashAttention |
| Ch11-12 | Streams, CUDA Graphs, Cooperative Groups |
| Ch13-14 | FP8, torch.compile, Triton |
| Ch15-20 | MoE, PagedAttention, Speculative Decoding |

---

## 🤝 Contributing

1. Run full analysis: `python run_full_analysis.py`
2. Make optimization changes
3. Re-run and compare: `python run_full_analysis.py`
4. Document improvement in PR

See `TODO_EXTENSIONS.md` for future features.

---

*This lab is the comprehensive demonstration of AI Performance Engineering techniques.*
