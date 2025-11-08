# AI Systems Performance Engineering: Code

Production playbook for standing up, validating, and tuning PyTorch LLM workloads on 8x NVIDIA B200 systems.

---

## Overview
**Target hardware:** 

- NVIDIA Blackwell (B200/B300, sm100)
- Grace Blackwell (GB200/GB300, sm103)
- DGX Spark (GB10, sm121)

**Reference stack:** CUDA 13+, PyTorch 2.9+, Triton 3.5+, and Python 3.10+

The repository packages everything needed to:
- Provision a reproducible software stack (`setup.sh`) for new lab machines.
- Exercise and benchmark the platform end-to-end before deploying workloads.

## Summary of Concepts and Examples
- **Total Chapters**: 20 (ch1–ch20)
- **Total Examples**: 264 baseline/optimized pairs

### Concepts by Chapter

| Chapter | Concepts Covered | # Examples |
|---------|------------------|------------|
| **ch1** | Performance basics, profiling, memory management, CUDA Graphs, batched operations, roofline model, arithmetic intensity optimization | 21 |
| **ch2** | GPU hardware architecture, NVLink, CPU-GPU coherency, memory hierarchy, hardware introspection | 6 |
| **ch3** | System tuning, NUMA binding, CPU governor, THP, PCIe settings, IRQ affinity, Docker/K8s configuration, topology discovery | 11 |
| **ch4** | Multi-GPU training, NCCL collectives, tensor/pipeline parallelism, NVSHMEM, symmetric memory, distributed training, bandwidth testing | 7 |
| **ch5** | Storage & IO optimization, GPUDirect Storage (GDS), DataLoader optimization, prefetching, caching, cuFile integration | 7 |
| **ch6** | CUDA basics, thread hierarchy, kernel launches, grid/block dimensions, occupancy, unified memory, 2D/3D indexing | 14 |
| **ch7** | Memory access patterns, coalescing, vectorization, shared memory, bank conflicts, tiling, memory bandwidth optimization | 20 |
| **ch8** | Occupancy tuning, instruction-level parallelism (ILP), loop unrolling, register pressure, warp divergence, resource limits | 13 |
| **ch9** | Kernel efficiency, arithmetic intensity, roofline model, micro-tiling, kernel fusion, CUTLASS, register blocking | 17 |
| **ch10** | Tensor Cores (tcgen05), TMA, async pipelines, double buffering, warp specialization, thread block clusters, persistent kernels | 17 |
| **ch11** | CUDA streams, concurrency, stream-ordered allocators, multi-stream pipelines, Hyper-Q, priority streams, overlap optimization | 11 |
| **ch12** | CUDA Graphs, graph capture/replay, conditional graphs, dynamic parallelism, device-side launches, load balancing, graph instantiation | 14 |
| **ch13** | PyTorch profiling, memory analysis, compiled autograd, FSDP, DataLoader optimization, quantization, mixed precision, custom allocators | 23 |
| **ch14** | torch.compile, Triton kernels, compiler modes, FP8 operations, TMA in Triton, custom kernel development | 5 |
| **ch15** | Disaggregated inference, prefill/decode separation, KV cache management, continuous batching, PagedAttention, prefix caching | 10 |
| **ch16** | Inference optimization, production serving (vLLM), FP8 quantization, speculative decoding, MoE benchmarking, request scheduling, monitoring | 8 |
| **ch17** | Dynamic routing, early exit, adaptive inference, complexity-based routing, roofline analysis, latency vs accuracy trade-offs | 13 |
| **ch18** | Advanced attention (FlashAttention, FlexAttention, MLA, PagedAttention), sliding window attention, custom attention patterns | 18 |
| **ch19** | FP4/FP6/FP8 quantization, low-precision training, dynamic precision switching, Transformer Engine, per-token precision, validation | 15 |
| **ch20** | End-to-end optimization, case studies, debugging strategies, production deployment, comprehensive workflows, multiple technique combinations | 14 |

### Major Concept Categories

### 1. Performance Fundamentals (ch1–ch3): 38 examples
- Profiling and measurement
- Hardware architecture understanding
- System-level tuning

### 2. Memory Optimization (ch7, ch19): 35 examples
- Memory access patterns and coalescing
- Quantization and precision reduction
- Memory bandwidth optimization

### 3. Parallelism & Distribution (ch4, ch11, ch13, ch15–ch17): 52 examples
- Multi-GPU training and communication
- CUDA streams and concurrency
- Data/pipeline/tensor parallelism
- Inference optimization and serving

### 4. Kernel Optimization (ch6–ch10, ch12): 81 examples
- CUDA programming fundamentals
- Memory patterns and bandwidth
- Occupancy and ILP
- Tensor cores and advanced CUDA features
- CUDA Graphs and dynamic parallelism

### 5. PyTorch & Compiler (ch13–ch14): 28 examples
- PyTorch profiling and optimization
- Compiled autograd and FSDP
- torch.compile and Triton kernels

### 6. Attention Mechanisms (ch18): 18 examples
- FlashAttention and FlexAttention
- MLA (Multi-head Latent Attention)
- PagedAttention and custom patterns

### 7. Advanced Techniques (ch19–ch20): 29 examples
- Low-precision training (FP4/FP6/FP8)
- End-to-end optimization workflows
- Production deployment patterns

---

## Quick Start

### Repository Structure

The repository provides a comprehensive production playbook for:
- Standing up PyTorch LLM workloads on 8x NVIDIA B200 systems
- Validating and tuning performance end-to-end
- Benchmarking the platform before deploying workloads
- Learning performance engineering through hands-on examples

Each chapter includes:
- Detailed README with learning objectives
- Baseline/optimized example pairs
- Performance analysis and expected results
- Common pitfalls and solutions
- Integration with unified benchmarking framework

### Prerequisites
- Root access to the host (the setup script installs NVIDIA driver 580+, CUDA 13.0, and dependencies)
- Python 3.10+ on the path (the setup script installs required packages in-place)
- Network access to fetch Python wheels and Nsight tooling

### Setup
1. Clone and enter the repository:
   ```bash
   git clone <repo-url> && cd ai-performance-engineering/code
   ```
2. Run the automated bootstrap:
   ```bash
   sudo ./setup.sh
   ```
3. If the script upgrades the driver, reboot and rerun `sudo ./setup.sh` to finish verification.


## Verification & Testing

### Quick Verification
Run the quick smoke tests after installation:
1. Confirm the hardware and driver:
   ```bash
   nvidia-smi
   ```
   Expect at least 1 Blackwell GPU and driver 580+.
2. Verify benchmarks can load (syntax + import check):
   ```bash
   python3 tools/verification/verify_all_benchmarks.py
   ```

### Running All Benchmarks

**Option 1: Typer-based CLI (`tools/cli/benchmark_cli.py`) - Recommended:**

```bash
# Full suite with structured logging + artifacts (run all chapters)
python tools/cli/benchmark_cli.py
# Or explicitly:
python tools/cli/benchmark_cli.py all

# Target one chapter with custom outputs
python tools/cli/benchmark_cli.py ch12 --artifacts-dir ./artifacts
# Or collect profiling traces explicitly (opt-in)
python tools/cli/benchmark_cli.py 12 --artifacts-dir ./artifacts --profile

# Slow systems: stretch every per-benchmark timeout by 2x
python tools/cli/benchmark_cli.py --timeout-multiplier 2.0

# Reproducible runs: set all seeds to 42 and enable deterministic algorithms
python tools/cli/benchmark_cli.py --reproducible

# Cold start: additional GPU cleanup (gc.collect()) for cold start measurements, including after failures (CUDA state is always reset between benchmarks by default)
python tools/cli/benchmark_cli.py --cold-start

# Combine flags: reproducible cold-start run with extended timeouts
python tools/cli/benchmark_cli.py --reproducible --cold-start --timeout-multiplier 2.0
```

Profiling (Nsight Systems/Compute + PyTorch profiler) is now disabled by default to keep throughput runs fast and avoid long GPU stalls. Pass `--profile` (or `--profile/--no-profile` in the Typer CLI) whenever you explicitly want trace captures.

### Skipping Profiling For A Run
Profiling stays off unless you explicitly enable it, but you can also force-disable it when you want to be sure no traces are captured:
- **CLI flag**: `python tools/cli/benchmark_cli.py ch12 --no-profile` (profiling disabled by default)

The CLI flag ensures nsys/ncu/PyTorch profiling is skipped for the entire invocation.

**What you get:**
- Discover every `baseline_*.py` / `optimized_*.py` pair.
- Run via `BenchmarkHarness`, automatically resetting CUDA state between benchmarks to prevent cascading failures.
- Measure baseline vs optimized performance and compute speedups.
- Detect/skip hardware or software limitations with clear notifications.
- Emit suite summaries:
  - `benchmark_test_results.json` (machine-readable)
  - `benchmark_test_results.md` (human-readable)
- Automatic artifact management: logs, manifests, and reports are stashed under `./artifacts/<run_id>` (override with `--artifacts-dir`).
- Reproducibility knobs baked in: every benchmark invocation is captured through `benchmark_with_manifest`, and the `--profile/--no-profile` toggles map directly to harness config.
- Additional controls:
  - `--timeout-multiplier`: Scale all benchmark timeouts (useful for slower systems)
  - `--suite-timeout`: Overall timeout for entire suite (default: 4 hours)
  - `--reproducible`: Set all seeds to 42 and enable deterministic algorithms
  - `--cold-start`: Additional GPU cleanup (includes gc.collect()) for cold start measurements, applied in all execution paths including after failures (CUDA state is always reset between benchmarks by default to prevent cascading failures)
  - `--log-level`: Control logging verbosity (DEBUG, INFO, WARNING, ERROR)

Exit codes remain `0` for success, `1` when any benchmark fails—ideal for CI/CD flows.

### Benchmark Harness Documentation

For detailed information about the benchmark harness architecture, features, and capabilities, see the [Benchmark Harness Guide](docs/benchmark_harness_guide.md).

### Running Integration Tests

To run the integration test suite that verifies end-to-end benchmark workflows:

```bash
pytest tests/integration
# or
python3 -m pytest tests/integration
```

The integration tests verify:
- Benchmark discovery across chapters
- Full benchmark execution pipeline
- Profiling workflows (nsys/ncu)
- Metrics collection and reporting
- Baseline vs optimized comparisons

### Multi-Metric Comparison

The benchmark comparison system supports comparing all collected metrics (not just timing) between baseline and optimized implementations.

**Basic Usage:**
```python
from common.python.benchmark_comparison import compare_and_display_all_metrics

# Compare all metrics and display results
comprehensive = compare_and_display_all_metrics(
    baseline_result=baseline_result,
    optimized_result=optimized_result,
    name="My Benchmark",
    format_style="both"  # Shows both table and summary
)
```

**Key Parameters:**
- `include_raw_metrics` (default: `False`): If `True`, includes raw profiler metrics (NCU/NSYS/Torch raw counters). Raw metrics are excluded by default to avoid noise from misclassified counters.
- `chapter` (default: `None`): Chapter identifier (e.g., `"ch7"`) to enable chapter-specific metrics from `performance_targets.py`. Chapter configs merge with base config, with chapter entries overriding base for same metric paths.

**Example with Raw Metrics:**
```python
# Enable raw profiler metrics
comprehensive = compare_and_display_all_metrics(
    baseline_result=baseline_result,
    optimized_result=optimized_result,
    name="My Benchmark",
    include_raw_metrics=True  # Include raw profiler counters
)
```

**Example with Chapter-Specific Metrics:**
```python
# Use chapter-specific metrics (e.g., ch7 for memory access patterns)
comprehensive = compare_and_display_all_metrics(
    baseline_result=baseline_result,
    optimized_result=optimized_result,
    name="My Benchmark",
    chapter="ch7"  # Loads metrics from performance_targets.py
)
```

**Percentage Calculation:**
The system uses delta-based percentage calculations (not ratio-based) for accurate reporting:
- For lower-is-better metrics: `improvement_pct = (baseline - optimized) / baseline * 100`
- For higher-is-better metrics: `improvement_pct = (optimized - baseline) / baseline * 100`
- Zero/None baselines are handled gracefully (percentage = None, no crashes)

**Metric Categories:**
- **Timing Metrics**: mean_ms, median_ms, p99_ms, etc.
- **Memory Metrics**: peak_mb, allocated_mb, reserved_mb
- **Profiler Metrics**: NCU (kernel_time_ms, sm_throughput_pct, etc.), NSYS, PyTorch profiler metrics

See `common/python/example_multi_metric_comparison.py` for more examples.

### Peak Performance Validation
During `setup.sh`, the system automatically runs `benchmark_peak.py` to capture actual peak hardware performance metrics:
- HBM memory bandwidth
- FP4 compute TFLOPS (if available)
- FP6 compute TFLOPS (if available)
- FP8 compute TFLOPS (if available)
- FP16 compute TFLOPS
- BF16 compute TFLOPS (if available)
- L2 cache bandwidth
- Shared memory (L1-equivalent) characteristics
- GPU hardware information (SMs, cache sizes, registers, etc.)
- NVLink bandwidth (if multi-GPU available)
- torch.compile speedup

These measured values are saved to `benchmark_peak_results_*.json` and used as dynamic performance targets instead of hardcoded values. The `performance_targets.py` system automatically loads these measured values and uses them for validation.

**Automatic execution**: If `benchmark_peak_results_*.json` files don't exist, `benchmark_cli.py` will automatically run peak detection (~30-60 seconds) before running benchmarks. The system gracefully continues even if peak detection fails.

To manually re-run peak benchmarks:
```bash
python tools/benchmarking/benchmark_peak.py
```

## Repository Layout
```text
code/
├── setup.sh                # End-to-end system bootstrap
├── ch1...ch20/             # Chapter walkthroughs with focused READMEs
├── scripts/                # Capture and profiling helpers
├── tools/                  # Verification utilities
└── tests/                  # PyTorch regression tests (`pytest -v tests/`)
```

## Cleanup Generated Artifacts
Remove generated artifacts, caches, and binaries:
```bash
python cleanup.py
```

## Troubleshooting

### GPU Reset
If you encounter GPU state issues, stuck processes, or need to fully reset NVIDIA GPUs:

```bash
sudo ./reset-gpu.sh
```

This script kills all processes running on the GPU, clears GPU state, reloads NVIDIA kernel modules, and performs hardware-level resets to restore GPUs to a clean state.

This script will:
- Stop GPU processes and clear compute applications
- Disable persistence mode temporarily
- Attempt NVML-based GPU resets
- Reload NVIDIA kernel modules
- Issue PCIe function-level resets if needed
- Restore original persistence mode settings

**Note:** Requires root privileges. The script will automatically use `sudo` if not run as root.

## Next Steps
- Record measured metrics or new findings for future reference
- For questions or new issues, escalate via the team's issue tracker
