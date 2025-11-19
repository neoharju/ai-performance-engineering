# Persistent Decode Lab

Progressively upgrade a decode path from per-token launches to a cooperative persistent kernel with double-buffered K/V tiles and CUDA Graph capture for prefill vs. decode. Both CUDA and Triton variants ship side-by-side, and everything runs through the shared harness (no manual profiling scripts needed).

## What it runs
- `baseline_persistent_decode.py`: per-token Python/Torch decode loop (launch per token).
- `optimized_persistent_decode_triton.py`: Triton persistent decode kernel with a device work queue.
- `optimized_persistent_decode_cuda.py`: CUDA extension persistent kernel (tiled dot) driven from Python.
- `optimized_persistent_decode_graphs.py`: wraps the Triton persistent path in CUDA Graphs (prefill captured separately from decode).
- **Pseudo TMA pair**: `baseline_tma_prefill_decode.py` / `optimized_tma_prefill_decode.py` (sleep-based async copy simulation with burst shaping + decode graphs).
- **Native TMA pair**: `baseline_native_tma_prefill_decode.py` / `optimized_native_tma_prefill_decode.py` (strict native async copy path via CUDA extension; raises if the GPU has no TMA support—no fallbacks).

## Run via harness (profiles automatically)
```bash
# Full lab
python tools/cli/benchmark_cli.py run --targets labs/persistent_decode --profile

# Specific example
python tools/cli/benchmark_cli.py run --targets labs/persistent_decode:persistent_decode_triton --profile

# Prefill-vs-decode + pseudo-TMA shaping sub-lab
python tools/cli/benchmark_cli.py run --targets labs/persistent_decode:tma_prefill_decode --profile

# Native TMA prefill-vs-decode sub-lab (requires TMA-capable GPU)
python tools/cli/benchmark_cli.py run --targets labs/persistent_decode:native_tma_prefill_decode --profile
```

The harness will emit Nsight Systems/Compute traces under `benchmark_profiles/labs/persistent_decode/` and JSON timings under `artifacts/<run_id>/`.

## Nsight Systems quick-start for the TMA sub-lab

The harness will emit separate baseline/optimized traces automatically when profiling is enabled:

```bash
python tools/cli/benchmark_cli.py run \
  --targets labs/persistent_decode:tma_prefill_decode \
  --profile \
  --iterations 2 --warmup 0
```

What to inspect:
- Prefill: A/B the NVTX ranges `prefill_baseline` vs. `prefill_shaped` to see stream overlap and reduced in-flight bursts.
- Decode: `decode_baseline` shows host launch gaps; `decode_graph` collapses them via CUDA Graph replay.

## Tunable knobs
- `SEQ_LEN` (default 32), `BATCH` (8), `HEAD_DIM` (64) in `persistent_decode_common.py`.
- `NUM_PROGRAMS`: number of persistent CTAs/programs (Triton path).
- `BLOCK_K`: tile size along head_dim (Triton/CUDA paths).
- Set `QUICK=1` to shrink shapes for fast CI smoke.

## Milestones mirrored from the write-up
1. Baseline per-token launches (harness run + Nsight via `--profile`).  
2. Persistent decode (device queue) in Triton.  
3. Persistent decode in CUDA (cooperative kernel).  
4. Persistent + CUDA Graphs for prefill vs. decode capture (single graph node for decode).  
5. Planned next drops: tiled attention (QKᵀ + softmax + V) inside the persistent loop, warp specialization + double-buffered K/V tiles, and cp.async/TMA pipelining with profile deltas.
6. TMA burst shaping + decode graphs (this sub-lab) to visualize NVLink/SMEM contention shaping.
