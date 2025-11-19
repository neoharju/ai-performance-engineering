# Dynamic Prefill/Decode Routing Lab

This lab mirrors the book chapter on inference routing. It gives you a toy baseline (no feedback) and an optimized router (prefill/decode pools, TTFT/TPOT scoring, KV-aware decode placement, migration budget).

## Files
- `baseline_router.py`: single-pool round-robin, no TTFT/TPOT, no KV locality or migration.
- `optimized_router.py`: EWMA-smoothed metrics, pool-aware routing, KV locality hints, migration planner.
- `driver.py`: synthetic simulator to compare baseline vs optimized.
- `baseline_dynamic_router_vllm.py` / `optimized_dynamic_router_vllm.py`: harness targets that drive real vLLM engines (requires CUDA GPUs, vLLM installed, and `VLLM_MODEL` pointing to a local HF model path/id).

## Run
```bash
python labs/dynamic_router/driver.py --mode baseline
python labs/dynamic_router/driver.py --mode optimized
# Or via the harness (preferred for consistency):
python tools/cli/benchmark_cli.py run --targets labs/dynamic_router
# vLLM pair (opt-in, requires model + 2 GPUs):
VLLM_MODEL=/path/to/model python tools/cli/benchmark_cli.py run --targets labs/dynamic_router:dynamic_router_vllm
```
The driver spins virtual GPUs, generates random requests, and logs average TTFT plus decode scores so you can see the control loop react.

## Concepts to watch
- **Prefill vs decode specialization:** Prefill GPUs handle bursty GEMMs; decode GPUs stay cache-hot for steady token streaming.
- **Signals:** TTFT as the latency leading indicator; TPOT (tokens per occupied time) as throughput indicator; queue depth and free HBM as congestion proxies.
- **Routing policy:** Score GPUs (`α/TTFT + β·TPOT + γ·mem − δ·queue`) separately for prefill and decode; add a KV-locality boost when a GPU already holds the sequence’s KV.
- **Admission shaping:** When TTFT inflates, narrow concurrent prefills; reopen when TTFT is healthy.
- **Migration budget:** Allow limited mid-batch moves only when the score gap is meaningful; cap moves per window to avoid thrash.

## Next steps (optional)
- Replace the virtual GPUs with real engines (vLLM/SGLang/TRT-LLM) by wiring their telemetry into `update_metrics` and swap enqueue/dequeue points with your scheduler hooks.
- Add logging for per-request pool transitions and migration latency to verify tail improvements.
