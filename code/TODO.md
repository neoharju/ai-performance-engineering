# TODO

## Objective
Standardize NVTX taxonomy across all chapters/labs, ensure minimal/speed-of-light profiling runs work with NVTX markers, and complete a full-suite benchmark run with the queue runner. Fix NVTX-related failures and re-queue failed targets until the suite is clean.

## Current Status
- Full-suite run in progress: `python -m cli.aisp bench run --targets all --profile minimal --ncu-metric-set minimal --update-expectations --run-id nvtx_full_minimal_20260131_012119` via `artifacts/parallel_runs/run_queue.sh`.
- Queue monitoring active via `artifacts/parallel_runs/queue.log` and run log via `artifacts/runs/nvtx_full_minimal_20260131_012119/logs/benchmark.log`.
- NVTX taxonomy standardized in core helpers and applied across chapters/labs (see recent changes).

## What’s Left
1. Let the full-suite run finish; keep monitoring queue/logs for failures and stalls.
2. Scan `queue.log` + `benchmark.log` for NVTX-related failures after completion.
3. Patch any NVTX failures or misleading markers discovered.
4. Re-queue failed targets using the same runner (`artifacts/parallel_runs/run_queue.sh --enqueue ...`).
5. Continue monitoring and auto-requeue until the suite completes without NVTX-related failures.

## Notes
- Multiple queue runners detected (`queue.sh` and `run_queue.sh`); per policy we should keep a single runner, but no termination was requested.

## Summary (NVTX Workstream)
We are standardizing NVTX taxonomy across chapters/labs and ensuring minimal/speed‑of‑light profiling runs succeed with meaningful NVTX markers, then completing a full‑suite benchmark run via the existing queue runner and re‑queueing any NVTX‑related failures until clean.

## Request recap (append-only)

### What I’m trying to achieve (short summary)
- Preserve the book‑parity warp‑specialized tcgen05 kernel while adding CUTLASS‑aligned reference + advanced (2‑SM) variants.
- Validate correctness with a direct functional check.
- Benchmark the three tcgen05 targets with minimal profiling and updated expectations.
- If speedup < 1.05×, increase sizes for baseline/optimized across all three targets and re‑queue.

### What’s left to do (short checklist)
1. Re‑run the direct functional check once the queue is idle (or isolate it with a longer timeout).
2. Confirm the queued tcgen05 minimal‑profile sweep starts/completes in `artifacts/parallel_runs/queue.log`.
3. If any speedup < 1.05×, bump sizes in all three baseline/optimized pairs and re‑queue with `--update-expectations`.
4. Summarize results and note which expectations file(s) changed.

---

## Summary (Artifacts + ch11 Smoke Test)
We are finishing the migration to `artifacts/runs` across tooling/docs and running a focused deep‑dive smoke test for `ch11:tensor_cores_streams` (baseline vs optimized) through the queue runner.

### What I’m trying to achieve (short summary)
- Normalize remaining tooling/docs references to `artifacts/runs`.
- Run `python -m cli.aisp bench run --profile deep_dive ... -t ch11:tensor_cores_streams` via the queue runner.
- Confirm the smoke test completes and report any failure logs.

### What’s left to do (short checklist)
1. Wait for the queue to go idle and the ch11 smoke test to actually start.
2. Confirm completion via `END` entry in `artifacts/parallel_runs/queue.log`.
3. If it fails, capture the log path from the queue log and report back.

---

## Summary (Queue Runner Multiline Fix)
We are fixing the queue runner so multiline commands (heredocs / embedded Python) are preserved as a single queue entry, then validating the new `--enqueue-file` workflow with a small non‑benchmark test.

### What I’m trying to achieve (short summary)
- Make `artifacts/parallel_runs/run_queue.sh` safe for multiline commands by auto‑encoding them and decoding at execution time.
- Add a first‑class `--enqueue-file` path to avoid quoting issues entirely.
- Verify the new behavior with a tiny queued script that writes `artifacts/parallel_runs/queue_multiline_test.txt`.

### What’s left to do (short checklist)
1. Re‑run the queue so the multiline test entry executes.
2. Confirm `artifacts/parallel_runs/queue_multiline_test.txt` exists with `ok`.
3. Note the successful execution in `artifacts/parallel_runs/queue.log`.

---

## Summary (ch12 Kernel Fusion Deep‑Dive Compare Re‑run)
We are re‑running the deep‑dive compare for `ch12:kernel_fusion_llm_dedicated_stream_and_prefetch_for_blackwell` with higher warmup/iterations, then reporting NCU/NSYS deltas.

### What I’m trying to achieve (short summary)
- Run the deep‑dive compare with `iterations=3` and `warmup=10`.
- Extract NCU/NSYS deltas from the new `deep_dive_compare.json`.
- Report those deltas back to you.

### What’s left to do (short checklist)
1. Let the current queued ch10 minimal run finish (it’s ahead of the compare in the queue).
2. Wait for the queued deep‑dive compare to complete and write:
   - `artifacts/parallel_runs/deep_dive_compare_ch12_kernel_fusion_llm_dedicated_stream_and_prefetch_for_blackwell.json`
   - `artifacts/.../reports/deep_dive_compare.json` under the compare run directory.
3. Parse the report and summarize NCU/NSYS deltas.

---

## Summary (MCP Tool Selection Clarity)
We are improving MCP tool selection accuracy by enriching tool descriptions with explicit “Selection” guidance and tightening schema descriptions for benchmark inputs/outputs to reduce ambiguous tool picks.

### What I’m trying to achieve (short summary)
- Add a small, consistent “Selection” hint to disambiguate commonly confused tools (triage vs status vs suggest, report vs export vs compare, nsys vs ncu vs torch, ask vs explain).
- Clarify benchmark target formats and results file parameters so callers choose the correct benchmark/report/export tool.
- Fix any confusing self-referential wording in tool descriptions.

### What’s left to do (short checklist)
1. Spot‑check `python -m mcp.mcp_server --list` output to confirm the new “Selection” hints read cleanly.
2. If any tool description feels noisy, trim or refine the hint text.

---

## Summary (MCP Compare + Profiling Metrics & Test Cleanup)
We are making MCP compare tools always return NCU/NSYS metrics when captured, ensuring profiling tools include metrics payloads, tightening fixture-based assertions, and driving the full test suite clean again.

### What I’m trying to achieve (short summary)
- Guarantee compare tools include NCU/NSYS metrics when profiles exist (and nudge descriptions to analyze those metrics).
- Ensure profiling tools return JSON metrics payloads consistently (nsys/ncu/torch/hta).
- Add real local fixtures under `artifacts/runs/mcp-fixtures/...` (not checked in) and tighten test assertions when fixtures exist.
- Fix all full-suite test failures introduced or surfaced by these changes.

### What’s left to do (short checklist)
1. Repair determinism audit checks: add explicit determinism toggle detection + allowlist comments in `core/scripts/audit_verification_compliance.py`, and ensure status/warnings align with `tests/test_audit_determinism_policy.py`.
2. Resolve baseline/optimized orphan check by excluding or relocating generated `llm_patches` dirs (policy says don’t delete; needs code-side ignore or move).
3. Reconcile defaults: align `BenchmarkDefaults` vs `BenchmarkConfig` for `enable_nsys/enable_ncu` and adjust config merge to accept the new `cli_ncu_replay_mode_provided` kwarg in `_compute_locked_fields`.
4. Fix virtualization enforcement: make env validation fail before timing cross‑validation (e.g., short‑circuit when invalid) so `test_environment_validation_enforcement_virtualization_detected` sees the correct error.
5. Add missing benchmark domain ops (`overview`, `compare`) or remove those engine_op references in `core/api/registry.py` so `tests/test_interface_consistency.py` passes.
6. Stabilize MCP slow‑tool tests: ensure `aisp_run_benchmarks` and `aisp_benchmark_deep_dive_compare` complete in the test timeout or adjust tool behavior for test mode.
7. Fix `discover_benchmark_pairs` in `core/scripts/validate_benchmark_pairs.py` to accept non‑validated files for test fixtures (or provide a toggle) so `tests/test_validate_benchmark_pairs_tools.py` passes.
8. Re-run full suite and tighten fixture-based assertions further once the above are resolved.

---

## Summary (High Findings Fixes: Dashboard UI ↔ API)
We are aligning the dashboard UI with the new backend API field names so the Software Stack pill and Dependencies widget display real data instead of undefined/always‑OK.

### What I’m trying to achieve (short summary)
- Map backend `system.software` keys to the UI expectations (`python_version`, `pytorch_version`, `cuda_version`, `triton_version`).
- Make the Dependencies widget reflect `issues`/`warnings` from the backend (or update the backend to emit the UI’s `missing`/`outdated` keys).
- Keep the simplified dashboard working end‑to‑end with the new API registry/envelope.

### HIGH findings to fix (from review)
1. Software Stack widget expects `python_version`/`pytorch_version`/`cuda_version`/`triton_version`, but backend returns `python`/`pytorch`/`cuda_runtime`/`triton`, so the pill renders `undefined`.
2. Dependencies widget checks `missing`/`outdated`, but backend returns `issues`/`warnings`, so the UI always shows “All deps OK” even when warnings exist.

### What’s left to do (short checklist)
1. Update the frontend to tolerate backend field names (or add a backend compatibility layer in the API response).
2. Normalize dependency health shape so `DependenciesWidget` reflects real warnings.
3. Re-run the dashboard smoke check after the mapping fix (manual run is enough).

### Recommendation
- Prefer a backend compatibility layer in `core/api/handlers.py` for `/api/system/software` and `/api/system/dependencies` so all clients (not just the dashboard) get stable, UI‑friendly field names without duplicating mapping logic across frontends.
