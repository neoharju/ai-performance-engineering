"""
vLLM-backed dynamic router runner.

This replaces the virtual simulator with real LLMEngine instances. It is a thin,
opt-in harness hook: if vLLM or the model is unavailable, it raises SKIPPED.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

try:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.llm_engine import LLMEngine
    from vllm.sampling_params import SamplingParams
except Exception as exc:  # pragma: no cover - optional dep
    EngineArgs = None  # type: ignore
    LLMEngine = None  # type: ignore
    SamplingParams = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from labs.dynamic_router.optimized_router import Router, SequenceInfo
from labs.dynamic_router.baseline_router import Request


def _skip(reason: str) -> None:
    raise RuntimeError(f"SKIPPED: {reason}")


@dataclass
class _RequestRuntime:
    req: Request
    gpu_id: str
    admitted_at: float
    ttft_ms: Optional[float] = None
    finished: bool = False


class _VllmWrapper:
    """Minimal wrapper around LLMEngine for metrics and request tracking."""

    def __init__(self, gpu_id: str, device_index: int, model_id: str) -> None:
        if EngineArgs is None or LLMEngine is None or SamplingParams is None:
            _skip(f"vLLM import failed: {_IMPORT_ERROR}")

        self.gpu_id = gpu_id
        self.device_index = device_index
        engine_args = EngineArgs(
            model=model_id,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.5,
            device=f"cuda:{device_index}",
            enforce_eager=True,
        )
        self.engine = LLMEngine.from_engine_args(engine_args)
        self._inflight: Dict[str, _RequestRuntime] = {}

    def add_request(self, rt: _RequestRuntime) -> None:
        params = SamplingParams(temperature=0.0, max_tokens=rt.req.expected_new_tokens)
        self.engine.add_request(
            request_id=rt.req.req_id,
            inputs=rt.req.req_id + " " + "x" * rt.req.prompt_tokens,
            params=params,
            arrival_time=rt.admitted_at,
        )
        self._inflight[rt.req.req_id] = rt

    def step(self, now: float) -> Tuple[int, List[float], int]:
        """Advance engine; return (finished_count, ttft_samples_ms, tokens_emitted)."""
        outputs = self.engine.step()
        ttft_samples: List[float] = []
        finished = 0
        tokens_emitted = 0
        for ro in outputs:
            rid = ro.request_id
            rt = self._inflight.get(rid)
            if rt is None:
                continue
            # Detect first token
            if rt.ttft_ms is None and ro.outputs:
                total_tokens = sum(len(o.token_ids) for o in ro.outputs)
                if total_tokens > 0:
                    rt.ttft_ms = (now - rt.admitted_at) * 1000.0
                    ttft_samples.append(rt.ttft_ms)
            # Track tokens
            if ro.outputs:
                tokens_emitted += sum(len(o.token_ids) for o in ro.outputs)
            if ro.finished:
                finished += 1
                rt.finished = True
                self._inflight.pop(rid, None)
        return finished, ttft_samples, tokens_emitted

    def queue_depth(self) -> int:
        return self.engine.get_num_unfinished_requests()

    def snapshot_metrics(self, ttft_ema: float, tpot_ema: float) -> Dict[str, float]:
        mem_free_gb = 0.0
        try:
            torch.cuda.synchronize(self.device_index)
            free_bytes, _ = torch.cuda.mem_get_info(self.device_index)
            mem_free_gb = free_bytes / (1024**3)
        except Exception:
            pass
        return {
            "ttft_ms": ttft_ema,
            "tpot": tpot_ema,
            "queue_depth": float(self.queue_depth()),
            "mem_free_gb": mem_free_gb,
            "kv_hit_rate": 0.0,
        }


def run_vllm_routing(mode: str, req_count: int = 16, max_tokens: int = 16) -> Dict[str, float]:
    """Run a small vLLM-backed routing demo. Raises SKIPPED if prerequisites missing."""
    if _IMPORT_ERROR is not None:
        _skip(f"vLLM import failed: {_IMPORT_ERROR}")
    if not torch.cuda.is_available():
        _skip("CUDA is required for vLLM routing demo.")
    if torch.cuda.device_count() < 2:
        _skip("vLLM routing demo requires at least 2 GPUs.")

    model_id = os.getenv("VLLM_MODEL")
    if not model_id:
        _skip("Set VLLM_MODEL to a local HF model path or repo id to run vLLM demo.")

    # Build two engines on GPU0/GPU1
    engines = {
        "gpu0": _VllmWrapper("gpu0", 0, model_id),
        "gpu1": _VllmWrapper("gpu1", 1, model_id),
    }

    # Router selection
    router = Router() if mode == "optimized" else None
    if router:
        for gid in engines:
            router.register_gpu(gid, is_prefill=True, is_decode=True)

    requests: Dict[str, _RequestRuntime] = {}
    ttft_samples: List[float] = []
    completed = 0
    tpot_ema: Dict[str, float] = {gid: 0.0 for gid in engines}
    alpha = 0.3

    # Submit all requests up front
    for i in range(req_count):
        rid = f"req-{i}"
        req = Request(req_id=rid, prompt_tokens=64, expected_new_tokens=max_tokens)
        admitted = time.time()
        if router:
            # Round-trip through Router for placement
            gid = router.choose_prefill_gpu() or "gpu0"
        else:
            gid = "gpu0" if i % 2 == 0 else "gpu1"
        rt = _RequestRuntime(req=req, gpu_id=gid, admitted_at=admitted)
        engines[gid].add_request(rt)
        requests[rid] = rt

    active = True
    while active:
        active = False
        for gid, eng in engines.items():
            now = time.time()
            finished, ttft_new, tokens = eng.step(now)
            if finished or eng.queue_depth() > 0:
                active = True
            if ttft_new:
                ttft_samples.extend(ttft_new)
            # Update simple TPOT EMA
            if tokens > 0:
                tpot_ema[gid] = alpha * (tokens) + (1.0 - alpha) * tpot_ema[gid]
            # Push metrics into router
            if router:
                router.update_metrics(gid, eng.snapshot_metrics(ttft_ema=tpot_ema[gid], tpot_ema=tpot_ema[gid]))
        time.sleep(0.01)
        # Count completed
        completed = sum(1 for r in requests.values() if r.finished)
        if completed >= req_count:
            break

    summary: Dict[str, float] = {
        "mode": mode,
        "requests": req_count,
        "completed": completed,
        "ttft_ms_mean": float(sum(ttft_samples) / len(ttft_samples)) if ttft_samples else 0.0,
    }
    if ttft_samples:
        summary["ttft_ms_p50"] = float(torch.tensor(ttft_samples).kthvalue(len(ttft_samples) // 2 + 1).item())
        summary["ttft_ms_p95"] = float(torch.tensor(ttft_samples).kthvalue(int(len(ttft_samples) * 0.95)).item())
    else:
        summary["ttft_ms_p50"] = 0.0
        summary["ttft_ms_p95"] = 0.0
    for gid in engines:
        summary[f"tpot_tok_per_step_{gid}"] = tpot_ema[gid]
    return summary
