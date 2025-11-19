"""
Synthetic driver to compare baseline vs optimized routing.

Usage:
    python labs/dynamic_router/driver.py --mode baseline
    python labs/dynamic_router/driver.py --mode optimized

What it does:
  - Spawns virtual GPUs with prefill/decode roles.
  - Generates synthetic requests (prompt + decode lengths).
  - Runs a short simulation loop, logging TTFT and TPOT estimates.

This is a teaching aid: the virtual GPUs are simple queues with fixed rates.
Swap in real engine hooks (vLLM/SGLang/TRT-LLM) at the INTEGRATION POINTS to
turn this into a live experiment.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from labs.dynamic_router.baseline_router import BaselineRouter, Request
from labs.dynamic_router.optimized_router import Router, SequenceInfo

TICK_SECONDS = 0.05  # wall-clock seconds per simulation tick


# ------------------------------------------------------------
# Virtual GPU model
# ------------------------------------------------------------


@dataclass
class PrefillTask:
    req_id: str
    remaining_time: float


@dataclass
class DecodeTask:
    req_id: str
    remaining_tokens: int
    first_token_emitted: bool = False


@dataclass
class VirtualGPU:
    gpu_id: str
    is_prefill: bool
    is_decode: bool
    prefill_rate: float  # prompt tokens processed per second
    decode_rate: float  # tokens per second
    prefill_q: List[PrefillTask] = field(default_factory=list)
    decode_q: List[DecodeTask] = field(default_factory=list)
    ttft_ema: float = 0.0
    tpot_ema: float = 0.0
    queue_depth_sum: float = 0.0
    queue_depth_max: int = 0
    queue_depth_samples: int = 0

    def enqueue_prefill(self, req: Request) -> None:
        prompt = max(1, req.prompt_tokens)
        remaining = prompt / self.prefill_rate
        self.prefill_q.append(PrefillTask(req.req_id, remaining))

    def enqueue_decode(self, req_id: str, expected_new_tokens: int) -> None:
        tokens = max(1, expected_new_tokens)
        self.decode_q.append(DecodeTask(req_id, tokens))

    def step(self, tick_s: float) -> Tuple[List[str], List[Tuple[str, int, bool]]]:
        """
        Advance one tick.

        Returns:
          - completed_prefills: list of req_ids that finished prefill
          - decode_events: list of (req_id, tokens_emitted, first_token_emitted)
        """
        completed_prefills: List[str] = []
        decode_events: List[Tuple[str, int, bool]] = []

        self._record_queue_depth()

        # Process prefill (one-at-a-time queue)
        if self.prefill_q:
            task = self.prefill_q[0]
            task.remaining_time -= tick_s
            if task.remaining_time <= 0:
                completed_prefills.append(task.req_id)
                self.prefill_q.pop(0)

        # Process decode tasks (simple FCFS, one-at-a-time)
        if self.decode_q:
            task = self.decode_q[0]
            tokens = min(task.remaining_tokens, math.ceil(self.decode_rate * tick_s))
            task.remaining_tokens -= tokens
            decode_events.append((task.req_id, tokens, not task.first_token_emitted))
            task.first_token_emitted = True
            if task.remaining_tokens <= 0:
                self.decode_q.pop(0)

        return completed_prefills, decode_events

    def update_smoothed_metrics(
        self, ttft_ms_samples: List[float], tokens_emitted: int, alpha: float = 0.3
    ) -> None:
        """Update simple EMAs for TTFT and tokens-per-occupied-second."""
        for sample in ttft_ms_samples:
            self.ttft_ema = alpha * sample + (1.0 - alpha) * self.ttft_ema
        occupied = 1.0 if (self.prefill_q or self.decode_q) else 0.0
        if occupied:
            tpot = tokens_emitted / TICK_SECONDS
            self.tpot_ema = alpha * tpot + (1.0 - alpha) * self.tpot_ema

    def _record_queue_depth(self) -> None:
        depth = len(self.prefill_q) + len(self.decode_q)
        self.queue_depth_samples += 1
        self.queue_depth_sum += depth
        self.queue_depth_max = max(self.queue_depth_max, depth)

    def queue_depth_avg(self) -> float:
        if self.queue_depth_samples == 0:
            return 0.0
        return self.queue_depth_sum / float(self.queue_depth_samples)

    def metrics_snapshot(self) -> Dict[str, float]:
        queue_depth = len(self.prefill_q) + len(self.decode_q)
        mem_free_gb = max(0.0, 40.0 - queue_depth * 0.5)  # toy model
        return {
            "ttft_ms": self.ttft_ema,
            "tpot": self.tpot_ema,
            "queue_depth": float(queue_depth),
            "mem_free_gb": mem_free_gb,
            "kv_hit_rate": 0.0,
        }


# ------------------------------------------------------------
# Simulation harness
# ------------------------------------------------------------


@dataclass
class RequestState:
    req: Request
    admitted_at: float
    prefill_gpu: Optional[str] = None
    decode_gpu: Optional[str] = None
    ttft_ms: Optional[float] = None
    finished: bool = False


def make_virtual_gpus() -> Dict[str, VirtualGPU]:
    """Create a small fleet: two prefill-oriented GPUs and two decode-oriented GPUs."""
    return {
        "pf0": VirtualGPU("pf0", is_prefill=True, is_decode=False, prefill_rate=8000, decode_rate=2000),
        "pf1": VirtualGPU("pf1", is_prefill=True, is_decode=False, prefill_rate=7000, decode_rate=1800),
        "dc0": VirtualGPU("dc0", is_prefill=False, is_decode=True, prefill_rate=4000, decode_rate=3200),
        "dc1": VirtualGPU("dc1", is_prefill=False, is_decode=True, prefill_rate=4000, decode_rate=2800),
    }


def build_optimized_router(gpus: Dict[str, VirtualGPU]) -> Router:
    r = Router()
    for gid, gpu in gpus.items():
        r.register_gpu(gid, is_prefill=gpu.is_prefill, is_decode=gpu.is_decode)
    return r


def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    assert 0.0 <= pct <= 100.0
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data_sorted[int(k)]
    d0 = data_sorted[int(f)] * (c - k)
    d1 = data_sorted[int(c)] * (k - f)
    return d0 + d1


def simulate(
    mode: str,
    num_ticks: int = 400,
    seed: int = 0,
) -> Dict[str, float]:
    random.seed(seed)
    gpus = make_virtual_gpus()

    baseline = BaselineRouter(gpus.keys()) if mode == "baseline" else None
    optimized = build_optimized_router(gpus) if mode == "optimized" else None

    requests: Dict[str, RequestState] = {}
    next_id = 0
    completed_ttfts: List[float] = []
    completed_count = 0

    for tick in range(num_ticks):
        now = tick * TICK_SECONDS

        # 1) Generate new requests
        new_requests = random.randint(0, 3)
        for _ in range(new_requests):
            req = Request(
                req_id=f"req-{next_id}",
                prompt_tokens=random.randint(100, 1200),
                expected_new_tokens=random.randint(32, 256),
                priority=random.randint(0, 2),
            )
            next_id += 1

            if mode == "baseline":
                gpu_id = baseline.route(req)  # type: ignore[union-attr]
                requests[req.req_id] = RequestState(req=req, admitted_at=now, prefill_gpu=gpu_id, decode_gpu=gpu_id)
                gpus[gpu_id].enqueue_prefill(req)
            else:
                gpu_id = optimized.choose_prefill_gpu()  # type: ignore[union-attr]
                if gpu_id is None:
                    gpu_id = "pf0"
                requests[req.req_id] = RequestState(req=req, admitted_at=now, prefill_gpu=gpu_id)
                gpus[gpu_id].enqueue_prefill(req)

        # 2) Update metrics -> Router for optimized
        if optimized:
            for gid, gpu in gpus.items():
                optimized.update_metrics(gid, gpu.metrics_snapshot())

        # 3) Step GPUs and route inter-stage traffic
        for gid, gpu in gpus.items():
            completed_prefills, decode_events = gpu.step(TICK_SECONDS)

            # Prefill completions -> decode admission
            for rid in completed_prefills:
                state = requests[rid]
                if mode == "baseline":
                    gpu.enqueue_decode(rid, state.req.expected_new_tokens)
                else:
                    seq = SequenceInfo(
                        seq_id=rid,
                        current_gpu=state.prefill_gpu or gid,
                        kv_gpus={gid},
                        expected_tokens_remaining=state.req.expected_new_tokens,
                        priority=state.req.priority,
                    )
                    dst_gpu = optimized.choose_decode_gpu(seq)  # type: ignore[union-attr]
                    dst_gpu = dst_gpu or gid
                    state.decode_gpu = dst_gpu
                    gpus[dst_gpu].enqueue_decode(rid, state.req.expected_new_tokens)

            # Decode events -> TTFT + throughput stats
            ttft_samples: List[float] = []
            tokens_emitted = 0
            for rid, tokens, first in decode_events:
                tokens_emitted += tokens
                state = requests[rid]
                if first and state.ttft_ms is None:
                    state.ttft_ms = (now - state.admitted_at) * 1000.0
                    completed_ttfts.append(state.ttft_ms)
                    ttft_samples.append(state.ttft_ms)
            gpu.update_smoothed_metrics(ttft_samples, tokens_emitted)

        # 4) Optional: migrate (optimized only)
        if optimized and tick % 5 == 0:
            active = []
            for state in requests.values():
                if state.ttft_ms is not None and state.decode_gpu is not None:
                    # Only consider active decode sequences
                    if any(t.req_id == state.req.req_id for t in gpus[state.decode_gpu].decode_q):
                        active.append(
                            SequenceInfo(
                                seq_id=state.req.req_id,
                                current_gpu=state.decode_gpu,
                                kv_gpus={state.decode_gpu},
                                expected_tokens_remaining=None,
                                priority=state.req.priority,
                            )
                        )
            migrations = optimized.plan_migrations(active, max_per_call=2)
            for rid, src, dst in migrations:
                src_gpu = gpus[src]
                dst_gpu = gpus[dst]
                for idx, task in enumerate(src_gpu.decode_q):
                    if task.req_id == rid:
                        dst_gpu.decode_q.append(task)
                        src_gpu.decode_q.pop(idx)
                        break

        # 5) Clean up finished requests
        finished = [
            rid
            for rid, state in requests.items()
            if state.req.req_id
            not in [t.req_id for gpu in gpus.values() for t in gpu.prefill_q + gpu.decode_q]
        ]
        for rid in finished:
            if mode == "baseline":
                baseline.complete(rid)  # type: ignore[union-attr]
            state = requests.pop(rid, None)
            if state:
                state.finished = True
                completed_count += 1

        # Optional slow logging
        if tick % 20 == 0:
            avg_ttft = [
                s.ttft_ms for s in requests.values() if s.ttft_ms is not None
            ]
            ttft_str = f"{sum(avg_ttft)/len(avg_ttft):.1f} ms" if avg_ttft else "n/a"
            if optimized:
                scores = {
                    gid: optimized._score_decode_gpu(gpus[gid], kv_local=False)  # type: ignore[attr-defined]
                    for gid, g in gpus.items()
                    if g.is_decode
                }
                print(
                    f"[tick {tick:03d}] mode={mode} active={len(requests)} "
                    f"avg_ttft={ttft_str} decode_scores={scores}"
                )
            else:
                print(f"[tick {tick:03d}] mode={mode} active={len(requests)} avg_ttft={ttft_str}")

        time.sleep(0.0)

    print(f"\nDone. Mode={mode} | completed={completed_count} | remaining={len(requests)}")

    # Summary metrics
    summary: Dict[str, float] = {
        "mode": mode,
        "seed": seed,
        "ticks": num_ticks,
        "completed": completed_count,
        "remaining": len(requests),
    }

    if completed_ttfts:
        summary.update(
            {
                "ttft_ms_mean": statistics.mean(completed_ttfts),
                "ttft_ms_p50": _percentile(completed_ttfts, 50.0),
                "ttft_ms_p95": _percentile(completed_ttfts, 95.0),
            }
        )
    else:
        summary.update({"ttft_ms_mean": 0.0, "ttft_ms_p50": 0.0, "ttft_ms_p95": 0.0})

    decode_tpots = [g.tpot_ema for g in gpus.values() if g.is_decode]
    prefill_tpots = [g.tpot_ema for g in gpus.values() if g.is_prefill]
    summary["avg_decode_tpot_tok_per_s"] = statistics.mean(decode_tpots) if decode_tpots else 0.0
    summary["avg_prefill_tpot_tok_per_s"] = statistics.mean(prefill_tpots) if prefill_tpots else 0.0

    for gid, gpu in gpus.items():
        summary[f"queue_depth_avg_{gid}"] = gpu.queue_depth_avg()
        summary[f"queue_depth_max_{gid}"] = float(gpu.queue_depth_max)
        summary[f"tpot_tok_per_s_{gid}"] = gpu.tpot_ema
        summary[f"ttft_ms_ema_{gid}"] = gpu.ttft_ema

    return summary


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic routing lab driver")
    p.add_argument("--mode", choices=["baseline", "optimized"], default="baseline")
    p.add_argument("--ticks", type=int, default=400, help="simulation ticks")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    simulate(
        args.mode,
        num_ticks=args.ticks,
        seed=args.seed,
    )
