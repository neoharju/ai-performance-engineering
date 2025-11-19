"""Optimized prefill vs. decode microbench with simple TMA burst shaping.

What it demonstrates for Nsight Systems:
- Prefill: double-buffered-ish pipeline using multiple streams + max_in_flight
  guard to show how shaping reduces contention.
- Decode: graph-captured token loop to trim host launch gaps.
"""

from __future__ import annotations

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)


class TmaBurstConfig:
    """Simple config holder for burst shaping knobs."""

    def __init__(self, chunk_k: int = 128, max_in_flight: int = 2, tma_sleep_cycles: int = 50_000) -> None:
        self.chunk_k = chunk_k
        self.max_in_flight = max_in_flight
        # torch.cuda._sleep argument is in cycles; keep high enough to visualize overlap.
        self.tma_sleep_cycles = tma_sleep_cycles


class OptimizedTmaPrefillDecodeBenchmark(BaseBenchmark):
    """Prefill with shaped pseudo-TMA + decode hosted in a CUDA Graph."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.prefill_chunks = 8
        self.prefill_chunk_elems = 128 * 128
        self.cfg = TmaBurstConfig()
        self.prefill_streams = [torch.cuda.Stream() for _ in range(self.cfg.max_in_flight)]
        self.decode_graph = torch.cuda.CUDAGraph()
        self.graph_q = None
        self.graph_k = None
        self.graph_v = None
        self.graph_out = None
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())

    def setup(self) -> None:
        self.inputs = build_inputs(self.device)
        self.prefill_src = torch.randn(
            self.prefill_chunks, self.prefill_chunk_elems, device=self.device
        )
        self.prefill_dst = torch.zeros_like(self.prefill_src)

        # Graph-captured decode loop to cut host gaps during profiling.
        self.graph_q = self.inputs.q.clone()
        self.graph_k = self.inputs.k.clone()
        self.graph_v = self.inputs.v.clone()
        self.graph_out = torch.zeros_like(self.inputs.out)

        capture_stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.graph(self.decode_graph, stream=capture_stream):
            self._decode_body(self.graph_q, self.graph_k, self.graph_v, self.graph_out)
        torch.cuda.synchronize()

    def _decode_body(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor
    ) -> None:
        # Simple per-token dot product.
        for t in range(self.seq_len):
            q_t = q[:, t, :]
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            dot = (q_t * k_t).sum(dim=-1, keepdim=True)
            out[:, t, :] = v_t * dot

    def _prefill_shaped(self) -> None:
        """Launch pseudo-TMA copies on multiple streams with a max_in_flight cap."""
        events = []
        for idx in range(self.prefill_chunks):
            stream = self.prefill_streams[idx % len(self.prefill_streams)]
            with torch.cuda.stream(stream):
                torch.cuda._sleep(self.cfg.tma_sleep_cycles)
                self.prefill_dst[idx].add_(self.prefill_src[idx])
            evt = torch.cuda.Event(enable_timing=False, blocking=False)
            evt.record(stream)
            events.append(evt)
            if len(events) > self.cfg.max_in_flight:
                events.pop(0).synchronize()

        # Drain remaining work.
        for evt in events:
            evt.synchronize()

    def _decode_graph(self) -> None:
        assert self.inputs is not None
        # Refresh graph inputs to show a realistic copy-before-replay pattern.
        self.graph_q.copy_(self.inputs.q)
        self.graph_k.copy_(self.inputs.k)
        self.graph_v.copy_(self.inputs.v)
        self.graph_out.zero_()
        self.decode_graph.replay()
        # Mirror back to inputs.out so validation stays consistent.
        self.inputs.out.copy_(self.graph_out)

    def benchmark_fn(self) -> None:
        if self.inputs is None:
            raise RuntimeError("Inputs not initialized")

        with self._nvtx_range("prefill_shaped"):
            self._prefill_shaped()
        with self._nvtx_range("decode_graph"):
            self._decode_graph()
        self._synchronize()

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=8, warmup=2)

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedTmaPrefillDecodeBenchmark()
