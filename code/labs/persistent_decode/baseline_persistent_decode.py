"""Baseline per-token decode: one kernel launch per timestep."""

from __future__ import annotations

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)


class BaselinePersistentDecodeBenchmark(BaseBenchmark):
    """Naive decode: host loop over tokens, launch work per timestep."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        batch, seq_len, _ = resolve_shapes()
        self.seq_len = seq_len
        self.batch = batch
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())

    def setup(self) -> None:
        self.inputs = build_inputs(self.device)
        self._synchronize()

    def _decode_step(self, t: int) -> None:
        assert self.inputs is not None
        # Compute a simple dot per sequence for timestep t, then scale V.
        q_t = self.inputs.q[:, t, :]  # [batch, head_dim]
        k_t = self.inputs.k[:, t, :]
        v_t = self.inputs.v[:, t, :]

        dot = (q_t * k_t).sum(dim=-1, keepdim=True)  # [batch, 1]
        self.inputs.out[:, t, :] = v_t * dot

    def benchmark_fn(self) -> None:
        if self.inputs is None:
            raise RuntimeError("Inputs not initialized")

        with self._nvtx_range("baseline_per_token"):
            for t in range(self.seq_len):
                self._decode_step(t)
            self._synchronize()

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None

    def get_config(self) -> BenchmarkConfig:
        # Keep iterations small; focus on relative speedups and profiling
        return BenchmarkConfig(iterations=12, warmup=4)

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselinePersistentDecodeBenchmark()
