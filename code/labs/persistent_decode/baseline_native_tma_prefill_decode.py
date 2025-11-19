"""Baseline native-TMA prefill vs. decode microbench (no fallbacks)."""

from __future__ import annotations

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)
from labs.persistent_decode.tma_extension import load_native_tma


class BaselineNativeTmaPrefillDecodeBenchmark(BaseBenchmark):
    """Sequential native-TMA copy/compute prefill + host decode."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.prefill_chunks = 8
        self.prefill_chunk_elems = 128 * 128
        self._tma_ext = None
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())

    def setup(self) -> None:
        self.inputs = build_inputs(self.device)
        self.prefill_src = torch.randn(
            self.prefill_chunks, self.prefill_chunk_elems, device=self.device
        )
        self.prefill_dst = torch.zeros_like(self.prefill_src)
        self._tma_ext = load_native_tma()  # raises if unsupported
        self._synchronize()

    def _prefill_native(self) -> None:
        for idx in range(self.prefill_chunks):
            self._tma_ext.tma_copy(self.prefill_src[idx], self.prefill_dst[idx])

    def _decode_host_loop(self) -> None:
        assert self.inputs is not None
        for t in range(self.seq_len):
            q_t = self.inputs.q[:, t, :]
            k_t = self.inputs.k[:, t, :]
            v_t = self.inputs.v[:, t, :]
            dot = (q_t * k_t).sum(dim=-1, keepdim=True)
            self.inputs.out[:, t, :] = v_t * dot

    def benchmark_fn(self) -> None:
        if self.inputs is None:
            raise RuntimeError("Inputs not initialized")

        with self._nvtx_range("prefill_native_baseline"):
            self._prefill_native()
        with self._nvtx_range("decode_baseline"):
            self._decode_host_loop()
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
    return BaselineNativeTmaPrefillDecodeBenchmark()
