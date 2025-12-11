"""Optimized FlashAttention lab: Gluon/Triton warp-specialized kernel (fallback to flash-attn)."""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.flashattention_gluon.flashattention_gluon_common import (
    FlashAttentionInputs,
    FlashAttentionKernel,
    build_flashattention_inputs,
    resolve_gluon_flash_attention,
)


class OptimizedFlashAttentionGluonBenchmark(BaseBenchmark):
    """Optimized attention: fused kernel (prefer Gluon warp specialization; fallback flash-attn)."""

    def __init__(self) -> None:
        super().__init__()
        self.batch = 2
        self.seq_len = 1024
        self.heads = 8
        self.head_dim = 64
        self.dtype = torch.float16
        self.inputs: Optional[FlashAttentionInputs] = None
        self.kernel: Optional[FlashAttentionKernel] = None
        self.output: Optional[torch.Tensor] = None
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(0)
        self.kernel = resolve_gluon_flash_attention()
        self.inputs = build_flashattention_inputs(
            batch=self.batch,
            seq_len=self.seq_len,
            heads=self.heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None or self.kernel is None:
            raise RuntimeError("FlashAttention inputs/kernel not initialized")

        with torch.inference_mode():
            with self._nvtx_range(f"flashattention_optimized_{self.kernel.provider}"):
                result = self.kernel.fn(self.inputs.q, self.inputs.k, self.inputs.v)
                self.output = result.detach().float().clone()
        self._synchronize()

    def teardown(self) -> None:
        self.inputs = None
        self.kernel = None
        self.output = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return {"provider": self.kernel.provider if self.kernel else "unset"}

    def validate_result(self) -> Optional[str]:
        if self.inputs is None or self.kernel is None:
            return "FlashAttention inputs/kernel not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        return self.output

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {
            "batch": self.batch,
            "seq_len": self.seq_len,
            "heads": self.heads,
            "head_dim": self.head_dim,
            "shapes": {
                "q": (self.batch, self.heads, self.seq_len, self.head_dim),
                "k": (self.batch, self.heads, self.seq_len, self.head_dim),
                "v": (self.batch, self.heads, self.seq_len, self.head_dim),
            },
            "dtypes": {"q": str(self.dtype), "k": str(self.dtype), "v": str(self.dtype)},
        }

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return OptimizedFlashAttentionGluonBenchmark()
