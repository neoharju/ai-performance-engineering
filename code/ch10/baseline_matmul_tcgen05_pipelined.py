"""Baseline matmul benchmark: Single-stage tcgen05 kernel.

CHAPTER 10 CONTEXT: This is the basic tcgen05 kernel without software
pipelining. Compare against the pipelined tcgen05 variant to see how
double-buffering and overlap help.

WHY THIS IS SLOWER:
1. Single-stage (no overlap between load/compute)
2. No double-buffered shared memory
3. Fewer scheduling optimizations
"""

from __future__ import annotations

from typing import Optional

import torch

from ch10.matmul_extension_tcgen05 import load_matmul_tcgen05_module
from ch10.optimized_matmul import resolve_device
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.tcgen05_requirements import check_tcgen05_support


class BaselineMatmulTCGen05PipelinedBenchmark(BaseBenchmark):
    """Single-stage tcgen05 baseline (no pipelining)."""

    def __init__(self) -> None:
        super().__init__()
        available, reason = check_tcgen05_support(
            loader=None,
            module_name="ch10 matmul tcgen05 kernels",
        )
        self._tcgen05_available = available
        self._skip_reason = reason or "SKIPPED: tcgen05 matmul unavailable"
        self.device = resolve_device()
        self.dtype = torch.float16
        # Match other matmul benchmarks (baseline_matmul.py uses n=8192)
        self.n = 8192
        self.size = self.n
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self.module = None
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(bytes_per_iteration=float(self.n * self.n * 2 * 3))

    def setup(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        if self.module is None:
            self.module = load_matmul_tcgen05_module()
        torch.manual_seed(0)
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=self.dtype)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=self.dtype)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        assert self.A is not None and self.B is not None and self.module is not None
        with self._nvtx_range("baseline_matmul_tcgen05_single_stage"):
            with torch.no_grad():
                _ = self.module.matmul_tcgen05(self.A, self.B)
        self._synchronize()

    def teardown(self) -> None:
        self.A = None
        self.B = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics."""
        flops = 2 * self.size ** 3
        return {
            "matrix_size": self.size,
            "theoretical_flops": flops,
            "library": "cuBLAS",
        }

    def validate_result(self) -> Optional[str]:
        if not self._tcgen05_available:
            return self._skip_reason
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification."""
        if self.output is None:
            raise RuntimeError("Output not available - run benchmark first")
        return self.output.float()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"size": self.size}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison - wider due to FP16."""
        return (0.5, 5.0)


def get_benchmark() -> BaselineMatmulTCGen05PipelinedBenchmark:
    return BaselineMatmulTCGen05PipelinedBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
