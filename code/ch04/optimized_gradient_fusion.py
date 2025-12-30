"""Optimized gradient fusion benchmark (single GPU, fused reductions)."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch04.gradient_fusion_common import GradientFusionBenchmark, attach_benchmark_metadata


def get_benchmark() -> BaseBenchmark:
    bench = GradientFusionBenchmark(fused=True, equivalence_group="ch04_gradient_fusion_single")
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
