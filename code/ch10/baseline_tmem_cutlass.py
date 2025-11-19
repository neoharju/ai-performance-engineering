"""Baseline CUTLASS TMEM benchmark (tcgen05) exposed for harness discovery.

This simply reuses the tcgen05 baseline matmul benchmark and surfaces it under
the `tmem_cutlass` example name within Chapter 10 (no cross-chapter aliasing).
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch10.baseline_matmul_tcgen05 import BaselineMatmulTCGen05Benchmark


class BaselineTmemCutlassBenchmark(BaselineMatmulTCGen05Benchmark):
    """Uses the torch matmul baseline as the CUTLASS/TMEM reference."""


def get_benchmark() -> BaselineTmemCutlassBenchmark:
    return BaselineTmemCutlassBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(
        f"\nBaseline TMEM CUTLASS: {result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
