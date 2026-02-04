"""Baseline tcgen05 TMEM benchmark exposed for harness discovery.

This simply reuses the tcgen05 baseline matmul benchmark and surfaces it under
the `tmem_tcgen05` example name within Chapter 10 (no cross-chapter aliasing).
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch10.baseline_matmul_tcgen05 import BaselineMatmulTCGen05Benchmark


class BaselineTmemTcgen05Benchmark(BaselineMatmulTCGen05Benchmark):
    """Uses the tcgen05 baseline kernel as the TMEM reference."""


def get_benchmark() -> BaselineTmemTcgen05Benchmark:
    return BaselineTmemTcgen05Benchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
