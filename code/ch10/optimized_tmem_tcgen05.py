"""Optimized TMEM comparison benchmark (cuBLAS) for harness discovery.

This re-exports the cuBLAS matmul benchmark under the `tmem_tcgen05` example
name within Chapter 10.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch10.optimized_matmul_tcgen05 import OptimizedMatmulTCGen05Benchmark


class OptimizedTmemTcgen05Benchmark(OptimizedMatmulTCGen05Benchmark):
    """cuBLAS matmul wrapper for TMEM comparison."""


def get_benchmark() -> OptimizedTmemTcgen05Benchmark:
    return OptimizedTmemTcgen05Benchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
