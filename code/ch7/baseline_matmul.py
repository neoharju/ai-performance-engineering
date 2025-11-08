"""Python harness wrapper for ch7's baseline_matmul.cu."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineMatmulBenchmark(CudaBinaryBenchmark):
    """Wraps the naïve matrix multiply kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_matmul",
            friendly_name="Ch7 Baseline Matmul",
            iterations=3,
            warmup=1,
            timeout_seconds=120,
            time_regex=None,
        )


def get_benchmark() -> BaselineMatmulBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineMatmulBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nCh7 Baseline Matmul: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
