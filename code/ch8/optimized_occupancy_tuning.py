"""Register-capped variant of occupancy_tuning for harness comparisons."""

from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch8.baseline_occupancy_tuning import OccupancyBinaryBenchmark


class OptimizedOccupancyTuningBenchmark(OccupancyBinaryBenchmark):
    """Builds with MAXRREGCOUNT=32 to force higher occupancy (at possible ILP cost)."""

    def __init__(self) -> None:
        super().__init__(
            friendly_name="Occupancy Tuning (maxrregcount=32)",
            build_env={"MAXRREGCOUNT": "32"},
        )


def get_benchmark() -> OptimizedOccupancyTuningBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedOccupancyTuningBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(f"\nOccupancy Tuning (maxrregcount=32): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
