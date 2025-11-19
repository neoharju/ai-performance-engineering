"""Occupancy tuning with 128-thread blocks to illustrate block-size effects."""

from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch8.baseline_occupancy_tuning import OccupancyBinaryBenchmark


class OptimizedOccupancyTuningBS128(OccupancyBinaryBenchmark):
    def __init__(self) -> None:
        super().__init__(
            friendly_name="Occupancy Tuning (block=128)",
            run_args=["--block-size", "128"],
        )


def get_benchmark() -> OptimizedOccupancyTuningBS128:
    """Factory for discover_benchmarks()."""
    return OptimizedOccupancyTuningBS128()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(
        f"\nOccupancy Tuning (block=128): {result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
