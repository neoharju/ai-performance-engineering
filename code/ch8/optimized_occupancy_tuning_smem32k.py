"""Occupancy tuning with 32KB dynamic shared memory to force SMEM as the limiter."""

from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch8.baseline_occupancy_tuning import OccupancyBinaryBenchmark


class OptimizedOccupancyTuningSMEM32K(OccupancyBinaryBenchmark):
    def __init__(self) -> None:
        super().__init__(
            friendly_name="Occupancy Tuning (smem=32KB)",
            run_args=["--smem-bytes", "32768"],
        )


def get_benchmark() -> OptimizedOccupancyTuningSMEM32K:
    """Factory for discover_benchmarks()."""
    return OptimizedOccupancyTuningSMEM32K()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(
        f"\nOccupancy Tuning (smem=32KB): {result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
