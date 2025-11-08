"""baseline_stream_ordered.py - Standard memory allocation (baseline).

Demonstrates standard memory allocation without stream-ordered allocator.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)
from common.python.stream_ordered import run_standard_allocator


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")


class BaselineStreamOrderedBenchmark(Benchmark):
    """Standard memory allocation - no stream ordering."""
    
    def __init__(self):
        self.device = resolve_device()
        self.N = 5_000_000
    
    def setup(self) -> None:
        """Setup placeholder to keep harness symmetry."""
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard memory allocation pattern."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("baseline_stream_ordered_standard", enable=enable_nvtx):
            # REAL cudaMalloc baseline workload
            run_standard_allocator(self.N, iterations=3)
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=30,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineStreamOrderedBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Stream Ordered (Standard Allocation): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
