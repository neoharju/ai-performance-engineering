"""baseline_numa_unaware.py - NUMA-unaware memory allocation (baseline).

Allocates memory without NUMA awareness - may access remote NUMA nodes.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


class BaselineNUMAUnawareBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol."""
    
    def __init__(self, size_mb: int = 512):
        self.size_mb = size_mb
        self.data = None
    
    def setup(self) -> None:
        """Setup: allocate memory without NUMA awareness."""
        # NUMA-unaware allocation - memory may be on remote NUMA nodes
        self.data = np.random.rand(self.size_mb * 1024 * 1024 // 8).astype(np.float64)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Simulate CPU-side processing (memory access)
        # Note: While NVTX is CUDA-specific, we use Python-level markers for CPU benchmarks
        # to maintain consistency with profiling infrastructure
        try:
            import torch
            if torch.cuda.is_available():
                # Use NVTX if CUDA available (for GPU-side profiling of CPU-GPU transfers)
                torch.cuda.nvtx.range_push("baseline_numa_unaware")
        except ImportError:
            pass
        
        try:
            _ = np.sum(self.data * 1.5 + 2.0)
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_pop()
            except ImportError:
                pass
    
    def teardown(self) -> None:
        """Cleanup."""
        del self.data
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data array not initialized"
        expected_size = self.size_mb * 1024 * 1024 // 8
        if self.data.size != expected_size:
            return f"Data size mismatch: expected {expected_size} elements, got {self.data.size}"
        if not np.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineNUMAUnawareBenchmark(size_mb=512)


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = BaselineNUMAUnawareBenchmark(size_mb=512)
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Baseline: NUMA-Unaware Memory Allocation")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
