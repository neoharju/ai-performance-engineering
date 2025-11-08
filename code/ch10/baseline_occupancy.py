"""baseline_occupancy.py - Low occupancy baseline using CUDA kernels.

Demonstrates inefficient thread block configuration leading to low GPU occupancy.
Occupancy: This baseline uses suboptimal thread block sizes, resulting in low occupancy.
Uses actual CUDA kernels from common.python.occupancy to demonstrate real occupancy differences.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)
from common.python.occupancy import run_low_occupancy


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")


class BaselineOccupancyBenchmark(Benchmark):
    """Baseline: Low occupancy kernel (inefficient thread block configuration)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.work_iters = 2048
        self.N = 16_777_216  # 16M elements (~64MB)
    
    def setup(self) -> None:
        """Setup: Initialize data."""
        torch.manual_seed(42)
        # Baseline: Low occupancy configuration
        # Uses CUDA kernels with suboptimal thread block sizes
        # Low occupancy means fewer active warps per SM, reducing GPU efficiency
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty_like(self.input)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Low occupancy computation using CUDA kernels."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("baseline_occupancy", enable=enable_nvtx):
            # Baseline: Low occupancy - inefficient thread block configuration
            # Uses CUDA kernels with small thread blocks, resulting in low GPU utilization
            # Low occupancy occurs when thread blocks are too small or use too much shared memory
            # This demonstrates real occupancy differences through actual CUDA kernel launches
            run_low_occupancy(self.input, self.output, self.work_iters)
            torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.input is None or self.output is None:
            return "Input/output tensors not initialized"
        if self.output.shape != self.input.shape:
            return "Shape mismatch between input and output"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineOccupancyBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Occupancy (Low): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

