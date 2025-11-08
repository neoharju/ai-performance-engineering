"""optimized_occupancy.py - High occupancy variant using CUDA kernels.

Demonstrates optimal thread block configuration for maximum GPU occupancy.
Occupancy: This optimized version uses optimal thread block sizes to maximize occupancy.
High occupancy improves GPU utilization and hides memory latency.
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

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)
from common.python.occupancy import run_high_occupancy


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")


class OptimizedOccupancyBenchmark(Benchmark):
    """Optimized: High occupancy kernel (optimal thread block configuration)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.work_iters = 2048
        self.N = 16_777_216  # 16M elements (~64MB)
    
    def setup(self) -> None:
        """Setup: Initialize data."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        # Optimization: High occupancy configuration
        # Uses CUDA kernels with optimal thread block sizes
        # High occupancy improves GPU utilization and hides memory latency
        # More active warps per SM means better latency hiding through warp scheduling
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty_like(self.input)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: High occupancy computation using CUDA kernels."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_occupancy", enable=enable_nvtx):
            # Optimization: High occupancy - optimal thread block configuration
            # Uses CUDA kernels with optimal thread block sizes for maximum GPU utilization
            # High occupancy hides memory latency through warp scheduling
            # More active warps per SM means better GPU resource utilization
            # High occupancy is achieved through:
            # 1. Optimal thread block sizes (typically 128-256 threads)
            # 2. Minimizing shared memory usage per block
            # 3. Maximizing register efficiency
            # 4. Using occupancy APIs like cudaOccupancyMaxActiveBlocksPerSM
            run_high_occupancy(self.input, self.output, self.work_iters)
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
    return OptimizedOccupancyBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Occupancy (High): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

