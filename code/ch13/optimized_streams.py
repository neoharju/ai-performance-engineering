"""optimized_streams.py - Optimized CUDA streams for parallel execution.

Demonstrates CUDA streams for parallel execution of independent operations.
Streams: Uses CUDA streams to overlap independent operations.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class OptimizedStreamsBenchmark(Benchmark):
    """Optimized: CUDA streams for parallel execution of independent operations.
    
    Streams: Uses CUDA streams to overlap independent operations.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input1 = None
        self.input2 = None
        self.stream1 = None
        self.stream2 = None
    
    def setup(self) -> None:
        """Setup: Initialize model and CUDA streams."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        # Optimization: CUDA streams allow parallel execution of independent operations
        # Multiple streams enable overlapping computation and memory transfers
        # This improves GPU utilization by executing operations concurrently
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        # Create CUDA streams for parallel execution
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()
        
        self.input1 = torch.randn(32, 1024, device=self.device)
        self.input2 = torch.randn(32, 1024, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Parallel execution with CUDA streams."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        # Optimization: Use CUDA streams for parallel execution
        # Operations on different streams can execute concurrently
        with nvtx_range("optimized_streams", enable=enable_nvtx):
            with torch.no_grad():
                # Process inputs in parallel using different streams
                with torch.cuda.stream(self.stream1):
                    output1 = self.model(self.input1)
                
                with torch.cuda.stream(self.stream2):
                    output2 = self.model(self.input2)
                
                # Synchronize both streams
                self.stream1.synchronize()
                self.stream2.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input1 = None
        self.input2 = None
        self.stream1 = None
        self.stream2 = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedStreamsBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)

