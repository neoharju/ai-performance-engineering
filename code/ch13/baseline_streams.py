"""baseline_streams.py - Baseline without CUDA streams for training.

Demonstrates operations without CUDA streams for parallel execution.
Streams: This baseline does not use CUDA streams.
Operations execute sequentially without overlap.
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

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class BaselineStreamsBenchmark(Benchmark):
    """Baseline: Sequential execution without CUDA streams.
    
    Streams: This baseline does not use CUDA streams.
    Operations execute sequentially without overlap.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input1 = None
        self.input2 = None
    
    def setup(self) -> None:
        """Setup: Initialize model without streams."""
        torch.manual_seed(42)
        # Baseline: No CUDA streams - sequential execution
        # CUDA streams allow parallel execution of independent operations
        # This baseline does not use streams
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        self.input1 = torch.randn(32, 1024, device=self.device)
        self.input2 = torch.randn(32, 1024, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential execution without streams."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        # Baseline: Sequential execution - no CUDA streams
        # Operations execute one after another without overlap
        with nvtx_range("baseline_streams", enable=enable_nvtx):
            with torch.no_grad():
                # Process inputs sequentially
                output1 = self.model(self.input1)
                torch.cuda.synchronize()  # Wait for completion
                output2 = self.model(self.input2)
                torch.cuda.synchronize()  # Wait for completion
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input1 = None
        self.input2 = None
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
    return BaselineStreamsBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)

