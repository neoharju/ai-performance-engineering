"""baseline_tensor_parallelism.py - Baseline model without tensor parallelism.

Demonstrates model inference without tensor parallelism.
Tensor parallelism: This baseline runs the entire model on a single GPU without sharding.
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
        raise RuntimeError("CUDA required for ch4")
    return torch.device("cuda")


class BaselineTensorParallelismBenchmark(Benchmark):
    """Baseline: Model inference without tensor parallelism (single GPU)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input_data = None
    
    def setup(self) -> None:
        """Setup: Initialize model on single GPU."""
        torch.manual_seed(42)
        # Baseline: Entire model on single GPU
        # Tensor parallelism shards model layers across multiple GPUs
        # This baseline runs the full model sequentially on one GPU
        self.model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        ).to(self.device).eval()
        
        self.input_data = torch.randn(32, 512, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential model inference without tensor parallelism."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        # Baseline: Process entire model on single GPU
        # No tensor parallelism - all layers on one device
        with nvtx_range("baseline_tensor_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                output = self.model(self.input_data)
        torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Cleanup: Clear CUDA cache."""
        if torch.cuda.is_available():
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
        if self.input_data is None:
            return "Input data not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineTensorParallelismBenchmark()


def main():
    """Run baseline tensor parallelism benchmark."""
    benchmark = get_benchmark()
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
    print("Processing: Sequential on single GPU (no tensor parallelism)")


if __name__ == "__main__":
    main()

