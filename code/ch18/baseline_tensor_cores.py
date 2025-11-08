"""baseline_tensor_cores.py - Baseline without tensor core acceleration.

Demonstrates matrix operations without tensor core acceleration.
Tensor cores: This baseline uses standard FP32 operations without tensor cores.
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
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class BaselineTensorCoresBenchmark(Benchmark):
    """Baseline: FP32 matrix operations without tensor cores."""
    
    def __init__(self):
        self.device = resolve_device()
        self.A = None
        self.B = None
        self.size = 4096
    
    def setup(self) -> None:
        """Setup: Initialize matrices in FP32."""
        torch.manual_seed(42)
        # Baseline: FP32 operations without tensor cores
        # Tensor cores accelerate FP16/BF16 matrix operations
        # This baseline uses FP32 which doesn't use tensor cores
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: FP32 matrix multiplication without tensor cores."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        # Baseline: FP32 matmul without tensor cores
        # Tensor cores accelerate FP16/BF16 operations
        with nvtx_range("baseline_tensor_cores", enable=enable_nvtx):
            _ = torch.matmul(self.A, self.B)
        torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A = None
        self.B = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineTensorCoresBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)

