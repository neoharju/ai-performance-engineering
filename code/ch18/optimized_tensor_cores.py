"""optimized_tensor_cores.py - Optimized tensor core acceleration.

Demonstrates tensor core acceleration using FP16/BF16.
Tensor cores: Uses tensor cores for accelerated matrix operations.
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

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class OptimizedTensorCoresBenchmark(Benchmark):
    """Optimized: Tensor core accelerated matrix operations."""
    
    def __init__(self):
        self.device = resolve_device()
        self.A = None
        self.B = None
        self.size = 4096
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    def setup(self) -> None:
        """Setup: Initialize matrices in FP16/BF16 for tensor cores."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        # Optimization: Tensor cores accelerate FP16/BF16 matrix operations
        # Tensor cores provide high throughput for mixed-precision operations
        # This uses FP16/BF16 to leverage tensor core acceleration
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=self.dtype)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=self.dtype)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Tensor core accelerated matrix multiplication."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        # Optimization: FP16/BF16 matmul with tensor cores
        # Tensor cores provide high throughput for these operations
        with nvtx_range("optimized_tensor_cores", enable=enable_nvtx):
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
    return OptimizedTensorCoresBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)

