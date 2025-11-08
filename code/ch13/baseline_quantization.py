"""baseline_quantization.py - Baseline FP32 precision without quantization.

Demonstrates full precision FP32 inference without quantization.
Quantization: This baseline uses FP32 precision without any quantization.
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


class BaselineQuantizationBenchmark(Benchmark):
    """Baseline: FP32 precision without quantization (full precision)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.data = None
        # Larger batch highlights quantization savings (Tensor Cores saturate better)
        self.N = 65536
    
    def setup(self) -> None:
        """Setup: Initialize model in FP32."""
        torch.manual_seed(42)
        # Baseline: Full FP32 precision - no quantization
        # Higher memory usage and slower computation
        # Quantization reduces precision to reduce memory and speed up inference
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        ).to(self.device).to(torch.float32)  # Explicit FP32
        
        self.model.eval()
        self.data = torch.randn(self.N, 256, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: FP32 inference without quantization."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("baseline_quantization", enable=enable_nvtx):
            # Baseline: Full precision FP32 inference
            # No quantization - uses full 32-bit floating point precision
            # Quantization would reduce precision to INT8 or INT4 for faster inference
            with torch.no_grad():
                _ = self.model(self.data)
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineQuantizationBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
