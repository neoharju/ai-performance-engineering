"""optimized_quantization.py - Optimized FP16 quantization for faster inference.

Demonstrates FP16 quantization for reduced memory and faster inference on GPU.
Quantization: This optimized version uses FP16 quantization to reduce model size and speed up inference.
FP16 quantization is GPU-friendly and enables Tensor Core acceleration.
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


class OptimizedQuantizationBenchmark(Benchmark):
    """Optimized: FP16 quantization for faster inference with reduced memory."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.quantized_model = None
        self.data = None
        # Match baseline batch size so FP16 Tensor Core acceleration pays off
        self.N = 65536
    
    def setup(self) -> None:
        """Setup: Initialize model and apply quantization."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        # Optimization: Quantization - reduce precision from FP32 to FP16
        # FP16 quantization reduces memory usage by 2x and speeds up inference on Tensor Cores
        # This is a practical GPU-friendly quantization approach (INT8 quantization requires CPU)
        # FP16 quantization is commonly used in production deployments
        # Match baseline architecture exactly - only difference is quantization
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        ).to(self.device).to(torch.float32)
        
        # Prepare model for quantization
        self.model.eval()
        
        # Apply FP16 quantization: FP32 -> FP16
        # FP16 quantization is GPU-friendly and enables Tensor Core acceleration
        # This is equivalent to post-training quantization (PTQ) but uses FP16 instead of INT8
        # Both weights and activations are quantized to FP16 for consistent dtype
        self.quantized_model = self.model.to(torch.float16)
        
        # Use FP16 data to match quantized model dtype
        # This ensures activations and weights share the same dtype/device
        self.data = torch.randn(self.N, 256, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Quantized FP16 inference."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_quantization", enable=enable_nvtx):
            # Optimization: Quantized inference - FP16 precision
            # FP16 quantization reduces memory bandwidth by 2x and enables Tensor Core acceleration
            # Typically 1.5-2x faster than FP32 with minimal accuracy loss
            # FP16 quantization is GPU-friendly and commonly used in production
            # Both weights and activations are FP16 for consistent dtype/device
            with torch.no_grad():
                _ = self.quantized_model(self.data)
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.quantized_model = None
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
        if self.quantized_model is None:
            return "Quantized model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedQuantizationBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
