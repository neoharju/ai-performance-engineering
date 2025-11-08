"""optimized_pinned_memory.py - Pinned memory transfer (optimized).

Demonstrates faster CPU-GPU memory transfer using pinned memory.
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
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch11")
    return torch.device("cuda")


class OptimizedPinnedMemoryBenchmark(Benchmark):
    """Pinned memory transfer - faster CPU-GPU transfers.
    
    Pinned memory (page-locked memory) allows direct memory access (DMA)
    transfers between CPU and GPU, eliminating the need for CPU staging buffers.
    This significantly speeds up H2D and D2H transfers.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.cpu_data = None
        self.gpu_data = None
        self.N = 10_000_000
    
    def setup(self) -> None:
        """Setup: Initialize CPU tensor with pinned memory."""
        torch.manual_seed(42)
        
        # Optimized: Pinned memory - page-locked CPU memory
        # pin_memory=True enables faster CPU-GPU transfers via DMA
        self.cpu_data = torch.randn(self.N, dtype=torch.float32, pin_memory=True)
        self.gpu_data = None
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Pinned memory transfer."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_pinned_memory_pinned", enable=enable_nvtx):
            # Optimized: Transfer pinned memory to GPU
            # This is faster because it uses DMA (Direct Memory Access)
            # Non-blocking transfer can overlap with computation
            self.gpu_data = self.cpu_data.to(self.device, non_blocking=True)
            torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.cpu_data = None
        self.gpu_data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.gpu_data is None:
            return "GPU tensor not initialized"
        if self.cpu_data is None:
            return "CPU tensor not initialized"
        if self.gpu_data.shape[0] != self.N:
            return f"GPU tensor shape mismatch: expected {self.N}, got {self.gpu_data.shape[0]}"
        if not torch.isfinite(self.gpu_data).all():
            return "GPU tensor contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedPinnedMemoryBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Pinned Memory (Pinned): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

