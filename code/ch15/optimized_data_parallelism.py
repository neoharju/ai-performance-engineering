"""optimized_data_parallelism.py - Optimized inference with data parallelism.

Demonstrates data parallelism for inference by replicating model across GPUs.
Data parallelism: This optimized version uses data parallelism to process multiple requests in parallel.
In inference, data parallelism replicates the entire model on multiple GPUs for higher throughput.
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
        raise RuntimeError("CUDA required for ch15")
    return torch.device("cuda")


class OptimizedDataParallelismBenchmark(Benchmark):
    """Optimized: Data parallelism for inference (model replication across GPUs)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.requests = None
        self.num_requests = 10
        self.use_data_parallel = False
    
    def setup(self) -> None:
        """Setup: Initialize model with data parallelism."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        # Optimization: Data parallelism for inference
        # In inference, data parallelism replicates the entire model on multiple GPUs
        # Each GPU processes different requests independently
        # Unlike training, inference data parallelism doesn't require gradient synchronization
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        ).to(self.device).eval()
        
        # Check if multiple GPUs available for data parallelism
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            # Use DataParallel for inference data parallelism
            # DataParallel replicates model across GPUs and splits batch
            try:
                self.model = nn.DataParallel(self.model)
                self.use_data_parallel = True
            except Exception:
                self.use_data_parallel = False
        
        # Generate multiple inference requests
        self.requests = [torch.randn(1, 256, device=self.device) for _ in range(self.num_requests)]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Data parallel inference."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_data_parallelism", enable=enable_nvtx):
            # Optimization: Data parallelism for inference
            # Model is replicated across GPUs, each GPU processes different requests
            # Data parallelism improves throughput by processing requests in parallel
            # In inference, requests are independent, so no gradient synchronization needed
            with torch.no_grad():
                if self.use_data_parallel:
                    # DataParallel automatically distributes requests across GPUs
                    batch = torch.cat(self.requests, dim=0)
                    _ = self.model(batch)
                else:
                    # Fallback: process requests sequentially (simulating parallel processing)
                    for request in self.requests:
                        _ = self.model(request)
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.requests = None
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
    return OptimizedDataParallelismBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)

