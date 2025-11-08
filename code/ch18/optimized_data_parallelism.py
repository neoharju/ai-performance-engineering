"""optimized_data_parallelism.py - Optimized data parallelism for multi-GPU inference.

Demonstrates data parallelism by replicating model across multiple GPUs.
Data parallelism: Replicates model across GPUs for parallel processing of different batches.
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

try:
    import ch18.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

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


class OptimizedDataParallelismBenchmark(Benchmark):
    """Optimized: Data parallelism with model replication across GPUs."""
    
    def __init__(self):
        self.device = resolve_device()
        self.models = None
        self.requests = None
        self.num_requests = 10
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    def setup(self) -> None:
        """Setup: Initialize model replicas on multiple GPUs."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        # Optimization: Data parallelism replicates model across multiple GPUs
        # Each GPU processes different requests in parallel
        # This enables higher throughput by processing multiple requests simultaneously
        self.models = []
        for gpu_id in range(self.num_gpus):
            model = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
            ).to(torch.device(f"cuda:{gpu_id}")).eval()
            self.models.append(model)
        
        # Generate multiple inference requests
        # Distribute requests across GPUs for parallel processing
        self.requests = []
        for i in range(self.num_requests):
            gpu_id = i % self.num_gpus
            request = torch.randn(1, 256, device=torch.device(f"cuda:{gpu_id}"))
            self.requests.append((gpu_id, request))
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Data parallelism processing across multiple GPUs."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        # Optimization: Process requests in parallel across GPUs
        # Data parallelism enables parallel processing of different requests
        with nvtx_range("optimized_data_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                # Process requests in parallel across GPUs
                for gpu_id, request in self.requests:
                    model = self.models[gpu_id]
                    _ = model(request)
        
        # Synchronize all GPUs
        for gpu_id in range(self.num_gpus):
            torch.cuda.synchronize(torch.device(f"cuda:{gpu_id}"))
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.models = None
        self.requests = None
        if torch.cuda.is_available():
            for gpu_id in range(self.num_gpus):
                torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.models is None or len(self.models) == 0:
            return "Models not initialized"
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

