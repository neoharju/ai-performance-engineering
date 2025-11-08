"""optimized_cuda_graphs.py - Optimized CUDA graphs for reduced launch overhead.

Demonstrates CUDA graphs for capturing and replaying sequences of operations.
CUDA graphs: Captures and replays operation sequences to reduce launch overhead.
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
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")


class OptimizedCudaGraphsBenchmark(Benchmark):
    """Optimized: CUDA graphs for reduced launch overhead."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input_data = None
        self.graph = None
        self.static_input = None
        self.batch_size = 8
    
    def setup(self) -> None:
        """Setup: Initialize model and capture CUDA graph."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        # Optimization: CUDA graphs capture and replay sequences of operations
        # This reduces launch overhead by eliminating repeated kernel launch setup
        # CUDA graphs are especially beneficial for inference workloads with fixed shapes
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device).eval()
        
        # Create static input for graph capture
        self.static_input = torch.randn(self.batch_size, 256, device=self.device)
        
        # Capture CUDA graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.no_grad():
                _ = self.model(self.static_input)
        
        # Create input for actual execution
        self.input_data = torch.randn(self.batch_size, 256, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: CUDA graph replay."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        # Optimization: Replay CUDA graph instead of launching kernels separately
        # CUDA graphs reduce launch overhead by replaying captured sequences
        with nvtx_range("optimized_cuda_graphs", enable=enable_nvtx):
            # Copy input to static input buffer
            self.static_input.copy_(self.input_data)
            # Replay graph
            self.graph.replay()
        torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input_data = None
        self.static_input = None
        self.graph = None
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
        if self.graph is None:
            return "CUDA graph not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedCudaGraphsBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)

