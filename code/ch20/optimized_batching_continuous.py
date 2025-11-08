"""optimized_batching_continuous.py - Continuous batching optimization (optimized).

Continuous batching processes requests as they arrive, without waiting for full batch.
Efficiently handles variable-length sequences with better GPU utilization.

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
from collections import defaultdict
from math import ceil

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
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")

class SimpleModel(nn.Module):
    """Simple model for batching demonstration."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class OptimizedBatchingContinuousBenchmark(Benchmark):
    """Continuous batching optimization - processes requests immediately."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization

        # Optimization: Compile model for kernel fusion and optimization

        self.request_queue = None
        self.hidden_dim = 1024
        self.batch_size = 64
        self.bucket_size = 32
        self.micro_batch = 4
    
    def setup(self) -> None:
        """Setup: Initialize model and prepare requests."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().eval()
        self.model = torch.compile(self.model, mode="reduce-overhead")
        
        # Simulate incoming requests with different sequence lengths
        # Continuous batching: process requests as they arrive, group by length buckets
        self.request_queue = []
        for i in range(self.batch_size):
            base = 64 + (i % 16) * 8
            burst = (i // 16) * 32
            seq_len = base + burst  # Span from 64 tokens up to ~280 tokens
            x = torch.randn(1, seq_len, self.hidden_dim, device=self.device, dtype=torch.float16)
            self.request_queue.append(x)
        
        # Warmup with a couple of micro-batches to trigger compilation
        with torch.no_grad():
            self._process_queue(self.request_queue[: self.micro_batch * 2])
        torch.cuda.synchronize()
    
    def _bucket_for_length(self, seq_len: int) -> int:
        return int(ceil(seq_len / self.bucket_size) * self.bucket_size)
    
    def _flush_bucket(self, bucket: int, requests: list[torch.Tensor]) -> None:
        if not requests:
            return
        padded = []
        for req in requests:
            cur_len = req.size(1)
            if cur_len == bucket:
                padded.append(req)
                continue
            pad_len = bucket - cur_len
            padding = torch.zeros(req.size(0), pad_len, self.hidden_dim, device=req.device, dtype=req.dtype)
            padded.append(torch.cat([req, padding], dim=1))
        batch = torch.cat(padded, dim=0)
        _ = self.model(batch)
    
    def _process_queue(self, queue: list[torch.Tensor]) -> None:
        buffers: dict[int, list[torch.Tensor]] = defaultdict(list)
        for req in queue:
            bucket = self._bucket_for_length(req.size(1))
            buffers[bucket].append(req)
            if len(buffers[bucket]) >= self.micro_batch:
                self._flush_bucket(bucket, buffers[bucket])
                buffers[bucket] = []
        for bucket, pending in buffers.items():
            if pending:
                self._flush_bucket(bucket, pending)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - continuous batching processes immediately."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_batching_continuous", enable=enable_nvtx):
            # Continuous batching: track buckets and execute micro-batches immediately
            with torch.no_grad():
                self._process_queue(self.request_queue)

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.request_queue
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
        if self.model is None:
            return "Model not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedBatchingContinuousBenchmark()

if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Continuous Batching: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
