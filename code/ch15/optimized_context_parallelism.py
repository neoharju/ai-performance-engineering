"""optimized_context_parallelism.py - Optimized processing with context parallelism.

Demonstrates context parallelism for long sequences by splitting across GPUs.
Context parallelism: This optimized version uses context parallelism to split long sequences across multiple GPUs for parallel processing.
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


class OptimizedContextParallelismBenchmark(Benchmark):
    """Optimized: Context parallelism for long sequences (split across GPUs)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.models = None
        self.input_sequence = None
        self.sequence_length = 8192  # Long sequence to demonstrate context parallelism benefit
        self.num_gpus = min(4, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    
    def setup(self) -> None:
        """Setup: Initialize models and split sequence for context parallelism."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        # Optimization: Context parallelism splits long sequences across GPUs
        # Each GPU processes a different portion of the sequence in parallel
        # This enables parallel processing of long contexts that don't fit efficiently on one GPU
        self.models = []
        for gpu_id in range(self.num_gpus):
            model = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
            ).to(torch.device(f"cuda:{gpu_id}")).eval()
            self.models.append(model)
        
        # Long sequence that benefits from context parallelism
        # Split sequence across GPUs for parallel processing
        self.input_sequence = torch.randn(self.sequence_length, 256, device=self.device)
        
        # Split sequence into chunks for each GPU
        tokens_per_gpu = self.sequence_length // self.num_gpus
        self.sequence_chunks = []
        for gpu_id in range(self.num_gpus):
            start_idx = gpu_id * tokens_per_gpu
            end_idx = start_idx + tokens_per_gpu if gpu_id < self.num_gpus - 1 else self.sequence_length
            chunk = self.input_sequence[start_idx:end_idx].to(torch.device(f"cuda:{gpu_id}"))
            self.sequence_chunks.append(chunk)
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Context parallelism processing of long sequence."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        # Optimization: Process sequence chunks in parallel across GPUs
        # Context parallelism enables parallel processing of long sequences
        with nvtx_range("optimized_context_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                outputs = []
                for gpu_id, (model, chunk) in enumerate(zip(self.models, self.sequence_chunks)):
                    output = model(chunk)
                    outputs.append(output)
        
        # Synchronize all GPUs
        for gpu_id in range(self.num_gpus):
            torch.cuda.synchronize(torch.device(f"cuda:{gpu_id}"))
    
    def teardown(self) -> None:
        """Cleanup: Clear CUDA cache."""
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
        if self.sequence_chunks is None or len(self.sequence_chunks) == 0:
            return "Sequence chunks not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedContextParallelismBenchmark()


def main():
    """Run optimized context parallelism benchmark."""
    benchmark = get_benchmark()
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
    print(f"Sequence length: {benchmark.sequence_length} tokens")
    print(f"GPUs used: {benchmark.num_gpus}")
    print("Processing: Parallel across GPUs (context parallelism)")


if __name__ == "__main__":
    main()

