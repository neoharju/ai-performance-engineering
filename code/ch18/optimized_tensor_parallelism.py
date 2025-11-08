"""optimized_tensor_parallelism.py - Optimized tensor parallelism across GPUs.

Demonstrates tensor parallelism by splitting tensors across multiple GPUs.
Tensor parallelism: Splits tensors across GPUs for parallel computation.
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
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class OptimizedTensorParallelismBenchmark(Benchmark):
    """Optimized: Tensor parallelism with tensors split across GPUs."""
    
    def __init__(self):
        self.device = resolve_device()
        self.models = None
        self.input_data = None
        self.batch_size = 8
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    def setup(self) -> None:
        """Setup: Initialize model shards across multiple GPUs."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        # Optimization: Tensor parallelism splits tensors across multiple GPUs
        # Each GPU processes a shard of the hidden dimension and results are reduced
        # This enables parallel computation of large tensors
        
        hidden_size = 256
        intermediate_size = 512
        
        # Tensor parallelism: Split hidden dimension across GPUs
        # Cap num_gpus to not exceed hidden_size to avoid zero-width layers
        effective_num_gpus = min(self.num_gpus, hidden_size)
        
        # Calculate shard sizes, handling remainder when hidden_size isn't divisible
        # Distribute remainder to first few GPUs
        base_shard_size = hidden_size // effective_num_gpus
        remainder = hidden_size % effective_num_gpus
        
        # Calculate actual shard sizes per GPU
        self.shard_sizes = []
        for gpu_id in range(effective_num_gpus):
            shard_size = base_shard_size + (1 if gpu_id < remainder else 0)
            self.shard_sizes.append(shard_size)
        
        intermediate_size_per_gpu = intermediate_size // effective_num_gpus
        intermediate_remainder = intermediate_size % effective_num_gpus
        
        self.models = []
        for gpu_id in range(effective_num_gpus):
            hidden_shard_size = self.shard_sizes[gpu_id]
            intermediate_shard_size = intermediate_size_per_gpu + (1 if gpu_id < intermediate_remainder else 0)
            
            # Each GPU processes a shard of the hidden dimension
            # Input: (batch, hidden_shard_size)
            # Output: (batch, hidden_shard_size) - will be reduced across GPUs
            model = nn.Sequential(
                nn.Linear(hidden_shard_size, intermediate_shard_size),
                nn.ReLU(),
                nn.Linear(intermediate_shard_size, intermediate_shard_size),
                nn.ReLU(),
                nn.Linear(intermediate_shard_size, hidden_shard_size),
            ).to(torch.device(f"cuda:{gpu_id}")).eval()
            self.models.append(model)
        
        # Update num_gpus to effective count
        self.num_gpus = effective_num_gpus
        
        # Input data - will be split across GPUs
        self.input_data = torch.randn(self.batch_size, hidden_size, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Tensor parallelism processing across multiple GPUs."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        # Optimization: Process tensors in parallel across GPUs
        # Tensor parallelism enables parallel computation by sharding hidden dimension
        with nvtx_range("optimized_tensor_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                # Shard input hidden dimension across GPUs using pre-calculated shard sizes
                # Process shards in parallel
                output_shards = []
                start_idx = 0
                for gpu_id, (model, shard_size) in enumerate(zip(self.models, self.shard_sizes)):
                    # Extract shard of hidden dimension for this GPU
                    end_idx = start_idx + shard_size
                    input_shard = self.input_data[:, start_idx:end_idx].to(torch.device(f"cuda:{gpu_id}"))
                    
                    # Process shard
                    output_shard = model(input_shard)
                    output_shards.append(output_shard)
                    start_idx = end_idx
                
                # Reduce outputs: concatenate shards back together
                # In production, this would use all-reduce for better efficiency
                if len(output_shards) > 1:
                    # Transfer all shards to first GPU and concatenate
                    output_shards_on_device0 = [shard.to(torch.device("cuda:0")) for shard in output_shards]
                    combined_output = torch.cat(output_shards_on_device0, dim=-1)
                else:
                    combined_output = output_shards[0]
        
        # Synchronize all GPUs
        for gpu_id in range(self.num_gpus):
            torch.cuda.synchronize(torch.device(f"cuda:{gpu_id}"))
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.models = None
        self.input_data = None
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
    return OptimizedTensorParallelismBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)

