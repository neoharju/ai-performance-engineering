"""optimized_tensor_parallelism.py - Optimized model with tensor parallelism.

Demonstrates tensor parallelism by sharding model layers across GPUs.
Tensor parallelism: This optimized version uses tensor parallelism to shard model layers across multiple GPUs for parallel processing.
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
        raise RuntimeError("CUDA required for ch4")
    return torch.device("cuda")


class OptimizedTensorParallelismBenchmark(Benchmark):
    """Optimized: Tensor parallelism for model inference (sharded across GPUs)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model_layers = None
        self.input_data = None
        self.num_gpus = min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    
    def setup(self) -> None:
        """Setup: Initialize model with tensor parallelism sharding."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        # Optimization: Tensor parallelism shards model layers across GPUs
        # Each GPU holds a portion of each layer's weights
        # This enables parallel computation across GPUs for large models
        self.model_layers = []
        
        # Layer 1: Split input dimension across GPUs (column-wise parallelism)
        for gpu_id in range(self.num_gpus):
            layer1 = nn.Linear(512 // self.num_gpus, 1024).to(torch.device(f"cuda:{gpu_id}"))
            self.model_layers.append([layer1])
        
        # Layer 2: Split hidden dimension across GPUs
        for gpu_id in range(self.num_gpus):
            layer2 = nn.Linear(1024, 1024 // self.num_gpus).to(torch.device(f"cuda:{gpu_id}"))
            self.model_layers[gpu_id].append(layer2)
        
        # Layer 3: Split output dimension across GPUs (row-wise parallelism)
        for gpu_id in range(self.num_gpus):
            layer3 = nn.Linear(1024 // self.num_gpus, 512).to(torch.device(f"cuda:{gpu_id}"))
            self.model_layers[gpu_id].append(layer3)
        
        # Split input across GPUs for tensor parallelism
        self.input_data = torch.randn(32, 512, device=self.device)
        self.input_chunks = []
        chunk_size = 512 // self.num_gpus
        for gpu_id in range(self.num_gpus):
            start_idx = gpu_id * chunk_size
            end_idx = start_idx + chunk_size if gpu_id < self.num_gpus - 1 else 512
            chunk = self.input_data[:, start_idx:end_idx].to(torch.device(f"cuda:{gpu_id}"))
            self.input_chunks.append(chunk)
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Tensor parallelism model inference."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        # Optimization: Process model with tensor parallelism
        # Each GPU processes its shard of the model in parallel
        # Communication happens between layers (all-reduce for column-wise, all-gather for row-wise)
        with nvtx_range("optimized_tensor_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                outputs = []
                for gpu_id in range(self.num_gpus):
                    x = self.input_chunks[gpu_id]
                    gpu_device = torch.device(f"cuda:{gpu_id}")
                    
                    # Process through layers on this GPU
                    for layer in self.model_layers[gpu_id]:
                        x = layer(x)
                        if isinstance(layer, nn.Linear):
                            x = torch.relu(x)
                    
                    outputs.append(x)
                
                # Aggregate outputs (simplified - real implementation would use all-reduce/all-gather)
                # For row-wise parallelism, we concatenate outputs
                output = torch.cat([out.to(self.device) for out in outputs], dim=-1)
        
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
        if self.model_layers is None or len(self.model_layers) == 0:
            return "Model layers not initialized"
        if self.input_chunks is None or len(self.input_chunks) == 0:
            return "Input chunks not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedTensorParallelismBenchmark()


def main():
    """Run optimized tensor parallelism benchmark."""
    benchmark = get_benchmark()
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
    print(f"GPUs used: {benchmark.num_gpus}")
    print("Processing: Parallel across GPUs (tensor parallelism)")


if __name__ == "__main__":
    main()

