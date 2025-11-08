"""optimized_pipeline_parallelism.py - Optimized pipeline parallelism across GPUs.

Demonstrates pipeline parallelism by splitting model layers across multiple GPUs.
Pipeline parallelism: Splits model layers across GPUs for parallel processing of microbatches.
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

from typing import Optional, List

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


class OptimizedPipelineParallelismBenchmark(Benchmark):
    """Optimized: Pipeline parallelism with layers split across GPUs."""
    
    def __init__(self):
        self.device = resolve_device()
        self.pipeline_stages = None
        self.input_data = None
        self.batch_size = 8
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    def setup(self) -> None:
        """Setup: Initialize pipeline stages across multiple GPUs."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        # Optimization: Pipeline parallelism splits model layers across GPUs
        # Each GPU processes different layers (pipeline stages)
        # Microbatches flow through the pipeline stages for parallel processing
        # This enables parallel processing of multiple microbatches across stages
        
        # Define layers for each pipeline stage
        layers_per_stage = [
            [nn.Linear(256, 512), nn.ReLU()],
            [nn.Linear(512, 512), nn.ReLU()],
            [nn.Linear(512, 512), nn.ReLU()],
            [nn.Linear(512, 256)],
        ]
        
        # Always create all stages, assigning multiple stages to same GPU if needed
        # This ensures the model architecture matches the baseline
        self.pipeline_stages: List[nn.Module] = []
        num_stages = len(layers_per_stage)
        if self.num_gpus > num_stages:
            self.num_gpus = num_stages
        
        for stage_id in range(num_stages):
            # Distribute stages across available GPUs (round-robin)
            # If num_gpus < num_stages, multiple stages will be on same GPU
            gpu_id = stage_id % self.num_gpus if self.num_gpus else 0
            stage = nn.Sequential(*layers_per_stage[stage_id]).to(torch.device(f"cuda:{gpu_id}")).eval()
            self.pipeline_stages.append(stage)
        
        # Input data for first pipeline stage
        self.input_data = torch.randn(self.batch_size, 256, device=torch.device("cuda:0"))
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Pipeline parallelism processing across multiple GPUs."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        # Optimization: Model partitioning across pipeline stages
        # IMPORTANT: This demonstrates model partitioning, not true pipeline parallelism.
        # True pipeline parallelism requires:
        # - Multiple microbatches in flight simultaneously (not implemented here)
        # - Asynchronous transfers between stages (uses blocking .to() transfers)
        # - Overlapping computation and communication (stages run sequentially)
        # This implementation partitions the model correctly but executes sequentially,
        # which is model parallelism, not pipeline parallelism.
        with nvtx_range("optimized_pipeline_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                # Process through pipeline stages sequentially (model partitioning only)
                # This does NOT demonstrate true pipelining with overlapping execution
                x = self.input_data
                for stage_idx, stage in enumerate(self.pipeline_stages):
                    x = stage(x)
                    # Transfer to next GPU if needed (blocking transfer - simplified)
                    if stage_idx < len(self.pipeline_stages) - 1:
                        next_stage_idx = stage_idx + 1
                        next_gpu_id = next_stage_idx % self.num_gpus
                        current_gpu_id = stage_idx % self.num_gpus
                        if next_gpu_id != current_gpu_id:
                            x = x.to(torch.device(f"cuda:{next_gpu_id}"))
        
        # Synchronize all GPUs
        for gpu_id in range(self.num_gpus):
            torch.cuda.synchronize(torch.device(f"cuda:{gpu_id}"))
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.pipeline_stages = None
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
        if self.pipeline_stages is None or len(self.pipeline_stages) == 0:
            return "Pipeline stages not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedPipelineParallelismBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
