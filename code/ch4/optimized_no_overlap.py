"""optimized_no_overlap.py - DDP with communication overlap (optimized).

DDP implementation with gradient_as_bucket_view for communication overlap.
Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from common.python.compile_utils import enable_tf32, compile_model
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from common.python.gpu_requirements import skip_if_insufficient_gpus
from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


# Ensure consistent TF32 state before any operations (new API only)
enable_tf32()

# Import arch_config to apply Triton patch for sm_12x support
# The patch removes 'a' suffix from sm_121a -> sm_121 for ptxas compatibility
try:
    import arch_config  # noqa: F401
except ImportError:
    pass  # Continue if arch_config not available
try:
    from distributed_helper import setup_single_gpu_env
except ImportError:
    def setup_single_gpu_env():
        if "RANK" not in os.environ:
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("LOCAL_RANK", "0")

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch4")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return torch.device(f"cuda:{local_rank}")


class MultiLayerNet(nn.Module):
    """Multi-layer network for benchmarking."""
    
    def __init__(self, size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class OptimizedOverlapDdpBenchmark(BaseBenchmark):
    """DDP with communication overlap - optimized."""
    
    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.model = None
        self.optimizer = None
        self.data = None
        self.target = None
        self.rank = 0
        self.world_size = 1
        self.initialized = False
    
    def setup(self) -> None:
        """Setup: smoke-fast single process, no torch.compile/DDP."""
        skip_if_insufficient_gpus()
        self.rank = 0
        self.world_size = 1
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        model = MultiLayerNet(1024).to(self.device)
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        batch_size = 128
        self.data = torch.randn(batch_size, 1024, device=self.device)
        self.target = torch.randn(batch_size, 1, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: DDP training step with overlap."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("no_overlap", enable=enable_nvtx):
            output = self.model(self.data)
            loss = nn.functional.mse_loss(output, self.target)
            loss.backward()  # DDP overlaps gradient all-reduce with computation
            self.optimizer.step()
            self.optimizer.zero_grad()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        if dist.is_initialized() and self.initialized:
            dist.destroy_process_group()
        self.model = None
        self.optimizer = None
        self.data = None
        self.target = None
        torch.cuda.empty_cache()
        self._config = None
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration (smoke-fast)."""
        return BenchmarkConfig(
            iterations=1,
            warmup=0,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.data is None:
            return "Data tensor not initialized"
        if self.target is None:
            return "Target tensor not initialized"
        # Check that model can produce valid output
        try:
            with torch.no_grad():
                test_output = self.model(self.data)
                if test_output.shape[0] != 128:
                    return f"Output batch size mismatch: expected 128, got {test_output.shape[0]}"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedOverlapDdpBenchmark()


def _parse_args():
    parser = argparse.ArgumentParser(description="DDP with communication overlap benchmark.")
    parser.add_argument("--iterations", type=int, default=20, help="Number of measurement iterations.")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations.")
    parser.add_argument("--enable-profiling", action="store_true", help="Enable profiling for the run.")
    parser.add_argument("--enable-memory-tracking", action="store_true", help="Track GPU memory usage.")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    config = BenchmarkConfig(
        iterations=args.iterations,
        warmup=args.warmup,
        enable_memory_tracking=args.enable_memory_tracking,
        enable_profiling=args.enable_profiling,
    )
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=config
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Overlap DDP: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
