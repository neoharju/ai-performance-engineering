"""Optimized batched GEMM - Single fused kernel for all batches.

This optimized version uses torch.bmm which launches a single efficient
kernel for all batched operations, avoiding multiple kernel launch overhead.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from typing import Optional

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


def matmul_batched_fused(a_batch: torch.Tensor, b_batch: torch.Tensor) -> torch.Tensor:
    """Single fused kernel for batched GEMM.
    
    torch.bmm uses a single efficient kernel for all batch elements,
    avoiding the overhead of multiple kernel launches.
    """
    return torch.bmm(a_batch, b_batch)


class OptimizedTritonPersistentBenchmark(BaseBenchmark):
    """Optimized: Single fused kernel for batched operations.
    
    Uses torch.bmm which efficiently handles all batch elements
    in a single kernel launch, avoiding launch overhead.
    """

    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None
        self.batch_size = 32  # Many small operations
        self.M = 256
        self.N = 256
        self.K = 256
        self.dtype = torch.float16
        self._last = 0.0
        
        flops = 2 * self.batch_size * self.M * self.N * self.K
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.M * self.N),
        )
        self.output = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.M * self.N),
        )

    def setup(self) -> None:
        """Setup: Initialize batched matrices."""
        torch.manual_seed(42)
        
        self.a = torch.randn(self.batch_size, self.M, self.K, device=self.device, dtype=self.dtype)
        self.b = torch.randn(self.batch_size, self.K, self.N, device=self.device, dtype=self.dtype)
        
        # Warmup
        for _ in range(3):
            _ = matmul_batched_fused(self.a, self.b)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: Single fused kernel."""
        self.output = matmul_batched_fused(self.a, self.b)
        self._last = float(self.output.sum())
        self._synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.a = None
        self.b = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_triton_metrics
        return compute_triton_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            block_size=getattr(self, 'BLOCK_SIZE', 1024),
            num_warps=getattr(self, 'num_warps', 4),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.a is None or self.b is None:
            return "Matrices not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("Output not available - run benchmark first")
        return self.output

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return {"batch_size": self.batch_size, "M": self.M, "N": self.N, "K": self.K}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedTritonPersistentBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
