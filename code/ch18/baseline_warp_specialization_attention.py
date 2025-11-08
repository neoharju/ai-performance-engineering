"""baseline_warp_specialization_attention.py - Baseline without warp specialization in FlexAttention/KV cache context.

Demonstrates operations without warp specialization.
Warp specialization: This baseline does not use warp specialization.
All warps perform the same work.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import math
import torch
import torch.nn as nn

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class BaselineWarpSpecializationAttentionBenchmark(Benchmark):
    """Baseline attention-style workload without warp specialization."""
    
    def __init__(self):
        self.device = resolve_device()
        self.q_proj: Optional[nn.Linear] = None
        self.k_proj: Optional[nn.Linear] = None
        self.v_proj: Optional[nn.Linear] = None
        self.out_proj: Optional[nn.Linear] = None
        self.input: Optional[torch.Tensor] = None
        self.scale = 1.0 / math.sqrt(64.0)
    
    def setup(self) -> None:
        """Setup: Initialize attention projections without warp specialization."""
        torch.manual_seed(42)
        hidden_dim = 256
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).eval()
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).eval()
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).eval()
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).eval()
        
        batch = 4
        seqlen = 128
        self.input = torch.randn(batch, seqlen, hidden_dim, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Pure PyTorch implementation without warp specialization."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("baseline_warp_specialization_attention", enable=enable_nvtx):
            assert self.input is not None
            assert self.q_proj and self.k_proj and self.v_proj and self.out_proj

            with torch.no_grad():
                q = self.q_proj(self.input)
                k = self.k_proj(self.input)
                v = self.v_proj(self.input)
                scores = torch.relu(q * k) * self.scale
                context = scores * v
                output = self.out_proj(context)
                _ = output.sum()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.out_proj = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
            use_subprocess=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.input is None:
            return "Input not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineWarpSpecializationAttentionBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    config = BenchmarkConfig(iterations=50, warmup=5)
    config.use_subprocess = False
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=config
    )
    benchmark = BaselineWarpSpecializationAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Warp Specialization")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
