"""baseline_moe.py - Baseline dense model without MoE.

Demonstrates standard dense model without Mixture of Experts (MoE).
MoE: This baseline uses a dense model without expert routing.
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

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class BaselineMoEBenchmark(Benchmark):
    """Baseline: Dense model without MoE (all computation on single path)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input_data = None
    
    def setup(self) -> None:
        """Setup: Initialize dense model."""
        torch.manual_seed(42)
        # Baseline: Dense model without MoE
        # MoE uses expert routing to activate only a subset of experts per token
        # This baseline uses a dense model that processes all tokens through the same path
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device).eval()
        
        self.input_data = torch.randn(32, 256, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Dense model inference without MoE."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        # Baseline: Dense model processes all tokens through same path
        # No MoE - no expert routing or selective activation
        with nvtx_range("baseline_moe", enable=enable_nvtx):
            with torch.no_grad():
                output = self.model(self.input_data)
        torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Cleanup: Clear CUDA cache."""
        if torch.cuda.is_available():
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
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineMoEBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)

