"""baseline_warp_specialization.py - Baseline without warp specialization in streams context.

Demonstrates sequential stream processing without warp specialization.
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
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch11")
    return torch.device("cuda")


class BaselineWarpSpecializationStreamsBenchmark(Benchmark):
    """Baseline: Sequential stream processing without warp specialization."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
        self.stream = None
    
    def setup(self) -> None:
        """Setup: Initialize model and stream without warp specialization."""
        torch.manual_seed(42)
        
        # Baseline: Sequential model - no warp specialization
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        self.input = torch.randn(32, 1024, device=self.device)
        self.stream = torch.cuda.Stream()
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential stream processing."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("baseline_warp_specialization_multistream", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Sequential processing in single stream
                # No warp specialization, no inter-kernel overlap
                with torch.cuda.stream(self.stream):
                    output = self.model(self.input)
                    _ = output.sum()
                self.stream.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        self.stream = None
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
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineWarpSpecializationStreamsBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    config = benchmark.get_config()
    config.use_subprocess = False
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Warp Specialization (Multi-Stream): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

