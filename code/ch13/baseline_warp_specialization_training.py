"""Baseline benchmark for Chapter 13 warp specialization training example.

This baseline intentionally avoids warp specialization. All warps execute the
same work (standard PyTorch ops) so we can compare against the optimized Triton
kernel that assigns producer/consumer roles.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import Benchmark, BenchmarkConfig

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class BaselineWarpSpecializationTrainingBenchmark(Benchmark):
    """Baseline training workload without warp specialization."""

    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.weight: Optional[torch.Tensor] = None

    def setup(self) -> None:
        """Allocate tensors and build a simple MLP to mimic training compute."""
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
        ).to(self.device).train()

        batch = 512
        width = 2048
        self.input = torch.randn(batch, width, device=self.device)
        self.weight = torch.randn_like(self.input)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Run the baseline forward pass (no warp specialization)."""
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("baseline_warp_specialization_training", enable=enable_nvtx):
            assert self.input is not None and self.weight is not None
            assert self.model is not None

            with torch.no_grad():
                fused = torch.relu(self.input * self.weight)
                output = self.model(fused)
                _ = output.sum()

    def teardown(self) -> None:
        """Release GPU resources."""
        self.model = None
        self.input = None
        self.weight = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return a standard benchmark configuration."""
        return BenchmarkConfig(iterations=50, warmup=5, use_subprocess=False)

    def validate_result(self) -> Optional[str]:
        """Sanity-check that buffers were initialized."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None or self.weight is None:
            return "Input tensors not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory for harness discovery."""
    return BaselineWarpSpecializationTrainingBenchmark()
