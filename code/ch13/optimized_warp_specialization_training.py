"""optimized_warp_specialization_training.py - Optimized warp specialization in training context.

Demonstrates REAL warp specialization using Triton kernels for training workloads.
Warp specialization: Assigns different roles to warps (producer/consumer).
Specialized warps improve efficiency through optimized execution patterns.
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

try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# Try to load Triton warp specialization
if TRITON_AVAILABLE:
    try:
        from ch13.warp_specialized_triton import warp_specialized_triton_forward_ch13
        TRITON_WARP_SPEC_AVAILABLE = True
    except ImportError:
        try:
            from warp_specialized_triton import warp_specialized_triton_forward_ch13
            TRITON_WARP_SPEC_AVAILABLE = True
        except ImportError:
            TRITON_WARP_SPEC_AVAILABLE = False
else:
    TRITON_WARP_SPEC_AVAILABLE = False

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")

class OptimizedWarpSpecializationTrainingBenchmark(Benchmark):
    """Optimized: REAL warp specialization in training context.
    
    Demonstrates REAL warp specialization using Triton kernels.
    Warp specialization: Assigns different roles to warps (producer/consumer).
    Specialized warps improve efficiency through optimized execution patterns.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
        self.weight = None
    
    def setup(self) -> None:
        """Setup: Initialize model with REAL warp specialization."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.manual_seed(42)
        
        # Model for post-processing (Triton kernel replaces forward computation)
        self.model = nn.Sequential(
            nn.Linear(2048, 2048),
        ).to(self.device).train()
        
        # Larger workload for training context
        self.input = torch.randn(512, 2048, device=self.device)
        self.weight = torch.randn_like(self.input)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: REAL warp specialization with Triton kernels."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_warp_specialization_training", enable=enable_nvtx):
            if not TRITON_WARP_SPEC_AVAILABLE:
                raise RuntimeError(
                    "REAL warp specialization requires Triton kernels! "
                    "Triton available: {}. Build Triton kernels for Chapter 13."
                    .format(TRITON_WARP_SPEC_AVAILABLE)
                )
            
            # REAL warp specialization: Use Triton kernel with warp_specialize=True
            # Based on Chapter 13's training pattern
            input_flat = self.input.flatten()
            weight_flat = self.weight.flatten()
            intermediate_flat = warp_specialized_triton_forward_ch13(input_flat, weight_flat)
            intermediate = intermediate_flat.view_as(self.input)
            
            # Apply model transformation
            output = self.model(intermediate)
            _ = output.sum()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        self.weight = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> Optional[BenchmarkConfig]:
        """Return benchmark configuration."""
        return BenchmarkConfig(use_subprocess=False)

def get_benchmark() -> Benchmark:
    """Return benchmark instance."""
    return OptimizedWarpSpecializationTrainingBenchmark()
