"""optimized_warp_specialization.py - Warp specialization with CUDA and Triton kernels.

Demonstrates warp specialization using custom CUDA kernels (based on ch10 examples)
and Triton kernels. Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import arch_config to apply Triton SM architecture patch
import arch_config  # noqa: F401

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# Try to load Triton warp specialization
if TRITON_AVAILABLE:
    try:
        from ch10.warp_specialized_triton import warp_specialized_triton_forward_ch10
        TRITON_WARP_SPEC_AVAILABLE = True
    except ImportError:
        TRITON_WARP_SPEC_AVAILABLE = False
else:
    TRITON_WARP_SPEC_AVAILABLE = False

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
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")
class OptimizedWarpSpecializationPipelineBenchmark(Benchmark):
    """
    Optimized: Warp specialization with intra-kernel pipelining.
    
    Demonstrates warp specialization for intra-kernel latency hiding.
    Based on Chapter 10's warp_specialized_pipeline_enhanced.cu:
    - 8 warps per block: 1 producer, 6 compute, 1 consumer
    - Double-buffered pipeline with CUDA Pipeline API
    - Producer/consumer overlap within single kernel
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
        self.use_triton_warp_spec = TRITON_WARP_SPEC_AVAILABLE
        self.use_triton = TRITON_AVAILABLE
    
    def setup(self) -> None:
        """Setup: Initialize model with warp specialization."""
        torch.manual_seed(42)
        
        # Model for post-processing (if using CUDA/Triton kernels)
        self.model = nn.Sequential(
            nn.Linear(2048, 2048),
        ).to(self.device).eval()
        
        # Larger workload to demonstrate warp specialization benefits
        self.input = torch.randn(512, 2048, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Warp specialization with CUDA or Triton."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        # FAIL FAST: No fallbacks - REAL warp specialization required
        if not self.use_triton_warp_spec:
            raise RuntimeError(
                "REAL warp specialization requires Triton kernels! "
                f"Triton available: {self.use_triton_warp_spec}. "
                "Build Triton kernels for Chapter 10."
            )
        
        with nvtx_range("optimized_warp_specialization_pipeline", enable=enable_nvtx):
            with torch.no_grad():
                # REAL warp specialization: Use Triton kernel with warp_specialize=True
                # Based on Chapter 10's warp specialization pattern
                # This demonstrates intra-kernel pipelining with warp specialization
                input_flat = self.input.flatten()
                intermediate_flat = warp_specialized_triton_forward_ch10(input_flat)
                intermediate = intermediate_flat.view_as(self.input)
                output = self.model(intermediate)
                _ = output.sum()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        if not self.use_triton_warp_spec:
            return "Triton warp specialization kernels not available"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedWarpSpecializationPipelineBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    config = benchmark.get_config()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Warp Specialization (Pipeline): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
