"""optimized_warp_specialization.py - Warp specialization with streams (Chapter 11).

Demonstrates warp specialization combined with CUDA streams for inter-kernel overlap.
Based on Chapter 11's warp_specialized_kernel_two_pipelines_multistream.cu.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import arch_config to apply Triton SM architecture patch (fixes sm_121a issue)
try:
    import arch_config  # noqa: F401
except ImportError:
    pass

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
        from ch11.warp_specialized_triton import warp_specialized_triton_forward_ch11
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
        raise RuntimeError("CUDA required for ch11")
    return torch.device("cuda")


class OptimizedWarpSpecializationStreamsBenchmark(Benchmark):
    """
    Optimized: Warp specialization with CUDA streams for inter-kernel overlap.
    
    Demonstrates two-level pipeline combining warp specialization with streams.
    Based on Chapter 11's warp_specialized_kernel_two_pipelines_multistream.cu:
    - Intra-kernel overlap: loader ↔ compute ↔ storer warps (warp specialization)
    - Inter-kernel overlap: Multiple streams process different batches concurrently
    - Two pipelines: loader->compute and compute->storer
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
        self.streams = None
        self.use_triton = TRITON_AVAILABLE
        self.use_triton_warp_spec = TRITON_WARP_SPEC_AVAILABLE
        self.num_streams = 3  # Multiple streams for inter-kernel overlap
    
    def setup(self) -> None:
        """Setup: Initialize model and streams with warp specialization."""
        torch.manual_seed(42)
        
        # Model for post-processing
        self.model = nn.Sequential(
            nn.Linear(1024, 1024),
        ).to(self.device).eval()
        
        self.input = torch.randn(32, 1024, device=self.device)
        
        # Create multiple streams for inter-kernel overlap
        # This demonstrates the two-level pipeline from Chapter 11:
        # - Intra-kernel: warp specialization within each kernel
        # - Inter-kernel: streams overlap different batches
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Warp specialization with multiple streams."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        # FAIL FAST: No fallbacks - REAL warp specialization required
        if not self.use_triton_warp_spec:
            raise RuntimeError(
                "REAL warp specialization requires Triton kernels! "
                f"Triton available: {self.use_triton_warp_spec}. "
                "Build Triton kernels for Chapter 11."
            )
        
        with nvtx_range("optimized_warp_specialization_multistream", enable=enable_nvtx):
            with torch.no_grad():
                # REAL warp specialization with multiple streams
                # Each stream processes the input with warp-specialized kernel
                # This demonstrates inter-kernel overlap (streams) + intra-kernel overlap (warp specialization)
                
                inputs_per_stream = torch.chunk(self.input, self.num_streams, dim=0)
                partial_sums = []
                for chunk, stream in zip(inputs_per_stream, self.streams):
                    with torch.cuda.stream(stream):
                        flat = chunk.reshape(-1).contiguous()
                        intermediate_flat = warp_specialized_triton_forward_ch11(flat)
                        intermediate = intermediate_flat.view_as(chunk)
                        out = self.model(intermediate)
                        partial_sums.append(out.sum())
                
                for stream in self.streams:
                    stream.synchronize()
                _ = torch.stack(partial_sums).sum()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        self.streams = None
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
        if self.streams is None:
            return "Streams not initialized"
        if not self.use_triton_warp_spec:
            return "Triton warp specialization kernels not available"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedWarpSpecializationStreamsBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    config = benchmark.get_config()
    config.use_subprocess = False
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Warp Specialization (Multi-Stream): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
