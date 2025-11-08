"""optimized_attention.py - Optimized attention with CUDA graphs in kernel launches context.

Demonstrates attention computation with CUDA graphs to reduce kernel launch overhead.
Attention: Uses CUDA graphs to capture and replay attention kernels.
Reduces kernel launch overhead significantly.
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
        raise RuntimeError("CUDA required for ch12")
    return torch.device("cuda")

class OptimizedAttentionBenchmark(Benchmark):
    """Optimized: Attention with CUDA graphs.
    
    Attention: Uses CUDA graphs to capture and replay attention kernels.
    Reduces kernel launch overhead significantly.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
        self._graph_callable = None
    
    def setup(self) -> None:
        """Setup: Initialize attention model with CUDA graphs."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Attention with CUDA graphs
        # Attention: uses CUDA graphs to reduce launch overhead
        
        hidden_dim = 256
        self.model = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        ).to(self.device).eval()
        
        # Attention: prepare input for CUDA graph capture
        self.input = torch.randn(4, 128, hidden_dim, device=self.device)
        
        # CUDA graphs: capture attention computation via make_graphed_callables
        if hasattr(torch.cuda, "make_graphed_callables"):
            try:
                self._graph_callable = torch.cuda.make_graphed_callables(
                    self.model, (self.input, self.input, self.input)
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "Failed to create graphed callable for optimized attention. "
                    "Ensure CUDA graphs are supported on this platform."
                ) from exc
        else:
            # Fallback to eager execution if graphs are unavailable
            self._graph_callable = None
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Attention with CUDA graphs."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_attention", enable=enable_nvtx):
            with torch.no_grad():
                if self._graph_callable is not None:
                    # CUDA graphs replay path
                    _ = self._graph_callable(self.input, self.input, self.input)
                else:
                    _ = self.model(self.input, self.input, self.input)[0]
                _ = self.input.sum()  # Use input for validation

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        self._graph_callable = None
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
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedAttentionBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Attention")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
