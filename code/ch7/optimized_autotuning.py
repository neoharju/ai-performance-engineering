"""optimized_autotuning.py - Optimized with autotuning in memory access/GEMM context.

Demonstrates autotuning for optimal kernel configuration.
Autotuning: Uses autotuning to find optimal kernel parameters.
Automatically searches for best configurations.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

# Import arch_config to apply Triton patch for sm_12x support
# The patch removes 'a' suffix from sm_121a -> sm_121 for ptxas compatibility
try:
    import arch_config  # noqa: F401
except ImportError:
    pass  # Continue if arch_config not available

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch7")
    return torch.device("cuda")


class OptimizedAutotuningBenchmark(Benchmark):
    """Optimized: Autotuning for optimal kernel configuration.
    
    Autotuning: Uses autotuning to find optimal kernel parameters.
    Automatically searches for best configurations.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.compiled_model = None
        self.input = None
        self._compile_warning: Optional[str] = None
    
    def setup(self) -> None:
        """Setup: Initialize model with autotuning."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Autotuning - automatic kernel optimization
        # Autotuning searches for optimal kernel configurations
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        # Optimization: Compile with autotuning
        # Autotuning: automatically finds optimal configurations
        # arch_config.py patches Triton to handle sm_12x architectures (removes 'a' suffix from sm_121a)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "121")
        # Select optimal compile mode based on GPU SM count
        from common.python.compile_utils import get_optimal_compile_mode
        compile_mode = get_optimal_compile_mode("max-autotune")
        try:
            self.compiled_model = torch.compile(
                self.model,
                mode=compile_mode,
            )
            self._compile_warning = None
        except (RuntimeError, Exception) as exc:
            # Catch various compilation errors including:
            # - TF32 API mixing errors (PyTorch 2.9+)
            # - Generator errors from torch.compile internals
            # - ptxas compatibility issues
            # - Other compilation failures
            error_msg = str(exc)
            if ("generator didn't stop" in error_msg or 
                "mix of the legacy and new APIs" in error_msg or
                "allow_tf32_new" in error_msg or
                "allowTF32CuBLAS" in error_msg):
                # Known PyTorch internal issues - fall back to eager mode
                self._compile_warning = (
                    "torch.compile(max-autotune) unavailable due to PyTorch TF32 API issue - "
                    "falling back to eager execution"
                )
            else:
                # Other compilation errors
                self._compile_warning = (
                    "torch.compile(max-autotune) unavailable - falling back to "
                    f"eager execution ({type(exc).__name__})"
                )
            self.compiled_model = self.model
            if self._compile_warning:
                print(self._compile_warning)
        
        self.input = torch.randn(32, 1024, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with autotuning."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        # Wrap in try-except to handle generator errors and TF32 API errors from torch.compile issues
        try:
            with nvtx_range("optimized_autotuning", enable=enable_nvtx):
                with torch.no_grad():
                    # Optimization: Autotuning
                    # Uses torch.compile with max-autotune to find optimal kernels
                    # Autotuning: automatically searches for best configurations
                    output = self.compiled_model(self.input)
                    
                    # Optimization: Autotuning benefits
                    # - Automatic kernel optimization
                    # - Finds optimal configurations
                    # - Better performance through autotuning
                    # - Optimized kernel parameters
                    _ = output.sum()
        except RuntimeError as e:
            error_msg = str(e)
            # Handle generator errors and TF32 API errors that can occur when torch.compile has issues
            if ("generator didn't stop" in error_msg or 
                "allow_tf32_new" in error_msg or
                "allowTF32CuBLAS" in error_msg or
                "mix of the legacy and new APIs" in error_msg):
                # Fall back to direct execution without NVTX range
                # These are known PyTorch internal issues - model should work in eager mode
                with torch.no_grad():
                    output = self.compiled_model(self.input)
                    _ = output.sum()
            else:
                raise

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.compiled_model = None
        self.input = None
        self._compile_warning = None
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
    return OptimizedAutotuningBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedAutotuningBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Autotuning")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
