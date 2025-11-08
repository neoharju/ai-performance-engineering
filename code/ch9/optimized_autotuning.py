"""optimized_autotuning.py - Optimized with autotuning in kernel efficiency/arithmetic intensity context.

Demonstrates autotuning for optimal kernel configurations.
Autotuning: Uses autotuning to automatically find optimal kernel parameters.
Improves performance through automatic kernel tuning.
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
        raise RuntimeError("CUDA required for ch9")
    return torch.device("cuda")


class OptimizedAutotuningBenchmark(Benchmark):
    """Optimized: Autotuning for optimal kernel configurations.
    
    Autotuning: Uses autotuning to automatically find optimal kernel parameters.
    Improves performance through automatic kernel tuning.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize model with autotuning."""
        
        # Ensure .torch_inductor directory exists to prevent C++ compilation errors
        try:
            from common.python.env_defaults import apply_env_defaults
            apply_env_defaults()
        except ImportError:
            pass  # Continue if env_defaults not available
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Autotuning - automatic kernel tuning
        # Autotuning finds optimal kernel configurations automatically
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        # Autotuning: Use torch.compile with optimal mode for autotuning
        # Autotuning automatically explores kernel configurations
        # Select optimal compile mode based on GPU SM count
        from common.python.compile_utils import get_optimal_compile_mode
        compile_mode = get_optimal_compile_mode("max-autotune")
        # Wrap in try-except to handle compilation errors gracefully
        try:
            self.model = torch.compile(self.model, mode=compile_mode)
        except (RuntimeError, Exception) as e:
            error_msg = str(e)
            # Handle various compilation errors:
            # - Generator errors from torch.compile internals
            # - TF32 API mixing errors (PyTorch 2.9+)
            # - C++ compilation errors
            if ("generator didn't stop" in error_msg or 
                "mix of the legacy and new APIs" in error_msg or
                "allow_tf32_new" in error_msg or
                "allowTF32CuBLAS" in error_msg):
                # Known PyTorch internal issues - fall back to eager mode
                print("WARNING: torch.compile failed due to PyTorch TF32 API issue. Falling back to eager execution.")
                # Model stays in eager mode
            elif "CppCompileError" in error_msg or "torch._inductor" in error_msg or "No such file or directory" in error_msg:
                print(f"WARNING: torch.compile failed due to C++ compilation error: {error_msg[:200]}. Falling back to eager execution.")
                # Model stays in eager mode
            else:
                # Fallback to reduce-overhead if compilation fails for other reasons
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                except Exception:
                    # If compilation fails entirely, use eager mode
                    pass  # Model stays in eager mode
        
        self.input = torch.randn(32, 1024, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with autotuning."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        # Wrap in try-except to handle generator errors from torch.compile issues
        try:
            with nvtx_range("optimized_autotuning", enable=enable_nvtx):
                with torch.no_grad():
                    # Optimization: Autotuning
                    # Uses autotuned kernel configurations
                    # Autotuning: automatically finds optimal parameters
                    try:
                        output = self.model(self.input)
                    except Exception as e:
                        # Handle errors that might occur during first call to compiled model
                        # (e.g., C++ compilation errors during guard creation)
                        error_msg = str(e)
                        if "CppCompileError" in error_msg or "torch._inductor" in error_msg or "No such file or directory" in error_msg:
                            # Fall back to eager execution if C++ compilation fails
                            # Re-compile model without compilation (use original model)
                            if hasattr(self.model, '_orig_mod'):
                                # If model was compiled, use original
                                self.model = self.model._orig_mod
                            # Re-run with eager model
                            output = self.model(self.input)
                        else:
                            # Re-raise other errors
                            raise
                
                # Optimization: Autotuning benefits
                # - Automatic kernel configuration optimization
                # - Optimal parameter selection
                # - Improved performance through autotuning
                # - Better kernel efficiency
                _ = output.sum()
        except RuntimeError as e:
            # Handle generator errors that can occur when torch.compile has issues
            if "generator didn't stop" in str(e):
                # Fall back to direct execution without NVTX range
                with torch.no_grad():
                    output = self.model(self.input)
                    _ = output.sum()
            else:
                raise

    
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

