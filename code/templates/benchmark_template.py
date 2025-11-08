"""Template for creating new benchmarks.

This template provides a complete example of a benchmark implementation
following the benchmark contract. Copy this file and modify it for your needs.

Usage:
    1. Copy this template to your chapter directory
    2. Rename the file (e.g., baseline_my_benchmark.py or optimized_my_benchmark.py)
    3. Implement the required methods: setup(), benchmark_fn(), teardown()
    4. Optionally implement get_config() and validate_result()
    5. Add a get_benchmark() function that returns your benchmark instance
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class MyBenchmark(BaseBenchmark):
    """Description of what this benchmark measures.
    
    This benchmark demonstrates [technique/optimization/pattern].
    
    Key aspects:
    - What it measures (e.g., "Matrix multiplication performance")
    - What technique it uses (e.g., "Naive implementation" or "Optimized with tensor cores")
    - Expected performance characteristics
    """
    
    def __init__(self):
        """Initialize benchmark with device resolution.
        
        Subclasses should call super().__init__() and then set up their own attributes.
        """
        super().__init__()
        # Add your benchmark-specific attributes here
        self.model: Optional[nn.Module] = None
        self.input_data: Optional[torch.Tensor] = None
        # ... other attributes
    
    def setup(self) -> None:
        """Setup phase: initialize models, data, etc.
        
        This method is called once before benchmarking begins.
        Use it to:
        - Initialize models, tensors, or other resources
        - Move data to the correct device
        - Set random seeds if needed
        - Perform any one-time setup
        
        Example:
            torch.manual_seed(42)
            self.model = nn.Linear(256, 256).to(self.device)
            self.input_data = torch.randn(32, 256, device=self.device)
            torch.cuda.synchronize()  # Ensure setup is complete
        """
        # Set random seed for reproducibility (optional)
        torch.manual_seed(42)
        
        # Initialize your model/data here
        self.model = nn.Linear(256, 256).to(self.device)
        self.input_data = torch.randn(32, 256, device=self.device)
        
        # Synchronize CUDA operations to ensure setup is complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark. Must be callable with no args.
        
        This method is called repeatedly during benchmarking.
        It should contain the code you want to measure.
        
        Best practices:
        - Keep it focused on the operation being measured
        - Use NVTX ranges for profiling (via self._nvtx_range())
        - Synchronize CUDA operations if needed (via self._synchronize())
        - Avoid unnecessary overhead (e.g., print statements)
        
        Example:
            with self._nvtx_range("my_operation"):
                output = self.model(self.input_data)
                self._synchronize()  # Wait for CUDA operations to complete
        """
        # Use NVTX ranges for profiling (automatically enabled when profiling is on)
        with self._nvtx_range("my_benchmark_operation"):
            # Your benchmark code here
            output = self.model(self.input_data)
            
            # Synchronize CUDA operations to ensure accurate timing
            self._synchronize()
    
    def teardown(self) -> None:
        """Cleanup phase.
        
        This method is called once after benchmarking completes.
        Use it to:
        - Free GPU memory
        - Clean up resources
        - Reset state
        
        Note: This is called even if benchmarking fails, so make it safe.
        """
        # Clean up resources
        self.model = None
        self.input_data = None
        
        # Optionally clear CUDA cache (usually not needed, harness handles this)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> Optional[BenchmarkConfig]:
        """Optional: return benchmark-specific config overrides.
        
        Use this to customize benchmark configuration for this specific benchmark.
        Common use cases:
        - Different iteration counts for slow benchmarks
        - Custom timeouts for benchmarks that take longer
        - Enabling memory tracking for memory-intensive benchmarks
        
        Returns:
            BenchmarkConfig with overrides, or None to use defaults
            
        Example:
            config = BenchmarkConfig()
            config.iterations = 50  # Fewer iterations for slow benchmark
            config.measurement_timeout_seconds = 30  # Longer timeout
            config.enable_memory_tracking = True  # Track memory usage
            return config
        """
        # Return None to use default config
        return None
        
        # Or return a custom config:
        # config = BenchmarkConfig()
        # config.iterations = 50
        # return config
    
    def validate_result(self) -> Optional[str]:
        """Optional: validate benchmark result, return error message if invalid.
        
        This method is called after benchmarking to validate the results.
        Use it to check:
        - Correctness of computations
        - Expected performance characteristics
        - Resource usage (memory, etc.)
        
        Returns:
            None if result is valid, or error message string if invalid
            
        Example:
            # Check that output is not NaN
            if torch.isnan(self.last_output).any():
                return "Output contains NaN values"
            return None
        """
        # Return None if validation passes
        return None
        
        # Or return an error message if validation fails:
        # if some_condition:
        #     return "Error description"


def get_benchmark():
    """Factory function that returns a benchmark instance.
    
    This function is required for benchmark discovery.
    It should return an instance of your benchmark class.
    
    Returns:
        Benchmark instance
    """
    return MyBenchmark()


# Optional: Add a main function for standalone testing
if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness
    
    # Create benchmark instance
    benchmark = get_benchmark()
    
    # Create harness with default config
    harness = BenchmarkHarness()
    
    # Run benchmark
    result = harness.benchmark(benchmark)
    
    # Print results
    if result.timing:
        print(f"Mean: {result.timing.mean_ms:.2f} ms")
        print(f"Median: {result.timing.median_ms:.2f} ms")
        print(f"Std: {result.timing.std_ms:.2f} ms")
    
    if result.memory:
        print(f"Peak memory: {result.memory.peak_mb:.2f} MB")

