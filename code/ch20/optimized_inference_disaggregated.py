"""optimized_inference_disaggregated.py - Disaggregated inference optimization (optimized).

Disaggregated inference separates prefill and decode phases onto different resources.
Prefill uses compute-intensive GPUs, decode uses latency-optimized GPUs.
Better resource utilization and throughput.

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
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")


class SimpleTransformer(nn.Module):
    """Simple transformer for inference demonstration."""
    
    def __init__(self, hidden_dim: int = 1024, num_layers: int = 6):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                dtype=torch.float16
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class OptimizedInferenceDisaggregatedBenchmark(Benchmark):
    """Disaggregated inference optimization - separate prefill and decode."""
    
    def __init__(self):
        self.device = resolve_device()
        self.prefill_model = None  # Optimized for throughput
        self.decode_model = None  # Optimized for latency
        self.prompts: list[torch.Tensor] = []
        self.prefill_stream: Optional[torch.cuda.Stream] = None
        self.decode_stream: Optional[torch.cuda.Stream] = None
        self.hidden_dim = 1024
        self.prompt_len = 512
        self.decode_steps = 10
        self.num_requests = 4
    
    def setup(self) -> None:
        """Setup: Initialize separate prefill and decode models."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        
        # In real disaggregated setup, these would be on different GPUs/nodes
        # Prefill model: optimized for batch processing
        self.prefill_model = SimpleTransformer(hidden_dim=self.hidden_dim, num_layers=6).to(self.device).half().eval()
        
        # Decode model: optimized for single-token latency (could be lighter/shared)
        # Use fewer layers to simulate latency-optimized shard
        self.decode_model = SimpleTransformer(hidden_dim=self.hidden_dim, num_layers=4).to(self.device).half().eval()
        
        self.prompts = [
            torch.randn(1, self.prompt_len, self.hidden_dim, device=self.device, dtype=torch.float16)
            for _ in range(self.num_requests)
        ]
        self.prefill_stream = torch.cuda.Stream()
        self.decode_stream = torch.cuda.Stream()
        
        # Warmup
        with torch.no_grad():
            _ = self.prefill_model(self.prompts[0])
            token = self.prompts[0][:, :1, :]
            _ = self.decode_model(token)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - disaggregated inference."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_inference_disaggregated", enable=enable_nvtx):
            with torch.no_grad():
                prefill_events = [torch.cuda.Event(blocking=False) for _ in range(self.num_requests)]
                cached_tokens: list[Optional[torch.Tensor]] = [None] * self.num_requests
                
                for idx, prompt in enumerate(self.prompts):
                    with torch.cuda.stream(self.prefill_stream):
                        prefill_output = self.prefill_model(prompt)
                        cached_tokens[idx] = prefill_output[:, -1:, :]
                        prefill_events[idx].record(self.prefill_stream)
                    
                    if idx > 0:
                        with torch.cuda.stream(self.decode_stream):
                            self.decode_stream.wait_event(prefill_events[idx - 1])
                            token = cached_tokens[idx - 1]
                            for _ in range(self.decode_steps):
                                token = self.decode_model(token)
                                token = token[:, -1:, :]
                
                # Flush final decode
                with torch.cuda.stream(self.decode_stream):
                    self.decode_stream.wait_event(prefill_events[-1])
                    token = cached_tokens[-1]
                    for _ in range(self.decode_steps):
                        token = self.decode_model(token)
                        token = token[:, -1:, :]
                
                self.prefill_stream.synchronize()
                self.decode_stream.synchronize()
    
    
    def teardown(self) -> None:
        """Cleanup."""
        del self.prefill_model, self.decode_model, self.prompts
        self.prefill_stream = None
        self.decode_stream = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.prefill_model is None:
            return "Prefill_Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedInferenceDisaggregatedBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
    mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    timing = result.timing
    if timing:
        print(f"\nOptimized Disaggregated Inference: {timing.mean_ms:.3f} ms")
    else:
        print("No timing data available")
