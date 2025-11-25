"""Baseline vLLM inference - out of box performance.

Uses vLLM with default settings for comparison against our
from-scratch implementation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from components.monitoring import MetricsCollector, InferenceMetrics


class BaselineVLLMBenchmark(BaseBenchmark):
    """Baseline: vLLM with default settings.
    
    vLLM internally implements many of the same optimizations we teach:
    - PagedAttention (Ch16)
    - Continuous batching (Ch17)
    - CUDA graphs (Ch12)
    - FlashAttention (Ch10)
    
    This baseline shows out-of-box vLLM performance.
    """
    
    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        tensor_parallel: int = 1,
        prompt_tokens: int = 2048,
        decode_tokens: int = 256,
    ):
        super().__init__()
        
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not installed. Install with: pip install vllm")
        
        self.model_name = model_name
        self.tensor_parallel = tensor_parallel
        self.prompt_tokens = prompt_tokens
        self.decode_tokens = decode_tokens
        
        self.llm: Optional[LLM] = None
        self.prompts: List[str] = []
        self.metrics_collector = MetricsCollector()
        
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(prompt_tokens + decode_tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize vLLM engine."""
        print(f"Initializing vLLM with {self.model_name}")
        print(f"  Tensor parallel: {self.tensor_parallel}")
        
        # vLLM automatically applies:
        # - PagedAttention for KV cache
        # - Continuous batching
        # - CUDA graphs (when applicable)
        # - FlashAttention (when available)
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel,
            trust_remote_code=True,
            # Use defaults for baseline comparison
        )
        
        # Prepare prompts
        base_prompt = "Explain the future of artificial intelligence and its impact on society."
        self.prompts = [base_prompt]
        
        # Warmup
        print("Running warmup...")
        sampling_params = SamplingParams(max_tokens=10, temperature=0)
        _ = self.llm.generate(self.prompts, sampling_params)
        
        print("vLLM baseline setup complete")
    
    def benchmark_fn(self) -> None:
        """Benchmark: Run vLLM inference."""
        self.metrics_collector.reset()
        self.metrics_collector.start_request("vllm_baseline")
        
        sampling_params = SamplingParams(
            max_tokens=self.decode_tokens,
            temperature=0,  # Greedy for reproducibility
        )
        
        start_time = time.perf_counter()
        
        with self._nvtx_range("vllm_baseline"):
            outputs = self.llm.generate(self.prompts, sampling_params)
        
        end_time = time.perf_counter()
        
        # Extract metrics
        output = outputs[0]
        output_tokens = len(output.outputs[0].token_ids)
        
        self.metrics_collector.record_first_token("vllm_baseline")
        self.metrics_collector.end_request(
            "vllm_baseline",
            self.prompt_tokens,
            output_tokens,
        )
        
        self.last_metrics = self.metrics_collector.compute_metrics()
        self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up vLLM."""
        del self.llm
        self.llm = None
        torch.cuda.empty_cache()
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(iterations=10, warmup=3)
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        """Return workload metadata."""
        return self._workload
    
    def validate_result(self) -> Optional[str]:
        """Validate result."""
        if self.llm is None:
            return "vLLM not initialized"
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Baseline vLLM Benchmark")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--decode-tokens", type=int, default=256)
    args = parser.parse_args()
    
    if not VLLM_AVAILABLE:
        print("Error: vLLM not installed")
        print("Install with: pip install vllm")
        return
    
    benchmark = BaselineVLLMBenchmark(
        model_name=args.model,
        tensor_parallel=args.tensor_parallel,
        prompt_tokens=args.prompt_tokens,
        decode_tokens=args.decode_tokens,
    )
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    
    print("=" * 60)
    print("Baseline vLLM Benchmark")
    print("=" * 60)
    
    result = harness.benchmark(benchmark)
    
    print()
    print("Results:")
    if result.timing:
        print(f"  Mean latency: {result.timing.mean_ms:.2f} ms")
        print(f"  Median: {result.timing.median_ms:.2f} ms")
    
    if hasattr(benchmark, 'last_metrics') and benchmark.last_metrics:
        m = benchmark.last_metrics
        print(f"  TTFT: {m.ttft_ms:.2f} ms")
        print(f"  Tokens/sec: {m.tokens_per_sec:.1f}")
    
    print("=" * 60)


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineVLLMBenchmark()


if __name__ == "__main__":
    main()

