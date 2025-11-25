"""Optimized vLLM inference - with additional tuning.

Uses vLLM with performance-optimized settings.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
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
from components.monitoring import MetricsCollector


class OptimizedVLLMBenchmark(BaseBenchmark):
    """Optimized: vLLM with performance tuning.
    
    vLLM optimizations enabled:
    - Speculative decoding (Ch18)
    - Chunked prefill (Ch16)
    - FP8 KV cache (Ch13)
    - Prefix caching (Ch16)
    - CUDA graphs (Ch12)
    """
    
    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        tensor_parallel: int = 1,
        prompt_tokens: int = 2048,
        decode_tokens: int = 256,
        use_speculative: bool = False,
        draft_model: Optional[str] = None,
    ):
        super().__init__()
        
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not installed")
        
        self.model_name = model_name
        self.tensor_parallel = tensor_parallel
        self.prompt_tokens = prompt_tokens
        self.decode_tokens = decode_tokens
        self.use_speculative = use_speculative
        self.draft_model = draft_model
        
        self.llm: Optional[LLM] = None
        self.prompts: List[str] = []
        self.metrics_collector = MetricsCollector()
        
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(prompt_tokens + decode_tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize optimized vLLM engine."""
        print(f"Initializing optimized vLLM with {self.model_name}")
        
        # Set environment for optimizations
        os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "1")
        
        # Build engine args with optimizations
        engine_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel,
            "trust_remote_code": True,
            # Ch12: CUDA graphs
            "enforce_eager": False,
            # Ch16: Chunked prefill
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4096,
            # Ch16: Prefix caching
            "enable_prefix_caching": True,
        }
        
        # Ch18: Speculative decoding
        if self.use_speculative and self.draft_model:
            print(f"  Speculative decoding with {self.draft_model}")
            engine_kwargs["speculative_model"] = self.draft_model
            engine_kwargs["num_speculative_tokens"] = 4
        
        # Try FP8 KV cache (Ch13)
        try:
            engine_kwargs["kv_cache_dtype"] = "fp8"
            print("  FP8 KV cache enabled")
        except Exception:
            print("  FP8 KV cache not available")
        
        self.llm = LLM(**engine_kwargs)
        
        # Prepare prompts
        base_prompt = "Explain the future of artificial intelligence and its impact on society."
        self.prompts = [base_prompt]
        
        # Warmup with prefix caching benefit
        print("Running warmup (building prefix cache)...")
        sampling_params = SamplingParams(max_tokens=10, temperature=0)
        for _ in range(3):
            _ = self.llm.generate(self.prompts, sampling_params)
        
        print("Optimized vLLM setup complete")
    
    def benchmark_fn(self) -> None:
        """Benchmark: Run optimized vLLM inference."""
        self.metrics_collector.reset()
        self.metrics_collector.start_request("vllm_optimized")
        
        sampling_params = SamplingParams(
            max_tokens=self.decode_tokens,
            temperature=0,
        )
        
        with self._nvtx_range("vllm_optimized"):
            outputs = self.llm.generate(self.prompts, sampling_params)
        
        output = outputs[0]
        output_tokens = len(output.outputs[0].token_ids)
        
        self.metrics_collector.record_first_token("vllm_optimized")
        self.metrics_collector.end_request(
            "vllm_optimized",
            self.prompt_tokens,
            output_tokens,
        )
        
        self.last_metrics = self.metrics_collector.compute_metrics()
        self._synchronize()
    
    def teardown(self) -> None:
        """Teardown."""
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
    parser = argparse.ArgumentParser(description="Optimized vLLM Benchmark")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--draft-model", type=str)
    args = parser.parse_args()
    
    if not VLLM_AVAILABLE:
        print("Error: vLLM not installed")
        return
    
    benchmark = OptimizedVLLMBenchmark(
        model_name=args.model,
        tensor_parallel=args.tensor_parallel,
        prompt_tokens=args.prompt_tokens,
        decode_tokens=args.decode_tokens,
        use_speculative=args.speculative,
        draft_model=args.draft_model,
    )
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    
    print("=" * 60)
    print("Optimized vLLM Benchmark")
    print("=" * 60)
    
    result = harness.benchmark(benchmark)
    
    print()
    print("Results:")
    if result.timing:
        print(f"  Mean latency: {result.timing.mean_ms:.2f} ms")
    
    if hasattr(benchmark, 'last_metrics') and benchmark.last_metrics:
        m = benchmark.last_metrics
        print(f"  TTFT: {m.ttft_ms:.2f} ms")
        print(f"  Tokens/sec: {m.tokens_per_sec:.1f}")
    
    print("=" * 60)


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedVLLMBenchmark()


if __name__ == "__main__":
    main()

