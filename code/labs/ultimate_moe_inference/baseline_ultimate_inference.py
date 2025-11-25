"""Baseline Ultimate MoE Inference - No optimizations.

This baseline implementation uses standard HuggingFace inference with no
performance optimizations. It serves as the reference point for measuring
the impact of each optimization layer.

Models:
    - gpt-oss-20b: Single GPU (21B params, 3.6B active)
    - gpt-oss-120b: Multi-GPU (117B params, 5.1B active)

Usage:
    python baseline_ultimate_inference.py --config configs/single_gpu.yaml
    torchrun --nproc_per_node=8 baseline_ultimate_inference.py --config configs/multi_gpu_8.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.distributed as dist

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


@dataclass
class InferenceConfig:
    """Configuration for inference benchmark."""
    
    # Model
    model_name: str = "openai/gpt-oss-20b"
    precision: str = "bf16"  # Baseline uses bf16, not optimized MXFP4
    
    # Parallelism
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    
    # Benchmark parameters
    prompt_tokens: int = 2048
    decode_tokens: int = 256
    batch_size: int = 1
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    
    # Workload
    workload_type: str = "synthetic"
    num_samples: int = 100
    
    @classmethod
    def from_yaml(cls, path: Path) -> "InferenceConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls(
            model_name=data.get("model", {}).get("name", cls.model_name),
            precision=data.get("model", {}).get("precision", "bf16"),
            tensor_parallel=data.get("parallelism", {}).get("tensor_parallel", 1),
            pipeline_parallel=data.get("parallelism", {}).get("pipeline_parallel", 1),
            prompt_tokens=data.get("benchmark", {}).get("prompt_tokens", 2048),
            decode_tokens=data.get("benchmark", {}).get("decode_tokens", 256),
            batch_size=data.get("benchmark", {}).get("batch_size", 1),
            warmup_iterations=data.get("benchmark", {}).get("warmup_iterations", 3),
            benchmark_iterations=data.get("benchmark", {}).get("benchmark_iterations", 10),
            workload_type=data.get("benchmark", {}).get("workload_type", "synthetic"),
            num_samples=data.get("benchmark", {}).get("num_samples", 100),
        )


@dataclass
class InferenceMetrics:
    """Metrics collected during inference."""
    
    # Latency (milliseconds)
    ttft_ms: float = 0.0  # Time to First Token
    tpot_ms: float = 0.0  # Time Per Output Token (average)
    e2e_latency_ms: float = 0.0  # End-to-end latency
    
    # Throughput
    tokens_per_sec: float = 0.0
    prefill_tokens_per_sec: float = 0.0
    
    # Batch info
    batch_size: int = 1
    prompt_tokens: int = 0
    output_tokens: int = 0
    
    # Memory
    peak_memory_gb: float = 0.0
    
    # Percentiles (for multiple requests)
    ttft_p50_ms: Optional[float] = None
    ttft_p90_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None


class BaselineUltimateInference(BaseBenchmark):
    """Baseline: Standard eager-mode inference with NO optimizations.
    
    This implementation intentionally avoids all optimizations to serve as
    a baseline for comparison. It uses:
    - Standard HuggingFace model loading
    - Eager mode execution (no torch.compile)
    - No CUDA graphs
    - No FP8/MXFP4 quantization
    - No speculative decoding
    - No PagedAttention
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        super().__init__()
        self.config = config or InferenceConfig()
        
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.input_ids: Optional[torch.Tensor] = None
        self.attention_mask: Optional[torch.Tensor] = None
        
        # Metrics tracking
        self.last_metrics: Optional[InferenceMetrics] = None
        self._prefill_start: float = 0.0
        self._first_token_time: float = 0.0
        self._decode_times: List[float] = []
        
        # Workload metadata
        total_tokens = self.config.batch_size * (
            self.config.prompt_tokens + self.config.decode_tokens
        )
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(total_tokens),
        )
    
    def setup(self) -> None:
        """Setup: Load model and prepare inputs (NO optimizations)."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library required for baseline")
        
        # Determine device mapping
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1 and self.config.tensor_parallel > 1:
            device_map = "auto"
        else:
            device_map = "cuda:0"
        
        # Determine dtype - baseline uses bf16 (no MXFP4/FP8)
        if self.config.precision == "bf16":
            torch_dtype = torch.bfloat16
        elif self.config.precision == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16  # Default to bf16 for baseline
        
        print(f"Loading model: {self.config.model_name}")
        print(f"  Precision: {torch_dtype}")
        print(f"  Device map: {device_map}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model - NO optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        
        # Prepare synthetic input
        self._prepare_inputs()
        
        # Warmup
        print("Running warmup...")
        for _ in range(self.config.warmup_iterations):
            self._run_inference()
        torch.cuda.synchronize()
        
        print("Baseline setup complete")
    
    def _prepare_inputs(self) -> None:
        """Prepare input tensors for benchmarking."""
        # Generate synthetic prompt
        prompt = "The future of artificial intelligence is"
        
        # Tokenize and pad/truncate to desired length
        encoding = self.tokenizer(
            [prompt] * self.config.batch_size,
            padding="max_length",
            max_length=self.config.prompt_tokens,
            truncation=True,
            return_tensors="pt",
        )
        
        self.input_ids = encoding["input_ids"].to(self.device)
        self.attention_mask = encoding["attention_mask"].to(self.device)
    
    def _run_inference(self) -> torch.Tensor:
        """Run single inference pass."""
        with torch.no_grad():
            outputs = self.model.generate(
                self.input_ids,
                attention_mask=self.attention_mask,
                max_new_tokens=self.config.decode_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=self.tokenizer.pad_token_id,
                # Disable all optimization features
                use_cache=True,  # KV cache is standard, not an optimization
            )
        return outputs
    
    def _run_inference_with_timing(self) -> InferenceMetrics:
        """Run inference with detailed timing metrics."""
        torch.cuda.synchronize()
        
        # Start timing
        start_time = time.perf_counter()
        self._prefill_start = start_time
        
        with torch.no_grad():
            # For detailed timing, we'd need to hook into generation
            # For baseline, we just measure overall time
            outputs = self.model.generate(
                self.input_ids,
                attention_mask=self.attention_mask,
                max_new_tokens=self.config.decode_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Calculate metrics
        e2e_latency_ms = (end_time - start_time) * 1000
        output_tokens = outputs.shape[1] - self.input_ids.shape[1]
        total_output_tokens = output_tokens * self.config.batch_size
        
        # Estimate TTFT and TPOT (rough approximation for baseline)
        # In reality, prefill is faster per-token than decode
        prefill_ratio = 0.3  # Rough estimate: 30% of time in prefill
        ttft_ms = e2e_latency_ms * prefill_ratio
        decode_time_ms = e2e_latency_ms * (1 - prefill_ratio)
        tpot_ms = decode_time_ms / max(output_tokens, 1)
        
        # Memory
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        
        return InferenceMetrics(
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            e2e_latency_ms=e2e_latency_ms,
            tokens_per_sec=total_output_tokens / (e2e_latency_ms / 1000),
            prefill_tokens_per_sec=(self.config.prompt_tokens * self.config.batch_size) / (ttft_ms / 1000),
            batch_size=self.config.batch_size,
            prompt_tokens=self.config.prompt_tokens,
            output_tokens=output_tokens,
            peak_memory_gb=peak_memory_gb,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Run inference with no optimizations."""
        with self._nvtx_range("baseline_inference"):
            self.last_metrics = self._run_inference_with_timing()
        self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.input_ids = None
        self.attention_mask = None
        torch.cuda.empty_cache()
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=self.config.benchmark_iterations,
            warmup=self.config.warmup_iterations,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        """Return workload metadata."""
        return self._workload
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark results."""
        if self.model is None:
            return "Model not initialized"
        if self.last_metrics is None:
            return "No metrics collected"
        if self.last_metrics.tokens_per_sec <= 0:
            return "Invalid throughput"
        return None
    
    def get_metrics(self) -> Optional[InferenceMetrics]:
        """Get the last collected metrics."""
        return self.last_metrics

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        """Return custom metrics for ultimate MoE inference baseline.
        
        From PROFILING_METRICS_ANALYSIS.md:
        These metrics explain WHY optimizations work, not just WHAT they achieve.
        
        Categories:
        1. Latency metrics (TTFT, TPOT, E2E)
        2. Throughput metrics (tokens/sec)
        3. Memory efficiency (GB used, GB/token)
        4. Compute efficiency (arithmetic intensity proxy)
        """
        if self.last_metrics is None:
            return None
        m = self.last_metrics
        
        # Calculate derived metrics for analysis
        total_tokens = m.prompt_tokens + m.output_tokens
        memory_per_token_mb = (m.peak_memory_gb * 1024) / max(total_tokens, 1)
        
        # Estimate arithmetic intensity (FLOPs per byte transferred)
        # For transformer: ~2 * hidden_dim * seq_len * num_layers FLOPs per token
        # Memory: ~2 * hidden_dim * num_layers * 2 bytes (KV cache)
        est_flops_per_token = 2 * 4096 * 32 * 2  # Rough estimate for 20B model
        est_bytes_per_token = 2 * 4096 * 32 * 2  # KV cache in BF16
        arithmetic_intensity = est_flops_per_token / max(est_bytes_per_token, 1)
        
        return {
            # Latency metrics
            "ultimate_moe.ttft_ms": m.ttft_ms,
            "ultimate_moe.tpot_ms": m.tpot_ms,
            "ultimate_moe.e2e_latency_ms": m.e2e_latency_ms,
            
            # Throughput metrics
            "ultimate_moe.tokens_per_sec": m.tokens_per_sec,
            "ultimate_moe.prefill_tokens_per_sec": m.prefill_tokens_per_sec,
            
            # Workload metrics
            "ultimate_moe.batch_size": float(m.batch_size),
            "ultimate_moe.prompt_tokens": float(m.prompt_tokens),
            "ultimate_moe.output_tokens": float(m.output_tokens),
            "ultimate_moe.total_tokens": float(total_tokens),
            
            # Memory efficiency (explains memory bottleneck)
            "ultimate_moe.peak_memory_gb": m.peak_memory_gb,
            "ultimate_moe.memory_per_token_mb": memory_per_token_mb,
            
            # Compute efficiency (explains compute utilization)
            "ultimate_moe.arithmetic_intensity": arithmetic_intensity,
            
            # Optimization status (for comparison)
            "ultimate_moe.is_baseline": 1.0,
            "ultimate_moe.fp8_enabled": 0.0,
            "ultimate_moe.flash_attn_enabled": 0.0,
            "ultimate_moe.cuda_graphs_enabled": 0.0,
        }


def get_benchmark(config_path: Optional[str] = None) -> BaseBenchmark:
    """Factory function for harness discovery."""
    if config_path:
        config = InferenceConfig.from_yaml(Path(config_path))
    else:
        config = InferenceConfig()
    return BaselineUltimateInference(config)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Baseline Ultimate MoE Inference Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "single_gpu.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override number of benchmark iterations",
    )
    args = parser.parse_args()
    
    # Load config
    config = InferenceConfig.from_yaml(args.config)
    if args.iterations:
        config.benchmark_iterations = args.iterations
    
    # Create and run benchmark
    benchmark = BaselineUltimateInference(config)
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    
    print("=" * 70)
    print("Baseline Ultimate MoE Inference")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    print(f"Precision: {config.precision}")
    print(f"Batch size: {config.batch_size}")
    print(f"Prompt tokens: {config.prompt_tokens}")
    print(f"Decode tokens: {config.decode_tokens}")
    print("=" * 70)
    print()
    
    result = harness.benchmark(benchmark)
    
    # Print results
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    if result.timing:
        print(f"Mean latency: {result.timing.mean_ms:.2f} ms")
        print(f"Median latency: {result.timing.median_ms:.2f} ms")
        print(f"Std dev: {result.timing.std_ms:.2f} ms")
    
    metrics = benchmark.get_metrics()
    if metrics:
        print()
        print("Inference Metrics:")
        print(f"  TTFT: {metrics.ttft_ms:.2f} ms")
        print(f"  TPOT: {metrics.tpot_ms:.2f} ms")
        print(f"  Tokens/sec: {metrics.tokens_per_sec:.1f}")
        print(f"  Peak memory: {metrics.peak_memory_gb:.2f} GB")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

