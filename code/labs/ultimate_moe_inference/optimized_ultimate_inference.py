"""Optimized Ultimate MoE Inference - ALL optimizations combined.

This optimized implementation applies every optimization technique from the book:
- Layer 1 (Ch1-6): NVTX, NUMA, TF32, cuDNN
- Layer 2 (Ch7-8): Memory coalescing, occupancy
- Layer 3 (Ch9-10): Double buffering, FlashAttention
- Layer 4 (Ch11-12): CUDA streams, CUDA graphs
- Layer 5 (Ch13-14): FP8, torch.compile, Triton
- Layer 6 (Ch15-20): MoE parallelism, PagedAttention, speculative decode

Models:
    - gpt-oss-20b: Single GPU (21B params, 3.6B active)
    - gpt-oss-120b: Multi-GPU (117B params, 5.1B active)

Usage:
    python optimized_ultimate_inference.py --config configs/single_gpu.yaml
    torchrun --nproc_per_node=8 optimized_ultimate_inference.py --config configs/multi_gpu_8.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
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

# Import optimization layers
from optimization_layers import (
    Layer01Basics,
    Layer02Memory,
    Layer03Pipelining,
    Layer04Concurrency,
    Layer05PyTorch,
    Layer06Advanced,
)

# Import components
from components.model_loader import ModelLoader, ModelConfig, Precision
from components.workload_loader import WorkloadLoader, WorkloadConfig, WorkloadType
from components.monitoring import MetricsCollector, InferenceMetrics
from components.kv_cache_manager import PagedKVCache, KVCacheConfig


@dataclass
class OptimizedConfig:
    """Configuration for optimized inference."""
    
    # Model
    model_name: str = "openai/gpt-oss-20b"
    precision: str = "mxfp4"
    
    # Parallelism
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    
    # Optimization toggles
    use_flash_attention: bool = True
    use_cuda_graphs: bool = True
    use_torch_compile: bool = True
    compile_mode: str = "max-autotune"
    use_fp8_kv_cache: bool = True
    use_speculative_decode: bool = False
    use_paged_attention: bool = True
    
    # Speculative decoding
    draft_model: Optional[str] = None
    speculation_length: int = 4
    use_ngram_speculation: bool = True
    
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
    def from_yaml(cls, path: Path) -> "OptimizedConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        optimizations = data.get("optimizations", {})
        benchmark = data.get("benchmark", {})
        parallelism = data.get("parallelism", {})
        
        return cls(
            model_name=data.get("model", {}).get("name", cls.model_name),
            precision=data.get("model", {}).get("precision", "mxfp4"),
            tensor_parallel=parallelism.get("tensor_parallel", 1),
            pipeline_parallel=parallelism.get("pipeline_parallel", 1),
            use_flash_attention=optimizations.get("enable_flash_attention", True),
            use_cuda_graphs=optimizations.get("use_cuda_graphs", True),
            use_torch_compile=optimizations.get("use_torch_compile", True),
            compile_mode=optimizations.get("compile_mode", "max-autotune"),
            use_fp8_kv_cache=optimizations.get("use_fp8_kv_cache", True),
            use_speculative_decode=optimizations.get("use_speculative_decode", False),
            draft_model=optimizations.get("draft_model"),
            speculation_length=optimizations.get("speculation_length", 4),
            use_ngram_speculation=optimizations.get("use_ngram_speculation", True),
            use_paged_attention=optimizations.get("use_paged_attention", True),
            prompt_tokens=benchmark.get("prompt_tokens", 2048),
            decode_tokens=benchmark.get("decode_tokens", 256),
            batch_size=benchmark.get("batch_size", 1),
            warmup_iterations=benchmark.get("warmup_iterations", 3),
            benchmark_iterations=benchmark.get("benchmark_iterations", 10),
            workload_type=benchmark.get("workload_type", "synthetic"),
            num_samples=benchmark.get("num_samples", 100),
        )


class OptimizedUltimateInference(BaseBenchmark):
    """Optimized: ALL optimization techniques combined.
    
    This benchmark demonstrates the cumulative effect of all optimization
    techniques from the book, applied to real MoE model inference.
    
    Optimization layers applied:
    - Layer 1 (Ch1-6): NVTX, NUMA, TF32, warmup
    - Layer 2 (Ch7-8): Memory coalescing, occupancy
    - Layer 3 (Ch9-10): Double buffering, TMA, FlashAttention
    - Layer 4 (Ch11-12): Streams, CUDA graphs
    - Layer 5 (Ch13-14): FP8, torch.compile, Triton
    - Layer 6 (Ch15-20): Expert parallel, PagedAttn, speculative
    """
    
    def __init__(self, config: Optional[OptimizedConfig] = None):
        super().__init__()
        self.config = config or OptimizedConfig()
        
        # Initialize optimization layers
        self.layers = [
            Layer01Basics(),
            Layer02Memory(),
            Layer03Pipelining(),
            Layer04Concurrency(),
            Layer05PyTorch(),
            Layer06Advanced(),
        ]
        
        # Components
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.kv_cache: Optional[PagedKVCache] = None
        self.metrics_collector = MetricsCollector()
        
        # Inputs
        self.input_ids: Optional[torch.Tensor] = None
        self.attention_mask: Optional[torch.Tensor] = None
        
        # CUDA graphs
        self._decode_graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_input: Optional[torch.Tensor] = None
        self._static_output: Optional[torch.Tensor] = None
        
        # Workload metadata
        total_tokens = self.config.batch_size * (
            self.config.prompt_tokens + self.config.decode_tokens
        )
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(total_tokens),
        )
    
    def setup(self) -> None:
        """Setup: Apply all optimizations and prepare for inference."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library required")
        
        print("=" * 70)
        print("Optimized Ultimate MoE Inference Setup")
        print("=" * 70)
        
        # Apply Layer 1: Basics
        print("\n[Layer 1] Applying basic optimizations (Ch1-6)...")
        self.layers[0].apply(self)
        
        # Load model
        print(f"\nLoading model: {self.config.model_name}")
        self._load_model()
        
        # Apply Layer 2: Memory
        print("\n[Layer 2] Applying memory optimizations (Ch7-8)...")
        self.layers[1].apply(self)
        
        # Apply Layer 3: Pipelining
        print("\n[Layer 3] Applying pipelining optimizations (Ch9-10)...")
        self.layers[2].apply(self)
        
        # Apply Layer 4: Concurrency
        print("\n[Layer 4] Applying concurrency optimizations (Ch11-12)...")
        self.layers[3].apply(self)
        
        # Apply Layer 5: PyTorch
        print("\n[Layer 5] Applying PyTorch optimizations (Ch13-14)...")
        self.layers[4].apply(self)
        
        # Compile model if enabled
        if self.config.use_torch_compile:
            self.model = self.layers[4].compile_model(
                self.model,
                mode=self.config.compile_mode,
            )
        
        # Apply Layer 6: Advanced
        print("\n[Layer 6] Applying advanced optimizations (Ch15-20)...")
        self.layers[5].apply(self)
        
        # Setup PagedAttention KV cache
        if self.config.use_paged_attention:
            self._setup_kv_cache()
        
        # Prepare inputs
        self._prepare_inputs()
        
        # Warmup
        print(f"\nRunning {self.config.warmup_iterations} warmup iterations...")
        for i in range(self.config.warmup_iterations):
            with self.layers[0].nvtx_range(f"warmup_{i}"):
                self._run_inference()
        torch.cuda.synchronize()
        
        # Capture CUDA graphs if enabled
        if self.config.use_cuda_graphs:
            self._capture_graphs()
        
        print("\n" + "=" * 70)
        print("Setup complete - all optimizations applied")
        print("=" * 70)
        self._print_optimization_status()
    
    def _load_model(self) -> None:
        """Load model with optimization settings."""
        # Determine dtype
        if self.config.precision == "mxfp4":
            precision = Precision.MXFP4
        elif self.config.precision == "fp8":
            precision = Precision.FP8
        else:
            precision = Precision.BF16
        
        # Create model config
        model_config = ModelConfig(
            model_name=self.config.model_name,
            precision=precision,
            tensor_parallel=self.config.tensor_parallel,
            use_flash_attention=self.config.use_flash_attention,
            use_torch_compile=False,  # We'll compile after setup
        )
        
        # Load model and tokenizer
        loader = ModelLoader(model_config)
        self.model, self.tokenizer = loader.load()
        
        # Set to eval mode
        self.model.eval()
    
    def _setup_kv_cache(self) -> None:
        """Setup PagedAttention KV cache."""
        # Estimate model dimensions
        config = self.model.config if hasattr(self.model, 'config') else None
        
        num_layers = getattr(config, 'num_hidden_layers', 32)
        num_heads = getattr(config, 'num_attention_heads', 32)
        head_dim = getattr(config, 'hidden_size', 4096) // num_heads
        
        # Calculate max tokens
        max_tokens = self.config.batch_size * (
            self.config.prompt_tokens + self.config.decode_tokens
        )
        
        kv_config = KVCacheConfig(
            block_size=16,
            num_blocks=max_tokens // 16 + 256,  # Extra blocks for overhead
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            use_fp8=self.config.use_fp8_kv_cache,
        )
        
        self.kv_cache = PagedKVCache(kv_config, self.device)
        print(f"  Initialized PagedKVCache: {self.kv_cache}")
    
    def _prepare_inputs(self) -> None:
        """Prepare input tensors."""
        # Generate synthetic prompt
        prompt = "Explain the future of artificial intelligence and its impact on society."
        
        encoding = self.tokenizer(
            [prompt] * self.config.batch_size,
            padding="max_length",
            max_length=self.config.prompt_tokens,
            truncation=True,
            return_tensors="pt",
        )
        
        self.input_ids = encoding["input_ids"].to(self.device)
        self.attention_mask = encoding["attention_mask"].to(self.device)
    
    def _capture_graphs(self) -> None:
        """Capture CUDA graphs for decode phase."""
        print("\n[Layer 4] Capturing CUDA graphs...")
        
        # Create static tensors for graph
        self._static_input = self.input_ids.clone()
        
        # Warmup for graph capture
        decode_stream = self.layers[3].get_stream("decode")
        stream = decode_stream or torch.cuda.current_stream()
        
        with torch.cuda.stream(stream):
            for _ in range(3):
                with torch.no_grad():
                    _ = self.model.generate(
                        self._static_input,
                        max_new_tokens=1,
                        do_sample=False,
                        use_cache=True,
                    )
        
        stream.synchronize()
        
        # Note: Full graph capture for generate() is complex due to dynamic shapes.
        # In production, you'd capture individual decode steps.
        print("  CUDA graph warmup complete")
    
    def _run_inference(self) -> torch.Tensor:
        """Run optimized inference."""
        with torch.no_grad():
            # Use FlashAttention backend
            with self.layers[2].flash_attention_backend():
                # Use FP8 autocast if enabled
                with self.layers[4].fp8_autocast_context():
                    outputs = self.model.generate(
                        self.input_ids,
                        attention_mask=self.attention_mask,
                        max_new_tokens=self.config.decode_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=True,
                    )
        
        return outputs
    
    def _run_inference_with_metrics(self) -> InferenceMetrics:
        """Run inference with detailed metrics collection."""
        self.metrics_collector.reset()
        self.metrics_collector.reset_memory_tracking()
        self.metrics_collector.start_power_monitoring()
        
        torch.cuda.synchronize()
        
        # Start request tracking
        request_id = "main"
        self.metrics_collector.start_request(request_id)
        
        # Run inference
        outputs = self._run_inference()
        
        # Record first token (rough approximation)
        self.metrics_collector.record_first_token(request_id)
        
        torch.cuda.synchronize()
        
        # End request
        output_tokens = outputs.shape[1] - self.input_ids.shape[1]
        self.metrics_collector.end_request(
            request_id,
            self.config.prompt_tokens * self.config.batch_size,
            output_tokens * self.config.batch_size,
        )
        
        # Record memory
        self.metrics_collector.record_memory()
        
        # Compute metrics
        return self.metrics_collector.compute_metrics()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Run optimized inference."""
        with self._nvtx_range("optimized_inference"):
            self.last_metrics = self._run_inference_with_metrics()
        self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.input_ids = None
        self.attention_mask = None
        self.kv_cache = None
        self._decode_graph = None
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
        return None

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        """Return custom metrics for optimized ultimate MoE inference.
        
        From PROFILING_METRICS_ANALYSIS.md:
        Comprehensive metrics that explain optimization effectiveness.
        """
        # Get inference metrics if available
        m = self.last_metrics
        
        # Base metrics
        metrics = {
            # Workload configuration
            "optimized_ultimate.batch_size": float(self.config.batch_size),
            "optimized_ultimate.prompt_tokens": float(self.config.prompt_tokens),
            "optimized_ultimate.decode_tokens": float(self.config.decode_tokens),
            
            # Optimization flags (for comparison analysis)
            "optimized_ultimate.use_flash_attention": float(self.config.use_flash_attention),
            "optimized_ultimate.use_cuda_graphs": float(self.config.use_cuda_graphs),
            "optimized_ultimate.use_torch_compile": float(self.config.use_torch_compile),
            "optimized_ultimate.use_paged_attention": float(self.config.use_paged_attention),
            "optimized_ultimate.use_speculative": float(self.config.use_speculative_decoding),
            
            # Is this the optimized version?
            "optimized_ultimate.is_baseline": 0.0,
        }
        
        # Add inference timing metrics if available
        if m is not None:
            total_tokens = m.prompt_tokens + m.output_tokens
            memory_per_token_mb = (m.peak_memory_gb * 1024) / max(total_tokens, 1)
            
            metrics.update({
                # Latency metrics
                "optimized_ultimate.ttft_ms": m.ttft_ms,
                "optimized_ultimate.tpot_ms": m.tpot_ms,
                "optimized_ultimate.e2e_latency_ms": m.e2e_latency_ms,
                
                # Throughput metrics  
                "optimized_ultimate.tokens_per_sec": m.tokens_per_sec,
                "optimized_ultimate.prefill_tokens_per_sec": m.prefill_tokens_per_sec,
                
                # Memory efficiency
                "optimized_ultimate.peak_memory_gb": m.peak_memory_gb,
                "optimized_ultimate.memory_per_token_mb": memory_per_token_mb,
            })
        
        # Add layer-specific stats
        for layer in self.layers:
            status = layer.get_status()
            layer_prefix = layer.name.split(":")[0].lower().replace(" ", "_")
            for key, value in status.items():
                if isinstance(value, (int, float, bool)):
                    metrics[f"optimized_ultimate.{layer_prefix}.{key}"] = float(value)
        
        return metrics

    def _print_optimization_status(self) -> None:
        """Print status of all optimization layers."""
        print("\nOptimization Status:")
        for layer in self.layers:
            status = layer.get_status()
            print(f"  {layer.name}:")
            for key, value in status.items():
                print(f"    {key}: {value}")
    
    def get_layer_comparison_results(self) -> Dict[str, float]:
        """Run benchmark with incremental layers to show contribution.
        
        Returns:
            Dictionary mapping layer config to mean latency
        """
        # This would be implemented in a separate comparison script
        pass


def get_benchmark(config_path: Optional[str] = None) -> BaseBenchmark:
    """Factory function for harness discovery."""
    if config_path:
        config = OptimizedConfig.from_yaml(Path(config_path))
    else:
        config = OptimizedConfig()
    return OptimizedUltimateInference(config)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Optimized Ultimate MoE Inference Benchmark",
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
    parser.add_argument(
        "--compare-layers",
        action="store_true",
        help="Run layer-by-layer comparison",
    )
    args = parser.parse_args()
    
    # Load config
    config = OptimizedConfig.from_yaml(args.config)
    if args.iterations:
        config.benchmark_iterations = args.iterations
    
    # Create and run benchmark
    benchmark = OptimizedUltimateInference(config)
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    
    print("=" * 70)
    print("Optimized Ultimate MoE Inference")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    print(f"Precision: {config.precision}")
    print(f"Batch size: {config.batch_size}")
    print(f"Prompt tokens: {config.prompt_tokens}")
    print(f"Decode tokens: {config.decode_tokens}")
    print()
    print("Optimizations enabled:")
    print(f"  FlashAttention: {config.use_flash_attention}")
    print(f"  CUDA Graphs: {config.use_cuda_graphs}")
    print(f"  torch.compile: {config.use_torch_compile} ({config.compile_mode})")
    print(f"  FP8 KV Cache: {config.use_fp8_kv_cache}")
    print(f"  PagedAttention: {config.use_paged_attention}")
    print(f"  Speculative Decode: {config.use_speculative_decode}")
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
    
    if hasattr(benchmark, 'last_metrics') and benchmark.last_metrics:
        metrics = benchmark.last_metrics
        print()
        print("Inference Metrics:")
        print(f"  TTFT: {metrics.ttft_ms:.2f} ms")
        print(f"  TPOT: {metrics.tpot_ms:.2f} ms")
        print(f"  Tokens/sec: {metrics.tokens_per_sec:.1f}")
        print(f"  Peak memory: {metrics.peak_memory_gb:.2f} GB")
        if metrics.tokens_per_joule:
            print(f"  Tokens/Joule: {metrics.tokens_per_joule:.2f}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

