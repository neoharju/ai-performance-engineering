"""Optimized Layer 3: Pipelining (Ch9-10) - FlashAttention, double buffering.

Incremental optimization showing Layers 1-3 contribution:
- Layer 1: TF32, cuDNN, NUMA
- Layer 2: Memory layout (implicit via model config)
- Layer 3: FlashAttention, double buffering

This demonstrates the intra-kernel pipelining techniques from Chapters 9-10.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from baseline_ultimate_inference import InferenceConfig
from optimization_layers import Layer01Basics, Layer02Memory, Layer03Pipelining
from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from components.monitoring import MetricsCollector, InferenceMetrics


class OptimizedLayer3Benchmark(BaseBenchmark):
    """Optimized: Layers 1-3 (Ch1-10 basics + pipelining).
    
    Adds pipelining optimizations:
    - FlashAttention via SDPA
    - Double buffering streams
    
    Expected improvement: 1.5-2x from FlashAttention alone
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        super().__init__()
        self.config = config or InferenceConfig()
        
        self.layer1 = Layer01Basics()
        self.layer2 = Layer02Memory()
        self.layer3 = Layer03Pipelining()
        
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.input_ids: Optional[torch.Tensor] = None
        self.attention_mask: Optional[torch.Tensor] = None
        self.last_metrics: Optional[InferenceMetrics] = None
        self.metrics_collector = MetricsCollector()
        
        total_tokens = self.config.batch_size * (self.config.prompt_tokens + self.config.decode_tokens)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(total_tokens),
        )
    
    def setup(self) -> None:
        """Setup with Layers 1-3 optimizations."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers required")
        
        # Layer 1
        print("[Layer 1] Applying basic optimizations...")
        self.layer1.enable_tf32()
        self.layer1.enable_cudnn_benchmark()
        self.layer1.bind_numa(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Layer 3: Enable FlashAttention during model load
        print("[Layer 3] Loading model with FlashAttention...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # Layer 3 optimization
        )
        self.model.eval()
        
        # Layer 2
        print("[Layer 2] Applying memory optimizations...")
        self.layer2.apply(self)
        
        # Layer 3 streams
        print("[Layer 3] Setting up double buffering...")
        self.layer3.setup_double_buffering()
        
        # Prepare inputs
        self._prepare_inputs()
        
        # Warmup
        print("Running warmup...")
        for _ in range(3):
            self._run_inference()
        torch.cuda.synchronize()
    
    def _prepare_inputs(self) -> None:
        prompt = "Explain the future of artificial intelligence."
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
        with torch.no_grad():
            # Use FlashAttention backend
            with self.layer3.flash_attention_backend():
                outputs = self.model.generate(
                    self.input_ids,
                    attention_mask=self.attention_mask,
                    max_new_tokens=self.config.decode_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
        return outputs
    
    def benchmark_fn(self) -> None:
        with self._nvtx_range("layer3_pipelining"):
            self.metrics_collector.reset()
            self.metrics_collector.start_request("main")
            
            outputs = self._run_inference()
            
            self.metrics_collector.record_first_token("main")
            output_tokens = outputs.shape[1] - self.input_ids.shape[1]
            self.metrics_collector.end_request(
                "main",
                self.config.prompt_tokens * self.config.batch_size,
                output_tokens * self.config.batch_size,
            )
            self.last_metrics = self.metrics_collector.compute_metrics()
        self._synchronize()
    
    def teardown(self) -> None:
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=3)
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        """Return custom metrics for layer 3 pipelining benchmark."""
        if self.last_metrics is None:
            return None
        m = self.last_metrics
        return {
            "layer3_pipelining.ttft_ms": m.ttft_ms,
            "layer3_pipelining.tpot_ms": m.tpot_ms,
            "layer3_pipelining.tokens_per_sec": m.tokens_per_sec,
            "layer3_pipelining.peak_memory_gb": m.peak_memory_gb,
        }


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedLayer3Benchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    
    print(f"\nLayers 1-3 (Ch1-10): {result.timing.mean_ms:.2f} ms" if result.timing else "No timing")

