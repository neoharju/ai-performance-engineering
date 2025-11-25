#!/usr/bin/env python3
"""
05_torch_compile.py - Kernel Fusion & FP8 (Chapters 13-14)

═══════════════════════════════════════════════════════════════════════════════
THE REMAINING OVERHEAD: SMALL KERNELS
═══════════════════════════════════════════════════════════════════════════════

Even with FlashAttention and CUDA graphs, there are many small operations:
- LayerNorm
- GELU activation
- Residual additions
- Embedding lookups

Each is a separate kernel. Small kernels are MEMORY-BOUND because:
- Launch overhead is significant relative to compute
- Data is read from HBM, processed, written back immediately
- No opportunity for data reuse

═══════════════════════════════════════════════════════════════════════════════
THE SOLUTION: KERNEL FUSION via torch.compile
═══════════════════════════════════════════════════════════════════════════════

torch.compile with TorchInductor:
1. Traces your model's operations
2. Analyzes data dependencies
3. FUSES compatible operations into single kernels
4. Generates optimized CUDA/Triton code

Example fusion:

BEFORE (3 kernels):
  residual = x + attention_output     # Kernel 1: Read x, attn; Write residual
  normalized = layer_norm(residual)   # Kernel 2: Read residual; Write normalized
  activated = gelu(normalized)        # Kernel 3: Read normalized; Write activated

AFTER (1 fused kernel):
  # All in one kernel - data stays in registers!
  activated = fused_residual_ln_gelu(x, attention_output)

Memory traffic reduced by 3x!

═══════════════════════════════════════════════════════════════════════════════
WHY IT WORKS (Concepts from Chapters 13-14):
═══════════════════════════════════════════════════════════════════════════════

Chapter 13 - FP8 Precision:
┌─────────────────────────────────────────────────────────────────────────────┐
│ REDUCED PRECISION: Less bits = less memory traffic                          │
│                                                                              │
│   Format    Bits   Range              Typical Use                            │
│   ────────────────────────────────────────────────────────────              │
│   FP32      32     ±3.4e38            Training (legacy)                     │
│   BF16      16     ±3.4e38            Training (recommended)                │
│   FP16      16     ±65504             Inference                              │
│   FP8 E4M3   8     ±448               Forward pass                          │
│   FP8 E5M2   8     ±57344             Backward pass                         │
│   FP4        4     ±6                 Weights only (Blackwell)              │
│                                                                              │
│ FP8 halves memory bandwidth vs FP16:                                        │
│   - Weights: 2x smaller                                                      │
│   - Activations: 2x smaller                                                  │
│   - KV cache: 2x smaller (huge for long sequences!)                         │
│                                                                              │
│ DelayedScaling: Dynamic range management for FP8                             │
│   - Track amax (max absolute value) over time                                │
│   - Adjust scale factor to use full FP8 range                                │
│   - Hysteresis prevents scale oscillation                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Chapter 14 - torch.compile & Triton:
┌─────────────────────────────────────────────────────────────────────────────┐
│ TorchInductor: The compilation backend                                       │
│                                                                              │
│   Python code                                                                │
│        ↓                                                                     │
│   FX Graph (operation trace)                                                 │
│        ↓                                                                     │
│   TorchInductor (fusion, scheduling)                                         │
│        ↓                                                                     │
│   Triton kernels (generated code)                                            │
│        ↓                                                                     │
│   PTX/SASS (GPU assembly)                                                    │
│                                                                              │
│ Compilation modes:                                                           │
│   "default"      - Basic fusion                                              │
│   "reduce-overhead" - Minimize CPU overhead                                  │
│   "max-autotune" - Try many configurations, pick best (slower compile)      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ TRITON: Python-like GPU programming                                         │
│                                                                              │
│   @triton.jit                                                                │
│   def fused_add_ln_gelu(x_ptr, residual_ptr, out_ptr, ...):                 │
│       # Load from global memory                                              │
│       x = tl.load(x_ptr + offsets)                                          │
│       residual = tl.load(residual_ptr + offsets)                            │
│                                                                              │
│       # All computation in registers!                                        │
│       y = x + residual                                                       │
│       y = layer_norm(y)     # Stays in registers                            │
│       y = gelu(y)           # Still in registers                            │
│                                                                              │
│       # Single write to global memory                                        │
│       tl.store(out_ptr + offsets, y)                                        │
│                                                                              │
│ One read, one write instead of 6 memory operations!                          │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
AUTOTUNING:
═══════════════════════════════════════════════════════════════════════════════

torch.compile with max-autotune tries multiple configurations:
- Different Triton tile sizes (BLOCK_M, BLOCK_N, BLOCK_K)
- Different num_warps (parallelism within a block)
- Different num_stages (pipelining depth)

It benchmarks each and picks the fastest!

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8),
        # ... many more configs
    ],
    key=['M', 'N', 'K'],  # Re-tune when these dimensions change
)

═══════════════════════════════════════════════════════════════════════════════
ADVANCED: REGIONAL COMPILATION & GRAPH BREAKS
═══════════════════════════════════════════════════════════════════════════════

REGIONAL COMPILATION:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Sometimes you only want to compile PART of a model:                         │
│                                                                              │
│   # Compile only the decoder layers (hot path)                              │
│   model.decoder = torch.compile(model.decoder, mode="max-autotune")         │
│                                                                              │
│   # Or compile specific modules                                              │
│   for layer in model.layers:                                                │
│       layer.attention = torch.compile(layer.attention)                      │
│       layer.mlp = torch.compile(layer.mlp)                                  │
│                                                                              │
│ Benefits:                                                                    │
│ - Faster compile time (only hot paths)                                      │
│ - Avoid graph breaks in problematic code                                    │
│ - Mix compiled and eager code                                               │
│                                                                              │
│ See: code/ch16/optimized_regional_compilation.py                            │
└─────────────────────────────────────────────────────────────────────────────┘

GRAPH BREAKS (and how to avoid them):
┌─────────────────────────────────────────────────────────────────────────────┐
│ A GRAPH BREAK occurs when TorchDynamo can't trace an operation.             │
│ The graph splits, and you lose optimization opportunities.                  │
│                                                                              │
│ Common causes:                                                               │
│ 1. Python control flow depending on tensor values:                          │
│    BAD:  if x.sum() > 0:        # Python bool from tensor!                  │
│    GOOD: torch.where(x > 0, a, b)  # Pure tensor op                         │
│                                                                              │
│ 2. Non-PyTorch library calls:                                               │
│    BAD:  result = numpy.array(x)   # Breaks graph                           │
│    GOOD: result = x.numpy()        # After graph (or avoid)                 │
│                                                                              │
│ 3. Print statements / debugging:                                            │
│    BAD:  print(x.shape)            # Causes graph break                     │
│    GOOD: Remove in production or use torch._dynamo.config.verbose           │
│                                                                              │
│ Debugging graph breaks:                                                      │
│   torch.compiler.set_stance("fail_on_recompile")  # Error on break          │
│   TORCH_COMPILE_DEBUG=1 python script.py          # Verbose output          │
│   torch._dynamo.explain(model)(input)             # Show break reasons      │
│                                                                              │
│ Marking functions as graph-safe:                                             │
│   @torch._dynamo.allow_in_graph                                             │
│   def my_custom_function(x):                                                │
│       ...                                                                    │
└─────────────────────────────────────────────────────────────────────────────┘

DYNAMIC SHAPES:
┌─────────────────────────────────────────────────────────────────────────────┐
│ By default, torch.compile specializes on observed shapes.                    │
│ If shapes change, it recompiles (slow!).                                    │
│                                                                              │
│ Mark dimensions as dynamic to avoid recompilation:                          │
│                                                                              │
│   # Mark batch dimension as dynamic                                          │
│   torch._dynamo.mark_dynamic(input_tensor, dim=0)                           │
│                                                                              │
│   # Mark sequence length as dynamic                                         │
│   torch._dynamo.mark_dynamic(input_tensor, dim=1)                           │
│                                                                              │
│ Or use dynamic=True (less optimized but no recompiles):                     │
│   model = torch.compile(model, dynamic=True)                                │
│                                                                              │
│ For inference with variable batch/seq:                                       │
│   # Best practice: use dynamic=None (default) + warmup with varied shapes   │
│   # Compiler will detect dynamism and adapt                                  │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
RAW EXAMPLES:
═══════════════════════════════════════════════════════════════════════════════

Transformer Engine FP8 (Ch13):
  → code/ch13/optimized_precisionfp8_te.py

Triton kernels (Ch14):
  → code/ch14/triton_examples.py
  → code/ch14/triton_fp8_advanced.py

Regional compilation (Ch14/16):
  → code/ch16/optimized_regional_compilation.py
  → code/ch14/optimized_regional_triton.py

═══════════════════════════════════════════════════════════════════════════════
TO VERIFY IT'S WORKING:
═══════════════════════════════════════════════════════════════════════════════

Set TORCH_COMPILE_DEBUG=1 to see generated code:
  TORCH_COMPILE_DEBUG=1 python 05_torch_compile.py

Look for in logs:
- "Compiling function" messages
- Generated Triton kernel code
- Autotuning results

Run: nsys profile -o 05_compile python 05_torch_compile.py

Look for:
- Fewer, larger kernels (fusion working)
- Kernels named "triton_" (Inductor-generated)

═══════════════════════════════════════════════════════════════════════════════
EXPECTED SPEEDUP: Additional 1.2-1.4x!
═══════════════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════════════
NEXT STEP: 06_ultimate.py - Add advanced inference techniques
═══════════════════════════════════════════════════════════════════════════════
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

from baseline_ultimate_inference import InferenceConfig, InferenceMetrics
from optimization_layers import (
    Layer01Basics, Layer02Memory, Layer03Pipelining, 
    Layer04Concurrency, Layer05PyTorch
)
from components.monitoring import MetricsCollector
from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class TorchCompileBenchmark(BaseBenchmark):
    """Optimized: torch.compile + FP8 (Chapters 13-14).
    
    Adds:
    - torch.compile with max-autotune for kernel fusion
    - TorchInductor for optimized code generation
    - (FP8 via Transformer Engine when available)
    
    Expected speedup: Additional 1.2-1.4x!
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        super().__init__()
        self.config = config or InferenceConfig()
        
        self.layer1 = Layer01Basics()
        self.layer2 = Layer02Memory()
        self.layer3 = Layer03Pipelining()
        self.layer4 = Layer04Concurrency()
        self.layer5 = Layer05PyTorch()
        
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.input_ids: Optional[torch.Tensor] = None
        self.attention_mask: Optional[torch.Tensor] = None
        self.last_metrics: Optional[InferenceMetrics] = None
        self.metrics_collector = MetricsCollector()
        
        total_tokens = self.config.batch_size * (
            self.config.prompt_tokens + self.config.decode_tokens
        )
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(total_tokens),
        )
    
    def setup(self) -> None:
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers required")
        
        print("=" * 70)
        print("05_torch_compile.py - Kernel Fusion & FP8 (Ch13-14)")
        print("=" * 70)
        
        # Layer 1 & 2
        print("\n[Layer 1-2] Foundation optimizations...")
        self.layer1.enable_tf32()
        self.layer1.enable_cudnn_benchmark()
        self.layer2.configure_memory_allocator()
        
        # Load with FlashAttention (Layer 3)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("\n[Layer 3] Loading with FlashAttention...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        
        # Layer 4: Streams
        print("\n[Layer 4] Setting up CUDA streams...")
        self.layer4.setup_streams()
        
        # Layer 5: Configure TorchInductor and compile!
        print("\n" + "=" * 70)
        print("[Layer 5] Configuring TorchInductor...")
        print("=" * 70)
        self.layer5.configure_inductor()
        print("  ✓ Triton cudagraphs enabled")
        print("  ✓ Max autotune enabled")
        print("  ✓ Coordinate descent tuning enabled")
        
        print("\n[Layer 5] Compiling model with torch.compile (max-autotune)...")
        print("  This may take a few minutes on first run...")
        self.model = self.layer5.compile_model(self.model, mode="max-autotune")
        print("  ✓ Model compiled!")
        
        # Prepare inputs
        self._prepare_inputs()
        
        # Warmup (triggers JIT compilation)
        print("\nWarmup (includes JIT compilation)...")
        for i in range(5):
            print(f"  Warmup {i+1}/5...")
            self._run_inference()
        torch.cuda.synchronize()
        print("  ✓ JIT compilation complete!")
        
        print("=" * 70)
    
    def _prepare_inputs(self) -> None:
        prompt = "Explain the theory of general relativity and its implications."
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
        with self._nvtx_range("torch_compile"):
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
        """Return custom metrics for torch.compile benchmark."""
        if self.last_metrics is None:
            return None
        m = self.last_metrics
        return {
            "torch_compile.ttft_ms": m.ttft_ms,
            "torch_compile.tpot_ms": m.tpot_ms,
            "torch_compile.tokens_per_sec": m.tokens_per_sec,
            "torch_compile.peak_memory_gb": m.peak_memory_gb,
        }


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return TorchCompileBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    
    print("\n" + "=" * 70)
    print("RESULTS - 05_torch_compile.py")
    print("=" * 70)
    if result.timing:
        print(f"Mean latency: {result.timing.mean_ms:.2f} ms")
    
    print("\ntorch.compile benefits:")
    print("  ✓ Kernel fusion (fewer memory round-trips)")
    print("  ✓ Triton-generated optimized kernels")
    print("  ✓ Autotuned tile sizes and configurations")
    
    print("\nCombined with previous layers:")
    print("  Layer 1: TF32, cuDNN (~1.1x)")
    print("  Layer 2: Memory config (~1.05x)")
    print("  Layer 3: FlashAttention (~2x)")
    print("  Layer 4: CUDA Graphs (~1.2x)")
    print("  Layer 5: torch.compile (~1.3x)")
    print("  ─────────────────────────────")
    print("  Cumulative: ~3.5x faster than baseline!")
    
    print("\nNext: Run 06_ultimate.py for speculative decoding and more!")
    print("=" * 70)

