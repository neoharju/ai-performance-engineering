#!/usr/bin/env python3
"""
03_flash_attention.py - The Big Win! (Chapters 9-10)

═══════════════════════════════════════════════════════════════════════════════
THE BREAKTHROUGH: TILED ATTENTION
═══════════════════════════════════════════════════════════════════════════════

FlashAttention solves the O(n²) memory problem by:

1. NEVER materializing the full attention matrix
2. Processing in TILES that fit in shared memory
3. Computing softmax INCREMENTALLY (online softmax)

Result: O(n) memory instead of O(n²), and 2-3x faster!

═══════════════════════════════════════════════════════════════════════════════
BEFORE (Standard Attention):
═══════════════════════════════════════════════════════════════════════════════

    # Compute full attention matrix (O(n²) memory!)
    scores = Q @ K.T / sqrt(d)          # [n, n] - Written to HBM
    probs = softmax(scores, dim=-1)     # [n, n] - Read + Write to HBM
    output = probs @ V                  # Read from HBM
    
Memory: O(n²)
For n=4096: 64 MB per head

═══════════════════════════════════════════════════════════════════════════════
AFTER (FlashAttention - Tiled):
═══════════════════════════════════════════════════════════════════════════════

    # Process in tiles that fit in shared memory
    for tile_q in tiles(Q):                    # Outer loop over Q tiles
        for tile_k, tile_v in tiles(K, V):     # Inner loop over K,V tiles
            # Compute partial scores (fits in SRAM)
            partial_scores = tile_q @ tile_k.T
            
            # Online softmax update (the magic!)
            new_max = max(running_max, partial_scores.max())
            scale = exp(running_max - new_max)
            running_sum = scale * running_sum + exp(partial_scores - new_max).sum()
            running_max = new_max
            
            # Accumulate output
            output += (exp(partial_scores - new_max) / running_sum) @ tile_v
            
Memory: O(n) - Only store running statistics!
For n=4096: ~4 KB per head (vs 64 MB)

═══════════════════════════════════════════════════════════════════════════════
WHY IT WORKS (Concepts from Chapters 9-10):
═══════════════════════════════════════════════════════════════════════════════

Chapter 9 - Tiling:
┌─────────────────────────────────────────────────────────────────────────────┐
│ TILING: Break large matrices into blocks that fit in fast memory            │
│                                                                              │
│   Global Memory (slow):     Shared Memory (fast):                           │
│   ┌───────────────────┐     ┌─────────┐                                     │
│   │ Full 4096×4096    │ --> │ 64×64   │ Process one tile at a time          │
│   │ attention matrix  │     │ tile    │ 16 KB fits in 48 KB SRAM!           │
│   └───────────────────┘     └─────────┘                                     │
│                                                                              │
│ Each tile: 64 × 64 × 4 bytes = 16 KB (fits in shared memory)                │
│ Total tiles: (4096/64)² = 4096 tiles, processed sequentially                │
│ Memory traffic: Read Q,K,V once, never write intermediate results!          │
└─────────────────────────────────────────────────────────────────────────────┘

Chapter 10 - Pipelining & Double Buffering:
┌─────────────────────────────────────────────────────────────────────────────┐
│ DOUBLE BUFFERING: Overlap load and compute                                   │
│                                                                              │
│ Time: ─────────────────────────────────────────────────────────────>        │
│                                                                              │
│ Naive:     [Load T1][Compute T1][Load T2][Compute T2][Load T3]...           │
│                                                                              │
│ Double-    [Load T1][Load T2   ][Load T3   ][Load T4   ]...                 │
│ Buffer:            [Compute T1 ][Compute T2 ][Compute T3 ]...               │
│                                                                              │
│ Compute and load happen IN PARALLEL - hides memory latency!                 │
└─────────────────────────────────────────────────────────────────────────────┘

Chapter 10 - Online Softmax:
┌─────────────────────────────────────────────────────────────────────────────┐
│ ONLINE SOFTMAX: Compute softmax WITHOUT seeing all values first             │
│                                                                              │
│ Problem: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))                │
│          Normally need ALL values to find max and sum!                       │
│                                                                              │
│ Solution: Track running max (m) and sum (l) as we process tiles             │
│                                                                              │
│   For each new tile:                                                         │
│   1. new_m = max(m, tile_max)                                               │
│   2. scale_old = exp(m - new_m)                                             │
│   3. scale_new = exp(tile_max - new_m)                                      │
│   4. new_l = scale_old * l + scale_new * tile_sum                           │
│   5. output = scale_old * output + scale_new * tile_output / new_l          │
│                                                                              │
│ Magic: We compute EXACT softmax by updating statistics incrementally!       │
└─────────────────────────────────────────────────────────────────────────────┘

Chapter 10 - CUDA Pipeline API (The key to double buffering):
┌─────────────────────────────────────────────────────────────────────────────┐
│ The CUDA Pipeline API enables fine-grained producer-consumer patterns:      │
│                                                                              │
│   #include <cuda/pipeline>                                                   │
│   cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipe_state;      │
│   auto pipe = cuda::make_pipeline(cta, &pipe_state);                        │
│                                                                              │
│ Key API calls:                                                               │
│                                                                              │
│   pipe.producer_acquire()  // Reserve next stage for writing                │
│   cuda::memcpy_async(...)  // Async copy to shared memory (cp.async)        │
│   pipe.producer_commit()   // Signal: "data is ready for consumers"         │
│                                                                              │
│   pipe.consumer_wait()     // Wait for committed data                       │
│   // ... use the data ...                                                    │
│   pipe.consumer_release()  // Release stage for reuse                       │
│                                                                              │
│ The "2" in pipeline_shared_state<..., 2> = two stages = double buffer!      │
│ Stage 0: being loaded (producer)                                             │
│ Stage 1: being computed (consumer)                                           │
│ They swap each iteration.                                                    │
│                                                                              │
│ Example: code/ch10/optimized_double_buffered_pipeline.py                    │
└─────────────────────────────────────────────────────────────────────────────┘

Chapter 10 - Warp Specialization (Advanced):
┌─────────────────────────────────────────────────────────────────────────────┐
│ WARP SPECIALIZATION: Different warps do different jobs                      │
│                                                                              │
│   Block of 256 threads = 8 warps                                            │
│                                                                              │
│   Warp 0-1: LOADER   - Fetch next K,V tiles from HBM via TMA                │
│   Warp 2-5: COMPUTE  - Matrix multiply Q @ K.T, softmax, @ V                │
│   Warp 6-7: STORER   - Write output tiles back to HBM                       │
│                                                                              │
│ Key insight: No __syncthreads() needed between different roles!             │
│ Loader warps use pipeline API to signal when data is ready.                 │
│ Compute warps wait on that signal, then process.                            │
│ Storer warps wait for compute, then write back.                             │
│                                                                              │
│ Implementation pattern (simplified):                                         │
│                                                                              │
│   int warp_id = threadIdx.x / 32;                                           │
│   if (warp_id < 2) {                                                        │
│       // LOADER: fetch tiles via TMA                                        │
│       pipe.producer_acquire();                                              │
│       cuda::memcpy_async(smem, global, shape, pipe);                        │
│       pipe.producer_commit();                                               │
│   } else if (warp_id < 6) {                                                 │
│       // COMPUTE: matrix multiply + softmax                                  │
│       pipe.consumer_wait();                                                 │
│       output = compute_tile(smem_q, smem_k, smem_v);                        │
│       pipe.consumer_release();                                              │
│   } else {                                                                  │
│       // STORER: write results back                                          │
│       write_output(global_out, output);                                     │
│   }                                                                         │
│                                                                              │
│ Example: code/ch10/optimized_warp_specialization_pipeline.py                │
└─────────────────────────────────────────────────────────────────────────────┘

Chapter 10 - Persistent Kernels:
┌─────────────────────────────────────────────────────────────────────────────┐
│ PERSISTENT KERNEL: One kernel that processes multiple work items            │
│                                                                              │
│   Traditional: Launch kernel per tile → lots of launch overhead             │
│   ┌────┐ gap ┌────┐ gap ┌────┐ gap ┌────┐                                   │
│   │ K1 │────>│ K2 │────>│ K3 │────>│ K4 │                                   │
│   └────┘     └────┘     └────┘     └────┘                                   │
│                                                                              │
│   Persistent: One kernel loops over all work                                 │
│   ┌────────────────────────────────────────┐                                │
│   │ K: loop { fetch_work(); process(); }   │                                │
│   └────────────────────────────────────────┘                                │
│                                                                              │
│ Implementation:                                                              │
│   __device__ int work_counter = 0;                                          │
│   __global__ void persistent_kernel(...) {                                  │
│       while (true) {                                                        │
│           int tile = atomicAdd(&work_counter, 1);                           │
│           if (tile >= num_tiles) return;                                    │
│           process_tile(tile);                                               │
│       }                                                                     │
│   }                                                                         │
│                                                                              │
│ FlashAttention-3 uses persistent kernels + warp specialization!             │
│ Example: code/ch10/optimized_cooperative_persistent.py                      │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
BLACKWELL-SPECIFIC: TMA & TMEM
═══════════════════════════════════════════════════════════════════════════════

TMA (Tensor Memory Accelerator):
- Hardware unit for async bulk memory transfers
- Loads directly into shared memory, no register staging
- Programmer specifies tensor layout, TMA handles addressing
- Code: cuda::memcpy_async() in ch10/optimized_warp_specialized_pipeline.cu

TMEM (Tensor Memory):
- Special memory for 2-CTA clusters on Blackwell
- Shared between thread block clusters
- Enables larger effective shared memory pools
- Used by cuBLAS/FlashAttention automatically

═══════════════════════════════════════════════════════════════════════════════
RAW CUDA EXAMPLES:
═══════════════════════════════════════════════════════════════════════════════

Tiling basics (Ch9):
  → code/ch9/baseline_cutlass_gemm.cu vs optimized_cutlass_gemm.cu

Double buffering (Ch10):
  → code/ch8/optimized_double_buffering_pipelined.cu

Warp specialization (Ch10):
  → code/ch10/baseline_warp_specialized_pipeline.cu
  → code/ch10/optimized_warp_specialized_pipeline.cu

TMA copy (Ch10):
  → code/ch7/optimized_tma_copy.cu

═══════════════════════════════════════════════════════════════════════════════
TO VERIFY IT'S WORKING:
═══════════════════════════════════════════════════════════════════════════════

Run: nsys profile -o 03_flash python 03_flash_attention.py

Look for:
- Attention kernels MUCH shorter
- Kernel names containing "flash" or "mem_efficient"
- Less DRAM traffic between attention ops

Run: ncu --set full --kernel-name ".*flash.*" python 03_flash_attention.py

Look for:
- Higher compute utilization (sm__pipe_tensor_cycles_active)
- Lower memory:compute ratio
- Good occupancy (>50%)

═══════════════════════════════════════════════════════════════════════════════
EXPECTED SPEEDUP: 2-3x over baseline!
═══════════════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════════════
NEXT STEP: 04_cuda_graphs.py - Eliminate remaining overhead
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
from optimization_layers import Layer01Basics, Layer02Memory, Layer03Pipelining
from components.monitoring import MetricsCollector
from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class FlashAttentionBenchmark(BaseBenchmark):
    """Optimized: FlashAttention via SDPA (Chapters 9-10).
    
    This is THE BIG WIN! FlashAttention uses all the concepts from Ch9-10:
    - Tiling to fit in shared memory
    - Online softmax for incremental computation
    - Double buffering to overlap load/compute
    - Warp specialization for parallel roles
    
    Expected speedup: 2-3x over baseline!
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
        print("03_flash_attention.py - The Big Win! (Ch9-10)")
        print("=" * 70)
        
        # Layer 1 & 2
        print("\n[Layer 1-2] Applying foundation optimizations...")
        self.layer1.enable_tf32()
        self.layer1.enable_cudnn_benchmark()
        self.layer2.configure_memory_allocator()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Layer 3: THE KEY CHANGE - Load with FlashAttention!
        print("\n" + "=" * 70)
        print("[Layer 3] Loading model with FLASH ATTENTION!")
        print("=" * 70)
        print("  → attn_implementation='flash_attention_2'")
        print("  → Tiled computation in SRAM")
        print("  → O(n) memory instead of O(n²)")
        print("  → Online softmax for incremental computation")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # THE MAGIC LINE!
        )
        self.model.eval()
        
        # Show what's happening
        print("\n" + "-" * 70)
        print("MEMORY COMPARISON:")
        print("-" * 70)
        seq_len = self.config.prompt_tokens
        standard_mem_mb = (seq_len ** 2 * 4 * 32) / 1e6  # 32 heads, FP32
        flash_mem_mb = (seq_len * 64 * 4 * 32) / 1e6  # 64 = tile size approx
        print(f"  Standard attention: {standard_mem_mb:.1f} MB per layer")
        print(f"  Flash attention:    ~{flash_mem_mb:.1f} MB per layer")
        print(f"  Reduction:          {standard_mem_mb / flash_mem_mb:.0f}x less memory!")
        print("-" * 70)
        
        # Prepare inputs
        self._prepare_inputs()
        
        # Warmup
        print("\nWarmup runs...")
        for _ in range(3):
            self._run_inference()
        torch.cuda.synchronize()
        
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
        with self._nvtx_range("flash_attention"):
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
        """Return custom metrics for flash attention benchmark."""
        if self.last_metrics is None:
            return None
        m = self.last_metrics
        return {
            "flash_attention.ttft_ms": m.ttft_ms,
            "flash_attention.tpot_ms": m.tpot_ms,
            "flash_attention.tokens_per_sec": m.tokens_per_sec,
            "flash_attention.peak_memory_gb": m.peak_memory_gb,
        }


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return FlashAttentionBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    
    print("\n" + "=" * 70)
    print("RESULTS - 03_flash_attention.py")
    print("=" * 70)
    if result.timing:
        print(f"Mean latency: {result.timing.mean_ms:.2f} ms")
        print(f"Expected: ~2-3x faster than baseline!")
    
    if benchmark.last_metrics:
        print(f"\nThroughput: {benchmark.last_metrics.tokens_per_sec:.1f} tokens/sec")
    
    print("\nThis is the BIG win! FlashAttention applies all Ch9-10 concepts:")
    print("  ✓ Tiling (process 64×64 blocks)")
    print("  ✓ Double buffering (overlap load/compute)")
    print("  ✓ Online softmax (incremental computation)")
    print("  ✓ Warp specialization (loader/compute/storer)")
    
    print("\nNext: Run 04_cuda_graphs.py to eliminate launch overhead")
    print("=" * 70)

