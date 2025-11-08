"""optimized_paged_attention.py - Optimized paged attention.

Demonstrates paged attention for efficient KV cache management.
Paged attention: Uses non-contiguous pages for efficient memory management.
Reduces fragmentation and improves memory utilization for variable-length sequences.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.python.compile_utils import enable_tf32

from typing import Optional, List, Tuple

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")

class PagedKVCache:
    """Paged KV cache - non-contiguous page-based storage."""
    
    def __init__(self, batch_size: int, page_size: int, num_heads: int, head_dim: int, device: torch.device):
        self.batch_size = batch_size
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.k_pages: List[torch.Tensor] = []
        self.v_pages: List[torch.Tensor] = []
        self.page_map: List[int] = []  # Maps sequence positions to page indices
    
    def allocate_page(self) -> int:
        """Allocate a new page and return its index."""
        page_shape = (self.batch_size, self.page_size, self.num_heads, self.head_dim)
        k_page = torch.zeros(page_shape, dtype=torch.float32, device=self.device)
        v_page = torch.zeros(page_shape, dtype=torch.float32, device=self.device)
        self.k_pages.append(k_page)
        self.v_pages.append(v_page)
        return len(self.k_pages) - 1
    
    def write(self, pos: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write K/V to cache at position using paged storage."""
        page_idx = pos // self.page_size
        offset = pos % self.page_size
        
        while len(self.k_pages) <= page_idx:
            self.allocate_page()
        
        # Ensure tensors are on the correct device/dtype and drop the seq dimension
        k_batch = k[:, 0, :, :].to(self.device, dtype=torch.float32)
        v_batch = v[:, 0, :, :].to(self.device, dtype=torch.float32)
        self.k_pages[page_idx][:, offset, :, :] = k_batch
        self.v_pages[page_idx][:, offset, :, :] = v_batch
        self.page_map.append(page_idx)
    
    def get_kv(self, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get K/V up to length, reconstructing from pages."""
        if length == 0 or not self.k_pages:
            empty = torch.empty(self.batch_size, 0, self.num_heads, self.head_dim, device=self.device)
            return empty, empty
        
        k_list = []
        v_list = []
        
        for pos in range(length):
            page_idx = self.page_map[pos] if pos < len(self.page_map) else 0
            offset = pos % self.page_size
            k_list.append(self.k_pages[page_idx][:, offset:offset+1, :, :])
            v_list.append(self.v_pages[page_idx][:, offset:offset+1, :, :])
        
        return torch.cat(k_list, dim=1), torch.cat(v_list, dim=1)

class OptimizedPagedAttentionBenchmark(Benchmark):
    """Optimized: Paged attention for efficient KV cache management.
    
    Paged attention: Uses non-contiguous pages for efficient memory management.
    Reduces fragmentation and improves memory utilization for variable-length sequences.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization

        # Optimization: Compile model for kernel fusion and optimization

        self.kv_cache = None
        self.inputs = None
        self.hidden_dim = 256
        self.num_heads = 8
        self.head_dim = self.hidden_dim // self.num_heads
        self.batch_size = 2
        self.page_size = 16
    
    def setup(self) -> None:
        """Setup: Initialize model and paged KV cache."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            enable_tf32()
        torch.manual_seed(42)
        # Optimization: Paged attention - non-contiguous page-based storage
        # Paged attention uses pages for efficient memory management
        
        # Simple attention model
        self.model = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True
        ).to(self.device).eval()
        
        # Optimization: Paged KV cache (paged attention)
        # Uses non-contiguous pages for efficient memory management
        self.kv_cache = PagedKVCache(self.batch_size, self.page_size, self.num_heads, self.head_dim, self.device)
        
        # Simulate autoregressive generation
        self.inputs = [
            torch.randn(self.batch_size, 1, self.hidden_dim, device=self.device)
            for _ in range(32)
        ]
        torch.cuda.synchronize()
    
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape (B, T, hidden) into (B, heads, T, head_dim)."""
        batch, seq_len, _ = tensor.shape
        return tensor.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
    
    def _project_qkv(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project inputs into Q, K, V using the MHA weights."""
        qkv = F.linear(query, self.model.in_proj_weight, self.model.in_proj_bias)
        return qkv.chunk(3, dim=-1)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Paged attention."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_paged_attention", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Paged attention
                # Uses non-contiguous pages for efficient memory management
                # Reduces fragmentation and improves memory utilization
                
                for step, query in enumerate(self.inputs):
                    q_proj, k_proj, v_proj = self._project_qkv(query)
                    q_heads = self._split_heads(q_proj)
                    k_new = self._split_heads(k_proj).permute(0, 2, 1, 3)
                    v_new = self._split_heads(v_proj).permute(0, 2, 1, 3)
                    
                    # Store in paged cache (paged attention)
                    self.kv_cache.write(step, k_new, v_new)
                    
                    # Retrieve K, V from pages (paged attention reconstruction)
                    k_all, v_all = self.kv_cache.get_kv(step + 1)
                    if k_all.shape[1] == 0:
                        continue
                    
                    k_heads = k_all.permute(0, 2, 1, 3).contiguous()
                    v_heads = v_all.permute(0, 2, 1, 3).contiguous()
                    
                    attn_scores = torch.matmul(
                        q_heads, k_heads.transpose(-2, -1)
                    ) / math.sqrt(self.head_dim)
                    attn_probs = torch.softmax(attn_scores, dim=-1)
                    context = torch.matmul(attn_probs, v_heads)
                    
                    _ = context.permute(0, 2, 1, 3).reshape(query.size(0), 1, self.hidden_dim)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.kv_cache = None
        self.inputs = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.kv_cache is None:
            return "KV cache not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedPagedAttentionBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedPagedAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: paged_attention")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
