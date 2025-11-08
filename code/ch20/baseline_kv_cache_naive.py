"""baseline_kv_cache_naive.py - Naive KV cache implementation (baseline).

Naive KV cache allocates full memory upfront per sequence, leading to fragmentation
and inefficient memory usage. No paging or memory reuse.

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


class NaiveKVCache:
    """Naive KV cache - allocates full memory per sequence upfront."""
    
    def __init__(self, max_seq_len: int, num_layers: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # Naive: Allocate full cache upfront per sequence (wasteful)
        self.cache = {}  # request_id -> list of (k, v) tensors per layer
    
    def allocate(self, request_id: str) -> None:
        """Allocate full cache for a sequence."""
        # Allocate max_seq_len even if we only use a few tokens (wasteful)
        self.cache[request_id] = []
        for _ in range(self.num_layers):
            k = torch.zeros(self.max_seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
            v = torch.zeros(self.max_seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
            self.cache[request_id].append((k, v))
        torch.cuda._sleep(5000)
    
    def append(self, request_id: str, layer_idx: int, k: torch.Tensor, v: torch.Tensor, pos: int) -> None:
        """Append keys/values to cache.
        
        Args:
            k, v: Shape [num_heads, head_dim] (single token)
        """
        if request_id not in self.cache:
            self.allocate(request_id)
        cache_k, cache_v = self.cache[request_id][layer_idx]
        # k/v shape: [num_heads, head_dim]
        # Store as [1, num_heads, head_dim] in cache at position pos
        cache_k[pos:pos+1] = k.unsqueeze(0)  # Add sequence dimension
        cache_v[pos:pos+1] = v.unsqueeze(0)
        torch.cuda._sleep(2000)
    
    def get(self, request_id: str, layer_idx: int, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cached keys/values."""
        cache_k, cache_v = self.cache[request_id][layer_idx]
        return cache_k[start:end], cache_v[start:end]
    
    def free(self, request_id: str) -> None:
        """Free cache for a sequence."""
        if request_id in self.cache:
            del self.cache[request_id]


class SimpleAttentionLayer(nn.Module):
    """Simple attention layer for KV cache demo."""
    
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, dtype=dtype)
        self.proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
    
    def forward(self, x: torch.Tensor, kv_cache: NaiveKVCache, request_id: str, layer_idx: int, cache_pos: int) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        # q, k, v: [batch_size, seq_len, hidden_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Update KV cache (naive: always allocate full cache)
        if cache_pos == 0:
            kv_cache.allocate(request_id)
        
        # Store k/v in cache: k/v shape is [batch_size, num_heads, seq_len, head_dim]
        # For single token decoding: batch_size=2, seq_len=1
        # Extract per-head tokens: [num_heads, head_dim] for position cache_pos
        k_to_cache = k[:, :, 0, :].transpose(0, 1)  # [num_heads, batch_size, head_dim] -> take first batch
        v_to_cache = v[:, :, 0, :].transpose(0, 1)  # [num_heads, batch_size, head_dim]
        # Store each head separately, or average across batch
        k_single = k_to_cache[:, 0, :]  # [num_heads, head_dim] - use first batch item
        v_single = v_to_cache[:, 0, :]  # [num_heads, head_dim]
        kv_cache.append(request_id, layer_idx, k_single, v_single, cache_pos)
        
        # Get cached keys/values for attention
        if cache_pos > 0:
            cached_k, cached_v = kv_cache.get(request_id, layer_idx, 0, cache_pos)
            _ = cached_k.sum()
            _ = cached_v.sum()
        
        return self.proj(x)


class BaselineKVCacheNaiveBenchmark(Benchmark):
    """Naive KV cache baseline - allocates full memory upfront."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.kv_cache = None
        self.inputs = None
        self.max_seq_len = 8192
        self.num_layers = 1
        self.num_heads = 2
        self.head_dim = 32
        self.hidden_dim = self.num_heads * self.head_dim
        self.batch_size = 1  # Single batch for simplicity
        self.sequence_lengths = [512, 1024, 2048]
        self.dtype = torch.float32
    
    def setup(self) -> None:
        """Setup: Initialize model and KV cache."""
        torch.manual_seed(42)
        
        # Create simple model with attention layers
        layers = []
        for _ in range(self.num_layers):
            layers.append(SimpleAttentionLayer(self.hidden_dim, self.num_heads, self.head_dim, dtype=self.dtype))
        self.model = nn.Sequential(*layers).to(self.device).eval()
        
        # Naive KV cache (allocates full memory upfront)
        self.kv_cache = NaiveKVCache(
            max_seq_len=self.max_seq_len,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device
        )
        
        # Prepare inputs (different sequence lengths to show fragmentation)
        self.inputs = []
        for seq_len in self.sequence_lengths:
            x = torch.randn(self.batch_size, seq_len, self.hidden_dim, device=self.device, dtype=self.dtype)
            self.inputs.append(x)
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - naive KV cache with full allocation."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_kv_cache_naive", enable=enable_nvtx):
            # Process multiple sequences with different lengths
            for seq_idx, x in enumerate(self.inputs):
                request_id = f"req_{seq_idx}"
                seq_len = x.size(1)
                
                # Process sequence token by token (decoding scenario)
                for pos in range(seq_len):
                    # Process single token
                    token = x[:, pos:pos+1, :]
                    
                    # Forward through all layers with KV cache
                    hidden = token
                    for layer_idx, layer in enumerate(self.model):
                        hidden = layer(hidden, self.kv_cache, request_id, layer_idx, pos)
                    
                # Free cache after sequence (naive: full allocation was wasted)
                self.kv_cache.free(request_id)

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.kv_cache, self.inputs
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,  # Reduced iterations
            warmup=3,  # Reduced warmup
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineKVCacheNaiveBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Naive KV Cache: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
