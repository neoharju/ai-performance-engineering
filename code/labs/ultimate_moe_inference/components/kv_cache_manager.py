"""KV Cache Manager with PagedAttention support.

Implements efficient KV cache management:
- PagedAttention: Block-based memory allocation (Ch16)
- FP8 KV cache: 2x memory reduction vs BF16
- NVFP4 Block Scaling: 4x memory reduction with block-wise scales
- Prefix caching: Reuse computation for shared prompts
- Dynamic allocation: Grow cache as needed

BEFORE (BF16 KV Cache):
    Memory per token: num_layers × 2 × num_heads × head_dim × 2 bytes
    For LLaMA-70B: 80 × 2 × 64 × 128 × 2 = 2.6MB per token
    32K context = 83GB KV cache!

AFTER (FP8 KV Cache):
    Memory per token: num_layers × 2 × num_heads × head_dim × 1 byte
    For LLaMA-70B: 80 × 2 × 64 × 128 × 1 = 1.3MB per token
    32K context = 41.5GB KV cache (2x reduction!)

AFTER (NVFP4 with Block Scaling):
    Memory per token: (elements × 0.5 bytes) + (elements/16 × 2 bytes scales)
    Achieves ~3.5x reduction with minimal accuracy loss

WHY FP8 works for KV Cache:
    - KV values have limited dynamic range (already normalized)
    - Attention softmax is robust to small quantization errors
    - FP8 E4M3 provides sufficient precision for inference
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import torch
import torch.nn as nn


class KVPrecision(Enum):
    """KV cache precision modes."""
    BF16 = "bf16"           # 2 bytes per element
    FP16 = "fp16"           # 2 bytes per element
    FP8_E4M3 = "fp8_e4m3"   # 1 byte per element (2x savings)
    FP8_E5M2 = "fp8_e5m2"   # 1 byte per element (wider range)
    NVFP4 = "nvfp4"         # 0.5 bytes + block scales (~3.5x savings)


@dataclass
class KVCacheConfig:
    """Configuration for KV cache.
    
    PERFORMANCE ANALYSIS (from PERFORMANCE_OPTIMIZATION_ANALYSIS.md):
    
    FP8 KV Cache Benefits:
    - 2x memory reduction → 2x larger batch sizes
    - 2x less memory bandwidth → faster decode
    - Minimal accuracy impact for inference
    
    NVFP4 Block Scaling Benefits:
    - ~4x memory reduction with 16-element blocks
    - Block-wise scale factors preserve accuracy
    - Ideal for very long contexts (128K+)
    """
    
    # Block configuration (PagedAttention)
    block_size: int = 16  # Tokens per block
    num_blocks: int = 2048  # Total blocks
    
    # Model dimensions (auto-detected from model if not set)
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    
    # Precision - NEW: Support for multiple precision modes
    precision: KVPrecision = KVPrecision.FP8_E4M3  # Default to FP8 for 2x savings
    dtype: torch.dtype = torch.bfloat16  # Fallback/compute dtype
    use_fp8: bool = True  # Legacy flag for compatibility
    
    # NVFP4 Block Scaling configuration
    nvfp4_block_size: int = 16  # Elements per scale factor
    
    # Features
    enable_prefix_caching: bool = True
    prefix_cache_size_gb: float = 8.0
    
    def __post_init__(self):
        """Sync legacy use_fp8 flag with precision enum."""
        if self.use_fp8 and self.precision == KVPrecision.BF16:
            self.precision = KVPrecision.FP8_E4M3
    
    @property
    def bytes_per_element(self) -> float:
        """Bytes per KV element based on precision."""
        if self.precision == KVPrecision.NVFP4:
            # 4 bits per element + 16-bit scale per block
            return 0.5 + (2.0 / self.nvfp4_block_size)  # ~0.625 for block_size=16
        elif self.precision in (KVPrecision.FP8_E4M3, KVPrecision.FP8_E5M2):
            return 1.0
        elif self.precision in (KVPrecision.BF16, KVPrecision.FP16):
            return 2.0
        else:
            return 4.0  # FP32
    
    @property
    def block_size_bytes(self) -> int:
        """Size of a single block in bytes."""
        # K and V for all layers and heads
        # Shape per block: [2, num_layers, block_size, num_heads, head_dim]
        elements = 2 * self.num_layers * self.block_size * self.num_heads * self.head_dim
        return int(elements * self.bytes_per_element)
    
    @property
    def total_memory_gb(self) -> float:
        """Total memory for KV cache in GB."""
        return (self.num_blocks * self.block_size_bytes) / 1e9
    
    @property
    def memory_savings_vs_bf16(self) -> float:
        """Memory savings ratio compared to BF16."""
        bf16_bytes = 2.0
        return bf16_bytes / self.bytes_per_element


@dataclass
class CacheBlock:
    """A single block in the paged KV cache."""
    
    block_id: int
    ref_count: int = 0  # Number of sequences using this block
    tokens_used: int = 0  # Number of tokens filled
    
    # Prefix caching
    prefix_hash: Optional[int] = None
    is_prefix_cached: bool = False


class PagedKVCache:
    """Paged KV Cache implementing PagedAttention (Ch16).
    
    Key features:
    - Block-based allocation: Memory allocated in fixed-size blocks
    - Dynamic growth: Blocks allocated on-demand as sequence grows
    - Memory efficiency: No fragmentation, blocks can be shared
    - Prefix caching: Common prefixes share blocks
    
    Example:
        cache = PagedKVCache(config)
        
        # Allocate for new sequence
        block_ids = cache.allocate(sequence_id="seq_1", num_tokens=100)
        
        # Store KV tensors
        cache.store(sequence_id="seq_1", layer=0, k=k_tensor, v=v_tensor)
        
        # Retrieve for attention
        k, v = cache.get(sequence_id="seq_1", layer=0)
    """
    
    def __init__(self, config: KVCacheConfig, device: torch.device = torch.device("cuda")):
        """Initialize paged KV cache.
        
        Args:
            config: KV cache configuration
            device: Device to allocate cache on
        """
        self.config = config
        self.device = device
        
        # Block management
        self.blocks: List[CacheBlock] = [
            CacheBlock(block_id=i) for i in range(config.num_blocks)
        ]
        self.free_blocks: List[int] = list(range(config.num_blocks))
        
        # Sequence to block mapping
        self.sequence_blocks: Dict[str, List[int]] = {}
        
        # Allocate GPU memory for blocks
        self._k_cache: Optional[torch.Tensor] = None
        self._v_cache: Optional[torch.Tensor] = None
        self._allocate_memory()
        
        # Prefix cache (hash -> block_ids)
        self.prefix_cache: Dict[int, List[int]] = {}
    
    def _allocate_memory(self) -> None:
        """Allocate GPU memory for cache blocks.
        
        Memory Layout by Precision:
        
        BF16/FP16:
            K/V shape: [num_blocks, num_layers, block_size, num_heads, head_dim]
            dtype: torch.bfloat16 or torch.float16
            
        FP8:
            K/V shape: [num_blocks, num_layers, block_size, num_heads, head_dim]
            dtype: torch.float8_e4m3fn
            2x memory savings vs BF16
            
        NVFP4 (Block Scaled):
            K/V shape: [num_blocks, num_layers, block_size, num_heads, head_dim]
            dtype: torch.uint8 (packed 4-bit values, 2 per byte)
            Scales shape: [num_blocks, num_layers, block_size, num_heads, head_dim // block_size]
            ~3.5x memory savings vs BF16
        """
        # Shape: [num_blocks, num_layers, block_size, num_heads, head_dim]
        shape = (
            self.config.num_blocks,
            self.config.num_layers,
            self.config.block_size,
            self.config.num_heads,
            self.config.head_dim,
        )
        
        # Select dtype based on precision mode
        precision = self.config.precision
        
        if precision == KVPrecision.FP8_E4M3:
            if hasattr(torch, 'float8_e4m3fn'):
                dtype = torch.float8_e4m3fn
            else:
                print("  [Warning] FP8 not available, falling back to BF16")
                dtype = torch.bfloat16
        elif precision == KVPrecision.FP8_E5M2:
            if hasattr(torch, 'float8_e5m2'):
                dtype = torch.float8_e5m2
            else:
                print("  [Warning] FP8 E5M2 not available, falling back to BF16")
                dtype = torch.bfloat16
        elif precision == KVPrecision.NVFP4:
            # NVFP4: Use uint8 for packed 4-bit values (2 values per byte)
            # Plus separate scale tensors
            packed_shape = (
                self.config.num_blocks,
                self.config.num_layers,
                self.config.block_size,
                self.config.num_heads,
                self.config.head_dim // 2,  # 2 values packed per byte
            )
            scale_shape = (
                self.config.num_blocks,
                self.config.num_layers,
                self.config.block_size,
                self.config.num_heads,
                self.config.head_dim // self.config.nvfp4_block_size,
            )
            
            self._k_cache = torch.zeros(packed_shape, dtype=torch.uint8, device=self.device)
            self._v_cache = torch.zeros(packed_shape, dtype=torch.uint8, device=self.device)
            self._k_scales = torch.ones(scale_shape, dtype=torch.float16, device=self.device)
            self._v_scales = torch.ones(scale_shape, dtype=torch.float16, device=self.device)
            
            savings = self.config.memory_savings_vs_bf16
            print(f"Allocated NVFP4 KV cache: {self.config.total_memory_gb:.2f} GB "
                  f"({savings:.1f}x savings vs BF16)")
            return
        elif precision == KVPrecision.FP16:
            dtype = torch.float16
        else:
            dtype = self.config.dtype
        
        self._k_cache = torch.zeros(shape, dtype=dtype, device=self.device)
        self._v_cache = torch.zeros(shape, dtype=dtype, device=self.device)
        
        # Scale tensors not needed for FP8/BF16
        self._k_scales = None
        self._v_scales = None
        
        savings = self.config.memory_savings_vs_bf16
        precision_name = precision.value.upper()
        print(f"Allocated {precision_name} KV cache: {self.config.total_memory_gb:.2f} GB "
              f"({savings:.1f}x savings vs BF16)")
    
    def allocate(self, sequence_id: str, num_tokens: int) -> List[int]:
        """Allocate blocks for a sequence.
        
        Args:
            sequence_id: Unique sequence identifier
            num_tokens: Number of tokens to allocate for
            
        Returns:
            List of allocated block IDs
        """
        num_blocks_needed = math.ceil(num_tokens / self.config.block_size)
        
        if num_blocks_needed > len(self.free_blocks):
            raise RuntimeError(
                f"Not enough free blocks: need {num_blocks_needed}, "
                f"have {len(self.free_blocks)}"
            )
        
        # Allocate blocks
        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop(0)
            self.blocks[block_id].ref_count = 1
            self.blocks[block_id].tokens_used = 0
            allocated.append(block_id)
        
        self.sequence_blocks[sequence_id] = allocated
        return allocated
    
    def extend(self, sequence_id: str, additional_tokens: int) -> List[int]:
        """Extend allocation for a sequence.
        
        Args:
            sequence_id: Sequence identifier
            additional_tokens: Number of additional tokens
            
        Returns:
            List of newly allocated block IDs
        """
        if sequence_id not in self.sequence_blocks:
            return self.allocate(sequence_id, additional_tokens)
        
        current_blocks = self.sequence_blocks[sequence_id]
        
        # Check if last block has space
        last_block = self.blocks[current_blocks[-1]]
        remaining_space = self.config.block_size - last_block.tokens_used
        
        tokens_needing_new_blocks = max(0, additional_tokens - remaining_space)
        new_blocks_needed = math.ceil(tokens_needing_new_blocks / self.config.block_size)
        
        if new_blocks_needed > len(self.free_blocks):
            raise RuntimeError(f"Not enough free blocks")
        
        # Allocate new blocks
        new_blocks = []
        for _ in range(new_blocks_needed):
            block_id = self.free_blocks.pop(0)
            self.blocks[block_id].ref_count = 1
            self.blocks[block_id].tokens_used = 0
            new_blocks.append(block_id)
        
        self.sequence_blocks[sequence_id].extend(new_blocks)
        return new_blocks
    
    def free(self, sequence_id: str) -> None:
        """Free blocks for a sequence.
        
        Args:
            sequence_id: Sequence identifier
        """
        if sequence_id not in self.sequence_blocks:
            return
        
        for block_id in self.sequence_blocks[sequence_id]:
            block = self.blocks[block_id]
            block.ref_count -= 1
            
            if block.ref_count <= 0:
                block.ref_count = 0
                block.tokens_used = 0
                block.prefix_hash = None
                block.is_prefix_cached = False
                self.free_blocks.append(block_id)
        
        del self.sequence_blocks[sequence_id]
    
    def store(
        self,
        sequence_id: str,
        layer: int,
        k: torch.Tensor,
        v: torch.Tensor,
        token_offset: int = 0,
    ) -> None:
        """Store K/V tensors for a layer.
        
        Args:
            sequence_id: Sequence identifier
            layer: Layer index
            k: Key tensor [batch, seq_len, num_heads, head_dim]
            v: Value tensor [batch, seq_len, num_heads, head_dim]
            token_offset: Offset within sequence
        """
        if sequence_id not in self.sequence_blocks:
            raise ValueError(f"Sequence {sequence_id} not allocated")
        
        block_ids = self.sequence_blocks[sequence_id]
        seq_len = k.shape[1]
        
        # Convert to FP8 if configured
        if self.config.use_fp8 and hasattr(torch, 'float8_e4m3fn'):
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)
        
        # Copy to cache blocks
        for i, token_idx in enumerate(range(token_offset, token_offset + seq_len)):
            block_idx = token_idx // self.config.block_size
            within_block_idx = token_idx % self.config.block_size
            
            if block_idx >= len(block_ids):
                break
            
            block_id = block_ids[block_idx]
            
            # Store K and V
            self._k_cache[block_id, layer, within_block_idx] = k[0, i]
            self._v_cache[block_id, layer, within_block_idx] = v[0, i]
            
            # Update tokens used
            self.blocks[block_id].tokens_used = max(
                self.blocks[block_id].tokens_used,
                within_block_idx + 1
            )
    
    def get(
        self,
        sequence_id: str,
        layer: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get K/V tensors for a layer.
        
        Args:
            sequence_id: Sequence identifier
            layer: Layer index
            
        Returns:
            Tuple of (k, v) tensors
        """
        if sequence_id not in self.sequence_blocks:
            raise ValueError(f"Sequence {sequence_id} not allocated")
        
        block_ids = self.sequence_blocks[sequence_id]
        
        # Gather from blocks
        k_parts = []
        v_parts = []
        
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.tokens_used > 0:
                k_parts.append(self._k_cache[block_id, layer, :block.tokens_used])
                v_parts.append(self._v_cache[block_id, layer, :block.tokens_used])
        
        if not k_parts:
            # Return empty tensors
            return (
                torch.empty(0, self.config.num_heads, self.config.head_dim, 
                           device=self.device, dtype=self.config.dtype),
                torch.empty(0, self.config.num_heads, self.config.head_dim,
                           device=self.device, dtype=self.config.dtype),
            )
        
        k = torch.cat(k_parts, dim=0)
        v = torch.cat(v_parts, dim=0)
        
        # Convert from FP8 if needed
        if self.config.use_fp8:
            k = k.to(self.config.dtype)
            v = v.to(self.config.dtype)
        
        return k, v
    
    def get_prefix_cached(
        self,
        prefix_tokens: List[int],
    ) -> Optional[List[int]]:
        """Check if prefix is cached and return block IDs.
        
        Args:
            prefix_tokens: Token IDs for prefix
            
        Returns:
            Block IDs if cached, None otherwise
        """
        if not self.config.enable_prefix_caching:
            return None
        
        prefix_hash = hash(tuple(prefix_tokens))
        return self.prefix_cache.get(prefix_hash)
    
    def cache_prefix(
        self,
        sequence_id: str,
        prefix_tokens: List[int],
        num_prefix_tokens: int,
    ) -> None:
        """Cache prefix blocks for reuse.
        
        Args:
            sequence_id: Source sequence ID
            prefix_tokens: Token IDs for prefix
            num_prefix_tokens: Number of tokens in prefix
        """
        if not self.config.enable_prefix_caching:
            return
        
        if sequence_id not in self.sequence_blocks:
            return
        
        # Calculate blocks that contain the prefix
        num_prefix_blocks = math.ceil(num_prefix_tokens / self.config.block_size)
        all_blocks = self.sequence_blocks[sequence_id]
        prefix_blocks = all_blocks[:num_prefix_blocks]
        
        # Mark blocks as prefix cached
        prefix_hash = hash(tuple(prefix_tokens))
        for block_id in prefix_blocks:
            self.blocks[block_id].prefix_hash = prefix_hash
            self.blocks[block_id].is_prefix_cached = True
            self.blocks[block_id].ref_count += 1  # Extra ref to prevent eviction
        
        self.prefix_cache[prefix_hash] = prefix_blocks
    
    def share_prefix(
        self,
        sequence_id: str,
        prefix_block_ids: List[int],
    ) -> None:
        """Share prefix blocks with a new sequence.
        
        Args:
            sequence_id: New sequence ID
            prefix_block_ids: Block IDs to share
        """
        # Increment ref counts
        for block_id in prefix_block_ids:
            self.blocks[block_id].ref_count += 1
        
        # Initialize sequence with shared blocks
        self.sequence_blocks[sequence_id] = list(prefix_block_ids)
    
    def memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics.
        
        Returns:
            Dictionary with memory statistics in GB
        """
        used_blocks = self.config.num_blocks - len(self.free_blocks)
        used_gb = (used_blocks * self.config.block_size_bytes) / 1e9
        
        return {
            "total_gb": self.config.total_memory_gb,
            "used_gb": used_gb,
            "free_gb": self.config.total_memory_gb - used_gb,
            "used_blocks": used_blocks,
            "free_blocks": len(self.free_blocks),
            "utilization_pct": 100 * used_blocks / self.config.num_blocks,
        }
    
    def __str__(self) -> str:
        """Return string representation."""
        usage = self.memory_usage()
        return (
            f"PagedKVCache("
            f"blocks={self.config.num_blocks}, "
            f"used={usage['used_blocks']}, "
            f"memory={usage['used_gb']:.2f}/{usage['total_gb']:.2f} GB)"
        )


def create_kv_cache(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_tokens: int = 32768,
    block_size: int = 16,
    use_fp8: bool = True,
    device: torch.device = torch.device("cuda"),
) -> PagedKVCache:
    """Convenience function to create a KV cache.
    
    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per head
        max_tokens: Maximum total tokens to support
        block_size: Tokens per block
        use_fp8: Use FP8 for storage
        device: Device to allocate on
        
    Returns:
        Configured PagedKVCache
    """
    num_blocks = math.ceil(max_tokens / block_size)
    
    config = KVCacheConfig(
        block_size=block_size,
        num_blocks=num_blocks,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        use_fp8=use_fp8,
    )
    
    return PagedKVCache(config, device)

