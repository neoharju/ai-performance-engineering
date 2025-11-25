"""Layer 03: Pipelining Optimizations (Ch9-10).

Intra-kernel pipelining and memory transfer optimizations:
- Tiling for data reuse
- Double buffering for latency hiding
- TMA (Tensor Memory Accelerator) for Blackwell
- FlashAttention via SDPA
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Generator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Use new SDPA API when available (PyTorch 2.2+)
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _NEW_SDPA_API = True
except ImportError:
    sdpa_kernel = None  # type: ignore[assignment]
    SDPBackend = None  # type: ignore[assignment]
    _NEW_SDPA_API = False


class Layer03Pipelining:
    """Layer 3: Pipelining optimizations from Chapters 9-10.
    
    These optimizations hide memory latency through overlapped
    execution and efficient attention computation.
    
    Techniques:
    - Ch9: Tiling, data reuse patterns
    - Ch10: Double buffering, producer-consumer pipelines, FlashAttention
    """
    
    name = "Layer 03: Pipelining (Ch9-10)"
    chapters = [9, 10]
    
    def __init__(self):
        """Initialize layer."""
        self._flash_attention_enabled = False
        self._double_buffer_streams: Dict[str, torch.cuda.Stream] = {}
        self._tma_available = False
        
        # Check TMA availability (Blackwell/SM 9.0+)
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability()
            self._tma_available = cc[0] >= 9
    
    def apply(self, benchmark: Any) -> None:
        """Apply pipelining optimizations.
        
        Args:
            benchmark: Benchmark instance to configure
        """
        model = getattr(benchmark, 'model', None)
        config = getattr(benchmark, 'config', None)
        
        if model is not None:
            self.enable_flash_attention(model)
        
        if config and getattr(config, 'enable_double_buffering', True):
            self.setup_double_buffering()
    
    def enable_flash_attention(self, model: nn.Module) -> None:
        """Enable FlashAttention via SDPA (Ch10).
        
        FlashAttention uses the same intra-kernel pipelining concepts:
        - Tiled computation in SRAM
        - Online softmax for memory efficiency
        - O(seq_len) memory instead of O(seq_len²)
        
        Args:
            model: Model to configure
        """
        # For HuggingFace models, FlashAttention is typically enabled via
        # attn_implementation="flash_attention_2" during loading.
        # Here we ensure SDPA is configured correctly.
        
        # Check if SDPA backends are available
        if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
            # Prefer FlashAttention
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)  # Disable slow fallback
            
            self._flash_attention_enabled = True
            print("  [Layer 03] Enabled FlashAttention via SDPA")
        else:
            print("  [Layer 03] FlashAttention not available, using default attention")
    
    def setup_double_buffering(self) -> None:
        """Setup streams for double buffering (Ch10).
        
        Double buffering overlaps memory transfers with computation
        by using two buffers and alternating between them.
        """
        if not torch.cuda.is_available():
            return
        
        # Create streams for double buffering
        self._double_buffer_streams = {
            "load": torch.cuda.Stream(),
            "compute": torch.cuda.Stream(),
        }
        
        print("  [Layer 03] Created double buffer streams")
    
    @contextmanager
    def double_buffer_context(
        self,
        phase: str = "compute"
    ) -> Generator[Optional[torch.cuda.Stream], None, None]:
        """Context manager for double buffer execution.
        
        Args:
            phase: "load" or "compute"
            
        Yields:
            CUDA stream for the phase
        """
        stream = self._double_buffer_streams.get(phase)
        if stream is None:
            yield None
            return
        
        with torch.cuda.stream(stream):
            yield stream
    
    def apply_sdpa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = True,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """Apply scaled dot-product attention with FlashAttention (Ch10).
        
        This uses PyTorch's SDPA which automatically selects the best
        backend (FlashAttention, Memory-Efficient, or Math).
        
        Args:
            query: Query tensor [batch, heads, seq_q, head_dim]
            key: Key tensor [batch, heads, seq_k, head_dim]
            value: Value tensor [batch, heads, seq_k, head_dim]
            is_causal: Apply causal mask
            dropout_p: Dropout probability
            
        Returns:
            Attention output [batch, heads, seq_q, head_dim]
        """
        return F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
    
    @contextmanager
    def flash_attention_backend(self) -> Generator[None, None, None]:
        """Context manager to force FlashAttention backend.
        
        Yields:
            None (context for FlashAttention)
        """
        if _NEW_SDPA_API and sdpa_kernel is not None:
            try:
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                    yield
            except RuntimeError:
                # Fallback to memory-efficient
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                    yield
        else:
            # No context manager available, just yield
            yield
    
    def get_tma_status(self) -> Dict[str, Any]:
        """Get TMA (Tensor Memory Accelerator) status.
        
        TMA is available on Blackwell (SM 9.0+) and provides
        hardware-accelerated async memory transfers.
        
        Returns:
            Dictionary with TMA status
        """
        return {
            "available": self._tma_available,
            "compute_capability": torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
        }
    
    def get_status(self) -> dict:
        """Get status of applied optimizations."""
        return {
            "flash_attention_enabled": self._flash_attention_enabled,
            "double_buffer_streams": len(self._double_buffer_streams),
            "tma_available": self._tma_available,
        }
    
    def __str__(self) -> str:
        """Return string representation."""
        return f"{self.name}: flash={self._flash_attention_enabled}, TMA={self._tma_available}"

