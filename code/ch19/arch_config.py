"""Shared Blackwell defaults for Chapter 19 examples."""

from __future__ import annotations

import os
import torch
from common.python.compile_utils import enable_tf32


# Prefer BF16 over FP16 for AMP paths on B200/B300.
PREFER_BFLOAT16: bool = True

# Toggle Transformer Engine usage for FP8 / NVFP4 helpers.
USE_TE_FP8: bool = True
USE_TE_FP4: bool = True

# Recommended torch.compile defaults for steady-state decode graphs.
TORCH_COMPILE_KW = {
    "mode": "reduce-overhead",
    "fullgraph": True,
    "dynamic": False,
}

# Expand FP32 matmul heuristics for fallback paths using new TF32 APIs.
enable_tf32()

# Stream-ordered allocator is the default best practice on Blackwell.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")


def prefer_bfloat16() -> bool:
    """Return whether BF16 should be preferred over FP16 in AMP contexts."""
    return PREFER_BFLOAT16


__all__ = [
    "PREFER_BFLOAT16",
    "USE_TE_FP8",
    "USE_TE_FP4",
    "TORCH_COMPILE_KW",
    "prefer_bfloat16",
]
