"""Shared probe/output factory for ch04 mixin usage."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def make_probe_output(device: torch.device, shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
    """Create a small deterministic probe tensor and zero output."""
    probe = torch.zeros(*shape, device=device, dtype=torch.float32)
    output = torch.zeros_like(probe)
    return {"probe": probe, "output": output}
