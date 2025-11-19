"""Common helpers for persistent decode lab."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for persistent decode lab")
    return torch.device("cuda")


def resolve_shapes() -> Tuple[int, int, int]:
    """Resolve (batch, seq_len, head_dim) with an optional QUICK override."""
    quick = os.getenv("QUICK", "0") == "1"
    batch = 4 if quick else 8
    seq_len = 16 if quick else 32
    head_dim = 64
    return batch, seq_len, head_dim


@dataclass
class DecodeInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    work_seq_ids: torch.Tensor
    work_steps: torch.Tensor
    work_counter: torch.Tensor


def build_inputs(device: torch.device) -> DecodeInputs:
    batch, seq_len, head_dim = resolve_shapes()
    torch.manual_seed(0)

    q = torch.randn(batch, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = torch.zeros_like(q)

    work_seq_ids = torch.arange(batch, device=device, dtype=torch.int32)
    work_steps = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    work_counter = torch.zeros(1, device=device, dtype=torch.int32)

    return DecodeInputs(
        q=q,
        k=k,
        v=v,
        out=out,
        work_seq_ids=work_seq_ids,
        work_steps=work_steps,
        work_counter=work_counter,
    )


def tokens_per_iteration() -> float:
    batch, seq_len, _ = resolve_shapes()
    return float(batch * seq_len)

