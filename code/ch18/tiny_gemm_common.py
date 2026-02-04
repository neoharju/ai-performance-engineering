"""Shared helpers for tiny GEMM (QKV/router) benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class TinyGemmConfig:
    batch_size: int = 4
    seq_len: int = 128
    hidden_size: int = 512
    dtype: torch.dtype = torch.bfloat16

    @property
    def tokens(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def tokens_per_iter(self) -> int:
        return self.tokens


def build_tiny_gemm_inputs(
    device: torch.device,
    cfg: TinyGemmConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(cfg.tokens, cfg.hidden_size, device=device, dtype=cfg.dtype)
    w_q = torch.randn(cfg.hidden_size, cfg.hidden_size, device=device, dtype=cfg.dtype)
    w_k = torch.randn(cfg.hidden_size, cfg.hidden_size, device=device, dtype=cfg.dtype)
    w_v = torch.randn(cfg.hidden_size, cfg.hidden_size, device=device, dtype=cfg.dtype)
    w_router = torch.randn(cfg.hidden_size, cfg.hidden_size, device=device, dtype=cfg.dtype)
    w_fused = torch.cat([w_q, w_k, w_v, w_router], dim=1)
    return x, w_q, w_k, w_v, w_router, w_fused
