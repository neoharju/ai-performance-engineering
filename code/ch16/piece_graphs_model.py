"""Shared model definitions for the piece-graphs benchmarks (Chapter 16)."""

from __future__ import annotations

import torch
import torch.nn as nn


class PieceGraphBlock(nn.Module):
    """Transformer-style block used by both head and tail regions."""

    def __init__(self, *, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


class RegionalPieceGraph(nn.Module):
    """Splits the model into two regions (head stack, tail stack)."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        n_layers: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        midpoint = n_layers // 2
        blocks = [PieceGraphBlock(hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(n_layers)]
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.region_head = nn.Sequential(*blocks[:midpoint])
        self.region_tail = nn.Sequential(*blocks[midpoint:])
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.region_head(x)
        x = self.region_tail(x)
        x = self.output_proj(x)
        return x

