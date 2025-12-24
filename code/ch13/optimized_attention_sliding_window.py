"""optimized_attention_sliding_window.py - FlexAttention sliding-window attention."""

from __future__ import annotations

import torch

from core.benchmark.flexattention_sliding_window import (
    SlidingWindowAttentionBenchmark,
    SlidingWindowConfig,
)
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedSlidingWindowAttentionBenchmark(SlidingWindowAttentionBenchmark):
    def __init__(self) -> None:
        cfg = SlidingWindowConfig(
            batch_size=2,
            num_heads=16,
            seq_len=2048,
            head_dim=64,
            window_size=256,
            dtype=torch.bfloat16,
        )
        super().__init__(use_flex=True, cfg=cfg)


def get_benchmark() -> BaseBenchmark:
    return OptimizedSlidingWindowAttentionBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
