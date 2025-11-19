"""Persistent decode with CUDA Graphs: prefill + decode captured separately."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.persistent_decode.optimized_persistent_decode_triton import persistent_decode_kernel
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)


class OptimizedPersistentDecodeGraphsBenchmark(BaseBenchmark):
    """Capture prefill and persistent decode in CUDA Graphs."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.block_k = 64
        self.num_programs = 8
        self.prefill_graph: torch.cuda.CUDAGraph | None = None
        self.decode_graph: torch.cuda.CUDAGraph | None = None
        self.prefill_out: torch.Tensor | None = None
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())

    def setup(self) -> None:
        self.inputs = build_inputs(self.device)
        self.prefill_out = torch.empty((self.batch, self.seq_len), device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

        # Capture prefill (toy: dot across head_dim per token)
        self.prefill_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.prefill_graph):
            qk = (self.inputs.q * self.inputs.k).sum(dim=-1)
            self.prefill_out.copy_(qk)

        # Capture persistent decode kernel
        self.decode_graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        grid = (self.batch,)
        BLOCK_K = self.block_k
        with torch.cuda.graph(self.decode_graph):
            persistent_decode_kernel[grid](
                self.inputs.q,
                self.inputs.k,
                self.inputs.v,
                self.inputs.out,
                self.inputs.work_seq_ids,
                self.inputs.work_steps,
                self.batch,
                head_dim=self.head_dim,
                max_steps=self.seq_len,
                BLOCK_K=BLOCK_K,
                num_warps=2,
                num_stages=1,
            )
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None or self.prefill_graph is None or self.decode_graph is None:
            raise RuntimeError("Graphs not initialized")

        with self._nvtx_range("persistent_decode_graphs"):
            self.prefill_graph.replay()
            self.decode_graph.replay()
            self._synchronize()

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None
        self.prefill_graph = None
        self.decode_graph = None
        self.prefill_out = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=12, warmup=4)

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedPersistentDecodeGraphsBenchmark()
