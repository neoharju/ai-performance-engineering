"""labs.moe_cuda/optimized_decode_attention.py - FlashMLA-inspired decode kernel."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older PyTorch fallback
    SDPBackend = None  # type: ignore[assignment]
    sdpa_kernel = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.compile_utils import compile_model, enable_tf32
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range


def _math_sdp_context():
    """Prefer the new sdpa_kernel API; fall back to no-op if unavailable."""
    if sdpa_kernel is None or SDPBackend is None:
        return nullcontext()
    backend = getattr(SDPBackend, "MATH", None)
    if backend is None:
        return nullcontext()
    return sdpa_kernel([backend])

# Prefer non-Flash SDPA backends on SM12x to avoid mismatched prebuilt kernels
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _SDPA_BACKENDS = [
        backend
        for backend in (
            getattr(SDPBackend, "CUDNN", None),
            getattr(SDPBackend, "EFFICIENT_ATTENTION", None),
            getattr(SDPBackend, "MATH", None),
        )
        if backend is not None
    ]
except Exception:  # pragma: no cover - older Torch
    sdpa_kernel = None  # type: ignore[assignment]
    _SDPA_BACKENDS = []


def _sdpa_safe_ctx():
    """Favor cudnn/efficient/math; avoid Flash kernels that may lack SM121 binaries."""
    if sdpa_kernel is None or not _SDPA_BACKENDS:
        return nullcontext()
    return sdpa_kernel(_SDPA_BACKENDS)


class FlashDecodeAttention(nn.Module):
    """Flash-style decoder using scaled_dot_product_attention."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )


class OptimizedDecodeAttentionBenchmark(BaseBenchmark):
    """Benchmark for fused decode attention."""

    def __init__(self) -> None:
        super().__init__()
        self.force_math = False
        self.batch = 32
        self.num_heads = 16
        self.head_dim = 128
        self.kv_seq = 2048
        self.module: Optional[nn.Module] = None
        self.q: Optional[torch.Tensor] = None
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        tokens = self.batch * self.kv_seq
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._history: Dict[str, List[float]] = {"latency_ms": []}

    def setup(self) -> None:
        enable_tf32()
        torch.manual_seed(42)
        major, _ = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        if major >= 12:
            # Avoid Flash/CUTLASS kernels that are not built for SM121; route to math backend.
            class _MathDecode(nn.Module):
                def forward(self, q, k, v):  # pragma: no cover - benchmark path
                    with _math_sdp_context():
                        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

            module = _MathDecode().to(self.device)
            self.force_math = True
        else:
            module = FlashDecodeAttention().to(self.device)
            # compile_model no-ops on SM>=12; keep path eager to avoid Triton/Inductor gaps
            module = compile_model(module, mode="max-autotune")
        self.module = module

        dtype = torch.bfloat16
        self.q = torch.randn(
            self.batch,
            self.num_heads,
            1,
            self.head_dim,
            device=self.device,
            dtype=dtype,
        )
        self.k = torch.randn(
            self.batch,
            self.num_heads,
            self.kv_seq,
            self.head_dim,
            device=self.device,
            dtype=dtype,
        )
        self.v = torch.randn_like(self.k)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if any(t is None for t in (self.module, self.q, self.k, self.v)):
            raise RuntimeError("Decode tensors missing")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("moe_cuda_decode_flash", enable=enable_nvtx):
            ctx = nullcontext() if self.force_math else _sdpa_safe_ctx()
            with torch.inference_mode(), ctx:
                start = self._record_start()
                _ = self.module(self.q, self.k, self.v)
                torch.cuda.synchronize(self.device)
                self._history["latency_ms"].append(self._record_stop(start))
        return {"decode_ms": self._history["latency_ms"]}

    def teardown(self) -> None:
        self.module = None
        self.q = None
        self.k = None
        self.v = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=8, warmup=3)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self._history["latency_ms"]:
            return None
        return {
            "decode.mean_ms": float(sum(self._history["latency_ms"]) / len(self._history["latency_ms"]))
        }

    def validate_result(self) -> Optional[str]:
        if any(t is None for t in (self.module, self.q, self.k, self.v)):
            return "Decode tensors missing"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedDecodeAttentionBenchmark()
