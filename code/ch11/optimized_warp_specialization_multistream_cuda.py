"""optimized_warp_specialization_multistream_cuda.py

CUDA-based warp specialization benchmark for Chapter 11's multi-stream pipeline pattern.
Uses the warp_specialized_multistream_extension.cu kernel to launch specialized loader,
compute, and consumer warps across multiple CUDA streams to overlap batches.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from common.python.compile_utils import enable_tf32
from common.python.cuda_capabilities import pipeline_runtime_allowed
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)

CUDA_EXT_NAME = "warp_specialized_multistream_cuda"


@functools.lru_cache(maxsize=1)
def _load_multistream_extension():
    """Build/load the CUDA extension for the Chapter 11 warp-specialized multi-stream kernel."""
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("warp_specialized_multistream_extension.cu")
    if not source.exists():
        raise FileNotFoundError(
            f"Missing CUDA source for Chapter 11 warp specialization: {source}"
        )

    extra_include_paths = [
        str(Path(__file__).parent),
        str(repo_root / "common" / "headers"),
    ]

    try:
        return load(
            name=CUDA_EXT_NAME,
            sources=[str(source)],
            extra_include_paths=extra_include_paths,
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to build warp_specialized_multistream CUDA extension. "
            "Ensure CUDA 13+ Pipeline APIs are available."
        ) from exc


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Chapter 11 warp specialization benchmark.")
    return torch.device("cuda")


class OptimizedWarpSpecializationMultistreamCUDABenchmark(Benchmark):
    """Optimized warp specialization benchmark combining intra-kernel and stream-level overlap."""

    def __init__(self, num_streams: int = 3):
        self.device = resolve_device()
        self.num_streams = max(1, num_streams)
        self.stream_model: Optional[nn.Module] = None
        self.input_a: Optional[torch.Tensor] = None
        self.input_b: Optional[torch.Tensor] = None
        self.cuda_extension = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        enable_tf32()

        supported, reason = pipeline_runtime_allowed()
        if not supported:
            raise RuntimeError(f"SKIPPED: CUDA Pipeline API unavailable ({reason})")

        try:
            self.cuda_extension = _load_multistream_extension()
        except Exception as exc:
            raise RuntimeError(
                "SKIPPED: Failed to build warp-specialized multi-stream CUDA extension "
                f"(CUDA Pipeline API required): {exc}"
            ) from exc

        self.stream_model = nn.Sequential(
            nn.Linear(1024, 1024),
        ).to(self.device).eval()

        self.input_a = torch.randn(32, 1024, device=self.device)
        self.input_b = torch.randn(32, 1024, device=self.device)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self.cuda_extension is None:
            raise RuntimeError("CUDA warp specialization extension not initialized.")

        with nvtx_range(
            "optimized_warp_specialization_multistream_cuda",
            enable=enable_nvtx,
        ):
            with torch.no_grad():
                intermediate = self.cuda_extension.warp_specialized_multistream_forward(
                    self.input_a.contiguous(),
                    self.input_b.contiguous(),
                    int(self.num_streams),
                )
                output = self.stream_model(intermediate)
                _ = output.sum()

    def teardown(self) -> None:
        self.stream_model = None
        self.input_a = None
        self.input_b = None
        self.cuda_extension = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
            use_subprocess=False,
        )

    def validate_result(self) -> Optional[str]:
        if self.stream_model is None:
            return "Post-processing model not initialized"
        if self.input_a is None or self.input_b is None:
            return "Input tensors not initialized"
        if self.cuda_extension is None:
            return "Warp specialization CUDA extension not available"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedWarpSpecializationMultistreamCUDABenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    config = benchmark.get_config()
    config.use_subprocess = False
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(benchmark)
    print(
        "\nOptimized Warp Specialization CUDA (Multi-Stream): "
        f"{result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
