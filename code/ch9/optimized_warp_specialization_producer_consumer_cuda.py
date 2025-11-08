"""optimized_warp_specialization_producer_consumer_cuda.py

CUDA-based warp specialization benchmark for the Chapter 9 producer/consumer pattern.
This version loads the custom CUDA extension (warp_specialized_cuda.cu) that assigns
dedicated producer, compute, and consumer warps using the CUDA Pipeline API.
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
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)

CUDA_EXT_NAME = "warp_specialized_cuda_ch9"


@functools.lru_cache(maxsize=1)
def _load_cuda_extension():
    """Build or load the warp specialization CUDA extension for Chapter 9."""
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("warp_specialized_cuda.cu")
    if not source.exists():
        raise FileNotFoundError(
            f"Missing CUDA source for warp specialization: {source}"
        )

    extra_include_paths = [
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
            "Failed to build warp_specialized_cuda extension for Chapter 9. "
            "Ensure nvcc is available and CUDA >= 12 is installed."
        ) from exc


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Chapter 9 warp specialization benchmark.")
    return torch.device("cuda")


class OptimizedWarpSpecializationProducerConsumerCUDABenchmark(Benchmark):
    """Optimized warp specialization benchmark using the CUDA extension."""

    def __init__(self):
        self.device = resolve_device()
        self.consumer_model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.cuda_extension = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        enable_tf32()

        self.cuda_extension = _load_cuda_extension()

        self.consumer_model = nn.Sequential(
            nn.Linear(2048, 2048),
        ).to(self.device).eval()

        self.input = torch.randn(1024, 2048, device=self.device)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self.cuda_extension is None:
            raise RuntimeError("CUDA extension not initialized for warp specialization.")

        with nvtx_range(
            "optimized_warp_specialization_producer_consumer_cuda",
            enable=enable_nvtx,
        ):
            with torch.no_grad():
                intermediate = self.cuda_extension.warp_specialized_ch9_forward(
                    self.input.contiguous()
                )
                output = self.consumer_model(intermediate)
                _ = output.sum()

    def teardown(self) -> None:
        self.consumer_model = None
        self.input = None
        self.cuda_extension = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
            use_subprocess=False,
        )

    def validate_result(self) -> Optional[str]:
        if self.consumer_model is None:
            return "Consumer model not initialized"
        if self.input is None:
            return "Input tensor not initialized"
        if self.cuda_extension is None:
            return "Warp specialization CUDA extension not available"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedWarpSpecializationProducerConsumerCUDABenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    config = benchmark.get_config()
    config.use_subprocess = False
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(benchmark)
    print(
        "\nOptimized Warp Specialization CUDA (Producer/Consumer): "
        f"{result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
