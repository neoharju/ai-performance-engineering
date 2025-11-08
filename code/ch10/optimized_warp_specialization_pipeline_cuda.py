"""optimized_warp_specialization_pipeline_cuda.py

CUDA-based warp specialization benchmark for Chapter 10's intra-kernel pipeline pattern.
Loads the warp_specialized_pipeline_enhanced CUDA extension that overlaps producer,
compute, and consumer warps via the CUDA Pipeline API with double buffering.
"""

from __future__ import annotations

import functools
import logging
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

CUDA_EXT_NAME = "warp_specialized_pipeline_enhanced_cuda"


@functools.lru_cache(maxsize=1)
def _load_pipeline_extension():
    """Build/load the CUDA extension for the Chapter 10 warp-specialized pipeline."""
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("warp_specialized_pipeline_enhanced_extension.cu")
    if not source.exists():
        raise FileNotFoundError(
            f"Missing CUDA source for Chapter 10 warp specialization: {source}"
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
            "Failed to build warp_specialized_pipeline_enhanced CUDA extension. "
            "Ensure nvcc is installed and the CUDA toolkit supports Pipeline APIs."
        ) from exc


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Chapter 10 warp specialization benchmark.")
    return torch.device("cuda")


class OptimizedWarpSpecializationPipelineCUDABenchmark(Benchmark):
    """Optimized intra-kernel pipeline benchmark driven by CUDA warp specialization."""

    def __init__(self):
        self.device = resolve_device()
        self.post_model: Optional[nn.Module] = None
        self.input_a: Optional[torch.Tensor] = None
        self.input_b: Optional[torch.Tensor] = None
        self.cuda_extension = None
        self._logger = logging.getLogger(__name__)

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        enable_tf32()
        supported, reason = pipeline_runtime_allowed()
        if not supported:
            raise RuntimeError(f"SKIPPED: CUDA Pipeline API unavailable ({reason})")

        try:
            self.cuda_extension = _load_pipeline_extension()
        except Exception as exc:
            raise RuntimeError(
                "SKIPPED: Failed to build warp-specialized CUDA extension "
                f"(CUDA Pipeline API required): {exc}"
            ) from exc

        self.post_model = nn.Sequential(
            nn.Linear(2048, 2048),
        ).to(self.device).eval()

        self.input_a = torch.randn(512, 2048, device=self.device)
        self.input_b = torch.randn(512, 2048, device=self.device)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self.cuda_extension is None:
            raise RuntimeError("CUDA warp specialization extension not initialized.")

        with nvtx_range(
            "optimized_warp_specialization_pipeline_cuda",
            enable=enable_nvtx,
        ):
            self._run_cuda_pipeline()

    def teardown(self) -> None:
        self.post_model = None
        self.input_a = None
        self.input_b = None
        self.cuda_extension = None
        torch.cuda.empty_cache()

    def _run_cuda_pipeline(self) -> None:
        with torch.no_grad():
            intermediate = (
                self.cuda_extension.warp_specialized_pipeline_enhanced_forward(
                    self.input_a.contiguous(),
                    self.input_b.contiguous(),
                )
            )
            output = self.post_model(intermediate)
            _ = output.sum()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=5,
            warmup=2,
        )

    def validate_result(self) -> Optional[str]:
        if self.post_model is None:
            return "Post-processing model not initialized"
        if self.input_a is None or self.input_b is None:
            return "Input tensors not initialized"
        if self.cuda_extension is None:
            return "Warp specialization CUDA extension not available"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedWarpSpecializationPipelineCUDABenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    config = benchmark.get_config()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(benchmark)
    print(
        "\nOptimized Warp Specialization CUDA (Pipeline): "
        f"{result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
