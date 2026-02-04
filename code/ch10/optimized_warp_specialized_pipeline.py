"""Python harness wrapper for optimized_warp_specialized_pipeline.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedWarpSpecializedPipelineBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_warp_specialized_pipeline",
            friendly_name="Optimized Warp Specialized Pipeline",
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "TILE_SIZE": 64,
                "PIPELINE_STAGES": 2,
                "WARPS_PER_BLOCK": 3,
                "tile_iter": 0,
                "shared_bytes": 3,
                "dtype": 'float32',
                "batch_size": 1,
            },
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedWarpSpecializedPipelineBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
