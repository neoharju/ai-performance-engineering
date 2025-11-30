"""Optimized: FP4 via Transformer Engine (Blackwell NVFP4)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata  # noqa: E402


def get_benchmark() -> DecodeBenchmark:
    os.environ.setdefault(
        "PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync,expandable_segments:True,max_split_size_mb:512"
    )
    cfg = DecodeConfig(
        batch_size=32,
        prompt_tokens=4096,
        decode_tokens=1024,
        hidden_size=6144,
        use_fp4=True,
        use_pinned_host=True,
        use_copy_stream=True,
        use_compute_stream=True,
        use_torch_compile=False,
        use_cuda_graphs=False,
        graph_full_iteration=False,
        label="optimized_decode_fp4",
    )
    return attach_benchmark_metadata(DecodeBenchmark(cfg), __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode  # noqa: E402

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    mean = result.timing.mean_ms if result.timing else 0.0
    print(f"\noptimized_decode_fp4: {mean:.3f} ms/iter")




