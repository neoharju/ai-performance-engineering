"""Integration tests for profiling workflows (nsys/ncu).

These tests intentionally run real Nsight tools (nsys/ncu) and verify that the
profiling runner can profile a benchmark imported as a dotted module name.

That exercises wrapper sys.path handling (a common failure mode for profiling
subprocesses).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch

from core.harness.benchmark_harness import BenchmarkConfig
from core.profiling.profiling_runner import (
    check_ncu_available,
    check_nsys_available,
    run_ncu_profiling,
    run_nsys_profiling,
)


def _load_temp_benchmark_module(tmp_path: Path) -> ModuleType:
    """Create and import a tiny CUDA benchmark as a package module."""
    package_root = tmp_path / "benchpkg"
    package_root.mkdir(parents=True, exist_ok=True)
    module_path = package_root / "tiny_bench.py"
    module_path.write_text(
        "import torch\n"
        "from core.harness.benchmark_harness import BaseBenchmark\n"
        "\n"
        "\n"
        "class TinyMatmulBenchmark(BaseBenchmark):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.x = None\n"
        "        self.y = None\n"
        "        self.output = None\n"
        "\n"
        "    def setup(self) -> None:\n"
        "        torch.manual_seed(42)\n"
        "        torch.cuda.manual_seed_all(42)\n"
        "        self.x = torch.randn(512, 512, device=self.device)\n"
        "        self.y = torch.randn(512, 512, device=self.device)\n"
        "        torch.cuda.synchronize()\n"
        "\n"
        "    def benchmark_fn(self) -> None:\n"
        "        self.output = self.x @ self.y\n"
        "\n"
        "\n"
        "def get_benchmark():\n"
        "    return TinyMatmulBenchmark()\n"
    )

    module_name = "benchpkg.tiny_bench"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_nsys_profiling_workflow(tmp_path: Path) -> None:
    assert torch.cuda.is_available(), "CUDA required - NVIDIA GPU and tools must be available"
    assert check_nsys_available(), "nsys must be available for profiling workflow tests"

    benchmark_module = _load_temp_benchmark_module(tmp_path)
    benchmark = benchmark_module.get_benchmark()

    output_dir = tmp_path / "profiles_nsys"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig(
        iterations=2,
        warmup=5,
        enable_profiling=True,
        enable_nsys=True,
        profiling_output_dir=str(output_dir),
        adaptive_iterations=False,
    )

    result = run_nsys_profiling(
        benchmark=benchmark,
        benchmark_module=benchmark_module,
        benchmark_class="TinyMatmulBenchmark",
        output_dir=output_dir,
        config=config,
        timeout_seconds=180,
    )
    assert result is not None
    nsys_rep = Path(result["profiling_outputs"]["nsys_rep"])
    assert nsys_rep.exists()
    assert result["metrics"].total_gpu_time_ms is not None
    assert result["metrics"].total_gpu_time_ms > 0.0


def test_ncu_profiling_workflow(tmp_path: Path) -> None:
    assert torch.cuda.is_available(), "CUDA required - NVIDIA GPU and tools must be available"
    assert check_ncu_available(), "ncu must be available for profiling workflow tests"

    benchmark_module = _load_temp_benchmark_module(tmp_path)
    benchmark = benchmark_module.get_benchmark()

    output_dir = tmp_path / "profiles_ncu"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig(
        iterations=1,
        warmup=5,
        enable_profiling=True,
        enable_ncu=True,
        profiling_output_dir=str(output_dir),
        adaptive_iterations=False,
        ncu_metric_set="minimal",
    )

    result = run_ncu_profiling(
        benchmark=benchmark,
        benchmark_module=benchmark_module,
        benchmark_class="TinyMatmulBenchmark",
        output_dir=output_dir,
        config=config,
        timeout_seconds=180,
    )
    assert result is not None
    ncu_rep = Path(result["profiling_outputs"]["ncu_rep"])
    assert ncu_rep.exists()
    assert result["metrics"].kernel_time_ms is not None
    assert result["metrics"].kernel_time_ms > 0.0
