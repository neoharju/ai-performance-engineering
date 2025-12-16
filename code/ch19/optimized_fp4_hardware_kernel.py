"""
Optimized FP4 hardware kernel benchmark (cuda_fp4.h intrinsics path).

Wraps the CUDA sample binary so it can be driven via aisp bench.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin


class OptimizedFP4HardwareKernelBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Invoke the fp4 intrinsics binary."""

    def __init__(self) -> None:
        super().__init__()
        self.chapter_dir = Path(__file__).parent
        self.bin_path = self.chapter_dir / "optimized_fp4_hardware_kernel"
        self._dump_path = self.chapter_dir / "fp4_hardware_kernel_optimized_output.bin"
        self.matrix_dim = 512
        self.output = None
        self._verify_input = None
        self._verification_payload = None
        self._ran = False
        self._workload = WorkloadMetadata(requests_per_iteration=1.0)
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        import torch
        subprocess.run(
            ["make", "USE_ARCH_SUFFIX=0", "ARCH=sm_100", "optimized_fp4_hardware_kernel"],
            cwd=self.chapter_dir,
            check=True,
        )
        if not self.bin_path.exists():
            raise RuntimeError(f"Binary not found at {self.bin_path}")
        # Dummy input tensor for aliasing checks (binary benchmarks have no inputs)
        self._verify_input = torch.tensor([0.0], dtype=torch.float32)
        self.output = None
        self._ran = False

    def benchmark_fn(self) -> None:
        with self._nvtx_range("optimized_fp4_hardware_kernel"):
            subprocess.run(
                [str(self.bin_path), "--dump-output", str(self._dump_path)],
                cwd=self.chapter_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        self._ran = True

    def capture_verification_payload(self) -> None:
        if not self._ran:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._verify_input is None:
            raise RuntimeError("setup() must run before capture_verification_payload()")
        if not self._dump_path.exists():
            raise RuntimeError(f"Expected output dump not found at {self._dump_path}")
        import numpy as np
        import torch
        data = np.fromfile(self._dump_path, dtype=np.float16)
        expected = self.matrix_dim * self.matrix_dim
        if data.size != expected:
            raise RuntimeError(f"Unexpected dump size: expected {expected} fp16 values, got {data.size}")
        self.output = torch.from_numpy(data).clone().reshape(self.matrix_dim, self.matrix_dim)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self.output,
            batch_size=1,
            parameter_count=0,
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=5, use_subprocess=True)

    def get_custom_metrics(self) -> Optional[dict]:
        return {"variant": "optimized_fp4_intrinsics"}

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if not self.bin_path.exists():
            return "Binary not found"
        return None


def get_benchmark() -> OptimizedFP4HardwareKernelBenchmark:
    return OptimizedFP4HardwareKernelBenchmark()
