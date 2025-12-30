"""Optimized DDP training with INT8 gradient compression (single GPU)."""

from __future__ import annotations

from pathlib import Path

from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark


def get_benchmark():
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "ddp_compression.py",
        base_args=[
            "--compression",
            "int8",
            "--extra-grad-mb",
            "1024",
            "--batch-size",
            "4",
            "--allow-single-gpu",
            "--simulate-single-gpu-comm",
        ],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:ddp_compression",
        default_nproc_per_node=1,
        multi_gpu_required=False,
        name="optimized_ddp_compression_int8",
    )
