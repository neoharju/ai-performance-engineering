"""Common definitions for the vLLM MoE environment scenarios.

These scenario modules are helpers for the MoE parallelism lab and are used by
legacy shim entry points. They are *not* comparable benchmarks; the primary
workflow is `aisp tools moe-parallelism`.
"""

from __future__ import annotations

import argparse
from typing import Dict

from labs.moe_parallelism.plan import (
    ClusterSpec,
    ModelSpec,
    ParallelismPlan,
    get_default_model_spec,
)

VALID_FABRICS = ("nvlink", "nvl72")
TARGET_NAME = "moe_vllm_env"

NVLINK_CLUSTER = ClusterSpec(
    name="GB200/B200 NVLink (single node)",
    nodes=1,
    gpus_per_node=8,
    nics_per_node=2,
    nic_bandwidth_gbps=800.0,
    nvlink_bandwidth_tbps=1.8,
    hbm_gb=192.0,
    interconnect="NVLink",
)

NVL72_CLUSTER = ClusterSpec(
    name="GB200 NVL72 (dual-rail IB)",
    nodes=9,
    gpus_per_node=8,
    nics_per_node=2,
    nic_bandwidth_gbps=800.0,
    nvlink_bandwidth_tbps=1.8,
    hbm_gb=192.0,
    interconnect="InfiniBand",
)

MODEL: ModelSpec = get_default_model_spec()


def _default_run_args(fabric: str) -> Dict[str, object]:
    if fabric == "nvl72":
        return {
            "ngpu": 72,
            "tp": 8,
            "pp": 3,
            "max_len": 8192,
            "max_seqs": 4096,
            "gpu_util": 0.92,
            "model_path": "gpt-oss-20b/original/",
        }
    return {
        "ngpu": 8,
        "tp": 8,
        "pp": 1,
        "max_len": 8192,
        "max_seqs": 1024,
        "gpu_util": 0.92,
        "model_path": "gpt-oss-20b/original/",
    }


def _build_arg_parser(default_fabric: str = "nvlink") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fabric",
        choices=VALID_FABRICS,
        default=default_fabric,
        help="Fabric to model: nvlink (single node) or nvl72 (multi-node InfiniBand).",
    )
    parser.add_argument("--ngpu", type=int, help="Number of GPUs to use in validation commands.")
    parser.add_argument("--model", dest="model_path", default=None, help="Model path for validation.")
    parser.add_argument("--tp", type=int, help="Tensor parallel degree for validation commands.")
    parser.add_argument("--pp", type=int, help="Pipeline parallel degree for validation commands.")
    parser.add_argument("--max-len", type=int, dest="max_len", help="Max sequence length for validation commands.")
    parser.add_argument("--max-seqs", type=int, dest="max_seqs", help="Max concurrent sequences for validation commands.")
    parser.add_argument("--gpu-util", type=float, dest="gpu_util", help="GPU memory utilization target for vLLM.")
    return parser


def build_plan(label: str, fabric: str) -> ParallelismPlan:
    cross_node = fabric == "nvl72"
    if fabric == "nvlink":
        return ParallelismPlan(
            name=f"{label} vLLM MoE env ({fabric})",
            dp=1,
            pp=1,
            tp=2,
            ep=4,
            microbatch_sequences=8,
            microbatches=6,
            experts_per_gpu=4,
            capacity_factor=1.15,
            dense_checkpoint_fraction=0.75,
            moe_checkpoint_fraction=0.9,
            stage_layers=None,
            cross_node_ep=False,
            notes=[
                "Single-node decode with top-2 routing; small microbatches keep MoE all-to-all visible.",
                "No cross-node experts; expect NVLink-only traffic when tuned correctly.",
            ],
        )

    return ParallelismPlan(
        name=f"{label} vLLM MoE env ({fabric})",
        dp=3,
        pp=3,
        tp=2,
        ep=4,
        microbatch_sequences=8,
        microbatches=8,
        experts_per_gpu=4,
        capacity_factor=1.15,
        dense_checkpoint_fraction=0.75,
        moe_checkpoint_fraction=0.9,
        stage_layers=None,
        cross_node_ep=cross_node,
        notes=[
            "Nine-node NVL72-style layout; experts span nodes so all-to-all rides IB.",
            "Router-heavy microbatches surface small-message latency and QP congestion.",
        ],
    )

