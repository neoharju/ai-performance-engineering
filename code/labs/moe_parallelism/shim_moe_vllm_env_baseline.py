"""Legacy shim for the MoE parallelism lab.

Use `labs/moe_parallelism/scenario_moe_vllm_env_baseline.py` instead.
"""

from __future__ import annotations

from labs.moe_parallelism import scenario_moe_vllm_env_baseline as _impl


def get_benchmark():
    return _impl.get_benchmark()
