"""Legacy shim for the MoE parallelism lab.

Use `labs/moe_parallelism/scenario_gpt_gb200_optimized.py` instead.
"""

from __future__ import annotations

from labs.moe_parallelism import scenario_gpt_gb200_optimized as _impl


def get_benchmark():
    return _impl.get_benchmark()
