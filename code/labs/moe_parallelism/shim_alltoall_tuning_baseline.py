"""Legacy shim for the MoE parallelism lab.

Use `labs/moe_parallelism/scenario_alltoall_tuning_baseline.py` instead.
"""

from __future__ import annotations

from labs.moe_parallelism import scenario_alltoall_tuning_baseline as _impl


def get_benchmark():
    return _impl.get_benchmark()
