"""Shared GPU requirement helpers for benchmarks and examples.

Use these helpers to enforce multi-GPU/NVLink prerequisites consistently.
"""

from __future__ import annotations

import sys

import torch


def skip_if_insufficient_gpus(min_gpus: int = 2) -> None:
    """Raise a standardized SKIPPED RuntimeError when not enough GPUs exist."""
    available = torch.cuda.device_count()
    if available < min_gpus:
        raise RuntimeError(
            f"SKIPPED: Distributed benchmark requires multiple GPUs (found {available} GPU)"
        )


def require_min_gpus(min_gpus: int, script_name: str | None = None) -> None:
    """
    Print a descriptive error and exit if the system does not meet the GPU requirement.

    Prefer skip_if_insufficient_gpus() when running under the benchmark harness so the
    limitation is recorded as a skip instead of a process exit.
    """
    available = torch.cuda.device_count()
    if available >= min_gpus:
        return

    script = script_name or sys.argv[0]
    message = [
        "╔" + "═" * 78 + "╗",
        f"║ {'GPU REQUIREMENT NOT MET':^76} ║",
        "╠" + "═" * 78 + "╣",
        f"║ Script: {script:<69} ║",
        f"║ Required GPUs: {min_gpus:<62} ║",
        f"║ Available GPUs: {available:<61} ║",
        "║" + " " * 78 + "║",
        f"║ This script requires at least {min_gpus} GPU(s) to run correctly."
        + " " * (35 - len(str(min_gpus))) + "║",
        f"║ Current system has {available} GPU(s) available."
        + " " * (41 - len(str(available))) + "║",
        "║" + " " * 78 + "║",
        "║ To run this script:" + " " * 58 + "║",
        f"║ • Use a system with {min_gpus}+ GPUs"
        + " " * (54 - len(str(min_gpus))) + "║",
        "║ • Or modify the script to work with fewer GPUs"
        + " " * 30 + "║",
        "╚" + "═" * 78 + "╝",
    ]
    for line in message:
        print(line, file=sys.stderr)
    sys.exit(1)


def warn_optimal_gpu_count(optimal_gpus: int, script_name: str | None = None) -> None:
    """Emit a warning if the system has fewer GPUs than the optimal configuration."""
    available = torch.cuda.device_count()
    if available < optimal_gpus:
        script = script_name or sys.argv[0]
        print(f"\nWARNING: {script}", file=sys.stderr)
        print(
            f"    Optimized for {optimal_gpus} GPUs but only {available} available; performance may be suboptimal.\n",
            file=sys.stderr,
        )


def require_peer_access(src: int, dst: int, script_name: str | None = None) -> None:
    """Fail if peer access between two devices is unavailable."""
    if torch.cuda.device_count() <= max(src, dst):
        require_min_gpus(max(src, dst) + 1, script_name=script_name)
    if not torch.cuda.can_device_access_peer(src, dst):
        raise RuntimeError(f"Peer access unavailable between GPU {src} and GPU {dst}")
