#!/usr/bin/env python3
"""Best-effort detector for the active GPU SM architecture."""
from __future__ import annotations

import subprocess
import sys


def map_cc(major: int, minor: int) -> str:
    if major >= 12:
        return "sm_121"
    if major == 10 and minor >= 3:
        return "sm_103"
    if major == 10:
        return "sm_100"
    return ""


def detect_with_torch() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return map_cc(props.major, props.minor)
    except Exception:
        return ""
    return ""


def detect_with_nvidia_smi() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        return ""
    if not out:
        return ""
    first = out.splitlines()[0]
    parts = first.strip().split(".")
    try:
        major = int(parts[0]) if parts else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return ""
    return map_cc(major, minor)


def main() -> int:
    arch = detect_with_torch()
    if not arch:
        arch = detect_with_nvidia_smi()
    sys.stdout.write(arch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
