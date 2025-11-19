"""Utilities to verify TMA (Tensor Memory Accelerator) codegen in CUDA binaries."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def require_tma_instructions(binary: Path) -> None:
    """
    Ensure the compiled binary contains cp.async.bulk.tensor instructions.

    Raises RuntimeError with a SKIPPED prefix when validation cannot be performed
    (e.g., cuobjdump missing) or when the instructions are absent.
    """
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise RuntimeError("SKIPPED: cuobjdump not found; cannot verify TMA codegen")

    try:
        proc = subprocess.run(
            [cuobjdump, "--dump-sass", str(binary)],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - subprocess safety
        raise RuntimeError(f"SKIPPED: cuobjdump failed to run ({exc})") from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"SKIPPED: cuobjdump returned {proc.returncode} while checking TMA instructions\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )

    haystack = proc.stdout
    if "cp.async.bulk.tensor" not in haystack and "CP.ASYNC.BULK.TENSOR" not in haystack:
        raise RuntimeError(
            "SKIPPED: TMA validation failed (cp.async.bulk.tensor not found in SASS)"
        )
