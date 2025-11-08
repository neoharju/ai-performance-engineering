#!/usr/bin/env python3
"""Best-effort GPU reset utility (user-space only).

Attempts to recover a wedged GPU without requiring root privileges by:
  * Listing compute processes via `nvidia-smi` and terminating them.
  * Clearing PyTorch CUDA caches (if torch is available).
  * Optionally invoking `nvidia-smi --gpu-reset` (will be ignored if permissions are insufficient).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def _log(msg: str) -> None:
    print(f"[reset-gpu.py] {msg}")


def _collect_gpu_pids() -> List[int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:  # pragma: no cover
        _log(f"nvidia-smi unavailable: {exc}")
        return []

    pids: List[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or not line.isdigit():
            continue
        pid = int(line)
        if pid == os.getpid():
            continue
        pids.append(pid)
    return pids


def _terminate_pid(pid: int) -> None:
    for sig, wait in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 0.5)):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return
        except PermissionError:
            _log(f"Insufficient permissions to signal PID {pid}")
            return
        time.sleep(wait)


def _flush_torch_cache() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            _log("Cleared torch CUDA cache")
    except Exception as exc:  # pragma: no cover
        _log(f"torch cuda cache flush failed: {exc}")


def _attempt_gpu_reset(device: int) -> None:
    try:
        subprocess.run(
            ["nvidia-smi", "--gpu-reset", "-i", str(device)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        # Ignore failures (often requires root)
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason", default="unspecified", help="Reason for reset request")
    parser.add_argument("--device", type=int, default=0, help="GPU index to reset (default: 0)")
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Only terminate processes / flush cache, skip nvidia-smi --gpu-reset",
    )
    args = parser.parse_args()

    _log(f"Starting best-effort GPU reset (reason: {args.reason})")

    pids = _collect_gpu_pids()
    if pids:
        _log(f"Terminating {len(pids)} GPU compute processes: {pids}")
        for pid in pids:
            _terminate_pid(pid)
    else:
        _log("No active compute processes found via nvidia-smi")

    _flush_torch_cache()

    if not args.skip_reset:
        _log("Attempting nvidia-smi --gpu-reset (may require elevated privileges)")
        _attempt_gpu_reset(args.device)

    _log("Best-effort GPU reset complete")


if __name__ == "__main__":
    # Ensure script works even when executed from other directories
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
