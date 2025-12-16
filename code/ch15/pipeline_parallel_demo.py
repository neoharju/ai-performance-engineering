"""pipeline_parallel_demo.py - Chapter 15 pipeline-parallel (PP) demo (tool).

Minimal 2-stage pipeline example using torch.distributed send/recv.

Run with torchrun, e.g.:
  torchrun --nproc_per_node 2 ch15/pipeline_parallel_demo.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.distributed as dist


def _init_dist() -> tuple[int, int, torch.device]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for pipeline_parallel_demo.py")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Run with torchrun (missing RANK/WORLD_SIZE env vars).")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return rank, world_size, device


def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 15 pipeline parallel demo (2-stage)")
    parser.add_argument("--batch", type=int, default=64, help="Batch size.")
    parser.add_argument("--hidden", type=int, default=4096, help="Hidden dimension.")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16", help="Compute dtype.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pipeline_parallel_demo.py")

    rank, world_size, device = _init_dist()
    if world_size != 2:
        raise RuntimeError(f"pipeline_parallel_demo.py requires world_size=2, got {world_size}")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    hidden = int(args.hidden)
    batch = int(args.batch)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Deterministic weights (both ranks create the same tensors).
    w1 = torch.randn(hidden, hidden, device=device, dtype=dtype)
    w2 = torch.randn(hidden, hidden, device=device, dtype=dtype)

    # Buffers for send/recv.
    act = torch.empty(batch, hidden, device=device, dtype=dtype)
    out = torch.empty(batch, hidden, device=device, dtype=dtype)

    x0 = None
    if rank == 0:
        x0 = torch.randn(batch, hidden, device=device, dtype=dtype)

    def stage0_forward(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ w1.t())

    def stage1_forward(x: torch.Tensor) -> torch.Tensor:
        return x @ w2.t()

    # Warmup.
    with torch.no_grad():
        for _ in range(int(args.warmup)):
            if rank == 0:
                assert x0 is not None
                act.copy_(stage0_forward(x0))
                dist.send(act, dst=1)
                dist.recv(out, src=1)
            else:
                dist.recv(act, src=0)
                out.copy_(stage1_forward(act))
                dist.send(out, dst=0)
        torch.cuda.synchronize(device)
        dist.barrier()

    # Timed loop (measure wall-clock on rank 0, take max over ranks).
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(int(args.iters)):
            if rank == 0:
                assert x0 is not None
                act.copy_(stage0_forward(x0))
                dist.send(act, dst=1)
                dist.recv(out, src=1)
            else:
                dist.recv(act, src=0)
                out.copy_(stage1_forward(act))
                dist.send(out, dst=0)
        torch.cuda.synchronize(device)
        dist.barrier()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(int(args.iters), 1)

    worst = torch.tensor([elapsed_ms], device=device, dtype=torch.float32)
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    worst_ms = float(worst.item())

    max_diff = 0.0
    if rank == 0:
        assert x0 is not None
        ref = stage1_forward(stage0_forward(x0))
        max_diff = float((out.float() - ref.float()).abs().max().item())
        print(f"pipeline_parallel_demo: world=2 -> {worst_ms:.3f} ms/iter, max_abs_diff={max_diff:.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

