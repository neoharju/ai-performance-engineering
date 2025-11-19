#!/usr/bin/env python3
"""Baseline symmetric-memory perf microbench (NCCL only).

Compares simple NCCL AllReduce latency/bandwidth across payload sizes.
Use the optimized variant to see the uplift when using SymmetricMemory + direct puts.
"""
from __future__ import annotations

import argparse
import datetime
import os
from typing import List, Tuple

import torch
import torch.distributed as dist


def init_distributed() -> Tuple[int, int, int]:
    """Initialize process group for a single-node demo."""
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60),
        )
        torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), torch.cuda.current_device()


def benchmark_allreduce(
    size_mb: float, iters: int, warmup: int, device: torch.device, world_size: int
) -> Tuple[float, float]:
    """Return (avg_ms, effective_gbps) for NCCL all_reduce."""
    numel = int((size_mb * 1024 * 1024) / 4)  # float32
    tensor = torch.ones(numel, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        dist.all_reduce(tensor)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / iters

    # Rough bytes moved: two traversals of the payload per rank (ring-style)
    bytes_moved = size_mb * 1024 * 1024 * 2 * iters
    gbps = (bytes_moved / (total_ms / 1000.0)) / 1e9
    return avg_ms, gbps


def parse_sizes(sizes: List[str]) -> List[float]:
    return [float(s) for s in sizes]


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline NCCL-only perf microbench")
    parser.add_argument(
        "--sizes-mb",
        nargs="+",
        default=["0.25", "1", "16", "64"],
        help="Payload sizes in MB (float).",
    )
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    args = parser.parse_args()

    rank, world_size, device_id = init_distributed()
    device = torch.device("cuda", device_id)

    if rank == 0:
        print(
            f"[baseline] world_size={world_size}, device={device}, "
            f"sizes_mb={args.sizes_mb}, iters={args.iters}"
        )

    sizes_mb = parse_sizes(args.sizes_mb)
    results = []
    for sz in sizes_mb:
        avg_ms, gbps = benchmark_allreduce(sz, args.iters, args.warmup, device, world_size)
        results.append((sz, avg_ms, gbps))

    dist.barrier()
    if rank == 0:
        print("\n[baseline] NCCL AllReduce results")
        print(f"{'MB':>8} | {'avg ms':>8} | {'GB/s':>8}")
        print("-" * 32)
        for sz, avg_ms, gbps in results:
            print(f"{sz:8.2f} | {avg_ms:8.3f} | {gbps:8.2f}")

    dist.barrier()


if __name__ == "__main__":
    main()


def get_benchmark():
    """Harness entrypoint."""
    def _run():
        main()
    return _run
