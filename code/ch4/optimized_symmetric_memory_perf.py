#!/usr/bin/env python3
"""Optimized symmetric-memory perf microbench (SymmetricMemory puts only).

Measures latency/bandwidth of direct peer writes using torch.distributed.nn.SymmetricMemory.
Fails if SymmetricMemory is unavailable.
"""
from __future__ import annotations

import argparse
import datetime
import os
from typing import List, Tuple

import torch
import torch.distributed as dist


def symmetric_memory_available() -> bool:
    return hasattr(dist, "nn") and hasattr(dist.nn, "SymmetricMemory")


def init_distributed() -> Tuple[int, int, int]:
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


def benchmark_symmetric_put(
    size_mb: float, iters: int, warmup: int, device: torch.device, peer_rank: int
) -> Tuple[float, float]:
    """Return (avg_ms, effective_gbps) for direct peer copy via SymmetricMemory."""
    numel = int((size_mb * 1024 * 1024) / 4)
    local_tensor = torch.ones(numel, device=device, dtype=torch.float32)

    handle = dist.nn.SymmetricMemory(local_tensor)
    peer_buffer = handle.get_buffer(peer_rank)

    # Warmup
    for _ in range(warmup):
        peer_buffer.copy_(local_tensor, non_blocking=True)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        peer_buffer.copy_(local_tensor, non_blocking=True)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / iters

    bytes_moved = size_mb * 1024 * 1024 * iters
    gbps = (bytes_moved / (total_ms / 1000.0)) / 1e9
    return avg_ms, gbps


def parse_sizes(sizes: List[str]) -> List[float]:
    return [float(s) for s in sizes]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized symmetric-memory perf microbench")
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

    if not symmetric_memory_available():
        raise RuntimeError(
            "SymmetricMemory not available. Install PyTorch with SymmetricMemory support "
            "or run on a platform that exposes dist.nn.SymmetricMemory."
        )

    if world_size < 2:
        raise RuntimeError("SymmetricMemory peer-put microbench requires world_size >= 2.")

    peer_rank = (rank + 1) % world_size
    if rank == 0:
        print(
            f"[optimized] world_size={world_size}, device={device}, "
            f"peer={peer_rank}, sizes_mb={args.sizes_mb}, iters={args.iters}"
        )

    sizes_mb = parse_sizes(args.sizes_mb)
    results = []
    for sz in sizes_mb:
        avg_ms, gbps = benchmark_symmetric_put(sz, args.iters, args.warmup, device, peer_rank)
        results.append((sz, avg_ms, gbps))

    dist.barrier()
    if rank == 0:
        print("\n[optimized] SymmetricMemory peer-put results")
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
