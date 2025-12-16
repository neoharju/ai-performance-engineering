import multiprocessing as mp
import os
import tempfile
from typing import Any, Dict, Tuple

import pytest
import torch

from core.benchmark.verify_runner import VerifyConfig, VerifyRunner
from core.benchmark.verification_mixin import VerificationPayloadMixin


class _DummyDistributedPipelineBenchmark(VerificationPayloadMixin):
    def __init__(self, *, rank: int, world_size: int):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._verification_payload = None
        self._input: torch.Tensor | None = None
        self._output: torch.Tensor | None = None

    def setup(self) -> None:
        self._input = torch.tensor([1.0])
        self._output = None

    def benchmark_fn(self) -> None:
        self._output = torch.tensor([float(self.rank)])

    def capture_verification_payload(self) -> None:
        if self._input is None or self._output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        boundaries = [(idx, idx) for idx in range(self.world_size)]
        self._set_verification_payload(
            inputs={"input": self._input},
            output=self._output,
            batch_size=1,
            parameter_count=1,
            output_tolerance=(0.0, 0.0),
            signature_overrides={
                "world_size": self.world_size,
                "ranks": list(range(self.world_size)),
                "pipeline_stages": self.world_size,
                "pipeline_stage_boundaries": boundaries,
            },
        )


def _dist_worker(
    rank: int,
    world_size: int,
    init_method: str,
    queue: "torch.multiprocessing.SimpleQueue[Tuple[int, bool, str | None]]",
    cache_dir: str,
) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
    )
    try:
        bench = _DummyDistributedPipelineBenchmark(rank=rank, world_size=world_size)
        runner = VerifyRunner(cache_dir=cache_dir)
        result = runner.verify_distributed(
            bench,
            world_size=world_size,
            rank=rank,
            config=VerifyConfig(seed=42),
        )
        queue.put((rank, bool(result.passed), result.reason))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_pipeline_parallel_distributed_verification_allows_rank_divergent_outputs() -> None:
    world_size = 2
    if not hasattr(torch.distributed, "init_process_group"):
        pytest.skip("torch.distributed init_process_group unavailable")

    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "dist_init")
        init_method = f"file://{init_file}"
        cache_dir = os.path.join(tmpdir, "verify_cache")
        queue: Any = mp.get_context("spawn").SimpleQueue()

        torch.multiprocessing.spawn(
            _dist_worker,
            args=(world_size, init_method, queue, cache_dir),
            nprocs=world_size,
            join=True,
            daemon=False,
        )

        results: Dict[int, Tuple[bool, str | None]] = {}
        for _ in range(world_size):
            rank, passed, reason = queue.get()
            results[int(rank)] = (bool(passed), reason)

        for rank in range(world_size):
            passed, reason = results[rank]
            assert passed, f"rank {rank} failed: {reason}"
