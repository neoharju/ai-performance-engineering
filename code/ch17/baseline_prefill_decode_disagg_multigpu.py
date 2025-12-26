"""Baseline disaggregated prefill/decode benchmark (multi-GPU torchrun).

Chapter 17: Scaling Disaggregated Prefill and Decode Pipelines

Baseline behavior is serialized: decode waits until all prefills complete.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.benchmark.verification import PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)

_ENV_WORLD_SIZE = "AISP_DISAGG_WORLD_SIZE"


@dataclass(frozen=True)
class PrefillDecodeConfig:
    hidden_size: int = 2048
    num_layers: int = 4
    batch_size: int = 1
    requests_per_rank: int = 4
    context_window: int = 256
    decode_tokens: int = 128
    dtype: torch.dtype = torch.bfloat16

    @property
    def tokens_per_request(self) -> int:
        return self.context_window + self.decode_tokens


@dataclass
class _LocalPair:
    prefill_device: torch.device
    decode_device: torch.device
    prefill_model: "TinyPrefillDecode"
    decode_model: "TinyPrefillDecode"
    prompts: torch.Tensor


class TinyPrefillDecode(nn.Module):
    """Simple prefill/decode model to emulate KV cache traffic."""

    def __init__(self, hidden_size: int, num_layers: int, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)
        ])
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to(device=device, dtype=dtype)

    def prefill(self, prompts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = prompts
        for layer in self.layers:
            x = torch.relu(layer(x))
        logits = self.proj(x)
        kv_cache = x
        seed = logits[:, -1, :]
        return kv_cache, seed

    def decode(self, seed: torch.Tensor, kv_cache: torch.Tensor, decode_tokens: int) -> torch.Tensor:
        x = seed
        context = kv_cache.shape[1]
        for step in range(decode_tokens):
            kv = kv_cache[:, step % context, :]
            x = x + kv
            for layer in self.layers:
                x = torch.relu(layer(x))
            x = self.proj(x)
        return x


def _resolve_world_size() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for disaggregated prefill/decode")
    available = torch.cuda.device_count()
    if available < 2:
        raise RuntimeError("prefill_decode_disagg_multigpu requires >=2 GPUs.")
    override = os.getenv(_ENV_WORLD_SIZE)
    if override is not None:
        requested = int(override)
        if requested < 2:
            raise RuntimeError(f"{_ENV_WORLD_SIZE} must be >= 2 (got {requested}).")
        if requested > available:
            raise RuntimeError(
                f"{_ENV_WORLD_SIZE}={requested} exceeds available GPUs ({available})."
            )
        if requested % 2 != 0:
            raise RuntimeError(f"{_ENV_WORLD_SIZE} must be even for prefill/decode pairing.")
        return requested
    if available % 2 != 0:
        raise RuntimeError(
            f"{_ENV_WORLD_SIZE} must be set to an even value when device_count={available}."
        )
    return available


def _init_distributed() -> Tuple[int, int, torch.device]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for disaggregated prefill/decode")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Run with torchrun (missing RANK/WORLD_SIZE env vars).")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def _run_prefill(
    cfg: PrefillDecodeConfig,
    model: TinyPrefillDecode,
    prompts: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    kv_chunks: List[torch.Tensor] = []
    seed_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for req_idx in range(cfg.requests_per_rank):
            request_prompt = prompts[req_idx]
            kv_cache, seed = model.prefill(request_prompt)
            kv_chunks.append(kv_cache)
            seed_chunks.append(seed)
    return kv_chunks, seed_chunks


def _run_decode(
    cfg: PrefillDecodeConfig,
    model: TinyPrefillDecode,
    kv_chunks: List[torch.Tensor],
    seed_chunks: List[torch.Tensor],
) -> List[torch.Tensor]:
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for kv_cache, seed in zip(kv_chunks, seed_chunks):
            outputs.append(model.decode(seed, kv_cache, cfg.decode_tokens))
    return outputs


def _run_torchrun_worker(
    cfg: PrefillDecodeConfig,
    *,
    overlap: bool,
    label: str,
    iters: int,
    warmup: int,
) -> None:
    rank, world_size, device = _init_distributed()
    expected_world_size = _resolve_world_size()
    if world_size != expected_world_size:
        raise RuntimeError(
            f"Expected world_size={expected_world_size} (set {_ENV_WORLD_SIZE}), got {world_size}."
        )
    if world_size % 2 != 0:
        raise RuntimeError("world_size must be even (prefill ranks + decode ranks)")

    num_pairs = world_size // 2
    is_prefill = rank < num_pairs
    pair_id = rank if is_prefill else rank - num_pairs
    peer_rank = pair_id + num_pairs if is_prefill else pair_id

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = TinyPrefillDecode(cfg.hidden_size, cfg.num_layers, device, cfg.dtype).eval()

    prompts: Optional[torch.Tensor] = None
    if is_prefill:
        prompts = torch.randn(
            cfg.requests_per_rank,
            cfg.batch_size,
            cfg.context_window,
            cfg.hidden_size,
            device=device,
            dtype=cfg.dtype,
        )

    def run_iteration() -> List[torch.Tensor]:
        if is_prefill:
            kv_chunks, seed_chunks = _run_prefill(cfg, model, prompts)
            if overlap:
                handles = []
                for kv_cache, seed in zip(kv_chunks, seed_chunks):
                    handles.append(dist.isend(kv_cache, dst=peer_rank))
                    handles.append(dist.isend(seed, dst=peer_rank))
                for handle in handles:
                    handle.wait()
            else:
                for kv_cache, seed in zip(kv_chunks, seed_chunks):
                    dist.send(kv_cache, dst=peer_rank)
                    dist.send(seed, dst=peer_rank)
            if not overlap:
                dist.barrier()
            return []

        kv_chunks = []
        seed_chunks = []
        for _ in range(cfg.requests_per_rank):
            kv_buf = torch.empty(
                (cfg.batch_size, cfg.context_window, cfg.hidden_size),
                device=device,
                dtype=cfg.dtype,
            )
            seed_buf = torch.empty((cfg.batch_size, cfg.hidden_size), device=device, dtype=cfg.dtype)
            dist.recv(kv_buf, src=peer_rank)
            dist.recv(seed_buf, src=peer_rank)
            kv_chunks.append(kv_buf)
            seed_chunks.append(seed_buf)

        if not overlap:
            dist.barrier()
        if overlap:
            outputs = []
            for kv_cache, seed in zip(kv_chunks, seed_chunks):
                outputs.append(model.decode(seed, kv_cache, cfg.decode_tokens))
            return outputs
        return _run_decode(cfg, model, kv_chunks, seed_chunks)

    dist.barrier()
    torch.cuda.synchronize(device)

    for _ in range(max(warmup, 0)):
        run_iteration()
    torch.cuda.synchronize(device)
    dist.barrier()

    start = time.perf_counter()
    for _ in range(max(iters, 1)):
        run_iteration()
    torch.cuda.synchronize(device)
    dist.barrier()
    elapsed = time.perf_counter() - start

    if rank == 0:
        total_requests = cfg.requests_per_rank * num_pairs * cfg.batch_size
        tokens_per_iter = total_requests * cfg.tokens_per_request
        tokens_per_s = tokens_per_iter * (max(iters, 1) / max(elapsed, 1e-9))
        time_per_iter_ms = (elapsed / max(iters, 1)) * 1000.0
        print(f"rank0 {label} tokens/s: {tokens_per_s:.2f} tokens/s")
        print(f"rank0 {label} time_per_iter_ms: {time_per_iter_ms:.3f}")

    dist.destroy_process_group()


class _PrefillDecodeMultiGPUBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Shared multi-GPU disaggregated prefill/decode harness."""

    def __init__(self, *, overlap: bool, label: str) -> None:
        super().__init__()
        self.cfg = PrefillDecodeConfig()
        self.world_size = _resolve_world_size()
        if self.world_size % 2 != 0:
            raise RuntimeError("world_size must be even for disaggregated prefill/decode")
        self.num_pairs = self.world_size // 2
        self.overlap = bool(overlap)
        self.label = label
        self._pairs: List[_LocalPair] = []
        self._output: Optional[torch.Tensor] = None
        self._verify_prompt: Optional[torch.Tensor] = None
        self._param_count: int = 0

        total_requests = self.cfg.requests_per_rank * self.num_pairs * self.cfg.batch_size
        tokens_per_iter = total_requests * self.cfg.tokens_per_request
        self.register_workload_metadata(
            requests_per_iteration=float(total_requests),
            tokens_per_iteration=float(tokens_per_iter),
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for disaggregated prefill/decode")
        if torch.cuda.device_count() < self.world_size:
            raise RuntimeError(
                f"SKIPPED: requires >= {self.world_size} GPUs (found {torch.cuda.device_count()})"
            )

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self._pairs = []
        total_params = 0
        for pair_id in range(self.num_pairs):
            prefill_device = torch.device(f"cuda:{pair_id}")
            decode_device = torch.device(f"cuda:{pair_id + self.num_pairs}")
            prefill_model = TinyPrefillDecode(
                self.cfg.hidden_size,
                self.cfg.num_layers,
                prefill_device,
                self.cfg.dtype,
            ).eval()
            decode_model = TinyPrefillDecode(
                self.cfg.hidden_size,
                self.cfg.num_layers,
                decode_device,
                self.cfg.dtype,
            ).eval()
            decode_model.load_state_dict(prefill_model.state_dict())
            prompts = torch.randn(
                self.cfg.requests_per_rank,
                self.cfg.batch_size,
                self.cfg.context_window,
                self.cfg.hidden_size,
                device=prefill_device,
                dtype=self.cfg.dtype,
            )
            total_params += sum(p.numel() for p in prefill_model.parameters())
            total_params += sum(p.numel() for p in decode_model.parameters())
            self._pairs.append(
                _LocalPair(
                    prefill_device=prefill_device,
                    decode_device=decode_device,
                    prefill_model=prefill_model,
                    decode_model=decode_model,
                    prompts=prompts,
                )
            )

        self._param_count = total_params
        if not self._pairs:
            raise RuntimeError("Failed to initialize prompts for verification")
        self._verify_prompt = self._pairs[0].prompts[0]
        for pair in self._pairs:
            torch.cuda.synchronize(pair.prefill_device)
            torch.cuda.synchronize(pair.decode_device)

    def benchmark_fn(self) -> None:
        if not self._pairs:
            raise RuntimeError("setup() must run before benchmark_fn()")

        outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for pair in self._pairs:
                if self.overlap:
                    for req_idx in range(self.cfg.requests_per_rank):
                        kv_cache, seed = pair.prefill_model.prefill(pair.prompts[req_idx])
                        kv_cache = kv_cache.to(pair.decode_device, non_blocking=True)
                        seed = seed.to(pair.decode_device, non_blocking=True)
                        outputs.append(
                            pair.decode_model.decode(seed, kv_cache, self.cfg.decode_tokens)
                        )
                else:
                    kv_chunks, seed_chunks = _run_prefill(self.cfg, pair.prefill_model, pair.prompts)
                    decoded = _run_decode(self.cfg, pair.decode_model, kv_chunks, seed_chunks)
                    outputs.extend(decoded)

        self._output = torch.stack([out.detach().cpu() for out in outputs], dim=0)

    def capture_verification_payload(self) -> None:
        if self._output is None or self._verify_prompt is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        tf32_enabled = torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32)
        meta_dtype = torch.float32
        self._set_verification_payload(
            inputs={
                "prompt": self._verify_prompt,
                "decode_tokens": torch.zeros((self.cfg.decode_tokens,), dtype=meta_dtype),
                "hidden_size": torch.zeros((self.cfg.hidden_size,), dtype=meta_dtype),
                "num_layers": torch.zeros((self.cfg.num_layers,), dtype=meta_dtype),
            },
            output=self._output,
            batch_size=int(self._output.shape[0]),
            parameter_count=int(self._param_count),
            precision_flags=PrecisionFlags(bf16=True, tf32=tf32_enabled),
            output_tolerance=(0.0, 0.0),
            signature_overrides={
                "world_size": self.world_size,
                "pipeline_stages": 2,
                "per_rank_batch_size": self.cfg.requests_per_rank,
                "collective_type": "send_recv",
            },
        )

    def validate_result(self) -> Optional[str]:
        if self._output is None:
            return "No output captured"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=self.world_size,
            iterations=4,
            warmup=3,
            multi_gpu_required=True,
            measurement_timeout_seconds=900,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        return TorchrunLaunchSpec(
            script_path=Path(__file__).resolve(),
            script_args=[],
            env={
                _ENV_WORLD_SIZE: str(self.world_size),
                "NCCL_DEBUG": "WARN",
                "NCCL_P2P_LEVEL": "NVL",
                "NCCL_P2P_DISABLE": "0",
            },
            parse_rank0_only=True,
            multi_gpu_required=True,
            name=self.label,
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )


class BaselinePrefillDecodeDisaggMultiGPUBenchmark(_PrefillDecodeMultiGPUBenchmark):
    """Serialized prefill then decode across multi-GPU ranks."""

    def __init__(self) -> None:
        super().__init__(overlap=False, label="baseline_prefill_decode_disagg_multigpu")


def get_benchmark() -> BaseBenchmark:
    return BaselinePrefillDecodeDisaggMultiGPUBenchmark()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _run_torchrun_worker(
        PrefillDecodeConfig(),
        overlap=False,
        label="baseline_prefill_decode_disagg_multigpu",
        iters=int(args.iters),
        warmup=int(args.warmup),
    )


if __name__ == "__main__":
    main()
