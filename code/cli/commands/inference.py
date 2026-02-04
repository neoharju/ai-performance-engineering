"""Inference CLI commands wired to the unified PerformanceEngine."""

from __future__ import annotations

import json
import subprocess
from typing import Any, Dict

from core.engine import get_engine


def _print_result(result: Dict[str, Any], json_output: bool) -> None:
    if json_output:
        print(json.dumps(result, indent=2))
        return
    print(json.dumps(result, indent=2))


def vllm_config(args: Any) -> int:
    """Generate vLLM configuration."""
    model_size = getattr(args, "model_size", None)
    if model_size is None:
        print("model_size is required (billions of parameters).")
        return 1
    result = get_engine().inference.vllm_config(
        model=getattr(args, "model", "model"),
        model_params_b=float(model_size),
        num_gpus=int(getattr(args, "gpus", 1)),
        gpu_memory_gb=float(getattr(args, "gpu_memory_gb", 80.0)),
        target=getattr(args, "target", "throughput"),
        max_seq_length=int(getattr(args, "max_seq_length", 8192)),
        quantization=getattr(args, "quantization", None),
        compare=bool(getattr(args, "compare", False)),
    )
    _print_result(result, getattr(args, "json", False))
    return 0 if result.get("success", True) else 1


def quantize(args: Any) -> int:
    """Quantization recommendations."""
    model_size = getattr(args, "model_size", None)
    result = get_engine().inference.quantization(model_size=model_size)
    _print_result(result, getattr(args, "json", False))
    return 0 if result.get("success", True) else 1


def deploy_config(args: Any) -> int:
    """Generate deployment configuration."""
    model_size = getattr(args, "model_size", None)
    if model_size is None:
        print("model_size is required (billions of parameters).")
        return 1
    params = {
        "model": getattr(args, "model", "model"),
        "model_params_b": float(model_size),
        "num_gpus": int(getattr(args, "gpus", 1)),
        "gpu_memory_gb": float(getattr(args, "gpu_memory_gb", 80.0)),
        "goal": getattr(args, "goal", "throughput"),
        "max_seq_length": int(getattr(args, "max_seq_length", 8192)),
    }
    result = get_engine().inference.deploy(params)
    _print_result(result, getattr(args, "json", False))
    return 0 if result.get("success", True) else 1


def estimate(args: Any) -> int:
    """Estimate inference performance."""
    model_size = getattr(args, "model_size", None)
    if model_size is None:
        print("model_size is required (billions of parameters).")
        return 1
    params = {
        "model": getattr(args, "model", "model"),
        "model_params_b": float(model_size),
        "num_gpus": int(getattr(args, "gpus", 1)),
        "gpu_memory_gb": float(getattr(args, "gpu_memory_gb", 80.0)),
        "goal": getattr(args, "goal", "throughput"),
        "max_seq_length": int(getattr(args, "max_seq_length", 8192)),
    }
    result = get_engine().inference.estimate(params)
    _print_result(result, getattr(args, "json", False))
    return 0 if result.get("success", True) else 1


def serve(args: Any) -> int:
    """Generate (and optionally run) an inference server command."""
    model_size = getattr(args, "model_size", None)
    if model_size is None:
        print("model_size is required (billions of parameters).")
        return 1
    params = {
        "model": getattr(args, "model", "model"),
        "model_params_b": float(model_size),
        "num_gpus": int(getattr(args, "gpus", 1)),
        "gpu_memory_gb": float(getattr(args, "gpu_memory_gb", 80.0)),
        "goal": getattr(args, "goal", "throughput"),
        "max_seq_length": int(getattr(args, "max_seq_length", 8192)),
    }
    result = get_engine().inference.deploy(params)
    if not result.get("success", True):
        _print_result(result, getattr(args, "json", False))
        return 1
    launch_cmd = result.get("launch_command")
    if not launch_cmd:
        print("No launch command available.")
        _print_result(result, getattr(args, "json", False))
        return 1
    if getattr(args, "run", False):
        return subprocess.run(launch_cmd, shell=True).returncode
    print(launch_cmd)
    return 0
