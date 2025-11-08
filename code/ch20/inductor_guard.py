"""Shared helpers for torch.compile guard rails in Chapter 20 benchmarks."""

from __future__ import annotations

from common.python import compile_utils as _compile_utils_patch  # noqa: F401
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import torch

InductorCudagraphState = Optional[Tuple[Any, Dict[str, Any]]]


def disable_inductor_cudagraph_features() -> InductorCudagraphState:
    """Force-disable Inductor's cudagraph helpers and return previous state."""
    try:
        import torch._inductor.config as inductor_config  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        return None

    triton_cfg = getattr(inductor_config, "triton", None)
    if triton_cfg is None:
        return None

    previous: Dict[str, Any] = {}
    changed = False
    for attr in ("cudagraphs", "cudagraph_trees"):
        if hasattr(triton_cfg, attr):
            previous[attr] = getattr(triton_cfg, attr)
            setattr(triton_cfg, attr, False)
            changed = True

    if not changed:
        return None

    return (triton_cfg, previous)


def restore_inductor_cudagraph_features(state: InductorCudagraphState) -> None:
    """Restore cudagraph settings if they were overridden."""
    if not state:
        return

    triton_cfg, previous = state
    for attr, value in previous.items():
        try:
            setattr(triton_cfg, attr, value)
        except Exception:
            continue


@contextmanager
def inductor_cudagraph_guard():
    """Context manager that temporarily disables Inductor cudagraph helpers."""
    state = disable_inductor_cudagraph_features()
    try:
        yield state
    finally:
        restore_inductor_cudagraph_features(state)
