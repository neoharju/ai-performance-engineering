"""Shared verification helper for ch15 multi-GPU/disaggregated benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class VerificationPayload:
    inputs: Dict[str, torch.Tensor]
    output: torch.Tensor
    batch_size: int
    parameter_count: int
    precision_flags: Dict[str, bool]


class VerificationPayloadMixin:
    """Mixin that provides strict verification methods for ch15 benchmarks."""

    def _set_verification_payload(
        self,
        *,
        inputs: Dict[str, torch.Tensor],
        output: torch.Tensor,
        batch_size: int,
        parameter_count: int = 0,
        precision_flags: Optional[Dict[str, bool]] = None,
    ) -> None:
        if not inputs:
            raise ValueError("inputs must be a non-empty dict of tensors")
        if output is None:
            raise ValueError("output tensor is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        flags = precision_flags or {
            "fp16": False,
            "bf16": False,
            "fp8": False,
            "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
        }
        self._verification_payload = VerificationPayload(
            inputs=inputs,
            output=output,
            batch_size=int(batch_size),
            parameter_count=int(parameter_count),
            precision_flags=flags,
        )

    def get_verify_inputs(self) -> Dict[str, torch.Tensor]:
        payload: VerificationPayload = getattr(self, "_verification_payload", None)
        if payload is None:
            raise RuntimeError("_set_verification_payload() must be called in setup()")
        return payload.inputs

    def get_verify_output(self) -> torch.Tensor:
        payload: VerificationPayload = getattr(self, "_verification_payload", None)
        if payload is None:
            raise RuntimeError("_set_verification_payload() must be called in setup()")
        return payload.output.detach().clone()

    def get_input_signature(self) -> Dict[str, object]:
        payload: VerificationPayload = getattr(self, "_verification_payload", None)
        if payload is None:
            raise RuntimeError("_set_verification_payload() must be called in setup()")
        shapes = {name: tuple(t.shape) for name, t in payload.inputs.items()}
        dtypes = {name: str(t.dtype) for name, t in payload.inputs.items()}
        shapes["output"] = tuple(payload.output.shape)
        dtypes["output"] = str(payload.output.dtype)
        return {
            "shapes": shapes,
            "dtypes": dtypes,
            "batch_size": payload.batch_size,
            "parameter_count": payload.parameter_count,
            "precision_flags": payload.precision_flags,
        }
