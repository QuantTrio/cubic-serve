"""Explicit eager linear algebra oracle."""

from __future__ import annotations

import torch
from torch import Tensor

from cubic_serve.executor.runner import require_reference_execution


def linear(
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    accumulator_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
) -> Tensor:
    """Compute ``inputs @ weight.T`` with an explicit accumulator dtype.

    This intentionally avoids ``torch.nn.functional.linear`` so the semantic
    decomposition remains visible to the guardrail oracle and replay tools.
    """

    require_reference_execution()

    if inputs.ndim < 1 or weight.ndim != 2:
        raise ValueError("linear expects rank>=1 inputs and rank-2 weight")
    if inputs.shape[-1] != weight.shape[1]:
        raise ValueError("linear input width does not match weight width")
    if not inputs.is_floating_point() or not weight.is_floating_point():
        raise TypeError("linear inputs and weight must be floating point")
    if inputs.device != weight.device:
        raise ValueError("linear inputs and weight must be on the same device")
    if inputs.numel() == 0 or weight.shape[0] == 0:
        raise ValueError("reference linear does not accept an empty execution row or output")
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError("linear bias must match output width")
        if bias.device != inputs.device:
            raise ValueError("linear bias must be on the input device")

    # Execute one canonical row at a time. A packed prefill row and the same isolated decode
    # row therefore present the same GEMM shape and reduction length to the backend.
    input_width = inputs.shape[-1]
    rows = inputs.reshape(-1, input_width).to(dtype=accumulator_dtype)
    transposed_weight = weight.to(dtype=accumulator_dtype).transpose(0, 1)
    accumulated_bias = bias.to(dtype=accumulator_dtype) if bias is not None else None
    outputs: list[Tensor] = []
    for row in rows.unbind(dim=0):
        output = torch.matmul(row.unsqueeze(0), transposed_weight).squeeze(0)
        if accumulated_bias is not None:
            output = output + accumulated_bias
        outputs.append(output)
    result = torch.stack(outputs, dim=0).reshape(*inputs.shape[:-1], weight.shape[0])
    return result.to(dtype=output_dtype or inputs.dtype)
