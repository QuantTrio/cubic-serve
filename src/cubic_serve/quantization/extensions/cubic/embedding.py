"""Format-owned immutable embedding packets; original Cubic codes are preserved."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from .linear import arguments


def row_source(operators, weight, device, dtype):
    """Return CPU packets, native operands and the format-owned decoder.

    Each row carries its original codes, metadata and curve coefficients. No
    re-quantization, scale fitting or change of logical vocabulary ownership.
    """
    gpu = operators.materialize_weight(weight, device)
    args, fmt = arguments(operators, gpu)
    if fmt != 2 or not 2 <= weight.encoding.bits <= 8:
        return None
    w, s, _, a, b, _, _, _, part = args
    n, k = weight.shape
    e = weight.encoding
    wb, mb = math.ceil(k * e.bits / 8), math.ceil(k / e.group_in) * 2
    stride = math.ceil((wb + mb + 16) / 128) * 128
    packets = torch.zeros((n, stride), dtype=torch.uint8)
    payload = {name: value.cpu() for name, value in weight.payload.items()}
    groups = torch.arange(n) // e.group_out
    partitions = part.cpu()[groups].long()
    packets[:, :wb] = payload["weight_packed"].reshape(n, wb)
    packets[:, wb : wb + mb] = (
        payload["weight_metadata"][groups].contiguous().view(torch.uint8).reshape(n, mb)
    )
    packets[:, wb + mb : wb + mb + 8] = (
        payload["weight_curve_a"]
        .reshape(-1, 4)[partitions]
        .contiguous()
        .view(torch.uint8)
        .reshape(n, 8)
    )
    packets[:, wb + mb + 8 : wb + mb + 16] = (
        payload["weight_curve_b"]
        .reshape(-1, 4)[partitions]
        .contiguous()
        .view(torch.uint8)
        .reshape(n, 8)
    )
    defines = (
        f"#define MIRROR_BITS {e.bits}\n#define MIRROR_GI {e.group_in}\n"
        f"#define MIRROR_GO {e.group_out}\n#define OUTPUT_BF16 {int(dtype == torch.bfloat16)}\n"
    )
    source = defines.encode() + Path(__file__).with_suffix(".cu").read_bytes()
    return (
        packets,
        (w, s, a, b, part),
        (wb, mb, stride),
        source,
        stride,
    )
