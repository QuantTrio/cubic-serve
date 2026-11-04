from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.quantization.common.binding import PackedOperatorBinding
    from cubic_serve.quantization.common.parameters import PackedLinearParameters

import torch
import triton  # type: ignore[import-untyped]

from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.quantization.common.kernels import _embedding, _weight_only_mm

from .kernels import load_weight


def arguments(
    operators: PackedLinearOperators, matrix: QuantizedMatrix
) -> tuple[tuple[Tensor, ...], int]:
    p = matrix.payload
    dummy = next(iter(p.values()))
    if dummy.dtype is torch.float8_e4m3fn:
        dummy = dummy.view(torch.uint8)
    return (
        p["qweight"],
        p["scales"],
        p["qzeros"],
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
    ), 0


def bind_linear(
    operators: PackedLinearOperators,
    x: Tensor,
    weight: QuantizedMatrix,
    output: Tensor,
    params: PackedLinearParameters,
    fp8_token_scales: Tensor | None = None,
) -> PackedOperatorBinding:
    e = weight.encoding
    n, k = weight.shape
    m = x.shape[0]
    bm, bn, bk = params.fixed_block_m, params.fixed_block_n, params.fixed_block_k
    options = dict(
        num_warps=params.fixed_warps, num_stages=params.fixed_stages, enable_fp_fusion=False
    )
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    compile_launch = _PackedLaunch.compile
    scratch: tuple[Tensor, ...] = ()
    calls: tuple[_PackedLaunch, ...]
    args, fmt = arguments(operators, weight)
    calls = (
        compile_launch(
            _weight_only_mm,
            (x, *args, output),
            (
                m,
                n,
                k,
                load_weight,
                fmt,
                e.bits,
                e.group_in,
                e.group_out,
                triton.cdiv(k * e.bits, 8),
                bm,
                bn,
                bk,
            ),
            grid,
            options,
        ),
    )
    return PackedOperatorBinding(output, calls, scratch=scratch)


def bind_embedding(
    operators: PackedLinearOperators,
    input_ids: Tensor,
    weight: QuantizedMatrix,
    output: Tensor,
    params: PackedLinearParameters,
) -> PackedOperatorBinding:
    args, fmt = arguments(operators, weight)
    e = weight.encoding
    n, k = weight.shape
    block = params.embedding_block_k
    call = _PackedLaunch.compile(
        _embedding,
        (input_ids, *args, output),
        (
            n,
            k,
            load_weight,
            fmt,
            e.bits,
            e.group_in,
            e.group_out,
            triton.cdiv(k * e.bits, 8),
            block,
        ),
        (input_ids.numel(), triton.cdiv(k, block)),
        {"num_warps": params.fixed_warps, "enable_fp_fusion": False},
    )
    return PackedOperatorBinding(output, (call,))
