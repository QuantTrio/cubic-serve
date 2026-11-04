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

from cubic_serve.models.quantization import ActivationFormat
from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.quantization.common.kernels import _embedding, _weight_only_mm

from .kernels import _nvfp4_mm, _nvfp4_quant, load_weight
from .layout import payload_names


def arguments(
    operators: PackedLinearOperators, matrix: QuantizedMatrix
) -> tuple[tuple[Tensor, ...], int]:
    p = matrix.payload
    packed, _, _, global_scale = payload_names(matrix.encoding)
    dummy = next(iter(p.values()))
    if dummy.dtype is torch.float8_e4m3fn:
        dummy = dummy.view(torch.uint8)
    return (
        p[packed],
        p["weight_scale"].view(torch.uint8),
        dummy,
        dummy,
        dummy,
        p[global_scale],
        dummy,
        dummy,
        dummy,
    ), 6 if matrix.encoding.metadata == "modelopt" else 5


def bind_native(
    operators: PackedLinearOperators,
    x: Tensor,
    weight: QuantizedMatrix,
    output: Tensor,
    params: PackedLinearParameters,
    fp8_token_scales: Tensor | None = None,
) -> PackedOperatorBinding | None:
    from .marlin import bind_linear as bind_marlin

    return bind_marlin(operators, x, weight, output, params)


def bind_linear(
    operators: PackedLinearOperators,
    x: Tensor,
    weight: QuantizedMatrix,
    output: Tensor,
    params: PackedLinearParameters,
    fp8_token_scales: Tensor | None = None,
) -> PackedOperatorBinding:
    candidate = bind_native(operators, x, weight, output, params)
    if candidate is not None:
        return candidate
    p, e = weight.payload, weight.encoding
    packed, _, input_global, weight_global = payload_names(e)
    direct = e.metadata == "modelopt"
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
    if e.activation is ActivationFormat.NVFP4_LOCAL:
        q = torch.empty_like(x, dtype=torch.uint8)
        scales = torch.empty((m, k // 16), dtype=torch.float8_e4m3fn, device=x.device)
        scratch = q, scales
        calls = (
            compile_launch(
                _nvfp4_quant,
                (x, p[input_global], q, scales.view(torch.uint8)),
                (k, direct),
                (m, k // 16),
                {"enable_fp_fusion": False},
            ),
            compile_launch(
                _nvfp4_mm,
                (
                    q,
                    scales.view(torch.uint8),
                    p[input_global],
                    p[packed],
                    p["weight_scale"].view(torch.uint8),
                    p[weight_global],
                    output,
                ),
                (m, n, k, bm, bn, direct),
                grid,
                options,
            ),
        )
    else:
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
