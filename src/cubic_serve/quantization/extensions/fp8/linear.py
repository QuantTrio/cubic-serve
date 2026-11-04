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

from .kernels import _fp8_mm, _fp8_quant, _fp8_token_scale, load_weight
from .marlin import bind_grouped as bind_grouped
from .marlin import bind_linear as bind_native


def arguments(
    operators: PackedLinearOperators, matrix: QuantizedMatrix
) -> tuple[tuple[Tensor, ...], int]:
    p = matrix.payload
    dummy = next(iter(p.values()))
    if dummy.dtype is torch.float8_e4m3fn:
        dummy = dummy.view(torch.uint8)
    return (
        p["weight"].view(torch.uint8),
        p["weight_scale"],
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
    ), 4


def bind_linear(
    operators: PackedLinearOperators,
    x: Tensor,
    weight: QuantizedMatrix,
    output: Tensor,
    params: PackedLinearParameters,
    fp8_token_scales: Tensor | None = None,
) -> PackedOperatorBinding:
    candidate = bind_native(operators, x, weight, output, params, fp8_token_scales)
    if candidate is not None:
        return candidate
    weight = operators.materialize_weight(weight, x.device)
    p, e = weight.payload, weight.encoding
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
    if e.activation is ActivationFormat.FP8_TOKEN:
        q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scales = (
            torch.empty(m, dtype=torch.float32, device=x.device)
            if fp8_token_scales is None
            else fp8_token_scales
        )
        scratch = (q, scales) if fp8_token_scales is None else (q,)
        calls = (
            compile_launch(
                _fp8_quant,
                (x, q.view(torch.uint8), scales),
                (k, triton.next_power_of_2(k), fp8_token_scales is not None),
                (m,),
                {"enable_fp_fusion": False},
            ),
            compile_launch(
                _fp8_mm,
                (
                    q.view(torch.uint8),
                    scales,
                    p["weight"].view(torch.uint8),
                    p["weight_scale"],
                    output,
                ),
                (m, n, k, bm, bn, bk),
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


def bind_token_scales(x: Tensor, scales: Tensor) -> PackedOperatorBinding:
    """Bind local scale production before a caller-owned topology MAX.

    max(round(max(abs(x_rank))/448), floor) followed by rank MAX is
    bit-identical to the scale from the concatenated finite logical input:
    correctly rounded division and the shared floor are monotone. No SUM
    reduction or weight mutation is involved.
    """
    if (
        x.ndim != 2
        or min(x.shape) <= 0
        or not x.is_cuda
        or x.dtype not in (torch.float16, torch.bfloat16)
        or not x.is_contiguous()
        or scales.shape != (x.shape[0],)
        or scales.dtype is not torch.float32
        or scales.device != x.device
        or not scales.is_contiguous()
        or scales.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    ):
        raise ValueError("FP8 token scale binding requires private FP32 [M] and FP16/BF16 [M,K]")
    return PackedOperatorBinding(
        scales,
        (
            _PackedLaunch.compile(
                _fp8_token_scale,
                (x, scales),
                (x.shape[1], triton.next_power_of_2(x.shape[1])),
                (x.shape[0],),
                {"enable_fp_fusion": False},
            ),
        ),
    )


def bind_routed_input(
    x: Tensor,
    *,
    row_parallel: bool = False,
) -> tuple[Tensor, Tensor, PackedOperatorBinding]:
    """Prepare expert route inputs; distributed owners override the scale domain."""
    if row_parallel:
        raise ValueError("row-parallel expert FP8 requires a topology scale producer")
    codes = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    binding = PackedOperatorBinding(
        codes,
        (
            _PackedLaunch.compile(
                _fp8_quant,
                (x, codes.view(torch.uint8), scales),
                (x.shape[1], triton.next_power_of_2(x.shape[1]), False),
                (x.shape[0],),
                {"enable_fp_fusion": False},
            ),
        ),
        scratch=(codes, scales),
    )
    return codes, scales, binding
