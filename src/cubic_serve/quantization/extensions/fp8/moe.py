from __future__ import annotations

from typing import TYPE_CHECKING

from cubic_serve.quantization.api import RoutedOperands

from .linear import arguments

if TYPE_CHECKING:
    from torch import Tensor

    from cubic_serve.models.quantization import QuantizationScheme
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.quantization.common.binding import PackedOperatorBinding

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]

from cubic_serve.models.quantization import ActivationFormat

from .kernels import fp8_e4m3fn_decode, load_weight


@triton.jit  # type: ignore[untyped-decorator]
def compute(  # type: ignore[no-untyped-def]
    X,
    XS,
    pointers,
    input_rows,
    valid,
    n,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    CONFIG: tl.constexpr,
):
    FORMAT: tl.constexpr = CONFIG[0]
    ACTIVATION: tl.constexpr = CONFIG[1]
    BITS: tl.constexpr = CONFIG[2]
    GI: tl.constexpr = CONFIG[3]
    GO: tl.constexpr = CONFIG[4]
    STRIDE: tl.constexpr = CONFIG[5]
    W, S, Z, A, B, GS, GA, GB, PART = pointers[:9]
    accumulator = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + tl.arange(0, BK)
        x = tl.load(
            X + input_rows[:, None] * K + k[None, :], valid[:, None] & (k[None, :] < K), 0.0
        )
        if FORMAT == -1 or ACTIVATION == 1:
            w = tl.load(W + n[None, :] * K + k[:, None], (n[None, :] < N) & (k[:, None] < K), 0.0)
        else:
            w = load_weight(
                W,
                S,
                Z,
                A,
                B,
                GS,
                GA,
                GB,
                PART,
                n[None, :],
                k[:, None],
                N,
                K,
                FORMAT,
                BITS,
                GI,
                GO,
                STRIDE,
            ).to(X.dtype.element_ty)
        if ACTIVATION == 1:
            x, w = fp8_e4m3fn_decode(x), fp8_e4m3fn_decode(w)
        accumulator = tl.dot(x, w, accumulator)
    if ACTIVATION == 1:
        accumulator = (
            accumulator
            * tl.load(XS + input_rows, valid, 0)[:, None]
            * tl.load(S + n, n < N, 0)[None, :]
        )
    return accumulator


def prepare(
    operators: PackedLinearOperators,
    x: Tensor,
    encoding: QuantizationScheme,
    row_parallel: bool,
    cache: dict[int, tuple[Tensor, Tensor, PackedOperatorBinding]],
) -> tuple[Tensor, Tensor, PackedOperatorBinding | None, int]:
    if encoding.activation is ActivationFormat.IDENTITY:
        return x, x, None, 0
    key = id(x)
    if key not in cache:
        cache[key] = operators.bind_routed_fp8_input(x, row_parallel=row_parallel)
    codes, scales, binding = cache[key]
    return codes.view(torch.uint8), scales, binding, 1


def operands(
    operators: PackedLinearOperators, weights: tuple[QuantizedMatrix, ...]
) -> RoutedOperands:
    matrix = weights[0]
    e = matrix.encoding
    args, variant = arguments(operators, matrix)
    arrays = tuple(arguments(operators, w)[0] for w in weights)
    activation = 0 if e.activation is ActivationFormat.IDENTITY else 1
    stride = matrix.payload.get("weight_packed", args[0]).shape[-1]
    return RoutedOperands(
        arrays, (variant, activation, e.bits, e.group_in, e.group_out, stride), compute
    )
