from __future__ import annotations

from typing import TYPE_CHECKING

from cubic_serve.quantization.api import RoutedOperands

from .layout import payload_names
from .linear import arguments
from .marlin import bind_grouped as bind_grouped
from .marlin import bind_routed as bind_routed

if TYPE_CHECKING:
    from torch import Tensor

    from cubic_serve.models.quantization import QuantizationScheme
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.quantization.common.binding import PackedOperatorBinding

import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]

from cubic_serve.models.quantization import ActivationFormat
from cubic_serve.quantization.extensions.fp8.kernels import fp8_e4m3fn_decode

from .kernels import _fp4, _nvfp4_codes_and_scale, load_weight


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
    if ACTIVATION == 2:
        XG = pointers[9]
        inverse = tl.load(XG)
        if FORMAT == 6:
            inverse = tl.div_rn(1.0, inverse)
        for group in range(K // 16):
            k = group * 16 + tl.arange(0, 16)
            x = tl.load(X + input_rows[:, None] * K + k[None, :], valid[:, None], 0).to(tl.float32)
            code, xs = _nvfp4_codes_and_scale(x, inverse)
            word = tl.load(W + n[None, :] * (K // 2) + k[:, None] // 2, n[None, :] < N, 0).to(
                tl.int32
            )
            wcode = (word >> ((k[:, None] % 2) * 4)) & 15
            partial = tl.dot(_fp4(code).to(tl.float16), _fp4(wcode).to(tl.float16))
            ws = fp8_e4m3fn_decode(tl.load(S + n * (K // 16) + group, n < N, 0)).to(tl.float32)
            accumulator += partial * xs[:, None] * ws[None, :]
        if FORMAT == 6:
            accumulator = accumulator * tl.load(XG) * tl.load(GS)
        else:
            accumulator = accumulator / inverse / tl.load(GS)
    else:
        for start in range(tl.cdiv(K, BK)):
            k = start * BK + tl.arange(0, BK)
            x = tl.load(
                X + input_rows[:, None] * K + k[None, :], valid[:, None] & (k[None, :] < K), 0.0
            )
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
            accumulator = tl.dot(x, w, accumulator)
    return accumulator


def prepare(
    operators: PackedLinearOperators,
    x: Tensor,
    encoding: QuantizationScheme,
    row_parallel: bool,
    cache: dict[int, tuple[Tensor, Tensor, PackedOperatorBinding]],
) -> tuple[Tensor, Tensor, PackedOperatorBinding | None, int]:
    return x, x, None, 2 if encoding.activation is ActivationFormat.NVFP4_LOCAL else 0


def operands(
    operators: PackedLinearOperators, weights: tuple[QuantizedMatrix, ...]
) -> RoutedOperands:
    matrix = weights[0]
    e = matrix.encoding
    args, variant = arguments(operators, matrix)
    arrays = tuple(arguments(operators, w)[0] for w in weights)
    arrays = tuple(
        (*row, w.payload[payload_names(w.encoding)[2]])
        for row, w in zip(arrays, weights, strict=True)
    )
    activation = 0 if e.activation is ActivationFormat.IDENTITY else 2
    stride = matrix.payload.get("weight_packed", args[0]).shape[-1]
    return RoutedOperands(
        arrays, (variant, activation, e.bits, e.group_in, e.group_out, stride), compute
    )
