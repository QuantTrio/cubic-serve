from __future__ import annotations

import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]

from cubic_serve.quantization.extensions.fp8.kernels import fp8_e4m3fn_decode, fp8_e4m3fn_encode


@triton.jit
def _fp4(code):  # type: ignore[no-untyped-def]
    magnitude = code & 7
    value = tl.where(
        magnitude < 4,
        magnitude * 0.5,
        tl.where(
            magnitude == 4, 2.0, tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0))
        ),
    )
    return tl.where((code & 8) != 0, -value, value)


@triton.jit  # type: ignore[untyped-decorator]
def load_weight(  # type: ignore[no-untyped-def]
    W,
    S,
    Z,
    A,
    B,
    GS,
    GA,
    GB,
    PART,
    row,
    col,
    N: tl.constexpr,
    K: tl.constexpr,
    FORMAT: tl.constexpr,
    BITS: tl.constexpr,
    GI: tl.constexpr,
    GO: tl.constexpr,
    STRIDE: tl.constexpr,
):
    valid = (row < N) & (col < K)
    code = (tl.load(W + row * (K // 2) + col // 2, valid, 0.0) >> (4 * (col % 2))) & 15
    scale = fp8_e4m3fn_decode(tl.load(S + row * (K // 16) + col // 16, valid, 0)).to(tl.float32)
    result = _fp4(code) * scale
    if FORMAT == 6:  # noqa: SIM108 - compile-time Triton format branch
        result = result * tl.load(GS)
    else:
        result = result / tl.load(GS)
    return tl.where(valid, result, 0.0)


@triton.jit
def _nvfp4_codes_and_scale(x, inverse):  # type: ignore[no-untyped-def]
    scale = fp8_e4m3fn_decode(
        fp8_e4m3fn_encode(
            tl.minimum(tl.div_rn(tl.max(tl.abs(x), len(x.shape) - 1) * inverse, 6.0), 448.0)
        )
    ).to(tl.float32)
    expanded = scale[:, None] if len(x.shape) == 2 else scale
    # E4M3 scale times each E2M1 midpoint is exact in FP32. Comparing
    # the original rounded numerator preserves RN-even codes without a division.
    y = tl.abs(x) * inverse
    code = (
        (y > expanded * 0.25).to(tl.int32)
        + (y >= expanded * 0.75).to(tl.int32)
        + (y > expanded * 1.25).to(tl.int32)
        + (y >= expanded * 1.75).to(tl.int32)
        + (y > expanded * 2.5).to(tl.int32)
        + (y >= expanded * 3.5).to(tl.int32)
        + (y > expanded * 5.0).to(tl.int32)
    )
    code = code | tl.where((x.to(tl.int32, bitcast=True) & -2147483648) != 0, 8, 0)
    return tl.where(expanded > 0, code, 0), scale


@triton.jit  # type: ignore[untyped-decorator]
def _nvfp4_quant(X, GLOBAL, Q, S, K: tl.constexpr, DIRECT: tl.constexpr = False):  # type: ignore[no-untyped-def]
    row, group = tl.program_id(0), tl.program_id(1)
    k = group * 16 + tl.arange(0, 16)
    x = tl.load(X + row * K + k).to(tl.float32)
    inverse = tl.load(GLOBAL)
    if DIRECT:
        inverse = tl.div_rn(1., inverse)
    code, scale = _nvfp4_codes_and_scale(x, inverse)
    tl.store(Q + row * K + k, code)
    tl.store(S + row * (K // 16) + group, fp8_e4m3fn_encode(scale))


@triton.jit  # type: ignore[untyped-decorator]
def _nvfp4_mm(  # type: ignore[no-untyped-def]
    X,
    XS,
    XG,
    W,
    WS,
    WG,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    DIRECT: tl.constexpr = False,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, 16)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for group in range(K // 16):
        k = group * 16 + kk
        xcode = tl.load(X + m[:, None] * K + k[None, :], m[:, None] < M, 0.0).to(tl.int32)
        word = tl.load(W + n[None, :] * (K // 2) + k[:, None] // 2, n[None, :] < N, 0.0).to(
            tl.int32
        )
        wcode = (word >> ((k[:, None] % 2) * 4)) & 15
        partial = tl.dot(_fp4(xcode).to(tl.float16), _fp4(wcode).to(tl.float16))
        xs = fp8_e4m3fn_decode(tl.load(XS + m * (K // 16) + group, m < M, 0)).to(tl.float32)
        ws = fp8_e4m3fn_decode(tl.load(WS + n * (K // 16) + group, n < N, 0)).to(tl.float32)
        accumulator += partial * xs[:, None] * ws[None, :]
    if DIRECT:
        value = accumulator * tl.load(XG) * tl.load(WG)
    else:
        value = accumulator / tl.load(XG) / tl.load(WG)
    tl.store(Y + m[:, None] * N + n[None, :], value, (m[:, None] < M) & (n[None, :] < N))
