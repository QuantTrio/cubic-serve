"""Portable E4M3FN conversion over original byte storage.

CUDA 13 cuda_fp8.hpp uses bias 7, three fraction bits, nearest-even rounding
and finite saturation. vLLM 89af4f1f quantization/w8a8/fp8/nvidia/quant_utils.cuh
and SGLang ba433bb4 per_token_quant_fp8.cuh use that conversion contract.
llama.cpp 458681e1 common.cuh also exposes the software exponent/mantissa
rule; its UE4M3 scale factor and NaN-to-zero convention do not apply here.

Byte carriers avoid Triton's SM89+ fp8e4nv IR restriction. No device selector,
weight expansion, intermediate FP16 rounding of FP32 input, or format change.
"""

from __future__ import annotations

import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from triton.language.extra import libdevice  # type: ignore[import-untyped]


@triton.jit
def fp8_e4m3fn_encode(x):  # type: ignore[no-untyped-def]
    """FP32 -> saturated E4M3FN bytes, RNE including subnormal ties."""
    x = x.to(tl.float32)
    bits = x.to(tl.uint32, bitcast=True)
    sign = (bits >> 24) & 0x80
    magnitude = tl.where(x == x, tl.minimum(tl.abs(x), 448.0), 0.0)
    word = magnitude.to(tl.uint32, bitcast=True)
    # Discard 20 FP32 fraction bits; add half-minus-one plus retained LSB
    # so a midpoint rounds to the even destination significand.
    normal = ((word + 0x7FFFF + ((word >> 20) & 1)) >> 20) - 960
    subnormal = libdevice.nearbyint(magnitude * 512.0).to(tl.uint32)
    encoded = tl.where(magnitude < 0.015625, subnormal, normal) | sign
    return tl.where(x == x, encoded, 0x7F).to(tl.uint8)


@triton.jit
def fp8_e4m3fn_decode(code):  # type: ignore[no-untyped-def]
    """E4M3FN bytes -> exact FP16 for every finite encoding, including -0."""
    raw = code.to(tl.uint32)
    exponent, fraction = (raw >> 3) & 15, raw & 7
    normal = ((exponent + 8) << 10) | (fraction << 7)
    subnormal = (fraction.to(tl.float32) * 0.001953125).to(tl.float16)
    half = tl.where(exponent == 0, subnormal.to(tl.uint16, bitcast=True), normal)
    half = tl.where((raw & 0x7F) == 0x7F, 0x7E00, half) | ((raw & 0x80) << 8)
    return half.to(tl.uint16).to(tl.float16, bitcast=True)


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
    result = fp8_e4m3fn_decode(tl.load(W + row * K + col, valid, 0)).to(tl.float32)
    result *= tl.load(S + row, row < N, 0.0)
    return tl.where(valid, result, 0.0)


@triton.jit
def _fp8_scale(max_abs):  # type: ignore[no-untyped-def]
    return tl.maximum(tl.div_rn(max_abs, 448.0), 1.0 / (448.0 * 512.0))


@triton.jit  # type: ignore[untyped-decorator]
def _fp8_token_scale(X, S, K: tl.constexpr, BLOCK: tl.constexpr):  # type: ignore[no-untyped-def]
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + k, k < K, 0.0).to(tl.float32)
    scale = _fp8_scale(tl.max(tl.abs(x), 0))
    tl.store(S + row, scale)


@triton.jit  # type: ignore[untyped-decorator]
def _fp8_quant(  # type: ignore[no-untyped-def]
    X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr, EXTERNAL_SCALE: tl.constexpr = False
):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + k, k < K, 0.0).to(tl.float32)
    if EXTERNAL_SCALE:  # noqa: SIM108 - retain an explicit compile-time branch
        scale = tl.load(S + row)
    else:
        scale = _fp8_scale(tl.max(tl.abs(x), 0))
    q = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.0), 448.0)
    tl.store(Q + row * K + k, fp8_e4m3fn_encode(q), k < K)
    if not EXTERNAL_SCALE:
        tl.store(S + row, scale)


@triton.jit  # type: ignore[untyped-decorator]
def _fp8_mm(  # type: ignore[no-untyped-def]
    X,
    XS,
    W,
    WS,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        x = fp8_e4m3fn_decode(
            tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0)
        )
        w = fp8_e4m3fn_decode(
            tl.load(W + n[None, :] * K + k[:, None], (n[None, :] < N) & (k[:, None] < K), 0)
        )
        accumulator = tl.dot(x, w, accumulator)
    value = accumulator * tl.load(XS + m, m < M, 0)[:, None] * tl.load(WS + n, n < N, 0.0)[None, :]
    tl.store(Y + m[:, None] * N + n[None, :], value, (m[:, None] < M) & (n[None, :] < N))
