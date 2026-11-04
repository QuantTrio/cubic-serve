from __future__ import annotations

import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]


@triton.jit  # type: ignore[untyped-decorator]
def load_levels(  # type: ignore[no-untyped-def]
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
    """Exact dtype-rounded magnitudes; the sign is an IEEE sign-bit XOR."""
    valid = (row < N) & (col < K)
    bit = col * BITS
    offset = row * STRIDE + bit // 8
    lo = tl.load(W + offset, valid, 0).to(tl.int32)
    hi = tl.load(W + offset + 1, valid & (bit // 8 + 1 < STRIDE), 0).to(tl.int32)
    raw = ((lo | (hi << 8)) >> (bit % 8)) & ((1 << BITS) - 1)
    code = tl.where(raw >= (1 << (BITS - 1)), raw - (1 << BITS), raw)
    code = tl.where(code == -(1 << (BITS - 1)), 0, code)
    group = (row // GO) * tl.cdiv(K, GI) + col // GI
    value = tl.load(S + group * (1 << (BITS - 1)) + tl.abs(code), valid, 0)
    bits = value.to(tl.uint16, bitcast=True) ^ tl.where(code < 0, 0x8000, 0).to(tl.uint16)
    return bits.to(S.dtype.element_ty, bitcast=True)


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
    bit = col * BITS
    offset = row * STRIDE + bit // 8
    lo = tl.load(W + offset, valid, 0.0).to(tl.int32)
    hi = tl.load(W + offset + 1, valid & (bit // 8 + 1 < STRIDE), 0.0).to(tl.int32)
    code = ((lo | (hi << 8)) >> (bit % 8)) & ((1 << BITS) - 1)
    if BITS == 1:
        code = code * 2 - 1
    else:
        code = tl.where(code >= (1 << (BITS - 1)), code - (1 << BITS), code)
        code = tl.where(code == -(1 << (BITS - 1)), 0, code)
    group = (row // GO) * tl.cdiv(K, GI) + col // GI
    part = tl.load(PART + row // GO, row < N, 0.0)
    if FORMAT == 1:  # compact int8 scale + signed int4 a/b
        scale = tl.load(S + group, valid, 0.0).to(tl.float32) * tl.load(GS + part)
        ab = tl.load(A + group, valid, 0.0).to(tl.int32)
        a, b = ab & 15, ab >> 4
        a, b = tl.where(a >= 8, a - 16, a), tl.where(b >= 8, b - 16, b)
        a = (1.0 + a.to(tl.float32) * tl.load(GA + part)).to(tl.float16).to(tl.float32)
        b = (b.to(tl.float32) * tl.load(GB + part)).to(tl.float16).to(tl.float32)
    elif FORMAT == 2:  # e5m9 scale plus curve index
        metadata = tl.load(S + group, valid, 0.0).to(tl.int32)
        scale = ((metadata & 0x3FFF) << 1).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        curve = metadata >> 14
        a = tl.load(A + part * 4 + curve).to(tl.float32)
        b = tl.load(B + part * 4 + curve).to(tl.float32)
    else:
        scale = tl.load(S + group, valid, 0.0)
        a = tl.load(A + group, valid, 0.0).to(tl.float32)
        b = tl.load(B + group, valid, 0.0).to(tl.float32)
    if BITS == 1:
        result = code.to(tl.float32) * scale
    else:
        if BITS <= 4:
            # Fold the tiny finite codebook into correctly rounded constants;
            # no division per weight and no extra device-memory dependency.
            # Keep the polynomial and weight-rounding boundaries unchanged.
            magnitude = tl.abs(code)
            t = tl.full(code.shape, 0.0, tl.float32)
            for level in tl.static_range(1, 1 << (BITS - 1)):
                t = tl.where(magnitude == level, level / ((1 << (BITS - 1)) - 1), t)
        else:
            # Reuse the finite quotient codebook within the CTA. Wide codes
            # otherwise invoke strict division for every reconstructed weight.
            levels = tl.div_rn(
                tl.arange(0, 1 << (BITS - 1)).to(tl.float32),
                float((1 << (BITS - 1)) - 1),
            )
            t = tl.gather(
                levels,
                tl.reshape(tl.abs(code), (code.numel,)),
                axis=0,
            ).reshape(code.shape)
        normalized = t * (a + t * (b + t * (1.0 - a - b)))
        result = tl.where(code < 0, -normalized, normalized) * scale
    return tl.where(valid, result, 0.0)


@triton.jit  # type: ignore[untyped-decorator]
def load_level_words(  # type: ignore[no-untyped-def]
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
    tl.static_assert(BITS == 4)
    if len(col.shape) == 2 and col.shape[1] <= GI and GI % col.shape[1] == 0:
        valid = (row < N) & (col < K)
        packed = tl.load(W + row * STRIDE + col // 2, valid, 0).to(tl.int32)
        raw = (packed >> ((col % 2) * 4)) & 15
        magnitude = tl.where(raw >= 8, 16 - raw, raw) & 7
        start = tl.min(col, 1)[:, None]
        group = (row // GO) * tl.cdiv(K, GI) + start // GI
        group_valid = (row < N) & (start < K)
        words = S.to(tl.pointer_type(tl.uint32))
        w0 = tl.load(words + group * 4, group_valid, 0)
        w1 = tl.load(words + group * 4 + 1, group_valid, 0)
        w2 = tl.load(words + group * 4 + 2, group_valid, 0)
        w3 = tl.load(words + group * 4 + 3, group_valid, 0)
        word = tl.where(
            (magnitude & 4) != 0,
            tl.where((magnitude & 2) != 0, w3, w2),
            tl.where((magnitude & 2) != 0, w1, w0),
        )
        bits = ((word >> ((magnitude & 1) * 16)) & 65535) ^ tl.where(raw > 8, 32768, 0).to(
            tl.uint32
        )
        return bits.to(tl.uint16).to(S.dtype.element_ty, bitcast=True)
    else:
        return load_levels(
            W, S, Z, A, B, GS, GA, GB, PART, row, col, N, K, FORMAT, BITS, GI, GO, STRIDE
        )
