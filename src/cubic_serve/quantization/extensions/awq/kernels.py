from __future__ import annotations

import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]


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
    lane = (row % 2) * 4 + (row % 8) // 2
    code = (tl.load(W + col * (N // 8) + row // 8, valid, 0.0) >> (lane * 4)) & 15
    zero = (tl.load(Z + (col // GI) * (N // 8) + row // 8, valid, 0.0) >> (lane * 4)) & 15
    scale = tl.load(S + (col // GI) * N + row, valid, 0.0).to(tl.float32)
    result = (code.to(tl.float32) - zero.to(tl.float32)) * scale
    return tl.where(valid, result, 0.0)
