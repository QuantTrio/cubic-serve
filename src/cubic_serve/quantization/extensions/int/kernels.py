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
    N,
    K: tl.constexpr,
    FORMAT: tl.constexpr,
    BITS: tl.constexpr,
    GI: tl.constexpr,
    GO: tl.constexpr,
    STRIDE: tl.constexpr,
):
    valid = (row < N) & (col < K)
    LEGACY: tl.constexpr = FORMAT // 2
    ASYM: tl.constexpr = FORMAT % 2
    if LEGACY:  # noqa: SIM108 -- Triton compile-time branch
        bit = col // (32 // BITS) * 32 + col % (32 // BITS) * BITS
    else:
        bit = col * BITS
    idx = bit // 32
    shift = bit % 32
    lo = tl.load(W + row * STRIDE + idx, valid, 0).to(tl.uint32)
    hi = tl.load(W + row * STRIDE + tl.minimum(idx + 1, STRIDE - 1), valid, 0).to(tl.uint32)
    q = ((lo >> shift) | tl.where(shift + BITS > 32, hi << (32 - shift), 0)) & ((1 << BITS) - 1)
    q = q.to(tl.float32) - (1 << (BITS - 1))
    groups: tl.constexpr = triton.cdiv(K, GI)
    scale = tl.load(S + row * groups + col // GI, valid, 0)
    if ASYM:
        if LEGACY:
            zb = row // (32 // BITS) * 32 + row % (32 // BITS) * BITS
            ZWORDS = (N + 32 // BITS - 1) // (32 // BITS)
        else:
            zb = row * BITS
            ZWORDS = (N * BITS + 31) // 32
        zi = zb // 32
        zs = zb % 32
        zl = tl.load(Z + zi * groups + col // GI, valid, 0).to(tl.uint32)
        zh = tl.load(Z + tl.minimum(zi + 1, ZWORDS - 1) * groups + col // GI, valid, 0).to(
            tl.uint32
        )
        z = ((zl >> zs) | tl.where(zs + BITS > 32, zh << (32 - zs), 0)) & ((1 << BITS) - 1)
        q = q - (z.to(tl.float32) - (1 << (BITS - 1)))
    value = (q * scale.to(tl.float32)).to(S.dtype.element_ty).to(tl.float32)
    return tl.where(valid, value, 0.0)
