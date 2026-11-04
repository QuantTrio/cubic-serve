from __future__ import annotations

import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]


@triton.jit  # type: ignore[untyped-decorator]
def _copy_projection_rows(  # type: ignore[no-untyped-def]
    X, Y, COUNT: tl.constexpr, N: tl.constexpr, PADDED_N: tl.constexpr, BLOCK: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr = 0, OUTPUT_OFFSET: tl.constexpr = 0,
    INPUT_GROUP_N: tl.constexpr = 0, INPUT_GROUP_PADDED_N: tl.constexpr = 0,
):
    """Copy logical projection rows between padded/grouped/strided buffers."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    column = index % N
    if INPUT_GROUP_N:
        column = column // INPUT_GROUP_N * INPUT_GROUP_PADDED_N + column % INPUT_GROUP_N
    value = tl.load(X + index // N * PADDED_N + column, index < COUNT, 0)
    stride = OUTPUT_STRIDE if OUTPUT_STRIDE else N
    tl.store(Y + index // N * stride + index % N + OUTPUT_OFFSET, value, index < COUNT)


@triton.jit  # type: ignore[untyped-decorator]
def _plain_mm(  # type: ignore[no-untyped-def]
    X,
    W,
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
        x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0.0)
        w = tl.load(W + n[None, :] * K + k[:, None], (n[None, :] < N) & (k[:, None] < K), 0.0)
        accumulator = tl.dot(x, w, accumulator)
    tl.store(Y + m[:, None] * N + n[None, :], accumulator, (m[:, None] < M) & (n[None, :] < N))


@triton.jit  # type: ignore[untyped-decorator]
def _plain_partition_mm(  # type: ignore[no-untyped-def]
    X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PARTITIONS: tl.constexpr,
):
    """Contiguous fixed K partitions; FP32 scratch, no atomics or output rounding."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, BK)
    partition = tl.program_id(2)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for step in range(K // (BK * PARTITIONS)):
        kk = partition * (K // PARTITIONS) + step * BK + offsets
        x = tl.load(X + rows[:, None] * K + kk[None, :], rows[:, None] < M, 0)
        w = tl.load(W + columns[None, :] * K + kk[:, None], columns[None, :] < N, 0)
        accumulator = tl.dot(x, w, accumulator)
    tl.store(Y + partition * M * N + rows[:, None] * N + columns[None, :],
             accumulator, (rows[:, None] < M) & (columns[None, :] < N))


@triton.jit  # type: ignore[untyped-decorator]
def _weight_only_mm(  # type: ignore[no-untyped-def]
    X,
    W,
    S,
    Z,
    A,
    B,
    GS,
    GA,
    GB,
    PART,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    LOADER: tl.constexpr,
    FORMAT: tl.constexpr,
    BITS: tl.constexpr,
    GI: tl.constexpr,
    GO: tl.constexpr,
    STRIDE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
    WEIGHT_K_CONTIGUOUS: tl.constexpr = False,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), tl.float32)
    partition = tl.program_id(2)
    for start in range(tl.cdiv(K, BK * SPLIT_K)):
        k = (start * SPLIT_K + partition) * BK + kk
        x = tl.load(X + rows[:, None] * K + k[None, :], (rows[:, None] < M) & (k[None, :] < K), 0.0)
        if WEIGHT_K_CONTIGUOUS:
            weight_rows, weight_cols = cols[:, None], k[None, :]
        else:
            weight_rows, weight_cols = cols[None, :], k[:, None]
        weight = LOADER(
            W,
            S,
            Z,
            A,
            B,
            GS,
            GA,
            GB,
            PART,
            weight_rows,
            weight_cols,
            N,
            K,
            FORMAT,
            BITS,
            GI,
            GO,
            STRIDE,
        ).to(X.dtype.element_ty)
        if WEIGHT_K_CONTIGUOUS:
            weight = weight.T
        accumulator = tl.dot(x, weight, accumulator)
    tl.store(
        Y + partition * M * N + rows[:, None] * N + cols[None, :],
        accumulator,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit  # type: ignore[untyped-decorator]
def _embedding(  # type: ignore[no-untyped-def]
    IDS,
    W,
    S,
    Z,
    A,
    B,
    GS,
    GA,
    GB,
    PART,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    LOADER: tl.constexpr,
    FORMAT: tl.constexpr,
    BITS: tl.constexpr,
    GI: tl.constexpr,
    GO: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    row = tl.load(IDS + token)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = LOADER(W, S, Z, A, B, GS, GA, GB, PART, row, col, N, K, FORMAT, BITS, GI, GO, STRIDE)
    tl.store(Y + token * K + col, value, col < K)


@triton.jit  # type: ignore[untyped-decorator]
def plain_compute(  # type: ignore[no-untyped-def]
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
    W = pointers[0]
    accumulator = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + tl.arange(0, BK)
        x = tl.load(
            X + input_rows[:, None] * K + k[None, :], valid[:, None] & (k[None, :] < K), 0.0
        )
        w = tl.load(W + n[None, :] * K + k[:, None], (n[None, :] < N) & (k[:, None] < K), 0.0)
        accumulator = tl.dot(x, w, accumulator)
    return accumulator


@triton.jit  # type: ignore[untyped-decorator]
def _merge_split_k(Parts, Y, SIZE: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):  # type: ignore[no-untyped-def]
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Fixed partition order, one FP32 sum, one output cast. No atomics.
    total = tl.full((BLOCK,), 0, tl.float32)
    for partition in tl.static_range(SPLIT_K):
        total += tl.load(Parts + partition * SIZE + offsets, offsets < SIZE, 0)
    tl.store(Y + offsets, total, offsets < SIZE)


@triton.jit  # type: ignore[untyped-decorator]
def _accumulate_weight_tile(X, w, acc, m, k, M: tl.constexpr, K: tl.constexpr):  # type: ignore[no-untyped-def]
    x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0)
    return tl.dot(x, w, acc)


@triton.jit  # type: ignore[untyped-decorator]
def _weight_only_mm_reuse_rows(  # type: ignore[no-untyped-def]
    X,
    W,
    S,
    Z,
    A,
    B,
    GS,
    GA,
    GB,
    PART,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    LOADER: tl.constexpr,
    FORMAT: tl.constexpr,
    BITS: tl.constexpr,
    GI: tl.constexpr,
    GO: tl.constexpr,
    STRIDE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
    WEIGHT_K_CONTIGUOUS: tl.constexpr = False,
    ROW_TILES: tl.constexpr = 2,
):
    tl.static_assert(ROW_TILES == 2)
    m = tl.program_id(0) * BM * ROW_TILES + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    partition = tl.program_id(2)
    acc0 = tl.zeros((BM, BN), tl.float32)
    if ROW_TILES >= 2:
        acc1 = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK * SPLIT_K)):
        k = (start * SPLIT_K + partition) * BK + kk
        if WEIGHT_K_CONTIGUOUS:
            wr, wk = n[:, None], k[None, :]
        else:
            wr, wk = n[None, :], k[:, None]
        w = LOADER(W, S, Z, A, B, GS, GA, GB, PART, wr, wk, N, K, FORMAT, BITS, GI, GO, STRIDE).to(
            X.dtype.element_ty
        )
        if WEIGHT_K_CONTIGUOUS:
            w = w.T
        acc0 = _accumulate_weight_tile(X, w, acc0, m, k, M, K)
        if ROW_TILES >= 2:
            acc1 = _accumulate_weight_tile(X, w, acc1, m + BM, k, M, K)
    tl.store(
        Y + partition * M * N + m[:, None] * N + n[None, :],
        acc0,
        (m[:, None] < M) & (n[None, :] < N),
    )
    if ROW_TILES >= 2:
        tl.store(
            Y + partition * M * N + (m[:, None] + BM) * N + n[None, :],
            acc1,
            (m[:, None] + BM < M) & (n[None, :] < N),
        )
