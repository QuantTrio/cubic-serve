"""Cubic W2-W8 -> per-channel INT8 weights and per-token INT8 activations.

Linear weights occupy N*K bytes plus N FP32 scales; original checkpoint payloads
remain the independent oracle. Each local K reduction is exact INT32, followed
by one FP32 scale product and one boundary cast. No architecture-specific branch.
"""

from __future__ import annotations

from math import ceil

import torch
import triton
import triton.language as tl

from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.quantization.common.int8 import encode


@triton.jit
def _quantize(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    v = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    q, scale = encode(v)
    tl.store(Q + row * K + k, q, k < K)
    tl.store(S + row, scale)


@triton.jit
def _prepare_carriers(
    W,
    S,
    A,
    B,
    MAXIMA,
    Q,
    WS,
    N: tl.constexpr,
    K: tl.constexpr,
    BITS: tl.constexpr,
    GI: tl.constexpr,
    GO: tl.constexpr,
    BLOCK: tl.constexpr,
    EMIT: tl.constexpr,
):
    n, part = tl.program_id(0), tl.program_id(1)
    k = part * BLOCK + tl.arange(0, BLOCK)
    stride: tl.constexpr = tl.cdiv(K * BITS, 8)
    byte = k * BITS // 8
    raw = tl.load(W + n * stride + byte, k < K, 0).to(tl.int32)
    if 8 % BITS != 0:
        high = tl.load(W + n * stride + byte + 1, (k < K) & (byte + 1 < stride), 0)
        raw = raw | (high.to(tl.int32) << 8)
    raw = (raw >> (k * BITS % 8)) & ((1 << BITS) - 1)
    signed = tl.where(raw >= (1 << (BITS - 1)), raw - (1 << BITS), raw)
    signed = tl.where(signed == -(1 << (BITS - 1)), 0, signed)
    group = (n // GO) * tl.cdiv(K, GI) + k // GI
    aa = tl.load(A + group, k < K, 0).to(tl.float32)
    bb = tl.load(B + group, k < K, 0).to(tl.float32)
    scale = tl.load(S + group, k < K, 0)
    t = tl.div_rn(tl.abs(signed).to(tl.float32), ((1 << (BITS - 1)) - 1) * 1.0)
    f = t * (aa + t * (bb + t * ((1.0 - aa) - bb)))
    value = tl.where(signed < 0, -(scale * f), scale * f)
    parts: tl.constexpr = tl.cdiv(K, BLOCK)
    if EMIT:
        indices = tl.arange(0, triton.next_power_of_2(parts))
        maxima = tl.load(MAXIMA + n * parts + indices, indices < parts, 0)
        ws = tl.div_rn(tl.max(maxima, 0), 127.0)
        ws = tl.where(ws > 0, ws, 1.0)
        carrier = tl.extra.cuda.libdevice.nearbyint(tl.div_rn(value, ws))
        tl.store(Q + n * K + k, tl.minimum(127.0, tl.maximum(-127.0, carrier)).to(tl.int8), k < K)
        if part == 0:
            tl.store(WS + n, ws)
    else:
        tl.store(MAXIMA + n * parts + part, tl.max(tl.abs(value), 0))


@triton.jit
def _mm(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    column_tile=None,
    k_partition=None,
    PARTIAL_OFFSET: tl.constexpr = 0,
):
    if column_tile is None:
        column_tile = tl.program_id(1)
    if k_partition is None:
        k_partition = tl.program_id(2)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = column_tile * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    tiles: tl.constexpr = tl.cdiv(tl.cdiv(K, BK), SPLIT)
    base = k_partition * tiles * BK
    acc = tl.full((BM, BN), 0, tl.int32)
    for step in range(tiles):
        k = base + step * BK + kk
        a = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0)
        b = tl.load(W + n[None, :] * K + k[:, None], (k[:, None] < K) & (n[None, :] < N), 0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    tl.store(
        Y + PARTIAL_OFFSET + (k_partition * M + m[:, None]) * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _finish(
    P,
    XS,
    WS,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_OFFSET: tl.constexpr = 0,
    PARTIAL_OFFSET: tl.constexpr = 0,
):
    i = (tl.program_id(0) - TILE_OFFSET) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.int32)
    for part in range(SPLIT):
        total += tl.load(P + PARTIAL_OFFSET + part * M * N + i, i < M * N, 0)
    sx = tl.load(XS + i // N, i < M * N, 0)
    sw = tl.load(WS + i % N, i < M * N, 0)
    tl.store(Y + i, total.to(tl.float32) * (sx * sw), i < M * N)


def _prepare(operators, weight, device):
    versions = []
    for t in weight.payload.values():
        try:
            versions.append(t._version)
        except RuntimeError:
            versions.append(None)
    key = ("cubic.wna8.channel.v1", id(weight), device, tuple(versions))
    keys = tuple((*key, name) for name in ("codes", "scales"))
    if all(k in operators.prepared_weights for k in keys):
        tensors = tuple(operators.prepared_weights[k][1] for k in keys)
        return tensors
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("WNA8 preparation must precede graph capture")
    # Metadata decoding is shared with the format; carrier generation and the
    # independent reference contraction are separate implementations.
    from .reference import cubic_metadata

    p = {name: t.cpu() for name, t in weight.payload.items() if name != "weight_packed"}
    scale, a, b = cubic_metadata(
        p, weight.encoding, weight.shape, weight.partitions or (weight.shape[0],)
    )
    n, k = weight.shape
    packed = weight.payload["weight_packed"].to(device).contiguous()
    aa, bb = a.to(device).contiguous(), b.to(device).contiguous()
    codes = torch.empty((n, k), device=device, dtype=torch.int8)
    original_scales = scale.to(device).contiguous()
    blocks = triton.cdiv(k, 1024)
    maxima = torch.empty((n, blocks), device=device, dtype=torch.float32)
    scales = torch.empty((n,), device=device, dtype=torch.float32)
    args = (
        packed,
        original_scales,
        aa,
        bb,
        maxima,
        codes,
        scales,
        n,
        k,
        weight.encoding.bits,
        weight.encoding.group_in,
        weight.encoding.group_out,
        1024,
    )
    _prepare_carriers[(n, blocks)](*args, False, num_warps=4, enable_fp_fusion=False)
    _prepare_carriers[(n, blocks)](*args, True, num_warps=4, enable_fp_fusion=False)
    prepared = codes, scales
    for cache_key, tensor in zip(keys, prepared, strict=True):
        operators.prepared_weights[cache_key] = (weight, tensor)
    return prepared


def bind_linear(operators, x, weight, output, params):
    e = weight.encoding
    m, (n, k) = x.shape[0], weight.shape
    if not 2 <= e.bits <= 8 or k * 127 * 127 > torch.iinfo(torch.int32).max:
        raise ValueError("Cubic WNA8 requires W2-W8 and an overflow-free INT32 dot")
    encoded = operators.int8_input(x) if x.dtype == torch.int8 else None
    if x.dtype not in (torch.float16, torch.bfloat16) and encoded is None:
        raise ValueError("Cubic WNA8 requires FP16/BF16 boundary tensors")
    codes, scales = _prepare(operators, weight, x.device)
    q = torch.empty_like(x, dtype=torch.int8) if encoded is None else encoded.codes
    xs = (
        torch.empty((m,), dtype=torch.float32, device=x.device)
        if encoded is None
        else encoded.scales
    )
    bm = min(128, max(16, triton.next_power_of_2(m)))
    bn, bk = 64, 128
    resources = torch.cuda.get_device_properties(x.device)
    split = min(
        params.wna8_max_k_partitions,
        triton.cdiv(k, bk),
        ceil(params.wna8_cta_supply_factor * resources.multi_processor_count / triton.cdiv(n, bn)),
    )
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.int32)
    calls = (
        ()
        if encoded is not None
        else (
            _PackedLaunch.compile(
                _quantize,
                (x, q, xs),
                (k, triton.next_power_of_2(k)),
                (m,),
                dict(num_warps=4, enable_fp_fusion=False),
            ),
        )
    )
    calls += (
        _PackedLaunch.compile(
            _mm,
            (q, codes, partial),
            (m, n, k, bm, bn, bk, split),
            (triton.cdiv(m, bm), triton.cdiv(n, bn), split),
            dict(num_warps=4, num_stages=2, enable_fp_fusion=False),
        ),
        _PackedLaunch.compile(
            _finish,
            (partial, xs, scales, output),
            (m, n, split, 256),
            (triton.cdiv(m * n, 256),),
            dict(num_warps=4, enable_fp_fusion=False),
        ),
    )
    # The integer partial is fully consumed before this projection completes.
    # Reuse the existing dispatch-private pool across sequential occurrences;
    # independent graph bodies/overlap lanes retain separate pools.
    scratch = (partial, q, xs) if encoded is None else (partial,)
    return PackedOperatorBinding(
        output,
        calls,
        scratch=scratch,
        reusable_scratch=True,
        row_tile_count=triton.cdiv(m, bm),
    )


@triton.jit
def _paired_mm(
    X,
    W0,
    W1,
    P,
    M: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
):
    tile = tl.program_id(1)
    columns0: tl.constexpr = tl.cdiv(N0, BN)
    columns1: tl.constexpr = tl.cdiv(N1, BN)
    if tile < columns0 * S0:
        _mm(X, W0, P, M, N0, K, BM, BN, BK, S0, tile % columns0, tile // columns0, 0)
    else:
        local = tile - columns0 * S0
        _mm(X, W1, P, M, N1, K, BM, BN, BK, S1, local % columns1, local // columns1, S0 * M * N0)


@triton.jit
def _paired_finish(
    P,
    XS,
    WS0,
    WS1,
    Y0,
    Y1,
    M: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tiles0: tl.constexpr = tl.cdiv(M * N0, BLOCK)
    if tl.program_id(0) < tiles0:
        _finish(P, XS, WS0, Y0, M, N0, S0, BLOCK)
    else:
        _finish(P, XS, WS1, Y1, M, N1, S1, BLOCK, tiles0, S0 * M * N0)


def bind_pair(operators, x, weights, outputs, params):
    """Pool two declared independent projections without concatenating weights.

    Each member keeps its original K partitions and numerical boundary. With
    Gi=ceil(Ni/BN), Si the existing supply rule, the grid contains exactly
    ceil(M/BM)*(G0*S0+G1*S1) CTAs. Scratch is M*(S0*N0+S1*N1) integers.
    This pools existing work into two launches, without padding either K grid.
    Larger groups retain their existing path; this is a binary primitive.
    """
    if len(weights) != 2:
        return None
    m, k = x.shape
    if any(w.shape[1] != k or not 2 <= w.encoding.bits <= 8 for w in weights):
        raise ValueError("paired Cubic WNA8 requires compatible W2-W8 full-K weights")
    if k * 127 * 127 > torch.iinfo(torch.int32).max:
        raise ValueError("paired WNA8 requires overflow-free INT32 contraction")
    encoded = operators.int8_input(x) if x.dtype == torch.int8 else None
    if x.dtype not in (torch.float16, torch.bfloat16) and encoded is None:
        raise ValueError("paired WNA8 requires FP16/BF16 boundaries or registered INT8")
    prepared = tuple(_prepare(operators, w, x.device) for w in weights)
    q = torch.empty_like(x, dtype=torch.int8) if encoded is None else encoded.codes
    xs = (
        torch.empty((m,), dtype=torch.float32, device=x.device)
        if encoded is None
        else encoded.scales
    )
    ns = tuple(w.shape[0] for w in weights)
    bm, bn, bk = min(128, max(16, triton.next_power_of_2(m))), 64, 128
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    splits = tuple(
        min(
            params.wna8_max_k_partitions,
            triton.cdiv(k, bk),
            ceil(params.wna8_cta_supply_factor * sms / triton.cdiv(n, bn)),
        )
        for n in ns
    )
    partial = torch.empty(
        sum(s * m * n for s, n in zip(splits, ns, strict=True)), dtype=torch.int32, device=x.device
    )
    calls = (
        ()
        if encoded is not None
        else (
            _PackedLaunch.compile(
                _quantize,
                (x, q, xs),
                (k, triton.next_power_of_2(k)),
                (m,),
                dict(num_warps=4, enable_fp_fusion=False),
            ),
        )
    )
    calls += (
        _PackedLaunch.compile(
            _paired_mm,
            (q, prepared[0][0], prepared[1][0], partial),
            (m, *ns, k, bm, bn, bk, *splits),
            (
                triton.cdiv(m, bm),
                sum(triton.cdiv(n, bn) * s for n, s in zip(ns, splits, strict=True)),
            ),
            dict(num_warps=4, num_stages=2, enable_fp_fusion=False),
        ),
        _PackedLaunch.compile(
            _paired_finish,
            (partial, xs, prepared[0][1], prepared[1][1], *outputs),
            (m, *ns, *splits, 256),
            (sum(triton.cdiv(m * n, 256) for n in ns),),
            dict(num_warps=4, enable_fp_fusion=False),
        ),
    )
    scratch = (partial, q, xs) if encoded is None else (partial,)
    return PackedOperatorBinding(
        outputs[0], calls, scratch=scratch, reusable_scratch=True, row_tile_count=triton.cdiv(m, bm)
    )
