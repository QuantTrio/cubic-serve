"""Group16-ordered FP4 Marlin binding, projection globals unchanged."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor

from cubic_serve.models.quantization import ActivationFormat
from cubic_serve.ops.cuda.compiled_artifact import bind_tensor
from cubic_serve.ops.cuda.native import CudaScalar, NativeLaunch
from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.quantization.common.marlin.compiler import bind_launch, compile_module
from cubic_serve.quantization.common.marlin.cost import ProbeFormat
from cubic_serve.quantization.common.marlin.cost import select as select_native_geometry
from cubic_serve.quantization.common.marlin.parameters import (
    MarlinParameters,
    shared_bytes,
)
from cubic_serve.quantization.common.marlin.repack import repack_words
from cubic_serve.quantization.common.reference import _positive_finite

from .kernels import _fp4, _nvfp4_codes_and_scale
from .layout import payload_names

if TYPE_CHECKING:
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.quantization.api import RoutedRows
    from cubic_serve.quantization.common.parameters import PackedLinearParameters

    from .policy import NVFP4MoEParameters


@triton.jit  # type: ignore[untyped-decorator]
def prepare(  # type: ignore[no-untyped-def]
    X,
    G,
    COUNTS,
    STARTS,
    ROWS,
    Q,
    E: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    GROUPS: tl.constexpr,
    DIRECT: tl.constexpr = False,
    PHYSICAL: tl.constexpr = 0,
):
    index = tl.program_id(0)
    projection = tl.program_id(1)
    if index >= tl.load(STARTS + E - 1) + tl.load(COUNTS + E - 1):
        return
    ids = tl.arange(0, triton.next_power_of_2(E))
    end = tl.load(STARTS + ids, ids < E, 0) + tl.load(COUNTS + ids, ids < E, 0)
    expert = tl.sum(((ids < E) & (index >= end)).to(tl.int32), 0)
    route = tl.load(ROWS + index)
    k = tl.arange(0, GROUPS)[:, None] * 16 + tl.arange(0, 16)[None, :]
    x = tl.load(X + (route // TOPK) * K + k, k < K, 0).to(tl.float32)
    inverse = tl.load(G + projection * E + expert)
    if DIRECT:
        inverse = tl.div_rn(1.0, inverse)
    code, scale = _nvfp4_codes_and_scale(x, inverse)
    stride: tl.constexpr = PHYSICAL if PHYSICAL else K
    tl.store(Q + (projection * R + route) * stride + k,
             _fp4(code) * scale[:, None], k < stride)


@triton.jit  # type: ignore[untyped-decorator]
def finish(  # type: ignore[no-untyped-def]
    P,
    IG,
    WG,
    COUNTS,
    STARTS,
    ROWS,
    Y,
    E: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    PROJECTIONS: tl.constexpr,
    BLOCK: tl.constexpr,
    DIRECT: tl.constexpr = False,
    PHYSICAL: tl.constexpr = 0,
):
    index = tl.program_id(0)
    projection = tl.program_id(1)
    if index >= tl.load(STARTS + E - 1) + tl.load(COUNTS + E - 1):
        return
    ids = tl.arange(0, triton.next_power_of_2(E))
    end = tl.load(STARTS + ids, ids < E, 0) + tl.load(COUNTS + ids, ids < E, 0)
    expert = tl.sum(((ids < E) & (index >= end)).to(tl.int32), 0)
    route = tl.load(ROWS + index)
    col = tl.arange(0, BLOCK)
    stride: tl.constexpr = PHYSICAL if PHYSICAL else N
    val = tl.load(P + (projection * R + route) * stride + col, col < N, 0)
    if DIRECT:
        val = val * tl.load(IG + projection * E + expert) * tl.load(WG + projection * E + expert)
    else:
        val = val / tl.load(IG + projection * E + expert) / tl.load(WG + projection * E + expert)
    tl.store(Y + route * (PROJECTIONS * N) + projection * N + col, val, col < N)


@triton.jit  # type: ignore[untyped-decorator]
def pad_projection_routes(  # type: ignore[no-untyped-def]
    COUNTS,
    STARTS,
    BLOCK_STARTS,
    TOTAL,
    ROWS,
    PADDED,
    EXPERTS,
    TOKENS,
    E: tl.constexpr,
    ROUTES: tl.constexpr,
    BM: tl.constexpr,
    PROJECTIONS: tl.constexpr,
):
    # Flatten only the private (projection, rank-local expert) batch axes.
    # Logical expert IDs and checkpoint/rank ownership never change. Each
    # projection retains its own prepared activation and original scale domain.
    expert, projection = tl.program_id(0), tl.program_id(1)
    count = tl.load(COUNTS + expert)
    start = tl.load(STARTS + expert)
    block = tl.load(BLOCK_STARTS + expert)
    blocks = tl.load(TOTAL)
    lane = tl.arange(0, BM)
    for b in range(tl.cdiv(count, BM)):
        row = b * BM + lane
        route = tl.load(ROWS + start + row, row < count, 0)
        route = tl.where(row < count, route + projection * ROUTES, PROJECTIONS * ROUTES)
        ordinal = projection * blocks + block + b
        tl.store(PADDED + ordinal * BM + lane, route)
        tl.store(EXPERTS + ordinal, projection * E + expert)
    if expert == 0 and projection == 0:
        tl.store(TOKENS, blocks * BM * PROJECTIONS)


def encode_scales(scales: Tensor) -> Tensor:
    """Lossless positive E4M3 -> Marlin S0E5M3 carrier; no clipping.

    Every nonnegative finite E4M3 value times 128 is exactly FP16. FP4's
    raw half carrier has a 2**-14 bias; the product is therefore scaled by
    2**-7. The native FP32 epilogue restores 128 before original IG/WG math.
    """
    if scales.dtype != torch.float8_e4m3fn:
        raise ValueError("NVFP4 Marlin scale encoding requires original E4M3")
    half = scales.to(torch.float16)
    _positive_finite(half, "NVFP4 block scales", zero=True)
    return ((half * 128).view(torch.int16) >> 7).to(torch.uint8)


def packed_weights(
    operators: PackedLinearOperators,
    groups: tuple[tuple[QuantizedMatrix, ...], ...],
    device: torch.device,
    physical_shape: tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    gate = groups[0]
    weights_all = tuple(w for group in groups for w in group)
    identity = tuple(
        bind_tensor(w.payload[name]) for w in weights_all for name in payload_names(w.encoding)
    )
    keys = tuple(
        ("nvfp4-marlin-lossless-rebias-v4", gate[0].encoding,
         device, identity, physical_shape, name)
        for name in ("codes", "scales", "input_global", "weight_global")
    )
    if all(key in operators.prepared_weights for key in keys):
        return cast(
            tuple[Tensor, Tensor, Tensor, Tensor],
            tuple(operators.prepared_weights[key][1] for key in keys),
        )
    e = len(gate)
    logical_n, logical_k = gate[0].shape
    n, k = physical_shape
    codes = torch.empty((len(groups), e, k // 16, n * 2), dtype=torch.int32, device=device)
    scales = torch.empty((len(groups), e, k // 16, n), dtype=torch.uint8, device=device)
    permutation = [a + 8 * b for a in range(8) for b in range(8)]
    for projection, weights in enumerate(groups):
        for expert, weight in enumerate(weights):
            # One logical matrix bounds cold staging, independent of the
            # repository's expert count. Repack directly into the final
            # projection/expert slot; never stack a second full device copy.
            words = weight.payload[payload_names(weight.encoding)[0]].to(device)
            # Fill only private tile tails; original checkpoint bytes and
            # group16 scale domains stay unchanged.
            if (n, k) != (logical_n, logical_k):
                padded = torch.zeros((n, k // 2), dtype=words.dtype, device=device)
                padded[:logical_n, :logical_k // 2].copy_(words)
                words = padded
            source = words.contiguous().view(torch.int32).T.contiguous()
            repack_words(source, bits=4, axis="k", output=codes[projection, expert])
            original_scale = weight.payload["weight_scale"].to(device=device)
            if (n, k) != (logical_n, logical_k):
                padded_scale = torch.zeros((n, k // 16), dtype=torch.uint8, device=device)
                padded_scale[:logical_n, :logical_k // 16].copy_(original_scale.view(torch.uint8))
                original_scale = padded_scale.view(torch.float8_e4m3fn)
            scale = (
                encode_scales(original_scale)
                .T.contiguous()
                .reshape(-1, 64)[:, permutation]
                .reshape(-1, 4)[:, [0, 2, 1, 3]]
                .reshape(k // 16, n)
            )
            scales[projection, expert].copy_(scale)
    ig = torch.stack([w.payload[payload_names(w.encoding)[2]].reshape(()) for w in weights_all])
    wg = torch.stack([w.payload[payload_names(w.encoding)[3]].reshape(()) for w in weights_all])
    result = codes, scales, ig.to(device), wg.to(device)
    for key, value in zip(keys, result, strict=True):
        operators.prepared_weights[key] = (gate[0], value)
    return result


def bind_projection_group(
    operators: PackedLinearOperators,
    x: Tensor,
    groups: tuple[tuple[QuantizedMatrix, ...], ...],
    output: Tensor,
    rows: RoutedRows,
    r: int,
    top_k: int,
    params: NVFP4MoEParameters,
) -> tuple[int, PackedOperatorBinding] | None:
    gate = groups[0]
    weights_all = tuple(w for group in groups for w in group)
    projections = len(groups)
    n, k = gate[0].shape
    e = len(gate)
    if not isinstance(params, MarlinParameters) or not params.marlin_enabled:
        return None
    if projections > 1 and not params.grouped_projections:
        return None
    bm, bn, bk = params.marlin_block_m, params.marlin_block_n, params.marlin_block_k
    # All warps own N. Each K16 MMA forms an independent partial followed by
    # an ordered FP32 add; launch geometry cannot change this format boundary.
    if bn != params.marlin_warps * 64:
        raise ValueError("NVFP4 ordered K16 requires all warps along N")
    if gate[0].encoding.activation is not ActivationFormat.NVFP4_LOCAL or k % 16:
        return None
    if any(
        w.encoding != gate[0].encoding
        or w.shape != (n, k)
        or w.payload["weight_scale"].dtype != torch.float8_e4m3fn
        or w.payload[payload_names(w.encoding)[2]].numel() != 1
        or w.payload[payload_names(w.encoding)[3]].numel() != 1
        for w in weights_all
    ):
        return None
    # Prefetch tiles may be padded, but the ordered mathematical K16 groups
    # remain unchanged. Cost calibration charges the actual padded footprint.
    logical_n, logical_k = n, k
    n = math.ceil(n / bn) * bn
    k = math.ceil(k / math.lcm(bk, 16)) * math.lcm(bk, 16)
    params = select_native_geometry(
        params,
        operators=operators,
        family="nvfp4_moe",
        scope=dict(n=logical_n, k=logical_k, experts=e, top_k=rows.routing_top_k or top_k),
        source=_kernel_source,
        spec=ProbeFormat(
            4, torch.float16, torch.float32, torch.uint8, 0x22222222, 120, 1 / 128, 128.0
        ),
        device=x.device,
        n=n,
        k=k,
        experts=e * projections,
        group=16,
        rows=r / e if rows.mean_rows_per_expert is None else rows.mean_rows_per_expert,
        routed=True,
    )
    bm, bn, bk = params.marlin_block_m, params.marlin_block_n, params.marlin_block_k
    shared = shared_bytes(params, bits=4, group=16)
    resource = torch.cuda.get_device_properties(x.device)
    if shared > int(
        getattr(resource, "shared_memory_per_block_optin", resource.shared_memory_per_block)
    ):
        return None
    packed, scales, ig, wg = packed_weights(operators, groups, x.device, (n, k))
    source = _kernel_source(params)
    module, entry = compile_module(source, x.device)
    q = torch.empty(projections, r, k, dtype=torch.float16, device=x.device)
    partial = torch.empty(projections, r, n, device=x.device)
    blocks = math.ceil((r + min(r, e) * (bm - 1)) / bm)
    ids = torch.empty(projections * blocks * bm, dtype=torch.int32, device=x.device)
    owners = torch.empty(projections * blocks, dtype=torch.int32, device=x.device)
    total = torch.empty(1, dtype=torch.int32, device=x.device)
    rw = torch.ones(projections * r, 1, device=x.device)
    pad = _PackedLaunch.compile(
        pad_projection_routes,
        (
            rows.counts,
            rows.starts,
            rows.block_starts,
            rows.total_blocks,
            rows.row_map,
            ids,
            owners,
            total,
        ),
        (e, r, bm, projections),
        (e, projections),
        {},
    )
    prep = _PackedLaunch.compile(
        prepare,
        (x, ig, rows.counts, rows.starts, rows.row_map, q),
        (e, r, logical_k, top_k, triton.next_power_of_2(k // 16),
         gate[0].encoding.metadata == "modelopt", k),
        (r, projections),
        {"num_warps": 4, "enable_fp_fusion": False},
    )
    correction = torch.full((projections * e,), 128.0, dtype=torch.float32, device=x.device)
    calls: list[_PackedLaunch | NativeLaunch] = [pad, prep]
    args = (
        q,
        packed,
        partial,
        None,
        None,
        None,
        scales,
        correction,
        None,
        None,
        ids,
        owners,
        total,
        rw,
        CudaScalar("i32", 1),
        CudaScalar("bool", False),
        CudaScalar("i32", k // 16),
        CudaScalar("i32", projections * r),
        CudaScalar("i32", n),
        CudaScalar("i32", k),
        None,
        CudaScalar("bool", False),
        CudaScalar("bool", False),
        CudaScalar("bool", True),
        CudaScalar("i32", 0),
    )
    calls.append(
        bind_launch(
            module,
            entry,
            args,
            operator_id="cubic.cuda.nvfp4.marlin.v1",
            params=params,
            output_tiles=owners.numel() * (n // bn),
            shared_bytes=shared,
        )
    )
    calls.append(
        _PackedLaunch.compile(
            finish,
            (partial, ig, wg, rows.counts, rows.starts, rows.row_map, output),
            (
                e,
                r,
                logical_n,
                projections,
                triton.next_power_of_2(logical_n),
                gate[0].encoding.metadata == "modelopt",
                n,
            ),
            (r, projections),
            {"num_warps": 4, "enable_fp_fusion": False},
        )
    )
    return bm, PackedOperatorBinding(
        output, tuple(calls), scratch=(q, partial, ids, owners, total, rw, correction)
    )


def bind_grouped(
    operators: PackedLinearOperators,
    x: Tensor,
    gate: tuple[QuantizedMatrix, ...],
    up: tuple[QuantizedMatrix, ...],
    output: Tensor,
    rows: RoutedRows,
    top_k: int,
    params: NVFP4MoEParameters,
) -> tuple[int, PackedOperatorBinding] | None:
    return bind_projection_group(
        operators, x, (gate, up), output, rows, x.shape[0] * top_k, top_k, params
    )


def bind_routed(
    operators: PackedLinearOperators,
    x: Tensor,
    weights: tuple[QuantizedMatrix, ...],
    output: Tensor,
    rows: RoutedRows,
    params: NVFP4MoEParameters,
) -> tuple[int, PackedOperatorBinding] | None:
    return bind_projection_group(operators, x, (weights,), output, rows, x.shape[0], 1, params)


@triton.jit  # type: ignore[untyped-decorator]
def prepare_linear(X, G, Q, K: tl.constexpr, KP: tl.constexpr,  # type: ignore[no-untyped-def]
                   GROUPS: tl.constexpr, DIRECT: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, GROUPS)[:, None] * 16 + tl.arange(0, 16)[None, :]
    value = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    inverse = tl.load(G)
    if DIRECT:
        inverse = tl.div_rn(1.0, inverse)
    code, scale = _nvfp4_codes_and_scale(value, inverse)
    tl.store(Q + row * KP + k, _fp4(code) * scale[:, None], k < KP)


@triton.jit  # type: ignore[untyped-decorator]
def finish_linear(P, IG, WG, Y, M: tl.constexpr, N: tl.constexpr,  # type: ignore[no-untyped-def]
                  NP: tl.constexpr, BLOCK: tl.constexpr, DIRECT: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(P + (offset // N) * NP + offset % N, offset < M * N, 0)
    value = (value * tl.load(IG) * tl.load(WG) if DIRECT
             else value / tl.load(IG) / tl.load(WG))
    tl.store(Y + offset, value, offset < M * N)


def bind_linear(
    operators: PackedLinearOperators,
    x: Tensor,
    weight: QuantizedMatrix,
    output: Tensor,
    params: PackedLinearParameters,
) -> PackedOperatorBinding | None:
    """Dense identity rows share the format decoder, never expert routing.

    The prepared FP16 operands exactly carry FP4 times E4M3. Each original
    K16 contributes one ordered FP32 partial, as in routed projections.
    """
    n, k = weight.shape
    if (not isinstance(params, MarlinParameters) or not params.marlin_enabled
            or weight.encoding.activation is not ActivationFormat.NVFP4_LOCAL
            or k % 16 or not x.is_contiguous() or not output.is_contiguous()):
        return None
    if params.marlin_block_n != params.marlin_warps * 64:
        raise ValueError("NVFP4 ordered K16 requires all warps along N")
    names = payload_names(weight.encoding)
    if (weight.payload["weight_scale"].dtype != torch.float8_e4m3fn
            or any(weight.payload[name].numel() != 1 for name in names[2:])):
        return None
    np = math.ceil(n / params.marlin_block_n) * params.marlin_block_n
    quantum = math.lcm(params.marlin_block_k, 16)
    kp = math.ceil(k / quantum) * quantum
    m = x.shape[0]
    native_params: MarlinParameters = select_native_geometry(
        params, operators=operators, family="nvfp4_linear", scope=dict(n=n, k=k),
        source=lambda p: _kernel_source(p, dense_linear=True),
        spec=ProbeFormat(4, torch.float16, torch.float32, torch.uint8,
                         0x22222222, 120, 1 / 128, 128.0, dense_linear=True),
        device=x.device, n=np, k=kp, experts=1, group=16, rows=m, routed=False,
    )
    shared = shared_bytes(native_params, bits=4, group=16, row_metadata=False, zero_points=False)
    hw = torch.cuda.get_device_properties(x.device)
    if shared > getattr(hw, "shared_memory_per_block_optin", hw.shared_memory_per_block):
        return None
    packed, scales, ig, wg = packed_weights(operators, ((weight,),), x.device, (np, kp))
    module, entry = compile_module(_kernel_source(native_params, dense_linear=True), x.device)
    q = torch.empty((m, kp), dtype=torch.float16, device=x.device)
    partial = torch.empty((m, np), dtype=torch.float32, device=x.device)
    correction = torch.full((1,), 128.0, device=x.device)
    direct = weight.encoding.metadata == "modelopt"
    prep = _PackedLaunch.compile(
        prepare_linear, (x, ig, q),
        (k, kp, triton.next_power_of_2(kp // 16), direct), (m,),
        {"num_warps": 4, "enable_fp_fusion": False},
    )
    args = (q, packed, partial, None, None, None, scales, correction, None, None,
            None, None, None, None, CudaScalar("i32", 1), CudaScalar("bool", False),
            CudaScalar("i32", kp // 16), CudaScalar("i32", m), CudaScalar("i32", np),
            CudaScalar("i32", kp), None, CudaScalar("bool", False),
            CudaScalar("bool", False), CudaScalar("bool", True),
        CudaScalar("i32", 0),
    )
    blocks = math.ceil(m / native_params.marlin_block_m)
    native = bind_launch(module, entry, args,
                         operator_id="cubic.cuda.nvfp4.marlin.linear.v1", params=native_params,
                         output_tiles=blocks * (np // native_params.marlin_block_n), shared_bytes=shared)
    finish_call = _PackedLaunch.compile(
        finish_linear, (partial, ig, wg, output), (m, n, np, 256, direct),
        (math.ceil(m * n / 256),), {"num_warps": 4, "enable_fp_fusion": False},
    )
    return PackedOperatorBinding(output, (prep, native, finish_call),
                                 scratch=(q, partial, correction), row_tile_count=blocks)


def _kernel_source(params: MarlinParameters, *, dense_linear: bool = False) -> str:
    bm, bn, bk = params.marlin_block_m, params.marlin_block_n, params.marlin_block_k
    threads, stages = params.marlin_warps * 32, params.marlin_stages
    return f"""#define MARLIN_NAMESPACE_NAME cubic_nvfp4
#define CUBIC_MARLIN_FP32_OUTPUT 1
#define CUBIC_MARLIN_ORDERED_K16 1
#define CUBIC_MARLIN_DENSE_LINEAR {int(dense_linear)}
#include "libtorch_stable/moe/marlin_moe_wna16/kernel.h"
#include "marlin_template.h"
namespace cubic_nvfp4 {{template __global__ void Marlin<vllm::kFloat16.id(),vllm::kFE2M1f.id(),
vllm::kFloat16.id(),vllm::kFE4M3fn.id(),{threads},{bm // 16},{bn // 16},{bk // 16},
false,{stages},1,false>
(MARLIN_KERNEL_PARAMS); }}"""
