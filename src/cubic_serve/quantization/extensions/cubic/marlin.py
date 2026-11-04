"""Cubic codes with format-owned polynomial decoding in the shared Marlin pipe."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from cubic_serve.ops.cuda.compiled_artifact import bind_tensor
from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.quantization.api import RoutedRows
from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.quantization.common.marlin.compiler import bind_launch, compile_module
from cubic_serve.quantization.common.marlin.cost import ProbeFormat, fixed_k_partitions, select
from cubic_serve.quantization.common.marlin.parameters import (
    MarlinParameters,
    pipeline_compatible,
    shared_bytes,
)
from cubic_serve.quantization.common.marlin.repack import (
    _merge_dense_partials,
    _pad_routes,
    repack_signed_codes,
)

from .marlin_metadata import CUDA_METADATA, FIXTURE_PACKERS, metadata_variant, prepare_metadata

if TYPE_CHECKING:
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators

    from .policy import CubicMoEParameters


def _prepare(
    operators: PackedLinearOperators,
    groups: tuple[tuple[QuantizedMatrix, ...], ...],
    device: torch.device,
    *,
    fragment_bits: int = 8,
) -> tuple[Tensor, Tensor]:
    weights = tuple(w for group in groups for w in group)
    key = (
        "cubic.marlin.compact.v1",
        fragment_bits,
        device,
        tuple(
            (
                w.encoding,
                w.shape,
                w.partitions,
                tuple((name, bind_tensor(t)) for name, t in sorted(w.payload.items())),
            )
            for w in weights
        ),
    )
    cache = operators.prepared_weights
    if (key, "codes") in cache:
        return cache[key, "codes"][1], cache[key, "metadata"][1]
    base = groups[0][0]
    e, n, k = len(groups[0]), sum(group[0].shape[0] for group in groups), base.shape[1]
    bits = base.encoding.bits
    if fragment_bits not in (4, 8) or bits > fragment_bits:
        raise ValueError("Cubic codes must fit the register fragment lane width")
    codes = torch.empty(e, k // 16, n * bits // 2, device=device, dtype=torch.int32)
    for expert in range(e):
        words = []
        for group in groups:
            w = group[expert]
            raw = w.payload["weight_packed"].to(device)
            words.append(
                repack_signed_codes(raw.contiguous(), bits=bits, fragment_bits=fragment_bits)
            )
        codes[expert].copy_(torch.cat(words, dim=1))
    metadata = prepare_metadata(groups, device)
    cache[key, "codes"] = weights, codes
    cache[key, "metadata"] = weights, metadata
    return codes, metadata


def _bind(
    operators: PackedLinearOperators,
    x: Tensor,
    groups: tuple[tuple[QuantizedMatrix, ...], ...],
    output: Tensor,
    rows: RoutedRows,
    top_k: int,
    params: CubicMoEParameters,
) -> tuple[int, PackedOperatorBinding] | None:
    base = groups[0][0]
    n, k, e = sum(group[0].shape[0] for group in groups), base.shape[1], len(groups[0])
    enc = base.encoding
    if (
        not params.marlin_enabled
        or enc.metadata
        not in (
            "int8-scale-int4-ab-global-fp32",
            "e5m9-scale-curve2-tensor-fp16",
            "float32-scale-float16-ab",
        )
        or not 2 <= enc.bits <= 8
        or enc.group_out != 1
        or x.dtype not in (torch.bfloat16, torch.float16)
    ):
        return None
    gi = min(enc.group_in, k)
    if (
        gi < 32
        or gi % 32
        or k % gi
        or k % params.marlin_block_k
        or n % params.marlin_block_n
        or any(w.encoding != enc or w.shape[1] != k for group in groups for w in group)
    ):
        return None
    # Keep a single reduction plan across all M and backend tile candidates.
    # N and warp count may grow together; K slicing is an arithmetic boundary.
    if params.marlin_warps * 64 != params.marlin_block_n * 4:
        raise ValueError("Cubic Marlin requires four fixed K partitions across all M")
    reuse = max(1, gi // params.marlin_block_k)
    stages = math.lcm(params.marlin_stages, reuse)
    if stages > 4:
        return None
    params = replace(params, marlin_stages=stages)
    if not pipeline_compatible(params, group=gi):
        return None
    dtype = "kBFloat16" if x.dtype == torch.bfloat16 else "kFloat16"
    params = select(
        params,
        operators=operators,
        family="cubic_moe",
        scope=dict(n=groups[0][0].shape[0], k=k, experts=e, top_k=rows.routing_top_k or top_k),
        source=lambda p: _kernel_source(p, dtype, enc.bits, gi),
        spec=ProbeFormat(
            8,
            x.dtype,
            x.dtype,
            torch.float16,
            -2122219135,
            1.0,
            float(torch.tensor(1 / ((1 << (enc.bits - 1)) - 1), dtype=x.dtype)),
            group_metadata=(1.0, 0.0, 1.0),
            storage_bits=enc.bits,
            group_storage_bytes=2 if metadata_variant(enc.metadata) else 8,
            metadata_packer=FIXTURE_PACKERS[metadata_variant(enc.metadata)],
        ),
        device=x.device,
        n=n,
        k=k,
        experts=e,
        group=gi,
        rows=(
            x.shape[0] * top_k / e
            if rows.mean_rows_per_expert is None
            else rows.mean_rows_per_expert
        ),
        routed=True,
    )
    shared = shared_bytes(params, bits=enc.bits, group=gi, metadata_planes=3)
    hw = torch.cuda.get_device_properties(x.device)
    if shared > getattr(hw, "shared_memory_per_block_optin", hw.shared_memory_per_block):
        return None
    module, entry = compile_module(_kernel_source(params, dtype, enc.bits, gi), x.device)
    weight, metadata = _prepare(operators, groups, x.device)
    m, bm = x.shape[0], params.marlin_block_m
    routes = m * top_k
    blocks = math.ceil((routes + min(routes, e) * (bm - 1)) / bm)
    ids = torch.empty(blocks * bm, device=x.device, dtype=torch.int32)
    owners = torch.empty(blocks, device=x.device, dtype=torch.int32)
    total = torch.empty(1, device=x.device, dtype=torch.int32)
    rw = torch.ones(m, top_k, device=x.device)
    align = _PackedLaunch.compile(
        _pad_routes,
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
        (routes, bm),
        (e,),
        {},
    )
    args = (
        x,
        weight,
        output,
        None,
        None,
        None,
        None,
        metadata,
        None,
        None,
        ids,
        owners,
        total,
        rw,
        CudaScalar("i32", top_k),
        CudaScalar("bool", False),
        CudaScalar("i32", k // gi),
        CudaScalar("i32", m),
        CudaScalar("i32", n),
        CudaScalar("i32", k),
        None,
        CudaScalar("bool", False),
        CudaScalar("bool", False),
        CudaScalar("bool", True),
        CudaScalar("i32", 0),
    )
    multiply = bind_launch(
        module,
        entry,
        args,
        operator_id="cubic.cuda.cubic.marlin.routed.v1",
        params=params,
        output_tiles=blocks * (n // params.marlin_block_n),
        shared_bytes=shared,
    )
    return bm, PackedOperatorBinding(output, (align, multiply), scratch=(ids, owners, total, rw))


def bind_grouped(
    operators: PackedLinearOperators,
    x: Tensor,
    gate: tuple[QuantizedMatrix, ...],
    up: tuple[QuantizedMatrix, ...],
    output: Tensor,
    rows: RoutedRows,
    top_k: int,
    params: CubicMoEParameters,
) -> tuple[int, PackedOperatorBinding] | None:
    if not params.grouped_projections:
        return None
    return _bind(operators, x, (gate, up), output, rows, top_k, params)


def bind_routed(
    operators: PackedLinearOperators,
    x: Tensor,
    weights: tuple[QuantizedMatrix, ...],
    output: Tensor,
    rows: RoutedRows,
    params: CubicMoEParameters,
) -> tuple[int, PackedOperatorBinding] | None:
    return _bind(operators, x, (weights,), output, rows, 1, params)


def _plan_linear(operators, x, matrix, params):
    """Seal an M-invariant K plan, then predict output tile cost from resources."""
    import struct

    enc = matrix.encoding
    n, k = matrix.shape
    gi = min(enc.group_in, k)
    if (
        enc.group_out != 1
        or not 2 <= enc.bits <= 8
        or gi < 32
        or gi % 32
        or k % gi
        or k % params.marlin_block_k
        or n % params.marlin_block_n
    ):
        return None
    stages = math.lcm(params.marlin_stages, max(1, gi // params.marlin_block_k))
    params = replace(params, marlin_stages=stages)
    if stages > 4 or not pipeline_compatible(params, group=gi):
        return None
    hw = torch.cuda.get_device_properties(x.device)
    # Price fractional SM coverage and final-wave work together. The supply
    # target bounds candidate partitions rather than excluding a partial wave.
    # This K plan remains invariant across M and independent of row-tile choice.
    # The cap bounds partial-buffer/merge traffic at large M, and legal divisors
    # retain checkpoint group boundaries for every format bit width.
    partitions = fixed_k_partitions(
        n=n,
        k=k,
        group=gi,
        block_n=params.marlin_block_n,
        block_k=params.marlin_block_k,
        sms=hw.multi_processor_count,
        supply_factor=params.marlin_cta_supply_factor,
        max_partitions=params.marlin_max_k_partitions,
    )
    fragment_bits = 4 if enc.bits <= 4 else 8
    value = (1 << (fragment_bits - 1)) + (1 << (enc.bits - 1)) - 1
    word = sum(value << (i * fragment_bits) for i in range(32 // fragment_bits))
    word = word if word < 2**31 else word - 2**32
    dtype = "kBFloat16" if x.dtype == torch.bfloat16 else "kFloat16"
    spec = ProbeFormat(
        fragment_bits,
        x.dtype,
        x.dtype,
        torch.float16,
        word,
        1.0,
        1.0,
        group_metadata=(struct.unpack("f", struct.pack("I", 0x3C00))[0], 1.0),
        fixed_partitions=partitions,
        dense_linear=True,
        storage_bits=enc.bits,
        group_storage_bytes=2 if metadata_variant(enc.metadata) else 8,
        metadata_packer=FIXTURE_PACKERS[metadata_variant(enc.metadata)],
    )
    params = select(
        params,
        operators=operators,
        family="cubic_linear",
        scope=dict(n=n, k=k),
        source=lambda p: _kernel_source(
            p,
            dtype,
            enc.bits,
            gi,
            fragment_bits=fragment_bits,
            dense_linear=True,
            fixed_partitions=partitions,
        ),
        spec=spec,
        device=x.device,
        n=n,
        k=k,
        experts=1,
        group=gi,
        rows=x.shape[0],
        routed=False,
    )
    return params, partitions


def bind_planned_linear(operators, x, matrix, output, params):
    plan = _plan_linear(operators, x, matrix, params)
    if plan is None:
        return None
    params, partitions = plan
    return bind_linear(operators, x, matrix, output, params, fixed_partitions=partitions)


def bind_planned_grouped(operators, x, weights, outputs, params):
    """Share dense launches without changing any member's K reduction plan."""
    if len(weights) < 2 or len(weights) != len(outputs):
        return None
    base = weights[0]
    if any(w.encoding != base.encoding or w.shape != base.shape for w in weights):
        return None
    # Price the original projection, not concatenated N: concatenation must not
    # change K slicing, MMA geometry or the format's rounding boundaries.
    plan = _plan_linear(operators, x, base, params)
    if plan is None:
        return None
    params, partitions = plan
    return _bind_dense(operators, x, weights, outputs, params, fixed_partitions=partitions)


def bind_linear(
    operators: PackedLinearOperators,
    x: Tensor,
    matrix: QuantizedMatrix,
    output: Tensor,
    params: MarlinParameters,
    *,
    fixed_partitions: int = 0,
) -> PackedOperatorBinding | None:
    return _bind_dense(
        operators, x, (matrix,), (output,), params, fixed_partitions=fixed_partitions
    )


def _bind_dense(operators, x, weights, outputs, params, *, fixed_partitions):
    """Bind dense rows without launching the expert route alignment kernel.

    Share the format's original-code decoder and prepared metadata with MoE.
    Dense row ownership is a compile-time contract, not a synthetic expert map.
    This candidate does not itself select or qualify a serving backend.
    """
    matrix, output = weights[0], outputs[0]
    enc = matrix.encoding
    m, n, k = x.shape[0], sum(w.shape[0] for w in weights), matrix.shape[1]
    if (
        not params.marlin_enabled
        or enc.metadata
        not in (
            "int8-scale-int4-ab-global-fp32",
            "e5m9-scale-curve2-tensor-fp16",
            "float32-scale-float16-ab",
        )
        or not 2 <= enc.bits <= 8
        or enc.group_out != 1
        or x.dtype not in (torch.bfloat16, torch.float16)
        or any(w.encoding != enc or w.shape[1] != k for w in weights)
    ):
        return None
    gi = min(enc.group_in, k)
    if gi < 32 or gi % 32 or k % gi or k % params.marlin_block_k or n % params.marlin_block_n:
        return None
    if params.marlin_warps * 64 != params.marlin_block_n * 4:
        raise ValueError("Cubic Marlin requires four fixed K partitions across all M")
    stages = math.lcm(params.marlin_stages, max(1, gi // params.marlin_block_k))
    if stages > 4:
        return None
    params = replace(params, marlin_stages=stages)
    if not pipeline_compatible(params, group=gi):
        return None
    if type(fixed_partitions) is not int or fixed_partitions < 0:
        raise ValueError("fixed Marlin partitions must be a nonnegative integer")
    if len(weights) > 1 and not fixed_partitions:
        raise ValueError("grouped dense projections require a fixed partial reduction")
    if fixed_partitions and k % (fixed_partitions * math.lcm(gi, params.marlin_block_k)):
        raise ValueError("fixed Marlin partitions must close K tiles and checkpoint groups")
    fragment_bits = 4 if enc.bits <= 4 else 8
    planes = 2 if fixed_partitions else 3
    shared = shared_bytes(
        params,
        bits=enc.bits,
        group=gi,
        metadata_planes=planes,
        row_metadata=False,
        zero_points=False,
    )
    hw = torch.cuda.get_device_properties(x.device)
    if shared > getattr(hw, "shared_memory_per_block_optin", hw.shared_memory_per_block):
        return None
    dtype = "kBFloat16" if x.dtype == torch.bfloat16 else "kFloat16"
    module, entry = compile_module(
        _kernel_source(
            params,
            dtype,
            enc.bits,
            gi,
            fragment_bits=fragment_bits,
            dense_linear=True,
            fixed_partitions=fixed_partitions,
        ),
        x.device,
    )
    weight, metadata = _prepare(
        operators,
        tuple((w,) for w in weights),
        x.device,
        fragment_bits=fragment_bits,
    )
    blocks = math.ceil(m / params.marlin_block_m)
    tiles = blocks * (n // params.marlin_block_n)
    partials = (
        torch.empty(
            tiles * fixed_partitions * params.marlin_block_m * params.marlin_block_n,
            device=x.device,
            dtype=torch.float32,
        )
        if fixed_partitions
        else None
    )
    args = (
        x,
        weight,
        output,
        partials,
        None,
        None,
        None,
        metadata,
        None,
        None,
        None,
        None,
        None,
        None,
        CudaScalar("i32", 1),
        CudaScalar("bool", False),
        CudaScalar("i32", k // gi),
        CudaScalar("i32", m),
        CudaScalar("i32", n),
        CudaScalar("i32", k),
        None,
        CudaScalar("bool", False),
        CudaScalar("bool", False),
        CudaScalar("bool", True),
        CudaScalar("i32", fixed_partitions),
    )
    if fixed_partitions:
        launch = module.bind(
            entry,
            args,
            operator_id="cubic.cuda.cubic.marlin.linear.partial.v1",
            grid=(tiles * fixed_partitions, 1, 1),
            block=(params.marlin_warps * 32, 1, 1),
            shared_bytes=shared,
        )
        widths = tuple(w.shape[0] for w in weights)
        finish = _PackedLaunch.compile(
            _merge_dense_partials,
            (partials, outputs if len(outputs) > 1 else output),
            (
                m, n, params.marlin_block_m, params.marlin_block_n, fixed_partitions, 256,
                *((widths, widths) if len(outputs) > 1 else ()),
            ),
            (math.ceil(m * n / 256),),
            {"num_warps": 4, "enable_fp_fusion": False},
        )
        return PackedOperatorBinding(
            output, (launch, finish), scratch=(partials,), reusable_scratch=True,
            row_tile_count=blocks,
        )
    launch = bind_launch(
        module,
        entry,
        args,
        operator_id="cubic.cuda.cubic.marlin.linear.v1",
        params=params,
        output_tiles=blocks * (n // params.marlin_block_n),
        shared_bytes=shared,
    )
    return PackedOperatorBinding(output, (launch,), row_tile_count=blocks)


def _kernel_source(
    p: MarlinParameters,
    dtype: str,
    bits: int,
    group: int,
    *,
    fragment_bits: int = 8,
    dense_linear: bool = False,
    fixed_partitions: int = 0,
) -> str:
    fragment_type = "kU4B8" if fragment_bits == 4 else "kU8B128"
    denominator = (1 << (bits - 1)) - 1
    shift = 23 + bits - 1
    hi = round(2**shift / denominator) * 2**-shift
    lo = float(torch.tensor(1 / denominator - hi, dtype=torch.float32))
    planes = 2 if fixed_partitions else 3
    metadata_read = (
        "const unsigned int ab = __float_as_uint(metadata[0][column]); "
        "const float a = __half2float(__ushort_as_half(ab & 65535)); "
        "const float b = __half2float(__ushort_as_half(ab >> 16)); "
        "const float s = metadata[1][column];"
        if fixed_partitions
        else "const float a = metadata[0][column], b = metadata[1][column], "
        "s = metadata[2][column];"
    )
    return f"""#define CUBIC_MARLIN_STORAGE_BITS {bits}
#define MARLIN_NAMESPACE_NAME cubic_polynomial
#define CUBIC_MARLIN_DENSE_LINEAR {int(dense_linear)}
#define CUBIC_MARLIN_FIXED_PARTITIONS {int(bool(fixed_partitions))}
#include "libtorch_stable/moe/marlin_moe_wna16/kernel.h"
namespace cubic_polynomial {{
{CUDA_METADATA}
struct PolynomialGroupDecoder : PackedCubicMetadata {{
  static constexpr int metadata_planes = {planes};
  template <vllm::ScalarTypeId type> __device__ static void apply(
      typename MarlinScalarType<type>::FragB& fragment,
      const float (&metadata)[{planes}][8], int column) {{
    using D = MarlinScalarType<type>;
    auto values = reinterpret_cast<typename D::scalar_t*>(&fragment);
    {metadata_read}
    #pragma unroll
    for (int z = 0; z < 4; ++z) {{
      float q = D::num2float(values[z]);
      if (q == -{1 << (bits - 1)}.f) q = 0.f;
      // Exact on the finite original-code domain; not an approximate division.
      float t = __fmaf_rn(fabsf(q), {hi.hex()}f, __fmul_rn(fabsf(q), {lo.hex()}f));
      float c = __fsub_rn(__fsub_rn(1.f, a), b);
      float v = __fmul_rn(t, __fadd_rn(a, __fmul_rn(t, __fadd_rn(b, __fmul_rn(t, c)))));
      values[z] = D::float2num(__fmul_rn(q < 0.f ? -v : v, s));
    }}
  }}
}};
}}
#define CUBIC_MARLIN_GROUP_DECODER PolynomialGroupDecoder
#include "marlin_template.h"
namespace cubic_polynomial {{
template __global__ void Marlin<vllm::{dtype}.id(), vllm::{fragment_type}.id(),
    vllm::{dtype}.id(), vllm::kFloat16.id(), {p.marlin_warps * 32}, {p.marlin_block_m // 16},
    {p.marlin_block_n // 16}, {p.marlin_block_k // 16}, false, {p.marlin_stages},
    {group // 16}, false>(MARLIN_KERNEL_PARAMS);
}}"""
