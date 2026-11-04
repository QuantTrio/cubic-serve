"""Compact W2-W8 affine decoding in the shared SM80 WNA16 Marlin pipeline."""

from __future__ import annotations

from typing_extensions import TypeVar

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from torch import Tensor

if TYPE_CHECKING:
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.ops.cuda.native import NativeLaunch

from functools import cache, partial

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]

from cubic_serve.ops.cuda.compiled_artifact import TensorBinding, bind_tensor
from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.ops.cuda.integer_division import invariant_divisor
from cubic_serve.ops.strategies import operator_strategy_scope, parameter_overrides
from cubic_serve.quantization.api import RoutedRows
from cubic_serve.quantization.common.binding import (
    PackedOperatorBinding, _PackedLaunch, concatenate_to_device,
)
from cubic_serve.quantization.common.int8 import (
    pad_int8_groups,
    quantize_rows,
    quantize_rows_padded_groups,
)
from cubic_serve.quantization.common.kernels import _copy_projection_rows as _unpad_output
from cubic_serve.quantization.common.marlin.compiler import bind_launch, compile_module
from cubic_serve.quantization.common.marlin.cost import (
    ProbeFormat, capacity_k_partitions, plan_dense_partitions, select,
)
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

from .checkpoint import fields
from .policy import IntegerLinearParameters, IntegerMoEParameters
from .reference import unpack
from .marlin_source import _a8_source, _source


METADATA_HEADER_WORDS = 4


def _fixture_metadata(
    metadata: Tensor, *, original_group: int, quantum: int, scale_dtype: str
) -> Tensor:
    e, planes, groups, n = metadata.shape
    count = math.ceil(groups * quantum / original_group)
    if n * groups > (1 << 31):
        raise ValueError("INT Marlin metadata indices exceed the signed-32-bit transport")
    # Probe planes are constant; the serialized carrier still owns exactly
    # scale/zero even when two K8 halves need separate register metadata.
    if scale_dtype == "F16":
        # Calibration scale=1, zero=0, encoded with the runtime carrier ABI.
        original = metadata[:, :1, :1].half().view(torch.int16).to(torch.int32)
        original = original.bitwise_and(65535).view(torch.float32).expand(e, 1, count, n)
    else:
        original = metadata[:, :2, :1].expand(e, 2, count, n)
    header = torch.tensor((*invariant_divisor(n), *invariant_divisor(original_group)),
                          device=metadata.device, dtype=torch.int32).view(torch.float32)
    return torch.cat((header, original.reshape(-1)))


@cache
def _fixture_packer(
    original_group: int, quantum: int, scale_dtype: str
) -> Callable[[Tensor], Tensor]:
    return partial(
        _fixture_metadata, original_group=original_group, quantum=quantum, scale_dtype=scale_dtype
    )


@dataclass(frozen=True)
class _PreparedSources:
    # Keep source identities alive for the identity-keyed, weight-owner-local cache.
    # Direct ABI checks avoid rebuilding/hashing a full descriptor for every M.
    weights: tuple[QuantizedMatrix, ...]
    operands: tuple[tuple[Tensor, TensorBinding], ...]

    def matches(self) -> bool:
        return all(binding.matches(tensor) for tensor, binding in self.operands)


def _prepare(
    operators: PackedLinearOperators,
    groups: tuple[tuple[QuantizedMatrix, ...], ...],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    weights = tuple(w for g in groups for w in g)
    key = ("int.marlin.v5", device, tuple(tuple(id(w) for w in g) for g in groups))
    cache = operators.prepared_weights
    cached = cache.get((key, "codes"))
    if cached is not None:
        owner = cached[0]
        if isinstance(owner, _PreparedSources) and owner.matches():
            return cached[1], cache[key, "metadata"][1]
    owner = _PreparedSources(weights, tuple(
        (tensor, bind_tensor(tensor)) for weight in weights for tensor in weight.payload.values()
    ))
    base = groups[0][0]
    bits = base.encoding.bits
    fragment = 4 if bits <= 4 else 8
    n = sum(g[0].shape[0] for g in groups)
    k = base.shape[1]
    experts = len(groups[0])
    count = math.ceil(k / min(k, base.encoding.group_in))
    codes = torch.empty((experts, k // 16, n * bits // 2), device=device, dtype=torch.int32)
    packed = fields(base.encoding)[2] == "F16"
    stored_planes = 1 if packed else 2
    metadata = torch.empty(
        METADATA_HEADER_WORDS + experts * stored_planes * count * n, device=device, dtype=torch.float32
    )
    # Exact division parameters are runtime transport, never kernel specialization.
    header = torch.tensor((*invariant_divisor(n),
                           *invariant_divisor(min(k, base.encoding.group_in))),
                          device=device, dtype=torch.int32).view(torch.float32)
    metadata[:METADATA_HEADER_WORDS].copy_(header)
    target = metadata[METADATA_HEADER_WORDS:].view(experts, stored_planes, count, n)
    permutation = torch.tensor([a + 8 * b for a in range(8) for b in range(8)], device=device)
    # Pack the complete physical output axis. A projection boundary need not
    # be an N64 boundary; experts retain their original outer-axis order.
    ordered = tuple(group[expert] for expert in range(experts) for group in groups)
    packing, symmetry, _ = fields(base.encoding)
    raw = concatenate_to_device(tuple(w.payload["weight_packed"] for w in ordered), device)
    source_bytes = raw.contiguous().view(torch.uint8)
    if packing != "legacy":
        # Dense checkpoint rows are word-padded; an odd-bit K16 tile may end
        # halfway through that final word. Repack consumes only logical bytes.
        source_bytes = source_bytes[:, :math.ceil(k * bits / 8)].contiguous()
    code = repack_signed_codes(
        source_bytes,
        bits=bits,
        fragment_bits=fragment,
        offset_binary=True,
        legacy_words=packing == "legacy",
        logical_k=k,
    )
    codes.copy_(code.view(k // 16, experts, n * bits // 2).transpose(0, 1))
    del raw, code
    scale = concatenate_to_device(tuple(w.payload["weight_scale"] for w in ordered), device)
    scale = scale.view(experts, n, count).transpose(1, 2).contiguous()
    if symmetry == "sym":
        zero = torch.zeros_like(scale)
    else:
        zeros = []
        for group in groups:
            rows = group[0].shape[0]
            raw_zero = concatenate_to_device(
                tuple(w.payload["weight_zero_point"] for w in group), device, dim=1,
            )
            zero = unpack(raw_zero, bits, rows, 0, packing).float()
            zeros.append(zero.view(rows, experts, count).permute(1, 2, 0))
        zero = torch.cat(zeros, dim=2)
    values: tuple[Tensor, ...]
    if packed:
        # Original F16 scale bits and exact integer zero share one uint32.
        scale_bits = scale.view(torch.int16).to(torch.int32).bitwise_and(65535)
        zero_bits = zero.half().view(torch.int16).to(torch.int32)
        values = ((scale_bits | (zero_bits << 16)).view(torch.float32),)
    else:
        values = (scale.float(), zero)
    for plane, value in enumerate(values):
        target[:, plane].copy_(value.reshape(-1, 64)[:, permutation].reshape(experts, count, n))
    cache[key, "codes"] = (owner, codes)
    cache[key, "metadata"] = (owner, metadata)
    return codes, metadata


def _prepare_a8_dense(
    operators: PackedLinearOperators, matrix: QuantizedMatrix, device: torch.device,
    padded_group: int,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Keep original compact codes, F16 scales and signed group zero points."""
    key = ("int.marlin.a8.v2", device, id(matrix), padded_group)
    cache = operators.prepared_weights
    cached = cache.get((key, "codes"))
    if cached is not None:
        owner = cached[0]
        if isinstance(owner, _PreparedSources) and owner.matches():
            return cached[1], cache[key, "scales"][1], cache[key, "zero_scales"][1]
    owner = _PreparedSources(
        (matrix,), tuple((tensor, bind_tensor(tensor)) for tensor in matrix.payload.values())
    )
    n, k = matrix.shape
    bits = matrix.encoding.bits
    group = matrix.encoding.group_in
    groups = math.ceil(k / group)
    padded_k = groups * padded_group
    payload = matrix.payload
    packing, symmetry, _ = fields(matrix.encoding)
    centered = symmetry == "asym" and bits < 8
    stored_bits = bits + int(centered)
    source = payload["weight_packed"].contiguous().view(torch.uint8)
    if packing == "dense":
        source = source[:, : math.ceil(k * bits / 8)]
    source = source.contiguous().to(device)
    fragment = 4 if stored_bits <= 4 else 8
    codes = repack_signed_codes(
        source, bits=stored_bits, source_bits=bits if centered else None,
        zero_points=payload["weight_zero_point"].to(device) if centered else None,
        fragment_bits=fragment, offset_binary=stored_bits != 8,
        logical_k=k, activation_8bit=True, legacy_words=packing == "legacy",
        source_group_in=group if centered or padded_k != k else 0,
        padded_group_in=padded_group if centered or padded_k != k else 0,
    ).view(1, padded_k // 16, n * stored_bits // 2)
    permutation = torch.tensor(
        [2 * a + b for a in range(4) for b in (0, 1, 8, 9, 16, 17, 24, 25)],
        device=device,
    )
    scales = payload["weight_scale"].to(device).view(n, groups).transpose(0, 1).contiguous()
    zero_scales = None
    if symmetry == "asym" and not centered:
        zero = unpack(payload["weight_zero_point"].to(device), bits, n, 0, packing)
        zero_scales = (
            zero.view(n, groups).transpose(0, 1).float() * scales.float()
        ).contiguous()
    scales = scales.reshape(-1, 32)[:, permutation].reshape(-1).contiguous()
    cache[key, "codes"] = (owner, codes)
    cache[key, "scales"] = (owner, scales)
    cache[key, "zero_scales"] = (owner, zero_scales)
    return codes, scales, zero_scales


def combine_a8_matrices(
    operators: PackedLinearOperators, weights: tuple[QuantizedMatrix, ...]
) -> QuantizedMatrix:
    """Make one checkpoint-shaped N axis for same-encoding projections."""
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix

    from .layout import pack

    if len(weights) < 2 or any(
        w.encoding != weights[0].encoding or w.shape[1] != weights[0].shape[1]
        for w in weights[1:]
    ):
        raise ValueError("A8 projection merge requires a common encoding and K")
    key = ("int.a8.grouped.checkpoint.v2", tuple(id(w) for w in weights))
    cached = operators.prepared_weights.get(key)
    if cached is not None and all(
        source is current for source, current in zip(cached[0], weights, strict=True)
    ):
        return cached[1]
    n, k = sum(w.shape[0] for w in weights), weights[0].shape[1]
    payload = {
        "weight_packed": torch.cat(tuple(w.payload["weight_packed"] for w in weights)),
        "weight_scale": torch.cat(tuple(w.payload["weight_scale"] for w in weights)),
        "weight_shape": torch.tensor((n, k), device=weights[0].payload["weight_packed"].device),
    }
    packing, symmetry, _ = fields(weights[0].encoding)
    if symmetry == "asym":
        zero = torch.cat(tuple(
            unpack(w.payload["weight_zero_point"], w.encoding.bits, w.shape[0], 0, packing)
            for w in weights
        ))
        payload["weight_zero_point"] = pack(zero, weights[0].encoding.bits, packing, axis=0)
    matrix = QuantizedMatrix(payload, weights[0].encoding, (n, k))
    operators.prepared_weights[key] = (weights, matrix)
    return matrix


def _prepare_a8_routed(
    operators: PackedLinearOperators, weights: tuple[QuantizedMatrix, ...],
    device: torch.device, padded_group: int,
    *, projection_groups: tuple[tuple[QuantizedMatrix, ...], ...] | None = None,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Write each expert directly into one Marlin owner buffer."""
    key = ("int.marlin.a8.routed.v3", device, tuple(id(w) for w in weights), padded_group)
    if projection_groups is not None:
        key = ("int.marlin.a8.routed.grouped.v1", device,
               tuple(tuple(id(w) for w in group) for group in projection_groups), padded_group)
    cache = operators.prepared_weights
    cached = cache.get((key, "codes"))
    if cached is not None:
        owner = cached[0]
        if isinstance(owner, _PreparedSources) and owner.matches():
            return (cached[1], cache[key, "scales"][1],
                    cache[key, "zero_scales"][1], cache[key, "packed_zeros"][1])
    sources = (weights if projection_groups is None else
               tuple(w for group in projection_groups for w in group))
    owner = _PreparedSources(
        sources, tuple((t, bind_tensor(t)) for w in sources for t in w.payload.values())
    )
    first = weights[0]
    n, k = first.shape
    if projection_groups is not None:
        n = sum(group[0].shape[0] for group in projection_groups)
    bits, group = first.encoding.bits, first.encoding.group_in
    groups = triton.cdiv(k, group)
    padded_k = groups * padded_group
    packing, symmetry, _ = fields(first.encoding)
    stored_bits = bits
    if symmetry == "asym" and bits < 8 and groups * n % 64:
        raise ValueError("A8 zero metadata requires complete per-expert permutation tiles")
    codes = torch.empty(
        (len(weights), padded_k // 16, n * stored_bits // 2),
        device=device, dtype=torch.int32,
    )
    scales = torch.empty((len(weights), groups * n), device=device, dtype=torch.float16)
    zero_scales = (
        torch.empty((len(weights), groups, n), device=device, dtype=torch.float32)
        if symmetry == "asym" and bits == 8 else None
    )
    fragment = 4 if bits <= 4 else 8
    lanes_per_word = 32 // fragment
    packed_zeros = (
        torch.empty((len(weights), groups, n // lanes_per_word),
                    device=device, dtype=torch.int32)
        if symmetry == "asym" and bits < 8 else None
    )
    permutation = torch.tensor(
        [2 * a + b for a in range(4) for b in (0, 1, 8, 9, 16, 17, 24, 25)],
        device=device,
    )
    # Every expert closes N32 tiles. Joining their N axes preserves each
    # kernel tile; split the resulting K32-major stream back in checkpoint order.
    # Bound transient device storage independently of expert/model/device names.
    free, _ = torch.cuda.mem_get_info(device)
    row_bytes = math.ceil(k / (32 // bits)) * 4 if packing == "legacy" else math.ceil(k * bits / 8)
    code_bytes = codes[0].numel() * codes.element_size()
    temporary_per_expert = n * row_bytes + 2 * code_bytes + 32 * groups * n
    chunk = max(1, min(len(weights), min(64 << 20, free // 32) // temporary_per_expert))
    zero_permutation = (None if packed_zeros is None else torch.tensor(
        ([lane * 8 + q for q in range(8) for lane in range(8)]
         if fragment == 4 else [q // 2 + (q % 2) * 8 + lane * 16
                               for q in range(16) for lane in range(4)]), device=device))
    shifts = torch.arange(lanes_per_word, dtype=torch.int32, device=device) * fragment
    for begin in range(0, len(weights), chunk):
        end = min(begin + chunk, len(weights))
        count = end - begin
        ordered = tuple(weights[e] for e in range(begin, end)) if projection_groups is None else tuple(
            projection[e] for e in range(begin, end) for projection in projection_groups)
        packed_parts = tuple(w.payload["weight_packed"].contiguous().view(torch.uint8) for w in ordered)
        if packing == "dense":
            packed_parts = tuple(p[:, :math.ceil(k * bits / 8)] for p in packed_parts)
        source = concatenate_to_device(packed_parts, device)
        prepared = repack_signed_codes(
            source, bits=stored_bits, fragment_bits=fragment,
            offset_binary=stored_bits != 8, logical_k=k, activation_8bit=True,
            legacy_words=packing == "legacy",
            source_group_in=group if padded_k != k else 0,
            padded_group_in=padded_group if padded_k != k else 0,
        )
        codes[begin:end].copy_(prepared.view(padded_k // 32, count, n * stored_bits)
                              .transpose(0, 1).reshape(count, padded_k // 16, n * stored_bits // 2))
        del source, prepared
        original_scale = concatenate_to_device(tuple(w.payload["weight_scale"] for w in ordered), device)
        original_scale = original_scale.view(count, n, groups).transpose(1, 2).contiguous()
        scales[begin:end].copy_(original_scale.reshape(-1, 32)[:, permutation].reshape(count, groups * n))
        if packed_zeros is not None or zero_scales is not None:
            # Encoded zero rows may be joined only at an exact packing boundary.
            # Otherwise decode each original row range before joining metadata.
            closes = all((w.shape[0] % (32 // bits) == 0 if packing == "legacy"
                          else w.shape[0] * bits % 32 == 0) for w in ordered)
            if closes:
                raw_zero = concatenate_to_device(tuple(w.payload["weight_zero_point"] for w in ordered), device)
                zero = unpack(raw_zero, bits, count * n, 0, packing)
            else:
                zero = torch.cat(tuple(unpack(w.payload["weight_zero_point"].to(device), bits,
                                              w.shape[0], 0, packing) for w in ordered))
            zero = zero.view(count, n, groups).transpose(1, 2).contiguous().to(torch.int32)
            if packed_zeros is not None:
                if fragment == 4:
                    zero += 1 << (fragment - 1)
                zero = zero.reshape(-1, 64)[:, zero_permutation].reshape(count, groups, n)
                lanes = zero.reshape(count, groups, n // lanes_per_word, lanes_per_word)
                # Disjoint integer bit fields: sum equals OR, including the sign bit.
                packed = ((lanes & ((1 << fragment) - 1)) << shifts).sum(-1, dtype=torch.int32)
                packed_zeros[begin:end].copy_(packed)
            if zero_scales is not None:
                zero_scales[begin:end].copy_(zero.float() * original_scale.float())
            del zero
            if closes:
                del raw_zero
            if packed_zeros is not None:
                del lanes, packed
        del original_scale
    cache[key, "codes"] = (owner, codes)
    cache[key, "scales"] = (owner, scales)
    cache[key, "zero_scales"] = (owner, zero_scales)
    cache[key, "packed_zeros"] = (owner, packed_zeros)
    return codes, scales, zero_scales, packed_zeros


@triton.jit
def _a8_group_sums(Q, SUMS, PADDED_K: tl.constexpr, GROUP: tl.constexpr,
                   GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1)
    index = tl.arange(0, BLOCK)
    value = tl.load(Q + row * PADDED_K + group * GROUP + index,
                    index < GROUP, 0).to(tl.int32)
    tl.store(SUMS + row * GROUPS + group, tl.sum(value, 0))


@triton.jit
def _a8_correct_zero(RAW, SUMS, XS, ZS, OUT,
                     N: tl.constexpr, GROUPS: tl.constexpr,
                     BG: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0)
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    group = tl.arange(0, BG)
    sums = tl.load(SUMS + row * GROUPS + group, group < GROUPS, 0).to(tl.float32)
    zero_scales = tl.load(ZS + group[:, None] * N + column[None, :],
                          (group[:, None] < GROUPS) & (column[None, :] < N), 0)
    correction = tl.sum(sums[:, None] * zero_scales, 0)
    native = tl.load(RAW + row * N + column, column < N, 0)
    scale = tl.load(XS + row)
    tl.store(OUT + row * N + column, native - scale * correction, column < N)


@triton.jit
def _a8_correct_zero_routed(RAW, SUMS, XS, ZS, IDS, OWNERS, TOTAL, OUT,
                            M: tl.constexpr, TOP_K: tl.constexpr, N: tl.constexpr,
                            GROUPS: tl.constexpr, BM: tl.constexpr,
                            BN: tl.constexpr):
    block = tl.program_id(0)
    lane = tl.arange(0, BM)
    padded = block * BM + lane
    active = padded < tl.load(TOTAL)
    route = tl.load(IDS + padded, active, M * TOP_K)
    expert = tl.load(OWNERS + block, block * BM < tl.load(TOTAL), 0)
    row = route // TOP_K
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    valid_row = route < M * TOP_K
    valid = valid_row[:, None] & (column[None, :] < N)
    correction = tl.full((BM, BN), 0.0, tl.float32)
    for group in range(GROUPS):
        sums = tl.load(SUMS + row * GROUPS + group, valid_row, 0).to(tl.float32)
        zero_scale = tl.load(ZS + (expert * GROUPS + group) * N + column,
                             (block * BM < tl.load(TOTAL)) & (column < N), 0)
        correction += sums[:, None] * zero_scale[None, :]
    native = tl.load(RAW + route[:, None] * N + column[None, :], valid, 0)
    scale = tl.load(XS + row, valid_row, 0)
    tl.store(OUT + route[:, None] * N + column[None, :],
             native - scale[:, None] * correction, valid)




Parameters = TypeVar("Parameters", IntegerLinearParameters, IntegerMoEParameters)


def _align_a8_geometry(
    params: Parameters, *, n: int, k: int, bits: int, group: int,
) -> Parameters:
    """Close Marlin's paired A8 K subtile load with an N tile that fits N."""
    fragment = 4 if bits <= 4 else 8
    pack_factor = 32 // fragment
    vectors = 1 if fragment == 4 else 2
    for bn in (params.marlin_block_n, 64, 128, 256, 512):
        if bn < params.marlin_block_n or n % bn or bn % 16:
            continue
        stride = bn * 16 // pack_factor // 2
        iterations = (
            stride * (params.marlin_block_k // 16)
            // 2 // (params.marlin_warps * 32 * vectors)
        )
        candidate = replace(params, marlin_block_n=bn)
        if (bn <= params.marlin_warps * 32
                and (iterations == 1 or (iterations >= 2 and iterations % 2 == 0))
                and pipeline_compatible(candidate, group=group, single_subtile=True)):
            return candidate
    return params


def _dense_arithmetic(operators, x, params, choose, *, key, minimum_partitions):
    """Retain the actual native reduction in the shared cross-M proof inventory.

    Only explicit audit anchors restore a recipe. Normal graph binding keeps
    the existing selector; replay never reads this metadata or audit context.
    """
    from dataclasses import asdict
    from cubic_serve.quantization.common.small_rows_cost import (
        audited_row_recipe, record_row_recipe,
    )

    selected = audited_row_recipe(key, x.shape[0])
    if selected is None:
        chosen, partitions = choose()
    else:
        if not isinstance(selected, tuple) or len(selected) != 3 or selected[0] != key[0]:
            raise ValueError("INT Marlin audit recipe has an unknown reduction contract")
        chosen = type(params)(**dict(selected[1]))
        partitions = selected[2]
        if (tuple(asdict(chosen).items()) != selected[1]
                or type(partitions) is not int or partitions < minimum_partitions):
            raise ValueError("INT Marlin audit recipe has malformed native parameters")
    recipe = (key[0], tuple(asdict(chosen).items()), partitions)
    record_row_recipe(operators, key, x.shape[0], recipe)
    return chosen, partitions


def _a8_dense_arithmetic(operators, x, matrix, output, params, choose, *, grouped=False):
    # Preserve the established A8 inventory identity and zero-partial sentinel.
    key = ("int.marlin-a8.row-arithmetic.v1", x.dtype, output.dtype,
           matrix.encoding, matrix.shape, matrix.partitions, params, grouped)
    return _dense_arithmetic(
        operators, x, params, choose, key=key, minimum_partitions=0,
    )


def _a16_dense_arithmetic(operators, x, groups, output, params, choose):
    # A grouped binding's full geometry matters, not only its first member.
    key = ("int.marlin-a16.row-arithmetic.v1", x.dtype, output.dtype,
           groups[0][0].encoding,
           tuple(tuple((w.shape, w.partitions) for w in group) for group in groups),
           params)
    return _dense_arithmetic(
        operators, x, params, choose, key=key, minimum_partitions=1,
    )


def bind_a8_dense_symmetric(
    operators: PackedLinearOperators,
    x: Tensor,
    matrix: QuantizedMatrix,
    output: Tensor,
    params: IntegerLinearParameters,
    *,
    grouped_outputs: tuple[Tensor, ...] | None = None,
) -> PackedOperatorBinding:
    """Native Marlin A8 for original dense W2-W8 groups, including affine zero."""
    n, k = matrix.shape
    logical_ns = tuple(y.shape[1] for y in grouped_outputs or ())
    if grouped_outputs is not None and (
        not grouped_outputs or sum(logical_ns) > n
        or any(y.shape[0] != x.shape[0] or y.dtype != output.dtype
               or y.device != output.device for y in grouped_outputs)
    ):
        raise ValueError("grouped A8 outputs disagree with the combined projection")
    packing, symmetry, scale_type = fields(matrix.encoding)
    inline_zero = symmetry == "asym" and matrix.encoding.bits < 8
    stored_bits = matrix.encoding.bits
    correction = symmetry == "asym" and not inline_zero
    group = matrix.encoding.group_in
    padded_group = triton.next_power_of_2(max(32, group))
    padded_k = triton.cdiv(k, group) * padded_group
    params = _align_k_tile(operators, params, n, padded_k, family="int_linear")
    params = _align_a8_geometry(
        params, n=n, k=padded_k, bits=stored_bits, group=padded_group
    )
    alignment = math.lcm(64, params.marlin_block_n)
    if n % alignment:
        padded_n = triton.cdiv(n, alignment) * alignment
        padded = _pad_output_rows(operators, matrix, padded_n)
        temporary = torch.empty(
            (x.shape[0], padded_n), device=x.device, dtype=output.dtype
        )
        binding = bind_a8_dense_symmetric(
            operators, x, padded, temporary, params,
            grouped_outputs=grouped_outputs,
        )
        if grouped_outputs is not None and binding.output is grouped_outputs[0]:
            return PackedOperatorBinding(
                grouped_outputs[0], binding._launches,
                scratch=(*binding.scratch, temporary),
                reusable_scratch=binding.reusable_scratch,
            reusable_scratch_indices=(*binding.reusable_scratch_indices, len(binding.scratch)),
            )
        crop = _PackedLaunch.compile(
            _unpad_output, (temporary, output),
            (output.numel(), n, padded_n, 256),
            (triton.cdiv(output.numel(), 256),), dict(num_warps=4),
        )
        return PackedOperatorBinding(
            output, (*binding._launches, crop), scratch=(*binding.scratch, temporary),
            reusable_scratch=binding.reusable_scratch,
            reusable_scratch_indices=(*binding.reusable_scratch_indices, len(binding.scratch)),
        )
    if (
        packing not in ("dense", "legacy") or scale_type != "F16"
        or not 2 <= matrix.encoding.bits <= 8 or not params.marlin_enabled
        or params.marlin_block_m < 16 or n % 64 or n % params.marlin_block_n
        or padded_k % params.marlin_block_k
        or not pipeline_compatible(params, group=padded_group, single_subtile=True)
    ):
        raise ValueError("native INT A8 Marlin requires packed W2-W8/F16 groups")
    encoded = operators.int8_input(x) if x.dtype == torch.int8 else None
    if encoded is None and x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("native INT A8 Marlin requires floating or registered INT8 input")
    fragment = 4 if stored_bits <= 4 else 8
    code = (1 << (fragment - 1)) + 1 if fragment == 4 else 1
    word = sum(code << (j * fragment) for j in range(32 // fragment))
    word = word if word < 2**31 else word - 2**32
    probe = ProbeFormat(
        fragment, torch.int8,
        torch.float32 if correction else output.dtype,
        torch.float16, word, 1.0, 1.0, activation_scale=1.0,
        dense_linear=True, storage_bits=stored_bits,
        group_storage_bytes=2 + (fragment / 8 if inline_zero else 0),
        zero_word=(((-2004318072) if fragment == 4 else 0) if inline_zero else None),
        min_block_m=16, single_subtile=True,
        native_dense_reduce=grouped_outputs is None or correction,
    )
    def choose(initial=params):
        params = initial
        partitions = plan_dense_partitions(
            params, operators=operators,
            source=lambda count: _a8_source(
                params, "kBFloat16" if output.dtype == torch.bfloat16 else "kFloat16",
                stored_bits, n, padded_k, padded_group,
                inline_zero=inline_zero, fp32_output=correction, fixed_partitions=count,
            ),
            spec=probe, device=x.device, n=n, k=padded_k,
            group=padded_group, partition_group=padded_group,
            max_partitions=params.marlin_max_k_partitions,
        )
        # A single full-K slice needs no partial buffer or merge.
        partitions = 0 if partitions == 1 else partitions
        params = select(
            params,
            operators=operators,
            family="int_linear",
            scope=dict(n=n, k=k),
            source=lambda candidate: _a8_source(
                candidate,
                "kBFloat16" if output.dtype == torch.bfloat16 else "kFloat16",
                stored_bits, n, padded_k, padded_group,
                inline_zero=inline_zero, fp32_output=correction, fixed_partitions=partitions,
            ),
            spec=replace(probe, fixed_partitions=partitions),
            device=x.device, n=n, k=padded_k, experts=1,
            group=padded_group, rows=x.shape[0], routed=False,
            # A wider row tile amortizes packed-code loads for large M. Its smaller
            # pipeline is a resource-compatible candidate, not a model selector.
            alternates=(replace(params, marlin_block_m=64, marlin_stages=2),),
        )
        # Price the work supplied by the final bound tile, rather than replicating
        # the sparse-row K scratch across an already populated physical graph.
        partitions = capacity_k_partitions(
            params, partitions=partitions or 1, rows=x.shape[0], n=n, k=padded_k,
            group=padded_group, spec=probe, device=x.device,
            source=lambda count: _a8_source(
                params, "kBFloat16" if output.dtype == torch.bfloat16 else "kFloat16",
                stored_bits, n, padded_k, padded_group,
                inline_zero=inline_zero, fp32_output=correction, fixed_partitions=0 if count == 1 else count,
            ),
        )
        partitions = 0 if partitions == 1 else partitions
        return params, partitions

    params, partitions = _a8_dense_arithmetic(
        operators, x, matrix, output, params, choose,
        grouped=grouped_outputs is not None,
    )
    if (n % params.marlin_block_n or padded_k % params.marlin_block_k
            or not pipeline_compatible(params, group=padded_group, single_subtile=True)
            or padded_k % ((partitions or 1) * math.lcm(padded_group, params.marlin_block_k))):
        raise ValueError("INT A8 bound reduction violates its native geometry")
    weight = matrix
    if inline_zero:
        codes, scales, zero_scales, packed_zeros = _prepare_a8_routed(operators, (weight,), x.device, padded_group)
    else:
        codes, scales, zero_scales = _prepare_a8_dense(operators, weight, x.device, padded_group)
        packed_zeros = None
    q = (
        encoded.codes if encoded is not None and padded_k == k
        else torch.empty((x.shape[0], padded_k), device=x.device, dtype=torch.int8)
    )
    xs = encoded.scales if encoded is not None else torch.empty(
        (x.shape[0],), device=x.device, dtype=torch.float32
    )
    dtype = "kBFloat16" if output.dtype == torch.bfloat16 else "kFloat16"
    raw = (torch.empty((x.shape[0], n), device=x.device, dtype=torch.float32)
           if zero_scales is not None else output)
    tiles = triton.cdiv(x.shape[0], params.marlin_block_m) * (n // params.marlin_block_n)
    partials = (
        torch.empty(
            tiles * partitions * params.marlin_block_m * params.marlin_block_n,
            device=x.device, dtype=torch.float32,
        ) if partitions else None
    )
    module, entry = compile_module(
        _a8_source(params, dtype, stored_bits, n, padded_k, padded_group,
                   inline_zero=inline_zero, fp32_output=zero_scales is not None, fixed_partitions=partitions),
        x.device,
    )
    args = (
        q, codes, raw, partials, None, xs, scales, None, packed_zeros, None,
        None, None, None, None,
        CudaScalar("i32", 1), CudaScalar("bool", False),
        CudaScalar("i32", padded_k // padded_group), CudaScalar("i32", x.shape[0]),
        CudaScalar("i32", n), CudaScalar("i32", padded_k), None,
        CudaScalar("bool", False), CudaScalar("bool", False), CudaScalar("bool", True),
        CudaScalar("i32", partitions),
    )
    shared = shared_bytes(
        params, bits=stored_bits, group=padded_group,
        row_metadata=False, zero_points=inline_zero
    )
    direct_outputs = grouped_outputs if partitions and zero_scales is None else None
    if partitions:
        launch = module.bind(
            entry, args, operator_id="cubic.cuda.int.marlin.linear.a8.v1",
            grid=(tiles * partitions, 1, 1),
            block=(params.marlin_warps * 32, 1, 1), shared_bytes=shared,
        )
        if direct_outputs is None:
            from cubic_serve.quantization.common.marlin.a8_partial_reduce import bind as bind_partial_reduce
            merge = bind_partial_reduce(partials, raw, m=x.shape[0], n=n,
                                        block_m=params.marlin_block_m,
                                        block_n=params.marlin_block_n, partitions=partitions)
        else:
            merge = _PackedLaunch.compile(
                _merge_dense_partials,
                (partials, raw if direct_outputs is None else direct_outputs),
                (x.shape[0], n, params.marlin_block_m, params.marlin_block_n,
                 partitions, 256,
                 () if direct_outputs is None else logical_ns,
                 () if direct_outputs is None else (*logical_ns[:-1], n - sum(logical_ns[:-1])),
                 False, True),
                (triton.cdiv(x.shape[0] * n, 256),),
                dict(num_warps=4, enable_fp_fusion=False),
            )
    else:
        launch = bind_launch(
            module, entry, args, operator_id="cubic.cuda.int.marlin.linear.a8.v1",
            params=params, output_tiles=tiles, shared_bytes=shared,
        )
        merge = None
    if encoded is not None and padded_k == k:
        calls = ()
    elif encoded is not None:
        calls = (_PackedLaunch.compile(
            pad_int8_groups, (x, q),
            (k, padded_k, group, padded_group, triton.next_power_of_2(padded_k)),
            (x.shape[0],), dict(num_warps=4, enable_fp_fusion=False),
        ),)
    elif padded_k != k:
        calls = (_PackedLaunch.compile(
            quantize_rows_padded_groups, (x, q, xs),
            (k, padded_k, group, padded_group, triton.next_power_of_2(padded_k)),
            (x.shape[0],), dict(num_warps=4, enable_fp_fusion=False),
        ),)
    else:
        calls = (_PackedLaunch.compile(
            quantize_rows, (x, q, xs), (k, triton.next_power_of_2(k)),
            (x.shape[0],), dict(num_warps=4, enable_fp_fusion=False),
        ),)
    scratch = ((q,) if encoded is not None and padded_k != k else
               () if encoded is not None else (q, xs))
    if partials is not None:
        # The merge consumes all partials within this binding. Keep activation
        # codes/scales private while exposing only partials to sealed reuse.
        scratch = (partials, *scratch)
    compute = (launch,) if merge is None else (launch, merge)
    if zero_scales is not None:
        groups = triton.cdiv(k, group)
        sums = torch.empty((x.shape[0], groups), device=x.device, dtype=torch.int32)
        sum_call = _PackedLaunch.compile(
            _a8_group_sums, (q, sums),
            (padded_k, padded_group, groups, triton.next_power_of_2(padded_group)),
            (x.shape[0], groups), dict(num_warps=4),
        )
        finish = _PackedLaunch.compile(
            _a8_correct_zero, (raw, sums, xs, zero_scales, output),
            (n, groups, triton.next_power_of_2(groups), 64),
            (x.shape[0], triton.cdiv(n, 64)),
            dict(num_warps=4, enable_fp_fusion=False),
        )
        return PackedOperatorBinding(
            output, (*calls, sum_call, *compute, finish), scratch=(*scratch, raw, sums),
            reusable_scratch=partials is not None,
            reusable_scratch_indices=tuple(range(len(scratch) + 2)),
        )
    return PackedOperatorBinding(
        direct_outputs[0] if direct_outputs is not None else output,
        (*calls, *compute), scratch=scratch,
        reusable_scratch=partials is not None,
        reusable_scratch_indices=tuple(range(len(scratch))),
    )




def bind_a8_routed(
    operators: PackedLinearOperators,
    x: Tensor,
    weights: tuple[QuantizedMatrix, ...],
    output: Tensor,
    rows: RoutedRows,
    params: IntegerMoEParameters,
    top_k: int = 1,
    *, projection_groups: tuple[tuple[QuantizedMatrix, ...], ...] | None = None,
) -> tuple[int, PackedOperatorBinding]:
    """Original expert ownership with the shared Marlin signed-byte MMA."""
    first = weights[0]
    n, k = first.shape
    encoding = first.encoding
    packing, symmetry, scale_type = fields(encoding)
    if projection_groups is not None:
        if not projection_groups or any(
            len(group) != len(weights) or not group
            or any(w.encoding != encoding or w.shape != group[0].shape
                   or w.shape[1] != k for w in group)
            for group in projection_groups
        ):
            raise ValueError("INT A8 grouped experts require common encoding, K and ownership")
        n = sum(group[0].shape[0] for group in projection_groups)
        # Zero points are packed along N. Concatenating encoded words is exact
        # only when every member ends at a complete format-owned word boundary.
        word_rows = 32 // (math.gcd(encoding.bits, 32) if packing == "dense"
                          else encoding.bits)
        if symmetry == "asym" and any(group[0].shape[0] % word_rows
                                      for group in projection_groups):
            combined = tuple(combine_a8_matrices(
                operators, tuple(group[expert] for group in projection_groups)
            ) for expert in range(len(weights)))
            return bind_a8_routed(operators, x, combined, output, rows, params, top_k)
    inline_zero = symmetry == "asym" and encoding.bits < 8
    stored_bits = encoding.bits
    group = encoding.group_in
    padded_group = triton.cdiv(max(32, group), 32) * 32
    power_two_group = triton.next_power_of_2(max(32, group))
    if padded_group != power_two_group:
        compact_k = triton.cdiv(k, group) * padded_group
        compact_bk = params.marlin_block_k
        if padded_group % compact_bk and compact_bk % padded_group:
            compact_bk = math.gcd(padded_group, compact_bk)
        group_tiles = max(1, padded_group // compact_bk)
        compact_stages = (params.marlin_stages if params.marlin_stages % group_tiles == 0
                          else group_tiles)
        with operator_strategy_scope(
            operators.config, device=operators.device_identity, model=operators.model_identity
        ):
            locked = parameter_overrides(
                "int_moe", n=n, k=compact_k, experts=len(weights),
                top_k=rows.routing_top_k or top_k,
            )
        if (not params.marlin_auto_geometry
                or {"marlin_block_k", "marlin_stages"}.intersection(locked)
                or compact_bk not in (16, 32, 64, 128, 256)
                or compact_stages not in (2, 3, 4)
                or compact_k % compact_bk):
            padded_group = power_two_group
        else:
            params = replace(
                params, marlin_block_k=compact_bk, marlin_stages=compact_stages
            )
    padded_k = triton.cdiv(k, group) * padded_group
    params = _align_k_tile(
        operators, params, n, padded_k, family="int_moe",
        experts=len(weights), top_k=rows.routing_top_k or top_k,
    )
    params = _align_a8_geometry(
        params, n=n, k=padded_k, bits=stored_bits, group=padded_group
    )
    alignment = math.lcm(64, params.marlin_block_n)
    if n % alignment:
        if projection_groups is not None:
            combined = tuple(combine_a8_matrices(
                operators, tuple(group[expert] for group in projection_groups)
            ) for expert in range(len(weights)))
            return bind_a8_routed(operators, x, combined, output, rows, params, top_k)
        padded_n = triton.cdiv(n, alignment) * alignment
        padded = tuple(_pad_output_rows(operators, w, padded_n) for w in weights)
        temporary = torch.empty(
            (x.shape[0] * top_k, padded_n), device=x.device, dtype=output.dtype
        )
        bm, binding = bind_a8_routed(
            operators, x, padded, temporary, rows, params, top_k
        )
        crop = _PackedLaunch.compile(
            _unpad_output, (temporary, output),
            (output.numel(), n, padded_n, 256),
            (triton.cdiv(output.numel(), 256),), dict(num_warps=4),
        )
        return bm, PackedOperatorBinding(
            output, (*binding._launches, crop), scratch=(*binding.scratch, temporary)
        )
    if (
        packing not in ("dense", "legacy") or scale_type != "F16"
        or not 2 <= encoding.bits <= 8 or not params.marlin_enabled
        or params.marlin_block_m < 16 or n % 64 or n % params.marlin_block_n
        or padded_k % params.marlin_block_k
        or not pipeline_compatible(params, group=padded_group, single_subtile=True)
        or (projection_groups is None and any(
            w.encoding != encoding or w.shape != (n, k) for w in weights))
    ):
        raise ValueError("native routed INT A8 Marlin requires aligned W2-W8/F16 groups")
    encoded = operators.int8_input(x) if x.dtype == torch.int8 else None
    if encoded is None and x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("native routed INT A8 Marlin requires floating or registered INT8 input")
    device = x.device
    codes, scales, zero_scales, packed_zeros = _prepare_a8_routed(
        operators, weights, device, padded_group, projection_groups=projection_groups
    )
    q = (encoded.codes if encoded is not None and padded_k == k else
         torch.empty((x.shape[0], padded_k), device=device, dtype=torch.int8))
    xs = encoded.scales if encoded is not None else torch.empty(
        (x.shape[0],), device=device, dtype=torch.float32
    )
    if encoded is not None and padded_k == k:
        calls: tuple[_PackedLaunch, ...] = ()
    elif encoded is not None:
        calls = (_PackedLaunch.compile(
            pad_int8_groups, (x, q),
            (k, padded_k, group, padded_group, triton.next_power_of_2(padded_k)),
            (x.shape[0],), dict(num_warps=4, enable_fp_fusion=False),
        ),)
    elif padded_k != k:
        calls = (_PackedLaunch.compile(
            quantize_rows_padded_groups, (x, q, xs),
            (k, padded_k, group, padded_group, triton.next_power_of_2(padded_k)),
            (x.shape[0],), dict(num_warps=4, enable_fp_fusion=False),
        ),)
    else:
        calls = (_PackedLaunch.compile(
            quantize_rows, (x, q, xs), (k, triton.next_power_of_2(k)),
            (x.shape[0],), dict(num_warps=4, enable_fp_fusion=False),
        ),)
    routes = x.shape[0] * top_k
    bm = params.marlin_block_m
    blocks = triton.cdiv(routes + min(routes, len(weights)) * (bm - 1), bm)
    ids = torch.empty(blocks * bm, device=device, dtype=torch.int32)
    owners = torch.empty(blocks, device=device, dtype=torch.int32)
    total = torch.empty(1, device=device, dtype=torch.int32)
    route_weights = torch.ones(x.shape[0], top_k, device=device)
    pad = _PackedLaunch.compile(
        _pad_routes,
        (rows.counts, rows.starts, rows.block_starts, rows.total_blocks,
         rows.row_map, ids, owners, total),
        (routes, bm), (len(weights),), {},
    )
    sums = None
    sum_call = None
    if zero_scales is not None and not inline_zero:
        groups = triton.cdiv(k, group)
        sums = torch.empty((x.shape[0], groups), device=device, dtype=torch.int32)
        sum_call = _PackedLaunch.compile(
            _a8_group_sums, (q, sums),
            (padded_k, padded_group, groups, triton.next_power_of_2(padded_group)),
            (x.shape[0], groups), dict(num_warps=4),
        )
    dtype = "kBFloat16" if output.dtype == torch.bfloat16 else "kFloat16"
    external_zero = zero_scales is not None and not inline_zero
    raw = (torch.empty((routes, n), device=device, dtype=torch.float32)
           if external_zero else output)
    module, entry = compile_module(
        _a8_source(params, dtype, stored_bits, n, padded_k, padded_group,
                   fp32_output=external_zero, dense=False,
                   inline_zero=inline_zero), device,
    )
    args = (
        q, codes, raw, None, None, xs, scales, None, packed_zeros, None,
        ids, owners, total, route_weights,
        CudaScalar("i32", top_k), CudaScalar("bool", False),
        CudaScalar("i32", padded_k // padded_group), CudaScalar("i32", x.shape[0]),
        CudaScalar("i32", n), CudaScalar("i32", padded_k), None,
        CudaScalar("bool", False), CudaScalar("bool", False), CudaScalar("bool", True),
        CudaScalar("i32", 0),
    )
    shared = shared_bytes(
        params, bits=stored_bits, group=padded_group,
        row_metadata=True, zero_points=inline_zero,
    )
    launch = bind_launch(
        module, entry, args, operator_id="cubic.cuda.int.marlin.routed.a8.v1",
        params=params, output_tiles=blocks * (n // params.marlin_block_n),
        shared_bytes=shared,
    )
    scratch: tuple[Tensor, ...] = (ids, owners, total, route_weights)
    if encoded is None:
        scratch += (q, xs)
    elif q is not encoded.codes:
        scratch += (q,)
    if external_zero:
        finish = _PackedLaunch.compile(
            _a8_correct_zero_routed,
            (raw, sums, xs, zero_scales, ids, owners, total, output),
            (x.shape[0], top_k, n, groups, bm, 64),
            (blocks, triton.cdiv(n, 64)),
            dict(num_warps=4, enable_fp_fusion=False),
        )
        return bm, PackedOperatorBinding(
            output, (*calls, pad, sum_call, launch, finish),
            scratch=(*scratch, raw, sums, codes, scales, zero_scales),
        )
    return bm, PackedOperatorBinding(
        output, (*calls, pad, launch),
        scratch=(*scratch, codes, scales,
                 *((packed_zeros,) if packed_zeros is not None else ())),
    )




def _bind(
    operators: PackedLinearOperators,
    x: Tensor,
    groups: tuple[tuple[QuantizedMatrix, ...], ...],
    output: Tensor,
    p: IntegerLinearParameters | IntegerMoEParameters,
    rows: RoutedRows | None = None,
    top_k: int = 1,
    *,
    logical_n: int | None = None,
    dense_outputs: tuple[Tensor, ...] | None = None,
) -> PackedOperatorBinding | tuple[int, PackedOperatorBinding] | None:
    if operators.config.compute_mode not in ("wna16", "wna8"):
        raise ValueError("INT Marlin requires WNA16 or WNA8")
    base = groups[0][0]
    enc = base.encoding
    n = sum(g[0].shape[0] for g in groups)
    k = base.shape[1]
    e = len(groups[0])
    gi = min(k, enc.group_in)
    alignment = math.gcd(gi, p.marlin_block_k)
    quantum = max(16, alignment)
    stored_planes = 1 if fields(enc)[2] == "F16" else 2
    planes = stored_planes * (2 if gi % quantum else 1)
    if n * math.ceil(k / quantum) > (1 << 31):
        raise ValueError("INT Marlin metadata indices exceed the signed-32-bit transport")
    if (
        not p.marlin_enabled
        or x.dtype not in (torch.float16, torch.bfloat16)
        or alignment < 8
        or alignment % 8
        or k % gi
        or k % p.marlin_block_k
        or n % 64
        or n % p.marlin_block_n
        or any(
            w.encoding != enc or w.shape != g[0].shape or w.shape[1] != k
            for g in groups
            for w in g
        )
    ):
        return None
    if (p.marlin_warps * 64) % p.marlin_block_n:
        raise ValueError("INT Marlin requires an integral warp K reduction geometry")
    if not pipeline_compatible(p, group=quantum, single_subtile=True):
        return None
    partitions = 0
    dtype = "kBFloat16" if x.dtype == torch.bfloat16 else "kFloat16"
    fragment = 4 if enc.bits <= 4 else 8
    code = (1 << (fragment - 1)) + 1
    word = sum(code << (j * fragment) for j in range(32 // fragment))
    word = word if word < 2**31 else word - 2**32
    spec = ProbeFormat(
        fragment,
        x.dtype,
        x.dtype,
        torch.float16,
        word,
        1.0,
        1.0,
        group_metadata=(1.0,) * planes if stored_planes == 1 else (1.0, 0.0) * (planes // 2),
        dense_linear=rows is None,
        storage_bits=enc.bits,
        group_storage_bytes=4 * stored_planes * quantum / gi,
        metadata_packer=_fixture_packer(gi, quantum, fields(enc)[2]),
        min_block_m=8,
        single_subtile=True,
    )
    capacity_dense = (
        rows is None and operators.config.compute_mode == "wna16"
        and operators.dense_capacity_partitions
    )

    def choose(initial=p, initial_spec=spec):
        p, spec = initial, initial_spec
        partitions = plan_dense_partitions(
            p,
            operators=operators,
            source=lambda count: _source(
                p, dtype, enc.bits, gi, quantum, fields(enc)[2], rows is None,
                count if rows is None or count > 1 else 0, n=n, k=k,
            ),
            spec=spec,
            device=x.device,
            n=n,
            k=k,
            group=quantum,
            # The decoder indexes original groups by absolute K, including
            # K8 halves. A compute slice need only align its metadata quantum;
            # no serialized group is split, rewritten or requantized.
            partition_group=quantum,
            max_partitions=p.marlin_max_k_partitions,
            experts=e,
            routing_top_k=1 if rows is None else (rows.routing_top_k or top_k),
        )
        if rows is not None and partitions == 1:
            partitions = 0
        spec = replace(spec, fixed_partitions=partitions)
        p = select(
            p,
            operators=operators,
            family="int_linear" if rows is None else "int_moe",
            scope=dict(
                n=base.shape[0] if logical_n is None else logical_n,
                k=k,
                **({} if rows is None else dict(experts=e, top_k=rows.routing_top_k or top_k)),
            ),
            source=lambda candidate: _source(
                candidate, dtype, enc.bits, gi, quantum, fields(enc)[2], rows is None, partitions,
                n=n, k=k,
            ),
            spec=spec,
            device=x.device,
            n=n,
            k=k,
            experts=e,
            group=quantum,
            rows=x.shape[0]
            if rows is None
            else (
                x.shape[0] * top_k / e
                if rows.mean_rows_per_expert is None
                else rows.mean_rows_per_expert
            ),
            routed=rows is not None,
        )
        if capacity_dense:
            partitions = capacity_k_partitions(
                p, partitions=partitions, rows=x.shape[0], n=n, k=k,
                group=quantum, spec=spec, device=x.device,
                source=lambda count: _source(
                    p, dtype, enc.bits, gi, quantum, fields(enc)[2], True,
                    count, n=n, k=k,
                ),
            )
        return p, partitions

    if capacity_dense:
        p, partitions = _a16_dense_arithmetic(
            operators, x, groups, output, p, choose,
        )
        if (n % p.marlin_block_n or k % p.marlin_block_k
                or not pipeline_compatible(p, group=quantum, single_subtile=True)
                or k % (partitions * math.lcm(quantum, p.marlin_block_k))):
            raise ValueError("INT A16 bound reduction violates its native geometry")
    else:
        p, partitions = choose()
    shared = shared_bytes(
        p,
        bits=enc.bits,
        group=quantum,
        metadata_planes=planes,
        row_metadata=rows is not None,
        zero_points=False,
    )
    hw = torch.cuda.get_device_properties(x.device)
    if shared > getattr(hw, "shared_memory_per_block_optin", hw.shared_memory_per_block):
        return None
    dtype = "kBFloat16" if x.dtype == torch.bfloat16 else "kFloat16"
    module, entry = compile_module(
        _source(
            p, dtype, enc.bits, gi, quantum, fields(enc)[2], rows is None, partitions,
            n=n, k=k,
        ), x.device
    )
    weight, metadata = _prepare(operators, groups, x.device)
    m = x.shape[0]
    bm = p.marlin_block_m
    calls: tuple[_PackedLaunch, ...] = ()
    scratch: tuple[Tensor, ...] = ()
    ids = owners = total = rw = None
    if rows is None:
        blocks = math.ceil(m / bm)
    else:
        routes = m * top_k
        blocks = math.ceil((routes + min(routes, e) * (bm - 1)) / bm)
        ids = torch.empty(blocks * bm, device=x.device, dtype=torch.int32)
        owners = torch.empty(blocks, device=x.device, dtype=torch.int32)
        total = torch.empty(1, device=x.device, dtype=torch.int32)
        rw = torch.ones(m, top_k, device=x.device)
        calls = (
            _PackedLaunch.compile(
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
            ),
        )
        scratch = (ids, owners, total, rw)
    tiles = blocks * (n // p.marlin_block_n)
    partials = (
        torch.empty(
            tiles * partitions * bm * p.marlin_block_n, device=x.device, dtype=torch.float32
        )
        if partitions
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
        ids,
        owners,
        total,
        rw,
        CudaScalar("i32", top_k),
        CudaScalar("bool", False),
        CudaScalar("i32", k // quantum),
        CudaScalar("i32", m),
        CudaScalar("i32", n),
        CudaScalar("i32", k),
        None,
        CudaScalar("bool", False),
        CudaScalar("bool", False),
        CudaScalar("bool", True),
        CudaScalar("i32", partitions),
    )
    if partitions:
        assert partials is not None
        partial_input: Tensor | tuple[Tensor, ...] = partials
        if rows is not None:
            assert ids is not None and total is not None
            partial_input = (partials, ids, total)
        launch = module.bind(
            entry,
            args,
            operator_id="cubic.cuda.int.marlin."
            + ("linear" if rows is None else "routed") + ".partial.v1",
            grid=(tiles * partitions, 1, 1),
            block=(p.marlin_warps * 32, 1, 1),
            shared_bytes=shared,
        )
        finish = _PackedLaunch.compile(
            _merge_dense_partials,
            (partial_input, output if dense_outputs is None else dense_outputs),
            (m if rows is None else m * top_k, n, bm, p.marlin_block_n, partitions, 256,
             () if dense_outputs is None else tuple(y.shape[1] for y in dense_outputs),
             () if dense_outputs is None else tuple(g[0].shape[0] for g in groups),
             rows is not None),
            (math.ceil((m if rows is None else blocks * bm) * n / 256),),
            dict(num_warps=4, enable_fp_fusion=False),
        )
        binding = PackedOperatorBinding(
            output if dense_outputs is None else dense_outputs[0],
            (*calls, launch, finish), scratch=(partials, *scratch),
            # The merge consumes every partial before the next projection.
            # Only that first tensor opts in to serial-layer reuse; routed
            # row/owner metadata keeps its original private lifetime.
            reusable_scratch=True,
        )
        return binding if rows is None else (bm, binding)
    launch = bind_launch(
        module,
        entry,
        args,
        operator_id="cubic.cuda.int.marlin." + ("linear" if rows is None else "routed") + ".v1",
        params=p,
        output_tiles=blocks * (n // p.marlin_block_n),
        shared_bytes=shared,
    )
    binding = PackedOperatorBinding(output, (*calls, launch), scratch=scratch)
    return binding if rows is None else (bm, binding)


def _pad_output_rows(
    operators: PackedLinearOperators, matrix: QuantizedMatrix, padded_n: int
) -> QuantizedMatrix:
    """Pad integer zero rows once, preserving all logical codes and group metadata."""
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix

    from .layout import pack

    key = (
        "int.n-padding",
        matrix.encoding,
        matrix.shape,
        matrix.partitions,
        padded_n,
        tuple((name, bind_tensor(t)) for name, t in sorted(matrix.payload.items())),
    )
    cached = operators.materialized_weights.get(key)
    if cached is not None:
        return cached[1]
    n, k = matrix.shape
    e = matrix.encoding
    packing, symmetry, _ = fields(e)
    p = matrix.payload
    device = p["weight_packed"].device
    codes = pack(torch.zeros((padded_n - n, k), dtype=torch.int8, device=device), e.bits, packing)
    scales = p["weight_scale"]
    payload = {
        "weight_packed": torch.cat((p["weight_packed"], codes)),
        "weight_scale": torch.cat((scales, scales.new_ones((padded_n - n, scales.shape[1])))),
        "weight_shape": torch.tensor((padded_n, k), device=device, dtype=torch.int64),
    }
    if symmetry == "asym":
        zero = unpack(p["weight_zero_point"], e.bits, n, 0, packing)
        zero = torch.cat((zero, zero.new_zeros((padded_n - n, zero.shape[1]))))
        payload["weight_zero_point"] = pack(zero, e.bits, packing, axis=0)
    padded = QuantizedMatrix(payload, e, (padded_n, k))
    operators.materialized_weights[key] = matrix, padded
    return padded


def _align_k_tile(
    operators: PackedLinearOperators, params: Parameters, n: int, k: int, *, family: str,
    **scope: int,
) -> Parameters:
    if params.marlin_enabled and params.marlin_auto_geometry and k % params.marlin_block_k:
        # Reuse the legal Marlin K tile family. The alignment equation determines
        # the smaller K tile and the N width needed to close its load pipeline;
        # neither public C nor a model/device name enters this startup choice.
        with operator_strategy_scope(
            operators.config, device=operators.device_identity, model=operators.model_identity
        ):
            locked = parameter_overrides(family, n=n, k=k, **scope)
        bk = math.gcd(k, params.marlin_block_k)
        # K16 requires the single-subtile load contract: a full two-subtile
        # load cannot fit its N-warp ownership. Other Dense geometries retain
        # their existing full-load contract; routing already permits one tile.
        load_quantum = params.marlin_warps * 32 * (
            32 if isinstance(params, IntegerMoEParameters) or bk == 16 else 64
        )
        bn = math.lcm(params.marlin_block_n, load_quantum // math.gcd(bk, load_quantum))
        if (
            bk in (16, 32, 64, 128, 256)
            and bn in (64, 128, 256, 512)
            and not {"marlin_block_n", "marlin_block_k"}.intersection(locked)
        ):
            params = replace(params, marlin_block_n=bn, marlin_block_k=bk)
    return params


def bind_linear(
    operators: PackedLinearOperators,
    x: Tensor,
    matrix: QuantizedMatrix,
    output: Tensor,
    params: IntegerLinearParameters,
) -> PackedOperatorBinding | None:
    n, k = matrix.shape
    group = min(k, matrix.encoding.group_in)
    params = _align_k_tile(operators, params, n, k, family="int_linear")
    if (
        params.marlin_enabled
        and n % params.marlin_block_n
        and k % params.marlin_block_k == 0
        and k % group == 0
        and math.gcd(group, params.marlin_block_k) % 8 == 0
    ):
        padded_n = triton.cdiv(n, params.marlin_block_n) * params.marlin_block_n
        padded = _pad_output_rows(operators, matrix, padded_n)
        # The existing grouped epilogue owns physical versus logical columns.
        # Dense always writes fixed K partials, so fold directly into the real
        # output without a padded output allocation and a separate crop launch.
        binding = cast(
            PackedOperatorBinding | None,
            _bind(operators, x, ((padded,),), output, params, logical_n=n,
                  dense_outputs=(output,)),
        )
        return binding
    return cast(PackedOperatorBinding | None, _bind(operators, x, ((matrix,),), output, params))


def _bind_routed_projection(
    operators: PackedLinearOperators,
    x: Tensor,
    weights: tuple[QuantizedMatrix, ...],
    output: Tensor,
    rows: RoutedRows,
    params: IntegerMoEParameters,
    top_k: int = 1,
) -> tuple[int, PackedOperatorBinding] | None:
    n, k = weights[0].shape
    params = _align_k_tile(
        operators, params, n, k, family="int_moe",
        experts=len(weights), top_k=rows.routing_top_k or top_k,
    )
    padded_n = triton.cdiv(n, params.marlin_block_n) * params.marlin_block_n
    if padded_n == n:
        return cast(
            tuple[int, PackedOperatorBinding] | None,
            _bind(operators, x, (weights,), output, params, rows, top_k),
        )
    padded = tuple(_pad_output_rows(operators, w, padded_n) for w in weights)
    temporary = torch.empty((x.shape[0] * top_k, padded_n), device=x.device, dtype=output.dtype)
    candidate = cast(
        tuple[int, PackedOperatorBinding] | None,
        _bind(operators, x, (padded,), temporary, params, rows, top_k, logical_n=n),
    )
    if candidate is None:
        return None
    bm, binding = candidate
    copy = _PackedLaunch.compile(
        _unpad_output, (temporary, output), (output.numel(), n, padded_n, 256),
        (triton.cdiv(output.numel(), 256),), dict(num_warps=4),
    )
    return bm, PackedOperatorBinding(
        output, (*binding._launches, copy), scratch=(*binding.scratch, temporary)
    )


def bind_grouped(
    operators: PackedLinearOperators,
    x: Tensor,
    gate: tuple[QuantizedMatrix, ...],
    up: tuple[QuantizedMatrix, ...],
    output: Tensor,
    rows: RoutedRows,
    top_k: int,
    params: IntegerMoEParameters,
) -> tuple[int, PackedOperatorBinding] | None:
    if params.grouped_projections and gate[0].encoding == up[0].encoding:
        # Logical eligibility precedes execution-layout normalization. A TP
        # slice can be unaligned even when the two projections are compatible.
        # Align the full physical N, keeping gate/up adjacent. Only trailing
        # integer-zero rows are needed, not a padding gap between projections.
        n, k = gate[0].shape
        if up[0].shape != (n, k):
            raise ValueError("INT grouped gate/up requires matching projection shapes")
        grouped_params = _align_k_tile(
            operators, params, n, k, family="int_moe",
            experts=len(gate), top_k=rows.routing_top_k or top_k,
        )
        alignment = math.lcm(64, grouped_params.marlin_block_n)
        physical_n = triton.cdiv(2 * n, alignment) * alignment
        groups = (gate, up) if physical_n == 2 * n else (
            gate, tuple(_pad_output_rows(operators, w, physical_n - n) for w in up)
        )
        target = output if physical_n == 2 * n else torch.empty(
            (x.shape[0] * top_k, physical_n), device=x.device, dtype=output.dtype
        )
        candidate = cast(
            tuple[int, PackedOperatorBinding] | None,
            _bind(
                operators, x, groups, target, grouped_params, rows, top_k, logical_n=n
            ),
        )
        if candidate is None:
            raise ValueError(
                f"INT compatible gate/up group requires one Marlin launch: "
                f"shape={(n, k)}, physical_n={physical_n}, encoding={gate[0].encoding}, "
                f"parameters={grouped_params}"
            )
        bm, binding = candidate
        if physical_n == 2 * n:
            return bm, binding
        finish = _PackedLaunch.compile(
            _unpad_output, (target, output),
            (output.numel(), 2 * n, physical_n, 256),
            (triton.cdiv(output.numel(), 256),), dict(num_warps=4),
        )
        return bm, PackedOperatorBinding(
            output, (*binding._launches, finish), scratch=(*binding.scratch, target)
        )
    # Different projection encodings retain separate affine decoders and weight
    # storage. Both launches consume the same declared expert/route ownership.
    # Share the configured row tile so the caller's one route prefix is valid
    # for both projections; neither projection may silently use a Triton GEMM.
    split_params = replace(params, marlin_auto_geometry=False)
    launches: list[_PackedLaunch | NativeLaunch] = []
    scratch: list[Tensor] = []
    width = sum(group[0].shape[0] for group in (gate, up))
    offset = 0
    for group in (gate, up):
        n = group[0].shape[0]
        temporary = torch.empty((x.shape[0] * top_k, n), device=x.device, dtype=output.dtype)
        candidate = _bind_routed_projection(
            operators, x, group, temporary, rows, split_params, top_k
        )
        if candidate is None:
            raise ValueError(
                f"INT MoE requires Marlin: shape={group[0].shape}, "
                f"encoding={group[0].encoding}, parameters={split_params}"
            )
        bm, binding = candidate
        launches.extend(binding._launches)
        launches.append(_PackedLaunch.compile(
            _unpad_output, (temporary, output),
            (temporary.numel(), n, n, 256, width, offset),
            (triton.cdiv(temporary.numel(), 256),), dict(num_warps=4),
        ))
        scratch.extend((*binding.scratch, temporary))
        offset += n
    return bm, PackedOperatorBinding(output, tuple(launches), scratch=tuple(scratch))


def bind_routed(
    operators: PackedLinearOperators,
    x: Tensor,
    weights: tuple[QuantizedMatrix, ...],
    output: Tensor,
    rows: RoutedRows,
    params: IntegerMoEParameters,
) -> tuple[int, PackedOperatorBinding] | None:
    candidate = _bind_routed_projection(operators, x, weights, output, rows, params)
    if candidate is None:
        raise ValueError(
            f"INT MoE requires Marlin: shape={weights[0].shape}, "
            f"encoding={weights[0].encoding}, parameters={params}"
        )
    return candidate
