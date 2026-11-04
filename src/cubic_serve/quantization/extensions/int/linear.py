"""Marlin INT GEMMs and format-owned packed embedding lookups."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor

if TYPE_CHECKING:
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators

from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.quantization.common.kernels import _embedding
from cubic_serve.quantization.registry import select_compute_mode

from . import small_rows
from .checkpoint import fields
from .kernels import load_weight
from .policy import IntegerLinearParameters

row_contraction = small_rows


@triton.jit  # type: ignore[untyped-decorator]
def _split_a8_group(  # type: ignore[no-untyped-def]
    X, OUTS, M: tl.constexpr, WIDTHS: tl.constexpr, TOTAL_N: tl.constexpr, BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = index // TOTAL_N, index % TOTAL_N
    value = tl.load(X + index, index < M * TOTAL_N, 0)
    start = 0
    for member in tl.static_range(len(WIDTHS)):
        width = WIDTHS[member]
        owned = (row < M) & (column >= start) & (column < start + width)
        tl.store(OUTS[member] + row * width + column - start, value, owned)
        start += width


def arguments(
    operators: PackedLinearOperators, matrix: QuantizedMatrix
) -> tuple[tuple[Tensor, ...], int]:
    p = matrix.payload
    packing, symmetry, _ = fields(matrix.encoding)
    w = p["weight_packed"]
    return (w, p["weight_scale"], p.get("weight_zero_point", w), w, w, w, w, w, w), int(
        symmetry == "asym"
    ) + 2 * int(packing == "legacy")


def bind_native(
    operators: PackedLinearOperators,
    x: Tensor,
    weight: QuantizedMatrix,
    output: Tensor,
    params: IntegerLinearParameters,
    fp8_token_scales: Tensor | None = None,
) -> PackedOperatorBinding | None:
    if fp8_token_scales is not None:
        raise ValueError("INT packed weights do not accept FP8 token scales")
    encoded = operators.int8_input(x) if x.dtype == torch.int8 else None
    mode = select_compute_mode(
        weight.encoding, "linear", operators.config.compute_mode,
        m=x.shape[0], n=weight.shape[0], k=weight.shape[1],
        dtype=encoded.boundary_dtype if encoded is not None else x.dtype, device=x.device,
        activation_quantized=encoded is not None,
    )
    if mode == "wna8":
        from .marlin import bind_a8_dense_symmetric

        return bind_a8_dense_symmetric(operators, x, weight, output, params)
    if mode != "wna16":
        raise ValueError("INT linear requires WNA16 or selected WNA8")
    from .marlin import bind_linear

    return bind_linear(operators, x, weight, output, params)


def bind_linear(
    operators: PackedLinearOperators,
    x: Tensor,
    weight: QuantizedMatrix,
    output: Tensor,
    params: IntegerLinearParameters,
    fp8_token_scales: Tensor | None = None,
) -> PackedOperatorBinding:
    binding = bind_native(operators, x, weight, output, params, fp8_token_scales)
    if binding is not None:
        return binding
    raise ValueError(
        "INT linear requires Marlin: "
        f"shape={weight.shape}, bits={weight.encoding.bits}, "
        f"group={weight.encoding.group_in}, dtype={x.dtype}, parameters={params}"
    )


def bind_grouped(
    operators: PackedLinearOperators,
    x: Tensor,
    weights: tuple[QuantizedMatrix, ...],
    outputs: tuple[Tensor, ...],
    params: IntegerLinearParameters,
) -> PackedOperatorBinding | None:
    """One INT Marlin and one fixed-order merge into independent projections."""
    encoded = operators.int8_input(x) if x.dtype == torch.int8 else None
    mode = select_compute_mode(
        weights[0].encoding, "linear", operators.config.compute_mode,
        m=x.shape[0], n=weights[0].shape[0], k=weights[0].shape[1],
        dtype=encoded.boundary_dtype if encoded is not None else x.dtype, device=x.device,
        activation_quantized=encoded is not None,
    )
    import math

    from cubic_serve.ops.strategies import operator_strategy_scope, parameter_overrides

    from .marlin import _align_k_tile, _bind, _pad_output_rows

    k = weights[0].shape[1]
    ns = tuple(w.shape[0] for w in weights)
    if any(w.shape[1] != k or w.encoding != weights[0].encoding for w in weights):
        return None
    if any(select_compute_mode(
        w.encoding, "linear", operators.config.compute_mode,
        m=x.shape[0], n=w.shape[0], k=k,
        dtype=encoded.boundary_dtype if encoded is not None else x.dtype, device=x.device,
        activation_quantized=encoded is not None,
    ) != mode for w in weights[1:]):
        return None
    # A user-specified member plan remains authoritative. A merged N must not
    # hide its scope or silently choose a different K reduction contract.
    with operator_strategy_scope(
        operators.config, device=operators.device_identity, model=operators.model_identity
    ):
        if any(parameter_overrides("int_linear", n=n, k=k) for n in ns):
            return None
    if mode == "wna8":
        if any(y.dtype != outputs[0].dtype for y in outputs):
            return None
        from .marlin import bind_a8_dense_symmetric, combine_a8_matrices

        combined = combine_a8_matrices(operators, weights)
        temporary = torch.empty(
            (x.shape[0], sum(ns)), device=x.device, dtype=outputs[0].dtype
        )
        binding = bind_a8_dense_symmetric(
            operators, x, combined, temporary, params,
            grouped_outputs=outputs,
        )
        if binding.output is outputs[0]:
            return binding
        split = _PackedLaunch.compile(
            _split_a8_group, (temporary, outputs),
            (x.shape[0], ns, sum(ns), 256),
            (triton.cdiv(temporary.numel(), 256),),
            dict(num_warps=4, enable_fp_fusion=False),
        )
        return PackedOperatorBinding(
            outputs[0], (*binding._launches, split), scratch=(*binding.scratch, temporary),
            reusable_scratch=binding.reusable_scratch,
            reusable_scratch_indices=(*binding.reusable_scratch_indices, len(binding.scratch)),
        )
    params = _align_k_tile(operators, params, sum(ns), k, family="int_linear")
    alignment = math.lcm(64, params.marlin_block_n)
    # Same-encoding projections form one physical output axis. The existing
    # merge retains the individual logical widths, including unaligned boundaries.
    tail = triton.cdiv(sum(ns), alignment) * alignment - sum(ns)
    physical = (*ns[:-1], ns[-1] + tail)
    padded = tuple(
        w if n == w.shape[0] else _pad_output_rows(operators, w, n)
        for w, n in zip(weights, physical, strict=True)
    )
    # The fixed-partition native epilogue writes FP32 scratch; final outputs
    # are written directly by the existing merge, with their original widths.
    binding = cast(PackedOperatorBinding | None, _bind(
        operators, x, tuple((w,) for w in padded), outputs[0], params,
        logical_n=sum(ns), dense_outputs=outputs,
    ))
    if binding is None:
        raise ValueError(f"INT grouped linear requires Marlin: shapes={[w.shape for w in weights]}")
    # Dense fixed partitions write only FP32 partials; their merge already
    # targets the individual outputs. Keep the native binding and its scratch
    # opt-in, without an unused physical-output allocation or ownership wrapper.
    if not binding.reusable_scratch or binding.output is not outputs[0]:
        raise ValueError("grouped INT projection requires a direct fixed-partition merge")
    return binding


def bind_embedding(
    operators: PackedLinearOperators,
    input_ids: Tensor,
    weight: QuantizedMatrix,
    output: Tensor,
    params: IntegerLinearParameters,
) -> PackedOperatorBinding:
    args, fmt = arguments(operators, weight)
    e = weight.encoding
    n, k = weight.shape
    call = _PackedLaunch.compile(
        _embedding,
        (input_ids, *args, output),
        (
            n,
            k,
            load_weight,
            fmt,
            e.bits,
            min(k, e.group_in),
            1,
            args[0].shape[1],
            params.embedding_block_k,
        ),
        (input_ids.numel(), triton.cdiv(k, params.embedding_block_k)),
        dict(num_warps=params.fixed_warps, enable_fp_fusion=False),
    )
    return PackedOperatorBinding(output, (call,))


@triton.jit  # type: ignore[untyped-decorator]
def _sharded_embedding(  # type: ignore[no-untyped-def]
    IDS,
    ACTIVE,
    STARTS,
    ENDS,
    WP,
    SP,
    ZP,
    SCALE,
    OUT,
    HEADS: tl.constexpr,
    K: tl.constexpr,
    COUNT: tl.constexpr,
    FORMAT: tl.constexpr,
    BITS: tl.constexpr,
    GI: tl.constexpr,
    STRIDE: tl.constexpr,
    ST: tl.constexpr,
    SB: tl.constexpr,
    BK: tl.constexpr,
):
    lookup = tl.program_id(0)
    if tl.load(ACTIVE + lookup // HEADS) != 0:
        token = tl.load(IDS + lookup)
        shard = tl.arange(0, SB)
        start = tl.load(STARTS + shard, shard < COUNT, 0)
        end = tl.load(ENDS + shard, shard < COUNT, 0)
        owns = (shard < COUNT) & (token >= start) & (token < end)
        owner = tl.max(tl.where(owns, shard, -1), 0)
        col = tl.program_id(1) * BK + tl.arange(0, BK)
        value = tl.full((BK,), 0, tl.float32)
        if owner >= 0:
            begin = tl.sum(tl.where(owns, start, 0), 0)
            n = tl.sum(tl.where(owns, end - start, 0), 0)
            w = tl.load(WP + owner).to(tl.pointer_type(tl.int32))
            s = tl.load(SP + owner).to(tl.pointer_type(ST))
            z = tl.load(ZP + owner).to(tl.pointer_type(tl.int32))
            value = load_weight(
                w, s, z, w, w, w, w, w, w, token - begin, col, n, K, FORMAT, BITS, GI, 1, STRIDE
            )
        value = value * tl.load(SCALE).to(tl.float32)
        tl.store(OUT + lookup * K + col, value, col < K)


def bind_scaled_sharded_embedding(
    ids: Tensor,
    shards: tuple[tuple[int, Tensor | QuantizedMatrix], ...],
    scale: Tensor,
    row_activity: Tensor,
    output: Tensor,
) -> PackedOperatorBinding:
    """One packed lookup over immutable original row owners; no full-table decode."""
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.embedding.immutable import embedding_pointer

    first = shards[0][1]
    if not isinstance(first, QuantizedMatrix):
        raise ValueError("INT embedding requires packed matrices")
    enc = first.encoding
    k = first.shape[1]
    payloads: list[Tensor] = []
    starts, ends, wp, sp, zp = [], [], [], [], []
    previous_end = 0
    stride = first.payload["weight_packed"].shape[1]
    scale_dtype = first.payload["weight_scale"].dtype
    for start, weight in shards:
        if (
            not isinstance(weight, QuantizedMatrix)
            or weight.encoding != enc
            or weight.shape[1] != k
            or type(start) is not int
            or start < previous_end
            or start + weight.shape[0] >= 2**63
        ):
            raise ValueError("INT embedding requires sorted disjoint homogeneous shard ranges")
        p = weight.payload
        w, s = p["weight_packed"], p["weight_scale"]
        z = p.get("weight_zero_point", w)
        if w.shape[1] != stride or s.dtype != scale_dtype:
            raise ValueError("INT embedding shard storage differs")
        payloads.extend((w, s, z))
        starts.append(start)
        previous_end = start + weight.shape[0]
        ends.append(previous_end)
        wp.append(embedding_pointer(w, ids.device))
        sp.append(embedding_pointer(s, ids.device))
        zp.append(embedding_pointer(z, ids.device))
    operands = (ids, scale, row_activity, output, *payloads)
    if (
        ids.device.type != "cuda"
        or ids.dtype != torch.int64
        or ids.ndim != 2
        or min(ids.shape) <= 0
        or row_activity.dtype != torch.int32
        or row_activity.shape != (ids.shape[0],)
        or scale.numel() != 1
        or scale.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or output.dtype not in (torch.float16, torch.bfloat16)
        or output.shape != (ids.shape[0], ids.shape[1] * k)
        or any(t.device != ids.device or not t.is_contiguous() for t in operands[:4])
        or any(not t.is_contiguous() for t in payloads)
    ):
        raise ValueError("INT embedding operand geometry/device/storage differs")
    if not 0 < float(scale.item()) < float("inf"):
        raise ValueError("INT embedding direct scale must be finite and positive")
    if any(
        t.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
        for t in (ids, scale, row_activity, *payloads)
    ):
        raise ValueError("INT embedding output may not alias its inputs")
    metadata = tuple(
        torch.tensor(v, dtype=torch.int64, device=ids.device) for v in (starts, ends, wp, sp, zp)
    )
    packing, symmetry, _ = fields(enc)
    fmt = int(symmetry == "asym") + 2 * int(packing == "legacy")
    st = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}[
        scale_dtype
    ]
    call = _PackedLaunch.compile(
        _sharded_embedding,
        (ids, row_activity, *metadata, scale, output),
        (
            ids.shape[1],
            k,
            len(shards),
            fmt,
            enc.bits,
            min(k, enc.group_in),
            stride,
            st,
            triton.next_power_of_2(len(shards)),
            256,
        ),
        (ids.numel(), triton.cdiv(k, 256)),
        dict(num_warps=4, enable_fp_fusion=False),
    )
    return PackedOperatorBinding(output, (call,), scratch=(*payloads, *metadata))
