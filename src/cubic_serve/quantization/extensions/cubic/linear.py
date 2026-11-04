from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.quantization.common.binding import PackedOperatorBinding
    from cubic_serve.quantization.common.parameters import PackedLinearParameters

import torch
import triton  # type: ignore[import-untyped]

from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.quantization.common.kernels import (
    _embedding,
    _merge_split_k,
    _weight_only_mm,
    _weight_only_mm_reuse_rows,
)

from . import small_rows
from .kernels import load_level_words, load_levels, load_weight
from .policy import CubicLinearParameters

row_contraction = small_rows


def _levels(
    operators: PackedLinearOperators,
    weight: QuantizedMatrix,
    args: tuple[Tensor, ...],
    dtype: torch.dtype,
) -> Tensor:
    """Prepare group values at cold binding, preserving the decoder's rounds.

    Only the small metadata/codebook is expanded, never an N-by-K weight.
    The cache retains its source matrix, preventing recycled-identity hits,
    and is shared by graph buckets through their resident weight owner.
    """
    versions: list[int | None] = []
    for tensor in weight.payload.values():
        try:
            versions.append(tensor._version)
        except RuntimeError:  # immutable inference-mode checkpoint tensors
            versions.append(None)
    key = ("cubic.levels.v1", id(weight), dtype, tuple(versions))
    cached = operators.prepared_weights.get(key)
    if cached is not None:
        return cached[1]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Cubic levels require cold preparation before capture")
    _, scale, _, ab, _, gs, ga, gb, parts = args
    ab, parts = ab.cpu().int(), parts.cpu().long()
    gs, ga, gb = gs.cpu(), ga.cpu(), gb.cpu()
    ac, bc = ab & 15, ab >> 4
    ac, bc = torch.where(ac >= 8, ac - 16, ac), torch.where(bc >= 8, bc - 16, bc)
    a = (1.0 + ac.float() * ga[parts, None]).half().float().unsqueeze(-1)
    b = (bc.float() * gb[parts, None]).half().float().unsqueeze(-1)
    scale = (scale.cpu().float() * gs[parts, None]).unsqueeze(-1)
    count = 1 << (weight.encoding.bits - 1)
    t = (torch.arange(count, dtype=torch.float64) / (count - 1)).float()
    levels = (t * (a + t * (b + t * ((1 - a) - b))) * scale).to(dtype)
    levels = levels.contiguous().to(args[0].device)
    operators.prepared_weights[key] = weight, levels
    return levels


def arguments(
    operators: PackedLinearOperators, matrix: QuantizedMatrix
) -> tuple[tuple[Tensor, ...], int]:
    p, e = matrix.payload, matrix.encoding
    dummy = next(iter(p.values()))
    if dummy.dtype is torch.float8_e4m3fn:
        dummy = dummy.view(torch.uint8)
    if id(matrix) not in operators._partitions:
        counts = matrix.partitions or (matrix.shape[0],)
        operators._partitions[id(matrix)] = torch.tensor(
            [i for i, count in enumerate(counts) for _ in range(count // e.group_out)],
            dtype=torch.int32,
            device=dummy.device,
        )
    parts = operators._partitions[id(matrix)]
    if e.metadata == "int8-scale-int4-ab-global-fp32":
        return (
            p["weight_packed"],
            p["weight_scale"],
            dummy,
            p["weight_ab"],
            dummy,
            p["weight_scale_global"],
            p["weight_a_global"],
            p["weight_b_global"],
            parts,
        ), 1
    if e.metadata == "e5m9-scale-curve2-tensor-fp16":
        return (
            p["weight_packed"],
            p["weight_metadata"],
            dummy,
            p["weight_curve_a"],
            p["weight_curve_b"],
            dummy,
            dummy,
            dummy,
            parts,
        ), 2
    return (
        p["weight_packed"],
        p["weight_scale"],
        dummy,
        p["weight_a"],
        p["weight_b"],
        dummy,
        dummy,
        dummy,
        parts,
    ), 3


def bind_native(operators, x, weight, output, params, fp8_token_scales=None):
    """Format-local admission; generic dispatch never names Cubic or Marlin."""
    if operators.config.compute_mode == "wna8":
        raise ValueError("Cubic WNA8 native linear kernel is unavailable")
    if not isinstance(params, CubicLinearParameters) or not params.marlin_enabled:
        return None
    from .marlin import bind_planned_linear

    return bind_planned_linear(operators, x, weight, output, params)


def bind_linear(
    operators: PackedLinearOperators,
    x: Tensor,
    weight: QuantizedMatrix,
    output: Tensor,
    params: PackedLinearParameters,
    fp8_token_scales: Tensor | None = None,
) -> PackedOperatorBinding:
    candidate = bind_native(operators, x, weight, output, params, fp8_token_scales)
    if candidate is not None:
        return candidate
    e = weight.encoding
    n, k = weight.shape
    m = x.shape[0]
    bm, bn, bk = params.fixed_block_m, params.fixed_block_n, params.fixed_block_k
    warps = params.fixed_warps
    split_k = min(
        params.fixed_split_k if isinstance(params, CubicLinearParameters) else 1,
        triton.cdiv(k, bk),
    )
    device = torch.cuda.get_device_properties(x.device)
    candidates = (
        params.output_tiles(m, n, k, split_k, device.multi_processor_count)
        if isinstance(params, CubicLinearParameters)
        else ((1.0, bm, bn, warps),)
    )
    compile_launch = _PackedLaunch.compile
    scratch: tuple[Tensor, ...] = ()
    calls: tuple[_PackedLaunch, ...]
    args, fmt = arguments(operators, weight)
    loader = load_weight
    decode_mode = (
        params.decode_mode(
            bits=e.bits,
            metadata=e.metadata,
            n=n,
            k=k,
            group_in=e.group_in,
            group_out=e.group_out,
            element_bytes=x.element_size(),
        )
        if isinstance(params, CubicLinearParameters)
        else "direct"
    )
    if decode_mode != "direct":
        levels = _levels(operators, weight, args, x.dtype)
        args = (args[0], levels, *args[2:])
        loader = load_level_words if decode_mode == "words" else load_levels
    partials = (
        torch.empty((split_k, m, n), dtype=torch.float32, device=x.device)
        if split_k > 1
        else output
    )
    choices = []
    for cost, bm, bn, warps in candidates:
        # Every Cubic decoder is a pure weight read with the same K reduction
        # contract. Sharing its result across rows is not a W4 capability;
        # compiled resource costs below govern whether the reuse is worthwhile.
        reuse = (
            isinstance(params, CubicLinearParameters)
            and params.row_reuse_max_tiles == 2
            and bm == params.fixed_max_block_m
            and bm > params.fixed_block_m
            and m > bm
        )
        for row_tiles in (1, 2) if reuse else (1,):
            constants: tuple[object, ...] = (
                m,
                n,
                k,
                loader,
                fmt,
                e.bits,
                e.group_in,
                e.group_out,
                triton.cdiv(k * e.bits, 8),
                bm,
                bn,
                bk,
                split_k,
                True,
            )
            kernel = _weight_only_mm
            if row_tiles > 1:
                kernel = _weight_only_mm_reuse_rows
                constants = (*constants, row_tiles)
            call = compile_launch(
                kernel,
                (x, *args, partials),
                constants,
                (triton.cdiv(m, bm * row_tiles), triton.cdiv(n, bn), split_k),
                dict(num_warps=warps, num_stages=params.fixed_stages, enable_fp_fusion=False),
            )
            # Compilation supplies actual register/shared/spill costs without
            # running autotuning or changing the K reduction plan across M.
            threads = warps * device.warp_size
            resident = max(
                1,
                min(
                    device.max_threads_per_multi_processor // threads,
                    device.regs_per_multiprocessor // max(1, call.kernel.n_regs * threads),  # type: ignore[attr-defined]
                    device.shared_memory_per_multiprocessor // max(1, call.kernel.metadata.shared),
                ),
            )
            tile_cost = (
                params.tile_cost(m, n, k, split_k, device.multi_processor_count, bm * row_tiles, bn)
                if isinstance(params, CubicLinearParameters)
                else cost
            )
            score = tile_cost * (1 + 1 / resident) * (1 + call.kernel.n_spills)
            choices.append((score, call))
    calls = (min(choices, key=lambda choice: choice[0])[1],)
    if split_k > 1:
        scratch = (partials,)
        calls += (
            compile_launch(
                _merge_split_k,
                (partials, output),
                (m * n, split_k, 256),
                (triton.cdiv(m * n, 256),),
                {"num_warps": 4, "enable_fp_fusion": False},
            ),
        )
    return PackedOperatorBinding(
        output,
        calls,
        scratch=scratch,
        reusable_scratch=bool(scratch),
        row_tile_count=calls[0].grid[0],
    )


def bind_embedding(
    operators: PackedLinearOperators,
    input_ids: Tensor,
    weight: QuantizedMatrix,
    output: Tensor,
    params: PackedLinearParameters,
) -> PackedOperatorBinding:
    args, fmt = arguments(operators, weight)
    e = weight.encoding
    n, k = weight.shape
    block = params.embedding_block_k
    call = _PackedLaunch.compile(
        _embedding,
        (input_ids, *args, output),
        (
            n,
            k,
            load_weight,
            fmt,
            e.bits,
            e.group_in,
            e.group_out,
            triton.cdiv(k * e.bits, 8),
            block,
        ),
        (input_ids.numel(), triton.cdiv(k, block)),
        {"num_warps": params.fixed_warps, "enable_fp_fusion": False},
    )
    return PackedOperatorBinding(output, (call,))


def bind_grouped(operators, x, weights, outputs, params):
    """Share same-input projections through the format's native compute mode."""
    if operators.config.compute_mode == "wna8":
        raise ValueError("Cubic WNA8 native grouped linear kernel is unavailable")
    if not isinstance(params, CubicLinearParameters) or not params.marlin_enabled:
        return None
    from .marlin import bind_planned_grouped

    return bind_planned_grouped(operators, x, weights, outputs, params)
