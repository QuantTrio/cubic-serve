"""Format-owned W4A16 grouped projection binding derived from Marlin.

Original checkpoint scale dtype and whole-K reduction ownership are invariant.
Only rank-local row metadata is adapted; expert ownership never changes.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from cubic_serve.ops.cuda.compiled_artifact import bind_tensor
from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.quantization.api import RoutedRows
from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.quantization.common.marlin.compiler import bind_launch, compile_module
from cubic_serve.quantization.common.marlin.cost import ProbeFormat
from cubic_serve.quantization.common.marlin.cost import select as select_native_geometry
from cubic_serve.quantization.common.marlin.parameters import MarlinParameters, shared_bytes
from cubic_serve.quantization.common.marlin.repack import _pad_routes, repack_words

from .policy import AwqMoEParameters

if TYPE_CHECKING:
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators


def _packed_group(
    operators: PackedLinearOperators,
    groups: tuple[tuple[QuantizedMatrix, ...], ...],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Checkpoint-lifetime cache of integer codes and original FP16 scales."""
    matrices = tuple(w for group in groups for w in group)
    identity = tuple(
        bind_tensor(w.payload[name]) for w in matrices for name in ("qweight", "scales", "qzeros")
    )
    keys = tuple(
        ("awq-marlin-group-v1", device, identity, name) for name in ("codes", "scales", "zeros")
    )
    cache = operators.prepared_weights
    if all(key in cache for key in keys):
        return tuple(cache[key][1] for key in keys)  # type: ignore[return-value]
    n = sum(group[0].shape[0] for group in groups)
    k = groups[0][0].shape[1]
    group = groups[0][0].encoding.group_in
    source_device = groups[0][0].payload["qweight"].device
    permutation = [a + 8 * b for a in range(8) for b in range(8)]
    undo = [0, 4, 1, 5, 2, 6, 3, 7]
    interleave = [0, 2, 4, 6, 1, 3, 5, 7]
    shifts = torch.arange(8, device=source_device) * 4
    codes, scales, zeros = [], [], []
    for members in zip(*groups, strict=True):
        source = torch.cat(tuple(w.payload["qweight"] for w in members), dim=1)
        packed = repack_words(source.to(device), bits=4)
        codes.append(packed)
        scales.append(
            torch.cat(tuple(w.payload["scales"] for w in members), dim=1)
            .reshape(-1, 64)[:, permutation]
            .reshape(k // group, n)
        )
        z = torch.cat(tuple(w.payload["qzeros"] for w in members), dim=1)
        raw = ((z[:, :, None] >> shifts) & 15).reshape(-1, 8)[:, undo]
        raw = raw.reshape(-1, 64)[:, permutation].reshape(-1, 8)[:, interleave]
        zeros.append((raw << shifts).sum(dim=1).int().reshape(k // group, n // 8))
    values = tuple(torch.stack(v).to(device) for v in (codes, scales, zeros))
    for key, value in zip(keys, values, strict=True):
        cache[key] = (matrices, value)
    return values  # type: ignore[return-value]


def _bind(
    operators: PackedLinearOperators,
    x: Tensor,
    groups: tuple[tuple[QuantizedMatrix, ...], ...],
    output: Tensor,
    rows: RoutedRows,
    top_k: int,
    params: AwqMoEParameters,
) -> tuple[int, PackedOperatorBinding] | None:
    """Return a startup-native candidate only for its declared weight/layout ABI."""
    if not params.marlin_enabled:
        return None
    weights = tuple(w for group in groups for w in group)
    encoding = groups[0][0].encoding
    group = encoding.group_in
    n = sum(group[0].shape[0] for group in groups)
    k = groups[0][0].shape[1]
    if (
        x.dtype not in (torch.bfloat16, torch.float16)
        or encoding.bits != 4
        or group < 16
        or group % 16
        or k % group
        or n % 64
        or k % 64
        or any(
            w.encoding != encoding
            or w.shape[1] != k or w.shape[0] % 64
            or w.partitions
            or w.payload["scales"].dtype != torch.float16
            for w in weights
        )
    ):
        return None
    bm, bn, bk = params.marlin_block_m, params.marlin_block_n, params.marlin_block_k
    threads, stages = params.marlin_warps * 32, params.marlin_stages
    if (
        bm not in (16, 32, 64)
        or bn < 64
        or bk < 64
        or stages < 2
        or bk * bn < threads * 32
        or (group < bk and bk % group)
    ):
        raise ValueError("AWQ Marlin tile/group/pipeline geometry is incompatible")
    if n % bn or k % bk:
        return None
    params = select_native_geometry(
        params, operators=operators, family="awq_moe",
        scope=dict(n=groups[0][0].shape[0], k=k, experts=len(groups[0]),
                   top_k=rows.routing_top_k or top_k),
        source=lambda p: _kernel_source(
            p, "kBFloat16" if x.dtype == torch.bfloat16 else "kFloat16", group),
        spec=ProbeFormat(4, x.dtype, x.dtype, torch.float16, 0x88888888 - 2**32,
                         0.03125, 0.03125, zero_word=0x77777777),
        device=x.device, n=n, k=k, experts=len(groups[0]), group=group,
        rows=(x.shape[0] * top_k / len(groups[0])
              if rows.mean_rows_per_expert is None else rows.mean_rows_per_expert),
        routed=True,
    )
    bm, bn, bk = params.marlin_block_m, params.marlin_block_n, params.marlin_block_k
    threads, stages = params.marlin_warps * 32, params.marlin_stages
    shared = shared_bytes(params, bits=4, group=group)
    resource = torch.cuda.get_device_properties(x.device)
    if shared > int(
        getattr(resource, "shared_memory_per_block_optin", resource.shared_memory_per_block)
    ):
        return None
    dtype = "kBFloat16" if x.dtype == torch.bfloat16 else "kFloat16"
    source = _kernel_source(params, dtype, group)
    module, entry = compile_module(source, x.device)
    w, s, z = _packed_group(operators, groups, x.device)
    m, experts = x.shape[0], len(groups[0])
    route_count = m * top_k
    blocks = math.ceil((route_count + min(route_count, experts) * (bm - 1)) / bm)
    ids = torch.empty(blocks * bm, dtype=torch.int32, device=x.device)
    owners = torch.empty(blocks, dtype=torch.int32, device=x.device)
    total = torch.empty(1, dtype=torch.int32, device=x.device)
    # Whole-K ownership needs no locks or cross-block reduction workspace.
    router_weights = torch.ones(m, top_k, dtype=torch.float32, device=x.device)
    prepare = _PackedLaunch.compile(
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
        (route_count, bm),
        (experts,),
        {},
    )
    arguments = (
        x,
        w,
        output,
        None,
        None,
        None,
        s,
        None,
        z,
        None,
        ids,
        owners,
        total,
        router_weights,
        CudaScalar("i32", top_k),
        CudaScalar("bool", False),
        CudaScalar("i32", k // group),
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
        arguments,
        operator_id="cubic.cuda.awq.marlin.grouped.v1",
        params=params,
        output_tiles=owners.numel() * (n // bn),
        shared_bytes=shared,
    )
    return bm, PackedOperatorBinding(
        output, (prepare, multiply), scratch=(ids, owners, total, router_weights)
    )


def bind_grouped(
    operators: PackedLinearOperators, x: Tensor, gate: tuple[QuantizedMatrix, ...],
    up: tuple[QuantizedMatrix, ...], output: Tensor, rows: RoutedRows, top_k: int,
    params: AwqMoEParameters,
) -> tuple[int, PackedOperatorBinding] | None:
    if not params.grouped_projections:
        return None
    return _bind(operators, x, (gate, up), output, rows, top_k, params)


def bind_routed(
    operators: PackedLinearOperators, x: Tensor, weights: tuple[QuantizedMatrix, ...],
    output: Tensor, rows: RoutedRows, params: AwqMoEParameters,
) -> tuple[int, PackedOperatorBinding] | None:
    return _bind(operators, x, (weights,), output, rows, 1, params)


def _kernel_source(params: MarlinParameters, dtype: str, group: int) -> str:
    bm, bn, bk = params.marlin_block_m, params.marlin_block_n, params.marlin_block_k
    threads, stages = params.marlin_warps * 32, params.marlin_stages
    return f"""#define MARLIN_NAMESPACE_NAME cubic_awq
    #include "libtorch_stable/moe/marlin_moe_wna16/kernel.h"
    #include "marlin_template.h"
    namespace cubic_awq {{
    template __global__ void Marlin<vllm::{dtype}.id(),vllm::kU4.id(),
    vllm::{dtype}.id(),vllm::kFloat16.id(),{threads},{bm // 16},{bn // 16},{bk // 16},
    false,{stages},{group // 16},false>(MARLIN_KERNEL_PARAMS);
    }}"""
