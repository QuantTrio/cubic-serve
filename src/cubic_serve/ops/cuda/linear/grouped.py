"""Group independent BF16 projections sharing one input and reduction plan."""

from __future__ import annotations

from dataclasses import replace

import torch
import triton
import triton.language as tl
from torch import Tensor

from cubic_serve.ops.cuda.compiled_artifact import (
    CompiledArtifactIdentity,
    LaunchReceipt,
    bind_tensor,
    project_triton_artifact,
    validate_launch_receipt,
)
from cubic_serve.ops.cuda.launch_cost import (
    LinearLaunchCost,
    LinearTile,
    MatmulGeometry,
    current_linear_cost,
    device_resources,
    device_work_cost,
    linear_geometries,
)
from cubic_serve.ops.cuda.linear.mmvf import _bf16_fixed_tile_at, _validate_layout
from cubic_serve.ops.strategies import (
    LinearParameters,
    launch_parameters,
    linear_parameters,
    linear_row_tile,
    parameter_overrides,
)
from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.quantization.common.marlin.cost import fixed_k_partitions


@triton.jit
def _grouped_partition_mm(
    x, weights, partials, active_rows, NS: tl.constexpr, K: tl.constexpr,
    M: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    PARTITIONS: tl.constexpr,
):
    """Independent projections, contiguous fixed K segments and FP32 partials."""
    tile = tl.program_id(0)
    first_tile = 0
    first_element = 0
    rows = (tl.program_id(1) * BM + tl.arange(0, BM)).to(tl.int64)
    active = tl.load(active_rows)
    valid = (active > 0) & (active <= M)
    offsets = tl.arange(0, BK)
    partition = tl.program_id(2).to(tl.int64)
    for group in tl.static_range(len(NS)):
        n = NS[group]
        if (tile >= first_tile) & (tile < first_tile + tl.cdiv(n, BN)):
            columns = ((tile - first_tile) * BN + tl.arange(0, BN)).to(tl.int64)
            accumulator = tl.zeros((BM, BN), tl.float32)
            for step in range(K // (BK * PARTITIONS)):
                kk = partition * (K // PARTITIONS) + step * BK + offsets
                left = tl.load(x + rows[:, None] * K + kk[None, :],
                               (rows[:, None] < active) & valid, 0)
                right = tl.load(weights[group] + columns[None, :] * K + kk[:, None],
                                columns[None, :] < n, 0)
                accumulator = tl.dot(left, right, accumulator)
            tl.store(partials + first_element + partition * M * n
                     + rows[:, None] * n + columns[None, :], accumulator,
                     (rows[:, None] < M) & (columns[None, :] < n) & valid)
        first_tile += tl.cdiv(n, BN)
        first_element += PARTITIONS * M * n


@triton.jit
def _grouped_partition_merge(
    partials, outputs, active_rows, NS: tl.constexpr, M: tl.constexpr,
    PARTITIONS: tl.constexpr, BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    first_tile = 0
    first_element = 0
    active = tl.load(active_rows).to(tl.int64)
    valid = (active > 0) & (active <= M)
    for group in tl.static_range(len(NS)):
        n = NS[group]
        if (tile >= first_tile) & (tile < first_tile + tl.cdiv(M * n, BLOCK)):
            offsets = ((tile - first_tile) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
            total = tl.full((BLOCK,), 0, tl.float32)
            for partition in tl.static_range(PARTITIONS):
                total += tl.load(partials + first_element + partition * M * n + offsets,
                                 (offsets < M * n) & valid, 0)
            tl.store(outputs[group] + offsets, tl.where(offsets < active * n, total, 0),
                     (offsets < M * n) & valid)
        first_tile += tl.cdiv(M * n, BLOCK)
        first_element += PARTITIONS * M * n


def bind_partitioned_linear_group(
    x: Tensor, weights: tuple[Tensor, ...], outputs: tuple[Tensor, ...], active_rows: Tensor,
) -> PackedOperatorBinding | None:
    """Reuse fixed-K supply and measured placement without changing cross-M math."""
    if len(weights) < 2 or len(weights) != len(outputs):
        raise ValueError("grouped linear requires matching independent projections")
    m, k = x.shape
    ns = tuple(weight.shape[0] for weight in weights)
    plans = tuple(linear_parameters(n=n, k=k) for n in ns)
    p = plans[0]
    if not all(plan.grouped_projections and plan == p for plan in plans):
        return None
    if any(key.startswith("fixed_") for n in ns
           for key in parameter_overrides("linear", n=n, k=k)):
        return None
    cost = current_linear_cost() or LinearLaunchCost(device_resources(x.device), 0.0, ())
    columns = sum(triton.cdiv(n, p.fixed_block_n) for n in ns)
    if (k < p.grouped_vector_min_k or k % p.fixed_block_k or columns >= cost.device.sms
            or 100 * sum(ns) <= p.grouped_vector_tile_fill_percent * columns * p.fixed_block_n):
        return None
    partitions = fixed_k_partitions(
        n=columns * p.fixed_block_n, k=k, group=p.fixed_block_k,
        block_n=p.fixed_block_n, block_k=p.fixed_block_k, sms=cost.device.sms,
        supply_factor=p.fixed_cta_supply_factor, max_partitions=p.fixed_max_k_partitions,
    )
    if partitions == 1:
        return None
    for weight, output in zip(weights, outputs, strict=True):
        _validate_layout(x, weight, output, active_rows, inspect_active=True,
                         allowed_dtypes=(torch.bfloat16, torch.float16))
    operands = (x, *weights, *outputs, active_rows)
    if len({t.data_ptr() for t in operands}) != len(operands):
        raise ValueError("grouped linear operands must not alias")
    partials = torch.empty((partitions * m * sum(ns),), device=x.device, dtype=torch.float32)
    baseline = MatmulGeometry(linear_row_tile(m, p.fixed_block_m, p.fixed_max_block_m),
                              p.fixed_block_n, p.fixed_block_k, p.fixed_warps, p.fixed_stages)
    points = linear_geometries(
        cost.device, rows=m, columns=ns, reduction=k, element_bytes=x.element_size(),
        baseline=baseline, parameters=p, partitions=partitions,
        communication=cost.communication, work_cost=device_work_cost(x.device, x.dtype),
    )
    calls: list[_PackedLaunch] = []
    tiles: list[LinearTile] = []
    for point in points:
        # Compiler spills justify a neighboring permitted warp count at the
        # same placement; no device generation or model identity is involved.
        while True:
            grid = (sum(triton.cdiv(n, point.block_n) for n in ns),
                    triton.cdiv(m, point.block_m), partitions)
            call = _PackedLaunch.compile(
                _grouped_partition_mm, (x, weights, partials, active_rows),
                (ns, k, m, point.block_m, point.block_n, point.block_k, partitions),
                grid, dict(num_warps=point.warps, num_stages=point.stages, enable_fp_fusion=False),
            )
            if not call.kernel.n_spills or point.warps * 2 > launch_parameters().max_warps:
                break
            point = replace(point, warps=point.warps * 2)
        if call.kernel.n_spills:
            continue
        calls.append(call)
        tiles.append(LinearTile(call.kernel, grid, point.block_m, point.block_n, point))
    if not tiles:
        raise RuntimeError("no spill-free fixed-partition grouped linear plan")
    selected = cost.select(
        tiles, x, weights, grouped=True, partitions=partitions,
        replays=launch_parameters().micro_calibration_replays,
    )
    call = calls[next(i for i, tile in enumerate(tiles) if tile is selected)]
    merge = _PackedLaunch.compile(
        _grouped_partition_merge, (partials, outputs, active_rows), (ns, m, partitions, 256),
        (sum(triton.cdiv(m * n, 256) for n in ns),),
        dict(num_warps=4, enable_fp_fusion=False),
    )
    return PackedOperatorBinding(
        outputs[0], (call, merge), scratch=(partials,), reusable_scratch=True,
        row_tile_count=selected.grid[1],
    )


def bind_grouped_linear(
    x: Tensor, weights: tuple[Tensor, ...], outputs: tuple[Tensor, ...], active_rows: Tensor
) -> BF16GroupedLinearPrimitive | None:
    """Group only compatible member plans; explicit per-shape tuning wins."""
    plans = tuple(linear_parameters(n=w.shape[0], k=w.shape[1]) for w in weights)
    if not plans or not all(p.grouped_projections and p == plans[0] for p in plans):
        return None
    return BF16GroupedLinearPrimitive(x, weights, outputs, active_rows)


@triton.jit
def _grouped_fixed_tile(
    x,
    weights,
    outputs,
    active_rows,
    NS: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    first = 0
    for group in tl.static_range(len(NS)):
        n = NS[group]
        if (tile >= first) & (tile < first + tl.cdiv(n, BN)):
            _bf16_fixed_tile_at(
                x,
                weights[group],
                outputs[group],
                active_rows,
                tl.program_id(1),
                tile - first,
                n,
                K,
                M,
                BK,
                BM,
                BN,
            )
        first += tl.cdiv(n, BN)


@triton.jit
def _grouped_row_column(
    x, weights, outputs, active_rows,
    NS: tl.constexpr, K: tl.constexpr, M: tl.constexpr,
    BK: tl.constexpr, ROWS: tl.constexpr,
):
    column = tl.program_id(0)
    row_base = tl.program_id(1) * ROWS
    offset = tl.arange(0, BK)
    active = tl.load(active_rows)
    valid = (active > 0) & (active <= M)
    first = 0
    for group in tl.static_range(len(NS)):
        n = NS[group]
        if (column >= first) & (column < first + n):
            weight = tl.load(
                weights[group] + (column - first).to(tl.int64) * K + offset,
                (offset < K) & valid, 0,
            ).to(tl.float32)
            # Keep the one-dimensional K layout for every row. A two-dimensional
            # row/K layout can change the reduction tree as ROWS changes.
            for lane in tl.range(0, ROWS, loop_unroll_factor=1):
                row = row_base + lane
                value = tl.load(
                    x + row.to(tl.int64) * K + offset,
                    (offset < K) & valid & (row < active), 0,
                ).to(tl.float32)
                result = tl.sum(weight * value, 0)
                tl.store(
                    outputs[group] + row.to(tl.int64) * n + column - first,
                    tl.where(row < active, result, 0), valid & (row < M),
                )
        first += n


class BF16GroupedLinearPrimitive:
    """No packed copies or workspace: each CTA owns one original output tile.

    Each member resolves its own ordinary-linear parameters from actual N/K.
    Grouping requires one common geometry; incompatible profiles stay separate.
    Group membership preserves independent weights and output rounding. Narrow
    projections bind a fixed row-local FP32 tree, independent of M and row reuse;
    other projections retain the ordered tensor-core K-block reduction.
    """

    def __init__(
        self,
        x: Tensor,
        weights: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        active_rows: Tensor,
    ) -> None:
        if len(weights) < 2 or len(weights) != len(outputs):
            raise ValueError("grouped linear requires matching independent projections")
        for weight, output in zip(weights, outputs, strict=True):
            _validate_layout(x, weight, output, active_rows, inspect_active=True,
                             allowed_dtypes=(torch.bfloat16, torch.float16))
        self._count = len(weights)
        self._device = x.device
        self._bindings = self._bind(x, weights, outputs, active_rows)
        if len({item.pointer for item in self._bindings}) != len(self._bindings):
            raise ValueError("grouped linear operands must not alias")
        ns = tuple(weight.shape[0] for weight in weights)
        m, k = x.shape
        parameters = tuple(linear_parameters(n=n, k=k) for n in ns)
        p = parameters[0]
        if not all(item.grouped_projections and item == p for item in parameters):
            raise ValueError("grouped linear requires enabled identical member plans")
        self._dtype_tag = "bf16" if x.dtype is torch.bfloat16 else "fp16"
        self._ordinal = 0
        self._token = object()
        self._last_bindings = self._bindings
        self._stream = None
        self._operator_id = f"cubic.cuda.{self._dtype_tag}-grouped-fixed-tile.triton.v1"
        # Family selection is independent of physical/logical M: switching
        # arithmetic families between buckets would break cross-M identity.
        columns = sum(ns)
        tiles = sum(triton.cdiv(n, p.fixed_block_n) for n in ns)
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        fixed_override = any(
            key.startswith("fixed_") for n in ns
            for key in parameter_overrides("linear", n=n, k=k)
        )
        if (not fixed_override and k >= p.grouped_vector_min_k and tiles < sms
                and 100 * columns <= p.grouped_vector_tile_fill_percent
                * tiles * p.fixed_block_n):
            self._bind_row_column(x, weights, outputs, active_rows, ns, p)
            return
        block_m = linear_row_tile(m, p.fixed_block_m, p.fixed_max_block_m)
        self._grid = (
            sum(triton.cdiv(n, p.fixed_block_n) for n in ns),
            triton.cdiv(m, block_m),
            1,
        )
        block_n = p.fixed_block_n
        warps, stages = p.fixed_warps, p.fixed_stages
        with torch.cuda.device(x.device):
            self._artifact = _grouped_fixed_tile.warmup(
                x,
                weights,
                outputs,
                active_rows,
                NS=ns,
                K=k,
                M=m,
                BM=block_m,
                BN=p.fixed_block_n,
                BK=p.fixed_block_k,
                grid=self._grid,
                num_warps=p.fixed_warps,
                num_stages=p.fixed_stages,
            )
            self._artifact._init_handles()
            cost = current_linear_cost()
            if cost is not None:
                baseline = MatmulGeometry(block_m, block_n, p.fixed_block_k, warps, stages)
                geometries = linear_geometries(
                    cost.device, rows=m, columns=ns, reduction=k,
                    element_bytes=x.element_size(), communication=cost.communication,
                    baseline=baseline, parameters=p,
                )
                tiles = [LinearTile(self._artifact, self._grid, block_m, block_n, baseline)]
                for point in geometries[1:]:
                    bm, bn = point.block_m, point.block_n
                    grid = (sum(triton.cdiv(n, bn) for n in ns), triton.cdiv(m, bm), 1)
                    candidate = _grouped_fixed_tile.warmup(
                        x, weights, outputs, active_rows, NS=ns, K=k, M=m,
                        BM=bm, BN=bn, BK=p.fixed_block_k, grid=grid,
                        num_warps=point.warps, num_stages=point.stages,
                    )
                    candidate._init_handles()
                    tiles.append(LinearTile(candidate, grid, bm, bn, point))
                selected = cost.select(tiles, x, weights, grouped=True)
                self._artifact, self._grid = selected.kernel, selected.grid
                block_m, block_n = selected.block_m, selected.block_n
                assert selected.geometry is not None
                warps, stages = selected.geometry.warps, selected.geometry.stages
        self._signature = (
            f"m={m},ns={ns},k={k},block_m={block_m},block_n={block_n},"
            f"block_k={p.fixed_block_k},num_warps={warps},num_stages={stages},"
            f"reduction=ordered-k-blocks-{self._dtype_tag}-dot-fp32,"
            f"out={self._dtype_tag},active_rows=device-int32"
        )

    def _bind_row_column(
        self, x: Tensor, weights: tuple[Tensor, ...], outputs: tuple[Tensor, ...],
        active_rows: Tensor, ns: tuple[int, ...], p: LinearParameters,
    ) -> None:
        m, k = x.shape
        block_k = triton.next_power_of_2(k)
        with torch.cuda.device(x.device):
            tiles = []
            rows = 1
            while rows <= min(m, p.canonical_max_rows):
                grid = (sum(ns), triton.cdiv(m, rows), 1)
                kernel = _grouped_row_column.warmup(
                    x, weights, outputs, active_rows, ns, k, m, block_k, rows,
                    grid=grid, num_warps=p.canonical_warps, enable_fp_fusion=False,
                )
                kernel._init_handles()
                if not kernel.n_spills:
                    tiles.append(LinearTile(kernel, grid, rows, 1))
                rows *= 2
            if not tiles:
                raise RuntimeError("no spill-free grouped row-column launch")
            cost = current_linear_cost()
            if cost is None:
                cost = LinearLaunchCost(device_resources(x.device), 0.0, ())
            selected = cost.select(
                tiles, x, weights, grouped=True,
                replays=launch_parameters().micro_calibration_replays,
            )
            self._artifact, self._grid = selected.kernel, selected.grid
        self._operator_id = f"cubic.cuda.{self._dtype_tag}-grouped-row-column.triton.v1"
        self._signature = (
            f"m={m},ns={ns},k={k},block_k={block_k},rows={selected.block_m},"
            f"num_warps={p.canonical_warps},reduction=fp32-row-column-tree,"
            f"out={self._dtype_tag},active_rows=device-int32"
        )

    @staticmethod
    def _bind(x, weights, outputs, active_rows):
        return tuple(bind_tensor(t) for t in (x, *weights, *outputs, active_rows))

    @property
    def launch_signature(self) -> str:
        return self._signature

    @property
    def row_tile_count(self) -> int:
        """Physical row CTAs declared by this binding, independent of dtype policy."""
        return self._grid[1]

    def project_artifact(self) -> CompiledArtifactIdentity:
        return project_triton_artifact(
            self._artifact,
            operator_id=self._operator_id,
            device=self._device,
            launch_signature=self._signature,
            grid=self._grid,
        )

    def launch(
        self,
        x: Tensor,
        weights: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        active_rows: Tensor,
    ) -> LaunchReceipt:
        if len(weights) != self._count or len(outputs) != self._count:
            raise ValueError("grouped linear membership differs from its binding")
        bindings = self._bind(x, weights, outputs, active_rows)
        for item, base in zip(bindings, self._bindings, strict=True):
            if (item.shape, item.stride, item.dtype, item.device, item.pointer % 16) != (
                base.shape,
                base.stride,
                base.dtype,
                base.device,
                base.pointer % 16,
            ):
                raise ValueError("grouped linear operands differ from compiled layout")
        with torch.cuda.device(self._device):
            stream = int(torch.cuda.current_stream(self._device).cuda_stream)
            self._artifact[self._grid](x, weights, outputs, active_rows, stream=stream)
        self._ordinal += 1
        self._last_bindings, self._stream = bindings, stream
        return LaunchReceipt(
            id(self._artifact),
            bindings,
            self._signature,
            self._grid,
            stream,
            self._ordinal,
            self._token,
        )

    def validate_receipt(self, receipt: LaunchReceipt) -> None:
        if self._stream is None:
            raise ValueError("grouped linear has no launch receipt")
        validate_launch_receipt(
            receipt,
            artifact_object_id=id(self._artifact),
            bindings=self._last_bindings,
            launch_signature=self._signature,
            grid=self._grid,
            stream=self._stream,
            ordinal=self._ordinal,
            primitive_token=self._token,
        )
