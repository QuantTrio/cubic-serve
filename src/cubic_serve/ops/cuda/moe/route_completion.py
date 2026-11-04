"""Ordered expert-route completion with row-local folding before output gather.

Transport ownership is temporary token-row ownership, never expert placement.
The caller owns one workspace per exclusively replayed stage/graph lane. No
module-global buffer, device/model selector, or alternate reduction tree exists.
"""

from __future__ import annotations

from typing_extensions import TypeAliasType

from collections.abc import Callable
from dataclasses import asdict, dataclass
import math
import sys
from statistics import median

import torch
import torch.distributed as dist
import triton  # type: ignore[import-untyped]
from torch import Tensor
from torch.distributed import ProcessGroup

from cubic_serve.ops.cuda.collective.fixed_rank_sum import (
    _fixed_rank_bf16_sum_kernel,
    partitioned_transport,
)
from cubic_serve.ops.cuda.compiled_artifact import (
    CompiledArtifactIdentity,
    LaunchReceipt,
    bind_tensor,
)
from cubic_serve.ops.cuda.launch_cost import RouteCompletionCost, TransferCost
from cubic_serve.ops.cuda.moe.fused import _route_combine
from cubic_serve.ops.strategies import CollectiveParameters, launch_parameters
from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch


@dataclass(frozen=True)
class RouteCompletionWorkspace:
    """Shared only by sequential route bindings of one fixed stage ABI."""

    group: ProcessGroup
    shape: tuple[int, int, int]
    index_dtype: torch.dtype
    exchanged: Tensor
    partition: Tensor
    folded: Tensor
    padded_routes: Tensor | None
    padded_ids: Tensor | None
    padded_weights: Tensor | None
    gathered: Tensor | None

    @classmethod
    def allocate(
        cls, routes: Tensor, group: ProcessGroup, index_dtype: torch.dtype,
    ) -> RouteCompletionWorkspace:
        if (routes.ndim != 3 or min(routes.shape) < 1 or not routes.is_cuda
                or not routes.is_contiguous()
                or routes.dtype not in (torch.bfloat16, torch.float16)
                or index_dtype not in (torch.int32, torch.int64)):
            raise ValueError("route completion requires contiguous CUDA FP16/BF16 [M,K,H]")
        ranks = dist.get_world_size(group)
        if ranks <= 1 or torch.cuda.is_current_stream_capturing():
            raise ValueError("route workspace requires a multi-rank group before capture")
        rows, top_k, width = routes.shape
        local = (rows + ranks - 1) // ranks
        padded = local * ranks
        dtype, device = routes.dtype, routes.device
        return cls(
            group, (rows, top_k, width), index_dtype,
            torch.empty((ranks, local, top_k, width), dtype=dtype, device=device),
            torch.empty((local, top_k, width), dtype=dtype, device=device),
            torch.empty((local, width), dtype=dtype, device=device),
            torch.zeros((padded, top_k, width), dtype=dtype, device=device)
            if padded != rows else None,
            torch.full((padded, top_k), -1, dtype=index_dtype, device=routes.device)
            if padded != rows else None,
            torch.zeros((padded, top_k), dtype=dtype, device=device) if padded != rows else None,
            torch.empty((padded, width), dtype=dtype, device=device) if padded != rows else None,
        )

    @property
    def scratch(self) -> tuple[Tensor, ...]:
        return tuple(x for x in (
            self.exchanged, self.partition, self.folded, self.padded_routes,
            self.padded_ids, self.padded_weights, self.gathered,
        ) if x is not None)


class RowPartitionedRouteCompletion(PackedOperatorBinding):
    """Original rank sum/store -> original weighted fold -> completed-row gather.

    Inputs must have identical route IDs and weights across this TP/ETP group;
    their producers and group domain are independently audited by the caller.
    Both kernels are retained, bound and preflighted before the first transfer.
    Primitive receipts are not a graph/topology qualification or a replay lease.
    """

    def __init__(
        self, routes: Tensor, route_ids: Tensor, route_weights: Tensor, output: Tensor,
        group: ProcessGroup, *, first_expert: int, experts: int,
        workspace: RouteCompletionWorkspace | None = None,
    ) -> None:
        workspace = workspace or RouteCompletionWorkspace.allocate(routes, group, route_ids.dtype)
        rows, top_k, width = workspace.shape
        ranks = workspace.exchanged.shape[0]
        local = workspace.partition.shape[0]
        if (workspace.group is not group or tuple(routes.shape) != workspace.shape
                or not routes.is_cuda or routes.dtype not in (torch.bfloat16, torch.float16)
                or workspace.partition.device != routes.device
                or workspace.partition.dtype != routes.dtype
                or route_ids.shape != (rows, top_k) or route_ids.dtype != workspace.index_dtype
                or route_weights.shape != (rows, top_k) or route_weights.dtype != routes.dtype
                or output.shape != (rows, width) or output.dtype != routes.dtype
                or any(x.device != routes.device or not x.is_contiguous()
                       for x in (routes, route_ids, route_weights, output))
                or first_expert < 0 or experts < 1):
            raise ValueError("route completion operands differ from the stage workspace ABI")
        if output.untyped_storage().data_ptr() in {
            x.untyped_storage().data_ptr() for x in
            (routes, route_ids, route_weights, *workspace.scratch)
        }:
            raise ValueError("route completion output must not alias input or private scratch")
        self.workspace = workspace
        self.routes, self.route_ids, self.route_weights = routes, route_ids, route_weights
        self.first_expert, self.experts = first_expert, experts
        self._rows = rows
        self._input = routes if workspace.padded_routes is None else workspace.padded_routes
        ids = route_ids if workspace.padded_ids is None else workspace.padded_ids
        weights = route_weights if workspace.padded_weights is None else workspace.padded_weights
        start = dist.get_rank(group) * local
        self._ids = ids[start:start + local]
        self._weights = weights[start:start + local]
        self._gathered = output if workspace.gathered is None else workspace.gathered
        self._operands = (routes, route_ids, route_weights, output, *workspace.scratch)
        self._bindings = tuple(bind_tensor(x) for x in self._operands)
        block = 256
        self._sum = PackedOperatorBinding(workspace.partition, (
            _PackedLaunch.compile(
                _fixed_rank_bf16_sum_kernel, (workspace.exchanged, workspace.partition),
                (workspace.partition.numel(), ranks, block),
                (triton.cdiv(workspace.partition.numel(), block),), {},
            ),
        ))
        self._fold = PackedOperatorBinding(workspace.folded, (
            _PackedLaunch.compile(
                _route_combine,
                (workspace.partition, self._ids, self._weights, workspace.folded),
                (first_expert, experts, local * width, width, top_k, block),
                (triton.cdiv(local * width, block),), {"enable_fp_fusion": False},
            ),
        ))
        super().__init__(output, (), scratch=workspace.scratch)

    def validate_operands(self) -> None:
        if tuple(bind_tensor(x) for x in self._operands) != self._bindings:
            raise ValueError("route completion input or scratch changed after startup binding")
        self._sum.validate_operands()
        self._fold.validate_operands()

    def project_artifacts(self) -> tuple[CompiledArtifactIdentity, ...]:
        return (*self._sum.project_artifacts(), *self._fold.project_artifacts())

    def launch(self) -> tuple[LaunchReceipt, ...]:
        self.validate_operands()
        w = self.workspace
        if w.padded_routes is not None:
            assert w.padded_ids is not None and w.padded_weights is not None
            w.padded_routes[:self._rows].copy_(self.routes)
            w.padded_ids[:self._rows].copy_(self.route_ids)
            w.padded_weights[:self._rows].copy_(self.route_weights)
        dist.all_to_all_single(w.exchanged.reshape(-1), self._input.reshape(-1), group=w.group)
        receipts = (*self._sum.launch(), *self._fold.launch())
        dist.all_gather_single(self._gathered.reshape(-1), w.folded.reshape(-1), group=w.group)
        if w.gathered is not None:
            self.output.copy_(w.gathered[:self._rows])
        return receipts


def _calibrate(
    top_k: int, width: int, dtype: torch.dtype, device: torch.device,
    group: ProcessGroup, policy: CollectiveParameters,
    index_dtype: torch.dtype,
) -> RouteCompletionCost:
    """Bounded private cold probes; keep neither their graphs nor GPU buffers."""
    ranks = dist.get_world_size(group)
    size = torch.empty((), dtype=dtype).element_size()
    index_size = torch.empty((), dtype=index_dtype).element_size()
    row_bytes = top_k * width * size
    small = max(ranks, ((policy.route_completion_calibration_bytes + row_bytes * ranks - 1)
                       // (row_bytes * ranks)) * ranks)
    samples = launch_parameters().calibration_samples
    repeats = policy.route_completion_calibration_replays

    def measure(fn: Callable[[], object]) -> float:
        for _ in range(2):
            fn()
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                # Price device work under continuous graph feeding, as in a
                # serving forward. A Python replay loop adds submission gaps
                # to tiny transfers and can corrupt the fitted startup cost.
                for _ in range(repeats):
                    fn()
            graph.replay()
            torch.cuda.synchronize(device)
            values = []
            for _ in range(samples):
                dist.barrier(group=group)
                start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
                end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                values.append(float(start.elapsed_time(end)) * 1000 / repeats)
            value = torch.tensor([median(values)], device=device)
            dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group)
            return float(value.item())
        finally:
            graph.reset()

    def probe(rows: int) -> tuple[int, tuple[float, ...], list[float], bool]:
        source = torch.full((rows, top_k, width), 0.125 * (dist.get_rank(group) + 1),
                            dtype=dtype, device=device)
        ids = torch.arange(top_k, dtype=index_dtype, device=device).expand(rows, top_k).contiguous()
        weights = torch.full((rows, top_k), 1 / top_k, dtype=dtype, device=device)
        full = torch.empty_like(source)
        gathered = torch.empty((ranks, *source.shape), dtype=dtype, device=device)
        output = torch.empty((rows, width), dtype=dtype, device=device)

        def gather() -> None:
            dist.all_gather_single(gathered.reshape(-1), source.reshape(-1), group=group)
            _fixed_rank_bf16_sum_kernel[(triton.cdiv(source.numel(), 256),)](
                gathered, full, ELEMENTS=source.numel(), RANKS=ranks, BLOCK=256,
            )
            _route_combine[(triton.cdiv(rows * width, 256),)](
                full, ids, weights, output, 0, top_k, rows * width, width, top_k, 256,
                enable_fp_fusion=False,
            )

        gather()
        # Independent scalar fixture: original rank additions, low store,
        # low weighted products and sequential low additions, all on the CPU.
        total = torch.tensor(0, dtype=torch.float32)
        for rank in range(ranks):
            total = total + torch.tensor(0.125 * (rank + 1), dtype=dtype).float()
        weighted = (total.to(dtype).float() * torch.tensor(1 / top_k, dtype=dtype).float())
        expected = torch.tensor(0, dtype=dtype)
        for _ in range(top_k):
            expected = expected + weighted.to(dtype)
        if not torch.equal(output.cpu(), expected.expand(rows, width)):
            raise RuntimeError("route calibration failed independent ordered arithmetic")
        if rows == 1:
            return rows, (float((ranks - 1) * row_bytes),), [measure(gather)], False
        fold = RowPartitionedRouteCompletion(source, ids, weights, torch.empty_like(output),
                                             group, first_expert=0, experts=top_k)
        w = fold.workspace

        def partition() -> None:
            dist.all_to_all_single(w.exchanged.reshape(-1), source.reshape(-1), group=group)
            fold._sum.launch()
            dist.all_gather_single(full.reshape(-1), w.partition.reshape(-1), group=group)
            _route_combine[(triton.cdiv(rows * width, 256),)](
                full, ids, weights, output, 0, top_k, rows * width, width, top_k, 256,
                enable_fp_fusion=False,
            )

        pad = RowPartitionedRouteCompletion(source[:-1], ids[:-1], weights[:-1], output[:-1],
                                            group, first_expert=0, experts=top_k)
        pw = pad.workspace
        assert pw.padded_routes is not None and pw.padded_ids is not None
        assert pw.padded_weights is not None and pw.gathered is not None
        padded_routes, padded_ids = pw.padded_routes, pw.padded_ids
        padded_weights, padded_output = pw.padded_weights, pw.gathered
        padded_output.zero_()

        def copies() -> None:
            padded_routes[:rows - 1].copy_(source[:-1])
            padded_ids[:rows - 1].copy_(ids[:-1])
            padded_weights[:rows - 1].copy_(weights[:-1])
            pad.output.copy_(padded_output[:rows - 1])

        times = []
        for fn in (gather, partition, fold.launch, copies):
            fn()
            actual = fold.output if fn == fold.launch else output
            if fn != copies and not torch.equal(actual.cpu(), expected.expand(rows, width)):
                raise RuntimeError("route calibration transport changed arithmetic")
            times.append(measure(fn))
        payload = rows * row_bytes
        wires = ((ranks - 1) * payload, 2 * (ranks - 1) * payload / ranks,
                 (ranks - 1) / ranks * rows * width * size * (top_k + 1),
                 2 * (rows - 1) * (top_k * (width * size + index_size + size) + width * size))
        return rows, wires, times, partitioned_transport(source, group, policy)
    observations = [probe(rows) for rows in (small, 4 * small, 8 * small, 1)]
    points = observations[:3]
    anchor = observations[3][1][0], observations[3][2][0]
    fits = [TransferCost.fit((anchor, (points[1][1][0], points[1][2][0])))]
    fits.extend(TransferCost.fit([(p[1][i], p[2][i]) for p in points[:2]]) for i in range(1, 4))
    errors = [0.0] * 4
    for i, fit in enumerate(fits):
        for j, (_rows, wires, times, partitioned) in enumerate(points):
            if (i == 0 and j == 1) or (i > 0 and j < 2):
                continue
            if i == 0 and partitioned:
                continue
            errors[i] = max(errors[i], abs(fit.predict(wires[i]) / times[i] - 1))
    return RouteCompletionCost(ranks, fits[0], fits[1], fits[2], fits[3],
                               max(errors[1:3]), errors[0], errors[3])


RouteCompletionCosts = TypeAliasType("RouteCompletionCosts", dict[tuple[object, ...], RouteCompletionCost | None])


class RouteCompletionPlanner:
    """Caller-owned shared CPU fits; exclusively owned stage GPU workspace."""

    def __init__(
        self, template: Tensor, group: ProcessGroup, policy: CollectiveParameters,
        costs: RouteCompletionCosts, *, index_dtype: torch.dtype = torch.int64,
    ) -> None:
        self.group, self.shape = group, tuple(template.shape)
        self.index_dtype = index_dtype
        self.workspace: RouteCompletionWorkspace | None = None
        self.selected = False
        self.cost: RouteCompletionCost | None = None
        self.predicted_us: tuple[float, float] | None = None
        if not policy.route_completion or dist.get_world_size(group) <= 1:
            return
        rows, top_k, width = template.shape
        from cubic_serve.ops.cuda.cache_identity import cuda_libraries, module_kernels
        from cubic_serve.ops.cuda.launch_cost import (
            _launch_calibration_path, _read_startup_calibration, _write_startup_calibration)
        images = (module_kernels(sys.modules[__name__]), cuda_libraries(("libnccl",)))
        entry = (str(template.dtype), str(index_dtype), top_k, width,
                 dist.get_world_size(group), dist.get_rank(group), asdict(policy))
        selection_path = _launch_calibration_path(("route-completion-plan", rows, entry), (),
                                                   template.device, kernels=images)
        selection_schema = "cubic.route-completion-plan.v1"
        selected = _read_startup_calibration(selection_path, selection_schema)
        valid = (selected is not None and type(selected.get("selected")) is bool
                 and (selected.get("predicted_us") is None or (
                     isinstance(selected["predicted_us"], (list, tuple))
                     and len(selected["predicted_us"]) == 2
                     and all(type(t) in (int, float) and math.isfinite(t) and t >= 0
                             for t in selected["predicted_us"]))))
        agreement = torch.tensor([int(valid)], dtype=torch.int32, device=template.device)
        dist.all_reduce(agreement, op=dist.ReduceOp.MIN, group=group)
        if bool(agreement.item()):
            self.selected = selected["selected"]
            prices = selected.get("predicted_us")
            self.predicted_us = None if prices is None else tuple(prices)
            return
        key = (group, template.device, template.dtype, index_dtype, top_k, width, policy,
               launch_parameters().calibration_samples)
        if key not in costs:
            from cubic_serve.ops.cuda.startup import calibration_allowed
            path = _launch_calibration_path(
                ("route-completion", entry), (), template.device, kernels=images)
            schema = "cubic.route-completion.v1"
            raw = _read_startup_calibration(path, schema)
            cached = False
            if raw is not None and "cost" in raw:
                try:
                    value = raw["cost"]
                    costs[key] = None if value is None else RouteCompletionCost(**dict(value,
                        **{name: TransferCost(**value[name]) for name in
                           ("gather", "partition", "folded", "padding_copy")}))
                    cached = True
                except (TypeError, KeyError, ValueError):
                    pass
            agreement = torch.tensor([int(cached)], dtype=torch.int32, device=template.device)
            dist.all_reduce(agreement, op=dist.ReduceOp.MIN, group=group)
            if not bool(agreement.item()):
                if not calibration_allowed():
                    costs[key] = None
                    _write_startup_calibration(path, schema,
                        dict(cost=None, origin="production_default"))
                else:
                    try:
                        costs[key] = _calibrate(
                            top_k, width, template.dtype, template.device, group, policy, index_dtype,
                        )
                    except ValueError as error:
                        if not str(error).startswith("transfer fit"):
                            raise
                        costs[key] = None
                    _write_startup_calibration(path, schema,
                        dict(cost=None if costs[key] is None else asdict(costs[key])))
        self.cost = costs[key]
        if self.cost is None:
            from cubic_serve.ops.cuda.startup import calibration_allowed
            _write_startup_calibration(selection_path, selection_schema,
                dict(selected=False, predicted_us=None,
                     **({} if calibration_allowed() else {"origin": "production_default"})))
            return
        partitioned = partitioned_transport(template, group, policy)
        self.selected = self.cost.beneficial(
            rows, top_k, width, template.element_size(), partitioned=partitioned,
            min_gain_percent=policy.route_completion_min_gain_percent,
            max_error_percent=policy.route_completion_max_fit_error_percent,
            index_bytes=torch.empty((), dtype=index_dtype).element_size(),
        )
        self.predicted_us = self.cost.predict(
            rows, top_k, width, template.element_size(), partitioned=partitioned,
            index_bytes=torch.empty((), dtype=index_dtype).element_size(),
        )
        _write_startup_calibration(selection_path, selection_schema,
                                  dict(selected=self.selected, predicted_us=self.predicted_us))

    def bind(
        self, routes: Tensor, ids: Tensor, weights: Tensor, output: Tensor,
        first_expert: int, experts: int,
    ) -> RowPartitionedRouteCompletion | None:
        if not self.selected:
            return None
        if tuple(routes.shape) != self.shape or ids.dtype != self.index_dtype:
            raise ValueError("route planner cannot bind a different graph capacity")
        if self.workspace is None:
            self.workspace = RouteCompletionWorkspace.allocate(routes, self.group, ids.dtype)
        return RowPartitionedRouteCompletion(routes, ids, weights, output, self.group,
                                             first_expert=first_expert, experts=experts,
                                             workspace=self.workspace)
