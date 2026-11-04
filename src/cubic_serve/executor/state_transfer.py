"""Sealed, bit-preserving row transfers for disjoint state owners.

This is memory coordination, not model computation. Callers retain the topology
lease through durable consumption and qualify the containing model transaction.
Construction and selection are startup-only; replay never captures or allocates
device storage. Source and destination sets must be disjoint.

Upstream contracts: vLLM 003c91e7 v1/worker/mamba_utils.py batches state copies;
SGLang ba433bb4 mem_cache/memory_pool.py requires disjoint, rounded checkpoints.
llama.cpp 8b4b3558 llama-memory-recurrent.cpp instead aliases logical tails.
"""

from __future__ import annotations

import logging
import math
import os
import statistics
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import triton
import triton.language as tl

from cubic_serve.ops.cuda.cache_identity import module_kernels
import sys
from cubic_serve.runtime.logging import diagnostic_event
from cubic_serve.runtime.cache.quantization import slot_storage_contiguous, storage_regions_overlap

_LOGGER = logging.getLogger(__name__)


@triton.jit
def _byte_pattern(
    field, columns, tags, flag, W: tl.constexpr, WRITE: tl.constexpr, B: tl.constexpr,
    STRIDE: tl.constexpr = 0,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    expected = tl.load(columns + col, col < W, 0) ^ tl.load(tags + row)
    if WRITE:
        tl.store(field + row * (STRIDE if STRIDE else W) + col, expected, col < W)
    else:
        actual = tl.load(field + row * (STRIDE if STRIDE else W) + col, col < W, 0)
        bad = tl.sum(((actual != expected) & (col < W)).to(tl.int32), 0)
        if bad:
            tl.atomic_or(flag, 1)


@triton.jit
def owner_copy(
    ctx,
    dec,
    source,
    dest,
    CS: tl.constexpr,
    W: tl.constexpr,
    MODE: tl.constexpr,
    B: tl.constexpr,
    TILES: tl.constexpr,
    DEPTH: tl.constexpr = 1,
    SOURCE_STRIDE: tl.constexpr = 0,
    DESTINATION_STRIDE: tl.constexpr = 0,
):
    r = tl.program_id(0)
    n = tl.load(ctx)
    owner = tl.load(ctx + 1 + r * CS + 1).to(tl.int64)
    a = tl.load(dec + r * 2)
    if MODE == 0:
        origin = owner
        valid = r < n
    elif MODE == 2:
        origin = tl.load(ctx + 1 + r * CS + 3 + tl.minimum(tl.maximum(a, 0), DEPTH - 1)).to(
            tl.int64
        )
        valid = (r < n) & (a >= 0) & (a < DEPTH)
    elif MODE == 4:
        origin = tl.load(ctx + 1 + r * CS + 3 + tl.minimum(tl.maximum(a - 1, 0), DEPTH - 1)).to(tl.int64)
        valid = (r < n) & (a > 0) & (a <= DEPTH)
    else:
        origin = tl.load(ctx + 1 + r * CS + 2).to(tl.int64) + a - 1
        valid = (r < n) & (a > 0) & (a <= DEPTH)
    if valid:
        for start in range(tl.program_id(1) * B, W, TILES * B):
            x = start + tl.arange(0, B)
            v = tl.load(source + origin * (SOURCE_STRIDE if SOURCE_STRIDE else W) + x, x < W, 0)
            tl.store(dest + owner * (DESTINATION_STRIDE if DESTINATION_STRIDE else W) + x, v, x < W)


def _profitable_for_range(
    samples: dict[int, tuple[float, float]],
    maximum: int,
) -> tuple[bool, dict[str, Any]]:
    """Bounded affine host-wall model for one fixed field geometry.

    T(R)=a+bR includes dispatch and transport. Field geometry and device/host
    resources bind its receipt. Endpoints fit the model; interior/non-anchor
    rows measure its error. Choose a graph only when its entire interval wins.
    This conservative rule does not extrapolate to another field layout.
    """
    from cubic_serve.ops.cuda.launch_cost import fit_nonnegative_cost, prediction_interval

    if maximum < 1 or 1 not in samples or maximum not in samples:
        raise ValueError("state-transfer cost model needs its range endpoints")
    if any(
        n < 1 or n > maximum or len(t) != 2 or any(not math.isfinite(v) or v <= 0 for v in t)
        for n, t in samples.items()
    ):
        raise ValueError("state-transfer measurements must be finite positive costs")
    anchors = sorted({1, maximum})
    models = []
    for method in (0, 1):
        if maximum == 1:
            intercept, slope = samples[1][method], 0.0
        else:
            intercept, slope = fit_nonnegative_cost(
                [[1.0, float(n)] for n in anchors],
                [samples[n][method] for n in anchors],
            )
        error = max(abs((intercept + slope * n) / t[method] - 1.0) for n, t in samples.items())
        models.append(dict(intercept_ms=intercept, per_owner_ms=slope, relative_error=error))
    wins = True
    for n in anchors:
        intervals = [
            prediction_interval(m["intercept_ms"] + m["per_owner_ms"] * n, m["relative_error"])
            for m in models
        ]
        wins &= intervals[1][1] < intervals[0][0]
    return bool(wins), dict(
        formula="T(R)=a+bR",
        owners_range=[1, maximum],
        models=models,
        samples={str(n): t for n, t in samples.items()},
    )


class CapturedStateTransfer:
    """One shared descriptor and a finite power-of-two transfer portfolio.

    Integer storage views preserve every bit, including NaN payloads, integer
    histories and noncanonical boolean bytes. No floating-point conversion is
    part of this operation. The parent portfolio supplies numerical admission.
    """

    def __init__(self, fields: Sequence[torch.Tensor], maximum: int) -> None:
        if type(maximum) is not int or not 1 <= maximum <= 2**31 - 1:
            raise ValueError("invalid state-transfer owner capacity")
        if not fields or any(t.ndim < 1 or not t.numel()
                             or not (t.is_contiguous() or slot_storage_contiguous(t)) for t in fields):
            raise ValueError("state transfer requires contiguous inner row fields")
        self.device = fields[0].device
        self.rows = min(t.shape[0] for t in fields)
        if (
            self.device.type != "cuda"
            or self.rows > 2**31 - 1
            or maximum > self.rows // 2
            or any(t.device != self.device for t in fields)
        ):
            raise ValueError("state-transfer fields cannot cover disjoint owner sets")
        intervals = []
        unique = []
        for t in fields:
            span = (t.data_ptr(), t.shape[0], t[0].numel() * t.element_size(),
                    t.stride(0) * t.element_size())
            if span in intervals:
                previous = unique[intervals.index(span)]
                if previous.shape[0] != t.shape[0]:
                    raise ValueError("aliased state fields disagree about row ownership")
                continue
            if any(storage_regions_overlap(t, previous) for previous in unique):
                raise ValueError("state-transfer fields partially overlap")
            intervals.append(span)
            unique.append(t)
        word_types = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
        self.fields = tuple(
            t.view(word_types[math.gcd(t.element_size(), 8)]).reshape(t.shape[0], -1)
            for t in unique
        )
        self.maximum = maximum
        self.capacity = 1 << (maximum - 1).bit_length()
        self.stream = torch.cuda.Stream(device=self.device)
        self.ctx = torch.full((1 + 4 * self.capacity,), -1, dtype=torch.int32, device=self.device)
        self.ctx[0] = 0
        self.decisions = torch.zeros((self.capacity, 2), dtype=torch.int64, device=self.device)
        self.hosts = tuple(
            torch.empty_like(self.ctx, device="cpu", pin_memory=True) for _ in range(2)
        )
        self.arrays = tuple(t.numpy() for t in self.hosts)
        self.events = tuple(torch.cuda.Event() for _ in self.hosts)
        self.recorded = [False] * len(self.hosts)
        self.next_host = 0
        self.done = torch.cuda.Event()
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.qualified = False
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            for cap in (1 << n for n in range(self.capacity.bit_length())):
                self._copy(cap)
                self.stream.synchronize()
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph, stream=self.stream):
                    self._copy(cap)
                graph.instantiate()
                self.graphs[cap] = graph
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

    def _copy(self, capacity: int) -> None:
        for field in self.fields:
            width = field.shape[1]
            tiles = min(triton.cdiv(width, 2048), 256)
            owner_copy[(capacity, tiles)](
                self.ctx,
                self.decisions,
                field,
                field,
                4,
                width,
                2,
                2048,
                tiles,
                1,
                field.stride(0),
                field.stride(0),
            )

    def run(self, pairs: Sequence[tuple[int, int]]) -> None:
        n = len(pairs)
        if not n:
            return
        sources, destinations = tuple(zip(*pairs, strict=True))
        if (
            n > self.maximum
            or len(set(destinations)) != n
            or set(sources) & set(destinations)
            or any(type(i) is not int or not 0 <= i < self.rows for i in (*sources, *destinations))
        ):
            raise ValueError("state-transfer rows violate disjoint ownership")
        cap = 1 << (n - 1).bit_length()
        if cap not in self.graphs:
            raise RuntimeError("state-transfer graph bucket is not sealed")
        slot = self.next_host
        if self.recorded[slot]:
            self.events[slot].synchronize()
        host = self.arrays[slot]
        host[0] = n
        rows = host[1:].reshape(self.capacity, 4)
        rows[:n].fill(-1)
        rows[:n, 1] = destinations
        rows[:n, 3] = sources
        current = torch.cuda.current_stream(self.device)
        self.ctx[: 1 + 4 * n].copy_(self.hosts[slot][: 1 + 4 * n], non_blocking=True)
        self.events[slot].record(current)
        self.recorded[slot] = True
        self.next_host = (slot + 1) % len(self.hosts)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            self.graphs[cap].replay()
            self.done.record()
        # The parent worker synchronizes its feeding stream before releasing
        # the topology lease. This dependency includes the complete transfer.
        current.wait_event(self.done)

    def close(self) -> None:
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()

    def qualify(self) -> None:
        """Independent byte oracle on actual fields, outside any model forward.

        CPU constructs the owner map and column/row byte tags. Every storage
        byte of each distinct geometry/program, including protected rows and
        graph padding, is checked after completion. Every actual field's exact
        original graph binding is checked. The same source corpus covers boundary occupancies
        and reversed physical row maps. No numerical tolerance is involved.
        Original values are restored before the parent model audits continue.
        """
        if self.qualified:
            return
        with torch.cuda.device(self.device):
            from types import SimpleNamespace
            from cuda.bindings import runtime as rt
            from cubic_serve.executor.resident_window import check
            from cubic_serve.guardrail.graph_observer import (
                graph_nodes, kernel_identity, prove_graph_clone,
            )
            from cubic_serve.runtime.quantized_qualification import (
                _DraftBucketSnapshot, _draft_witness_budget,
            )
            started = time.perf_counter()
            ranks = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            byte_budget = _draft_witness_budget(ranks)
            groups = {}
            # Every original field keeps its exact pointer binding. Only the pure
            # byte mechanism is shared between equal shapes/dtypes, never data/state.
            by_pointer = {field.data_ptr(): i for i, field in enumerate(self.fields)}
            signatures, original_graphs, private, proxies = {}, self.graphs, {}, {}
            field_nodes = {}
            cases = snapshot_bytes = spilled_bytes = 0
            flag = torch.zeros((), dtype=torch.int32, device=self.device)
            shapes = {(t.shape[0], t[0].numel() * t.element_size()) for t in self.fields}
            phases = max(1, (max(max(rows, width) for rows, width in shapes).bit_length() + 7) // 8)
            try:
                for capacity, graph in original_graphs.items():
                    raw = graph.raw_cuda_graph()
                    nodes = graph_nodes(raw)
                    if len(nodes) != len(self.fields):
                        raise RuntimeError("state-transfer graph field inventory differs")
                    bindings = {}
                    for node in nodes:
                        if check(rt.cudaGraphNodeGetType(node))[0] != rt.cudaGraphNodeType.cudaGraphNodeTypeKernel:
                            raise RuntimeError("state-transfer graph contains an unowned node")
                        identity = kernel_identity(node)
                        args = tuple(value for _, value in identity[-1])
                        if len(args) != 6 or any(len(value) != 8 for value in args) or any(
                                any(value) for value in args[4:]):
                            raise RuntimeError("state-transfer graph argument schema differs")
                        ctx, decisions, source, destination = (
                            int.from_bytes(value, sys.byteorder) for value in args[:4])
                        index = by_pointer.get(source)
                        if (ctx != self.ctx.data_ptr() or decisions != self.decisions.data_ptr()
                                or index is None or destination != source or index in bindings):
                            raise RuntimeError("state-transfer graph field binding differs")
                        field = self.fields[index]
                        geometry = tuple(field.shape), field.dtype
                        tiles = min(triton.cdiv(field.shape[1], 2048), 256)
                        if identity[1:4] != (capacity, tiles, 1):
                            raise RuntimeError("state-transfer graph grid differs")
                        # Same compiled function and complete non-field launch/ABI
                        # bind the geometry proof to every independently owned field.
                        signature = (*identity[:-1], args[:2], args[4:])
                        signatures[capacity, index] = signature
                        bindings[index] = int(node)
                    if set(bindings) != set(range(len(self.fields))):
                        raise RuntimeError("state-transfer graph lost a field binding")
                    clone = check(rt.cudaGraphClone(raw))[0]
                    private[capacity] = [clone, None, None]
                    proof = prove_graph_clone(raw, clone)
                    mapped = dict(proof['nodes'])
                    executable = check(rt.cudaGraphInstantiateWithFlags(clone, 0))[0]
                    private[capacity][1] = executable
                    field_nodes[capacity] = {index: rt.cudaGraphNode_t(mapped[node])
                                             for index, node in bindings.items()}
                    for node in field_nodes[capacity].values():
                        check(rt.cudaGraphNodeSetEnabled(executable, node, 0))
                    def replay(executable=executable):
                        check(rt.cudaGraphLaunch(executable, self.stream.cuda_stream))
                    proxies[capacity] = SimpleNamespace(replay=replay)
                # Alignment or another compiled-program difference requires its
                # own byte witness even when tensor dimensions happen to match.
                for index, field in enumerate(self.fields):
                    geometry = tuple(field.shape), field.dtype
                    programs = tuple(signatures[capacity, index] for capacity in original_graphs)
                    groups.setdefault((geometry, programs), index)
                expected_context = self.ctx.detach().to('cpu', copy=True)
                expected_decisions = self.decisions.detach().to('cpu', copy=True)
                self.graphs = proxies
                for ((shape, _dtype), _programs), index in groups.items():
                    field = self.fields[index]
                    rows, width = shape[0], field[0].numel() * field.element_size()
                    for capacity, (_clone, executable, previous) in private.items():
                        if previous is not None:
                            check(rt.cudaGraphNodeSetEnabled(executable, field_nodes[capacity][previous], 0))
                        check(rt.cudaGraphNodeSetEnabled(executable, field_nodes[capacity][index], 1))
                        private[capacity][2] = index
                    indices = torch.arange(rows, device=self.device)
                    saved = _DraftBucketSnapshot(((str(index), field, indices),),
                        byte_budget=byte_budget, local_ranks=ranks)
                    snapshot_bytes += saved.total_bytes
                    if saved.file is not None:
                        spilled_bytes += saved.total_bytes - saved.zero_bytes
                    try:
                        columns = {}
                        for phase in range(phases):
                            column = torch.empty(width, dtype=torch.uint8, device=self.device)
                            chunk = max(1, byte_budget // 128)
                            for first in range(0, width, chunk):
                                stop = min(width, first + chunk)
                                ids = np.arange(first, stop, dtype=np.uint64)
                                raw = ((ids >> (8 * phase)) ^ (ids * 17)).astype(np.uint8)
                                column[first:stop].copy_(torch.from_numpy(raw))
                            columns[phase] = column
                        for capacity in self.graphs:
                            orders = ((n, order)
                                for n in sorted({min(capacity, self.maximum),
                                                 min(capacity // 2 + 1, self.maximum)})
                                for order in dict.fromkeys((tuple(range(n)), tuple(reversed(range(n))))))
                            for _n, order in orders:
                                pairs = tuple((self.maximum + i, destination)
                                              for i, destination in enumerate(order))
                                for phase in range(phases):
                                    ids = np.arange(rows, dtype=np.uint64)
                                    initial = torch.from_numpy((ids >> (8 * phase)).astype(np.uint8)).to(self.device)
                                    for source, destination in pairs:
                                        ids[destination] = source
                                    expected = torch.from_numpy((ids >> (8 * phase)).astype(np.uint8)).to(self.device)
                                    raw = field.view(torch.uint8).reshape(rows, -1)
                                    _byte_pattern[(rows, triton.cdiv(width, 4096))](
                                        raw, columns[phase], initial, flag, width, True, 4096,
                                        raw.stride(0))
                                    # The original host/context/stream binder runs;
                                    # only this private executable selects one field.
                                    expected_context[0] = len(pairs)
                                    bound = expected_context[1:].reshape(self.capacity, 4)[:len(pairs)]
                                    bound.fill_(-1)
                                    for row, (source, destination) in enumerate(pairs):
                                        bound[row, 1], bound[row, 3] = destination, source
                                    self.run(pairs)
                                    flag.zero_()
                                    _byte_pattern[(rows, triton.cdiv(width, 4096))](
                                        raw, columns[phase], expected, flag, width, False, 4096,
                                        raw.stride(0))
                                    if flag.item():
                                        raise RuntimeError("state transfer differs from independent byte oracle")
                                    if (not torch.equal(self.ctx.cpu(), expected_context)
                                            or not torch.equal(self.decisions.cpu(), expected_decisions)):
                                        raise RuntimeError("state transfer changed protected owner metadata")
                                    cases += 1
                    finally:
                        try:
                            self.stream.synchronize()
                            saved.restore()
                        finally:
                            saved.close()
                            torch.cuda.synchronize(self.device)
            finally:
                self.graphs = original_graphs
                self.stream.synchronize()
                for clone, executable, _previous in private.values():
                    try:
                        if executable is not None:
                            check(rt.cudaGraphExecDestroy(executable))
                    finally:
                        check(rt.cudaGraphDestroy(clone))
            self.qualified = True
            self._byte_proof = dict(all_field_bindings_exact=True, fields=len(self.fields),
                geometries=len(groups), cases=cases, host_byte_budget=byte_budget,
                snapshot_payload_bytes=snapshot_bytes, spilled_nonzero_bytes=spilled_bytes,
                seconds=time.perf_counter() - started)
            diagnostic_event(_LOGGER, "state_transfer.byte_proof",
                len(self.fields), self.maximum, cases, self._byte_proof['seconds'])

    @classmethod
    def select(cls, fields: Sequence[torch.Tensor], maximum: int) -> CapturedStateTransfer | None:
        """Cold host-wall selection; a hot receipt directly supplies the decision."""
        from cubic_serve.ops.cuda.launch_cost import (
            _launch_calibration_path,
            _read_startup_calibration,
            _write_startup_calibration,
        )

        schema = "cubic.state-transfer.v1"
        if not fields:
            raise ValueError("state transfer requires fields")
        started = time.perf_counter()
        geometry = tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype)) for t in fields)
        key = (
            schema,
            geometry,
            maximum,
            tuple(sorted(os.sched_getaffinity(0))),
        )
        path = _launch_calibration_path(key, (False, True), fields[0].device,
                                        kernels=module_kernels(sys.modules[__name__]))
        receipt = _read_startup_calibration(path, schema)
        if receipt is not None and type(receipt.get("graph_enabled")) is bool:
            diagnostic_event(_LOGGER, "state_transfer.cached",
                receipt["graph_enabled"],
                len(fields),
                maximum,
            )
            return cls(fields, maximum) if receipt["graph_enabled"] else None
        from cubic_serve.ops.cuda.startup import calibration_allowed
        if not calibration_allowed():
            _write_startup_calibration(path, schema,
                                       dict(graph_enabled=False, basis="retained_default",
                                            origin="production_default"))
            return None
        plan = cls(fields, maximum)
        # Only cold-start owner destinations are touched. Restore their exact
        # bytes before subsequent model audits; no additional device state pool.
        from cubic_serve.runtime.quantized_qualification import (
            _DraftBucketSnapshot, _draft_witness_budget,
        )
        ranks = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        indices = torch.arange(maximum, device=plan.device)
        try:
            backup = _DraftBucketSnapshot(tuple((str(i), tensor, indices)
                for i, tensor in enumerate(plan.fields)),
                byte_budget=_draft_witness_budget(ranks), local_ranks=ranks)
        except BaseException:
            plan.close()
            raise
        points = sorted({1, maximum, min(maximum, 3), max(1, maximum // 2 + 1)})
        keep_plan = False
        try:
            samples = {}
            try:
                for n in points:
                    pairs = tuple((maximum + i, n - i - 1) for i in range(n))

                    def direct(pairs=pairs) -> None:
                        for source, destination in pairs:
                            for field in plan.fields:
                                field[destination].copy_(field[source])

                    measured = []
                    for operation in (direct, lambda pairs=pairs: plan.run(pairs)):
                        for _ in range(2):
                            operation()
                        torch.cuda.synchronize(plan.device)
                        times = []
                        for _ in range(5):
                            start = time.perf_counter()
                            operation()
                            torch.cuda.synchronize(plan.device)
                            times.append((time.perf_counter() - start) * 1000)
                        measured.append(statistics.median(times))
                    samples[n] = tuple(measured)
                enabled, model = _profitable_for_range(samples, maximum)
            finally:
                try:
                    plan.stream.synchronize()
                    backup.restore()
                finally:
                    backup.close()
                    torch.cuda.synchronize(plan.device)
            _write_startup_calibration(path, schema, dict(graph_enabled=enabled, cost_model=model))
            diagnostic_event(_LOGGER, "state_transfer.prepared",
                enabled,
                len(fields),
                maximum,
                time.perf_counter() - started,
            )
            if not enabled:
                return None
            keep_plan = True
            return plan
        finally:
            if not keep_plan:
                plan.close()
