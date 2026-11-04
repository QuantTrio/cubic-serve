"""Address-invariant FP16/BF16 tensor-parallel sum completion.

The transport gathers rank-major FP16/BF16 partials.  This primitive completes the
sum pointwise in canonical process-group rank order, accumulating in FP32 and
rounding once when the result is stored in the input dtype.  Keeping transport and math
separate makes the numerical contract independent of NCCL's reduction tree.
"""

from __future__ import annotations

import math
import os
import socket
from contextlib import nullcontext
from dataclasses import asdict, replace
from statistics import median
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor
from torch.distributed import ProcessGroup
from triton.language.extra import libdevice  # type: ignore[import-untyped]

from cubic_serve.ops.cuda.compiled_artifact import stable_digest
from cubic_serve.ops.cuda.launch_cost import (
    OrderedCollectiveCost,
    TransferCost,
    _launch_calibration_device,
    _launch_calibration_path,
    _read_startup_calibration,
    _write_startup_calibration,
    predicted_gain,
    prediction_interval,
)
from .host_sum import HostSum, _host_numa_capabilities
from cubic_serve.ops.cuda.startup import calibration_allowed, require_calibration_permission
from cubic_serve.ops.cuda.normalization.rms_norm import (
    _bf16_residual_zero_centered_rms_norm,
)
from cubic_serve.ops.strategies import (
    CollectiveParameters,
    collective_parameters,
    launch_parameters,
    parameter_overrides,
)

# CPU coefficients only. Weak group ownership cannot keep NCCL communicators,
# CUDA graphs or private probe buffers alive after their executor is destroyed.
_TRANSPORT_COSTS: WeakKeyDictionary = WeakKeyDictionary()
_TRANSPORT_IDENTITIES: WeakKeyDictionary = WeakKeyDictionary()


def _transport_group_identity(device, group):
    if group not in _TRANSPORT_IDENTITIES:
        backing = ("ordered_host_backing_v1", _host_numa_capabilities(device, group))
        local = (_launch_calibration_device(device), socket.gethostname(),
                 tuple(sorted(os.sched_getaffinity(0))), torch.cuda.nccl.version(),
                 tuple(sorted((k, v) for k, v in os.environ.items() if k.startswith("NCCL_"))),
                 backing)
        peers = [None] * dist.get_world_size(group)
        dist.all_gather_object(peers, local, group=group)
        _TRANSPORT_IDENTITIES[group] = tuple(peers)
    return _TRANSPORT_IDENTITIES[group]


def _read_transport_cost(path):
    raw = _read_startup_calibration(path, "cubic.ordered-transport-cost.v1")
    if raw is None:
        return None
    try:
        value = dict(raw["cost"])
        for name in ("gather", "partition", "host", "host_partition"):
            value[name] = (None if value[name] is None else
                           tuple(TransferCost(**point) for point in value[name]))
        value["transition_bytes"] = tuple(value["transition_bytes"])
        value["interval_error_fractions"] = tuple(tuple(x) for x in value["interval_error_fractions"])
        if value["legacy_error_fractions"] is not None:
            value["legacy_error_fractions"] = tuple(value["legacy_error_fractions"])
        if any(type(x) is not int for x in (value["minimum_bytes"], value["maximum_bytes"],
                                            *value["transition_bytes"])):
            return None
        return OrderedCollectiveCost(**value)
    except (KeyError, TypeError, ValueError):
        return None


def _transport_key(template: Tensor, policy: CollectiveParameters, ranks: int) -> tuple:
    """Keep small-message fits stable while covering larger cold graph shapes."""
    alignment = ranks * template.element_size()
    budget = max(32 * alignment, policy.transport_calibration_bytes)
    payload = template.numel() * template.element_size()
    extent = 1 << max(0, (payload - 1).bit_length())
    # Refine only below the existing first anchor. Measure the actual aligned
    # payload at the local fit's central anchor: rounding it to a power of two
    # can place a latency transition solely in an interpolated holdout interval.
    # Keep the same probe count and preserve anchors for already-covered sizes.
    if payload < budget // 32:
        budget = min(budget, max(32 * alignment,
                                 math.ceil(payload / alignment) * alignment))
    maximum = max(4 * budget, extent)
    return (template.device, template.dtype, policy,
            launch_parameters().calibration_samples, maximum)


def _transport_probe_sizes(budget: int, maximum: int, alignment: int) -> list[int]:
    # Small payloads get a smaller latency-domain fit. Large payloads retain
    # the existing low anchors while extending only the high bandwidth domain.
    budget = min(budget, maximum // 4)
    return sorted({max(alignment, math.ceil(b / alignment) * alignment)
                   for b in (budget / 32, budget / 4, budget / 2, 3 * budget / 4,
                             budget, maximum / 2, maximum)})


def _transport_cost(template: Tensor, group: ProcessGroup,
                    policy: CollectiveParameters) -> OrderedCollectiveCost | None:
    """Preserve the primary fit; refine a competitive unqualified local interval once."""
    ranks = dist.get_world_size(group)
    payload = template.numel() * template.element_size()
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("transport planning belongs before graph capture")
    key = _transport_key(template, policy, ranks)
    costs = _TRANSPORT_COSTS.setdefault(group, {})

    def measured(probe_key: tuple) -> OrderedCollectiveCost | None:
        if probe_key not in costs or (costs[probe_key] is None and calibration_allowed()):
            try:
                path, cached = None, None
                if template.device.type == "cuda" and dist.is_initialized():
                    peers = _transport_group_identity(template.device, group)
                    path = _launch_calibration_path(("ordered-transport", probe_key, peers), (), template.device)
                    cached = _read_transport_cost(path)
                    witnesses = [None] * ranks
                    dist.all_gather_object(witnesses,
                        None if cached is None else stable_digest(asdict(cached)), group=group)
                    if any(x is None for x in witnesses) or len(set(witnesses)) != 1:
                        cached = None
                if cached is None:
                    if not calibration_allowed():
                        costs[probe_key] = None
                        return None
                    from cubic_serve.ops.cuda.startup import startup_cuda_lock
                    guard = (startup_cuda_lock(template.device)
                             if template.device.type == "cuda" else nullcontext())
                    with guard:
                        cached = _calibrate_transport(
                            template.device, template.dtype, group, policy, maximum_bytes=probe_key[-1])
                    if cached is not None:
                        _write_startup_calibration(path, "cubic.ordered-transport-cost.v1", {"cost": asdict(cached)})
                costs[probe_key] = cached
            except ValueError as error:
                if not str(error).startswith("transfer fit"):
                    raise
                costs[probe_key] = None
        return costs[probe_key]

    cost = measured(key)
    alignment = ranks * template.element_size()
    local_maximum = 4 * max(32 * alignment, math.ceil(payload / alignment) * alignment)
    # A supplement may extend above the primary maximum when its actual
    # payload anchor is nearer that upper bound. Only an identical probe
    # domain is redundant; the primary coefficients remain independently cached.
    if cost is None or local_maximum == key[-1]:
        return cost
    allowed = (True, True, policy.host_sum, policy.host_sum and policy.host_sum_partition)
    curves = (cost.gather, cost.partition, cost.host, cost.host_partition)
    available = [i for i, fits in enumerate(curves) if allowed[i] and fits is not None]
    limit = policy.transport_max_fit_error_percent / 100
    qualified = [cost._predict(curves[i], payload, cost.transition_bytes)
                 for i in available if cost._error(i, payload) <= limit]
    ceiling = min(qualified, default=math.inf) * (1 - policy.transport_min_gain_percent / 100)
    unresolved = [i for i in available if cost._error(i, payload) > limit
                  and cost._predict(curves[i], payload, cost.transition_bytes) < ceiling]
    # A qualified faster candidate still cannot replace an unqualified
    # incumbent: select_host must validate both sides. Refine that incumbent
    # too, rather than leaving the comparison permanently blocked by its fit.
    incumbent = int(cost.select(
        payload, fallback=policy.partitioned_sum(payload, ranks),
        min_gain_percent=policy.transport_min_gain_percent,
        max_error_percent=policy.transport_max_fit_error_percent,
    ))
    incumbent_cost = cost._predict(curves[incumbent], payload, cost.transition_bytes)
    if (cost._error(incumbent, payload) > limit and qualified
            and min(qualified) < incumbent_cost * (1 - policy.transport_min_gain_percent / 100)):
        unresolved.append(incumbent)
    if not unresolved:
        return cost
    local = measured((*key[:-1], local_maximum))
    if local is not None:
        local_curves = (local.gather, local.partition, local.host, local.host_partition)
        if (any(local_curves[i] is not None and local._error(i, payload) <= limit
                for i in unresolved)
                and all(local._error(i, payload) <= limit for i in available
                        if local_curves[i] is not None)):
            return local
    return cost


def partitioned_transport(template: Tensor, group: ProcessGroup,
                          policy: CollectiveParameters | None = None) -> bool:
    """Bind one shared transport rule before capture for plain and routed sums."""
    policy = policy or collective_parameters()
    ranks = dist.get_world_size(group)
    payload = template.numel() * template.element_size()
    if ranks <= 1 or template.numel() % ranks:
        return False
    fallback = policy.partitioned_sum(payload, ranks)
    if (not policy.transport_calibration
            or "exchange_latency_bytes" in parameter_overrides("collective")):
        return fallback
    cost = _transport_cost(template, group, policy)
    return fallback if cost is None else cost.select(
        payload, fallback=fallback, min_gain_percent=policy.transport_min_gain_percent,
        max_error_percent=policy.transport_max_fit_error_percent,
    )


def _calibrate_transport(device: torch.device, dtype: torch.dtype, group: ProcessGroup,
                         policy: CollectiveParameters, *, maximum_bytes: int) -> OrderedCollectiveCost:
    """Explicit offline graph probes, interleaved controls and slowest-rank costs.

    vLLM89af4f1/custom_all_reduce.py guards topology capabilities; SGLang
    ba433bb/parallel_state.py separates transport selection; llama.cpp8b4b355
    ggml-cuda.cu groups NCCL calls. Their floating reduction trees are not our
    canonical arithmetic. Here both transports share the same rank-sum kernel.
    """
    require_calibration_permission("ordered transport candidates")
    ranks = dist.get_world_size(group)
    element_bytes = torch.empty((), dtype=dtype).element_size()
    alignment = ranks * element_bytes
    budget = max(32 * alignment, policy.transport_calibration_bytes)
    sizes = _transport_probe_sizes(budget, maximum_bytes, alignment)
    observations = []
    repeats = policy.transport_calibration_replays
    host_available = policy.host_sum
    for size in sizes:
        elements = size // element_bytes
        source = torch.full((elements,), 0.125 * (dist.get_rank(group) + 1),
                            dtype=dtype, device=device)
        output = torch.empty_like(source)
        gathered = torch.empty((ranks, elements), dtype=dtype, device=device)
        exchanged = torch.empty_like(source)
        partition = torch.empty((elements // ranks,), dtype=dtype, device=device)

        def gather(source=source, gathered=gathered, output=output, elements=elements) -> None:
            dist.all_gather_single(gathered.reshape(-1), source, group=group)
            _fixed_rank_bf16_sum_kernel[(triton.cdiv(elements, 256),)](
                gathered, output, ELEMENTS=elements, RANKS=ranks, BLOCK=256,
            )

        def exchange(source=source, exchanged=exchanged, partition=partition,
                     output=output, elements=elements) -> None:
            dist.all_to_all_single(exchanged, source, group=group)
            _fixed_rank_bf16_sum_kernel[(triton.cdiv(elements // ranks, 256),)](
                exchanged, partition, ELEMENTS=elements // ranks, RANKS=ranks, BLOCK=256,
            )
            dist.all_gather_single(output, partition, group=group)

        expected = torch.tensor(0.0, dtype=torch.float32)
        for rank in range(ranks):
            expected += torch.tensor(0.125 * (rank + 1), dtype=dtype).float()
        expected = expected.to(dtype).expand(elements)
        host = HostSum.create(source, group, policy.host_sum_threads) if host_available else None
        if host is not None and host.workload_resource_identity != host.planned_workload_resource_identity:
            # A declined NUMA allocation is a different cost domain. Do not
            # publish that fallback's timings as prices of the planned backing.
            host.close()
            host = None
        host_available = host is not None
        if host is not None:
            host.bind(source, output)
        host_partition = (HostSum.create(source, group, policy.host_sum_threads, partitioned=True)
                          if host is not None and policy.host_sum_partition else None)
        if (host_partition is not None and host_partition.workload_resource_identity
                != host_partition.planned_workload_resource_identity):
            host_partition.close()
            host_partition = None
        if host_partition is not None:
            host_partition.bind(source, output)
        functions = (gather, exchange) + ((lambda: host.launch(source, output),) if host else ())
        if host_partition is not None:
            functions += (lambda: host_partition.launch(source, output),)
        graphs = []
        events = []
        try:
            for fn in functions:
                for _ in range(2):
                    fn()
                if not torch.equal(output.cpu(), expected):
                    raise RuntimeError("transport calibration changed ordered arithmetic")
                torch.cuda.synchronize(device)
                dist.barrier(group=group)
                graph = torch.cuda.CUDAGraph()
                graphs.append(graph)
                start = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                events.append((start, end))
                with torch.cuda.graph(graph):
                    # Meet peers before timing steady transport. Host submission
                    # skew is not transfer latency and can invert the cost fit.
                    fn()
                    start.record()
                    for _ in range(repeats):
                        fn()
                    end.record()
                graph.replay()
                torch.cuda.synchronize(device)
            times: list[list[float]] = [[] for _ in functions]
            for _ in range(launch_parameters().calibration_samples):
                for index in (*range(len(functions)), *reversed(range(len(functions)))):
                    dist.barrier(group=group)
                    start, end = events[index]
                    graphs[index].replay()
                    end.synchronize()
                    times[index].append(start.elapsed_time(end) * 1000 / repeats)
            values = torch.tensor([median(t) for t in times], device=device,
                                  dtype=torch.float64)
            dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
            observations.append((size, tuple(values.cpu().tolist())))
        finally:
            for graph in graphs:
                graph.reset()
            if host is not None:
                host.close()
            if host_partition is not None:
                host_partition.close()
    return OrderedCollectiveCost.fit(
        [observations[i] for i in (0, 2, 4, 6)],
        [observations[i] for i in (1, 3, 5)],
    )


@triton.jit
def _rn_add(left, right):  # type: ignore[no-untyped-def]
    """One non-contractable IEEE FP32 round-to-nearest addition."""

    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit  # type: ignore[untyped-decorator]
def _fixed_rank_bf16_sum_kernel(  # type: ignore[no-untyped-def]
    gathered,
    output,
    ELEMENTS: tl.constexpr,
    RANKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEMENTS
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    # Do not rotate the source pointers by the local rank: every participant
    # must execute the same rank order for a bitwise-identical result.
    for rank in tl.static_range(0, RANKS):
        partial = tl.load(
            gathered + rank * ELEMENTS + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        total = _rn_add(total, partial)
    tl.store(output + offsets, total, mask=mask)


@triton.jit  # type: ignore[untyped-decorator]
def _fixed_rank_bf16_sum_residual_rms_norm_kernel(  # type: ignore[no-untyped-def]
    gathered,
    residual,
    summed,
    weight,
    output,
    ROWS: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    RANKS: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Complete a canonical TP sum then preserve two input-dtype rounding boundaries.

    The rank-order sum is deliberately rounded to the input dtype in registers before
    the residual addition.  The residual result is likewise rounded before
    the RMS reduction.  Those casts are the same two observable arithmetic
    boundaries as the unfused fixed-rank completion followed by Cubic's
    BF16ResidualRMSNormPrimitive; `value` itself is intentionally not written.
    """

    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < ROW_WIDTH
    offset = row * ROW_WIDTH + columns
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for rank in tl.static_range(0, RANKS):
        partial = tl.load(
            gathered + rank * ROWS * ROW_WIDTH + offset,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        total = _rn_add(total, partial)
    # This cast is semantically required: the old sum kernel stores the input dtype to
    # `value`, then the residual kernel reloads that rounded value.
    reduced = total.to(summed.dtype.element_ty).to(tl.float32)
    residual_value = tl.load(residual + offset, mask=mask, other=0.0).to(tl.float32)
    # This cast likewise matches the old residual destination before its
    # standalone RMSNorm reads the row.
    values = (residual_value + reduced).to(summed.dtype.element_ty)
    tl.store(summed + offset, values, mask=mask)
    values_fp32 = values.to(tl.float32)
    variance = libdevice.div_rn(tl.sum(values_fp32 * values_fp32, axis=0), float(ROW_WIDTH))
    inverse_rms = libdevice.rsqrt_rn(variance + EPSILON)
    rms_weight = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output + offset,
        values_fp32 * inverse_rms * (rms_weight + 1.0),
        mask=mask,
    )


class FixedRankBF16SumPrimitive:
    """Caller-owned, graph-capturable all-gather plus fixed-order sum.

    One instance is bound to one fixed tensor shape and process group.  The
    caller owns the output tensor while this object owns the stable gather
    workspace used by eager warmup and CUDA Graph replay.
    """

    def __init__(self, template: Tensor, group: ProcessGroup, *,
                 calibrate_transport: bool = True) -> None:
        if (
            template.dtype not in (torch.bfloat16, torch.float16)
            or template.device.type != "cuda"
            or not template.is_contiguous()
            or template.numel() <= 0
        ):
            raise ValueError("fixed-rank sum requires non-empty contiguous CUDA FP16/BF16")
        ranks = dist.get_world_size(group)
        if ranks < 1:
            raise ValueError("fixed-rank sum requires a non-empty process group")
        self._shape = tuple(template.shape)
        self.input_shapes = (self._shape,)
        self._dtype = template.dtype
        self._device = template.device
        self._group = group
        self._ranks = ranks
        self._elements = template.numel()
        self._rows = template.shape[0] if template.ndim == 2 else 0
        self._row_width = template.shape[1] if template.ndim == 2 else 0
        policy = collective_parameters()
        if not calibrate_transport:
            # Request-time eager shapes cannot capture calibration graphs while
            # another thread is replaying the resident model's CUDA graphs.
            # The analytical transport choice preserves the ordered sum math.
            policy = replace(policy, transport_calibration=False)
        self._use_partitioned_sum = (
            ranks > 1
            and self._elements % ranks == 0
            and partitioned_transport(template, group, policy)
        )
        self._gathered = (
            None
            if self._use_partitioned_sum or ranks == 1
            else torch.empty(
                (ranks, self._elements),
                dtype=self._dtype,
                device=template.device,
            )
        )
        self._exchanged = (
            torch.empty(self._elements, dtype=self._dtype, device=template.device)
            if self._use_partitioned_sum
            else None
        )
        self._partition = (
            torch.empty(
                self._elements // ranks,
                dtype=self._dtype,
                device=template.device,
            )
            if self._use_partitioned_sum
            else None
        )
        self._block = 256
        self._host = None
        cost = (_transport_cost(template, group, policy) if ranks > 1 and policy.transport_calibration
                and "exchange_latency_bytes" not in parameter_overrides("collective") else None)
        if (policy.host_sum and policy.transport_calibration
                and "exchange_latency_bytes" not in parameter_overrides("collective")
                and cost is not None):
            payload = self._elements * template.element_size()
            margins = dict(partitioned=self._use_partitioned_sum,
                           min_gain_percent=policy.transport_min_gain_percent,
                           max_error_percent=policy.transport_max_fit_error_percent)
            host = cost.select_host(payload, **margins)
            host_partition = policy.host_sum_partition and cost.select_host_partition(
                payload, host_selected=host, **margins,
            )
            if host or host_partition:
                self._host = HostSum.create(template, group, policy.host_sum_threads,
                                            partitioned=host_partition)
                if (self._host is not None and self._host.workload_resource_identity
                        != self._host.planned_workload_resource_identity):
                    # Retain the admitted canonical NCCL transport if the
                    # measured host backing could not be admitted this process.
                    self._host.close()
                    self._host = None

        self._estimated_us: float | None = None
        self._estimated_interval_us: tuple[float, float] | None = None
        if (cost is not None and policy.transport_calibration
                and cost.minimum_bytes <= self._elements * template.element_size()
                <= cost.maximum_bytes):
            payload = self._elements * template.element_size()
            transport = (3 if self._host.partitioned else 2) if self._host is not None else int(
                self._use_partitioned_sum
            )
            curves = (cost.gather, cost.partition, cost.host, cost.host_partition)[transport]
            error = cost._error(transport, payload)
            if curves is not None and error * 100 <= policy.transport_max_fit_error_percent:
                self._estimated_us = cost._predict(curves, payload, cost.transition_bytes)
                self._estimated_interval_us = prediction_interval(self._estimated_us, error)

    @classmethod
    def coalesced(
        cls, template: Tensor, members: tuple[FixedRankBF16SumPrimitive, ...]
    ) -> FixedRankBF16SumPrimitive | None:
        """Price one contiguous transport without adding independent outputs together.

        Startup only. The caller owns disjoint contiguous member views and must
        complete every producer before launch. Unknown or invalid fits decline.
        All ranks agree before changing their collective sequence.
        """
        if len(members) < 2:
            raise ValueError("coalesced rank sum requires independent members")
        first = members[0]
        if (first._ranks < 2 or any(
            member._group is not first._group or member._device != template.device
            or member._dtype != template.dtype or len(member.input_shapes) != 1
            for member in members
        ) or sum(member._elements for member in members) != template.numel()):
            return None
        candidate = cls(template, first._group)
        estimates = tuple(member._estimated_interval_us for member in members)
        # Every original completion is serial. Sum its own interval rather
        # than treating uncertain center values as guaranteed saved latency.
        original = tuple(sum(value[i] for value in estimates if value is not None)
                         for i in (0, 1))
        selected = (candidate._estimated_interval_us is not None
                    and all(value is not None for value in estimates)
                    and predicted_gain(candidate._estimated_interval_us, original,
                                       collective_parameters().transport_min_gain_percent))
        vote = torch.tensor([int(selected)], dtype=torch.int32, device=template.device)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=first._group)
        if not vote.item():
            return None
        candidate.input_shapes = tuple(member._shape for member in members)
        return candidate

    def input_views(self, value: Tensor) -> tuple[Tensor, ...]:
        """Logical member views for deferred numerical and profiling observers."""
        flat = value.view(-1)
        offset = 0
        parts = []
        for shape in self.input_shapes:
            count = math.prod(shape)
            parts.append(flat[offset:offset + count].view(shape))
            offset += count
        if offset != value.numel():
            raise ValueError("rank sum logical member layout differs")
        return tuple(parts)

    @property
    def launch_signature(self) -> str:
        """Seal the chosen transport, not incidental probe timings/addresses."""
        identity = (
            "ordered_rank_sum", self._shape, str(self._dtype), self._ranks,
            self._use_partitioned_sum, self._block,
            self._host.signature if self._host is not None else None,
        )
        return stable_digest(identity if len(self.input_shapes) == 1
                             else (identity, self.input_shapes))

    @property
    def gathered(self) -> Tensor:
        """Stable rank-major workspace, exposed for qualification tests."""

        if self._gathered is None:
            raise RuntimeError("partitioned transport has no full rank-major workspace")
        return self._gathered

    def _partitioned_sum(self, value: Tensor, output: Tensor) -> None:
        exchanged, partition = self._exchanged, self._partition
        if exchanged is None or partition is None:
            raise RuntimeError("partitioned fixed-rank transport is not bound")
        chunk = self._elements // self._ranks
        dist.all_to_all_single(exchanged, value.reshape(self._elements), group=self._group)
        _fixed_rank_bf16_sum_kernel[(triton.cdiv(chunk, self._block),)](
            exchanged,
            partition,
            ELEMENTS=chunk,
            RANKS=self._ranks,
            BLOCK=self._block,
        )
        dist.all_gather_single(output.reshape(self._elements), partition, group=self._group)

    def launch(self, value: Tensor) -> None:
        self._validate_value(value)
        if self._ranks == 1:
            return
        if self._host is not None:
            self._host.launch(value)
            return
        if self._use_partitioned_sum:
            self._partitioned_sum(value, value)
            return
        assert self._gathered is not None
        dist.all_gather_single(
            self._gathered.view(-1),
            value.reshape(self._elements),
            group=self._group,
        )
        _fixed_rank_bf16_sum_kernel[(triton.cdiv(self._elements, self._block),)](
            self._gathered,
            value,
            ELEMENTS=self._elements,
            RANKS=self._ranks,
            BLOCK=self._block,
        )

    def launch_residual_rms_norm(
        self,
        value: Tensor,
        residual: Tensor,
        summed: Tensor,
        weight: Tensor,
        output: Tensor,
        *,
        epsilon: float,
    ) -> None:
        """Fuse canonical completion with the following FP16/BF16 residual RMSNorm.

        This is deliberately a separate ABI from :meth:`launch`: callers that
        need the reduced tensor as an observable value retain the standalone
        completion.  The fused ABI accepts only one fixed, contiguous [M, H]
        launch shape and fails closed rather than changing its reduction tree.
        """

        self._validate_value(value)
        if (
            self._rows <= 0
            or self._row_width <= 0
            or triton.next_power_of_2(self._row_width) > 8192
        ):
            raise ValueError("fused fixed-rank residual RMSNorm requires [M,H] with H <= 8192")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(
                "fused fixed-rank residual RMSNorm epsilon must be finite and positive"
            )
        operands = residual, summed, weight, output
        if (
            any(not isinstance(tensor, Tensor) for tensor in operands)
            or any(tensor.device != self._device for tensor in operands)
            or any(tensor.dtype != self._dtype for tensor in operands)
            or residual.shape != self._shape
            or summed.shape != self._shape
            or output.shape != self._shape
            or weight.shape != (self._row_width,)
            or any(not tensor.is_contiguous() for tensor in operands)
            or summed.untyped_storage().data_ptr()
            in {
                value.untyped_storage().data_ptr(),
                residual.untyped_storage().data_ptr(),
            }
        ):
            raise ValueError(
                "expected non-aliasing contiguous same-device CUDA FP16/BF16 "
                "fused residual RMSNorm operands"
            )
        block = triton.next_power_of_2(self._row_width)
        if self._ranks == 1 or self._use_partitioned_sum:
            completed = value
            if self._ranks > 1:
                assert self._exchanged is not None
                completed = self._exchanged.view_as(value)
                self._partitioned_sum(value, completed)
            _bf16_residual_zero_centered_rms_norm[(self._rows,)](
                residual,
                completed,
                summed,
                weight,
                output,
                residual.stride(0),
                width=self._row_width,
                epsilon=float(epsilon),
                block=block,
                num_warps=8 if block >= 4096 else 4,
            )
            return
        assert self._gathered is not None
        dist.all_gather_single(
            self._gathered.view(-1),
            value.reshape(self._elements),
            group=self._group,
        )
        _fixed_rank_bf16_sum_residual_rms_norm_kernel[(self._rows,)](
            self._gathered,
            residual,
            summed,
            weight,
            output,
            ROWS=self._rows,
            ROW_WIDTH=self._row_width,
            RANKS=self._ranks,
            EPSILON=float(epsilon),
            BLOCK=block,
            num_warps=8 if block >= 4096 else 4,
        )

    def _validate_value(self, value: Tensor) -> None:
        if (
            tuple(value.shape) != self._shape
            or value.dtype != self._dtype
            or value.device != self._device
            or not value.is_contiguous()
        ):
            raise ValueError("fixed-rank sum value differs from its captured ABI")
