"""Static routed matmuls over original packed expert payloads.

Reuse the canonical router/compaction/combine and format-owned reconstruction.
Expert IDs and ownership are supplied by the model plan. Payload pointer tables
preserve each expert's codes, scale groups and per-projection global metadata.
"""

from __future__ import annotations

from cubic_serve.ops.cuda.startup import weight_bound_calibration_key

import math
from collections.abc import Callable
from typing import Any, cast

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor

from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.cuda.activation import softmax_type
from cubic_serve.ops.cuda.compiled_artifact import bind_tensor
from cubic_serve.ops.cuda.launch_cost import select_launch
from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
from cubic_serve.ops.cuda.moe.fused import (
    _compact_local_routes,
    _count_local_routes,
    _prefix_local_routes,
    _route_combine,
    _scatter_local_routes,
    _sort_and_normalize_topk,
    _stable_topk_logits,
    _stable_topk_routes,
)
from cubic_serve.ops.cuda.moe.route_completion import RouteCompletionPlanner
from cubic_serve.ops.cuda.pointwise import BF16PackedSwiGLUPrimitive
from cubic_serve.ops.strategies import (
    MoEParameters,
    PackedMatmulParameters,
    launch_parameters,
    operator_strategy_scope,
)
from cubic_serve.quantization.api import RoutedRows
from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.quantization.common.kernels import plain_compute
from cubic_serve.quantization.common.policy import placement_candidates
from cubic_serve.quantization.registry import get_extension, require_compute_mode


@triton.jit  # type: ignore[untyped-decorator]
def _zero_counts(counts, cursors, E: tl.constexpr, BLOCK: tl.constexpr):  # type: ignore[no-untyped-def]
    i = tl.arange(0, BLOCK)
    tl.store(counts + i, 0, i < E)
    tl.store(cursors + i, 0, i < E)


def _bind_route_selection(
    logits: Tensor,
    activity: Tensor,
    active_rows: Tensor,
    top_logits: Tensor,
    top_ids: Tensor,
    route_ids: Tensor,
    route_weights: Tensor,
    counts: Tensor,
    cursors: Tensor,
    *,
    choices: dict[tuple[object, ...], int],
    initialize_counts: bool = True,
) -> tuple[_PackedLaunch, ...]:
    """Choose complete routing cost before capture, preserving every output bit.

    Private idempotent replays amortize graph-launch timing noise. Logits are
    producer outputs immediately consumed by routing; do not evict them as if
    they were streamed checkpoint weights. No request operand or RNG is touched.
    """
    rows, experts = logits.shape
    top_k, local = top_ids.shape[1], counts.numel()
    policy = launch_parameters()
    activation = softmax_type(logits.dtype, logits.device)
    key = (
        logits.device,
        logits.dtype,
        rows,
        experts,
        top_k,
        local,
        str(activation),
        policy.max_warps,
        policy.max_candidates,
        policy.calibration_samples,
        policy.micro_calibration_replays,
        initialize_counts,
    )
    targets = (top_logits, top_ids, route_ids, route_weights, counts, cursors)

    def compile_prefix(warps: int, operands: tuple[Tensor, ...]) -> tuple[_PackedLaunch, ...]:
        source, active, limit, *destinations = operands
        compile = _PackedLaunch.compile
        if warps == 0:
            calls = (
                compile(
                    _stable_topk_logits,
                    (source, active, limit, *destinations[:2]),
                    (experts, top_k, triton.next_power_of_2(experts)),
                    (rows,),
                    {},
                ),
                compile(
                    _sort_and_normalize_topk,
                    tuple(destinations[:4]),
                    (top_k, triton.next_power_of_2(top_k), activation),
                    (rows,),
                    {},
                ),
            )
            return calls + ((compile(
                    _zero_counts,
                    tuple(destinations[4:]),
                    (local, triton.next_power_of_2(local)),
                    (1,),
                    {},
                ),) if initialize_counts else ())
        return (
            compile(
                _stable_topk_routes,
                operands,
                (
                    experts,
                    top_k,
                    local,
                    triton.next_power_of_2(experts),
                    triton.next_power_of_2(top_k),
                    triton.next_power_of_2(local),
                    activation,
                    initialize_counts,
                ),
                (rows,),
                {"num_warps": warps},
            ),
        )

    if key not in choices:
        generator = torch.Generator(device=logits.device).manual_seed(137)
        private_logits = torch.randn(
            logits.shape, device=logits.device, dtype=logits.dtype, generator=generator
        )
        private_targets = tuple(torch.empty_like(t) for t in targets)
        private = (
            private_logits,
            torch.ones_like(activity),
            torch.full_like(active_rows, rows),
            *private_targets,
        )
        selected: PackedOperatorBinding | None = None

        def bind(warps: int) -> None:
            nonlocal selected
            selected = PackedOperatorBinding(private_targets[3], compile_prefix(warps, private))

        def launch() -> None:
            assert selected is not None
            for _ in range(policy.micro_calibration_replays):
                selected.launch()

        candidates = (0, *(w for w in (1, 2, 4, 8) if w <= policy.max_warps))[
            : policy.max_candidates
        ]
        choices[key] = select_launch(
            candidates,
            device=logits.device,
            bind=bind,
            launch=launch,
            snapshot=lambda: private_targets if initialize_counts else private_targets[:4],
            cache_key=weight_bound_calibration_key(("moe-route-prefix", key)),
        )
    return compile_prefix(choices[key], (logits, activity, active_rows, *targets))


@triton.jit  # type: ignore[untyped-decorator]
def _routed_mm(  # type: ignore[no-untyped-def]
    X,
    XS,
    PTRS,
    COUNTS,
    STARTS,
    BLOCK_STARTS,
    TOTAL,
    ROWS,
    Y,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    OUTPUT_OFFSET: tl.constexpr,
    ROUTED: tl.constexpr,
    CONFIG: tl.constexpr,
    TYPES: tl.constexpr,
    COMPUTE: tl.constexpr,
    PROJECTIONS: tl.constexpr = 1,
):
    block = tl.program_id(0)
    if block >= tl.load(TOTAL):
        return
    ids = tl.arange(0, triton.next_power_of_2(E))
    counts = tl.load(COUNTS + ids, ids < E, 0)
    starts = tl.load(BLOCK_STARTS + ids, ids < E, 0)
    expert = tl.sum(((ids < E) & (block >= starts + tl.cdiv(counts, BM))).to(tl.int32), 0)
    if expert >= E:
        return
    offset = (block - tl.load(BLOCK_STARTS + expert)) * BM + tl.arange(0, BM)
    valid = offset < tl.load(COUNTS + expert)
    routes = tl.load(ROWS + tl.load(STARTS + expert) + offset, valid, 0)
    input_rows = routes if ROUTED else routes // TOP_K
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    projection = tl.program_id(2) if PROJECTIONS > 1 else 0
    pointers = ()  # type: tuple[Any, ...]
    for i in tl.static_range(len(TYPES)):
        pointers += (
            tl.load(PTRS + (projection * E + expert) * len(TYPES) + i).to(
                tl.pointer_type(TYPES[i])
            ),
        )
    accumulator = COMPUTE(X, XS, pointers, input_rows, valid, n, N, K, BM, BN, BK, CONFIG)
    tl.store(
        Y + routes[:, None] * OUTPUT_STRIDE + OUTPUT_OFFSET + projection * N + n[None, :],
        accumulator,
        valid[:, None] & (n[None, :] < N),
    )


def _type(tensor: Tensor) -> Any:
    return {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
        torch.int32: tl.int32,
        torch.int8: tl.int8,
        torch.uint8: tl.uint8,
        torch.uint16: tl.uint16,
    }[tensor.dtype]


class PackedRoutedExpertsPrimitive:
    """One sealed local expert slice; arithmetic strategy is separate from plain MoE."""

    def __init__(
        self,
        hidden: Tensor,
        logits: Tensor,
        activity: Tensor,
        active_rows: Tensor,
        gate: tuple[Tensor | QuantizedMatrix, ...],
        up: tuple[Tensor | QuantizedMatrix, ...],
        down: tuple[Tensor | QuantizedMatrix, ...],
        output: Tensor,
        *,
        first_expert: int,
        top_k: int,
        operators: PackedLinearOperators,
        reduce_routes: Callable[[Tensor], None] | None = None,
        route_completion: RouteCompletionPlanner | None = None,
        route_workspace: Tensor | None = None,
        projection_scratch: Callable[[Tensor], Tensor] | None = None,
        workspace: Callable[[str, tuple[int, ...], torch.dtype], Tensor] | None = None,
    ) -> None:
        if (route_completion is not None and route_completion.selected
                and reduce_routes is not None):
            raise ValueError("route completion must replace, not follow, the legacy route sum")
        self._reduce_routes = reduce_routes
        # Complete-K activation scales belong to the expert tensor partition,
        # independently of which transport completes its partial route outputs.
        row_parallel_down = reduce_routes is not None or (
            route_completion is not None and route_completion.selected
        )
        # Both route completion and complete-K input scales belong to this
        # dependency boundary. A concurrent shared branch may itself use TP;
        # it must not race another collective from this routed branch.
        self.has_collectives = row_parallel_down
        m, h = hidden.shape
        e = len(gate)
        if not e or len(up) != e or len(down) != e or not 0 < top_k <= logits.shape[1]:
            raise ValueError("packed expert ownership or top-k is invalid")
        intermediate = gate[0].shape[0]
        if (
            first_expert < 0
            or first_expert + e > logits.shape[1]
            or logits.shape[0] != m
            or output.shape != hidden.shape
            or hidden.dtype not in (torch.float16, torch.bfloat16)
            or logits.dtype != hidden.dtype
            or output.dtype != hidden.dtype
            or activity.shape != (m,)
            or active_rows.shape != (1,)
            or activity.dtype != torch.int32
            or active_rows.dtype != torch.int32
            or not hidden.is_cuda
            or any(
                t.device != hidden.device or not t.is_contiguous()
                for t in (hidden, logits, activity, active_rows, output)
            )
        ):
            raise ValueError("packed expert activation ABI is invalid")
        for weights, shape in (
            (gate, (intermediate, h)),
            (up, (intermediate, h)),
            (down, (h, intermediate)),
        ):
            if any(w.shape != shape for w in weights):
                raise ValueError("local expert projections have inconsistent logical geometry")
        if operators.config.compute_mode != "wna16":
            for encoding in {weight.encoding for group in (gate, up, down)
                             for weight in group if isinstance(weight, QuantizedMatrix)}:
                require_compute_mode(encoding, "moe", operators.config.compute_mode)
        self._operands = hidden, logits, activity, active_rows, output
        self._bindings = tuple(bind_tensor(t) for t in self._operands)
        self._weights = gate, up, down
        self._weight_tensors = tuple(
            t
            for group in self._weights
            for w in group
            for t in ((w,) if isinstance(w, Tensor) else tuple(w.payload.values()))
        )
        if any(
            (
                w.device != hidden.device
                if isinstance(w, Tensor)
                else any(
                    t.device != hidden.device and t.device.type != "cpu" for t in w.payload.values()
                )
            )
            for group in self._weights
            for w in group
        ):
            raise ValueError("packed expert payloads must reside on the input device")

        pooled_storage: set[int] = set()
        operand_storage = ({t.untyped_storage().data_ptr()
                            for t in (*self._operands, *self._weight_tensors)
                            if t.device == hidden.device} if workspace is not None else set())

        def empty(*shape: int, dtype: torch.dtype = hidden.dtype,
                  role: str | None = None) -> Tensor:
            if workspace is None or role is None:
                return torch.empty(shape, dtype=dtype, device=hidden.device)
            value = workspace(role, shape, dtype)
            if (value.shape != shape or value.dtype != dtype or value.device != hidden.device
                    or not value.is_contiguous()):
                raise ValueError("expert workspace differs from its declared ABI")
            pointer = value.untyped_storage().data_ptr()
            if pointer in operand_storage or pointer in pooled_storage:
                raise ValueError("expert workspace aliases a live operand or scratch role")
            pooled_storage.add(pointer)
            return value

        self.top_logits = empty(m, top_k, dtype=torch.float32)
        self.top_ids = empty(m, top_k, dtype=torch.int64)
        self.route_ids = empty(m, top_k, dtype=torch.int64)
        self.route_weights = empty(m, top_k)
        self.counts = empty(e, dtype=torch.int32)
        self.starts = empty(e, dtype=torch.int32)
        self.block_starts = empty(e, dtype=torch.int32)
        self.cursors = empty(e, dtype=torch.int32)
        self.total = empty(1, dtype=torch.int32)
        self.row_map = empty(m * top_k, dtype=torch.int32)
        # These activations are fully consumed by launch(), including route
        # completion. Serial callers may reuse them after that boundary; route
        # metadata and native projection scratch retain their own contracts.
        self.packed = empty(m * top_k, 2 * intermediate, role="gate_up")
        self.activated = empty(m * top_k, intermediate, role="activated")
        if route_workspace is not None:
            if (route_workspace.shape != (m, top_k, h)
                    or route_workspace.dtype != hidden.dtype
                    or route_workspace.device != hidden.device
                    or not route_workspace.is_contiguous()):
                raise ValueError("external route workspace differs from the routed ABI")
            start = route_workspace.data_ptr()
            end = start + route_workspace.numel() * route_workspace.element_size()
            if any(t.device == hidden.device and start < t.data_ptr() + t.numel()
                   * t.element_size() and t.data_ptr() < end
                   for t in (*self._operands, *self._weight_tensors)):
                raise ValueError("external route workspace aliases a routed operand")
            if route_workspace.untyped_storage().data_ptr() in pooled_storage:
                raise ValueError("external route workspace aliases expert activation scratch")
        self.routes = (empty(m, top_k, h, role="routes")
                       if route_workspace is None else route_workspace)
        self.swiglu = BF16PackedSwiGLUPrimitive(self.packed, self.activated)
        compile = _PackedLaunch.compile
        # A lane owns at most one route in the single-CTA sort; limit the
        # histogram scan to one warp's counters per lane. These register/work
        # bounds depend on metadata geometry and the configured thread budget,
        # not public concurrency, model identity, TP count or SM generation.
        route_warps = triton.next_power_of_2(triton.cdiv(m * top_k, 32))
        compact = (route_warps <= launch_parameters().max_warps
                   and triton.next_power_of_2(e) <= route_warps * 32 * 32)
        calls = [
            *_bind_route_selection(
                logits, activity, active_rows, self.top_logits, self.top_ids,
                self.route_ids, self.route_weights, self.counts, self.cursors,
                choices=operators._routing_choices,
                initialize_counts=not compact,
            ),
        ]
        if not compact:
            calls.append(compile(
                _count_local_routes,
                (self.route_ids, self.counts),
                (first_expert, e, m * top_k, 256),
                (triton.cdiv(m * top_k, 256),),
                {},
            ))
        self.routing = PackedOperatorBinding(output, tuple(calls))
        self.scatter = PackedOperatorBinding(
            output,
            () if compact else (
                compile(
                    _scatter_local_routes,
                    (self.route_ids, self.starts, self.cursors, self.row_map),
                    (first_expert, e, m * top_k, 256),
                    (triton.cdiv(m * top_k, 256),),
                    {},
                ),
            ),
        )
        self.projections: list[
            tuple[int, PackedOperatorBinding | None, PackedOperatorBinding | None]
        ] = []
        gate_launch_data = None
        self.prefixes: dict[int, PackedOperatorBinding] = {}
        self._metadata_tensors: list[Tensor] = []
        self._pointer_tables = []
        prep_cache: dict[int, tuple[Tensor, Tensor, PackedOperatorBinding]] = {}
        for group in (gate, up, down):
            exemplar = group[0]
            n, k = exemplar.shape
            if any(
                type(w) is not type(exemplar)
                or (
                    isinstance(w, QuantizedMatrix)
                    and (
                        not isinstance(exemplar, QuantizedMatrix)
                        or w.encoding != exemplar.encoding
                        or w.partitions != exemplar.partitions
                    )
                )
                for w in group
            ):
                raise ValueError(
                    "each routed projection requires one declared encoding across its local experts"
                )
        native_candidates: dict[
            str, tuple[int, PackedOperatorBinding | None, PackedOperatorBinding | None]
        ] = {}
        native_scratch: tuple[Tensor, ...] = ()

        def reuse_projection(binding: PackedOperatorBinding) -> None:
            # Format-owned opt-in promises complete consumption by this
            # binding. The caller owns the serial lifetime and graph lease;
            # route metadata and exported projection outputs stay private.
            if projection_scratch is not None and binding.reusable_scratch:
                binding.rebind_private_scratch(projection_scratch(binding.scratch[0]))

        if (
            isinstance(gate[0], QuantizedMatrix)
            and isinstance(up[0], QuantizedMatrix)
            and gate[0].encoding.format == up[0].encoding.format
        ):
            extension = get_extension(gate[0].encoding.format)
            factory = getattr(extension.module("moe"), "bind_grouped", None)
            if factory is not None:
                native_params = operators.config.resolve(
                    extension.family("moe"),
                    device=operators.device_identity,
                    model=operators.model_identity,
                    n=intermediate,
                    k=h,
                    experts=e,
                    top_k=top_k,
                )
                candidate = factory(
                    operators,
                    hidden,
                    gate,
                    up,
                    self.packed,
                    RoutedRows(
                        self.counts, self.starts, self.block_starts, self.total, self.row_map,
                        m * top_k / logits.shape[1],
                        top_k,
                    ),
                    top_k,
                    native_params,
                )
                if candidate is not None:
                    bm, binding = candidate
                    reuse_projection(binding)
                    if bm not in self.prefixes:
                        prefix = compile(
                            _prefix_local_routes,
                            (self.counts, self.starts, self.block_starts, self.total),
                            (e, bm, triton.next_power_of_2(e)),
                            (1,),
                            {},
                        )
                        self.prefixes[bm] = PackedOperatorBinding(output, (prefix,))
                    native_candidates["gate"] = (bm, None, binding)
                    native_candidates["up"] = (bm, None, None)
                    native_scratch = binding.scratch
        # A format may bind a native route-local projection independently of
        # grouped gate/up. The caller retains expert ownership and row metadata.
        if isinstance(down[0], QuantizedMatrix):
            extension = get_extension(down[0].encoding.format)
            factory = getattr(extension.module("moe"), "bind_routed", None)
            if factory is not None:
                native_params = operators.config.resolve(
                    extension.family("moe"),
                    device=operators.device_identity,
                    model=operators.model_identity,
                    n=h,
                    k=intermediate,
                    experts=e,
                    top_k=top_k,
                )
                candidate = factory(
                    operators,
                    self.activated,
                    down,
                    self.routes,
                    RoutedRows(
                        self.counts, self.starts, self.block_starts, self.total, self.row_map,
                        m * top_k / logits.shape[1],
                        top_k,
                    ),
                    native_params,
                )
                if candidate is not None:
                    bm, binding = candidate
                    reuse_projection(binding)
                    if bm not in self.prefixes:
                        prefix = compile(
                            _prefix_local_routes,
                            (self.counts, self.starts, self.block_starts, self.total),
                            (e, bm, triton.next_power_of_2(e)),
                            (1,),
                            {},
                        )
                        self.prefixes[bm] = PackedOperatorBinding(output, (prefix,))
                    native_candidates["down"] = (bm, None, binding)
                    native_scratch = (*native_scratch, *binding.scratch)
        # An INT A8 gate/up pair can carry different checkpoint encodings.
        # Bind each to native Marlin and scatter only its final projection
        # values into the existing interleaved gate/up output.
        for role, group, offset in (("gate", gate, 0), ("up", up, intermediate)):
            if role in native_candidates or not isinstance(group[0], QuantizedMatrix):
                continue
            if operators.config.compute_mode != "wna8":
                continue
            extension = get_extension(group[0].encoding.format)
            factory = getattr(extension.module("moe"), "bind_routed", None)
            if factory is None:
                raise ValueError(f"{role} WNA8 has no native routed kernel")
            native_params = operators.config.resolve(
                extension.family("moe"), device=operators.device_identity,
                model=operators.model_identity, n=intermediate, k=h,
                experts=e, top_k=top_k,
            )
            temporary = torch.empty(
                (m * top_k, intermediate), device=hidden.device, dtype=hidden.dtype
            )
            candidate = factory(
                operators, hidden, group, temporary,
                RoutedRows(
                    self.counts, self.starts, self.block_starts, self.total, self.row_map,
                    m * top_k / logits.shape[1], top_k,
                ), native_params,
            )
            if candidate is None:
                raise ValueError(f"{role} WNA8 native routed kernel is unavailable")
            bm, binding = candidate
            if bm not in self.prefixes:
                prefix = compile(
                    _prefix_local_routes,
                    (self.counts, self.starts, self.block_starts, self.total),
                    (e, bm, triton.next_power_of_2(e)), (1,), {},
                )
                self.prefixes[bm] = PackedOperatorBinding(output, (prefix,))
            from cubic_serve.quantization.common.kernels import _copy_projection_rows

            scatter = _PackedLaunch.compile(
                _copy_projection_rows, (temporary, self.packed),
                (m * top_k * intermediate, intermediate, intermediate, 256,
                 2 * intermediate, offset),
                (triton.cdiv(m * top_k * intermediate, 256),),
                dict(num_warps=4),
            )
            routed = PackedOperatorBinding(
                self.packed, (*binding._launches, scatter),
                scratch=(*binding.scratch, temporary),
            )
            native_candidates[role] = (bm, None, routed)
            native_scratch = (*native_scratch, *routed.scratch)
        for role, group, x, y, stride, offset in (
            ("gate", gate, hidden, self.packed, 2 * intermediate, 0),
            ("up", up, hidden, self.packed, 2 * intermediate, intermediate),
            ("down", down, self.activated, self.routes, h, 0),
        ):
            exemplar = group[0]
            n, k = exemplar.shape
            if role in native_candidates:
                self.projections.append(native_candidates[role])
                continue
            prep = None
            arrays: list[tuple[Tensor, ...]]
            params: PackedMatmulParameters
            if isinstance(exemplar, Tensor):
                plain_group = cast(tuple[Tensor, ...], group)
                if any(w.dtype != hidden.dtype for w in plain_group):
                    raise ValueError("plain expert weights must follow execution dtype")
                plain_params = operators.config.resolve(
                    "moe",
                    device=operators.device_identity,
                    model=operators.model_identity,
                    hidden=h,
                    intermediate=intermediate,
                    experts=e,
                    global_experts=logits.shape[1],
                    top_k=top_k,
                )
                if not isinstance(plain_params, MoEParameters):
                    raise TypeError("plain expert strategy differs")
                params = PackedMatmulParameters(
                    plain_params.block_m,
                    plain_params.compact_down_block_n if role == "down" else plain_params.block_n,
                    plain_params.block_k,
                    plain_params.gemm_warps,
                    plain_params.gemm_stages,
                )
                compute_kernel = plain_compute
                family = "moe"
                arrays = [(w,) for w in plain_group]
                math_constants = ()
            else:
                packed_group = tuple(
                    operators.materialize_weight(w, hidden.device)
                    for w in cast(tuple[QuantizedMatrix, ...], group)
                )
                extension = get_extension(exemplar.encoding.format)
                family = extension.family("moe")
                compute_kernel = extension.module("moe").compute
                resolved = operators.config.resolve(
                    family,
                    device=operators.device_identity,
                    model=operators.model_identity,
                    n=n,
                    k=k,
                    experts=e,
                    top_k=top_k,
                )
                if not isinstance(resolved, PackedMatmulParameters):
                    raise TypeError("packed expert strategy differs")
                params = resolved
                operands = extension.module("moe").operands(operators, packed_group)
                arrays = list(operands.tensors)
                math_constants = operands.constants
                compute_kernel = operands.compute
                enc = exemplar.encoding
                x, xs, prep, act = extension.module("moe").prepare(
                    operators,
                    x,
                    enc,
                    role == "down" and row_parallel_down,
                    prep_cache,
                )
            if isinstance(exemplar, Tensor):
                xs = hidden
            table = torch.tensor(
                [[t.data_ptr() for t in row] for row in arrays],
                dtype=torch.int64,
                device=hidden.device,
            )
            self._metadata_tensors.extend(t for row in arrays for t in row)
            self._pointer_tables.append(table)
            with operator_strategy_scope(
                operators.config,
                device=operators.device_identity,
                model=operators.model_identity,
            ):
                params = self._select_parameters(
                    operators,
                    params,
                    family,
                    x,
                    xs,
                    table,
                    y,
                    n=n,
                    k=k,
                    e=e,
                    global_experts=logits.shape[1],
                    top_k=top_k,
                    m=m,
                    stride=stride,
                    offset=offset,
                    routed=role == "down",
                    math_constants=math_constants,
                    types=tuple(_type(t) for t in arrays[0]),
                    compute=compute_kernel,
                )
            bm, bn, bk = params.fixed_block_m, params.fixed_block_n, params.fixed_block_k
            if bm not in self.prefixes:
                self.prefixes[bm] = PackedOperatorBinding(
                    output,
                    (
                        compile(
                            _prefix_local_routes,
                            (self.counts, self.starts, self.block_starts, self.total),
                            (e, bm, triton.next_power_of_2(e)),
                            (1,),
                            {},
                        ),
                    ),
                )
            constants = (
                e,
                top_k,
                n,
                k,
                bm,
                bn,
                bk,
                stride,
                offset,
                role == "down",
                math_constants,
                tuple(_type(t) for t in arrays[0]),
                compute_kernel,
            )
            launch = compile(
                _routed_mm,
                (
                    x,
                    xs,
                    table,
                    self.counts,
                    self.starts,
                    self.block_starts,
                    self.total,
                    self.row_map,
                    y,
                ),
                constants,
                # Each nonempty block owns at least one route. Large expert
                # inventories must not force E empty CTAs for a tiny batch.
                (min(m * top_k, triton.cdiv(m * top_k, bm) + e), triton.cdiv(n, bn)),
                {
                    "num_warps": params.fixed_warps,
                    "num_stages": params.fixed_stages,
                    "enable_fp_fusion": False,
                },
            )
            self.projections.append((bm, prep, PackedOperatorBinding(output, (launch,))))
            if role == "gate":
                gate_launch_data = (x, xs, table, constants, params, prep)
            elif role == "up" and gate_launch_data is not None:
                gx, gs, gt, gc, gp, gprep = gate_launch_data
                # Group compatible projection axes, never logical experts.
                # Same CTA math, private output columns, and no reconstructed
                # weight copy. Mixed formats/plans remain independent launches.
                if (
                    getattr(params, "grouped_projections", False)
                    and gp == params
                    and x is gx
                    and xs is gs
                    and prep is gprep
                    and gc[:8] == constants[:8]
                    and gc[9:] == constants[9:]
                    and gc[8] == 0
                    and offset == n
                    and stride == 2 * n
                ):
                    table = torch.cat((gt, table), dim=0)
                    self._pointer_tables[-2:] = [table]
                    grouped = compile(
                        _routed_mm,
                        (
                            x,
                            xs,
                            table,
                            self.counts,
                            self.starts,
                            self.block_starts,
                            self.total,
                            self.row_map,
                            y,
                        ),
                        (*gc, 2),
                        (min(m * top_k, triton.cdiv(m * top_k, bm) + e), triton.cdiv(n, bn), 2),
                        {
                            "num_warps": params.fixed_warps,
                            "num_stages": params.fixed_stages,
                            "enable_fp_fusion": False,
                        },
                    )
                    self.projections[0] = (bm, prep, PackedOperatorBinding(output, (grouped,)))
                    self.projections[1] = (bm, None, None)
        if compact:
            self.prefixes = {
                bm: PackedOperatorBinding(output, (compile(
                    _compact_local_routes,
                    (self.route_ids, self.counts, self.starts, self.block_starts,
                     self.total, self.row_map),
                    (first_expert, e, m * top_k, bm, triton.next_power_of_2(e),
                     triton.next_power_of_2(m * top_k)),
                    (1,), {"num_warps": route_warps},
                ),))
                for bm in self.prefixes
            }
        completed = (None if route_completion is None else route_completion.bind(
            self.routes, self.route_ids, self.route_weights, output, first_expert, e,
        ))
        self.combine = completed if completed is not None else PackedOperatorBinding(
            output,
            (
                compile(
                    _route_combine,
                    (self.routes, self.route_ids, self.route_weights, output),
                    (first_expert, e, m * h, h, top_k, 256),
                    (triton.cdiv(m * h, 256),),
                    {"enable_fp_fusion": False},
                ),
            ),
        )
        self._needs_route_zero = reduce_routes is not None or completed is not None
        self.scratch = (
            self.top_logits,
            self.top_ids,
            self.route_ids,
            self.route_weights,
            self.counts,
            self.starts,
            self.block_starts,
            self.cursors,
            self.total,
            self.row_map,
            self.packed,
            self.activated,
            self.routes,
            *native_scratch,
            *(completed.scratch if completed is not None else ()),
            *(t for _, _, prep in prep_cache.values() for t in prep.scratch),
        )

        # Pointer-table kernels consume these actual materialized tensors.
        # Native bindings own and validate their canonical code/metadata operands
        # themselves. Checkpoint source tensors are construction inputs, not
        # launch operands after repacking, and remain owned by self._weights.
        self._weight_tensors = tuple(self._metadata_tensors)
        self._weight_bindings = tuple(bind_tensor(t) for t in self._weight_tensors)
        self._scratch_bindings = tuple(bind_tensor(t) for t in self.scratch)
        self._pointer_bindings = tuple(bind_tensor(t) for t in self._pointer_tables)

    @staticmethod
    def _select_parameters(
        operators: PackedLinearOperators,
        params: PackedMatmulParameters,
        family: str,
        x: Tensor,
        xs: Tensor,
        table: Tensor,
        y: Tensor,
        *,
        n: int,
        k: int,
        e: int,
        global_experts: int,
        top_k: int,
        m: int,
        stride: int,
        offset: int,
        routed: bool,
        math_constants: tuple[Any, ...],
        types: tuple[Any, ...],
        compute: Any,
    ) -> PackedMatmulParameters:
        """Price packed arithmetic on private routes, never reassign real experts.

        Expected local route population bounds geometry, as in ordinary MoE.
        Actual format decoding and compiled layout participate in cold timings.
        The per-projection K, scales, activation codes and output stride stay fixed.
        """
        expected = max(1, math.ceil(m * top_k * e / global_experts))
        groups = min(e, expected)
        scope = dict(experts=e, top_k=top_k)
        override_fields = ("fixed_block_m", "fixed_block_n", "fixed_warps", "fixed_stages")
        if family == "moe":
            scope.update(
                hidden=n if routed else k,
                intermediate=k if routed else n,
                global_experts=global_experts,
            )
            override_fields = (
                "block_m",
                "compact_down_block_n" if routed else "block_n",
                "gemm_warps",
                "gemm_stages",
            )
        candidates = placement_candidates(
            params,
            family=family,
            device=x.device,
            rows=math.ceil(expected / groups),
            columns=n,
            reduction=k,
            element_bytes=y.element_size(),
            groups=groups,
            override_fields=override_fields,
            **scope,
        )
        key = (
            "routed",
            x.device,
            x.dtype,
            y.dtype,
            m,
            n,
            k,
            e,
            global_experts,
            top_k,
            stride,
            offset,
            routed,
            math_constants,
            types,
            compute.cache_key,
            tuple(candidates),
        )
        cache = operators._routed_choices
        if key in cache:
            return cache[key]
        if len(candidates) == 1:
            cache[key] = params
            return params
        generator = torch.Generator(device=x.device).manual_seed(0)
        sample = (
            torch.randint(1, 32, x.shape, dtype=x.dtype, device=x.device, generator=generator)
            if not x.is_floating_point()
            else torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator).mul_(
                0.01
            )
        )
        scales = torch.ones_like(xs)
        output = torch.zeros_like(y)
        # A deterministic local population fixture, independent of observed traffic.
        counts_cpu = [expected // e + int(i < expected % e) for i in range(e)]
        counts = torch.tensor(counts_cpu, dtype=torch.int32, device=x.device)
        starts = torch.empty_like(counts)
        block_starts = torch.empty_like(counts)
        total = torch.empty(1, dtype=torch.int32, device=x.device)
        rows = torch.arange(expected, dtype=torch.int32, device=x.device)
        selected = None

        def bind(point: PackedMatmulParameters) -> None:
            nonlocal selected
            bm, bn, bk = point.fixed_block_m, point.fixed_block_n, point.fixed_block_k
            prefix = _PackedLaunch.compile(
                _prefix_local_routes,
                (counts, starts, block_starts, total),
                (e, bm, triton.next_power_of_2(e)),
                (1,),
                {},
            )
            call = _PackedLaunch.compile(
                _routed_mm,
                (sample, scales, table, counts, starts, block_starts, total, rows, output),
                (
                    e,
                    top_k,
                    n,
                    k,
                    bm,
                    bn,
                    bk,
                    stride,
                    offset,
                    routed,
                    math_constants,
                    types,
                    compute,
                ),
                (min(m * top_k, triton.cdiv(m * top_k, bm) + e), triton.cdiv(n, bn)),
                dict(
                    num_warps=point.fixed_warps,
                    num_stages=point.fixed_stages,
                    enable_fp_fusion=False,
                ),
            )
            selected = PackedOperatorBinding(output, (prefix, call))

        def launch() -> None:
            assert selected is not None
            selected.launch()

        def prepare() -> None:
            output.zero_()

        chosen = select_launch(
            candidates,
            device=x.device,
            bind=bind,
            launch=launch,
            prepare=prepare,
            snapshot=lambda: (output,),
            cache_key=weight_bound_calibration_key(("packed-moe", key)),
        )
        cache[key] = chosen
        return chosen

    def launch(self) -> None:
        if (
            tuple(bind_tensor(t) for t in self._operands) != self._bindings
            or any(not binding.matches(tensor) for binding, tensor in
                   zip(self._weight_bindings, self._weight_tensors, strict=True))
            or tuple(bind_tensor(t) for t in self.scratch) != self._scratch_bindings
            or tuple(bind_tensor(t) for t in self._pointer_tables) != self._pointer_bindings
        ):
            raise ValueError("routed expert operands changed after startup binding")
        # Reject stale direct operands before route producers or collectives
        # execute, including native prepared weights absent from pointer tables.
        for _, prep, gemm in self.projections:
            for binding in (prep, gemm):
                if binding is not None:
                    binding.validate_operands()
        if self._needs_route_zero:
            self.routes.zero_()
        self.routing.launch()
        previous_bm = None
        seen_preps = set()
        for index, (bm, prep, gemm) in enumerate(self.projections):
            if index == 2:
                self.swiglu.launch(self.packed, self.activated)
            if previous_bm != bm:
                self.prefixes[bm].launch()
                previous_bm = bm
            if index == 0:
                self.scatter.launch()
            if prep is not None and id(prep) not in seen_preps:
                prep.launch()
                seen_preps.add(id(prep))
            if gemm is not None:
                gemm.launch()
        if self._reduce_routes is not None:
            # All ranks have identical route slots in TP-only expert execution.
            self._reduce_routes(self.routes)
        self.combine.launch()
