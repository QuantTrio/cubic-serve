"""Deferred cross-M identity witnesses anchored to locally witnessed M1.

Temporary captures retain only immutable stream-ordered snapshots. Rank schedule,
reference comparisons and write-set checks happen after the complete forward.
Nothing in this module runs after serving readiness. The anchor uses candidate
boundary continuation, not an independently advancing full-model trajectory.
"""

from __future__ import annotations

from typing_extensions import TypeAliasType

import gc
import logging
import os
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist

from cubic_serve.executor.text_serving import FlashNextDecodeStage
from cubic_serve.guardrail.compare import tensor_digest
from cubic_serve.models.qwen4_exp.model import Qwen4ExpOperationBinder
from cubic_serve.ops.cuda.attention.sparse_indexer import IndexerPreparation
from cubic_serve.ops.cuda.collective import FixedRankBF16SumPrimitive
from cubic_serve.ops.cuda.moe.quantized import PackedRoutedExpertsPrimitive
from cubic_serve.quantization.common.small_rows_cost import (
    row_arithmetic_plan,
    row_arithmetic_scope,
)
from cubic_serve.runtime.text_qualification import boundary_snapshot, state_audit_horizon

from .text import audit_original, preserve_observer_collections

Boundary = TypeAliasType("Boundary", dict[str, str])
RowWitness = TypeAliasType("RowWitness", tuple[Boundary, tuple[tuple[str, str], ...]])


class _NumericalAttributionUnavailable(ValueError):
    """Different witnessed values still require their own numerical proof."""


def _attention_arithmetic_plan(body: Any) -> tuple[tuple[int, int, int], ...]:
    """Read the bound partial-reduction geometry, excluding independent rows."""
    buffers = getattr(getattr(body, "helper", None), "_buffers", {})
    result = set()
    for (role, shape, dtype) in buffers:
        if role != "qsa.attention_partial":
            continue
        if (len(shape) != 4 or shape[0] != body.captured_m or dtype is not torch.float32
                or any(type(value) is not int or value <= 0 for value in shape)
                or shape[2] & (shape[2] - 1)):
            raise ValueError("attention arithmetic partial inventory is invalid")
        result.add(tuple(shape[1:]))
    return tuple(sorted(result))


@contextmanager
def _attention_arithmetic_scope(plan: Sequence[tuple[int, int, int]]) -> Iterator[None]:
    """Bind the witnessed partition to the reference row; never reprice C1."""
    if not plan:
        yield
        return
    if any(len(item) != 3 or any(type(value) is not int or value <= 0 for value in item)
           or item[1] & (item[1] - 1) for item in plan):
        raise ValueError("attention arithmetic plan is invalid")
    splits = {item[1] for item in plan}
    if len(splits) != 1:
        raise ValueError("attention arithmetic requires an unambiguous partition binding")
    # The typed graph's QSA layers share this partition geometry. Both the
    # producer and merge consume the same bound workspace count. This override
    # exists only while constructing numerical witnesses, never during serving.
    with patch("cubic_serve.ops.cuda.attention.paged_decode.paged_gqa_kv_splits",
               return_value=next(iter(splits))):
        yield


def _require_m1_identity(witness: Boundary, wanted: Boundary, context: Any) -> None:
    if witness.keys() != wanted.keys():
        raise ValueError("locally witnessed M1 boundary inventory differs")
    differences = [name for name in witness if witness[name] != wanted[name]]
    if differences:
        # Different reduction plans may change stored values legitimately.
        # Hashes cannot supply the missing actual-operand numerical proof.
        raise _NumericalAttributionUnavailable(
            f"cross-M numerical attribution unavailable: {context}, fields={differences}"
        )


def _initial_bytes_equal(value: torch.Tensor, initial: Any) -> bool:
    expected = torch.tensor([initial], dtype=value.dtype, device=value.device).view(torch.uint8)
    actual = value.contiguous().view(torch.uint8).reshape(-1, value.element_size())
    return bool((actual == expected).all())


def _coordinate(error: tuple[str, str] | None, group: Any) -> None:
    errors: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    failures = [item for item in errors if item is not None]
    if failures:
        if any(kind == "harness" for kind, _ in failures):
            raise RuntimeError(f"typed cross-M harness failure: {failures}")
        if any(kind == "execution" for kind, _ in failures):
            raise RuntimeError(f"typed cross-M execution failure: {failures}")
        raise AssertionError(f"typed cross-M numerical failure: {failures}")


@contextmanager
def _observe(
    events: list[tuple[str, torch.Tensor]], *, input_geometry: tuple[int, int] | None = None,
) -> Iterator[None]:
    old_sum = FixedRankBF16SumPrimitive.launch
    old_route = PackedRoutedExpertsPrimitive.launch
    old_bind = Qwen4ExpOperationBinder.bind
    old_index_finish = IndexerPreparation.finish
    scope = "unscoped"
    counts: dict[str, int] = {}
    input_scope = False

    def named(kind: str) -> str:
        index = counts.get(kind, 0)
        counts[kind] = index + 1
        return f"{scope}/{kind}/{index}"

    def bound(binder: Any, operation: str, weight_prefix: str, *, layer_id: int | None) -> Any:
        launch = old_bind(binder, operation, weight_prefix, layer_id=layer_id)

        def observed() -> None:
            nonlocal scope, counts, input_scope
            previous = scope, counts, input_scope
            scope, counts = f"{weight_prefix}/{operation}", {}
            input_scope = operation == "embedding"
            try:
                launch()
            finally:
                scope, counts, input_scope = previous

        return observed

    def summed(op: Any, value: torch.Tensor) -> None:
        result = old_sum(op, value)
        for part in op.input_views(value):
            # input_views preserves declared logical member order even when
            # one transport completes several independent fields together.
            if input_geometry is not None and (scope == "unscoped" or input_scope):
                rows, streams = input_geometry
                if part.shape[0] == rows * streams:
                    part = part.reshape(rows, streams, *part.shape[1:])
            if scope == "unscoped" and input_geometry is not None:
                index = counts.get("input_sum", 0)
                counts["input_sum"] = index + 1
                site = f"draft_input/sum/{index}"
            else:
                site = named("sum")
            events.append((site, part.detach().clone()))
        return result

    def routed(op: Any) -> None:
        result = old_route(op)
        events.extend(
            (named(name), getattr(op, name).detach().clone())
            for name in ("route_ids", "route_weights", "routes")
        )
        return result

    def index_finished(op: Any) -> None:
        result = old_index_finish(op)
        # Snapshot only. These are fully materialized row outputs: unlike the
        # owner-only normalized-key scratch they are defined on every active row.
        events.append((named("index_rotated_query"), op._finish.output.detach().clone()))
        events.append((named("index_pooled_key"), op.pooled.detach().clone()))
        return result

    with (
        patch.object(IndexerPreparation, "finish", index_finished),
        patch.object(FixedRankBF16SumPrimitive, "launch", summed),
        patch.object(PackedRoutedExpertsPrimitive, "launch", routed),
        patch.object(Qwen4ExpOperationBinder, "bind", bound),
    ):
        yield


def _intermediate_witness(
    events: Sequence[tuple[str, torch.Tensor]], stage: Any, physical_row: int,
    *, snapshot_rows: int | None = None,
) -> tuple[tuple[str, str], ...]:
    """Match semantic sites after rank schedule proof, never launch ordinals."""
    snapshots = dict(events)
    if len(snapshots) != len(events) or any(n.startswith("unscoped/") for n in snapshots):
        raise ValueError("duplicate or unscoped cross-M observer site")
    intermediate = []
    for name, value in events:
        extent = stage.captured_m if snapshot_rows is None else snapshot_rows
        if value.shape[0] != extent:
            raise ValueError(f"observer lacks an explicit row layout: {name}")
        logical_value = value[physical_row : physical_row + 1]
        prefix, kind, ordinal = name.rsplit("/", 2)
        if kind == "routes":
            ids = snapshots[f"{prefix}/route_ids/{ordinal}"][physical_row : physical_row + 1]
            local = stage.config.text.moe.expert_count // stage.topology.ep_size
            first = stage.topology.expert_coordinates(stage.rank)[0] * local
            owned = (ids >= first) & (ids < first + local)
            logical_value = torch.where(owned[..., None], logical_value, 0)
        intermediate.append((name, tensor_digest(logical_value)))
    # Coalesced route/shared completion can move a shared sum across the route
    # observer. This does not change either logical field or any value bit.
    return tuple(sorted(intermediate))


def _host_events(
    events: Sequence[tuple[str, torch.Tensor]], capacity: int, rows: Sequence[int],
) -> list[tuple[str, torch.Tensor]]:
    """Copy immutable completed-forward events once, in explicit row-map order.

    The caller must first prove every rank's event schedule. This is never an
    observer callback. Slicing bounds the transfer; indexing retains arbitrary
    permutations and sparse physical row maps without changing any value bits.
    """
    first, end = min(rows), max(rows) + 1
    if first < 0 or end > capacity:
        raise ValueError("observer physical row map exceeds its captured capacity")
    indices = torch.tensor([row - first for row in rows], dtype=torch.int64)
    result = []
    for name, value in events:
        if value.shape[0] != capacity:
            raise ValueError(f"observer lacks an explicit row layout: {name}")
        host = value.detach()[first:end].to("cpu", copy=True).index_select(0, indices)
        result.append((name, host))
    return result


def _replicated_witness_rows(stage, physical, labels, events):
    """Prove every repeated corpus row bitwise before reusing its CPU witness.

    Representatives still pass the independent original-weight oracle. This
    lossless equality proof touches every logical row and field after completed
    forward/schedule agreement; it never evaluates reference math in a callback.
    On a difference retain the original full per-row attribution path.
    """
    if len(physical) != len(labels) or not physical:
        raise ValueError("row-equivalence corpus map differs from physical rows")
    representatives, classes, mapping = [], {}, []
    for index, label in enumerate(labels):
        if label not in classes:
            classes[label] = len(representatives)
            representatives.append(index)
        mapping.append(classes[label])
    identity = (tuple(range(len(physical))), tuple(range(len(physical))))
    if len(representatives) == len(physical):
        return identity
    device, b, arena = stage.hidden.device, stage.buffers, stage.arena
    actual = torch.tensor(physical, device=device, dtype=torch.int64)
    expected = torch.tensor([physical[representatives[c]] for c in mapping],
                            device=device, dtype=torch.int64)
    pages = b.page_destination
    owner = b.state_slot_maps[0][1, 0]
    page_indices = (pages.index_select(0, actual).long(), pages.index_select(0, expected).long())
    owner_indices = (owner.index_select(0, actual).long(), owner.index_select(0, expected).long())
    row_indices = (actual, expected)
    fields = [("hidden", stage.hidden, row_indices),
              ("key", arena.key, page_indices), ("value", arena.value, page_indices),
              ("written", arena.written, page_indices)]
    if stage.logits is not None:
        fields.append(("logits", stage.logits, row_indices))
    feedback = getattr(stage, "draft_feedback", None)
    if feedback is not None:
        fields.append(("draft_feedback", feedback, row_indices))
    fields.extend(("state:" + name, value, owner_indices)
                  for name, value in arena.states.items())
    fields.extend(("page:" + name, value, page_indices)
                  for name, value in arena.page_fields.items())
    for name, value in events:
        if value.shape[0] != stage.captured_m:
            raise ValueError(f"observer lacks an explicit row layout: {name}")
        fields.append((name, value, row_indices))
    # The existing path accepts strided tensors. Keep that path when a producer
    # has not supplied contiguous immutable snapshots; do not copy a whole arena.
    if any(not value.is_contiguous() for _, value, _ in fields):
        return identity
    equal = torch.ones((), dtype=torch.bool, device=device)
    for slots in b.state_slot_maps[1:]:
        equal.logical_and_((slots[1, 0].index_select(0, actual).long() == owner_indices[0]).all())
    sites = dict(events)
    for name, value, (indices, anchors) in fields:
        if value.numel() == 0:
            continue
        raw = value.view(torch.uint8).reshape(value.shape[0], -1)
        # Include two gathers, byte comparison and optional route masks. Work
        # is bounded by tensor geometry and bytes, independent of model/GPU/M.
        chunk = max(1, (64 << 20) // (8 * raw.shape[1]))
        route_ids = None
        if name.rsplit("/", 2)[-2:-1] == ["routes"]:
            prefix, _, ordinal = name.rsplit("/", 2)
            route_ids = sites[f"{prefix}/route_ids/{ordinal}"]
        for first in range(0, len(physical), chunk):
            left = raw.index_select(0, indices[first:first + chunk])
            right = raw.index_select(0, anchors[first:first + chunk])
            if route_ids is not None:
                local = stage.config.text.moe.expert_count // stage.topology.ep_size
                begin = stage.topology.expert_coordinates(stage.rank)[0] * local
                values = []
                for bits, rows in ((left, indices), (right, anchors)):
                    ids = route_ids.index_select(0, rows[first:first + chunk])
                    owned = (ids >= begin) & (ids < begin + local)
                    shaped = bits.view(value.dtype).reshape(-1, *value.shape[1:])
                    values.append(torch.where(owned[..., None], shaped, 0).view(torch.uint8))
                left, right = values
            equal.logical_and_((left == right).all())
    if not bool(equal):
        return identity
    return tuple(representatives), tuple(mapping)


def _boundary_witnesses(
    stage: Any, rows: Sequence[int], *, max_workers: int,
) -> list[Boundary]:
    """Read on the observing thread; hash only completed immutable CPU copies.

    Pending snapshots are bounded by the worker count. The forward stays paused
    until every hash completes, and result order remains the declared row order.
    """
    def digest(snapshot: dict[str, torch.Tensor]) -> Boundary:
        return {name: tensor_digest(value) for name, value in snapshot.items()}

    if max_workers <= 1:
        return [digest(boundary_snapshot(stage, row + 1)) for row in rows]
    result = []
    pending = deque()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for row in rows:
            snapshot = boundary_snapshot(stage, row + 1)
            pending.append(pool.submit(digest, snapshot))
            if len(pending) >= max_workers:
                result.append(pending.popleft().result())
        while pending:
            result.append(pending.popleft().result())
    return result


def _row_failure(actual: RowWitness, expected: RowWitness) -> tuple[str, str] | None:
    if actual == expected:
        return None
    boundaries, intermediate = actual
    wanted, anchors = expected
    if boundaries.keys() != wanted.keys() or tuple(n for n, _ in intermediate) != tuple(
        n for n, _ in anchors
    ):
        return "harness", "cross-M semantic witness inventory differs"
    fields = [n for n in boundaries if boundaries[n] != wanted[n]]
    fields.extend(n for (n, a), (_, b) in zip(intermediate, anchors, strict=True) if a != b)
    return "harness", "cross-M numerical attribution unavailable: " + ", ".join(fields)


def _temporary(
    source: Any, events: list[tuple[str, torch.Tensor]], groups: dict[str, Any],
    *, draft_provider: Any = None,
) -> Any:
    geometry = (source.captured_m, source.config.text.hyperconnection.stream_count)
    with _observe(events, input_geometry=geometry if draft_provider is not None else None):
        options = dict(
            tp_group=groups.get("tp"),
            pp_group=None,
            ep_group=groups.get("ep"),
            expert_tp_group=groups.get("etp"),
            captured_m=source.captured_m,
            table_pages=source.buffers.page_table.shape[1],
            arena=source.arena,
            page_size=source.arena.layout.block_size_tokens,
            operator_config=source.operators.config,
            recurrent_state_dtype=source.recurrent_state_dtype,
            mrope_tables=source.mrope_tables,
        )
        if source.arena.layout.storage.quantized:
            options["cache_storage"] = source.arena.layout.storage
        if draft_provider is None:
            stage: Any = FlashNextDecodeStage(
                source.config, source.weights, source.topology, **options,
            )
        else:
            stage = draft_provider.build_stage(
                source.weights.original, source.topology, **options,
            )
        launch = stage.launch

        def observed_launch(*, produce_logits: bool = True) -> None:
            if not produce_logits:
                with preserve_observer_collections((events,)):
                    launch(produce_logits=False)
                return
            events.clear()
            launch()

        stage.launch = observed_launch
        stage.capture()
    return stage


def _segment_snapshot(stage: Any, row: int, consumed: int) -> Boundary:
    """Read a completed checkpoint, excluding later tentative page writes.

    Every row has its own recurrent checkpoint. KV is shared by the segment;
    compare its consumed prefix here and its complete write-set separately.
    This changes only independent host snapshots, never observed device storage.
    """
    snapshot = boundary_snapshot(stage, row + 1)
    layout = stage.arena.layout
    cache_fields = set()
    for name, grouped in (("key", layout.key_encoding), ("value", layout.value_encoding)):
        if not grouped.spec.dtype.quantized:
            snapshot[name][:, :, consumed:] = 0
            continue
        complete = grouped.groups_for_tokens(tuple(range(consumed)), complete=True)
        mask = torch.zeros(grouped.group_count, dtype=torch.bool)
        mask[list(complete)] = True
        snapshot[name][..., ~mask, :] = 0
        field = f"kv_{name}_scales"
        cache_fields.add(field)
        snapshot[f"page:{field}"][..., ~mask] = 0
    snapshot["written"][:, :, consumed:] = 0
    plan = stage.indexer_plan
    completed_groups = consumed // plan.compression
    owned_groups = max(0, (completed_groups + plan.size - 1 - plan.rank) // plan.size)
    for spec in stage.arena.layout.page_fields:
        if spec.field_id not in cache_fields:
            snapshot[f"page:{spec.field_id}"][:, owned_groups:] = 0
    return {name: tensor_digest(value) for name, value in snapshot.items()}


def _run_segments(
    stage: Any, requests: int, depth: int, corpora: Sequence[Sequence[int]],
    oracle: Sequence[Sequence[Boundary]], anchors: Sequence[Any], groups: dict[str, Any],
    *, reverse: bool, events: Sequence[tuple[str, torch.Tensor]],
    feedback_rows: Sequence[torch.Tensor] = (),
) -> None:
    """Witness real continuation rows and every accepted-prefix checkpoint."""
    arena, b, cap = stage.arena, stage.buffers, stage.captured_m
    device, group = arena.device, groups["tp"]
    if requests * depth > cap or requests > arena.spec.state_slot_capacity:
        raise ValueError("segment witness exceeds distinct durable owners")
    arena.reset_state_rows(torch.arange(arena.spec.state_slot_capacity, device=device))
    arena.reset_page_rows(torch.arange(arena.spec.page_capacity, device=device))
    stage.prepare_scratch()
    physical = []
    tokens = torch.zeros(cap, device=device, dtype=torch.int64)
    if feedback_rows:
        b.target_hidden.zero_()
    for owner in range(requests):
        start = cap - (owner + 1) * depth if reverse else owner * depth
        for position in range(depth):
            row = start + position
            physical.append(row)
            tokens[row] = corpora[owner % len(corpora)][position]
            b.row_activity[row] = 1
            b.segment_lengths[row] = depth if position == 0 else 0
            b.page_source[row] = b.page_destination[row] = owner
            b.page_table[row].fill_(owner)
            b.position_ids[:, row] = position
            b.sequence_lengths[row] = position + 1
            b.token_offsets[row] = position
            for slots in b.state_slot_maps:
                slots[0, :, row] = owner
                if position == depth - 1:
                    slots[1, :, row] = owner
            if feedback_rows:
                b.target_hidden[row] = feedback_rows[position].reshape(-1)
    stage.fill_token_ids(tokens)
    stage.hidden.zero_()
    if reverse:
        inactive = b.row_activity == 0
        stage.hidden[inactive] = float("nan")
        private_states = b.state_slot_maps[0][1, 0, inactive].long()
        private_pages = b.page_destination[inactive].long()
        for value in arena.states.values():
            if value.is_floating_point():
                value.index_fill_(0, private_states, float("nan"))
        for value in (arena.key, arena.value, *arena.page_fields.values()):
            if value.is_floating_point():
                value.index_fill_(0, private_pages, float("nan"))
    stage.stream.wait_stream(torch.cuda.current_stream(device))
    dist.barrier(group=group)
    stage.replay()
    stage.synchronize()
    dist.barrier(group=group)
    schedule = tuple((name, tuple(value.shape), str(value.dtype)) for name, value in events)
    schedules: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(schedules, schedule, group=group)
    _coordinate(None if all(item == schedule for item in schedules) else
                ("harness", "segmented rank observer schedules differ"), group)
    error = None
    try:
        representatives, row_map = _replicated_witness_rows(
            stage, physical,
            tuple((row // depth % len(corpora), row % depth) for row in range(len(physical))),
            events,
        )
        selected_rows = tuple(physical[row] for row in representatives)
        host_events = _host_events(events, cap, selected_rows)
        selected_boundaries = tuple(
            _segment_snapshot(stage, physical[row], row % depth + 1) for row in representatives)
        selected_intermediates = tuple(_intermediate_witness(
            host_events, stage, row, snapshot_rows=len(selected_rows),
        ) for row in range(len(selected_rows)))
        for logical_row in range(len(physical)):
            owner, position = divmod(logical_row, depth)
            pattern = owner % len(corpora)
            selected = row_map[logical_row]
            actual = selected_boundaries[selected]
            _require_m1_identity(actual, oracle[pattern][position],
                                 ("segment", cap, requests, depth, reverse, owner, position))
            intermediate = selected_intermediates[selected]
            failure = _row_failure((actual, intermediate), anchors[pattern][position][0])
            if failure is not None:
                raise _NumericalAttributionUnavailable(
                    f"segmented producer attribution: {failure}"
                )
        for name, value in arena.states.items():
            if not _initial_bytes_equal(value[requests:arena.spec.state_slot_capacity],
                                        arena.state_specs[name].initial_value):
                raise AssertionError(f"segmented foreign owner write: {name}")
        for name, value in (("key", arena.key), ("value", arena.value),
                            ("written", arena.written), *arena.page_fields.items()):
            if bool(value[requests:arena.spec.page_capacity].contiguous().view(torch.uint8).any()):
                raise AssertionError(f"segmented foreign page write: {name}")
        wanted = torch.zeros_like(arena.written[:requests])
        wanted[..., :depth] = True
        if not torch.equal(arena.written[:requests], wanted):
            raise AssertionError("segmented owned KV write-set")
    except AssertionError as exc:
        error = ("execution", str(exc))
    except ValueError as exc:
        error = ("harness", str(exc))
    except Exception as exc:
        error = ("harness", repr(exc))
    _coordinate(error, group)


def _run(
    stage: Any,
    logical: int,
    corpora: tuple[tuple[int, ...], ...],
    oracle: list[list[Boundary]],
    groups: dict[str, Any],
    *,
    offset: int = 0,
    reverse: bool = False,
    events: Sequence[tuple[str, torch.Tensor]] = (),
    input_feature_rows: Sequence[bytes] = (),
    position_axes: Sequence[tuple[int, int, int]] = (),
    feedback_rows: Sequence[torch.Tensor] = (),
) -> list[list[RowWitness]]:
    arena, b, cap = stage.arena, stage.buffers, stage.captured_m
    group = groups["tp"]
    device = arena.device
    owners = torch.arange(logical, device=device, dtype=torch.int64)
    block = arena.layout.block_size_tokens
    pages_per_owner = (len(corpora[0]) + block - 1) // block
    owned_pages = logical * pages_per_owner
    if owned_pages > arena.spec.page_capacity or pages_per_owner > b.page_table.shape[1]:
        raise ValueError("cross-M trajectory exceeds its admitted page ownership")
    page_ids = owners[:, None] * pages_per_owner + torch.arange(
        pages_per_owner, device=device, dtype=torch.int64,
    )[None, :]
    # Include non-prefix physical rows in the permutation; metadata remains valid.
    physical = torch.arange(logical, device=device)
    if reverse:
        physical = cap - 1 - physical
    arena.reset_state_rows(torch.arange(arena.spec.state_slot_capacity, device=device))
    arena.reset_page_rows(torch.arange(arena.spec.page_capacity, device=device))
    results = []
    protected_pages = []
    page_values = (("key", arena.key), ("value", arena.value),
                   ("written", arena.written), *arena.page_fields.items())
    for position in range(len(corpora[0])):
        stage.prepare_scratch()
        tokens = torch.zeros(cap, device=device, dtype=torch.int64)
        tokens[physical] = torch.tensor(
            [corpora[(row + offset) % len(corpora)][position] for row in range(logical)],
            device=device,
        )
        stage.fill_token_ids(tokens)
        b.row_activity[physical] = 1
        b.segment_lengths[physical] = 1
        current_pages = page_ids[:, position // block]
        b.page_source[physical] = current_pages.to(b.page_source.dtype)
        b.page_destination[physical] = current_pages.to(b.page_destination.dtype)
        b.page_table[physical[:, None], torch.arange(pages_per_owner, device=device)] = (
            page_ids.to(b.page_table.dtype)
        )
        for slots in b.state_slot_maps:
            slots[:, :, physical] = owners[None, None].to(slots.dtype)
        b.position_ids[:, physical] = position
        if position_axes:
            b.position_ids[:, physical] = torch.tensor(
                position_axes[position], device=device, dtype=torch.int64
            )[:, None]
        if input_feature_rows and input_feature_rows[position]:
            value = torch.frombuffer(bytearray(input_feature_rows[position]),
                                     dtype=torch.bfloat16).to(device)
            stage.feature_rows[physical] = value.to(stage.feature_rows.dtype)
            stage.embedding_mask[physical] = 0
        b.sequence_lengths[physical] = position + 1
        b.token_offsets[physical] = position % block
        stage.hidden.zero_()
        if feedback_rows:
            b.target_hidden.zero_()
            b.target_hidden[physical] = feedback_rows[position].reshape(1, -1)
        if reverse and logical < cap:
            inactive = b.row_activity == 0
            stage.hidden[inactive] = float("nan")
            private_states = b.state_slot_maps[0][1, 0, inactive].long()
            private_pages = b.page_destination[inactive].long()
            for value in arena.states.values():
                if value.is_floating_point():
                    value.index_fill_(0, private_states, float("nan"))
            for value in (arena.key, arena.value, *arena.page_fields.values()):
                if value.is_floating_point():
                    value.index_fill_(0, private_pages, float("nan"))
        stage.stream.wait_stream(torch.cuda.current_stream(device))
        dist.barrier(group=group)
        stage.replay()
        stage.synchronize()
        dist.barrier(group=group)
        # No diagnostic collective, host read or assertion occurs in a callback.
        schedule = tuple((name, tuple(value.shape), str(value.dtype)) for name, value in events)
        schedules: list[Any] = [None] * dist.get_world_size(group)
        dist.all_gather_object(schedules, schedule, group=group)
        _coordinate(
            None
            if all(item == schedule for item in schedules)
            else ("harness", "rank event schedules differ"),
            group,
        )
        error = None
        step = []
        try:
            physical_rows = physical.cpu().tolist()
            representatives, row_map = _replicated_witness_rows(
                stage, physical_rows,
                tuple((row + offset) % len(corpora) for row in range(logical)), events,
            )
            selected_rows = tuple(physical_rows[row] for row in representatives)
            host_events = _host_events(events, cap, selected_rows)
            workers = min(4, len(selected_rows), max(
                1, len(os.sched_getaffinity(0)) // dist.get_world_size(group),
            ))
            witnesses = _boundary_witnesses(stage, selected_rows, max_workers=workers)
            intermediates = tuple(_intermediate_witness(
                host_events, stage, row, snapshot_rows=len(selected_rows),
            ) for row in range(len(selected_rows)))
            for row, selected in enumerate(row_map):
                witness = witnesses[selected]
                wanted = oracle[(row + offset) % len(corpora)][position]
                _require_m1_identity(witness, wanted, (cap, logical, position, row))
                step.append((witness, intermediates[selected]))
            for name, value in arena.states.items():
                initial = arena.state_specs[name].initial_value
                assert _initial_bytes_equal(
                    value[logical : arena.spec.state_slot_capacity], initial,
                ), (
                    "unowned state write",
                    name,
                    cap,
                    logical,
                    position,
                )
            for name, value in (
                ("key", arena.key),
                ("value", arena.value),
                ("written", arena.written),
                *arena.page_fields.items(),
            ):
                assert not bool(
                    value[owned_pages : arena.spec.page_capacity]
                    .contiguous().view(torch.uint8).any()
                ), (
                    "unowned page write",
                    name,
                    cap,
                    logical,
                    position,
                )
            for ids, snapshots in protected_pages:
                for name, value in page_values:
                    observed = value.index_select(0, ids).detach().to("cpu", copy=True)
                    assert torch.equal(observed.contiguous().view(torch.uint8),
                                       snapshots[name].contiguous().view(torch.uint8)), (
                        "closed page mutation", name, cap, logical, position,
                    )
            if (position + 1) % block == 0:
                ids = page_ids[:, position // block]
                protected_pages.append((ids, {
                    name: value.index_select(0, ids).detach().to("cpu", copy=True)
                    for name, value in page_values
                }))
            wanted_written = torch.zeros_like(arena.written[:owned_pages]).reshape(
                logical, pages_per_owner, *arena.written.shape[1:],
            )
            for page_index in range(pages_per_owner):
                count = max(0, min(block, position + 1 - page_index * block))
                wanted_written[:, page_index, ..., :count] = True
            assert torch.equal(arena.written[:owned_pages],
                               wanted_written.reshape_as(arena.written[:owned_pages])), (
                "owned KV write set"
            )
        except _NumericalAttributionUnavailable as exc:
            error = ("harness", str(exc))
        except AssertionError as exc:
            error = ("execution", str(exc))
        except Exception as exc:
            error = ("harness", repr(exc))
        _coordinate(error, group)
        results.append(step)
    return results


def audit_cross_m(
    executors: dict[int, Any], admission: Any, groups: dict[str, Any], m1_report: dict[str, Any],
    *, capacities: Sequence[int] | None = None, capture_bucket: Callable[[int], Any] | None = None,
    visual_report: dict[str, Any] | None = None,
    visual_tokens: tuple[int, ...] = (), input_feature_rows: tuple[bytes, ...] = (),
    position_axes: tuple[tuple[int, int, int], ...] = (),
    draft_provider: Any = None, feedback_rows: tuple[torch.Tensor, ...] = (),
    reference_math_memo: Any = None,
    completed_witness: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Check every admitted bucket boundary with a bounded, shared request corpus."""
    if (capacities is None) != (capture_bucket is None):
        raise ValueError("deferred audit capture requires capacities and a capture callback")
    if completed_witness is not None and not callable(completed_witness):
        raise ValueError("completed numerical witness consumer must be callable")
    buckets = tuple(sorted(executors if capacities is None else capacities,
                           reverse=capture_bucket is not None))
    if not buckets or 1 not in buckets or len(set(buckets)) != len(buckets):
        raise ValueError("cross-M audit requires unique capacities including M1")
    groups = {**groups, "tp": groups.get("tp", groups.get("sampling_tp"))}
    source = executors[1].body
    if source.topology.pp_size != 1:
        raise ValueError("typed cross-M identity currently requires PP1")
    steps = min(
        source.arena.layout.block_size_tokens, max(4, source.config.text.qsa.compression_ratio)
    )
    vocabulary = source.config.text.vocabulary_size
    if draft_provider is None:
        if feedback_rows or m1_report.get("draft_method") is not None:
            raise ValueError("draft cross-M witness requires its provider and fixed feedback")
    elif (
        m1_report.get("draft_method") != draft_provider.spec.method
        or m1_report.get("feedback_recurrence")
        or len(feedback_rows) < steps
        or visual_report is not None
    ):
        raise ValueError("draft cross-M anchor must witness the same fixed feedback trace")
    full_feedback_rows = feedback_rows
    feedback_rows = feedback_rows[:steps]
    corpora = tuple(
        tuple((17 + (row * steps + p) * 7919) % vocabulary for p in range(steps))
        for row in range(min(4, max(buckets)))
    )
    if len(m1_report["steps"]) < steps:
        raise ValueError("local M1 witness does not cover the cross-M horizon")
    seal_steps = state_audit_horizon(steps, admission.max_sequence_tokens,
                                     source.arena.layout.storage)
    if len(m1_report["steps"]) < seal_steps:
        raise ValueError("local M1 witness does not cover column sealing")
    seal_corpora = (tuple((17 + p * 7919) % vocabulary for p in range(seal_steps)),)
    seal_feedback = ()
    if draft_provider is not None:
        if len(full_feedback_rows) < seal_steps:
            raise ValueError("fixed feedback does not cover column sealing")
        seal_feedback = full_feedback_rows[:seal_steps]
    visual_corpora = (visual_tokens[:steps],) if visual_report is not None else ()
    visual_oracle = ([[step["boundary_digests"] for step in visual_report["steps"][:steps]]]
                     if visual_report is not None else [])
    if visual_report is not None and min(len(visual_tokens), len(input_feature_rows),
                                         len(position_axes), len(visual_report["steps"])) < steps:
        raise ValueError("visual M1 witness does not cover the cross-M horizon")
    # Establish immutable intermediate witnesses on the locally audited M1 path.
    def plan_identity(body):
        plan = row_arithmetic_plan(body.operators, body.captured_m)
        attention_plan = _attention_arithmetic_plan(body)
        local = (tuple(sorted((repr(key), recipe) for key, recipe in plan.items())),
                 attention_plan)
        joint: list[Any] = [None] * dist.get_world_size(groups["tp"])
        dist.all_gather_object(joint, local, group=groups["tp"])
        return plan, attention_plan, tuple(joint)

    def make_anchors(plan, local_oracle, local_visual_oracle, *,
                     attention_plan, selected_corpora=corpora, selected_feedback=feedback_rows):
        events: list[tuple[str, torch.Tensor]] = []
        with row_arithmetic_scope(plan), _attention_arithmetic_scope(attention_plan):
            stage = _temporary(source, events, groups, draft_provider=draft_provider)
        anchors = []
        visual_anchor = None
        try:
            for offset in range(len(selected_corpora)):
                anchors.append(_run(
                    stage, 1, selected_corpora, local_oracle, groups, offset=offset, events=events,
                    feedback_rows=selected_feedback,
                ))
            if visual_report is not None and selected_corpora is corpora:
                visual_anchor = _run(
                    stage, 1, visual_corpora, local_visual_oracle, groups, events=events,
                    input_feature_rows=input_feature_rows, position_axes=position_axes,
                )
            return anchors, visual_anchor
        finally:
            stage.close()
            del stage
            events.clear()
            gc.collect()
            torch.cuda.empty_cache()

    base_plan, base_attention, base_identity = plan_identity(source)
    # Each corpus owns far fewer exact-input memo entries than the complete
    # Cartesian matrix. Keep it hot across plans instead of evicting reusable
    # independent arithmetic while walking unrelated corpora. Every actual
    # selected plan is bound before any shared reference result can be reused.
    planned = {}
    plan_members = {base_identity: (source, base_plan, base_attention)}
    for capacity in buckets:
        executor = (executors[capacity] if capture_bucket is None else capture_bucket(capacity))
        plan, attention_plan, identity = plan_identity(executor.body)
        planned[capacity] = executor, plan, attention_plan, identity
        plan_members.setdefault(identity, (executor.body, plan, attention_plan))
    reports_by_plan = {identity: ([m1_report] if identity == base_identity else [])
                       for identity in plan_members}
    from .capture_session import AuditCaptures
    from cubic_serve.runtime.quantized_qualification import _draft_reference_headroom
    # Private construction reuse is bounded independently of model size. Every
    # borrow still runs the complete original causal/free/state proof. Leave
    # three quarters of available headroom for captures and reference work.
    observations = AuditCaptures(min(1 << 30, _draft_reference_headroom(
        source.arena.device) // 4), reserve_bytes=512 << 20)
    # One mathematical problem is shared only while its exact typed producing
    # inputs/outputs, initial state, rounded transitions and weight versions are
    # live. Every consumer owns its native state and independently bounds its
    # actual attention partition. No trajectory or final verdict cache is used.
    from .step_observation import current_plan_reports
    shared_current = (not source.arena.layout.storage.quantized
                      and source.topology.ep_size == 1 and seal_steps == steps
                      and len(plan_members) > 1)
    # Count declared completed obligations, including exact-input transports;
    # this is useful-work liveness, never a numerical verdict or throughput.
    work_total = (len(plan_members) * (len(corpora) + (visual_report is not None))
                  + (len(plan_members) - 1 if shared_current else 0) + len(buckets))
    work_completed = 0
    if completed_witness is not None:
        completed_witness(0, work_total)

    def completed(count=1):
        nonlocal work_completed
        work_completed += count
        if not 0 < work_completed <= work_total:
            raise RuntimeError("typed numerical completed-work schedule differs")
        if completed_witness is not None:
            completed_witness(work_completed, work_total)
    current_resources = dict(prepared_single_step_captures=0, transported_reports=0,
                             complete_original_reports=0, input_witness_peak_bytes=0,
                             visual_transported_reports=0, fallback_reasons={})

    def original_report(member, tokens, selected_feedback, *, operator_observers=None,
                        selected_features=(), selected_axes=()):
        body, plan, attention_plan = member
        with _attention_arithmetic_scope(attention_plan):
            return audit_original(
                body, admission, groups, token_ids=tokens,
                diagnose_independent_trajectory=True, row_plan=plan,
                reference_math_memo=reference_math_memo, capture_pool=observations,
                draft_provider=draft_provider, feedback_rows=selected_feedback,
                column_completion_names=(() if draft_provider is None else
                                         draft_provider.reference_columns),
                operator_observers=operator_observers, input_feature_rows=selected_features,
                position_axes=selected_axes,
            )

    try:
        bindings = {}
        if shared_current:
            # Construct each private observation graph through one complete
            # original step first. It cannot issue a multi-step/portfolio receipt.
            # The current corpus below establishes full state propagation.
            for identity, member in plan_members.items():
                if identity == base_identity:
                    continue
                original_report(member, (corpora[0][0],), feedback_rows[:1])
                current_resources['prepared_single_step_captures'] += 1
                completed()
                body = member[0]
                candidates = [binding for binding in observations.entries
                              if binding[0] == id(body)]
                if len(candidates) == 1:
                    bindings[identity] = candidates[0]
            # The previously completed M1 report remains the separate startup
            # numerical witness. Re-evaluate this corpus once while the actual
            # current operator values are live for the other plan consumers.
            reports_by_plan[base_identity] = []
        def grouped_reports(tokens, selected_feedback, *, selected_features=(), selected_axes=()):
            def audited(member, **kwargs):
                return original_report(member, tokens, selected_feedback,
                    selected_features=selected_features, selected_axes=selected_axes, **kwargs)
            reports, measured = current_plan_reports(
                plan_members, bindings, observations, names, groups['tp'], audited)
            current_resources['complete_original_reports'] += measured['complete_original_reports']
            current_resources['input_witness_peak_bytes'] = max(
                current_resources['input_witness_peak_bytes'], measured['input_witness_peak_bytes'])
            for reason, count in measured['fallback_reasons'].items():
                reasons = current_resources['fallback_reasons']
                reasons[reason] = reasons.get(reason, 0) + count
            current_resources.setdefault('current_original_anchor_counts', []).append(measured['groups'])
            return reports, measured['transported_reports']

        if shared_current:
            weights = source.weights if draft_provider is None else source.weights.original
            names = {id(weights[binding.tensor_name]): binding.tensor_name
                     for binding in admission.parallel_plan.bindings_for_rank(source.rank)}
        for offset, tokens in enumerate(corpora):
            if shared_current:
                current_reports, transported = grouped_reports(tokens, feedback_rows)
                current_resources['transported_reports'] += transported
                for identity, report in current_reports.items():
                    if not report["cross_step_error_propagation_qualified"]:
                        raise ValueError("selected arithmetic lacks independent state propagation")
                    reports_by_plan[identity].append(report)
                completed(len(current_reports))
                continue
            for identity, member in plan_members.items():
                if identity == base_identity and offset == 0:
                    completed()
                    continue
                selected_tokens = (seal_corpora[0] if offset == 0
                                   and identity != base_identity and seal_steps > steps else tokens)
                report = original_report(member, selected_tokens,
                    seal_feedback if len(selected_tokens) == seal_steps else feedback_rows)
                current_resources['complete_original_reports'] += 1
                if identity != base_identity and not report["cross_step_error_propagation_qualified"]:
                    raise ValueError("selected row arithmetic lacks independent state propagation")
                reports_by_plan[identity].append(report)
                completed()
        visual_by_plan = {base_identity: visual_report}
        if visual_report is not None:
            if shared_current:
                visual_by_plan, transported = grouped_reports(
                    visual_tokens[:steps], (), selected_features=input_feature_rows[:steps],
                    selected_axes=position_axes[:steps])
                current_resources['visual_transported_reports'] += transported
                visual_oracle = [[step['boundary_digests']
                                  for step in visual_by_plan[base_identity]['steps']]]
            else:
                for identity, member in plan_members.items():
                    if identity == base_identity:
                        continue
                    visual_by_plan[identity] = original_report(
                        member, visual_tokens[:steps], (),
                        selected_features=input_feature_rows[:steps], selected_axes=position_axes[:steps])
                    current_resources['complete_original_reports'] += 1
            if any(not report["cross_step_error_propagation_qualified"]
                   for report in visual_by_plan.values()):
                raise ValueError("selected visual row arithmetic lacks state propagation")
            completed(len(visual_by_plan))
    finally:
        observations.clear()
        gc.collect()
    reports = reports_by_plan[base_identity]
    oracle = [[step["boundary_digests"] for step in report["steps"][:steps]] for report in reports]
    retirements = []
    anchors, visual_anchor = make_anchors(
        base_plan, oracle, visual_oracle, attention_plan=base_attention,
    )
    seal_oracle = [[step["boundary_digests"] for step in m1_report["steps"][:seal_steps]]]
    seal_anchors = (None if seal_steps == steps else make_anchors(
        base_plan, seal_oracle, [], selected_corpora=seal_corpora,
        selected_feedback=seal_feedback, attention_plan=base_attention,
    )[0])
    cohorts = {base_identity: (oracle, visual_oracle, anchors, visual_anchor,
                               seal_oracle, seal_anchors)}
    arithmetic_reports = [dict(
        source_capacity=1, rank_plans=base_identity, original_weight_reports=reports,
        visual_original_weight_report=visual_by_plan[base_identity],
    )]
    retirements.append(dict(
        phase="anchor", capacity=1,
        allocated_bytes=torch.cuda.memory_allocated(source.arena.device),
    ))
    records = []
    for capacity in buckets:
        executor, plan, attention_plan, identity = planned[capacity]
        if identity not in cohorts:
            # A new arithmetic plan gets an independent original-weight causal
            # and free-trajectory witness. Then every real bucket row must match
            # that witnessed plan through the entire state trace, permutations
            # and padding checks below. A plan change alone is not an FP verdict.
            selected_reports = reports_by_plan[identity]
            local_oracle = [[s["boundary_digests"] for s in r["steps"][:steps]]
                            for r in selected_reports]
            local_visual_oracle = []
            selected_visual = visual_by_plan.get(identity)
            if visual_report is not None:
                local_visual_oracle = [[s["boundary_digests"]
                                       for s in selected_visual["steps"]]]
            local_anchors, local_visual_anchor = make_anchors(
                plan, local_oracle, local_visual_oracle, attention_plan=attention_plan,
            )
            local_seal_oracle = [[s["boundary_digests"]
                                  for s in selected_reports[0]["steps"][:seal_steps]]]
            local_seal_anchors = (None if seal_steps == steps else make_anchors(
                plan, local_seal_oracle, [], selected_corpora=seal_corpora,
                selected_feedback=seal_feedback, attention_plan=attention_plan,
            )[0])
            cohorts[identity] = (local_oracle, local_visual_oracle,
                                 local_anchors, local_visual_anchor,
                                 local_seal_oracle, local_seal_anchors)
            arithmetic_reports.append(dict(
                source_capacity=capacity, rank_plans=identity,
                original_weight_reports=selected_reports,
                visual_original_weight_report=selected_visual,
            ))
        oracle, visual_oracle, anchors, visual_anchor, seal_oracle, seal_anchors = cohorts[identity]
        events = []
        stage = _temporary(executor.body, events, groups, draft_provider=draft_provider)
        try:
            independent_limit = min(capacity, source.arena.spec.state_slot_capacity)
            for logical in sorted({1, independent_limit, max(1, independent_limit // 2 + 1)}):
                for reverse in (False, True):
                    # First prove that observation does not change the resident graph result.
                    _run(executor.body, logical, corpora, oracle, groups, reverse=reverse,
                         feedback_rows=feedback_rows)
                    actual = _run(
                        stage, logical, corpora, oracle, groups, reverse=reverse, events=events,
                        feedback_rows=feedback_rows,
                    )
                    error = None
                    for position, rows in enumerate(actual):
                        for row, value in enumerate(rows):
                            failure = _row_failure(value, anchors[row % len(corpora)][position][0])
                            if failure is not None and error is None:
                                error = (
                                    failure[0],
                                    f"M{capacity} logical{logical} position{position} row{row}: "
                                    + failure[1],
                                )
                    _coordinate(error, groups["tp"])
                    if seal_anchors is not None:
                        _run(executor.body, logical, seal_corpora, seal_oracle, groups,
                             reverse=reverse, feedback_rows=seal_feedback)
                        sealed_rows = _run(
                            stage, logical, seal_corpora, seal_oracle, groups,
                            reverse=reverse, events=events, feedback_rows=seal_feedback,
                        )
                        error = None
                        for position, rows in enumerate(sealed_rows):
                            for row, value in enumerate(rows):
                                failure = _row_failure(value, seal_anchors[0][position][0])
                                if failure is not None and error is None:
                                    error = (failure[0], f"column seal M{capacity} "
                                             f"logical{logical} "
                                             f"position{position} row{row}: {failure[1]}")
                        _coordinate(error, groups["tp"])
                    if visual_anchor is not None:
                        options = dict(input_feature_rows=input_feature_rows,
                                       position_axes=position_axes, reverse=reverse)
                        _run(executor.body, logical, visual_corpora, visual_oracle, groups,
                             **options)
                        visual_rows = _run(stage, logical, visual_corpora, visual_oracle,
                                           groups, events=events, **options)
                        error = None
                        for position, rows in enumerate(visual_rows):
                            for row, value in enumerate(rows):
                                failure = _row_failure(value, visual_anchor[position][0])
                                if failure is not None and error is None:
                                    error = (failure[0], f"visual M{capacity} logical{logical} "
                                             f"position{position} row{row}: {failure[1]}")
                        _coordinate(error, groups["tp"])
                    records.append(
                        dict(
                            capacity=capacity,
                            logical=logical,
                            reverse=reverse,
                            steps=steps,
                            column_seal_steps=seal_steps,
                            column_seal_distinct_corpora=int(seal_steps > steps),
                            rows=len(corpora),
                            visual_features_and_positions=visual_anchor is not None,
                            attention_split_workspaces=[
                                dict(shape=list(shape), dtype=str(dtype))
                                for (role, shape, dtype) in stage.helper._buffers
                                if role == "qsa.attention_partial"
                            ],
                        )
                    )
                    logging.getLogger(__name__).info(
                        "typed cross-M witness rank=%d M%d logical=%d reverse=%s",
                        source.rank,
                        capacity,
                        logical,
                        reverse,
                    )
            # Multi-token prefill is present at MTP0 too. Every available
            # independently witnessed prefix must exercise real continuation
            # rows; draft depth must not suppress this state/ownership audit.
            for depth in range(2, min(capacity, steps) + 1):
                limit = min(capacity // depth, source.arena.spec.state_slot_capacity)
                for requests in sorted({1, limit, max(1, limit // 2 + 1)}):
                    for reverse in (False, True):
                        _run_segments(
                            stage, requests, depth, corpora, oracle, anchors, groups,
                            reverse=reverse, events=events, feedback_rows=feedback_rows,
                        )
                        records.append(dict(capacity=capacity, logical=requests * depth,
                                            segment_depth=depth, requests=requests,
                                            reverse=reverse, accepted_checkpoints=depth))
                        logging.getLogger(__name__).info(
                            "typed segmented witness rank=%d M%d requests=%d depth=%d reverse=%s",
                            source.rank, capacity, requests, depth, reverse,
                        )
        finally:
            stage.close()
            # Temporary capture callbacks retain bound methods and tensors in
            # cycles. Closing the CUDA graph alone does not release that owner.
            # Retire it before constructing the next, larger audit bucket.
            del stage
            events.clear()
            gc.collect()
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(source.arena.device)
            retirements.append(dict(
                phase="bucket", capacity=capacity, allocated_bytes=allocated,
            ))
            logging.getLogger(__name__).info(
                "cross-M temporary retired rank=%d M%d allocated_MiB=%.1f",
                source.rank, capacity, allocated / (1 << 20),
            )
        completed()
    source.arena.reset_state_rows(
        torch.arange(source.arena.spec.state_slot_capacity, device=source.arena.device)
    )
    source.arena.reset_page_rows(
        torch.arange(source.arena.spec.page_capacity, device=source.arena.device)
    )
    if work_completed != work_total:
        raise RuntimeError("typed numerical completed-work schedule is incomplete")
    return {
        "scope": "independently witnessed arithmetic plans with concurrent cross-M identity",
        "independent_trajectory": False,
        "cross_step_error_propagation_qualified": False,
        "draft_method": None if draft_provider is None else draft_provider.spec.method,
        "fixed_feedback_trace": bool(feedback_rows),
        "cases": records,
        "arithmetic_cohorts": arithmetic_reports,
        "temporary_retirements": retirements,
        "current_operator_math_resources": current_resources,
    }
