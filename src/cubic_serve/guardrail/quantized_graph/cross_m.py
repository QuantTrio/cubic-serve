"""Cross-capacity exact ownership and independently witnessed state trajectories.

Temporary instrumented captures exist only before readiness. Their ordinary
outputs and state must first match the resident uninstrumented graph exactly.
They are destroyed after each witness and never enter the serving portfolio.
"""

from __future__ import annotations

import logging
import math
import time

import torch
import torch.distributed as dist

from cubic_serve.guardrail.compare import tensor_digest
from cubic_serve.ops.cuda.linear.quantized import PackedProjectionDispatch
from cubic_serve.runtime.performance import _logical_m_values

from . import _ReferenceTrajectory, audit_original, byte_exact, ep, route_completion
from .linear import Recorder as LinearRecorder


def audit_cross_m(executors, admission, groups, *, steps=4, logical_ms=(),
                  completed_witness=None):
    if type(steps) is not int or steps < 1:
        raise ValueError("cross-M witness requires positive bounded steps")
    if not executors:
        raise ValueError("cross-M witness requires a graph portfolio")
    if completed_witness is not None and not callable(completed_witness):
        raise ValueError("completed cross-M witness consumer must be callable")
    storage = next(iter(executors.values())).body.arena.layout.storage
    horizon = max(4, storage.column_group_alignment + 1) if storage.quantized else 4
    if steps > horizon:
        raise ValueError("cross-M witness exceeds its cache-boundary horizon")
    if storage.quantized:
        steps = min(max(steps, storage.column_group_alignment + 1), admission.max_sequence_tokens)
    topology = admission.parallel_plan.topology
    groups = {**groups, "tp": groups.get("tp", groups.get("sampling_tp"))}
    if topology.pp_size != 1:
        raise ValueError("cross-M packed witness does not qualify this topology yet")
    observer = ep if topology.ep_size > 1 else route_completion
    capacities = tuple(executors)
    if any(type(m) is not int or not 1 <= m <= max(capacities) for m in logical_ms):
        raise ValueError("logical audit rows must fit the graph portfolio")
    cases = []
    for capacity, executor in executors.items():
        domain = _logical_m_values(capacity, capacities)
        selected = {domain[0], domain[-1]} | {m for m in logical_ms if m in domain}
        # Qualify both independent request corpora together before testing the
        # smaller padded occupancy. Reuse still requires every boundary byte.
        cases.extend((executor.body, logical) for logical in sorted(selected, reverse=True))
    total = sum(1 + min(2, logical) * steps for _, logical in cases)
    completed = 0

    def proved(count):
        nonlocal completed
        completed += count
        if completed_witness is not None:
            if not 0 < completed <= total:
                raise RuntimeError("cross-M completed witness count exceeds its plan")
            completed_witness(completed, total)

    common = []
    oracle = _TrajectoryOracle(admission, groups, steps, completed_witness=proved)
    records = []
    plans = {}
    with torch.inference_mode():
        for capacity, executor in executors.items():
            plans[str(capacity)] = _projection_plans(executor.body)
        for stage, logical in cases:
            # Keep the distinct-token ownership/permutation corpus intact.
            _case(stage, logical, steps, observer, groups, common, records)
            proved(1)
            # Like the typed graph auditor, use two independently witnessed
            # request corpora across every row, padding map and capacity.
            _case(stage, logical, steps, observer, groups, [], records, oracle=oracle)
    if completed_witness is not None and completed != total:
        raise RuntimeError("cross-M completed witness schedule is incomplete")
    records.append({"original_weight_trajectory_witnesses": oracle.reports,
                    "projection_plans": plans})
    return records


def _retain_common(common, m, cap, actual, comparisons):
    """Share immutable CPU prefixes only after the existing all-field proof.

    These snapshots have already been detached from live model storage. Keep
    every case's metadata and every differing image; reuse no producer result or
    numerical verdict. Reuse the comparisons already made, without another scan.
    """
    retained = actual
    for (previous_m, _, previous), comparison in zip(common, comparisons, strict=True):
        if previous_m >= m and not comparison["differing_fields"]:
            retained = {key: value[:m] for key, value in previous.items()}
            break
    for index, ((previous_m, previous_cap, _), comparison) in enumerate(
        zip(common, comparisons, strict=True),
    ):
        if previous_m < m and not comparison["differing_fields"]:
            common[index] = (
                previous_m, previous_cap,
                {key: value[:previous_m] for key, value in retained.items()},
            )
    common.append((m, cap, retained))


def _projection_plans(stage):
    names = {id(w): n for n, w in stage.body.model.weights.source._weights.items()}
    seen, plans = set(), []
    for dispatch in vars(stage.body).values():
        if not isinstance(dispatch, PackedProjectionDispatch) or id(dispatch) in seen:
            continue
        seen.add(id(dispatch))
        for key, (weight, binding) in dispatch._entries.items():
            plans.append(dict(
                weight=names[id(weight)], input_shape=key[0][2],
                artifacts=[dict(operator_id=a.operator_id, compiled_hash=a.compiled_hash,
                                source_hash=a.source_hash, specialization=a.specialization_json,
                                launch=a.launch_signature, grid=a.grid)
                           for a in binding.project_artifacts()],
            ))
    return plans


def _cpu_rows(value, m, key, *, repeated_corpus=False, reverse=False):
    """Prove every repeated row before transferring immutable representatives.

    Run only after the observed forward and schedule agreement. Distinct-token
    ownership snapshots retain every row. No floating comparison or producer
    assumption can substitute for the full storage-bit repetition check.
    """
    value = value[:m]
    if repeated_corpus:
        _require_row_patterns(key, value, m)
        patterns = min(2, m)
        value = value[m - patterns:m] if reverse else value[:patterns]
    return value.cpu().clone()


def _snapshot(stage, m, steps, *, check_ownership=True, repeated_corpus=False, reverse=False):
    arena = stage.arena
    spec = arena.spec
    b = stage.body.buffers
    rank = stage.hidden.device.index
    cap = stage.captured_m
    block = arena.layout.block_size_tokens
    ids = b.page_table[:m, :math.ceil(steps / block)].long()

    def page_rows(tensor):
        selected = tensor[ids].transpose(1, 2)
        return _cpu_rows(selected.reshape(m, tensor.shape[1], -1, *tensor.shape[3:]),
                         m, "KV", repeated_corpus=repeated_corpus, reverse=reverse)

    def cpu(value, key):
        return _cpu_rows(value, m, key, repeated_corpus=repeated_corpus, reverse=reverse)

    if check_ownership:
        owned = torch.zeros(spec.page_capacity, dtype=torch.bool, device=arena.key.device)
        owned[ids.flatten()] = True
        for name, tensor in (("key", arena.key), ("value", arena.value),
                             ("written", arena.written), *arena.page_fields.items()):
            dirty = tensor[:spec.page_capacity].contiguous().view(torch.uint8).reshape(
                spec.page_capacity, -1).any(dim=1)
            assert not bool(dirty[~owned].any()), ("unowned KV storage", name, rank, cap)
        logical_written = arena.written[ids].transpose(1, 2).reshape(m, -1, ids.shape[1] * block)
        expected = torch.arange(ids.shape[1] * block, device=arena.written.device) < steps
        assert torch.equal(logical_written, expected[None, None].expand_as(logical_written)), (
            "logical KV write set", rank, cap)
        for name, t in arena.states.items():
            assert not bool(t[m:spec.state_slot_capacity].contiguous().view(torch.uint8).any()), (
                "unowned state", name, rank, cap)
    return {
        "output": cpu(b.logits if stage.last else b.hidden, "output"),
        "key": page_rows(arena.key),
        "value": page_rows(arena.value),
        **{f"kv_field:{name}": page_rows(tensor)
           for name, tensor in arena.page_fields.items()},
        **(
            {
                name: cpu(torch.stack(
                    [getattr(op, attr)[:m] for op in stage.body._packed_experts], dim=1,
                ), name)
                for name, attr in (
                    ("route_ids", "route_ids"),
                    ("route_weights", "route_weights"),
                )
            }
            if stage.body._packed_experts
            else {}
        ),
        **{k: cpu(t, k) for k, t in arena.states.items()},
    }


def _collective_fields(result, collective, linear, m, captured_rows,
                       *, repeated_corpus=False, reverse=False):
    # A matching rank sum/final state must not hide a different local producer.
    # These are existing immutable snapshots, read only after the full forward
    # and topology-wide observation-schedule agreement.
    def store(key, value):
        result[key] = _cpu_rows(value, m, key, repeated_corpus=repeated_corpus, reverse=reverse)

    for index, (_weight, x, y) in enumerate(linear.linear):
        store(f"projection_input:{index}", x)
        store(f"projection_output:{index}", y)
    for index, (_weight, ids, y) in enumerate(linear.embeddings):
        store(f"embedding_ids:{index}", ids)
        store(f"embedding_output:{index}", y)
    for index, (_weights, x, ids, packed, activated, routes) in enumerate(linear.experts):
        rows = ids.shape[0]
        for name, value in (("input", x), ("ids", ids), ("gate_up", packed),
                            ("activated", activated), ("down", routes)):
            store(f"expert_{name}:{index}", value.reshape(rows, -1))
    for i, (semantic, (_, _source, out)) in enumerate(
        zip(collective.event_semantics, collective.events, strict=True)
    ):
        key = semantic[:2] if semantic[0] == "expert_routes" else semantic
        factor = out.shape[0] // captured_rows if semantic[0] == "expert_scale" else 1
        store(f"collective:{i}:{key!r}", out[:m * factor].reshape(m, -1))
    for index, scale in linear.int8_scales.items():
        store(f"int8_codes:{index}", linear.linear[index][1])
        store(f"int8_scales:{index}", scale)
    return result


def _row_digests(values, row):
    result = []
    for key, value in sorted(values.items()):
        selected = value[row:row + 1].contiguous()
        # Singleton strides are ignored by is_contiguous(), but byte views
        # still require unit stride. Normalize the view without a data copy.
        selected = selected.as_strided((selected.numel(),), (1,)).reshape(selected.shape)
        result.append((key, tensor_digest(selected)))
    return tuple(result)


def _require_row_patterns(key, value, m):
    """Exact full-row proof on the field's device, including NaN payload bits."""
    patterns = min(2, m)
    if value.shape[0] != m:
        raise RuntimeError("cross-M row-pattern snapshot layout differs")
    if value.is_contiguous() and value.stride(-1) == 1:
        payload = value.view(torch.uint8).reshape(m, -1)
        dtype = next(dtype for size, dtype in (
            (8, torch.int64), (4, torch.int32), (2, torch.int16), (1, torch.uint8),
        ) if payload.shape[1] % size == 0 and payload.storage_offset() % size == 0)
        words = payload.view(dtype)
        matches = all(torch.equal(words[c::2], words[c:c + 1].expand_as(words[c::2]))
                      for c in range(patterns))
    else:
        matches = all(byte_exact(value[c::2], value[c:c + 1].expand_as(value[c::2]),
                                 chunk_bytes=16 << 20) for c in range(patterns))
    if not matches:
        raise RuntimeError("cross-M row lacks its corpus boundary witness: " + key)


def _trajectory_rows(values, m):
    """Prove repeated corpus rows bit-for-bit before reusing their digests."""
    patterns = min(2, m)
    for key, value in values.items():
        if value.device.type != "cpu":
            raise RuntimeError("cross-M row-pattern snapshot layout differs")
        _require_row_patterns(key, value, m)
    digests = tuple(_row_digests(values, c) for c in range(patterns))
    return tuple(digests[row % 2] for row in range(m))


class _TrajectoryOracle:
    """Reuse a causal witness only for identical complete per-step boundaries."""

    def __init__(self, admission, groups, steps, *, completed_witness=None):
        self.admission, self.groups, self.steps = admission, groups, steps
        self.anchors = {0: [], 1: []}
        self.reports = []
        self.independent = {}
        self.completed_witness = completed_witness

    def qualify(self, stage, timelines):
        patterns = min(2, len(timelines[0]))
        rows = [tuple(timelines[position][corpus] for position in range(self.steps))
                for corpus in range(patterns)]
        missing = tuple(values not in self.anchors[corpus] for corpus, values in enumerate(rows))
        schedules = [None] * dist.get_world_size(self.groups["tp"])
        dist.all_gather_object(schedules, missing, group=self.groups["tp"])
        if any(len(item) != patterns for item in schedules):
            raise RuntimeError("cross-M independent corpus schedules differ")
        # Preserve the actual first two logical row positions in the proof.
        # Moving only a missing second corpus to row zero can select a different
        # permitted routed/projection reduction and cannot witness that row.
        needed = tuple(range(patterns)) if any(any(item) for item in schedules) else ()
        if needed:
            logical = len(needed)
            anchors = [[] for _ in needed]

            def completed(position, current, report, linear, collective):
                values = _collective_fields(
                    _snapshot(current, logical, position + 1, check_ownership=False),
                    collective, linear, logical, logical,
                )
                for row, anchor in enumerate(anchors):
                    anchor.append(_row_digests(values, row))
                if self.completed_witness is not None:
                    self.completed_witness(logical)

            independent = self.independent.setdefault(logical, _ReferenceTrajectory())
            report = audit_original(
                stage, self.admission, self.groups, prefix_length=0,
                logical_m=logical, steps=self.steps, token_offset=13 * needed[0],
                completed_step_witness=completed,
                _independent=independent if self.steps > 1 else None,
            )
            if self.steps > 1 and not report["cross_step_error_propagation_qualified"]:
                raise AssertionError("cross-M trajectory state propagation is incomplete")
            for corpus, anchor in zip(needed, anchors, strict=True):
                if tuple(anchor) != rows[corpus]:
                    raise AssertionError("cross-M actual rows differ from the causal witness")
            for corpus, anchor in zip(needed, anchors, strict=True):
                if tuple(anchor) not in self.anchors[corpus]:
                    self.anchors[corpus].append(tuple(anchor))
            self.reports.append(dict(
                physical_m=stage.captured_m, corpora=needed, original_weight=report,
                boundary_digests=anchors,
            ))
        receipts = [self.anchors[corpus].index(values) for corpus, values in enumerate(rows)]
        for position, rows in enumerate(timelines):
            for row, value in enumerate(rows):
                if value != self.anchors[row % 2][receipts[row % 2]][position]:
                    raise AssertionError("cross-M row lacks its original-weight trajectory")
        if self.completed_witness is not None and patterns > len(needed):
            # Reused witnesses advance only after exact complete-trajectory
            # comparison, never from a hash hit or matching final state alone.
            self.completed_witness((patterns - len(needed)) * self.steps)
        return receipts


def _case(stage, m, steps, tp_observer, groups, common, records, *, oracle=None):
    started = time.monotonic()
    cap = stage.captured_m
    rank = stage.hidden.device.index
    stage.select_decode_overlap(False)
    arena = stage.arena
    spec = arena.spec
    b = stage.body.buffers
    weights = stage.body.model.weights.source
    gdn_count = len(stage.body.gdn_layers)
    block = arena.layout.block_size_tokens
    page_count = math.ceil(steps / block)
    permuted = False

    def tokens_for(reverse=False):
        tokens = torch.zeros(cap, dtype=torch.int64, device=rank)
        tokens[:m] = torch.tensor(
            [100 + (i % 2 if oracle is not None else i) * 13 for i in range(m)],
            device=rank,
        )
        if reverse:
            tokens[:m] = tokens[:m].flip(0)
        return tokens

    def fill(poison=False, reverse=False, *, prefix_steps=None):
        nonlocal permuted
        permuted = reverse
        stage.prepare_scratch()
        arena.key[: spec.page_capacity].zero_()
        arena.value[: spec.page_capacity].zero_()
        arena.written[: spec.page_capacity].zero_()
        for t in arena.page_fields.values():
            t[:spec.page_capacity].zero_()
        for t in arena.states.values():
            t[: spec.state_slot_capacity].zero_()
        tokens = tokens_for(reverse)
        stage.bind_tokens(tokens)
        b.row_activity.zero_()
        b.row_activity[:m] = 1
        pages = torch.arange(m, device=rank, dtype=torch.int32)
        logical_pages = torch.arange(m * page_count, device=rank, dtype=torch.int32).reshape(
            m, page_count)
        b.page_source[:m] = logical_pages[:, 0]
        b.page_destination[:m] = logical_pages[:, 0]
        b.page_table[:m, :page_count] = logical_pages
        local = torch.arange(gdn_count, device=rank, dtype=torch.int32)
        slots = (pages[:, None] * gdn_count + local[None, :]).T
        b.conv_slots[:, :, :m] = slots.unsqueeze(0)
        b.recurrent_slots[:, :, :m] = slots.unsqueeze(0)
        if poison:
            for tensor in (arena.key, arena.value):
                tensor[spec.storage_page_capacity + m:].fill_(
                    255 if tensor.dtype == torch.uint8 else float("nan"))
            for tensor in arena.page_fields.values():
                tensor[spec.storage_page_capacity + m:].fill_(float("nan"))
            for t in arena.states.values():
                t[spec.state_slot_capacity + m :].fill_(float("nan"))
        for step in range(steps - 1 if prefix_steps is None else prefix_steps):
            torch.cuda.synchronize()
            dist.barrier()
            stage.replay()
            torch.cuda.synchronize()
            b.position_ids.fill_(step + 1)
            b.sequence_lengths.fill_(step + 2)
            b.token_offsets.fill_((step + 1) % block)
            current = logical_pages[:, (step + 1) // block]
            b.page_source[:m] = current
            b.page_destination[:m] = current
            stage.bind_tokens(tokens + step + 1)
        torch.cuda.synchronize()
        dist.barrier()

    def snap():
        return _snapshot(stage, m, steps, repeated_corpus=oracle is not None, reverse=permuted)

    def observed(poison=False, reverse=False):
        fill(poison, reverse)
        linear = LinearRecorder()
        collective = tp_observer.Recorder()
        collective.names = {id(w): n for n, w in weights._weights.items()}
        graph = torch.cuda.CUDAGraph()
        try:
            with (
                collective.observe(stage, linear),
                linear.observe(),
                torch.cuda.graph(graph, stream=stage.stream),
            ):
                stage.launch()
            fill(poison, reverse, prefix_steps=0 if oracle is not None else None)
            timeline = []
            positions = range(steps) if oracle is not None else (steps - 1,)
            for position in positions:
                b.position_ids.fill_(position)
                b.sequence_lengths.fill_(position + 1)
                b.token_offsets.fill_(position % block)
                current = b.page_table[:m, position // block]
                b.page_source[:m] = current
                b.page_destination[:m] = current
                stage.bind_tokens(tokens_for(reverse) + position)
                stage.stream.wait_stream(torch.cuda.current_stream(stage.hidden.device))
                with torch.cuda.stream(stage.stream):
                    graph.replay()
                stage.stream.synchronize()
                if stage.topology.ep_size > 1:
                    collective.validate(groups["tp"], m, cap, expert_groups=groups)
                else:
                    collective.validate(groups["tp"], m, cap)
                result = _collective_fields(
                    _snapshot(stage, m, position + 1, repeated_corpus=oracle is not None,
                              reverse=reverse),
                    collective, linear, m, cap, repeated_corpus=oracle is not None,
                    reverse=reverse,
                )
                if oracle is not None:
                    patterns = min(2, m)
                    digests = _trajectory_rows(result, patterns)
                    first = m - patterns if reverse else 0
                    timeline.append(tuple(digests[(row - first) % patterns] for row in range(m)))
            return result, timeline
        finally:
            graph.reset()

    fill()
    stage.replay()
    torch.cuda.synchronize()
    actual = snap()
    assert torch.isfinite(actual["output"]).all(), ("finite", rank, cap)
    same, timeline = observed()
    assert all(byte_exact(actual[k], same[k]) for k in actual), ("capture identity", rank, m, cap)
    del actual
    actual = same
    fill(True, True)
    stage.replay()
    torch.cuda.synchronize()
    perm_graph = snap()
    assert all((byte_exact(actual[k], v.flip(0)) for k, v in perm_graph.items())), (
        "graph padding/permutation",
        rank,
        m,
        cap,
    )
    perm, perm_timeline = observed(True, True)
    assert all(byte_exact(perm_graph[k], perm[k]) for k in perm_graph), (
        "permuted capture identity",
        rank,
        m,
        cap,
    )
    assert all(byte_exact(actual[k], perm[k].flip(0)) for k in actual), (
        "collective padding/permutation",
        rank,
        m,
        cap,
    )
    comparisons = []
    for previous_m, previous_cap, previous in common:
        shared = min(previous_m, m)
        assert actual.keys() == previous.keys(), ("cross-M semantic fields", rank)
        differences = [key for key in actual
                       if not byte_exact(previous[key][:shared], actual[key][:shared])]
        comparisons.append(dict(logical_m=previous_m, physical_m=previous_cap,
                                common_requests=shared, differing_fields=differences))
    proof_receipts = None
    if oracle is not None:
        assert all(tuple(reversed(rows)) == permuted for rows, permuted in
                   zip(timeline, perm_timeline, strict=True)), "per-step row permutation"
        proof_receipts = oracle.qualify(stage, timeline)
    overlap = False
    if stage._overlap_resources is not None:
        _overlap_case(stage, m, groups, fill, snap, actual, repeated_corpus=oracle is not None)
        overlap = True
    record = {
        "rank": rank,
        "logical_m": m,
        "physical_m": cap,
        "steps": steps,
        "route_fold_selected": None
        if stage.route_completion is None
        else stage.route_completion.selected,
        "graph_capture_identity": True,
        "permuted_capture_identity": True,
        "padding_permutation_all_fields_exact": True,
        "scope": "distinct-token exact ownership" if oracle is None else
                 "cross-M original-weight causal trajectory",
        "distinct_token_cross_m_identity_diagnostic": comparisons,
        "original_weight_trajectory_receipts": proof_receipts,
        "numerical_corpus": None if oracle is None else (100, 113),
        "cross_step_error_propagation_qualified": oracle is not None and steps > 1,
        "semantic_collective_keys": True,
        "fields": list(actual),
        "kv_write_set_and_unowned_slots_checked": True,
        "interleaved_all_fields_and_collectives_exact": overlap,
        "seconds": round(time.monotonic() - started, 3),
    }
    if oracle is None:
        _retain_common(common, m, cap, actual, comparisons)
    records.append(record)
    logging.getLogger(__name__).warning(
        "cross-M audit passed rank=%d logical=%d physical=%d interleaved=%s seconds=%.2f",
        rank, m, cap, overlap, record["seconds"],
    )
    del same, perm, perm_graph
    dist.barrier()


def _overlap_case(stage, m, groups, fill, snap, baseline, *, repeated_corpus=False):
    """Witness every child lane against the already audited ordinary graph.

    Record only stream-ordered immutable outputs during the forward. Comparing
    the concatenated child collectives also checks intermediate rank sums, not
    merely final logits. No child lane is granted an independent replay lease.
    """
    _, children, _, launch = stage._overlap_resources
    keys = [key for key in baseline if key.startswith("collective:")]
    if any("'scale'" in key or "'expert_" in key for key in keys):
        raise ValueError("interleaved qualification requires row-independent linear collectives")
    original = [child.body._tp_reduce for child in children]
    events = [[] for _ in children]
    graph = torch.cuda.CUDAGraph()
    try:
        stage.select_decode_overlap(True)
        fill(True, True)
        stage.replay()
        stage.synchronize()
        actual = snap()
        for key, value in actual.items():
            if not byte_exact(baseline[key], value.flip(0)):
                raise AssertionError(("interleaved graph padding/state", m, stage.captured_m, key))
        for child, reduce, recorded in zip(children, original, events, strict=True):
            if reduce is None:
                raise ValueError("interleaved TP qualification requires an explicit collective")

            def observe(x, reduce=reduce, recorded=recorded):
                result = reduce(x)
                recorded.append(x.clone())
                return result

            child.body._tp_reduce = observe
        fill(True, True)
        with torch.cuda.graph(graph, stream=stage.stream):
            launch()
        fill(True, True)
        with torch.cuda.stream(stage.stream):
            graph.replay()
        stage.synchronize()
        schedule = [[(tuple(x.shape), str(x.dtype)) for x in lane] for lane in events]
        schedules = [None] * dist.get_world_size(groups["tp"])
        dist.all_gather_object(schedules, schedule, group=groups["tp"])
        if any(value != schedule for value in schedules):
            raise RuntimeError("interleaved observation schedule differs between ranks")
        observed = snap()
        if not all(byte_exact(value, observed[key]) for key, value in actual.items()):
            raise AssertionError("interleaved observer changes output or durable state")
        if any(len(lane) != len(keys) for lane in events):
            raise RuntimeError("interleaved collective coverage differs from ordinary graph")
        for index, key in enumerate(keys):
            value = torch.cat([lane[index] for lane in events], dim=0)[:m].reshape(m, -1)
            value = _cpu_rows(value, m, key, repeated_corpus=repeated_corpus, reverse=True).flip(0)
            if not byte_exact(baseline[key], value):
                raise AssertionError(("interleaved collective differs", m, stage.captured_m, key))
    finally:
        for child, reduce in zip(children, original, strict=True):
            child.body._tp_reduce = reduce
        graph.reset()
        stage.select_decode_overlap(False)
