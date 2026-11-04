"""Independent original-weight and coherent-producer qualification of live graphs.

Observers retain immutable GPU snapshots only. Ordered event schedules are
checked on every rank after forward completion, before reference math or
collective verification. Raw eager differences and named arithmetic allowances
remain separate; capture identity alone never issues a numerical verdict.
"""

from __future__ import annotations

import math
import pickle
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cubic_serve.executor.runner import ReferenceExecutionContext
from cubic_serve.guardrail.quantization import with_reference_integer_geometry
from cubic_serve.models.qwen3_5.model import Qwen3_5ReferenceModel
from cubic_serve.ops.strategies import operator_strategy_scope

from . import ep, nonlinear, replay, route_completion
from .nonlinear import observe_cache_producers, state_rows


def byte_exact(a, b, *, chunk_bytes=1 << 20):
    """Compare storage bits without a full-size strided GPU materialization."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if chunk_bytes < 1:
        raise ValueError("byte comparison chunk budget must be positive")
    if (
        a.device == b.device and a.is_contiguous() and b.is_contiguous()
    ) or a.numel() * a.element_size() <= max(chunk_bytes, a.element_size()):
        left = a.contiguous().as_strided((a.numel(),), (1,)).view(torch.uint8)
        right = b.to(a.device).contiguous().as_strided((b.numel(),), (1,)).view(torch.uint8)
        # Equal aligned integer words iff every constituent storage byte is
        # equal. This changes no floating arithmetic, NaN or signed-zero rule.
        dtype = next(dtype for size, dtype in (
            (8, torch.int64), (4, torch.int32), (2, torch.int16), (1, torch.uint8),
        ) if all(value % size == 0 for value in (
            left.numel(), left.storage_offset(), right.storage_offset(),
            left.data_ptr(), right.data_ptr(),
        )))
        return torch.equal(left.view(dtype), right.view(dtype))
    axis = max(range(a.ndim), key=lambda dim: a.shape[dim])
    slice_bytes = a.numel() // a.shape[axis] * a.element_size()
    step = max(1, chunk_bytes // slice_bytes)
    return all(
        byte_exact(
            a.narrow(axis, start, min(step, a.shape[axis] - start)),
            b.narrow(axis, start, min(step, b.shape[axis] - start)),
            chunk_bytes=chunk_bytes,
        )
        for start in range(0, a.shape[axis], step)
    )


def _replay_after_fill(stage):
    """Order caller-stream input/state writes before the named replay stream."""
    stage.stream.wait_stream(torch.cuda.current_stream(stage.hidden.device))
    stage.replay()


def _packed_written_indices(arena, active_pages, owners):
    """Witness actual page identities, excluding views of owned state bytes."""
    shared = getattr(arena, "shared_storage", None)
    if shared is None:
        return None
    pool = shared.allocator
    if (not 0 < owners <= pool.state_capacity
            or not 0 < active_pages <= arena.spec.page_capacity):
        raise ValueError("packed oracle ownership exceeds its admitted arena")
    if any(s and b - pool.state_capacity * s < active_pages * p
           for b, p, s in pool.rank_geometries):
        raise ValueError("packed oracle page/state inputs overlap; isolate its arena")
    excluded = 0
    for geometry in pool.rank_geometries:
        for row in range(owners):
            excluded |= pool._overlap("state", row, geometry)
    return torch.tensor([p for p in range(arena.spec.page_capacity)
                         if p < active_pages or not excluded & (1 << p)],
                        device=arena.device, dtype=torch.int64)


def retain_logical_rows(linear, nonlinear, m):
    cap = linear.embeddings[0][1].shape[0]
    linear.linear = [(w, x[:m], y[:m]) for w, x, y in linear.linear]
    linear.int8_scales = {i: s[:m] for i, s in linear.int8_scales.items()}
    linear.embeddings = [(w, x[:m], y[:m]) for w, x, y in linear.embeddings]
    new = []
    for groups, x, ids, packed, activated, routes in linear.experts:
        rows, topk = ids.shape
        new.append(
            (
                groups,
                x[:m],
                ids[:m],
                packed.reshape(rows, topk, -1)[:m].reshape(m * topk, -1),
                activated.reshape(rows, topk, -1)[:m].reshape(m * topk, -1),
                routes.reshape(rows, topk, -1)[:m].reshape(m * topk, -1),
            )
        )
    linear.experts = new
    constants = {"conv": {1}, "gdn": {3, 4}, "norm": {2}, "zero": {1}}
    for kind, events in nonlinear.events.items():
        nonlinear.events[kind] = [
            tuple(
                (
                    t
                    if i in constants.get(kind, set())
                    else t[: m * (t.shape[0] // cap)]
                    if kind == "zero"
                    else t[:m]
                    for i, t in enumerate(event)
                )
            )
            for event in events
        ]


def _cache_identity(previous, current):
    """Exact stored-state induction, including the rounded recurrent boundary."""
    assert (previous.sequence_length, previous.batch_size) == (
        current.sequence_length, current.batch_size
    ), "trajectory cache geometry"
    assert len(previous.layers) == len(current.layers), "trajectory layer schedule"
    for old, new in zip(previous.layers, current.layers, strict=True):
        for name in ("key", "value", "convolution", "recurrent"):
            left, right = getattr(old, name), getattr(new, name)
            assert (left is None) == (right is None), ("trajectory state field", name)
            if left is not None:
                assert byte_exact(left, right), ("trajectory stored-state continuation", name)


class _ReferenceTrajectory:
    """Bounded reuse of independent reference work, never candidate proofs."""

    def __init__(self, max_bytes=256 << 20):
        self.scope = None
        self.entries = {}
        self.storages = {}
        self.max_bytes = max_bytes

    def bind(self, scope):
        if self.scope is None:
            self.scope = scope
        elif self.scope != scope:
            raise RuntimeError("independent reference trajectory scope differs")

    def lookup(self, key, cache):
        if key not in self.entries:
            return None
        previous, expected, observations = self.entries[key]
        _cache_identity(previous, cache)
        return expected, observations

    def retain(self, key, cache, expected, observations):
        if key in self.entries:
            return
        payload = (cache, expected, tuple(observations))
        storages = dict(self.storages)

        def account(value):
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                storages[value.device, storage.data_ptr()] = storage.nbytes()
            elif is_dataclass(value):
                for field in fields(value):
                    account(getattr(value, field.name))
            elif isinstance(value, tuple | list):
                for item in value:
                    account(item)

        account(payload)
        if sum(storages.values()) <= self.max_bytes:
            self.entries[key] = payload
            self.storages = storages


def _reference_lookup(trajectory, key, cache, group):
    """All ranks either reuse the reference or enter its collectives together."""
    import torch.distributed as dist

    cached, failure = None, None
    try:
        cached = trajectory.lookup(key, cache)
    except AssertionError as error:
        failure = ("harness", str(error))
    except RuntimeError as error:
        failure = ("execution", str(error))
    schedule = (key, cached is not None, failure)
    schedules = [None] * dist.get_world_size(group)
    dist.all_gather_object(schedules, schedule, group=group)
    if any(item[0] != key for item in schedules):
        raise RuntimeError("harness: independent reference schedule differs")
    failures = [item[2] for item in schedules if item[2] is not None]
    if failures:
        raise RuntimeError("independent reference cache lookup failed: " + str(failures))
    return cached if all(item[1] for item in schedules) else None


@with_reference_integer_geometry
def audit_original(
    stage, admission, groups, *, prefix_length=64, logical_m=None, steps=1,
    token_offset=0, completed_step_witness=None, _independent=None,
    token_ids=None, position_axes=(), prefix_store=None, prefix=None,
    prefix_capacity_tokens=None, completed_transition=None,
):
    """Audit an actual captured bucket before readiness; no runtime recapture."""
    if logical_m is not None and (
        type(logical_m) is not int or not 1 <= logical_m <= stage.captured_m
    ):
        raise ValueError("logical audit rows must fit the captured bucket")
    storage = stage.arena.layout.storage
    horizon = max(4, storage.column_group_alignment + 1) if storage.quantized else 4
    if type(steps) is not int or not 1 <= steps <= horizon:
        raise ValueError("trajectory witness exceeds the declared cache-boundary horizon")
    if steps > 1 and prefix_length != 0:
        raise ValueError("trajectory witness must start from cold state")
    if type(token_offset) is not int or token_offset < 0:
        raise ValueError("trajectory token offset must be a nonnegative integer")
    if steps > 1 and steps > admission.max_sequence_tokens:
        raise ValueError("trajectory witness exceeds the admitted context")
    topology = admission.parallel_plan.topology
    groups = {**groups, "tp": groups.get("tp", groups.get("sampling_tp"))}
    if topology.pp_size != 1:
        raise ValueError("independent packed graph oracle does not qualify this topology yet")
    if stage.hidden.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("independent packed graph oracle requires FP16 or BF16")
    live = completed_transition is not None
    if (prefix_store is not None or prefix is not None or token_ids is not None
            or position_axes or prefix_capacity_tokens is not None) and not live:
        raise ValueError("explicit live audit inputs require a completed transition consumer")
    restored = None
    if live:
        if (not callable(completed_transition) or prefix_store is None or steps != 1
                or logical_m != 1 or _independent is not None
                or admission.config.graph.text.moe is not None):
            raise ValueError("live packed transitions require one dense independently witnessed owner")
        vocabulary = admission.config.graph.text.vocabulary_size
        if (type(token_ids) is not tuple or len(token_ids) != 1
                or type(token_ids[0]) is not int or not 0 <= token_ids[0] < vocabulary
                or len(position_axes) != 1 or len(position_axes[0]) != 3
                or any(type(p) is not int or not 0 <= p < admission.max_sequence_tokens
                       for p in position_axes[0])):
            raise ValueError("live packed transition differs from its fixed token/position trace")
        if (type(prefix_capacity_tokens) is not int
                or not 1 <= prefix_capacity_tokens <= admission.max_sequence_tokens):
            raise ValueError("live packed transition requires admitted prefix capacity")
        prefix_binding = (id(stage), id(stage.body.model.weights.source),
                          repr(admission.operator_config), storage.identity,
                          prefix_capacity_tokens)
        restored = (None if prefix is None else
                    prefix.restore(prefix_store, prefix_binding, stage.hidden.device))
        prefix_length = 0 if restored is None else restored['position']
        if prefix_length >= prefix_capacity_tokens:
            raise ValueError("live audit prefix exceeds its admitted observation capacity")
    support = (
        ep
        if topology.ep_size > 1
        else SimpleNamespace(
            Replay=replay.Replay,
            Reference=replay.Reference,
            NonlinearReplay=nonlinear.NonlinearReplay,
        )
    )
    observer = ep if topology.ep_size > 1 else route_completion
    original_norm = stage.body.hidden_norm
    original_operators = stage.body.model.default_operators
    if _independent is not None:
        if prefix_length != 0 or steps == 1:
            raise ValueError("reference trajectory reuse requires a cold multistep witness")
        model = stage.body.model
        _independent.bind((
            type(model), pickle.dumps(model.config, protocol=5), model.recurrent_state_dtype,
            id(model.weights.source), frozenset(original_operators.row_weights),
            stage._vocab_span, admission.operator_config, logical_m, steps, storage.identity,
            tuple(sorted((name, id(group)) for name, group in groups.items())),
        ))
    stage.select_decode_overlap(False)
    owned = None
    try:
        with (
            observer.prepare(),
            torch.inference_mode(),
            operator_strategy_scope(
                admission.operator_config, device=stage.hidden.device, model=""
            ),
        ):
            if topology.ep_size > 1:
                owned = ep.configure(stage)
            stage.prepare_scratch()
            continuation = ({} if restored is None else restored) if live else (
                {} if steps > 1 else None)
            if live:
                continuation['live_prefix'] = True
                continuation['restore_native'] = restored is not None
                continuation['table_pages'] = math.ceil(
                    prefix_capacity_tokens / stage.arena.layout.block_size_tokens)
            if continuation is not None and storage.quantized:
                # Keep physical owners stable over the complete seal/page
                # boundary trajectory, including its initially empty pages.
                continuation["table_pages"] = math.ceil(
                    steps / stage.arena.layout.block_size_tokens)
            reports = []
            for position in range(steps):
                report = _audit(
                    stage, admission, groups, support, observer, owned,
                    min(prefix_length + position, admission.max_sequence_tokens - 1),
                    logical_m, continuation, token_offset, completed_step_witness, _independent,
                    token_ids=token_ids, position_axes=position_axes,
                    completed_transition=(None if not live else
                        (prefix_store, prefix_binding, completed_transition)),
                )
                stage.body.hidden_norm = original_norm
                stage.body.model.default_operators = original_operators
                reports.append(report)
            if steps == 1:
                return reports[0]
            return {
                "status": "passed", "steps": reports,
                "cross_step_error_propagation_qualified": True,
                "propagation_contract": (
                    "original-weight actual-input enclosures; exact rounded KV, convolution "
                    "and recurrent continuation from cold state; independently advancing "
                    "reference trajectory"
                ),
            }
    finally:
        stage.body.hidden_norm = original_norm
        stage.body.model.default_operators = original_operators
        if owned is not None:
            del stage.body.model._moe._audit_owned
        stage.prepare_scratch()


def _audit(
    stage, admission, groups, support, tp_observer, owned, prefix_length,
    logical_m=None, continuation=None, token_offset=0, completed_step_witness=None,
    independent_trajectory=None,
    *, token_ids=None, position_axes=(), completed_transition=None,
):
    import torch.distributed as dist

    cfg = admission.config
    weights = stage.body.model.weights.source
    rank = stage.hidden.device.index
    a = SimpleNamespace(
        logical_m=min(7, stage.captured_m) if logical_m is None else logical_m,
        physical_m=stage.captured_m,
        prefix_length=prefix_length,
        page_size=stage.arena.layout.block_size_tokens,
        poison_padding=True,
        full_model=True,
        tp=stage.topology.tp_size,
    )
    table_pages = (continuation.get("table_pages", math.ceil((prefix_length + 1) / a.page_size))
                   if continuation is not None else math.ceil((prefix_length + 1) / a.page_size))
    report = {"status": "harness_failure"}
    b = stage.body.buffers
    arena = stage.arena
    encoded = arena.layout.storage.quantized
    carry = (continuation is not None and prefix_length > 0
             and (encoded or continuation.get('live_prefix', False)))
    m = a.logical_m
    cap = a.physical_m
    pages = torch.arange(m, device=rank, dtype=torch.int32)
    owner_slice = slice(0, m)
    shared = getattr(arena, "shared_storage", None)
    if shared is not None:
        state_bytes = 0 if shared.state_layout is None else shared.state_layout.entry_bytes
        if (m > arena.spec.state_slot_capacity or
                m * state_bytes + m * table_pages * shared.page_layout.entry_bytes
                > shared.budget_bytes):
            raise ValueError("independent oracle owners exceed the shared byte budget")
        owner_slice = slice(arena.spec.state_slot_capacity - m, arena.spec.state_slot_capacity)
        pages = pages + owner_slice.start
    logical_pages = (
        torch.arange(m * table_pages, device=rank, dtype=torch.int32)
        .reshape(m, table_pages)
        .flip(1)
    )
    written_indices = _packed_written_indices(arena, m * table_pages, m)
    b.row_activity.zero_()
    b.row_activity[:m] = 1
    b.page_table[:m].zero_()
    b.page_table[:m, :table_pages].copy_(logical_pages)

    def kv_rows(t, *, row_slice=slice(None), layer=None):
        # Restrict the source before gathering: an identity check must not
        # materialize every request/layer merely to inspect one of them.
        if layer is not None:
            t = t[:, layer : layer + 1]
        indices = logical_pages[row_slice]
        z = t[indices.long()]
        return z.permute(0, 2, 1, 3, 4, 5).reshape(
            indices.shape[0], z.shape[2], table_pages * a.page_size, z.shape[-2], z.shape[-1]
        )

    def position(step):
        b.position_ids.fill_(step)
        b.sequence_lengths.fill_(step + 1)
        b.token_offsets.fill_(step % a.page_size)
        current = logical_pages[:, step // a.page_size]
        b.page_source[:m].copy_(current)
        b.page_destination[:m].copy_(current)

    count = len(stage.body.gdn_layers)
    local = torch.arange(count, device=rank, dtype=torch.int32)
    slots = (pages[:, None] * count + local[None, :]).T
    b.conv_slots[:, :, :m].copy_(slots.unsqueeze(0).expand_as(b.conv_slots[:, :, :m]))
    b.recurrent_slots[:, :, :m].copy_(slots.unsqueeze(0).expand_as(b.recurrent_slots[:, :, :m]))
    storage = (
        arena.key[: m * table_pages],
        arena.value[: m * table_pages],
        arena.written[: m * table_pages],
        *(t[: m * table_pages] for t in arena.page_fields.values()),
        *(t[owner_slice] for t in arena.states.values()),
    )
    # This cold audit owns the arena. Earlier qualification may have marked
    # other pages; start a fresh full-pool write witness before either replay.
    # Keep protected KV/state payloads intact and retain the exact all-page
    # comparison below, so any new unowned write still fails.
    if carry:
        # Only the preceding fully witnessed frame grants this receipt. Check
        # every owned code, scale, rounded state and native-tail byte before
        # avoiding the former repeated cold-prefix execution.
        for old, tensor in zip(continuation["storage"], storage, strict=True):
            if continuation.get('restore_native', False):
                tensor.copy_(old)
            assert byte_exact(old, tensor), "trajectory physical stored-state continuation"
    else:
        if written_indices is None:
            arena.written[: arena.spec.page_capacity].zero_()
        else:
            arena.written.index_fill_(0, written_indices, False)
        for tensor in storage:
            tensor.zero_()
    if a.poison_padding:
        for data in (arena.key, arena.value):
            data[arena.spec.storage_page_capacity + m:].fill_(
                255 if data.dtype == torch.uint8 else float("nan"))
        for tensor in arena.states.values():
            tensor[arena.spec.state_slot_capacity + m :].fill_(float("nan"))

    def bind_logical(tokens):
        padded = torch.zeros(cap, device=rank, dtype=torch.int64)
        padded[:m] = tokens
        stage.bind_tokens(padded)

    prefix_length = a.prefix_length
    native_histories = {side: [[] for _ in stage.body.attentions] for side in ("key", "value")}
    prefix_scope = (observe_cache_producers(stage.body) if encoded and not carry else
                    nullcontext(None))
    with prefix_scope as prefix_records:
        for step in range(0 if carry else prefix_length):
            bind_logical(
                100 + token_offset + torch.arange(m, device=rank, dtype=torch.int64) * 13 + step
            )
            position(step)
            if encoded:
                stage.launch()
            else:
                _replay_after_fill(stage)
            torch.cuda.synchronize()
    if encoded and not carry:
        schedule = tuple((layer, tuple(key.shape), str(key.dtype), tuple(value.shape),
                          str(value.dtype)) for layer, key, value in prefix_records)
        schedules = [None] * dist.get_world_size(groups["tp"])
        dist.all_gather_object(schedules, schedule, group=groups["tp"])
        if any(item != schedule for item in schedules):
            raise RuntimeError("harness: native prefix observation schedules differ")
        expected_layers = tuple(range(len(stage.body.attentions))) * prefix_length
        assert tuple(item[0] for item in prefix_records) == expected_layers, "prefix coverage"
        for layer, key, value in prefix_records:
            native_histories["key"][layer].append(key[:m])
            native_histories["value"][layer].append(value[:m])

    native_prefixes = {}

    def native_prefix(side):
        if side in native_prefixes:
            return native_prefixes[side]
        if not prefix_length:
            value = torch.empty(
                (m, len(stage.body.attentions), 0, arena.layout.num_kv_heads,
                 arena.layout.head_dim),
                device=arena.device, dtype=arena.layout.dtype)
        elif carry:
            value = torch.stack([
                getattr(layer, side).transpose(1, 2)
                for layer in continuation["candidate"].layers if layer.key is not None
            ], dim=1)
        else:
            value = torch.stack([torch.stack(values, dim=1)
                                 for values in native_histories[side]], dim=1)
        native_prefixes[side] = value
        return value

    frozen = tuple(t.clone() for t in storage)
    from cubic_serve.models.qwen3_5.model import Qwen3_5LayerCache, Qwen3_5ReferenceCache

    prefix_layers = []
    gi = fi = 0
    for kind in cfg.graph.text.layer_schedule:
        if kind.value == "gated-delta-net":
            prefix_layers.append(
                Qwen3_5LayerCache(
                    convolution=state_rows(stage.body.conv_state, b.conv_slots[0, gi, :m], 3),
                    recurrent=state_rows(stage.body.recurrent_state,
                                         b.recurrent_slots[0, gi, :m], 4),
                )
            )
            gi += 1
        else:
            prefix_layers.append(
                Qwen3_5LayerCache()
                if prefix_length == 0
                else Qwen3_5LayerCache(
                    key=(native_prefix("key")[:, fi] if encoded else kv_rows(
                        arena.key, layer=fi)[:, 0, :prefix_length]).transpose(1, 2).clone(),
                    value=(native_prefix("value")[:, fi] if encoded else
                           kv_rows(arena.value, layer=fi)[:, 0, :prefix_length]
                           ).transpose(1, 2).clone(),
                )
            )
            fi += 1
    prefix_cache = Qwen3_5ReferenceCache(prefix_length, m, tuple(prefix_layers))
    if continuation is not None and prefix_length:
        _cache_identity(continuation["candidate"], prefix_cache)
    ids = (100 + token_offset + torch.arange(m, device=rank, dtype=torch.int64) * 13 + prefix_length
           if token_ids is None else torch.tensor(token_ids, device=rank, dtype=torch.int64))
    bind_logical(ids)
    position(prefix_length)
    if position_axes:
        b.position_ids[:, :m].copy_(torch.tensor(position_axes, device=rank).T)
    _replay_after_fill(stage)
    torch.cuda.synchronize()
    actual = b.logits[:m].clone()
    graph_conv = [
        state_rows(stage.body.conv_state, b.conv_slots[1, i, :m], 3).cpu()
        for i in range(len(stage.body._gdn_local))
    ]
    graph_recurrent = [
        state_rows(stage.body.recurrent_state, b.recurrent_slots[1, i, :m], 4).cpu()
        for i in range(len(stage.body._gdn_local))
    ]
    graph_storage = (tuple(t.clone() for t in storage)
                     if encoded or completed_transition is not None else None)
    if encoded:
        # A newly sealed group may replace its former native tail. Previously
        # closed bytes and scales remain immutable, independently of encoding.
        for row in range(m):
            for page_index in range(table_pages):
                physical = int(logical_pages[row, page_index])
                count = min(a.page_size, max(0, prefix_length - page_index * a.page_size))
                offsets = tuple(range(count))
                for is_key, data, old, side in (
                    (True, arena.key, frozen[0], "key"),
                    (False, arena.value, frozen[1], "value"),
                ):
                    mask = arena.layout.storage_token_mask(offsets, key=is_key, device=data.device)
                    assert byte_exact(data[physical][:, mask], old[physical][:, mask]), (
                        "closed prefix mutation", side)
                    spec = arena.layout.storage.key if is_key else arena.layout.storage.value
                    if spec.dtype.quantized:
                        name = f"kv_{side}_scales"
                        index = tuple(arena.page_fields).index(name) + 3
                        assert byte_exact(arena.page_fields[name][physical][:, mask.any(-1)],
                                          frozen[index][physical][:, mask.any(-1)]), (
                            "closed prefix scale mutation", side)
    else:
        graph_key = kv_rows(arena.key)[:, :, : prefix_length + 1].clone()
        graph_value = kv_rows(arena.value)[:, :, : prefix_length + 1].clone()
        assert byte_exact(
            graph_key[:, :, :prefix_length], kv_rows(frozen[0])[:, :, :prefix_length]), (
            "graph prefix key mutation"
        )
        assert byte_exact(
            graph_value[:, :, :prefix_length], kv_rows(frozen[1])[:, :, :prefix_length]
        ), "graph prefix value mutation"
    for tensor, old in zip(storage, frozen, strict=True):
        tensor.copy_(old)
    # The independent prefix cache owns its own snapshots. The restore copy
    # has no remaining consumer; do not retain another full recurrent state
    # while the observer records all layers below.
    del old, frozen
    candidate = []
    independent = []
    original_norm = stage.body.hidden_norm

    def norm_launch(x, w, y):
        original_norm.launch(x, w, y)
        candidate.append((x[:m].detach().clone(), y[:m].detach().clone()))

    stage.body.hidden_norm = SimpleNamespace(
        launch=norm_launch,
    )
    from .linear import ProducerMismatch, Recorder
    from .nonlinear import AttentionRecorder as NonlinearRecorder

    recorder = support.LinearRecorder(owned) if owned is not None else Recorder()
    Replay, Reference, NonlinearReplay = (
        support.Replay,
        support.Reference,
        support.NonlinearReplay,
    )
    nonlinear = NonlinearRecorder(table_pages)
    # Only logical owners are mathematical inputs to this witness. Padding is
    # poisoned and checked separately; retaining its recurrent copies until
    # after the forward can double the audit's largest allocation.
    nonlinear.logical_m = m
    tp_recorder = tp_observer.Recorder()
    tp_recorder.names = {id(w): n for n, w in weights._weights.items()}
    with (
        tp_recorder.observe(stage, recorder),
        recorder.observe(),
        nonlinear.observe(stage.body, recorder),
    ):
        stage.launch()
    torch.cuda.synchronize()
    import torch.distributed as dist
    embedding_schedule = (
        recorder.immutable_embedding_group is not None,
        [(tuple(x.shape), str(x.dtype), tuple(y.shape), str(y.dtype))
         for _, x, y in recorder.embeddings],
    )
    schedules = [None] * dist.get_world_size(groups["tp"])
    dist.all_gather_object(schedules, embedding_schedule, group=groups["tp"])
    if any(s != embedding_schedule for s in schedules):
        raise RuntimeError("embedding observation schedule differs across ranks")
    if owned is not None:
        tp_recorder.validate(groups["tp"], m, cap, expert_groups=groups)
    else:
        tp_recorder.validate(groups["tp"], m, cap)
    if owned is not None:
        support.local_snapshots(recorder, nonlinear, tp_recorder, owned)
        assert len(recorder.experts) == len(tp_recorder.ep_events), "EP observer coverage"
    else:
        if stage.topology.tp_size == 1:
            tp_recorder.routes = [(record[-1], record[-1]) for record in recorder.experts]
        for i, event in enumerate(nonlinear.events["routes"]):
            nonlinear.events["routes"][i] = (*event[:2], tp_recorder.routes[i][1], event[3])
        assert len(recorder.experts) == len(tp_recorder.routes), "route observer coverage"
        for i, snapshot in enumerate(recorder.experts):
            recorder.experts[i] = (*snapshot[:-1], tp_recorder.routes[i][0])
    if encoded:
        for old, actual_storage in zip(graph_storage, storage, strict=True):
            assert byte_exact(old, actual_storage), "observer changed encoded graph storage"
        current_keys = torch.stack([event[1][:m] for event in nonlinear.events["attention"]], dim=1)
        current_values = torch.stack(
            [event[2][:m] for event in nonlinear.events["attention"]], dim=1)
        graph_key = torch.cat((native_prefix("key"), current_keys[:, :, None]), dim=2)
        graph_value = torch.cat((native_prefix("value"), current_values[:, :, None]), dim=2)
        for index, event in enumerate(nonlinear.events["attention"]):
            nonlinear.events["attention"][index] = (
                event[0], graph_key[:, index], graph_value[:, index], *event[3:])
        for physical in logical_pages.flatten():
            index = int(physical)
            arena.layout.validate_page_storage(
                arena.key[index], arena.value[index], arena.written[index],
                {name: data[index] for name, data in arena.page_fields.items()},
                device=arena.device)
    retain_logical_rows(recorder, nonlinear, m)
    assert all(
        (byte_exact(v, nonlinear.events["conv"][i][-1]) for i, v in enumerate(graph_conv))
    ), "graph conv identity"
    assert all(
        (byte_exact(v, nonlinear.events["gdn"][i][7]) for i, v in enumerate(graph_recurrent))
    ), "graph recurrent identity"
    if not encoded:
        for row in range(m):
            rows = slice(row, row + 1)
            assert byte_exact(
                graph_key[rows], kv_rows(arena.key, row_slice=rows)[:, :, : prefix_length + 1]
            ) and byte_exact(
                graph_value[rows], kv_rows(arena.value, row_slice=rows)[:, :, : prefix_length + 1]
            ), ("graph KV identity", row)
    assert byte_exact(actual, b.logits[:m]), "observer changed graph body output"
    recorder.freeze()
    nonlinear.freeze()
    recorder.attention_events = nonlinear.events["attention"]
    recorder.attention_meta = nonlinear.attention_meta
    recorder.cache_reads = nonlinear.cache_reads
    assert bool((b.row_activity[:m] == 1).all()) and (not bool(b.row_activity[m:].any())), (
        "logical activity map"
    )
    actual_written = (arena.written[: arena.spec.page_capacity] if written_indices is None
                      else arena.written.index_select(0, written_indices))
    expected_writes = torch.zeros_like(actual_written)
    for token in range(prefix_length + 1):
        expected_writes[logical_pages[:, token // a.page_size].long(), :, token % a.page_size] = (
            True
        )
    assert byte_exact(expected_writes, actual_written), (
        "logical KV write set and unowned pages"
    )
    tp_recorder.retain_logical(m, cap)
    tp_recorder.attach_expert_scales(recorder)
    tp_recorder.map_outputs(recorder)
    assert len(candidate) == 2 * cfg.graph.text.layer_count + 1
    from .replay import reference_model_for_token_domain
    model = reference_model_for_token_domain(stage.body.model, cfg.graph.text.vocabulary_size)
    expert_rows = {
        id(weight)
        for name, weight in weights._weights.items()
        if ".experts." in name and name.endswith(".down_proj.weight")
    }
    reference = (
        Reference(
            model.default_operators.row_weights, stage._vocab_span, groups["tp"],
            ep_group=groups.get("ep", groups["tp"]),
            etp_group=groups.get("etp", groups["tp"]),
            expert_rows=expert_rows,
        )
        if owned is not None else
        Reference(model.default_operators.row_weights, stage._vocab_span, groups["tp"])
    )
    if encoded:
        from cubic_serve.ops.eager.kv_cache import ReferenceCacheOperators
        reference = ReferenceCacheOperators(reference, arena.layout.storage)
    model.default_operators = reference
    old_layer = Qwen3_5ReferenceModel.forward_layer

    def reference_layer(self, *args, **kw):
        layer = kw["layer_index"]

        def observe(name, value):
            if name in ("layer_input", "input_norm", "attention", "mlp_input", "mlp"):
                independent.append((layer, name, value.detach().clone()))

        return old_layer(self, *args, **kw, observer=observe)

    reference_cache = (continuation.get("reference", prefix_cache)
                       if continuation is not None else prefix_cache)
    reference_key = (None if independent_trajectory is None else
                     (prefix_length, tuple(ids.cpu().tolist())))
    cached = (None if independent_trajectory is None else _reference_lookup(
        independent_trajectory, reference_key, reference_cache, groups["tp"],
    ))
    with ReferenceExecutionContext():
        if cached is None:
            with patch.object(Qwen3_5ReferenceModel, "forward_layer", reference_layer):
                expected = model.forward(ids[:, None], cache=reference_cache, operators=reference,
                    **({} if not position_axes else {'position_ids': torch.tensor(
                        position_axes, device=rank, dtype=torch.int64).T[:, :, None]}))
        else:
            expected, observations = cached
            independent.extend(observations)
    torch.cuda.synchronize()
    reference_witness = (None if independent_trajectory is None or cached is not None else
                         (reference_cache, expected, tuple(independent)))

    def delta(x, y):
        x = x.reshape(-1).float()
        y = y.reshape(-1).float()
        d = (x - y).abs()
        return {
            "max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "unequal": int((x != y).sum()),
            "elements": x.numel(),
        }

    layers = []
    observations = {(layer, name): value for layer, name, value in independent}
    for layer in range(cfg.graph.text.layer_count):
        x, y = candidate[2 * layer]
        after, z = candidate[2 * layer + 1]
        layers.append(
            {
                "layer": layer,
                "input": delta(x, observations[layer, "layer_input"]),
                "input_norm": delta(y, observations[layer, "input_norm"]),
                "after_attention": delta(after, observations[layer, "attention"]),
                "mlp_input": delta(z, observations[layer, "mlp_input"]),
            }
        )
    report.update(
        status="diagnostic_complete",
        logits=delta(actual, expected.logits),
        layers=layers,
        candidate_top1=int(actual.argmax()),
        reference_top1=int(expected.logits.argmax()),
        candidate_layer_events=len(candidate),
        oracle=(
            "QuantizedReferenceOperators + Qwen3_5ReferenceModel complete forward "
            "on original packed weights"
        ),
    )
    # The unrestricted reference has now produced its scalar diagnostic.
    # Its returned full cache is not the frozen input for causal replay.
    # Release that second trajectory before constructing the witnessed one.
    if continuation is not None:
        continuation["reference"] = expected.cache
        if completed_transition is not None:
            independent_logits = expected.logits.detach().to('cpu', copy=True)
            independent_feedback = expected.hidden_states.detach().to('cpu', copy=True)
    del expected
    independent.clear()
    observations.clear()
    candidate.clear()
    replay = Replay(
        recorder, {id(w): n for n, w in weights._weights.items()}, tp_recorder, stage._vocab_span
    )
    model.default_operators = replay
    try:
        nonlin_replay = NonlinearReplay(nonlinear)
        if owned is not None:
            nonlin_replay.ep_events = tp_recorder.ep_events
        with ReferenceExecutionContext(), nonlin_replay.install(), replay.scale_witness():
            witnessed = model.forward(ids[:, None], cache=prefix_cache, operators=replay,
                **({} if not position_axes else {'position_ids': torch.tensor(
                    position_axes, device=rank, dtype=torch.int64).T[:, :, None]}))
        assert replay.attention_used == len(recorder.attention_events), (
            "unconsumed attention events"
        )
        assert replay.embedding_used == len(recorder.embeddings), "unconsumed embedding events"
        assert len(replay.used) == sum(len(v) for v in recorder.events.values()), (
            "unconsumed linear events"
        )
        for kind in ("zero", "conv", "gdn", "norm", "attention_gate", "combine"):
            assert nonlin_replay.indices[kind] == len(nonlinear.events[kind]), (
                "unconsumed nonlinear events " + kind
            )
        assert len(nonlin_replay.route_reference) == len(nonlinear.events["routes"]), (
            "unconsumed route events"
        )
        gi = fi = 0
        for layer_cache in witnessed.cache.layers:
            if layer_cache.convolution is not None:
                assert byte_exact(layer_cache.convolution, graph_conv[gi]), (
                    "causal convolution state"
                )
                assert byte_exact(layer_cache.recurrent, graph_recurrent[gi]), (
                    "causal recurrent state"
                )
                gi += 1
            else:
                assert byte_exact(
                    layer_cache.key.transpose(1, 2).reshape_as(graph_key[:, fi]), graph_key[:, fi]
                ), "causal KV key"
                assert byte_exact(
                    layer_cache.value.transpose(1, 2).reshape_as(graph_value[:, fi]),
                    graph_value[:, fi],
                ), "causal KV value"
                if encoded:
                    from cubic_serve.runtime.cache.quantization import CacheGroupAxis
                    for side, native, spec in (
                        ("key", graph_key[:, fi], arena.layout.storage.key),
                        ("value", graph_value[:, fi], arena.layout.storage.value),
                    ):
                        if spec.dtype.quantized and spec.axis is CacheGroupAxis.TOKENS:
                            tail = arena.states[f"kv_{side}_tail"]
                            for row in range(m):
                                owner = int(b.conv_slots[1, 0, row]) // b.conv_slots.shape[1]
                                wanted = torch.zeros_like(tail[owner, fi])
                                remaining = (prefix_length + 1) % spec.group_size
                                if remaining:
                                    values = native[row, -remaining:]
                                    if arena.layout.storage.enable_rotation:
                                        from cubic_serve.ops.eager.rotation import (
                                            rotate_cache_reference,
                                        )

                                        with ReferenceExecutionContext():
                                            values = rotate_cache_reference(values)
                                    wanted[:remaining] = values
                                assert byte_exact(tail[owner, fi], wanted), (
                                    "causal native tail checkpoint", side, row, fi)
                fi += 1
        assert byte_exact(actual.reshape_as(witnessed.logits), witnessed.logits), (
            "causal final logits"
        )
        if owned is not None:
            assert replay.ep_used == len(recorder.experts), "EP output witness schedule"
        report["causal_replay"] = {
            "status": "complete",
            "linear_checks": len(replay.reports),
            "embedding_checks": replay.embedding_used,
            "logits_exact": True,
            "complete_event_consumption": True,
            "KV_and_recurrent_and_conv_state_exact": True,
        }
    except ProducerMismatch:
        report["causal_replay"] = {
            "status": "unwitnessed_producer",
            "first_boundary": replay.failure,
        }
    except (AssertionError, RuntimeError) as error:
        report["causal_replay"] = {
            "status": "diagnostic_incomplete",
            "error": str(error),
            "linear_checks": len(replay.reports),
        }
    report["nonlinear_proved"] = nonlin_replay.proved
    report["routed_activation_proofs"] = replay.activation_proofs
    report["scope"] = (
        f"{cfg.graph.text.layer_count}-layer model scope, TP{a.tp} rank{rank} "
        f"logicalM{m}/physicalM{cap} step{prefix_length + 1} "
        f"from identical frozen prefix; page_size{a.page_size}"
    )
    report["prefix_length"] = prefix_length
    report["page_size"] = a.page_size
    report["table_pages"] = table_pages
    report["full_model"] = a.full_model
    report["reversed_physical_kv_pages"] = True
    report["logical_m"] = m
    report["physical_m"] = cap
    report["padding_poison"] = a.poison_padding
    report["logical_activity_and_kv_write_set_checked"] = True
    report["collective_proofs"] = tp_recorder.proofs
    report["semantic_collective_schedule"] = tp_recorder.schedule
    report["expert_complete_k_scale_events"] = len(tp_recorder.expert_scales)
    report["fold_proofs"] = tp_recorder.fold_proofs
    report["fold_observation_schedule"] = tp_recorder.fold_schedule
    report["attention_proofs"] = replay.attention_proofs
    report["nonlinear_intervals"] = nonlinear.proofs
    if report["causal_replay"]["status"] != "complete":
        raise AssertionError(
            "independent producer proof incomplete: " + str(report["causal_replay"])
        )
    if continuation is not None:
        continuation["candidate"] = witnessed.cache
        if encoded or completed_transition is not None:
            for old, tensor in zip(graph_storage, storage, strict=True):
                assert byte_exact(old, tensor), "deferred reference changed physical cache storage"
            continuation["storage"] = graph_storage
        report["stored_state_continuation_exact"] = True
    report["status"] = "passed"
    if independent_trajectory is not None:
        if cached is None:
            independent_trajectory.retain(reference_key, *reference_witness)
        report["independent_reference_reused"] = cached is not None
    if completed_step_witness is not None:
        from cubic_serve.guardrail.compare import tensor_digest

        report["linear_operand_digests"] = [
            dict(name=weights_name, occurrence=index, input=tensor_digest(x),
                 output=tensor_digest(y), input_dtype=str(x.dtype), output_dtype=str(y.dtype))
            for wid, events in recorder.events.items()
            for index, (_, x, y) in enumerate(events)
            for weights_name in (replay.names[wid],)
        ]
        completed_step_witness(prefix_length, stage, report, recorder, tp_recorder)
    if completed_transition is not None:
        from cubic_serve.guardrail.compare import tensor_digest
        store, binding, consumer = completed_transition
        continuation['position'] = prefix_length + 1
        continuation['restore_native'] = False
        boundaries = dict(logits=actual.detach().to('cpu', copy=True),
                          draft_feedback=b.norm[:m].detach().to('cpu', copy=True))
        report['candidate_boundary_continuation_state_and_logits_exact'] = True
        report['independent_trajectory_diagnostic'] = dict(
            route_branch_schedule_exact=True, logits=report['logits'],
            scope='separately advancing original-weight dense reference')
        report['boundary_digests'] = {name: tensor_digest(value)
                                      for name, value in boundaries.items()}
        consumer(dict(position=prefix_length, token=int(ids[0]), report=report,
            prefix=store.freeze(binding, continuation), boundaries=boundaries,
            independent_logits=independent_logits, independent_feedback=independent_feedback))
    return report
