"""Negative controls for independent packed graph qualification."""

import hashlib
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from cubic_serve.executor.runner import ReferenceExecutionContext
from cubic_serve.guardrail.quantization import ReferenceMathMemo
from cubic_serve.guardrail.quantized_graph import ep
from cubic_serve.guardrail.quantized_graph.bounds import check
from cubic_serve.guardrail.quantized_graph.nonlinear import ColdReplay
from cubic_serve.models.adapter import ModelDefinition
from cubic_serve.models.quantization import QuantizationScheme
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.models.qwen3_5_moe import model
from cubic_serve.ops.eager.quantization import QuantizedReferenceOperators
from cubic_serve.quantization.extensions.int.layout import pack


@pytest.mark.parametrize("flattened_ndim,trailing", [(3, (5, 3)), (4, (2, 3, 4))])
@pytest.mark.parametrize("layers", [3, 48])
def test_layered_audit_state_snapshots_decompose_logical_rows_and_remain_immutable(
        flattened_ndim, trailing, layers):
    import math
    from cubic_serve.guardrail.quantized_graph.nonlinear import state_rows
    shape = (2, layers, *trailing)
    strides = list(torch.empty(shape).stride())
    strides[0] += 7
    raw = torch.arange(2 * strides[0], dtype=torch.float32)
    field = raw.as_strided(shape, strides)
    indices = torch.tensor([2 * layers - 1, 0, layers - 1, layers])
    with pytest.raises(IndexError):
        field[indices]  # the old flattened-index operation is invalid
    flat = field.contiguous().reshape(2 * layers, *trailing)
    expected = flat[indices].clone()
    observed = state_rows(field, indices, flattened_ndim)
    contiguous = state_rows(flat, indices, flattened_ndim)
    raw.fill_(float("nan")); flat.zero_()
    assert torch.equal(observed, expected) and torch.equal(contiguous, expected)
    assert observed.numel() == 4 * math.prod(trailing)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("scratch", [1, 4])
def test_numerical_witness_preserves_shared_field_pitches_and_protects_live_pages(index, scratch):
    from cubic_serve.guardrail.quantized_graph.text import _audit_cache_arena
    from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout, StateSlotSpec
    from cubic_serve.runtime.cache.shared_pool import SharedCacheStorage, SharedCacheField

    schemas = tuple((CacheArenaSpec(str(i), 4, 4, (
        StateSlotSpec("history", torch.int64, (3,), initial_value=-1),
        StateSlotSpec("matrix", torch.bfloat16, (4 + i, 8)),
    ), scratch_capacity=128), KVLayout((i,), 2, 1, 4 + i)) for i in range(2))
    storage = SharedCacheStorage(schemas, budget_bytes=65536, device="cpu",
                                extra_state_fields=(SharedCacheField("feedback", torch.bfloat16, (7,)),))
    original = CacheArena(storage.arena_specs[index], schemas[index][1], "cpu", rank_id=0,
                          shared_storage=storage, shared_index=index)
    before = storage.backing.clone()
    witness = _audit_cache_arena(SimpleNamespace(arena=original), 3, scratch_capacity=scratch)
    assert witness.shared_storage is not storage
    assert witness.spec.state_slot_capacity == 1
    assert witness.spec.scratch_capacity == scratch
    assert witness.key.stride() == original.key.stride()
    for name in original.states:
        assert witness.states[name].stride() == original.states[name].stride()
    witness.key[:3].fill_(19)
    witness.value[:3].fill_(23)
    for name, value in witness.states.items():
        value[0].fill_(7)
    assert (witness.key[:3] == 19).all() and (witness.value[:3] == 23).all()
    assert torch.equal(storage.backing, before)
    assert witness.shared_storage.allocator.live_bits == {"page": 7, "state": 1}
    assert witness.shared_storage.backing.numel() < storage.backing.numel()

    from cubic_serve.guardrail.quantized_graph import _packed_written_indices
    indices = _packed_written_indices(witness, 2, 1)
    pool = witness.shared_storage.allocator
    start, stop = pool.state_offset_bytes, pool.state_offset_bytes + pool.state_bytes
    expected = [p for p in range(witness.spec.page_capacity)
                if p < 2 or (p + 1) * pool.page_bytes <= start or p * pool.page_bytes >= stop]
    assert indices.tolist() == expected and 2 in expected  # a real foreign guard page
    states = {name: value[0].clone() for name, value in witness.states.items()}
    witness.written.index_fill_(0, indices, False)
    assert all(torch.equal(witness.states[name][0], value) for name, value in states.items())
    overlap = storage.allocator.state_offset_bytes // storage.allocator.page_bytes + 1
    with pytest.raises(ValueError, match="inputs overlap"):
        _packed_written_indices(original, overlap, 1)
    assert torch.equal(storage.backing, before)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 65, 66, 255, 256])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16,
                                    torch.int8, torch.int64])
def test_repeated_audit_rows_preserve_full_permutation_digests(device, m, reverse, strided, dtype):
    from cubic_serve.guardrail.quantized_graph import byte_exact
    from cubic_serve.guardrail.quantized_graph.cross_m import _cpu_rows, _trajectory_rows

    if device == "cuda" and os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1":
        pytest.skip("explicit CUDA audit required")
    # Include different NaN payloads and signed zero;no floating equality suffices.
    bits = torch.tensor([0, -2147483648, 2143289345, 2143289346,
                         1065353216, -1082130432, 1, -1], dtype=torch.int32)
    original = bits.reshape(2, 4).repeat((m + 1) // 2, 1)[:m].to(device)
    value = original.view(dtype)
    if strided:
        value = value.repeat_interleave(2, dim=1)[:, ::2]
    unpermuted = value.cpu().clone()
    if reverse:
        value = value.flip(0)
        if strided:
            value = value.repeat_interleave(2, dim=1)[:, ::2]
    full = value.cpu().clone()
    compact = _cpu_rows(value, m, "field", repeated_corpus=True, reverse=reverse)
    count = min(m, 2)
    first = m - count if reverse else 0
    assert byte_exact(compact, full[first:first + count])
    sampled = _trajectory_rows({"field": compact}, count)
    expanded = tuple(sampled[(row - first) % count] for row in range(m))
    assert expanded == _trajectory_rows({"field": full}, m)
    if reverse:
        assert byte_exact(compact.flip(0), unpermuted[:count])
    value.zero_()
    assert byte_exact(compact, full[first:first + count])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("bits", [1, -2147483648, 2143289345])
def test_repeated_audit_rows_reject_unrepresented_row_corruption_before_copy(device, bits):
    from cubic_serve.guardrail.quantized_graph.cross_m import _cpu_rows

    if device == "cuda" and os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1":
        pytest.skip("explicit CUDA audit required")
    value = torch.zeros((65, 7), device=device)
    value.view(torch.int32)[33, 6] = bits
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("premature CPU copy")),
        pytest.raises(RuntimeError, match="corpus boundary witness: corrupted-state"),
    ):
        _cpu_rows(value, 65, "corrupted-state", repeated_corpus=True)


def test_repeated_audit_rows_keep_distinct_ownership_rows_and_unaligned_storage():
    from cubic_serve.guardrail.quantized_graph import byte_exact
    from cubic_serve.guardrail.quantized_graph.cross_m import _cpu_rows

    distinct = torch.arange(35, dtype=torch.int64).reshape(5, 7)
    full = _cpu_rows(distinct, 5, "ownership")
    assert byte_exact(full, distinct) and full.data_ptr() != distinct.data_ptr()
    # A contiguous view can have an offset unaligned to the widest integer word.
    source = torch.zeros(1 + 5 * 4, dtype=torch.int16)
    offset = source[1:].reshape(5, 4)
    compact = _cpu_rows(offset, 5, "offset", repeated_corpus=True)
    assert byte_exact(compact, offset[:2])


@pytest.mark.parametrize("order", [(1, 8, 3, 7), (8, 1, 7, 3)])
def test_cross_m_common_snapshots_share_only_proved_immutable_prefixes(order):
    from cubic_serve.guardrail.quantized_graph import byte_exact
    from cubic_serve.guardrail.quantized_graph.cross_m import _retain_common

    # Include storage distinctions that floating equality cannot establish.
    bits = torch.tensor([0, -2147483648, 2143289345, 2143289346], dtype=torch.int32)
    original = {"output": bits.repeat(8).view(torch.float32).reshape(8, 4),
                "state": torch.arange(24, dtype=torch.int64).reshape(8, 3)}
    common, frozen = [], []
    for m in order:
        actual = {key: value[:m].clone() for key, value in original.items()}
        comparisons = [{"differing_fields": [key for key in actual if not
                        byte_exact(previous[key][:min(n, m)], actual[key][:min(n, m)])]}
                       for n, _, previous in common]
        frozen.append({key: value.clone() for key, value in actual.items()})
        _retain_common(common, m, 1 << (m - 1).bit_length(), actual, comparisons)
    assert [(m, cap) for m, cap, _ in common] == [
        (m, 1 << (m - 1).bit_length()) for m in order
    ]
    for (_, _, snapshot), expected in zip(common, frozen, strict=True):
        assert snapshot.keys() == expected.keys()
        assert all(byte_exact(snapshot[key], value) for key, value in expected.items())
    for key in original:
        assert len({snapshot[key].untyped_storage().data_ptr()
                    for _, _, snapshot in common}) == 1
    assert all(byte_exact(value, frozen[order.index(8)][key])
               for key, value in original.items())


@pytest.mark.parametrize("field", ["output", "state"])
@pytest.mark.parametrize("bit", [1, -2147483648])
def test_cross_m_common_snapshots_preserve_each_differing_field(field, bit):
    from cubic_serve.guardrail.quantized_graph import byte_exact
    from cubic_serve.guardrail.quantized_graph.cross_m import _retain_common

    previous = {"output": torch.zeros((2, 4)), "state": torch.zeros((2, 4))}
    actual = {key: value.clone() for key, value in previous.items()}
    actual[field].view(torch.int32)[0, 0] = bit
    common = [(2, 4, previous)]
    differences = [key for key in actual if not byte_exact(previous[key], actual[key])]
    assert differences == [field]
    _retain_common(common, 2, 8, actual, [{"differing_fields": differences}])
    assert [(m, cap) for m, cap, _ in common] == [(2, 4), (2, 8)]
    assert common[0][2] is previous and common[1][2] is actual
    assert all(previous[key].untyped_storage().data_ptr() !=
               actual[key].untyped_storage().data_ptr() for key in actual)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_batched_swiglu_witness_preserves_scalar_receipts_and_freezes_operands(
    dtype, activation_dtype,
):
    from cubic_serve.guardrail.quantized_graph.bounds import (
        SwiGLUWitnessBatch,
        swiglu_bounds,
    )

    batch = SwiGLUWitnessBatch(max_pending=3, max_bytes=4096)
    receipts, expected = [], []
    generator = torch.Generator().manual_seed(741)
    for ordinal, shape in enumerate(((1, 17), (2, 9), (1, 160), (3, 7), (1, 320))):
        gate = torch.randn(shape, generator=generator).to(dtype)
        up = torch.randn(shape, generator=generator).to(dtype)
        lo, hi = swiglu_bounds(gate, up, activation_dtype=activation_dtype)
        actual = lo.to(dtype)
        expected.append(check(actual, lo, hi))
        receipts.append(batch.add(gate, up, actual, activation_dtype=activation_dtype,
                                  consumer=str(ordinal)))
        gate.fill_(float("nan"))
        up.zero_()
        actual.fill_(float("inf"))
    batch.finish()
    assert not batch.pending and batch.bytes == 0
    for receipt, proof in zip(receipts, expected, strict=True):
        assert receipt == {"kind": "swiglu", "consumer": receipt["consumer"], **proof}


@pytest.mark.parametrize("corruption", ["outside", "nonfinite_gate", "nonfinite_output"])
def test_batched_swiglu_witness_rejects_corruption_after_bounded_flush(corruption):
    from cubic_serve.guardrail.quantized_graph.bounds import (
        SwiGLUWitnessBatch,
        swiglu_bounds,
    )

    batch = SwiGLUWitnessBatch(max_pending=1)
    gate = torch.ones((1, 7), dtype=torch.bfloat16)
    up = gate.clone()
    lo, _ = swiglu_bounds(gate, up, activation_dtype=gate.dtype)
    actual = lo.to(gate.dtype)
    if corruption == "outside":
        actual[0, 0] = 32
    elif corruption == "nonfinite_gate":
        gate[0, 0] = float("nan")
    else:
        actual[0, 0] = float("nan")
    batch.add(gate, up, actual, activation_dtype=gate.dtype, consumer="bad")
    assert not batch.pending
    with pytest.raises(AssertionError, match="swiglu_witness_failures"):
        batch.finish()


def test_cross_m_hash_difference_requires_numerical_attribution_and_stays_closed():
    from cubic_serve.guardrail.quantized_graph.text_cross_m import (
        _NumericalAttributionUnavailable, _require_m1_identity,
    )

    _require_m1_identity({"state": "same"}, {"state": "same"}, (2, 1, 0, 0))
    with pytest.raises(_NumericalAttributionUnavailable, match="numerical attribution unavailable"):
        _require_m1_identity({"state": "different"}, {"state": "same"}, (2, 1, 0, 0))
    with pytest.raises(ValueError, match="inventory differs"):
        _require_m1_identity({"state": "same"}, {"output": "same"}, (2, 1, 0, 0))


@pytest.mark.parametrize("consumed", [1, 3, 4, 5, 8])
@pytest.mark.parametrize("swap_axes", [False, True])
def test_segment_snapshot_masks_logical_tokens_in_each_format_geometry(consumed, swap_axes):
    from cubic_serve.guardrail.quantized_graph import text_cross_m
    from cubic_serve.runtime.cache import KVLayout
    from cubic_serve.runtime.cache.pages import PageFieldSpec
    from cubic_serve.runtime.cache.quantization import (
        CacheDType, CacheGroupAxis, CacheStorageSpec, KVCacheStorageSpec,
    )
    axes = (CacheGroupAxis.TOKENS, CacheGroupAxis.CHANNELS)
    if swap_axes:
        axes = axes[::-1]
    layout = KVLayout((0,), 8, 1, 7, storage=KVCacheStorageSpec(
        CacheStorageSpec(CacheDType.INT6, axes[0], 4),
        CacheStorageSpec(CacheDType.FP8, axes[1], 4)),
        page_fields=(PageFieldSpec("index", torch.bfloat16, (2, 3)),))
    snapshot = {"key": torch.ones((1, *layout.storage_shape(key=True)), dtype=torch.uint8),
                "value": torch.ones((1, *layout.storage_shape(key=False)), dtype=torch.uint8),
                "written": torch.ones((1, 1, 8), dtype=torch.bool)}
    snapshot.update({f"page:{spec.field_id}": torch.ones((1, *spec.shape), dtype=spec.dtype)
                     for spec in layout.page_fields})
    stage = SimpleNamespace(arena=SimpleNamespace(layout=layout),
                            indexer_plan=SimpleNamespace(compression=4, size=2, rank=0))
    with patch.object(text_cross_m, "boundary_snapshot", return_value=snapshot):
        text_cross_m._segment_snapshot(stage, 0, consumed)
    assert snapshot["written"].tolist() == [[[p < consumed for p in range(8)]]]
    for name, grouped in (("key", layout.key_encoding), ("value", layout.value_encoding)):
        mask = snapshot[name][0, 0, :, 0].bool()
        for t in range(8):
            for d in range(7):
                group, _ = grouped.group_location(t, 0, d)
                valid = ((t // 4 + 1) * 4 <= consumed
                         if grouped.spec.axis is CacheGroupAxis.TOKENS else t < consumed)
                assert bool(mask[group]) == valid
        assert torch.equal(snapshot[f"page:kv_{name}_scales"][0, 0].bool(), mask)


@pytest.mark.parametrize("kind", ["harness", "execution", "numerical"])
def test_cross_m_reports_harness_execution_and_numerical_failures_separately(kind):
    from cubic_serve.guardrail.quantized_graph import text_cross_m

    def gather(output, error, **kwargs):
        output[0] = error

    with (
        patch.object(text_cross_m.dist, "get_world_size", return_value=1),
        patch.object(text_cross_m.dist, "all_gather_object", side_effect=gather),
        pytest.raises(AssertionError if kind == "numerical" else RuntimeError,
                      match=f"cross-M {kind} failure"),
    ):
        text_cross_m._coordinate((kind, "fixture"), object())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_protected_state_initial_value_compares_storage_bits_including_signed_zero(dtype):
    from cubic_serve.guardrail.quantized_graph.text_cross_m import _initial_bytes_equal

    value = torch.zeros((3, 4), dtype=dtype)
    assert _initial_bytes_equal(value, 0)
    assert _initial_bytes_equal(value[:0], 0)
    value[1, 2] = -0.0
    assert bool((value == 0).all())
    assert not _initial_bytes_equal(value, 0)
    assert _initial_bytes_equal(torch.full((3, 4), -0.0, dtype=dtype), -0.0)


def _reference_expert_schedules():
    topology = SimpleNamespace(
        tp_size=4, ep_size=2, expert_coordinates=lambda rank: (rank // 2, rank % 2),
    )
    decisions = {"layer.mlp": ((0, 3), (1, 3))}
    schedules = []
    for rank in range(4):
        events = [("layer.mlp.gate.weight", 0, (2, 8), (2, 4))]
        counts = {}
        for row in decisions["layer.mlp"]:
            for eid in row:
                if eid // 2 == rank // 2:
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        name = f"layer.mlp.experts.{eid}.{projection}.weight"
                        ordinal = counts.get(name, 0)
                        events.append((name, ordinal, (1, 8), (1, 8)))
                        counts[name] = ordinal + 1
        schedules.append((events, decisions.copy()))
    return topology, schedules


def test_independent_expert_oracle_checks_checkpoint_owners_not_equal_rank_work():
    from cubic_serve.guardrail.quantized_graph.text import _expert_reference_schedule_error

    topology, schedules = _reference_expert_schedules()
    assert schedules[0][0] != schedules[2][0]
    assert _expert_reference_schedule_error(schedules, topology, 4) is None


def test_independent_expert_oracle_proves_structure_before_producing_route_math():
    from cubic_serve.guardrail.quantized_graph.text import _expert_reference_schedule_error

    topology, schedules = _reference_expert_schedules()
    structural = [(events, None) for events, _ in schedules]
    assert _expert_reference_schedule_error(structural, topology, 4) is None
    structural[0][0][1] = ("layer.mlp.experts.0.gate_proj.weight", 2, (1, 8), (1, 8))
    assert _expert_reference_schedule_error(structural, topology, 4) is not None


@pytest.mark.parametrize("corruption", ["missing", "foreign", "ordinal", "shape", "route"])
def test_independent_expert_oracle_rejects_changed_ownership_or_producing_events(corruption):
    from cubic_serve.guardrail.quantized_graph.text import _expert_reference_schedule_error

    topology, schedules = _reference_expert_schedules()
    if corruption == "missing":
        schedules[0][0].pop()
    elif corruption == "foreign":
        schedules[0][0].append(("layer.mlp.experts.3.gate_proj.weight", 0, (1, 8), (1, 8)))
    elif corruption == "ordinal":
        for rank in (0, 1):
            event = schedules[rank][0][1]
            schedules[rank][0][1] = (event[0], 2, event[2], event[3])
    elif corruption == "shape":
        schedules[0][0][1] = (*schedules[0][0][1][:2], (1, 4), (1, 8))
    else:
        schedules[0][1]["layer.mlp"] = ((1, 3), (1, 3))
    assert _expert_reference_schedule_error(schedules, topology, 4) is not None


def test_cross_m_boundary_hashes_keep_reads_serial_copies_owned_and_rows_ordered():
    import threading

    from cubic_serve.guardrail.quantized_graph.text_cross_m import _boundary_witnesses

    observer_thread = threading.get_ident()
    original = torch.arange(8 * 16, dtype=torch.int64).reshape(8, 16) + 2**54
    reads = []
    def snapshot(stage, logical):
        assert threading.get_ident() == observer_thread, "GPU reads stay on observer thread"
        reads.append(logical)
        return {"state": original[logical - 1:logical].clone()}

    rows = (7, 2, 5, 0)
    with patch("cubic_serve.guardrail.quantized_graph.text_cross_m.boundary_snapshot", snapshot):
        serial = _boundary_witnesses(None, rows, max_workers=1)
        parallel = _boundary_witnesses(None, rows, max_workers=4)
    assert parallel == serial
    assert reads == [row + 1 for row in rows] * 2


@pytest.mark.parametrize("rows", ((0,), (7,), (0, 1, 2), (7, 6, 5), (7, 2, 5)))
def test_cross_m_host_events_preserve_bits_row_maps_and_expert_ownership(rows):
    from cubic_serve.guardrail.quantized_graph.text_cross_m import (
        _host_events,
        _intermediate_witness,
    )

    capacity = 8
    stage = SimpleNamespace(
        captured_m=capacity, rank=1,
        topology=SimpleNamespace(ep_size=2, expert_coordinates=lambda rank: (rank, 0)),
        config=SimpleNamespace(text=SimpleNamespace(moe=SimpleNamespace(expert_count=8))),
    )
    ids = torch.tensor([[0, 5]]).expand(capacity, -1).clone()
    routes = torch.arange(capacity * 2 * 4).reshape(capacity, 2, 4).to(torch.bfloat16)
    routes[:, 0] = float("nan")  # Unowned expert lanes must stay semantically zero.
    prefix = "layer/moe"
    events = [(prefix + "/route_ids/0", ids), (prefix + "/routes/0", routes),
              (prefix + "/sum/0", torch.arange(capacity)[:, None] + 2**54)]
    expected = [_intermediate_witness(events, stage, row) for row in rows]
    host = _host_events(events, capacity, rows)
    assert expected == [
        _intermediate_witness(host, stage, row, snapshot_rows=len(rows))
        for row in range(len(rows))
    ]
    # A later replay may replace the captured event tensors; host witnesses own
    # their bytes and retain exact INT64 values, BF16 lanes and row ordering.
    ids.zero_()
    routes.zero_()
    assert expected == [
        _intermediate_witness(host, stage, row, snapshot_rows=len(rows))
        for row in range(len(rows))
    ]
    with pytest.raises(ValueError, match="row map"):
        _host_events(events, capacity, (capacity,))


def test_cross_m_observer_matches_semantic_sites_across_coalesced_transport():
    from cubic_serve.guardrail.quantized_graph.text_cross_m import (
        _intermediate_witness,
        _observe,
        _row_failure,
    )
    from cubic_serve.models.qwen4_exp.model import Qwen4ExpOperationBinder as Binder
    from cubic_serve.ops.cuda.collective import FixedRankBF16SumPrimitive as Sum
    from cubic_serve.ops.cuda.moe.quantized import PackedRoutedExpertsPrimitive as Route

    def capture(joint, capacity, row):
        events = []
        routes = torch.full((capacity, 2, 4), float("nan"))
        shared = torch.full((capacity, 4), float("nan"))
        routes[row] = torch.arange(8.).reshape(2, 4)
        shared[row] = torch.arange(4.) + 10
        op = SimpleNamespace(
            route_ids=torch.tensor([[0, 1]]).expand(capacity, -1).clone(),
            route_weights=torch.tensor([[.25, .75]]).expand(capacity, -1).clone(),
            routes=routes,
        )
        summed = SimpleNamespace(input_views=lambda value: (routes, shared) if joint else (routes,))

        def route(_op):
            Sum.launch(summed, routes)

        def forward():
            Route.launch(op)
            if not joint:
                Sum.launch(SimpleNamespace(input_views=lambda value: (shared,)), shared)

        with (
            patch.object(Sum, "launch", lambda *args: None),
            patch.object(Route, "launch", route),
            patch.object(Binder, "bind", lambda *args, **kwargs: forward),
            _observe(events),
        ):
            for layer in (3, 4):
                Binder.bind(None, "moe", f"model.layers.{layer}.mlp", layer_id=layer)()
        stage = SimpleNamespace(
            captured_m=capacity, rank=0,
            config=SimpleNamespace(text=SimpleNamespace(moe=SimpleNamespace(expert_count=2))),
            topology=SimpleNamespace(ep_size=1, expert_coordinates=lambda rank: (0, 0)),
        )
        return events, stage, _intermediate_witness(events, stage, row)

    first, _, anchor = capture(True, 1, 0)
    second, stage, actual = capture(False, 4, 3)
    assert [n for n, _ in first] != [n for n, _ in second]
    assert actual == anchor
    assert len(actual) == 10
    assert _row_failure(({}, actual), ({}, anchor)) is None
    # A changed bit still fails closed at its site; a digest is not a
    # numerical oracle for a potentially changed reduction plan.
    second[0][1][3, 0, 0] = -0.0
    damaged = _intermediate_witness(second, stage, 3)
    failure = _row_failure(({}, damaged), ({}, anchor))
    assert failure is not None and failure[0] == "harness"
    assert "numerical attribution unavailable" in failure[1]
    assert "model.layers.3.mlp/moe/sum/0" in failure[1]
    assert _row_failure(({}, actual[:-1]), ({}, anchor))[0] == "harness"
    with pytest.raises(ValueError, match="duplicate"):
        _intermediate_witness([*second, second[0]], stage, 3)


def test_cross_m_named_routes_mask_only_checkpoint_nonowners():
    from cubic_serve.guardrail.quantized_graph.text_cross_m import _intermediate_witness

    prefix = "model.layers.7.mlp/moe"
    ids = torch.tensor([[0, 3]])
    routes = torch.tensor([[[1., 2.], [float("nan"), float("nan")]]])
    events = [(prefix + "/routes/0", routes), (prefix + "/route_ids/0", ids)]
    stage = SimpleNamespace(
        captured_m=1, rank=0,
        config=SimpleNamespace(text=SimpleNamespace(moe=SimpleNamespace(expert_count=4))),
        topology=SimpleNamespace(ep_size=2, expert_coordinates=lambda rank: (0, 0)),
    )
    anchor = _intermediate_witness(events, stage, 0)
    routes[0, 1].fill_(123)
    assert _intermediate_witness(events, stage, 0) == anchor
    routes[0, 0, 0] += 1
    assert _intermediate_witness(events, stage, 0) != anchor


def test_fp32_convolution_witness_covers_real_bf16_midpoint_and_rejects_corruption():
    from cubic_serve.guardrail.quantized_graph.state_bounds import fp32_conv_silu_range

    # Frozen TP4 layer42 convolution: FP32 reduction order crosses a BF16
    # midpoint after SiLU. These are original operands, not fitted tolerances.
    x = torch.tensor([[0.33203125]], dtype=torch.bfloat16)
    state = torch.tensor([[[0.6171875, 0.53125, 0.6484375]]], dtype=x.dtype)
    weight = torch.tensor(
        [[4.029273986816406e-05, 0.0059814453125, 0.0458984375, -0.03759765625]],
        dtype=x.dtype,
    )
    lower, upper = fp32_conv_silu_range(x, weight, state, dtype=x.dtype)
    eager = torch.tensor([[0.01031494140625]], dtype=x.dtype)
    candidate = torch.tensor([[0.0103759765625]], dtype=x.dtype)
    assert not torch.equal(eager, candidate)
    assert check(eager, lower, upper)["outside_enclosure"] == 0
    assert check(candidate, lower, upper)["outside_enclosure"] == 0
    corrupted = torch.nextafter(candidate, torch.full_like(candidate, float("inf")))
    assert check(corrupted, lower, upper)["outside_enclosure"] == 1
    changed_weight = weight.clone()
    changed_weight[0, 0] += 0.25
    changed = fp32_conv_silu_range(x, changed_weight, state, dtype=x.dtype)
    assert check(candidate, *changed)["outside_enclosure"] == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp32_convolution_witness_handles_silu_turning_point_and_exp_overflow(dtype):
    from cubic_serve.guardrail.quantized_graph.state_bounds import fp32_conv_silu_range

    x = torch.tensor([[-200.0, -1.28, 0.0, 200.0]], dtype=dtype)
    weight = torch.zeros((4, 4), dtype=dtype)
    weight[:, -1] = 1
    state = torch.zeros((1, 4, 3), dtype=dtype)
    lower, upper = fp32_conv_silu_range(x, weight, state, dtype=dtype)
    independent = (x.double() / (1 + torch.exp(-x.double()))).float().to(dtype)
    assert check(independent, lower, upper)["outside_enclosure"] == 0


def test_immutable_embedding_observation_belongs_to_replay_not_reference():
    from cubic_serve.guardrail.quantized_graph import replay

    ids = torch.tensor([0, 3])
    weight = torch.tensor([[1., 2.], [3., 4.]])
    local_expected = torch.tensor([[1., 2.], [0., 0.]])
    expected = torch.tensor([[1., 2.], [7., 8.]])
    group = object()
    reference = replay.Reference(set(), range(0, 2), group)
    with patch.object(replay, "ordered_sum", return_value=expected) as reduce:
        assert torch.equal(reference.embedding(ids, weight, dtype=torch.float32), expected)
        assert torch.equal(reduce.call_args.args[0], local_expected)
        assert reduce.call_args.args[1] is group

    recorder = SimpleNamespace(
        events={}, activation_sources={}, encoded_inputs={},
        immutable_embedding_group=group, embeddings=[(weight, ids, expected.clone())],
        attention_events=[], attention_meta=[],
    )
    witness = replay.Replay(recorder, {}, SimpleNamespace(), range(0, 2))
    with patch.object(replay, "ordered_sum", return_value=expected):
        assert torch.equal(witness.embedding(ids, weight, dtype=torch.float32), expected)
        assert witness.embedding_used == 1
        with pytest.raises(RuntimeError, match="schedule exhausted"):
            witness.embedding(ids, weight, dtype=torch.float32)
        witness.embedding_used = 0
        recorder.embeddings[0][2][0, 0] += 1
        with pytest.raises(AssertionError, match="original distributed rows"):
            witness.embedding(ids, weight, dtype=torch.float32)


@pytest.mark.parametrize("compute_mode", ["wna16", "wna8"])
@pytest.mark.parametrize("audit", [False, True])
def test_startup_numerical_audit_requires_explicit_opt_in(tmp_path, compute_mode, audit):
    from cubic_serve.models.quantization import WeightFormat
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.runtime import quantized_graph as runtime
    from tests.models.test_quantization import fp8_config

    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": fp8_config()}))
    from cubic_serve.models.qwen3_5 import Qwen3_5Config

    config = Mock(spec=Qwen3_5Config)
    from cubic_serve.models.qwen3_5.model import Qwen3_5GraphVariant
    from cubic_serve.models.qwen3_5.serving import text_graph_definition

    config.graph = SimpleNamespace(
        text=SimpleNamespace(max_position_embeddings=1024), variant=Qwen3_5GraphVariant.DENSE,
    )
    config.text_config = config.graph.text
    checkpoint = SimpleNamespace(weights=(), quantization=SimpleNamespace(rules=(
        SimpleNamespace(encoding=SimpleNamespace(format=WeightFormat.CUBIC, bits=4)),
    )))
    checkpoint.payload_view = lambda: checkpoint

    class WorkersNotStarted(Exception):
        pass

    with (
        patch.object(runtime, "parse_model_config_json", return_value=config),
        patch.object(runtime, "resolve_model_definition", return_value=ModelDefinition(
            "fixture", lambda _: config, lambda _: (), lambda *args, **kwargs: (),
            text_graph=text_graph_definition,
        )),
        patch.object(runtime.CheckpointWeights, "admit", return_value=checkpoint),
        patch.object(runtime, "_ProcessController", side_effect=WorkersNotStarted) as controller,
        pytest.raises(WorkersNotStarted),
    ):
        runtime.QuantizedGraphExecutor(
            tmp_path, devices=(0,), tp=1, pp=1, ep=1, max_requests=1,
            max_context=None, prefill_chunk=None,
            operator_config=OperatorConfig(compute_mode=compute_mode),
            eos_token_ids=(0,), execution_mode="cuda",
            **({"startup_numerical_audit": True} if audit else {}),
        )
    # Inspect the real executor's worker admission, not just CLI forwarding.
    assert controller.call_args.args[0].startup_numerical_audit is audit
    assert controller.call_args.args[0].mamba_ssm_cache_dtype == "bfloat16"


def test_scale_witness_preserves_int8_operand_provenance():
    from cubic_serve.guardrail.quantized_graph import linear
    from cubic_serve.guardrail.quantized_graph.replay import Replay

    witness = object.__new__(Replay)
    witness.scales = {}
    q = torch.tensor([[1, -2]], dtype=torch.int8)
    scale = torch.tensor([0.25])
    weight = torch.ones(1, 2)
    output = torch.ones(1, 1, dtype=torch.bfloat16)
    with patch.object(linear, "check_linear_enclosure", return_value={}) as checker:
        with witness.scale_witness():
            linear.check_linear_enclosure(q, weight, output, int8_token_scales=scale)
        assert checker.call_args.kwargs["int8_token_scales"] is scale
        assert checker.call_args.args[0] is q


def test_int8_boundary_rejects_unwitnessed_scale_and_codes():
    from cubic_serve.guardrail.quantized_graph.linear import check_int8_boundary

    value = torch.tensor([[0., 127., -127.]])
    q, scale = value.to(torch.int8), torch.ones(1)
    check_int8_boundary(q, scale, value, value)
    with pytest.raises(AssertionError, match="scale"):
        check_int8_boundary(q, scale * 2, value, value)
    q[0, 0] = 1
    with pytest.raises(AssertionError, match="codes"):
        check_int8_boundary(q, scale, value, value)


def test_int8_reference_rounds_through_integer_zero():
    from cubic_serve.models.quantization import QuantizationScheme, WeightFormat
    from cubic_serve.quantization.extensions.cubic.reference import wna8_activation

    encoding = QuantizationScheme(WeightFormat.CUBIC, 4, 32, 1, "float32-scale-float16-ab")
    values = torch.tensor([[-0.0, -0.25, 127.]], dtype=torch.bfloat16)
    restored = wna8_activation(values, encoding)
    assert torch.equal(restored, torch.tensor([[0., 0., 127.]]))
    assert not torch.signbit(restored).any()


def test_strided_byte_comparison_preserves_nan_payload_and_signed_zero():
    from cubic_serve.guardrail.quantized_graph import byte_exact

    bits = torch.tensor([0, -32768, 0x7FC1, 0x7FC2], dtype=torch.int16)
    a = bits.repeat(17, 3).view(torch.bfloat16).T
    b = a.clone()
    assert byte_exact(a, b, chunk_bytes=16)
    assert byte_exact(a[:2, :2], b[:2, :2], chunk_bytes=1)
    b.view(torch.int16)[-1, -1] = 0x7FC1
    assert not byte_exact(a, b, chunk_bytes=16)
    assert not byte_exact(torch.tensor(0.0), torch.tensor(-0.0))
    assert byte_exact(a[:, :0], b[:, :0])


def test_strided_byte_comparison_bounds_contiguous_scratch():
    from cubic_serve.guardrail.quantized_graph import byte_exact

    a = torch.arange(4096, dtype=torch.int32).reshape(16, 256).T
    b = a.clone()
    original = torch.Tensor.contiguous
    copied = []

    def bounded(tensor, *args, **kwargs):
        if not tensor.is_contiguous():
            copied.append(tensor.numel() * tensor.element_size())
        return original(tensor, *args, **kwargs)

    with patch.object(torch.Tensor, "contiguous", bounded):
        assert byte_exact(a, b, chunk_bytes=256)
    assert copied and max(copied) <= 256


@pytest.mark.parametrize("width,offset", [(1, 0), (3, 1), (8, 0), (16, 4), (24, 1), (32, 8)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_word_byte_comparison_checks_every_byte_with_unaligned_storage(width, offset, device):
    if device == "cuda" and os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1":
        pytest.skip("explicit CUDA audit required")
    if device == "cuda":
        from cubic_serve.executor.runner import configure_reference_math

        configure_reference_math()
    from cubic_serve.guardrail.quantized_graph import byte_exact

    storage = torch.arange(width + offset, dtype=torch.uint8, device=device)
    original = storage[offset:]
    same = original.clone()
    assert byte_exact(original, same)
    for byte in range(width):
        changed = same.clone()
        changed[byte] ^= 128
        assert not byte_exact(original, changed), (width, offset, byte, device)


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
def test_host_witness_byte_comparison_bounds_transfers():
    from cubic_serve.executor.runner import configure_reference_math
    from cubic_serve.guardrail.quantized_graph import byte_exact

    configure_reference_math()
    host = torch.arange(4096, dtype=torch.int32).reshape(16, 256).T
    device = host.cuda()
    original = torch.Tensor.to
    transfers = []

    def bounded(tensor, *args, **kwargs):
        if args and isinstance(args[0], torch.device) and tensor.device != args[0]:
            transfers.append(tensor.numel() * tensor.element_size())
        return original(tensor, *args, **kwargs)

    with patch.object(torch.Tensor, "to", bounded):
        assert byte_exact(host, device, chunk_bytes=256)
        assert byte_exact(device, host, chunk_bytes=256)
    assert transfers and max(transfers) <= 256
    device[-1, -1] += 1
    assert not byte_exact(host, device, chunk_bytes=256)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_offloaded_gdn_state_preserves_transport_and_producer_checks(device):
    if device == "cuda" and os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1":
        pytest.skip("explicit CUDA audit required")
    if device == "cuda":
        from cubic_serve.executor.runner import configure_reference_math

        configure_reference_math()
    q, k, v = [torch.zeros(2, 1, 1, 2, device=device) for _ in range(3)]
    a, b = [torch.zeros(2, 1, device=device) for _ in range(2)]
    alog, bias = [torch.zeros(1, device=device) for _ in range(2)]
    state = torch.zeros(2, 1, 2, 2, device=device)
    new = torch.ones_like(state, device="cpu")
    x = torch.cat((q.flatten(2), k.flatten(2), v.flatten(2)), dim=-1)
    y = torch.zeros(2, 1, 2, device=device)
    recorder = SimpleNamespace(
        events={"gdn": [(x, a, b, alog, bias, state.cpu(), y, new, None)]},
        proofs=[
            {
                "checks": {
                    key: {"outside_enclosure": 0}
                    for key in (
                        *ColdReplay.qk_proofs,
                        ColdReplay.state_proof,
                        "output_fp32_reduction",
                    )
                }
            }
        ],
    )
    # This fixture tests snapshot transport, not the separate interval proof.
    _, output = ColdReplay(recorder).gdn(q, k, v, a, b, alog, bias, state)
    assert output.device == state.device and torch.equal(output.cpu(), new)
    output.zero_()
    assert bool((new == 1).all()), "replay must not mutate immutable snapshots"
    with pytest.raises(AssertionError, match="GDN state"):
        ColdReplay(recorder).gdn(q, k, v, a, b, alog, bias, state + 1)


def test_observer_bounds_state_snapshots_without_truncating_forward():
    from cubic_serve.guardrail.quantized_graph.nonlinear import ColdRecorder

    state = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    initial = state.clone()
    src, dst = torch.tensor([3, 1, 7, 6]), torch.tensor([0, 4, 5, 2])
    x, y = torch.zeros(4, 2), torch.zeros(4, 2)

    def launch(x, packed, a, b, alog, bias, state, src, dst, segments, y):
        state[dst] = state[src] + 1
        y.copy_(state[dst])

    op = SimpleNamespace(launch=launch)
    body = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(
                graph=SimpleNamespace(text=SimpleNamespace(rms_norm_epsilon=1e-6))
            )
        ),
        gdn=op,
    )
    recorder = ColdRecorder()
    recorder.logical_m = 2
    with recorder.observe(body, None):
        body.gdn.launch(x, x, x, x, x, x, state, src, dst, src, y)
    assert body.gdn is op
    assert torch.equal(y, initial[src] + 1), "observation must not trim actual work"
    event = recorder.events["gdn"][0]
    assert torch.equal(event[5], initial[src[:2]])
    assert torch.equal(event[7], initial[src[:2]] + 1)
    state.zero_()
    assert torch.equal(event[7], initial[src[:2]] + 1), "snapshots must stay immutable"


@pytest.mark.parametrize("lo,hi", [(float("nan"), 1), (0, float("inf")), (2, 1)])
def test_invalid_interval_never_admits_a_finite_candidate(lo, hi):
    with pytest.raises(AssertionError, match="invalid reference interval"):
        check(torch.tensor([0.5]), torch.tensor([lo]), torch.tensor([hi]))


def test_nonfinite_candidate_and_swapped_request_state_are_rejected():
    assert (
        check(torch.tensor([float("nan")]), torch.tensor([0.0]), torch.tensor([1.0]))[
            "outside_enclosure"
        ]
        == 1
    )
    checker = ColdReplay(SimpleNamespace(events={}))
    state = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="request state"):
        checker.equal(state, state.flip(0), "request state")


def test_ep_reference_preserves_the_canonical_route_observer_boundary():
    """The EP ownership extension must not bypass reference routing observers."""
    width, prefix = 2, "mlp"
    weights = {f"{prefix}.gate.weight": torch.eye(width, dtype=torch.bfloat16)}
    for name in ("experts.0", "experts.1", "shared_expert"):
        for role in ("gate", "up", "down"):
            weights[f"{prefix}.{name}.{role}_proj.weight"] = torch.eye(width, dtype=torch.bfloat16)
    weights[f"{prefix}.shared_expert_gate.weight"] = torch.ones(1, width, dtype=torch.bfloat16)
    owner = SimpleNamespace(
        config=SimpleNamespace(
            graph=SimpleNamespace(
                text=SimpleNamespace(
                    moe=SimpleNamespace(experts_per_token=1),
                )
            )
        ),
        weights=weights,
        _audit_owned=(0, 2),
    )
    operators = SimpleNamespace(linear=lambda x, w: x @ w.T, ep_output=lambda x, _: x)
    inputs = torch.tensor([[1, 0], [0, 1]], dtype=torch.bfloat16)
    with (
        ReferenceExecutionContext(),
        patch.object(
            model,
            "route_qwen3_5_moe",
            wraps=model.route_qwen3_5_moe,
        ) as route,
    ):
        result = ep.ep_reference(owner, inputs, prefix=prefix, operators=operators)
    assert route.call_count == 1
    assert result.shape == inputs.shape and torch.isfinite(result).all()


def test_portfolio_lease_includes_interleaved_graph_and_real_streams():
    from cubic_serve.runtime.quantized_qualification import portfolio_resources

    def stage(stream):
        return SimpleNamespace(
            rank=0,
            stream=SimpleNamespace(cuda_stream=stream),
            buffers=object(),
            body=object(),
            arena=object(),
        )

    root, child = stage(1), stage(2)
    root._graph = object()
    schedule = SimpleNamespace(
        compute=SimpleNamespace(cuda_stream=3),
        communication=SimpleNamespace(cuda_stream=4),
    )
    root._overlap_resources = (object(), [child], schedule, None)
    sampler = SimpleNamespace(
        graphs=[object(), object()], stream=SimpleNamespace(cuda_stream=200),
        grammar_candidate_graph=None, grammar_random_graph=None, token_mask=None,
    )
    portfolio = {1: SimpleNamespace(
        slot=root, _input_update_graph=object(),
        _sampling=SimpleNamespace(greedy_plans=[sampler]),
    )}
    first = portfolio_resources(portfolio)
    # Children are parts of the parent capture, not independent graph owners.
    assert not hasattr(child, "graph")
    schedule.communication.cuda_stream = 5
    second = portfolio_resources(portfolio)
    assert first.stream_lane_id != second.stream_lane_id
    assert first.graph_instance_id == second.graph_instance_id
    root._overlap_resources = (object(), [child], schedule, None)
    assert portfolio_resources(portfolio).graph_instance_id != second.graph_instance_id
    third = portfolio_resources(portfolio)
    portfolio[1] = SimpleNamespace(
        slot=root, _input_update_graph=object(),
        _sampling=SimpleNamespace(greedy_plans=[sampler]),
    )
    fourth = portfolio_resources(portfolio)
    assert fourth.graph_instance_id != third.graph_instance_id
    assert fourth.buffer_set_id != third.buffer_set_id
    assert fourth.workspace_arena_id != third.workspace_arena_id
    # Replacing only the input graph also invalidates the graph portfolio lease.
    old_input_graph = portfolio[1]._input_update_graph
    portfolio[1]._input_update_graph = object()
    fifth = portfolio_resources(portfolio)
    assert fifth.graph_instance_id != fourth.graph_instance_id
    assert fifth.buffer_set_id == fourth.buffer_set_id
    assert old_input_graph is not portfolio[1]._input_update_graph
    before = portfolio_resources(portfolio)
    sampler.graphs[0] = object()
    after = portfolio_resources(portfolio)
    assert after.graph_instance_id != before.graph_instance_id
    assert after.buffer_set_id == before.buffer_set_id
    sampler.stream.cuda_stream += 1
    assert portfolio_resources(portfolio).stream_lane_id != after.stream_lane_id

    draft_stage = stage(300)
    draft_stage._graph = object()
    draft_stage._overlap_resources = None
    binding_executor = SimpleNamespace(_input_update_graph=object())
    draft = SimpleNamespace(
        slot=draft_stage, _sampling=SimpleNamespace(greedy_plans=[]),
        _draft_binding=SimpleNamespace(executor=binding_executor),
    )
    joint = portfolio_resources(portfolio, {1: draft})
    assert joint.graph_instance_id != portfolio_resources(portfolio).graph_instance_id
    assert joint.buffer_set_id != portfolio_resources(portfolio).buffer_set_id
    binding_executor._input_update_graph = object()
    assert portfolio_resources(portfolio, {1: draft}).graph_instance_id != joint.graph_instance_id
    same_stage = portfolio_resources(portfolio, {1: draft})
    draft._draft_binding.executor = SimpleNamespace(_input_update_graph=object())
    replaced = portfolio_resources(portfolio, {1: draft})
    assert replaced.buffer_set_id != same_stage.buffer_set_id
    assert replaced.workspace_arena_id != same_stage.workspace_arena_id

    before_round = portfolio_resources(portfolio, {1: draft})
    portfolio[1]._pipelined_draft_round = SimpleNamespace(
        depth=1, graph=object(), candidate_graph=object(), pool=object(), target=portfolio[1])
    with_round = portfolio_resources(portfolio, {1: draft})
    assert with_round.graph_instance_id != before_round.graph_instance_id
    assert with_round.buffer_set_id != before_round.buffer_set_id
    assert with_round.workspace_arena_id != before_round.workspace_arena_id
    before_candidate = portfolio_resources(portfolio, {1: draft})
    portfolio[1]._pipelined_draft_round.candidate_graph = object()
    changed_candidate = portfolio_resources(portfolio, {1: draft})
    assert changed_candidate.graph_instance_id != before_candidate.graph_instance_id
    native_window = SimpleNamespace(graph=11, exec=12, children=(object(),),
                                    host=object(), meta=object(), staging={})
    candidate_window = SimpleNamespace(graph=21, exec=22, children=(object(),),
                                       host=native_window.host, meta=object(), staging={})
    portfolio[1]._pipelined_draft_round._resident_window = native_window
    portfolio[1]._pipelined_draft_round._candidate_window = candidate_window
    both_windows = portfolio_resources(portfolio, {1: draft})
    candidate_window.exec = 23
    changed_window = portfolio_resources(portfolio, {1: draft})
    assert changed_window.graph_instance_id != both_windows.graph_instance_id
    candidate_window.staging = {"new-input": object()}
    changed_ingress = portfolio_resources(portfolio, {1: draft})
    assert changed_ingress.buffer_set_id != changed_window.buffer_set_id
    saved_pool = portfolio[1]._pipelined_draft_round.pool
    portfolio[1]._pipelined_draft_round.pool = object()
    replaced_pool = portfolio_resources(portfolio, {1: draft})
    assert replaced_pool.buffer_set_id != with_round.buffer_set_id
    assert saved_pool is not portfolio[1]._pipelined_draft_round.pool


@pytest.mark.parametrize(
    "policy",
    [
        {"attention_dtype": "fp32"},
        {"activation_dtype": "fp32"},
        {"softmax_dtype": "auto"},
    ],
)

def test_unqualified_arithmetic_is_rejected_before_weights_not_as_numerical_failure(
    tmp_path, policy,
):
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.runtime import quantized_graph as runtime
    from tests.models.test_quantization import fp8_config

    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": fp8_config()}))
    from cubic_serve.models.qwen3_5 import Qwen3_5Config

    config = Mock(spec=Qwen3_5Config)
    with (
        patch.object(runtime, "parse_model_config_json", return_value=config),
        patch.object(runtime.CheckpointWeights, "admit") as admit,
        pytest.raises(ValueError, match="independent text performance qualification"),
    ):
        runtime.QuantizedGraphExecutor(
            tmp_path,
            devices=(0,),
            tp=1,
            pp=1,
            ep=1,
            max_requests=1,
            max_context=None,
            prefill_chunk=None,
            operator_config=OperatorConfig(**policy),
            eos_token_ids=(0,),
            execution_mode="cuda",
        )
    admit.assert_not_called()


@pytest.mark.parametrize(
    ("tp", "ep"), [(tp, ep) for tp in (1, 2, 4, 8) for ep in (1, 2, 4, 8) if tp % ep == 0],
)
@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, "bfloat16"), ("auto", "float32"), ("float32", "float32"), ("bfloat16", "bfloat16")],
)
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("request_limit", [1, 128, 256, 4096])
def test_typed_admission_uses_topology_axes_without_ep_whitelist(
    tmp_path, tp, ep, requested, expected, quantized, request_limit,
):
    from cubic_serve.runtime import quantized_graph as runtime
    from tests.models.test_quantization import fp8_config

    document = {"quantization_config": fp8_config()} if quantized else {}
    (tmp_path / "config.json").write_text(json.dumps(document))
    from cubic_serve.models.qwen4_exp import Qwen4ExpConfig
    from cubic_serve.models.qwen4_exp.serving import text_graph_definition

    config = Mock(spec=Qwen4ExpConfig)
    config.text = SimpleNamespace(
        max_position_embeddings=1024,
        gated_delta_net=SimpleNamespace(recurrent_state_dtype="float32"),
        qsa=SimpleNamespace(compression_ratio=4),
        dtype="bfloat16",
    )

    config.text_config = config.text

    class WorkersNotStarted(Exception):
        pass

    checkpoint = SimpleNamespace()
    checkpoint.payload_view = lambda: checkpoint
    with (
        patch.object(runtime, "parse_model_config_json", return_value=config),
        patch.object(runtime, "resolve_model_definition", return_value=ModelDefinition(
            "fixture", lambda _: config, lambda _: (), lambda *a, **kw: (),
            text_graph=text_graph_definition,
        )),
        patch.object(runtime.CheckpointWeights, "admit", return_value=checkpoint) as admit,
        patch.object(runtime, "_ProcessController", side_effect=WorkersNotStarted) as controller,
        pytest.raises(WorkersNotStarted),
    ):
        runtime.QuantizedGraphExecutor(
            tmp_path, devices=tuple(range(tp)), tp=tp, pp=1, ep=ep, max_requests=request_limit,
            max_context=None, prefill_chunk=None, operator_config=None,
            eos_token_ids=(0,), execution_mode="cuda",
            **({"mamba_ssm_cache_dtype": requested} if requested is not None else {}),
        )
    admission = controller.call_args.args[0]
    assert admission.parallel_plan.topology.expert_tensor_size == tp // ep
    assert admission.max_running_requests == request_limit
    assert admission.graph_capacities == tuple(1 << i for i in range(11))
    assert admission.graph_request_capacity == 1024
    assert admission.mamba_ssm_cache_dtype == expected
    assert (
        admission.operator_config.activation_dtype,
        admission.operator_config.attention_dtype,
        admission.operator_config.softmax_dtype,
    ) == ("auto", "auto", "fp32")
    assert not admission.startup_numerical_audit
    assert admit.call_args.kwargs["require_quantization"] == quantized


@pytest.mark.parametrize("budget", [1, 3, 16, 255, 512, 1000, 1024, 4096])
def test_text_graph_budget_keeps_all_anchors_without_exceeding_admin_limit(budget):
    from cubic_serve.runtime.quantized_graph import _text_graph_capacities
    capacities = _text_graph_capacities(budget, 0)
    assert capacities[0] == 1
    assert capacities[-1] <= budget < 2 * capacities[-1]
    assert all(b == 2 * a for a, b in zip(capacities, capacities[1:]))


def test_automatic_graph_rows_follow_headroom_and_tensor_geometry():
    from cubic_serve.runtime.quantized_graph import _auto_text_graph_capacities

    geometry = dict(vocabulary_size=248320, hidden_size=5120,
                    element_bytes=2, tp_size=2, bodies=2, max_speculative_tokens=3)

    def choose(available, **changes):
        return _auto_text_graph_capacities(available, **(geometry | changes))

    assert choose(6 << 30)[-1] == 256
    assert choose(60 << 30)[-1] == 1024
    assert choose(6 << 30, vocabulary_size=2 * geometry['vocabulary_size'])[-1] < 256
    assert choose(6 << 30, bodies=1)[-1] > 256
    assert choose(6 << 30, tp_size=4)[-1] > 256
    assert choose(6 << 30, element_bytes=4)[-1] < 256
    limits = [choose(n << 28)[-1] for n in range(100)]
    assert limits == sorted(limits)
    assert limits[0] == 4
    assert all(m & (m - 1) == 0 for m in limits)
    # Neither the client ceiling nor context length is an input to this policy.
    assert choose(6 << 30) == tuple(1 << i for i in range(9))


@pytest.mark.parametrize("change", [dict(available_bytes=-1), dict(vocabulary_size=0),
                                  dict(hidden_size=0), dict(element_bytes=0),
                                  dict(tp_size=0), dict(bodies=0)])
def test_automatic_graph_rows_reject_invalid_resources(change):
    from cubic_serve.runtime.quantized_graph import _auto_text_graph_capacities

    inputs = dict(available_bytes=6 << 30, vocabulary_size=248320, hidden_size=5120,
                  element_bytes=2, tp_size=2, bodies=2, max_speculative_tokens=3)
    with pytest.raises(ValueError, match="geometry"):
        _auto_text_graph_capacities(**(inputs | change))


@pytest.mark.parametrize("budget,depth", [(1, 1), (3, 2), (8, 8)])
def test_text_graph_budget_rejects_incomplete_speculative_transaction(budget, depth):
    from cubic_serve.runtime.quantized_graph import _text_graph_capacities
    with pytest.raises(ValueError, match="complete draft review"):
        _text_graph_capacities(budget, depth)


@pytest.mark.parametrize("budget", [16, 255, 512, 1000, 1024, 4096])
@pytest.mark.parametrize("depth", [0, 1, 2, 3, 7])
def test_text_graph_owner_cohort_fits_every_complete_review(budget, depth):
    from cubic_serve.runtime.quantized_graph import (
        _text_graph_capacities, _text_graph_owner_capacity,
    )
    capacities = _text_graph_capacities(budget, depth)
    owners = _text_graph_owner_capacity(capacities, depth)
    assert owners in capacities
    assert owners * (depth + 1) <= capacities[-1] < 2 * owners * (depth + 1)
    assert all(next(m for m in capacities if owners * (k + 1) <= m) <= capacities[-1]
               for k in range(depth + 1))
    if budget == 1024 and depth == 3:
        # The accepted default admission256 plan had1024 physical rows and256
        # complete-review owners; separating admission must preserve that plan.
        assert owners == 256


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, None])
def test_text_graph_budget_rejects_invalid_limit_before_checkpoint_io(budget):
    from cubic_serve.runtime.quantized_graph import TextGraphExecutor
    with pytest.raises(ValueError, match="token budget"):
        TextGraphExecutor('/does-not-exist', devices=(0,), tp=1, pp=1, ep=1,
                          max_requests=4096, max_context=None, prefill_chunk=None,
                          operator_config=None, eos_token_ids=(0,),
                          cuda_graph_max_tokens=budget)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_route_enclosure_rejects_changed_weights_and_logits(dtype):
    from cubic_serve.guardrail.quantized_graph.bounds import route_softmax_bounds

    logits = torch.tensor([[0.0, -1.0, -2.0, -3.0]], dtype=dtype)
    expected = torch.softmax(logits.double(), dim=-1).to(dtype)
    lo, hi = route_softmax_bounds(logits, dtype)
    assert check(expected, lo, hi)["outside_enclosure"] == 0
    adjacent = expected.clone()
    adjacent[0, 0] = torch.nextafter(adjacent[0, 0], torch.tensor(float("inf"), dtype=dtype))
    assert check(adjacent, lo, hi)["outside_enclosure"] > 0
    corrupted = expected.clone()
    corrupted[0, 0] += 0.01
    assert check(corrupted, lo, hi)["outside_enclosure"] > 0
    assert check(expected.flip(-1), lo, hi)["outside_enclosure"] > 0
    wrong_lo, wrong_hi = route_softmax_bounds(logits.flip(-1), dtype)
    assert check(expected, wrong_lo, wrong_hi)["outside_enclosure"] > 0


def test_rounding_boundary_keeps_fp16_resolution():
    from cubic_serve.guardrail.quantized_graph.bounds import R

    x = torch.tensor([1.0009765625], dtype=torch.float64)
    assert torch.equal(R(x, torch.float16), x)
    assert not torch.equal(R(x, torch.bfloat16), x)


@pytest.mark.parametrize("logical_m", [0, -1, 65, 1.5, True])
def test_independent_audit_rejects_invalid_logical_rows(logical_m):
    from cubic_serve.guardrail.quantized_graph import audit_original

    with pytest.raises(ValueError, match="logical audit rows"):
        audit_original(SimpleNamespace(captured_m=64), None, None, logical_m=logical_m)


def test_cross_m_includes_requested_interior_rows_without_losing_boundaries():
    from cubic_serve.guardrail.quantized_graph.cross_m import audit_cross_m
    from cubic_serve.runtime.cache.quantization import KVCacheStorageSpec

    executors = {m: SimpleNamespace(body=SimpleNamespace(
        captured_m=m, arena=SimpleNamespace(layout=SimpleNamespace(storage=KVCacheStorageSpec()))
    )) for m in (1, 2, 4, 8, 16, 32, 64)}
    admission = SimpleNamespace(
        parallel_plan=SimpleNamespace(topology=SimpleNamespace(pp_size=1, ep_size=1, tp_size=4))
    )
    with (patch("cubic_serve.guardrail.quantized_graph.cross_m._case") as case,
          patch("cubic_serve.guardrail.quantized_graph.cross_m._projection_plans")):

        audit_cross_m(executors, admission, {}, logical_ms=(37, 37))
    rows = [(call.args[0].captured_m, call.args[1]) for call in case.call_args_list]
    assert (64, 37) in rows and rows.count((64, 37)) == 2
    assert (64, 33) in rows and (64, 64) in rows and (1, 1) in rows


def test_cross_m_progress_counts_only_completed_ownership_and_numerical_witnesses():
    from cubic_serve.guardrail.quantized_graph.cross_m import audit_cross_m
    from cubic_serve.runtime.cache.quantization import KVCacheStorageSpec

    executors = {m: SimpleNamespace(body=SimpleNamespace(
        captured_m=m, arena=SimpleNamespace(layout=SimpleNamespace(storage=KVCacheStorageSpec()))
    )) for m in (1, 2)}
    admission = SimpleNamespace(
        parallel_plan=SimpleNamespace(topology=SimpleNamespace(pp_size=1, ep_size=1, tp_size=2))
    )
    progress = []

    def validated_case(stage, logical, steps, *args, oracle=None):
        if oracle is not None:
            for _ in range(min(2, logical) * steps):
                oracle.completed_witness(1)

    with (patch("cubic_serve.guardrail.quantized_graph.cross_m._case", validated_case),
          patch("cubic_serve.guardrail.quantized_graph.cross_m._projection_plans")):
        audit_cross_m(executors, admission, {}, steps=2,
                      completed_witness=lambda done, total: progress.append((done, total)))
    assert progress == [(step, 11) for step in range(1, 12)]

    # An ownership/execution failure cannot publish its completion or let
    # later cases renew the rank's inactivity lease.
    progress.clear()
    with (patch("cubic_serve.guardrail.quantized_graph.cross_m._case",
                side_effect=RuntimeError("failed ownership witness")),
          patch("cubic_serve.guardrail.quantized_graph.cross_m._projection_plans")):
        with pytest.raises(RuntimeError, match="failed ownership"):
            audit_cross_m(executors, admission, {}, steps=2,
                          completed_witness=lambda *item: progress.append(item))
    assert progress == []


def test_trajectory_rejects_state_changed_after_bf16_writeback():
    from cubic_serve.guardrail.quantized_graph import _cache_identity
    from cubic_serve.models.qwen3_5.model import Qwen3_5LayerCache, Qwen3_5ReferenceCache

    stored = torch.tensor([1.0], dtype=torch.bfloat16)

    def cache(value):
        return Qwen3_5ReferenceCache(
            1, 1, (Qwen3_5LayerCache(convolution=stored.clone(), recurrent=value),)
        )

    previous = cache(stored.clone())
    _cache_identity(previous, cache(stored.clone()))
    changed = torch.nextafter(stored, torch.full_like(stored, float("inf")))
    with pytest.raises(AssertionError, match="stored-state continuation"):
        _cache_identity(previous, cache(changed))
    with pytest.raises(AssertionError, match="stored-state continuation"):
        _cache_identity(previous, cache(stored.float()))


def test_independent_reference_reuse_checks_initial_and_continued_state():
    from cubic_serve.guardrail.quantized_graph import _ReferenceTrajectory
    from cubic_serve.models.qwen3_5.model import (
        Qwen3_5LayerCache,
        Qwen3_5ReferenceCache,
        Qwen3_5ReferenceStep,
    )

    memo = _ReferenceTrajectory()
    memo.bind(("checkpoint", "owners", torch.bfloat16))
    state = torch.zeros(2, dtype=torch.bfloat16)
    previous = Qwen3_5ReferenceCache(0, 1, (Qwen3_5LayerCache(recurrent=state),))
    for position in range(4):
        # Independent rounded recurrence; the cache must preserve each step.
        next_state = (previous.layers[0].recurrent.float() + .1).to(torch.bfloat16)
        following = Qwen3_5ReferenceCache(
            position + 1, 1, (Qwen3_5LayerCache(recurrent=next_state),)
        )
        expected = Qwen3_5ReferenceStep(next_state.clone(), next_state.clone(), following)
        observations = ((0, "layer_input", previous.layers[0].recurrent.clone()),)
        memo.retain(position, previous, expected, observations)
        value, retained = memo.lookup(position, previous)
        assert torch.equal(value.logits, next_state) and retained == observations
        bad_state = previous.layers[0].recurrent.clone()
        bad_state[0] = torch.nextafter(bad_state[0], torch.full_like(bad_state[0], float("inf")))
        bad = Qwen3_5ReferenceCache(position, 1, (Qwen3_5LayerCache(recurrent=bad_state),))
        with pytest.raises(AssertionError, match="stored-state continuation"):
            memo.lookup(position, bad)
        previous = following


def test_independent_reference_reuse_rejects_changed_scope():
    from cubic_serve.guardrail.quantized_graph import _ReferenceTrajectory

    memo = _ReferenceTrajectory()
    memo.bind(("checkpoint", "owners", torch.bfloat16))
    memo.bind(("checkpoint", "owners", torch.bfloat16))
    for scope in (("other-checkpoint", "owners", torch.bfloat16),
                  ("checkpoint", "other-owners", torch.bfloat16),
                  ("checkpoint", "owners", torch.float16)):
        with pytest.raises(RuntimeError, match="reference trajectory scope differs"):
            memo.bind(scope)


def test_independent_reference_reuse_bounds_retained_storage():
    from cubic_serve.guardrail.quantized_graph import _ReferenceTrajectory
    from cubic_serve.models.qwen3_5.model import Qwen3_5ReferenceCache

    memo = _ReferenceTrajectory(max_bytes=4)
    cache = Qwen3_5ReferenceCache(0, 1, ())
    memo.retain(0, cache, torch.ones(8), ())
    assert memo.lookup(0, cache) is None and not memo.storages


def test_independent_reference_mixed_rank_hits_recompute_together(monkeypatch):
    from cubic_serve.guardrail.quantized_graph import _reference_lookup

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    key, cached = (0, (100,)), ("independent result", ())
    trajectory = SimpleNamespace(lookup=lambda *a: cached)

    def gather(items, value, **kw):
        items[:] = [value, (key, False, None)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    assert _reference_lookup(trajectory, key, None, None) is None

    def all_hit(items, value, **kw):
        items[:] = [value, value]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_hit)
    assert _reference_lookup(trajectory, key, None, None) is cached


@pytest.mark.parametrize("steps,prefix", [(0, 0), (5, 0), (True, 0), (2, 64)])
def test_trajectory_requires_bounded_cold_start(steps, prefix):
    from cubic_serve.guardrail.quantized_graph import audit_original
    from cubic_serve.runtime.cache.quantization import KVCacheStorageSpec

    stage = SimpleNamespace(captured_m=1, arena=SimpleNamespace(
        layout=SimpleNamespace(storage=KVCacheStorageSpec())))
    with pytest.raises(ValueError, match="trajectory witness"):
        audit_original(stage, None, None, prefix_length=prefix, steps=steps)


@pytest.mark.parametrize("first_corpus_retained", [False, True])
def test_cross_m_reuses_only_complete_causal_trajectories(monkeypatch, first_corpus_retained):
    from cubic_serve.guardrail.quantized_graph import cross_m

    monkeypatch.setattr(cross_m.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(cross_m.dist, "all_gather_object",
                        lambda items, value, **kw: items.__setitem__(0, value))
    calls = []

    def value(position, offset, m=1):
        # Different physical rows may have different permitted arithmetic.
        # Re-labelling the second corpus as row zero would lose this witness.
        return {"state": torch.tensor([[offset + position + row * 14] for row in range(m)],
                                      dtype=torch.bfloat16)}

    def independent(stage, admission, groups, **kw):
        calls.append((kw["logical_m"], kw["token_offset"]))
        for position in range(kw["steps"]):
            stage.position = position
            stage.offset = kw["token_offset"]
            stage.rows = kw["logical_m"]
            kw["completed_step_witness"](
                position, stage, {},
                SimpleNamespace(int8_scales={}, linear=[], embeddings=[], experts=[]),
                SimpleNamespace(events=[], event_semantics=[]),
            )
        return {"cross_step_error_propagation_qualified": True}

    monkeypatch.setattr(cross_m, "audit_original", independent)
    monkeypatch.setattr(cross_m, "_snapshot",
                        lambda stage, *a, **kw: value(stage.position, stage.offset, stage.rows))
    completed = []
    oracle = cross_m._TrajectoryOracle(None, {"tp": None}, 2,
                                      completed_witness=completed.append)
    stage = SimpleNamespace(captured_m=2)
    timeline = [tuple(cross_m._row_digests(value(position, 0, 2), corpus)
                      for corpus in range(2)) for position in range(2)]
    if first_corpus_retained:
        assert oracle.qualify(stage, [rows[:1] for rows in timeline]) == [0]
        assert calls == [(1, 0)]
        completed.clear()
    assert oracle.qualify(stage, timeline) == [0, 0]
    assert completed == [2, 2]
    assert oracle.qualify(stage, timeline) == [0, 0]
    assert completed == [2, 2, 4]
    assert calls == ([(1, 0)] if first_corpus_retained else []) + [(2, 0)]
    # A changed cold-step boundary cannot hide behind matching final state.
    corrupt = list(timeline)
    corrupt[0] = (cross_m._row_digests(value(1, 0), 0), timeline[0][1])
    with pytest.raises(AssertionError, match="actual rows differ"):
        oracle.qualify(stage, corrupt)
    assert calls[-1] == (2, 0)


@pytest.mark.parametrize("dtype,width", [
    (torch.uint8, 1), (torch.uint8, 3), (torch.uint8, 8),
    (torch.bfloat16, 1), (torch.bfloat16, 3), (torch.int32, 1), (torch.int64, 1),
])
@pytest.mark.parametrize("strided", [False, True])
def test_cross_m_lossless_row_compression_checks_every_row(dtype, width, strided):
    from cubic_serve.guardrail.quantized_graph.cross_m import _row_digests, _trajectory_rows

    x = torch.arange(2 * width).reshape(2, width).to(dtype).repeat(4, 1)
    if strided:
        x = torch.stack((x, x), dim=-1)[..., 0]
    values = {"state": x}
    assert _trajectory_rows(values, 8) == tuple(_row_digests(values, row) for row in range(8))
    x[6, -1] += 1
    with pytest.raises(RuntimeError, match="lacks its corpus boundary witness"):
        _trajectory_rows(values, 8)


def test_cross_m_lossless_row_compression_keeps_signed_zero_and_nan_payloads():
    from cubic_serve.guardrail.quantized_graph.cross_m import _row_digests, _trajectory_rows

    bits = torch.tensor([[0, 0x7FC1], [-32768, 0x7FC2]], dtype=torch.int16).repeat(4, 1)
    values = {"state": bits.view(torch.bfloat16)}
    assert _trajectory_rows(values, 8) == tuple(_row_digests(values, row) for row in range(8))
    for column, replacement in ((0, -32768), (1, 0x7FC2)):
        changed = bits.clone()
        changed[6, column] = replacement
        with pytest.raises(RuntimeError, match="lacks its corpus boundary witness"):
            _trajectory_rows({"state": changed.view(torch.bfloat16)}, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA stream ordering audit")
def test_named_audit_stream_waits_for_delayed_input_fill():
    from cubic_serve.guardrail.quantized_graph import _replay_after_fill

    device = torch.device("cuda:0")
    inputs = torch.zeros(64, device=device)
    outputs = torch.zeros_like(inputs)
    replay_stream, fill_stream = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=replay_stream):
        outputs.copy_(inputs)
    torch.cuda.synchronize()

    def replay():
        with torch.cuda.stream(replay_stream):
            graph.replay()

    stage = SimpleNamespace(hidden=inputs, stream=replay_stream, replay=replay)
    with torch.cuda.stream(fill_stream):
        torch.cuda._sleep(20000000)
        inputs[:37].fill_(7)
        _replay_after_fill(stage)
    torch.cuda.synchronize()
    assert torch.equal(outputs[:37], torch.full_like(outputs[:37], 7))
    assert not outputs[37:].any()


def test_attention_arithmetic_witness_preserves_actual_partition_and_nested_scope(monkeypatch):
    from cubic_serve.guardrail.quantized_graph import text_cross_m as audit
    from cubic_serve.ops.cuda.attention import paged_decode

    def body(capacity, splits):
        return SimpleNamespace(captured_m=capacity, helper=SimpleNamespace(_buffers={
            ("qsa.attention_partial", (capacity, 12, splits, 256), torch.float32): None,
        }))

    plan32 = audit._attention_arithmetic_plan(body(4, 32))
    plan16 = audit._attention_arithmetic_plan(body(16, 16))
    assert plan32 != plan16
    assert plan16 == audit._attention_arithmetic_plan(body(32, 16))
    def original(*args, **kwargs):
        return 7

    monkeypatch.setattr(paged_decode, "paged_gqa_kv_splits", original)
    with audit._attention_arithmetic_scope(plan32):
        assert paged_decode.paged_gqa_kv_splits(1, max_tokens=128) == 32
        with audit._attention_arithmetic_scope(plan16):
            assert paged_decode.paged_gqa_kv_splits(1, max_tokens=128) == 16
        assert paged_decode.paged_gqa_kv_splits(1, max_tokens=128) == 32
    assert paged_decode.paged_gqa_kv_splits is original
    with audit._attention_arithmetic_scope(()):
        assert paged_decode.paged_gqa_kv_splits is original


@pytest.mark.parametrize("capacity,splits,dtype", [
    (8, 16, torch.float32), (16, 3, torch.float32), (16, 16, torch.bfloat16),
])
def test_attention_arithmetic_witness_rejects_invalid_partition_inventory(capacity, splits, dtype):
    from cubic_serve.guardrail.quantized_graph import text_cross_m as audit

    body = SimpleNamespace(captured_m=16, helper=SimpleNamespace(_buffers={
        ("qsa.attention_partial", (capacity, 12, splits, 256), dtype): None,
    }))
    with pytest.raises(ValueError, match="attention arithmetic"):
        audit._attention_arithmetic_plan(body)


def test_attention_arithmetic_scope_rejects_ambiguous_partition_binding():
    from cubic_serve.guardrail.quantized_graph import text_cross_m as audit

    with pytest.raises(ValueError, match="attention arithmetic"):
        with audit._attention_arithmetic_scope(((12, 16, 256), (6, 32, 256))):
            pytest.fail("ambiguous partition plan cannot bind a reference graph")


def test_row_arithmetic_witness_uses_sealed_recipe_and_expanded_row_layout(monkeypatch):
    from fractions import Fraction

    from cubic_serve.quantization.common import small_rows_cost as selection

    operators = SimpleNamespace(config=SimpleNamespace(compute_mode='wna16'))
    weight = SimpleNamespace(encoding='format-owned', shape=(128, 256), partitions=(128,))
    source_x = SimpleNamespace(shape=(8, 256), dtype=torch.bfloat16)
    key = selection._plan_key(operators, source_x, (weight,), 7)
    operators._packed_row_plans = {(key, 8): ('simt', 8, 2)}
    plan = selection.row_arithmetic_plan(operators, 4)
    assert plan == {(key, Fraction(2, 1)): ('simt', 8, 2)}
    adapter = SimpleNamespace(prepare=lambda op, w, device: ('original-payload', w))
    linear = SimpleNamespace(row_contraction=adapter)
    audited_x = SimpleNamespace(shape=(2, 256), dtype=torch.bfloat16, device='candidate-device')
    received = []
    monkeypatch.setattr(selection, 'bind', lambda x, weights, y, **recipe:
                        received.append((x, weights, y, recipe)) or 'sealed-binding')
    monkeypatch.setattr(selection, '_choose_simt',
                        lambda *args: pytest.fail('audit must not reprice a C1 workload'))
    with selection.row_arithmetic_scope(plan):
        assert selection.choose(operators, audited_x, (weight,), ('out',), 7, linear) == 'sealed-binding'
        with selection.row_arithmetic_scope({}):
            with pytest.raises(ValueError, match='absent'):
                selection.choose(operators, audited_x, (weight,), ('out',), 7, linear)
        assert selection.choose(operators, audited_x, (weight,), ('out',), 7, linear) == 'sealed-binding'
    assert selection._audit_plan.get() is None
    assert received[0][-1] == {'warps': 8, 'partitions': 2}
    assert received[0][1] == (('original-payload', weight),)
    assert selection.row_arithmetic_plan(operators, 4) == plan


def test_row_arithmetic_witness_keeps_baseline_and_rejects_unknown_recipe():
    from fractions import Fraction

    from cubic_serve.quantization.common import small_rows_cost as selection

    operators = SimpleNamespace(config=SimpleNamespace(compute_mode='wna8'))
    weight = SimpleNamespace(encoding='format-owned', shape=(64, 256), partitions=(64,))
    x = SimpleNamespace(shape=(1, 256), dtype=torch.int8)
    key = selection._plan_key(operators, x, (weight,), 7), Fraction(1, 1)
    linear = SimpleNamespace(row_contraction=object())
    with selection.row_arithmetic_scope({key: ('baseline', 0, 0)}):
        # None returns ownership to the existing format/mode binding; it never
        # binds a WNA16 operator or substitutes a reference operation here.
        assert selection.choose(operators, x, (weight,), ('out',), 7, linear) is None
    with selection.row_arithmetic_scope({key: ('unknown', 0, 0)}):
        with pytest.raises(ValueError, match='unknown'):
            selection.choose(operators, x, (weight,), ('out',), 7, linear)
    with selection.row_arithmetic_scope({key: ('simt', 4, 1)}):
        with pytest.raises(ValueError, match='explicit WNA16'):
            selection.choose(operators, x, (weight,), ('out',), 7, linear)


@pytest.mark.parametrize('capacity,streams', [(1, 1), (1, 4), (4, 2), (4, 3)])
def test_draft_input_observer_keeps_declared_rows_inside_embedding_scope(capacity, streams):
    from cubic_serve.guardrail.quantized_graph.text_cross_m import _host_events, _observe
    from cubic_serve.models.qwen4_exp.model import Qwen4ExpOperationBinder as Binder
    from cubic_serve.ops.cuda.collective import FixedRankBF16SumPrimitive as Sum

    width = 5
    embedding = torch.arange(capacity * width, dtype=torch.bfloat16).reshape(capacity, width)
    projected = torch.arange(capacity * streams * width, dtype=torch.bfloat16).reshape(
        capacity * streams, width)
    events = []
    def forward():
        for tensor in (embedding, embedding + 7, projected):
            Sum.launch(SimpleNamespace(input_views=lambda value: (value,)), tensor)
    with (
        patch.object(Sum, 'launch', lambda *args: None),
        patch.object(Binder, 'bind', lambda *args, **kwargs: forward),
        _observe(events, input_geometry=(capacity, streams)),
    ):
        Binder.bind(None, 'embedding', 'input', layer_id=None)()
    assert [name for name, _value in events] == [f'input/embedding/sum/{i}' for i in range(3)]
    assert all(value.shape[0] == capacity for _name, value in events)
    assert torch.equal(events[2][1], projected.reshape(capacity, streams, width))
    rows = tuple(reversed(range(capacity)))
    host = _host_events(events, capacity, rows)
    assert torch.equal(host[2][1], projected.reshape(capacity, streams, width)[list(rows)])
    projected.zero_()
    assert events[2][1].count_nonzero() == capacity * streams * width - 1


def test_draft_input_observer_does_not_guess_unrelated_collective_layout():
    from cubic_serve.guardrail.quantized_graph.text_cross_m import _host_events, _observe
    from cubic_serve.models.qwen4_exp.model import Qwen4ExpOperationBinder as Binder
    from cubic_serve.ops.cuda.collective import FixedRankBF16SumPrimitive as Sum

    tensor = torch.zeros((8, 5), dtype=torch.bfloat16)
    events = []
    def forward():
        Sum.launch(SimpleNamespace(input_views=lambda value: (value,)), tensor)
    with (
        patch.object(Sum, 'launch', lambda *args: None),
        patch.object(Binder, 'bind', lambda *args, **kwargs: forward),
        _observe(events, input_geometry=(4, 2)),
    ):
        Binder.bind(None, 'moe', 'block', layer_id=1)()
    assert events[0][1].shape == (8, 5)
    with pytest.raises(ValueError, match='explicit row layout'):
        _host_events(events, 4, (0,))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
def test_geometry_tiled_full_enclosure_preserves_proof_and_detects_corruption(
    device, dtype, packed,
):
    """Larger witness tiles must check every column, including partial tails."""
    if device == "cuda" and os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1":
        pytest.skip("explicit CUDA audit required")
    from cubic_serve.guardrail.quantization import check_linear_enclosure
    from cubic_serve.models.quantization import QuantizationScheme
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.quantization.extensions.int.layout import pack
    from cubic_serve.quantization.extensions.int.reference import decode

    generator = torch.Generator().manual_seed(731)
    codes = torch.randint(-32, 32, (520, 320), generator=generator, dtype=torch.int8)
    scales = torch.full((520, 5), .015625, dtype=dtype)
    scale_dtype = "BF16" if dtype is torch.bfloat16 else "F16"
    encoding = QuantizationScheme("int-pack-v1", 6, 64, metadata="dense:sym:" + scale_dtype)
    matrix = QuantizedMatrix({"weight_packed": pack(codes, 6, "dense"),
                              "weight_scale": scales,
                              "weight_shape": torch.tensor([520, 320])}, encoding, (520, 320))
    weights = decode(matrix.payload, matrix.encoding, matrix.shape).to(device=device, dtype=dtype)
    weight = matrix.to(device) if packed else weights
    inputs = (torch.randn(3, 320, generator=generator) * .125).to(device=device, dtype=dtype)
    output = (inputs.float() @ weights.float().T).to(dtype)
    fixed = check_linear_enclosure(inputs, weight, output, output_rows=128)
    automatic = check_linear_enclosure(inputs, weight, output)
    fields = ("rows", "outputs", "reduction_length", "allowance")
    assert {key: automatic[key] for key in fields} == {key: fixed[key] for key in fields}
    for column in (0, 127, 128, 519):
        corrupted = output.clone()
        corrupted[2, column] += 100
        for tile in (128, None):
            with pytest.raises(AssertionError, match="exceeds FP32/format/RNE enclosure"):
                check_linear_enclosure(inputs, weight, corrupted, output_rows=tile)


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bits", [None, 6, 8])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_original_format_arithmetic_and_owned_results(dtype, bits, device):
    if device == "cuda" and os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1":
        pytest.skip("explicit CUDA audit required")
    g = torch.Generator().manual_seed(173)
    if bits is None:
        w = torch.randn(72, 128, generator=g).to(dtype)
    else:
        codes = torch.randint(-16, 16, (72, 128), generator=g, dtype=torch.int8)
        w = QuantizedMatrix(
            {
                "weight_packed": pack(codes, bits, "dense"),
                "weight_scale": torch.full((72, 2), 0.015625, dtype=dtype),
                "weight_shape": torch.tensor([72, 128]),
            },
            QuantizationScheme(
                "int-pack-v1",
                bits,
                64,
                metadata="dense:sym:" + ("F16" if dtype == torch.float16 else "BF16"),
            ),
            (72, 128),
        )
    x = torch.randn(2, 128, generator=g).to(device=device, dtype=dtype)
    if bits is None:
        w = w.to(device)
    op = QuantizedReferenceOperators(output_rows=32)
    m = ReferenceMathMemo()
    original = QuantizedReferenceOperators.linear
    with (
        ReferenceExecutionContext(),
        torch.inference_mode(),
        patch.object(QuantizedReferenceOperators, "linear", side_effect=original) as spy,
    ):
        m._linear = QuantizedReferenceOperators.linear
        expected = original(op, x, w)
        first = m.linear(op, x, w)
        second = m.linear(op, x.clone(), w)
        exact(first, expected)
        exact(second, expected)
        assert spy.call_count == 1 and m.hits == 1
        first.fill_(10)
        second.fill_(20)
        exact(m.linear(op, x, w), expected)
        changed = x.clone()
        changed[0, 0] += 1
        exact(m.linear(op, changed, w), original(op, changed, w))
        assert spy.call_count == 2
        if bits is None:
            w.add_(0.125)
        else:
            w.payload["weight_scale"].mul_(2)
        exact(m.linear(op, x, w), original(op, x, w))
        assert spy.call_count == 3
        op.output_rows = 64
        exact(m.linear(op, x, w), original(op, x, w))
        assert spy.call_count == 4


def test_signed_zero_collision_and_budget():
    w = torch.eye(2, dtype=torch.bfloat16)
    x = torch.tensor([[0.0, 1.0]], dtype=w.dtype)
    m = ReferenceMathMemo(16)
    op = QuantizedReferenceOperators()
    original = QuantizedReferenceOperators.linear

    class Collision:
        def digest(self):
            return b"collision"

    with (
        ReferenceExecutionContext(),
        torch.inference_mode(),
        patch.object(hashlib, "blake2s", return_value=Collision()),
        patch.object(QuantizedReferenceOperators, "linear", side_effect=original) as spy,
    ):
        m._linear = QuantizedReferenceOperators.linear
        m.linear(op, x, w)
        m.linear(op, x, w)
        assert spy.call_count == 1
        signed = x.clone()
        signed[0, 0] = -0.0
        m.linear(op, signed, w)
        assert spy.call_count == 2
        for i in range(6):
            m.linear(op, x + i, w)
            assert m.bytes <= 16
    assert len(m.entries) <= 2


def test_missing_version_and_disabled_budget_recompute():
    op = QuantizedReferenceOperators()
    x = torch.ones(1, 2)
    with torch.inference_mode():
        w = torch.eye(2)
    for budget in [0, 1024]:
        m = ReferenceMathMemo(budget)
        with ReferenceExecutionContext(), torch.inference_mode():
            exact(m.linear(op, x, w), x)
            exact(m.linear(op, x, w), x)
        assert m.hits == 0 and not m.entries


def test_reference_context_required_even_on_hit():
    x = torch.ones(1, 2)
    w = torch.eye(2)
    op = QuantizedReferenceOperators()
    m = ReferenceMathMemo()
    with ReferenceExecutionContext(), torch.inference_mode():
        m.linear(op, x, w)
    with pytest.raises(RuntimeError):
        m.linear(op, x, w)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"))])
def test_reference_sigmoid_memo_preserves_typed_values_and_owns_results(dtype, device):
    from cubic_serve.ops.eager import activation

    x = torch.tensor([-float("inf"), -65504., -2., -0., 0., 2., 65504.,
                      float("inf"), float("nan")], dtype=dtype, device=device)
    memo = ReferenceMathMemo()
    with ReferenceExecutionContext(), torch.inference_mode():
        expected = activation.sigmoid(x)
        with (activation.reference_activation_memo(memo),
              patch.object(activation, "_sigmoid", wraps=activation._sigmoid) as rule):
            first = activation.sigmoid(x)
            exact(first, expected)
            first.zero_()
            second = activation.sigmoid(x.clone())
            exact(second, expected)
            second.fill_(4)
            exact(activation.sigmoid(x), expected)
            assert rule.call_count == 1 and memo.hits == 2
        # Leaving the audit restores the ordinary independent reference path.
        hits = memo.hits
        exact(activation.sigmoid(x), expected)
        assert memo.hits == hits
    with activation.reference_activation_memo(memo), pytest.raises(RuntimeError):
        activation.sigmoid(x)


def test_reference_sigmoid_memo_checks_collision_bytes_layout_policy_and_budget():
    from cubic_serve.ops.eager import activation
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    class Collision:
        def digest(self):
            return b"collision"

    memo = ReferenceMathMemo(max_bytes=1024, max_entries=4)
    with (ReferenceExecutionContext(), torch.inference_mode(),
          activation.reference_activation_memo(memo),
          patch.object(hashlib, "blake2s", return_value=Collision()),
          patch.object(activation, "_sigmoid", wraps=activation._sigmoid) as rule):
        x = torch.zeros(1, 4, dtype=torch.bfloat16)
        activation.sigmoid(x)
        activation.sigmoid(x.clone())
        assert rule.call_count == 1
        x[0, 0] = -0.
        activation.sigmoid(x)
        assert rule.call_count == 2
        activation.sigmoid(x.expand(2, 4))
        assert rule.call_count == 3
        with operator_strategy_scope(OperatorConfig(activation_dtype="fp32"), device="", model=""):
            assert activation.sigmoid(x).dtype is torch.float32
        assert rule.call_count == 4
        for value in range(12):
            activation.sigmoid(x + value)
            assert memo.bytes <= memo.max_bytes and len(memo.entries) <= memo.max_entries
    memo.clear()
    assert not memo.entries and memo.bytes == 0


def test_reference_sigmoid_memo_scope_resets_on_failure_and_gradients_recompute():
    from cubic_serve.guardrail.quantization import with_reference_integer_geometry
    from cubic_serve.ops.eager import activation

    memo = ReferenceMathMemo()

    @with_reference_integer_geometry
    def failing(*, reference_math_memo):
        activation.sigmoid(torch.ones(4))
        activation.sigmoid(torch.ones(4))
        raise ValueError("audit failure")

    with ReferenceExecutionContext(), torch.inference_mode():
        with pytest.raises(ValueError, match="audit failure"):
            failing(reference_math_memo=memo)
        assert memo.hits == 1
        activation.sigmoid(torch.ones(4))
        assert memo.hits == 1
    memo.clear()
    with ReferenceExecutionContext(), activation.reference_activation_memo(memo):
        x = torch.ones(4, requires_grad=True)
        activation.sigmoid(x).sum().backward()
        assert x.grad is not None and not memo.entries


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_enclosure_reuse_checks_full_observed_output_and_versions(dtype):
    g = torch.Generator().manual_seed(13)
    x = torch.randn(2, 128, generator=g).to(dtype)
    w = torch.randn(72, 128, generator=g).to(dtype)
    m = ReferenceMathMemo()
    op = QuantizedReferenceOperators()
    original = m._enclosure
    with (
        ReferenceExecutionContext(),
        torch.inference_mode(),
        patch.object(m, "_enclosure", wraps=original) as spy,
    ):
        y = op.linear(x, w)
        a = m.enclosure(x, w, y)
        b = m.enclosure(x.clone(), w, y.clone())
        assert a == b and spy.call_count == 1
        b["max_bound"] = -1
        assert m.enclosure(x, w, y)["max_bound"] >= 0
        bad = y.clone()
        bad[-1, -1] += 100
        with pytest.raises(AssertionError):
            m.enclosure(x, w, bad)
        assert spy.call_count == 2
        m.enclosure(x, w, y, output_rows=8)
        assert spy.call_count == 3
        w.mul_(0.5)
        m.enclosure(x, w, op.linear(x, w))
        assert spy.call_count == 4


def test_grad_and_storage_layout_do_not_share_memo_results():
    x = torch.ones(1, 4, requires_grad=True)
    w = torch.eye(4)
    m = ReferenceMathMemo()
    op = QuantizedReferenceOperators()
    with ReferenceExecutionContext():
        m.linear(op, x, w)
        second = m.linear(op, x, w)
        assert m.hits == 0 and not m.entries
        second.sum().backward()
        assert x.grad is not None
    with ReferenceExecutionContext(), torch.inference_mode():
        a = x.detach()
        b = torch.ones(1, 8)[:, ::2]
        exact(a, b)
        exact(m.linear(op, a, w), m.linear(op, b, w))
        assert m.hits == 0


def test_reference_math_memo_entry_budget_and_clear_release_owned_snapshots():
    memo = ReferenceMathMemo(max_bytes=1024, max_entries=2)
    op = QuantizedReferenceOperators()
    weight = torch.eye(2)
    with ReferenceExecutionContext(), torch.inference_mode():
        for value in range(5):
            operand = torch.full((1, 2), float(value))
            exact(memo.linear(op, operand, weight), operand)
            assert len(memo.entries) <= 2 and memo.bytes <= 1024
    memo.clear()
    assert not memo.entries and memo.bytes == 0


@pytest.mark.parametrize("tp", [1, 4])
def test_causal_audit_status_keeps_immediate_non_none_errors_and_resets_each_generation(tp):
    import ast
    import inspect

    from cubic_serve.guardrail.quantized_graph.text import audit_original

    node = next(n for n in ast.walk(ast.parse(inspect.getsource(audit_original)))
                if isinstance(n, ast.FunctionDef) and n.name == "coordinated")
    flag = torch.full((), 2**20, dtype=torch.int32) if tp > 1 else None
    group = object()
    for rank in range(tp):
        values = [None] * tp
        calls = []

        def reduce_status(actual, *, op, group, values=values, rank=rank, calls=calls):
            assert actual is flag and actual.dtype == torch.int32
            assert op == torch.distributed.ReduceOp.MAX
            assert actual.item() == int(values[rank] is not None)
            actual.fill_(int(any(value is not None for value in values)))
            calls.append("status")

        def gather_details(destination, error, *, group, values=values, rank=rank, calls=calls):
            assert len(destination) == tp and error is values[rank]
            destination[:] = values
            calls.append("details")

        ns = dict(
            tp=tp, group=group, audit_error_status=flag,
            dist=SimpleNamespace(all_reduce=reduce_status, all_gather_object=gather_details,
                                 ReduceOp=torch.distributed.ReduceOp),
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), "causal_audit_status", "exec"), ns)
        synchronize = ns["coordinated"]
        for generation, error in enumerate((None, "", False, 0, [], {"step": 7}, "error")):
            values[:] = [None] * tp
            values[generation % tp] = error
            calls.clear()
            if error is None:
                assert synchronize(values[rank]) is None
            else:
                with pytest.raises(AssertionError) as caught:
                    synchronize(values[rank])
                assert str(caught.value) == str(values)
            assert calls == (["status"] if tp > 1 else []) + (
                ["details"] if error is not None or tp == 1 else [])
            values[:] = [None] * tp
            calls.clear()
            assert synchronize(None) is None
            assert calls == (["status"] if tp > 1 else ["details"])


def test_integer_reference_geometry_scope_and_exception_release():
    from cubic_serve.quantization.common.reference import (
        IntegerGeometryMemo,
        integer_geometry,
        integer_geometry_scope,
    )

    calls = []

    def build():
        calls.append(True)
        return torch.tensor([len(calls)], dtype=torch.int64)

    a, b = IntegerGeometryMemo(), IntegerGeometryMemo()
    first = integer_geometry(("test",), build)
    with pytest.raises(RuntimeError, match="test interruption"), integer_geometry_scope(a):
        cached = integer_geometry(("test",), build)
        assert integer_geometry(("test",), build) is cached
        with integer_geometry_scope(b):
            nested = integer_geometry(("test",), build)
            assert nested is not cached
        assert integer_geometry(("test",), build) is cached
        raise RuntimeError("test interruption")
    assert integer_geometry(("test",), build) is not first
    assert len(calls) == 4
    a.clear()
    b.clear()
    assert a.bytes == b.bytes == 0 and not a.entries and not b.entries


@pytest.mark.parametrize("byte_budget,entry_budget", [(0, 8), (64, 0), (64, 2)])
def test_integer_geometry_budget_eviction_and_no_floating_results(byte_budget, entry_budget):
    from cubic_serve.quantization.common.reference import IntegerGeometryMemo

    memo = IntegerGeometryMemo(byte_budget, entry_budget)
    for i in range(10):
        result = memo.get((i,), lambda: torch.arange(4, dtype=torch.int64))
        assert torch.equal(result, torch.arange(4))
        assert memo.bytes <= byte_budget and len(memo.entries) <= entry_budget
    with pytest.raises(TypeError, match="non-integer"):
        memo.get(("floating",), lambda: torch.ones(1))
    memo.clear()
    assert not memo.entries and memo.bytes == 0


def test_reference_math_and_integer_geometry_share_configured_resource_envelope():
    from cubic_serve.quantization.common.reference import integer_geometry

    memo = ReferenceMathMemo(max_bytes=4096, max_entries=64)
    weight = torch.eye(2)
    calls = []

    def compute():
        values = integer_geometry(("indices",), lambda: torch.arange(16, dtype=torch.int64))
        calls.append(int(values[-1]))
        return torch.ones(1)

    with torch.inference_mode():
        for i in range(100):
            memo.evaluate("test", (), (torch.tensor([float(i)]),), weight, compute)
            assert memo.bytes + memo.integer_geometry.bytes <= memo.max_bytes
            assert len(memo.entries) + len(memo.integer_geometry.entries) <= memo.max_entries
    assert calls and memo.integer_geometry.hits
    memo.clear()
    assert not memo.entries and not memo.integer_geometry.entries
    assert memo.bytes == memo.integer_geometry.bytes == 0


def test_independent_audit_scope_releases_geometry_before_return_and_preserves_abi():
    import inspect
    import weakref

    from cubic_serve.guardrail.quantization import with_reference_integer_geometry
    from cubic_serve.quantization.common.reference import integer_geometry

    memo = ReferenceMathMemo()
    refs = []

    def original(*, reference_math_memo):
        values = integer_geometry(("owned",), lambda: torch.arange(8))
        refs.append(weakref.ref(values))
        return "original-result"

    wrapped = with_reference_integer_geometry(original)
    assert inspect.signature(wrapped) == inspect.signature(original)
    assert wrapped(reference_math_memo=memo) == "original-result"
    assert refs[0]() is None and memo.integer_geometry.bytes == 0
    assert not memo.integer_geometry.entries


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1",
                    reason="explicit CUDA audit required")
def test_integer_geometry_never_lends_evicted_device_indices_to_a_captured_graph():
    import weakref

    from cubic_serve.quantization.common.reference import (
        IntegerGeometryMemo,
        integer_geometry_scope,
    )
    from cubic_serve.quantization.extensions.int.reference import unpack

    packed = torch.tensor(
        [[0x12345678, 0x7FFFFFFF], [0x23456789, 0x01020304]],
        dtype=torch.int32, device="cuda",
    )
    geometry = IntegerGeometryMemo()
    with integer_geometry_scope(geometry):
        unpack(packed, 4, 16)
        references = [weakref.ref(t) for _, value in geometry.entries.values()
                      for t in ((value,) if isinstance(value, torch.Tensor) else value)
                      if t is not None]
        hits = geometry.hits
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = unpack(packed, 4, 16)
        assert geometry.hits == hits
    geometry.clear()
    assert all(ref() is None for ref in references)
    # Reuse ordinary allocator blocks; captured indices must belong to the graph.
    poison = [torch.zeros(64, dtype=torch.int64, device="cuda") for _ in range(64)]
    packed[0, 0] ^= 1
    graph.replay()
    torch.cuda.synchronize()
    rows = packed.cpu().tolist()
    expected = torch.tensor([[(word >> (4 * slot) & 15) - 8
                              for word in row for slot in range(8)] for row in rows],
                            dtype=torch.int8)
    assert torch.equal(output.cpu(), expected)
    assert poison


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_deferred_trajectory_metrics_preserve_values_ownership_and_partial_flush(device, dtype):
    if device == "cuda" and os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1":
        pytest.skip("explicit CUDA audit required")
    from cubic_serve.guardrail.quantized_graph.text import _TrajectoryDifferences

    metrics = _TrajectoryDifferences(max_pending=3, max_bytes=192)
    source = torch.tensor([[-0., 99.], [0., 99.], [2., 99.], [-2., 99.]],
                          dtype=dtype, device=device)
    reference = torch.tensor([0., 0., 0., -2.], dtype=dtype, device=device)
    records = [metrics(source[:, 0], reference) for _ in range(2)]
    assert len(metrics.pending) == 2 and metrics.bytes == 128
    # Every value belongs to the completed observed step, including noncontiguous
    # operands. Subsequent source reuse must not alter pending report receipts.
    source.zero_()
    reference.zero_()
    metrics.flush()
    expected = dict(elements=4, candidate_dtype=str(dtype), reference_dtype=str(dtype),
                    different=1, max_abs=2., max_abs_flat_index=2,
                    candidate_at_peak=2., reference_at_peak=0.,
                    rms=1., reference_rms=1.)
    assert records == [expected, expected]
    assert metrics.pending == [] and metrics.bytes == 0
    for size in (0, 1, 2, 4, 12, 13):
        value = torch.zeros(size, dtype=dtype, device=device)
        record = metrics(value, value)
        metrics.flush()
        assert record["elements"] == size and record["different"] == 0
        assert record["max_abs"] == record["rms"] == record["reference_rms"] == 0
        assert record["max_abs_flat_index"] == (0 if size else None)
        assert metrics.pending == [] and metrics.bytes == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_deferred_trajectory_metrics_fail_on_nonfinite_and_release_receipts(device, value):
    if device == "cuda" and os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1":
        pytest.skip("explicit CUDA audit required")
    from cubic_serve.guardrail.quantized_graph.text import _TrajectoryDifferences

    metrics = _TrajectoryDifferences(max_pending=4)
    clean = torch.zeros(7, device=device)
    metrics(clean, clean)
    damaged = clean.clone()
    damaged[-1] = value
    metrics(clean, damaged)
    with pytest.raises(AssertionError, match="independent trajectory contains nonfinite values"):
        metrics.flush()
    assert metrics.pending == [] and metrics.bytes == 0
    with pytest.raises(ValueError, match="boundary geometry differs"):
        metrics(clean, clean[:1])


def test_encoded_cache_snapshot_compacts_pages_and_preserves_immutable_operands():
    from cubic_serve.guardrail.quantized_graph.nonlinear import snapshot_cache_read
    from cubic_serve.ops.cuda.cache_quantization import PagedKVEncoding, PagedKVReadWindow
    from cubic_serve.runtime.cache.quantization import (
        CacheDType, CacheGroupAxis, CacheStorageSpec, GroupedCacheLayout,
    )

    key_layout = GroupedCacheLayout(
        CacheStorageSpec(CacheDType.INT6, CacheGroupAxis.TOKENS, 4), 8, 1, 7)
    value_layout = GroupedCacheLayout(
        CacheStorageSpec(CacheDType.BF16, CacheGroupAxis.CHANNELS, 4), 8, 1, 7)
    key = torch.arange(3 * 2 * key_layout.group_count * key_layout.packed_group_bytes)
    key = key.to(torch.uint8).reshape(3, 2, key_layout.group_count, key_layout.packed_group_bytes)
    value = torch.arange(3 * 2 * 8 * 7).bfloat16().reshape(3, 2, 8, 1, 7)
    scales = torch.arange(3 * 2 * key_layout.group_count).float().reshape(3, 2, -1) + 1
    encoding = PagedKVEncoding(key_layout, value_layout, scales, None)
    incoming = torch.arange(14).bfloat16().reshape(2, 1, 7)
    tail = torch.arange(3 * 2 * 4 * 7).bfloat16().reshape(3, 2, 4, 1, 7)
    window = PagedKVReadWindow(
        incoming, incoming + 2, torch.tensor([7, 7]), torch.arange(2, dtype=torch.int32),
        torch.tensor([2, 1]), key_tail=tail,
    )
    table = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32)
    ids = table.flatten().long()
    expected = (key[ids, 1].clone(), value[ids, 1].clone(), scales[ids, 1].clone(),
                incoming.clone(), tail[:, 1:2].clone())
    frame = snapshot_cache_read(SimpleNamespace(_encoding=encoding, _window=window, layer=1),
                                key, value, table, 2)
    for tensor in (key, value, scales, incoming, tail, table, window.start_positions):
        tensor.zero_()
    copied_encoding, copied_window, copied_key, copied_value, copied_table = frame
    assert torch.equal(copied_table, torch.tensor([[0, 1], [2, 3]], dtype=torch.int32))
    for actual, wanted in zip((copied_key[:, 0], copied_value[:, 0],
                              copied_encoding.key_scales[:, 0], copied_window.key,
                              copied_window.key_tail), expected, strict=True):
        assert torch.equal(actual, wanted)
    assert copied_encoding.value_scales is None
    assert copied_window.start_positions.tolist() == [7, 7]


@pytest.mark.parametrize("rotation", [False, True])
def test_prefix_cache_observer_retains_only_ordered_native_producer_snapshots(rotation):
    from cubic_serve.guardrail.quantized_graph.nonlinear import observe_cache_producers

    values = [torch.full((2, 1, 7), float(layer)) for layer in (1, 2)]
    original = tuple(SimpleNamespace(
        _window=SimpleNamespace(
            key=value + 100 if rotation else value, value=value + 103 if rotation else value + 3,
            producer_key=value if rotation else None,
            producer_value=value + 3 if rotation else None,
        ), launch=lambda: None,
    ) for value in values)
    body = SimpleNamespace(attentions=original)
    with observe_cache_producers(body) as records:
        for op in body.attentions:
            op.launch()
        values[0].fill_(9)
        for op in body.attentions:
            op.launch()
    assert body.attentions is original
    assert [record[0] for record in records] == [0, 1, 0, 1]
    assert records[0][1].flatten().tolist() == [1.0] * 14
    assert records[2][1].flatten().tolist() == [9.0] * 14


@pytest.mark.parametrize("expert_count", [1, 3])
@pytest.mark.parametrize("token_position", [0, 1, 2])
@pytest.mark.parametrize("recurrent", [False, True])
def test_independent_draft_feedback_survives_expert_event_processing(
    expert_count, token_position, recurrent,
):
    # Exercise the auditor's actual expert bookkeeping and feedback expression
    # without loading a real checkpoint. Distinct sentinels make a candidate
    # feedback leak into the independent trajectory observable.
    import ast
    import copy
    import inspect

    from cubic_serve.guardrail.quantized_graph.text import audit_original

    tree = ast.parse(inspect.getsource(audit_original))
    expert_loop, = (node for node in ast.walk(tree) if isinstance(node, ast.For)
                   and isinstance(node.iter, ast.Call)
                   and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "enumerate"
                   and len(node.iter.args) == 1 and isinstance(node.iter.args[0], ast.Name)
                   and node.iter.args[0].id == "base_experts")
    reference_call, = (node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == "reference_forward"
                      and node.args and isinstance(node.args[0], ast.Name)
                      and node.args[0].id == "independent_model")
    weights = [torch.empty(1) for _ in range(expert_count)]
    events = [(((weight,),), "old") for weight in weights]
    candidate, reference = object(), object()
    scope = dict(
        ordinal=token_position, position=token_position, feedback_recurrence=recurrent,
        feedback=candidate, independent_feedback=reference, independent_feedback_rows=(),
        base_experts=events, recorder=SimpleNamespace(experts=list(events)),
        names={id(w): "layer.mlp.experts.0.gate_proj.weight" for w in weights},
        local_prefixes=set(), early_partials=iter(range(expert_count)),
    )
    exec(compile(ast.Module(body=[copy.deepcopy(expert_loop)], type_ignores=[]),
                 "<actual audit expert bookkeeping>", "exec"), scope)
    selected = eval(compile(ast.Expression(copy.deepcopy(reference_call.args[-1])),
                            "<actual audit independent feedback>", "eval"), scope)
    assert selected is (reference if recurrent and token_position else candidate)


@pytest.mark.parametrize("corruption", (None, "shared", "gate", "routed", "output", "nan"))
def test_completed_moe_shared_combine_keeps_provenance_and_rejects_corruption(corruption):
    from cubic_serve.guardrail.quantized_graph.bounds import moe_shared_combine_bounds
    from cubic_serve.guardrail.quantized_graph.text import _completed_moe_combine_proof

    shared = torch.tensor([[0.5, -0.75, 1.25]], dtype=torch.bfloat16)
    gate = torch.tensor([[0.4375]], dtype=shared.dtype)
    routed = torch.tensor([[1.5, 0.5, -0.25]], dtype=shared.dtype)
    expected = dict(expected_shared=shared.clone(), expected_gate=gate.clone(),
                    expected_routed=routed.clone())
    lo, _ = moe_shared_combine_bounds(shared, gate, routed, activation_dtype=shared.dtype)
    actual = lo.to(shared.dtype)
    if corruption in ("shared", "gate", "routed"):
        {"shared": shared, "gate": gate, "routed": routed}[corruption].add_(1)
    elif corruption == "output":
        actual.add_(8)
    elif corruption == "nan":
        actual.fill_(float("nan"))
    if corruption is None:
        proof = _completed_moe_combine_proof(
            shared, gate, routed, actual, **expected, activation_dtype=shared.dtype,
        )
        assert proof["outside_enclosure"] == 0
    else:
        with pytest.raises(AssertionError):
            _completed_moe_combine_proof(
                shared, gate, routed, actual, **expected, activation_dtype=shared.dtype,
            )


@pytest.mark.parametrize("segmented", [False, True])
def test_completed_cross_m_hash_difference_is_a_closed_harness_outcome(monkeypatch, segmented):
    from cubic_serve.guardrail.quantized_graph import text_cross_m as audit

    history = []
    arena = SimpleNamespace(
        device=torch.device("cpu"), layout=SimpleNamespace(block_size_tokens=4),
        spec=SimpleNamespace(state_slot_capacity=2, page_capacity=2),
        states={"state": torch.zeros(2, 1)},
        state_specs={"state": SimpleNamespace(initial_value=0)}, page_fields={},
        key=torch.zeros(2, 1, 4, 1, 1), value=torch.zeros(2, 1, 4, 1, 1),
        written=torch.zeros(2, 1, 4, dtype=torch.bool),
        reset_state_rows=lambda rows: None, reset_page_rows=lambda rows: None,
    )
    b = SimpleNamespace(
        row_activity=torch.zeros(2, dtype=torch.int32),
        segment_lengths=torch.zeros(2, dtype=torch.int32),
        page_source=torch.zeros(2, dtype=torch.int32),
        page_destination=torch.zeros(2, dtype=torch.int32),
        page_table=torch.zeros(2, 2, dtype=torch.int32),
        state_slot_maps=(torch.zeros(2, 1, 2, dtype=torch.int32),),
        position_ids=torch.zeros(3, 2, dtype=torch.int64),
        sequence_lengths=torch.zeros(2, dtype=torch.int32),
        token_offsets=torch.zeros(2, dtype=torch.int32),
    )
    stage = SimpleNamespace(
        arena=arena, buffers=b, captured_m=2, hidden=torch.zeros(2, 1),
        stream=SimpleNamespace(wait_stream=lambda stream: None),
        prepare_scratch=lambda: None, fill_token_ids=lambda tokens: None,
        replay=lambda: history.append("replay"),
        synchronize=lambda: history.append("completed"),
    )

    def gather(output, value, **kwargs):
        output[:] = [value]
        history.append("agreement")

    monkeypatch.setattr(audit.torch.cuda, "current_stream", lambda device: None)
    monkeypatch.setattr(audit.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(audit.dist, "get_world_size", lambda *args: 1)
    monkeypatch.setattr(audit.dist, "all_gather_object", gather)
    monkeypatch.setattr(audit, "_replicated_witness_rows", lambda stage, physical, *args:
                        (list(range(len(physical))), list(range(len(physical)))))
    monkeypatch.setattr(audit, "_host_events", lambda *args: ())
    monkeypatch.setattr(audit, "_boundary_witnesses", lambda stage, rows, **kwargs:
                        [{"state": "different"} for row in rows])
    monkeypatch.setattr(audit, "_segment_snapshot", lambda *args: {"state": "different"})
    monkeypatch.setattr(audit, "_intermediate_witness", lambda *args, **kwargs: ())
    corpus = ((17, 23),)
    oracle = [[{"state": "anchor"}, {"state": "anchor"}]]
    with pytest.raises(RuntimeError, match="cross-M harness failure.*attribution unavailable"):
        if segmented:
            audit._run_segments(stage, 1, 2, corpus, oracle,
                                [[(value, ()) for value in oracle[0]]],
                                {"tp": object()}, reverse=False, events=())
        else:
            audit._run(stage, 1, corpus, oracle, {"tp": object()})
    assert history[:4] == ["replay", "completed", "agreement", "agreement"]
