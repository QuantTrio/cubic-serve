"""Independent byte/owner oracle for the sealed state-transfer primitive."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cubic_serve.executor.state_transfer import CapturedStateTransfer, _profitable_for_range


def test_cost_model_rejects_transport_without_complete_range_benefit():
    # Same payload can have different host dispatch counts. No field-count,
    # model, device or public concurrency threshold chooses an implementation.
    assert _profitable_for_range(
        {1: (0.44, 0.12), 3: (1.32, 0.28), 16: (7.04, 0.82), 32: (14.08, 1.57)}, 32
    )[0]
    assert not _profitable_for_range(
        {1: (0.028, 0.048), 3: (0.150, 0.168), 16: (0.766, 0.780), 32: (1.520, 1.529)}, 32
    )[0]
    # An unmodelled interior cost must widen the interval and reject a false win.
    assert not _profitable_for_range({1: (2.0, 1.0), 3: (6.0, 30.0), 8: (16.0, 8.0)}, 8)[0]
    assert _profitable_for_range({1: (2.0, 1.0)}, 1)[0]


@pytest.mark.parametrize(
    "samples,maximum",
    [({}, 2), ({1: (1.0, 2.0)}, 2), ({1: (float("nan"), 1.0)}, 1), ({1: (0.0, 1.0)}, 1)],
)
def test_cost_model_rejects_invalid_measurements(samples, maximum):
    with pytest.raises(ValueError):
        _profitable_for_range(samples, maximum)


def _fields():
    generator = np.random.default_rng(827)
    tensors, expected = [], []
    for dtype, shape in [
        (torch.bfloat16, (17, 3, 7)),
        (torch.float16, (17, 9)),
        (torch.float32, (17, 11)),
        (torch.float64, (17, 5)),
        (torch.int64, (17, 2)),
        (torch.int8, (17, 13)),
        (torch.bool, (17, 23)),
        (torch.int64, (17,)),
        (torch.complex128, (17, 3)),
    ]:
        t = torch.empty(shape, dtype=dtype, device="cuda:0")
        raw = generator.integers(
            0, 256, size=(17, t.numel() // 17 * t.element_size()), dtype=np.uint8
        )
        t.view(torch.uint8).reshape(17, -1).copy_(torch.from_numpy(raw))
        tensors.append(t)
        expected.append(raw.copy())
    return tensors, expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sealed CUDA state transport")
@pytest.mark.parametrize("counts", [(1,), (2, 3, 7, 1, 3), (7, 3, 2, 7)])
def test_all_bytes_protected_rows_padding_and_row_permutations(counts):
    fields, expected = _fields()
    plan = CapturedStateTransfer(fields, 7)
    assert set(plan.graphs) == {1, 2, 4, 8}
    plan.qualify()
    assert plan.qualified
    for n in counts:
        pairs = tuple((10 + i, n - i - 1) for i in range(n))
        for raw in expected:
            old = raw.copy()
            for source, destination in pairs:
                raw[destination] = old[source]
        plan.run(pairs)
    torch.cuda.synchronize()
    for t, raw in zip(fields, expected, strict=True):
        actual = t.view(torch.uint8).reshape(17, -1).cpu().numpy()
        assert actual.tobytes() == raw.tobytes()
    plan.close()
    with pytest.raises(RuntimeError, match="not sealed"):
        plan.run(((10, 0),))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sealed wide-row state transport")
def test_wide_row_transport_crosses_tile_loop_with_exact_byte_ownership():
    # A single primitive boundary, not a model x capacity x dtype matrix.
    # owner_copy limits its grid to 256 tiles of 2048 words; the final three
    # words must be reached by the next loop trip and keep their exact bits.
    width, rows = 256 * 2048 + 3, 17
    raw = np.random.default_rng(827).integers(0, 256, size=(rows, width * 2), dtype=np.uint8)
    field = torch.empty((rows, width), dtype=torch.int16, device="cuda:0")
    field.view(torch.uint8).copy_(torch.from_numpy(raw))
    plan = CapturedStateTransfer((field,), 7)
    try:
        plan.qualify()
        assert plan.qualified
        # Qualification is non-destructive even for the wide loop boundary.
        assert field.view(torch.uint8).cpu().numpy().tobytes() == raw.tobytes()
        pairs = ((16, 0), (15, 2), (14, 1), (13, 3), (12, 4), (11, 6), (10, 5))
        for origin, owner in pairs:
            raw[owner] = raw[origin]
        plan.run(pairs)
        torch.cuda.synchronize()
        # Compare all bytes: selected destinations, non-selected owners,
        # source rows, the second loop trip, and the non-vector-aligned tail.
        assert field.view(torch.uint8).cpu().numpy().tobytes() == raw.tobytes()
    finally:
        plan.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="draft checkpoint transport")
@pytest.mark.parametrize("dtype,width", [
    (torch.bfloat16, 2051), (torch.int64, 19),
    (torch.bfloat16, 256 * 2048 + 3),
])
def test_draft_checkpoint_copy_modes_preserve_exact_owner_bytes(dtype, width):
    """Prove backup/repair/selected-prefix transport without running a model."""
    import triton

    from cubic_serve.executor.state_transfer import owner_copy

    rows, capacity, depth, guard = 32, 16, 3, 16
    owners, accepted = (5, 3, 1, 4, 0, 2), (-1, 0, 1, 2, 3, 4)
    cs = depth + 3
    itemsize = torch.empty((), dtype=dtype).element_size()
    source = torch.arange(rows * width * itemsize).remainder(251).to(torch.uint8)
    destination = source ^ 0xA5
    prefix = torch.full((guard * itemsize,), 0x6D, dtype=torch.uint8)
    source_store = torch.cat((prefix, source, prefix))
    destination_store = torch.cat((prefix, destination, prefix))
    xstore = source_store.to("cuda:0").view(dtype)
    ystore = destination_store.to("cuda:0").view(dtype)
    x, y = (store[guard:-guard].view(rows, width) for store in (xstore, ystore))
    context = torch.full((1 + capacity * cs,), -1, dtype=torch.int32)
    context[0] = len(owners)
    entries = context[1:].view(capacity, cs)
    for r, owner in enumerate(owners):
        entries[r, 1] = owner
        entries[r, 2] = 8 + r * depth
        entries[r, 3:] = torch.arange(8 + r * depth + depth - 1, 8 + r * depth - 1, -1)
    decisions = torch.zeros((capacity, 2), dtype=torch.int64)
    decisions[:len(accepted), 0] = torch.tensor(accepted)
    ctx, dec = context.cuda(), decisions.cuda()
    for mode in (0, 2, 3):
        inplace = mode == 2
        expected = (source_store if inplace else destination_store).clone()
        expected_rows = expected[guard * itemsize:-guard * itemsize].view(rows, -1)
        for r, (owner, accept) in enumerate(zip(owners, accepted, strict=True)):
            origin = None
            if mode == 0:
                origin = owner
            elif mode == 2 and 0 <= accept < depth:
                origin = int(entries[r, 3 + accept])
            elif mode == 3 and 0 < accept <= depth:
                origin = int(entries[r, 2]) + accept - 1
            if origin is not None:
                expected_rows[owner] = source.view(rows, -1)[origin]
        out, storage = (x, xstore) if inplace else (y, ystore)
        tiles = min(triton.cdiv(width, 2048), 256)

        def launch():
            owner_copy[(capacity, tiles)](
                ctx, dec, x, out, cs, width, mode, 2048, tiles, depth,
            )

        launch()
        torch.cuda.synchronize()
        assert torch.equal(storage.view(torch.uint8).cpu(), expected)
        if not inplace:
            assert torch.equal(xstore.view(torch.uint8).cpu(), source_store)
        xstore.view(torch.uint8).copy_(source_store)
        ystore.view(torch.uint8).copy_(destination_store)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                launch()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(storage.view(torch.uint8).cpu(), expected)
            if not inplace:
                assert torch.equal(xstore.view(torch.uint8).cpu(), source_store)
            assert torch.equal(ctx.cpu(), context)
            assert torch.equal(dec.cpu(), decisions)
        finally:
            graph.reset()
        xstore.view(torch.uint8).copy_(source_store)
        ystore.view(torch.uint8).copy_(destination_store)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sealed CUDA state transport")
@pytest.mark.parametrize(
    "pairs",
    [
        ((10, 0), (11, 0)),
        ((10, 11), (11, 1)),
        ((17, 0),),
        ((-1, 0),),
        ((True, 0),),
        tuple((10 + i, i) for i in range(8)),
    ],
)
def test_invalid_ownership_fails_before_writing(pairs):
    fields, expected = _fields()
    plan = CapturedStateTransfer(fields, 7)
    with pytest.raises(ValueError, match="ownership"):
        plan.run(pairs)
    torch.cuda.synchronize()
    for t, raw in zip(fields, expected, strict=True):
        assert t.view(torch.uint8).reshape(17, -1).cpu().numpy().tobytes() == raw.tobytes()
    plan.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sealed CUDA state transport")
@pytest.mark.parametrize("enabled", [False, True])
def test_cold_selection_restores_state_hot_selection_reuses_decision(
    monkeypatch, tmp_path, enabled
):
    import cubic_serve.executor.state_transfer as transfer
    import cubic_serve.ops.cuda.launch_cost as costs

    fields, expected = _fields()
    monkeypatch.setattr(costs, "_launch_calibration_path", lambda *_, **kw: tmp_path / "receipt.json")
    fits = []

    def fit(samples, maximum):
        fits.append((samples, maximum))
        return enabled, {"test_fixture": True}

    monkeypatch.setattr(transfer, "_profitable_for_range", fit)
    from cubic_serve.ops.cuda.startup import startup_weight_identity
    with startup_weight_identity(None, allow_calibration=True):
        cold = CapturedStateTransfer.select(fields, 7)
    assert (cold is not None) == enabled
    assert len(fits) == 1
    for t, raw in zip(fields, expected, strict=True):
        assert t.view(torch.uint8).reshape(17, -1).cpu().numpy().tobytes() == raw.tobytes()
    if cold is not None:
        cold.close()
    monkeypatch.setattr(transfer, "_profitable_for_range", lambda *_: pytest.fail("hot refit"))
    hot = CapturedStateTransfer.select(fields, 7)
    assert (hot is not None) == enabled
    if hot is not None:
        hot.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sealed CUDA state transport")
def test_field_alias_and_storage_validation():
    field = torch.zeros((17, 3), device="cuda:0")
    plan = CapturedStateTransfer((field, field.view(torch.int32)), 7)
    assert len(plan.fields) == 1
    plan.close()
    with pytest.raises(ValueError, match="overlap"):
        CapturedStateTransfer((field, field[1:]), 7)
    with pytest.raises(ValueError, match="contiguous"):
        CapturedStateTransfer((field.T,), 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sealed shared-field state transport")
def test_interleaved_fields_match_independent_whole_backing_byte_oracle():
    from cubic_serve.runtime.cache.shared_pool import SharedCacheField, SharedCacheLayout
    layout = SharedCacheLayout((SharedCacheField("matrix", torch.bfloat16, (8, 16)),
                                SharedCacheField("history", torch.int64, (3,)),
                                SharedCacheField("flags", torch.bool, (7,))))
    raw = torch.full((17 * layout.entry_bytes,), 0x5A, dtype=torch.uint8, device="cuda:0")
    fields = layout.views(raw, 17)
    for index in range(17):
        for value in fields.values():
            value[index].view(torch.uint8).fill_(index)
    before = bytes(raw.cpu().tolist())
    plan = CapturedStateTransfer(tuple(fields.values()), 7)
    plan.qualify()
    assert bytes(raw.cpu().tolist()) == before
    pairs = ((10, 2), (11, 0), (12, 1))
    expected = bytearray(before)
    for source, destination in pairs:
        for field in layout.fields:
            source_offset = source * layout.entry_bytes + layout.offsets[field.name]
            destination_offset = destination * layout.entry_bytes + layout.offsets[field.name]
            expected[destination_offset:destination_offset + field.nbytes] = before[
                source_offset:source_offset + field.nbytes]
    plan.run(pairs)
    torch.cuda.synchronize()
    assert bytes(raw.cpu().tolist()) == bytes(expected)
    plan.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sealed CUDA state transport")
def test_bound_state_repair_preserves_exact_target_provider_and_feedback(monkeypatch):
    from contextlib import ExitStack
    from types import SimpleNamespace as S

    from cubic_serve.executor.draft_graph import prepare_state_transfers
    from cubic_serve.runtime.performance import _repair_speculative_state

    fields, expected = _fields()
    target_fields = {str(i): t for i, t in enumerate(fields[:4])}
    provider_fields = {str(i): t for i, t in enumerate(fields[4:-1])}
    arena = S(states=target_fields, spec=S(state_slot_capacity=7))
    targets = {n: S(body=S(arena=arena), _target_hidden_slots=fields[-1]) for n in (4, 8)}
    providers = {n: S(state_fields=provider_fields) for n in (1, 2, 4, 8)}
    constructions = []

    def select(fs, maximum):
        plan = CapturedStateTransfer(fs, maximum)
        constructions.append(plan)
        return plan

    monkeypatch.setattr(CapturedStateTransfer, "select", select)
    with ExitStack() as lifetime:
        prepare_state_transfers(providers, targets, lifetime)
        assert len(constructions) == 1
        rows = tuple(S(state_destination_slot=n) for n in (10, 0, 11, 1, 12, 2))
        segments = tuple(S(start=2 * i, stop=2 * i + 2, draft_count=1) for i in range(3))
        _repair_speculative_state(
            targets[8], fields[-1], rows, segments, (0, 1, 0), draft_state_fields=provider_fields
        )
        torch.cuda.synchronize()
        for t, raw in zip(fields, expected, strict=True):
            old = raw.copy()
            raw[0] = old[10]
            raw[2] = old[12]
            assert t.view(torch.uint8).reshape(17, -1).cpu().numpy().tobytes() == raw.tobytes()
        # A repeated startup bind aliases the same qualified storage and must
        # neither capture another graph nor reinitialize the live state.
        prepare_state_transfers(providers, targets, lifetime)
        assert len(constructions) == 1
        for t, raw in zip(fields, expected, strict=True):
            assert t.view(torch.uint8).reshape(17, -1).cpu().numpy().tobytes() == raw.tobytes()
        with pytest.raises(RuntimeError, match="provider storage"):
            _repair_speculative_state(
                targets[8],
                fields[-1],
                rows,
                segments,
                (0, 1, 0),
                draft_state_fields=dict(provider_fields),
            )
        with pytest.raises(RuntimeError, match="target storage"):
            _repair_speculative_state(
                targets[8],
                fields[-1].clone(),
                rows,
                segments,
                (0, 1, 0),
                draft_state_fields=provider_fields,
            )
    assert not constructions[0].graphs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bounded CUDA audit restore")
def test_transfer_failed_byte_proof_restores_all_fields_from_disk(monkeypatch, tmp_path):
    from cubic_serve.runtime import quantized_qualification as q

    fields, originals = _fields()
    plan = CapturedStateTransfer(fields, 7)
    images = []
    original_image = q._DraftBucketSnapshot
    def snapshot(*args, **kwargs):
        kwargs["directory"] = tmp_path
        image = original_image(*args, **kwargs)
        assert image.file is not None and image.memory is None
        images.append(image)
        return image
    monkeypatch.setattr(q, "_draft_witness_budget", lambda ranks: 512)
    monkeypatch.setattr(q, "_DraftBucketSnapshot", snapshot)
    def fail(_pairs):
        raise RuntimeError("injected transport execution failure")
    monkeypatch.setattr(plan, "run", fail)
    try:
        with pytest.raises(RuntimeError, match="injected transport"):
            plan.qualify()
        assert not plan.qualified
        assert len(images) == 1 and images[0].closed
        assert not tuple(tmp_path.iterdir())
        for field, expected in zip(fields, originals, strict=True):
            assert field.view(torch.uint8).reshape(17, -1).cpu().numpy().tobytes() == expected.tobytes()
    finally:
        plan.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sealed copy binding proof")
def test_geometry_shared_byte_proof_checks_every_actual_field_binding():
    from cubic_serve.guardrail.graph_observer import graph_nodes
    from cubic_serve.executor.resident_window import check
    from cuda.bindings import runtime as rt
    fields = [torch.arange(17 * 19, device='cuda:0', dtype=torch.int64).reshape(17, 19)
              for _ in range(4)]
    originals = [field.clone() for field in fields]
    plan = CapturedStateTransfer(fields, 7)
    try:
        plan.qualify()
        assert plan._byte_proof['all_field_bindings_exact']
        assert plan._byte_proof['fields'] == 4 and plan._byte_proof['geometries'] == 1
        assert plan._byte_proof['snapshot_payload_bytes'] == fields[0].numel() * fields[0].element_size()
        assert all(torch.equal(field, old) for field, old in zip(fields, originals, strict=True))
        assert all(len(graph_nodes(graph.raw_cuda_graph())) == 4 for graph in plan.graphs.values())
        # An extra copy node cannot hide behind a successful representative.
        plan.qualified = False
        check(rt.cudaGraphAddEmptyNode(plan.graphs[1].raw_cuda_graph(), [], 0))
        with pytest.raises(RuntimeError, match='field inventory differs'):
            plan.qualify()
        assert not plan.qualified
        assert all(torch.equal(field, old) for field, old in zip(fields, originals, strict=True))
    finally:
        plan.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="copy program alignment")
def test_equal_shapes_with_different_pointer_alignment_keep_their_own_program_witness():
    fields = [torch.arange(17 * 19 + offset, device='cuda:0', dtype=torch.int64)[offset:]
              .reshape(17, 19) for offset in (0, 1)]
    originals = [field.cpu().clone() for field in fields]
    plan = CapturedStateTransfer(fields, 7)
    try:
        plan.qualify()
        assert 1 <= plan._byte_proof['geometries'] <= 2
        pairs = tuple((7 + i, i) for i in range(7))
        plan.run(pairs)
        torch.cuda.synchronize()
        for field, expected in zip(fields, originals, strict=True):
            source = expected.clone()
            for origin, destination in pairs:expected[destination] = source[origin]
            assert torch.equal(field.cpu(), expected)
    finally:
        plan.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual copy pointer provenance")
def test_geometry_proof_rejects_duplicate_actual_field_arguments_before_mutating_data():
    import ctypes as ct
    from cubic_serve.guardrail.graph_observer import graph_nodes, kernel_identity
    from cubic_serve.ops.cuda.launch_cost import _driver, _check, _KernelNodeParams

    fields = [torch.arange(17 * 19, device='cuda:0', dtype=torch.int64).reshape(17, 19)
              for _ in range(2)]
    originals = [field.clone() for field in fields]
    plan = CapturedStateTransfer(fields, 7)
    try:
        raw = plan.graphs[1].raw_cuda_graph()
        nodes = graph_nodes(raw)
        first = kernel_identity(nodes[0])
        node = nodes[1]
        identity = kernel_identity(node)
        args = [ct.create_string_buffer(value, len(value)) for _, value in identity[-1]]
        # Borrow the first field's source/destination, leaving the second field
        # absent from the real graph despite an otherwise equal geometry.
        for i in (2, 3):args[i] = ct.create_string_buffer(first[-1][i][1], 8)
        pointers = (ct.c_void_p * len(args))(*(ct.addressof(arg) for arg in args))
        params = _KernelNodeParams()
        driver = _driver()
        driver.cuGraphKernelNodeSetParams_v2.argtypes = [ct.c_void_p, ct.POINTER(_KernelNodeParams)]
        _check(driver.cuGraphKernelNodeGetParams_v2(int(node), ct.byref(params)))
        params.parameters = ct.addressof(pointers)
        _check(driver.cuGraphKernelNodeSetParams_v2(int(node), ct.byref(params)))
        with pytest.raises(RuntimeError, match='field binding differs'):
            plan.qualify()
        assert not plan.qualified
        assert all(torch.equal(field, old) for field, old in zip(fields, originals, strict=True))
    finally:
        plan.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cold copy selector restore")
@pytest.mark.parametrize('failed', [False, True])
def test_cold_selector_uses_bounded_backup_and_restores_exactly(monkeypatch, tmp_path, failed):
    from cubic_serve.executor import state_transfer as module
    from cubic_serve.runtime import quantized_qualification as q
    from cubic_serve.ops.cuda import launch_cost, startup

    fields, originals = _fields()
    monkeypatch.setattr(launch_cost, '_read_startup_calibration', lambda *a: None)
    monkeypatch.setattr(launch_cost, '_write_startup_calibration', lambda *a: None)
    monkeypatch.setattr(startup, 'calibration_allowed', lambda: True)
    monkeypatch.setattr(module, '_profitable_for_range', lambda *a: (True, {}))
    monkeypatch.setattr(q, '_draft_witness_budget', lambda ranks: 512)
    original_image, images, plans = q._DraftBucketSnapshot, [], []
    def image(*args, **kwargs):
        value = original_image(*args, **kwargs, directory=tmp_path)
        assert value.file is not None
        images.append(value)
        return value
    monkeypatch.setattr(q, '_DraftBucketSnapshot', image)
    original_init = CapturedStateTransfer.__init__
    def initialize(self, *a, **kw):
        original_init(self, *a, **kw)
        plans.append(self)
    monkeypatch.setattr(CapturedStateTransfer, '__init__', initialize)
    if failed:
        def fail(self, pairs):raise RuntimeError('injected selector copy failure')
        monkeypatch.setattr(CapturedStateTransfer, 'run', fail)
    plan = None
    try:
        if failed:
            with pytest.raises(RuntimeError, match='injected selector'):
                CapturedStateTransfer.select(fields, 7)
            assert len(plans) == 1 and not plans[0].graphs
        else:
            plan = CapturedStateTransfer.select(fields, 7)
            assert plan is not None and not plan.qualified
        assert len(images) == 1 and images[0].closed and not tuple(tmp_path.iterdir())
        for field, expected in zip(fields, originals, strict=True):
            assert field.view(torch.uint8).reshape(17, -1).cpu().numpy().tobytes() == expected.tobytes()
    finally:
        if plan is not None:plan.close()
