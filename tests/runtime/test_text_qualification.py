"""Typed identity witnesses must observe auxiliary integer and floating state."""

from types import SimpleNamespace

import pytest
import torch

from cubic_serve.guardrail.quantized_graph import text_cross_m as _seal_audit
from cubic_serve.runtime.text_qualification import boundary_snapshot, compare_snapshot


def _startup_receipt_session(monkeypatch, root, *, recording=False, key="b2s256:test"):
    from cubic_serve.runtime import quantized_qualification as q

    monkeypatch.setattr(q.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(q.dist, "all_gather_object", lambda out, value: out.__setitem__(0, value))
    return q.StartupQualification(key, 0, recording=recording, root=root)


def _native_receipt_owner(image="unit-cubin", **values):
    artifact = SimpleNamespace(backend="cuda-driver", stage_hashes=(("cubin", image),))
    return SimpleNamespace(**values, project_artifacts=lambda: (artifact,))


def test_compaction_reproves_fresh_instance_without_portable_executable_images(monkeypatch, tmp_path):
    from cubic_serve.executor.resident_window import CapturedRowCompaction

    actions = []
    for _ in range(2):
        session = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
        owner = CapturedRowCompaction.__new__(CapturedRowCompaction)
        session.compaction(owner, (4, "same-geometry"), lambda: actions.append("proof"))
        assert owner.qualified is True
        assert session.hits == 0
        session.finish()
    assert actions == ["proof", "proof"]


def test_failed_fresh_compaction_cannot_restore_qualification_or_replace_prior(monkeypatch, tmp_path):
    from cubic_serve.executor.resident_window import CapturedRowCompaction

    first = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
    first.compaction(CapturedRowCompaction.__new__(CapturedRowCompaction), 4, lambda: None)
    first.finish()
    prior = first.path.read_bytes()
    second = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
    owner = CapturedRowCompaction.__new__(CapturedRowCompaction)
    def failed():
        raise RuntimeError("fresh compaction proof failed")
    with pytest.raises(RuntimeError, match="fresh compaction proof failed"):
        second.compaction(owner, 4, failed)
    assert getattr(owner, "qualified", False) is False
    with pytest.raises(RuntimeError, match="cannot publish"):
        second.finish()
    assert first.path.read_bytes() == prior


@pytest.mark.parametrize("sealed", [False, True])
def test_bound_plan_memoization_is_owned_by_a_sealed_stage(sealed):
    from cubic_serve.runtime.quantized_qualification import _bound_composition_plan

    calls = []
    artifact = SimpleNamespace(artifact_digest="image-digest")
    class Program:
        def project_artifacts(self):
            calls.append(1)
            return (artifact,)
    stage = SimpleNamespace(program=Program(), _captured_once=sealed)
    first = _bound_composition_plan(stage)
    assert _bound_composition_plan(stage) == first
    assert len(calls) == (1 if sealed else 2)
    assert hasattr(stage, "_sealed_launch_plan_digest") is sealed


def test_kernel_domain_is_a_set_and_workload_keeps_order_count_and_geometry():
    from cubic_serve.runtime.quantized_qualification import _bound_composition_plan as plan
    def artifact(image, geometry):
        return SimpleNamespace(backend="cuda-driver", stage_hashes=(("cubin", image),),
            specialization_json='{"width": %d}' % geometry, grid=(1, 1, 1),
            compiled_hash="compiler-cache-key-irrelevant")
    a, b = artifact("binary-A", 64), artifact("binary-B", 128)
    class Program:
        def __init__(self, artifacts): self.artifacts = artifacts
        def project_artifacts(self): return self.artifacts
    stages = [SimpleNamespace(program=Program(x)) for x in ((a, b), (b, a, a),
               (artifact("binary-A", 256), b))]
    domains = [plan(s, kernels_only=True) for s in stages]
    workloads = [plan(s, kernels_only=True, workload=True) for s in stages]
    assert len(set(domains)) == 1
    assert len(set(workloads)) == 3
    a.compiled_hash = "new-compiler-version"
    assert plan(stages[0], kernels_only=True) == domains[0]


def test_vendor_workload_is_portable_between_cuda_ordinals():
    import json
    from cubic_serve.runtime.quantized_qualification import _bound_composition_plan as plan
    artifact = SimpleNamespace(backend="torch-cublas", backend_binary_hashes=(("libcublas", "image"),),
        specialization_json=json.dumps({"device": ["cuda:0", "GPU"], "x": [[1, 64], "bf16"]}),
        grid=(1, 1, 1))
    class Program:
        def project_artifacts(self): return (artifact,)
    stage = SimpleNamespace(program=Program())
    initial = plan(stage, kernels_only=True, workload=True)
    artifact.specialization_json = json.dumps({"device": ["cuda:7", "GPU"], "x": [[1, 64], "bf16"]})
    assert plan(stage, kernels_only=True, workload=True) == initial


def test_backing_resource_changes_workload_without_changing_kernel_namespace():
    from cubic_serve.runtime.quantized_qualification import _bound_composition_plan as plan

    artifact = SimpleNamespace(backend="cuda-driver", stage_hashes=(("cubin", "same-image"),),
                               specialization_json='{}', grid=(1, 1, 1))
    class Program:
        def project_artifacts(self):
            return (artifact,)
    owner = Program()
    owner.workload_resource_identity = ("mmap", 4096)
    image = plan(owner, kernels_only=True)
    workload = plan(owner, kernels_only=True, workload=True)
    owner.workload_resource_identity = ("host-numa", 2097152, (0, 0))
    assert plan(owner, kernels_only=True) == image
    assert plan(owner, kernels_only=True, workload=True) != workload


def test_primitive_cost_workload_binds_geometry_and_retained_jit_image():
    import triton.language as tl
    from triton.compiler.compiler import CompiledKernel
    from cubic_serve.ops.cuda.linear_attention.gated_delta_decode import GatedDeltaDecodePrimitive
    from cubic_serve.runtime.quantized_qualification import _bound_composition_plan as plan

    op = GatedDeltaDecodePrimitive.__new__(GatedDeltaDecodePrimitive)
    op._meta = (512, 4, 12, 128, 128, 1280, torch.device('cuda:0'), 128**-.5, 1e-6, True)
    op._abis = (((512, 2560), (2560, 1), torch.bfloat16, torch.device('cuda:0')),)*11
    op._activation_type, op._attention_precision = tl.bfloat16, 'auto'
    op._block_v, op._block_k, op._warps, op._stages = 64, 128, 4, 1
    kernel = CompiledKernel.__new__(CompiledKernel)
    kernel.asm, kernel.hash = {'cubin': b'first compiled image'}, 'compiler-key'
    op._compiled_recurrence = kernel
    stage = SimpleNamespace(program=(op,))
    image = plan(stage, kernels_only=True)
    initial = plan(stage, kernels_only=True, workload=True)
    op._block_v = 128
    assert plan(stage, kernels_only=True) == image
    assert plan(stage, kernels_only=True, workload=True) != initial
    kernel.asm = {'cubin': b'different compiled image'}
    assert plan(stage, kernels_only=True) != image



@pytest.mark.parametrize("kernels_only", [False, True])
def test_dense_stage_plan_follows_bound_program_and_nested_overlap(kernels_only):
    from cubic_serve.executor.quantized_model_runner import QuantizedDecodeStage
    from cubic_serve.executor.text_graph import BoundTextBlock
    from cubic_serve.models.qwen3_5.model import Qwen3_5ReferenceModel
    from cubic_serve.runtime.quantized_qualification import _bound_composition_plan as plan

    class Primitive:
        def __init__(self, image):
            self.image = image

        def project_artifacts(self):
            return (SimpleNamespace(
                backend="cuda-driver", stage_hashes=(("cubin", self.image),),
                artifact_digest=self.image,
            ),)

    class OpaqueExecution:
        def __call__(self):
            pytest.fail("identity inspection must never execute a callback")

    def stage(image):
        value = QuantizedDecodeStage.__new__(QuantizedDecodeStage)
        value.body = BoundTextBlock(torch.empty(0), (Primitive(image),))
        value._overlap_factory = {"model_factory": Qwen3_5ReferenceModel}
        value._overlap_resources = None
        return value

    parent = stage("parent")
    children = [stage(f"child-{i}") for i in range(4)]
    initial = plan(parent, kernels_only=kernels_only)
    # Startup-only constructors must not become live execution edges.
    parent._overlap_factory["model_factory"] = OpaqueExecution
    assert plan(parent, kernels_only=kernels_only) == initial
    parent._overlap_resources = (None, tuple(children), None, None)
    nested = plan(parent, kernels_only=kernels_only)
    assert nested != initial
    for child in children:
        original = child.body
        child.body = BoundTextBlock(torch.empty(0), (Primitive("changed-child"),))
        assert plan(parent, kernels_only=kernels_only) != nested
        child.body = original
    child = children[-1]
    child.body = BoundTextBlock(torch.empty(0), (OpaqueExecution(),))
    with pytest.raises(TypeError, match="unbound external execution callable"):
        plan(parent, kernels_only=kernels_only)


def test_startup_receipt_restores_flags_and_complete_costs_without_running_fixtures(monkeypatch, tmp_path):
    session = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
    original = _native_receipt_owner(qualified=False, evidence={})
    costs = {(4, False): [12.5, 12.75], (4, True, "schema"): [14.25]}
    def qualify():
        original.qualified = True
        original.evidence = {"costs": costs, "cases": [(1, "padded")]}
    original.qualify = qualify
    session.component(original, ("artifact", (4, 16)), attributes=("qualified", "evidence"))
    assert session.run("target-costs", (4,), lambda: ((4, 8.5),)) == ((4, 8.5),)
    session.finish()

    hot = _startup_receipt_session(monkeypatch, tmp_path)
    def forbidden():
        pytest.fail("normal startup executed an engineering fixture or timing run")
    restored = _native_receipt_owner(qualified=False, evidence={}, qualify=forbidden)
    hot.component(restored, ("artifact", (4, 16)), attributes=("qualified", "evidence"))
    assert restored.qualified
    assert restored.evidence == original.evidence
    assert hot.run("target-costs", (4,), forbidden) == ((4, 8.5),)
    hot.finish()
    assert hot.hits == 2


@pytest.mark.parametrize("fault", ["missing", "corrupt", "changed-identity", "unconsumed"])
def test_startup_receipt_misses_never_turn_into_tests(monkeypatch, tmp_path, fault):
    session = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
    session.run("capture", ("binary", 4), lambda: "passed")
    session.finish()
    if fault == "missing":
        session.path.unlink()
    elif fault == "corrupt":
        session.path.write_text('{"body": {}}')
    def forbidden():
        pytest.fail("cache miss ran a fixture")
    with pytest.raises(RuntimeError, match="startup evidence"):
        hot = _startup_receipt_session(monkeypatch, tmp_path)
        if fault == "unconsumed":
            hot.finish()
        else:
            hot.run("capture", ("different-binary", 4), forbidden)


def test_startup_receipt_is_not_published_by_partial_qualification(monkeypatch, tmp_path):
    session = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
    session.run("first", (), lambda: True)
    with pytest.raises(ValueError):
        session.run("failed", (), lambda: (_ for _ in ()).throw(ValueError("fixture failed")))
    assert not session.path.exists()
    with pytest.raises(RuntimeError, match="cannot publish"):
        session.finish()


def test_recording_reuses_complete_native_proofs_and_runs_missing_operations(monkeypatch, tmp_path):
    import json
    first = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
    value = {"qualified": True, "costs": {(4, False): [10.25, 10.5]}}
    first.run("native", ("binary", 4, "BF16"), lambda: value)
    first.finish()
    prior_cohort = first.receipt_set
    second = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
    def forbidden():
        pytest.fail("matching completed native proof was recomputed")
    assert second.run("native", ("binary", 4, "BF16"), forbidden) == value
    calls = []
    assert second.run("new-proof", ("new-obligation",),
                      lambda: calls.append("executed") or "witness") == "witness"
    second.finish()
    assert calls == ["executed"] and second.hits == 1
    body = json.loads(second.path.read_text())["body"]
    assert set(body["records"]) == {"native:0", "new-proof:0"}
    assert body["reused_from"] == {"receipt_set": prior_cohort, "operations": ["native:0"]}
    hot = _startup_receipt_session(monkeypatch, tmp_path)
    assert hot.run("native", ("binary", 4, "BF16"), forbidden) == value
    assert hot.run("new-proof", ("new-obligation",), forbidden) == "witness"
    hot.finish()


def test_recording_component_rechecks_a_changed_executable_image(monkeypatch, tmp_path):
    calls = []
    for image in ("first-cubin", "changed-cubin"):
        owner = _native_receipt_owner(image, qualified=False)
        def qualify(owner=owner):
            calls.append(image)
            owner.qualified = True
        owner.qualify = qualify
        session = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
        session.component(owner, ("unchanged-source-and-shape",))
        session.finish()
        assert owner.qualified and session.hits == 0
    assert calls == ["first-cubin", "changed-cubin"]


def test_recording_component_without_bound_images_never_reuses_flags(monkeypatch, tmp_path):
    calls = []
    for _ in range(2):
        owner = SimpleNamespace(qualified=False)
        def qualify(owner=owner):
            calls.append(1)
            owner.qualified = True
        owner.qualify = qualify
        session = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
        session.component(owner, ("opaque-shape",))
        session.finish()
        assert session.hits == 0
    assert calls == [1, 1]
    hot = _startup_receipt_session(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="startup evidence"):
        hot.component(owner, ("opaque-shape",))
    assert calls == [1, 1]


def test_unknown_composition_identity_cannot_restore_a_prior_verdict(monkeypatch, tmp_path):
    for expected in ("first", "second"):
        session = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
        assert session.run("composition", None, lambda: expected) == expected
        session.finish()
        assert session.hits == 0


def test_composition_key_binds_admitted_protection_extent_not_only_virtual_shape(monkeypatch):
    from cuda.bindings import driver
    from cubic_serve.runtime import quantized_qualification as q
    from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout

    arena = CacheArena(CacheArenaSpec("proof", 4, 0, (), scratch_capacity=4,
                                    reserved_page_capacity=8), KVLayout((0,), 16, 1, 2),
                       "cpu", rank_id=0)
    body = SimpleNamespace(arena=arena, hidden=torch.empty(0))
    stage = SimpleNamespace(body=body)
    round_ = SimpleNamespace(depth=3, stride=4, C=1)
    admission = SimpleNamespace(max_sequence_tokens=64, mamba_ssm_cache_dtype="bf16",
        operator_config=SimpleNamespace(digest="config", activation_dtype="bf16",
                                       attention_dtype="bf16", softmax_dtype="fp32"))
    identity = SimpleNamespace(inventory=SimpleNamespace(inventory_digest="inventory"),
        parallel_plan=SimpleNamespace(semantic_plan_digest="semantics"),
        sealed=SimpleNamespace(topology_digest="topology", partition_digest="partition"))
    monkeypatch.setattr(q, "execution_identity", lambda _: identity)
    monkeypatch.setattr(q, "_bound_composition_plan", lambda *a, **k: "bound-image")
    monkeypatch.setattr(q.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(q.dist, "all_gather_object", lambda out, value: out.__setitem__(0, value))
    monkeypatch.setattr(q.torch.cuda, "get_device_properties", lambda _: SimpleNamespace(
        uuid="unit-device", multi_processor_count=4, L2_cache_size=4096))
    monkeypatch.setattr(q.torch.cuda, "get_device_capability", lambda _: (8, 0))
    monkeypatch.setattr(driver, "cuDriverGetVersion", lambda: (0, 13000))
    shape = tuple(arena.key.shape)
    before = q._draft_composition_receipt_key({(4, 3): round_}, {1: stage}, {4: stage}, admission)
    assert before is not None
    arena.finalize_page_capacity(5)
    assert tuple(arena.key.shape) == shape
    after = q._draft_composition_receipt_key({(4, 3): round_}, {1: stage}, {4: stage}, admission)
    assert after is not None and after != before


@pytest.mark.parametrize("fault", ["missing", "cohort", "identity", "payload", "configuration"])
def test_recording_recomputes_unusable_prior_proofs(monkeypatch, tmp_path, fault):
    import json
    from cubic_serve.guardrail.compare import stable_digest
    first = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
    first.run("native", ("binary", 4), lambda: "old")
    first.finish()
    if fault == "missing":
        first.path.unlink()
    elif fault in ("cohort", "payload"):
        raw = json.loads(first.path.read_text())
        body = raw["body"]
        if fault == "payload":
            body["records"]["native:0"]["value"] = {"invalid-packing": 1}
            body["receipt_set"] = stable_digest([stable_digest(body["records"])])
        else:
            body["receipt_set"] = "different-cohort"
        raw["digest"] = stable_digest(body)
        first.path.write_text(json.dumps(raw))
    key = "b2s256:different-configuration" if fault == "configuration" else "b2s256:test"
    second = _startup_receipt_session(monkeypatch, tmp_path, recording=True, key=key)
    calls = []
    identity = ("binary", 8) if fault == "identity" else ("binary", 4)
    assert second.run("native", identity, lambda: calls.append(1) or "new") == "new"
    second.finish()
    assert calls == [1] and second.hits == 0


@pytest.mark.parametrize("fault", ["failed-extension", "unconsumed-prior", "unportable-result"])
def test_recording_cannot_replace_complete_receipt_with_partial_extension(monkeypatch, tmp_path, fault):
    first = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
    first.run("native", ("binary",), lambda: True)
    first.run("other-native", ("other-binary",), lambda: True)
    first.finish()
    original = first.path.read_bytes()
    second = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
    assert second.run("native", ("binary",), lambda: pytest.fail("reran native"))
    if fault == "failed-extension":
        with pytest.raises(ValueError, match="new proof failed"):
            second.run("new-proof", (), lambda: (_ for _ in ()).throw(ValueError("new proof failed")))
    elif fault == "unportable-result":
        with pytest.raises(TypeError, match="nonportable"):
            second.run("new-proof", (), lambda: torch.empty(0))
    with pytest.raises(RuntimeError, match="cannot publish"):
        second.finish()
    assert first.path.read_bytes() == original


def test_recording_rejects_a_mixed_rank_cohort_before_reuse(monkeypatch, tmp_path):
    from cubic_serve.runtime import quantized_qualification as q
    first = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
    first.run("native", (), lambda: "old")
    first.finish()
    monkeypatch.setattr(q.dist, "get_world_size", lambda: 2)
    def mixed(out, value):
        out[:] = [value, (1, "other-cohort", "other-records")]
    monkeypatch.setattr(q.dist, "all_gather_object", mixed)
    recording = q.StartupQualification("b2s256:test", 0, recording=True, root=tmp_path)
    assert not recording._can_reuse
    calls = []
    assert recording.run("native", (), lambda: calls.append(1) or "new") == "new"
    assert calls == [1] and recording.hits == 0
    with pytest.raises(RuntimeError, match="incomplete qualification cohort"):
        q.StartupQualification("b2s256:test", 0, root=tmp_path)


def test_hot_startup_restores_segmented_verifiers_in_every_bucket(monkeypatch, tmp_path):
    from cubic_serve.runtime import performance

    drafts = {m: _native_receipt_owner(_verifier_qualified=False)
              for m in (1, 2, 4, 8, 16, 32, 64, 128)}
    def qualify(drafts, targets):
        for draft in drafts.values():
            draft._verifier_qualified = True
    monkeypatch.setattr(performance, "_qualify_draft_verification", qualify)
    session = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
    session.draft_verifier(drafts, {m: None for m in drafts})
    session.finish()
    for draft in drafts.values():
        draft._verifier_qualified = False
    def forbidden(*args):
        pytest.fail("hot startup reran verifier fixtures")
    monkeypatch.setattr(performance, "_qualify_draft_verification", forbidden)
    hot = _startup_receipt_session(monkeypatch, tmp_path)
    hot.draft_verifier(drafts, {m: None for m in drafts})
    hot.finish()
    assert all(draft._verifier_qualified for draft in drafts.values())


def test_hot_startup_restores_resident_row_ownership_postcondition(monkeypatch, tmp_path):
    from cubic_serve.runtime import quantized_qualification as q

    monkeypatch.setattr(q, "_bound_composition_plan", lambda body, **kwargs: "bound-image")
    executor = SimpleNamespace(body=None, captured_m=8, max_sequence_tokens=256, _bound_active_rows=0)
    window = SimpleNamespace(e=executor, stochastic=False, qualified=False,
                             program_qualified=False, segmented_program_qualified=False)
    def qualify():
        window.qualified = window.program_qualified = window.segmented_program_qualified = True
        executor._bound_active_rows = 8
    window.qualify = qualify
    session = _startup_receipt_session(monkeypatch, tmp_path, recording=True)
    session.window(window)
    session.finish()
    window.qualified = window.program_qualified = window.segmented_program_qualified = False
    executor._bound_active_rows = 0
    window.qualify = lambda: pytest.fail("hot startup ran resident fixtures")
    hot = _startup_receipt_session(monkeypatch, tmp_path)
    hot.window(window)
    hot.finish()
    assert executor._bound_active_rows == 8
    assert window.qualified and window.program_qualified and window.segmented_program_qualified


def test_normal_startup_reads_a_miss_without_launching_calibration(monkeypatch, tmp_path):
    from cubic_serve.ops.cuda.launch_cost import _read_startup_calibration
    from cubic_serve.ops.cuda.startup import startup_weight_identity

    path = tmp_path / "missing.json"
    with startup_weight_identity(("source",), allow_calibration=False):
        assert _read_startup_calibration(path, "example") is None
    assert _read_startup_calibration(path, "example") is None


@pytest.mark.parametrize("base,maximum,key_axis,key_group,value_axis,value_group,expected", [
    (8, 4096, "tokens", 32, "channels", 32, 33),
    (8, 4096, "tokens", 64, "channels", 32, 65),
    (8, 4096, "channels", 128, "channels", 128, 8),
    (8, 4096, "tokens", 3, "tokens", 5, 8),
    (1, 4096, "tokens", 3, "tokens", 5, 6),
    (64, 4096, "tokens", 32, "channels", 32, 64),
    (8, 16, "tokens", 32, "channels", 32, 16),
])
def test_state_audit_horizon_covers_column_sealing_without_extending_row_groups(
    base, maximum, key_axis, key_group, value_axis, value_group, expected,
):
    from cubic_serve.runtime.cache.quantization import (
        CacheDType,
        CacheGroupAxis,
        CacheStorageSpec,
        KVCacheStorageSpec,
    )
    from cubic_serve.runtime.text_qualification import state_audit_horizon

    storage = KVCacheStorageSpec(
        CacheStorageSpec(CacheDType.INT6, CacheGroupAxis(key_axis), key_group),
        CacheStorageSpec(CacheDType.INT4, CacheGroupAxis(value_axis), value_group),
    )
    assert state_audit_horizon(base, maximum, storage) == expected


def test_native_state_audit_horizon_preserves_declared_extent_and_rejects_invalid_counts():
    from cubic_serve.runtime.cache.quantization import KVCacheStorageSpec
    from cubic_serve.runtime.text_qualification import state_audit_horizon

    storage = KVCacheStorageSpec().resolve(torch.bfloat16)
    assert state_audit_horizon(8, 4096, storage) == 8
    for base, maximum in ((0, 4096), (8, 0), (True, 4096), (8, 1.5)):
        with pytest.raises(ValueError):
            state_audit_horizon(base, maximum, storage)


@pytest.mark.parametrize("order", [(4, 1, 2), (2, 4, 1)])
def test_composition_fixture_selects_smallest_provider_independent_of_capture_order(
    monkeypatch, order
):
    import numpy as np

    from cubic_serve.runtime import performance
    from cubic_serve.runtime import quantized_qualification as module

    class ObservedSelection(Exception):
        pass

    arena = SimpleNamespace(spec=SimpleNamespace(page_capacity=8), written=torch.zeros(8, 1))
    target = SimpleNamespace(
        body=SimpleNamespace(arena=arena, hidden=torch.zeros(1)),
        _sampling=SimpleNamespace(constraint_bound=False),
        _block_size=32,
    )
    pool = SimpleNamespace(
        metadata=[None],
        tokens=torch.zeros(1, 1),
        scores=torch.zeros(1, 1, 1),
        ids=torch.zeros(1, 1, 1),
    )
    sampler = SimpleNamespace(invalidate_metadata=lambda: None)
    round_ = SimpleNamespace(
        target=target,
        align=SimpleNamespace(body=SimpleNamespace(arena=arena)),
        head=SimpleNamespace(_sampling=SimpleNamespace(constraint_bound=False)),
        pool=pool,
        C=1,
        depth=1,
        stride=2,
        head_greedy=sampler,
        target_greedy=sampler,
    )
    monkeypatch.setattr(module, "_draft_window_fixture_fields", lambda *a: ((), torch.tensor([0]), None, None))
    monkeypatch.setattr(module, "_initialize_draft_window_fixture", lambda *a, **kw: None)
    monkeypatch.setattr(module, "_snapshot_draft_fixture_fields", lambda *a, **kw: {})
    monkeypatch.setattr(module, "_restore_draft_fixture_fields", lambda *a: None)
    monkeypatch.setattr(module, "_draft_window_audit_rows", lambda *a, **kw: ())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)

    def observe(command, executor, providers, hidden, admission, capacities, **kwargs):
        assert performance._bucket_for_logical_m(1, capacities) == 1
        raise ObservedSelection

    monkeypatch.setattr(performance, "_execute_pipelined_draft_batch", observe)
    with pytest.raises(ObservedSelection):
        module._qualify_draft_window_case(
            2, round_, SimpleNamespace(host_flags=np.ones(1)), {},
            {capacity: None for capacity in order}, None, None, (0,), rank=0, sampler_rank=0,
        )


def _native_layout(block, page_fields=()):
    from cubic_serve.runtime.cache import KVLayout

    layout = KVLayout((0, 1), block, 1, 8)
    return SimpleNamespace(
        block_size_tokens=block, storage=layout.storage,
        key_encoding=layout.key_encoding, value_encoding=layout.value_encoding,
        page_fields=page_fields,
    )


def _qualify_mock_rounds(module, rounds):
    return module.qualify_draft_windows(
        rounds, {round_.C: None for round_ in rounds.values()},
        {capacity: None for capacity, _ in rounds}, None, None,
        rank=0, sampler_rank=0,
    )


def _stage():
    arena = SimpleNamespace(
        key=torch.zeros(3, 2, 4, 1, 8, dtype=torch.bfloat16),
        value=torch.zeros(3, 2, 4, 1, 8, dtype=torch.bfloat16),
        written=torch.zeros(3, 2, 4, dtype=torch.bool),
        states={
            "history": torch.full((3, 2), 2**54 + 1, dtype=torch.int64),
            "recurrent": torch.zeros(3, 2, 4, 4, dtype=torch.float32),
        },
        page_fields={"index_keys": torch.zeros(3, 1, 8, dtype=torch.bfloat16)},
    )
    buffers = SimpleNamespace(
        page_destination=torch.tensor([2, 1], dtype=torch.int32),
        state_slot_maps=(torch.tensor([[[2, 1]], [[2, 1]]], dtype=torch.int32),),
    )
    return SimpleNamespace(
        captured_m=2, arena=arena, buffers=buffers,
        hidden=torch.zeros(2, 8, dtype=torch.bfloat16),
        logits=torch.zeros(2, 12, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("field", ["history", "recurrent", "index_keys", "written"])
def test_snapshot_observes_typed_owner_fields_without_aliasing(field):
    stage = _stage()
    before = boundary_snapshot(stage, 1)
    original = compare_snapshot(before, boundary_snapshot(stage, 1))
    if field in stage.arena.states:
        stage.arena.states[field][2].add_(1)
    elif field in stage.arena.page_fields:
        stage.arena.page_fields[field][2].add_(1)
    else:
        stage.arena.written[2].fill_(True)
    with pytest.raises(ValueError, match=field):
        compare_snapshot(before, boundary_snapshot(stage, 1))
    assert compare_snapshot(before, before) == original


def test_padding_nan_is_not_an_active_owner_but_active_nan_fails():
    stage = _stage()
    before = boundary_snapshot(stage, 1)
    stage.arena.states["recurrent"][1].fill_(float("nan"))
    stage.arena.page_fields["index_keys"][1].fill_(float("nan"))
    compare_snapshot(before, boundary_snapshot(stage, 1))
    stage.arena.states["recurrent"][2].fill_(float("nan"))
    with pytest.raises(ValueError, match="recurrent"):
        compare_snapshot(before, boundary_snapshot(stage, 1))


def test_draft_feedback_is_a_separate_audited_boundary():
    stage = _stage()
    stage.draft_feedback = torch.zeros(2, 16, dtype=torch.bfloat16)
    before = boundary_snapshot(stage, 1)
    stage.draft_feedback[0, 5] = 1
    with pytest.raises(ValueError, match="draft_feedback"):
        compare_snapshot(before, boundary_snapshot(stage, 1))


@pytest.mark.parametrize("failed", [False, True])
def test_feedback_capture_keeps_full_forward_immutable_snapshot_references(failed):
    from cubic_serve.guardrail.quantized_graph.text import preserve_observer_collections

    tensor = torch.tensor([3])
    events, sites, names = [("head", tensor)], {"head": tensor}, {"head"}
    def capture():
        with preserve_observer_collections((events, sites, names)):
            events.clear()
            events.append(("feedback", torch.tensor([7])))
            sites.clear()
            sites["feedback"] = torch.tensor([11])
            names.add("feedback")
            if failed:
                raise RuntimeError("capture failed")
    if failed:
        with pytest.raises(RuntimeError, match="capture failed"):
            capture()
    else:
        capture()
    assert events[0][1] is tensor and sites["head"] is tensor
    assert names == {"head"} and list(sites) == ["head"]


def test_segment_snapshot_prunes_only_future_page_writes_without_mutating_storage():
    from cubic_serve.guardrail.compare import tensor_digest
    from cubic_serve.guardrail.quantized_graph.text_cross_m import _segment_snapshot

    stage = _stage()
    stage.arena.layout = _native_layout(
        4, (SimpleNamespace(field_id="index_keys", shape=(1, 8)),),
    )
    stage.indexer_plan = SimpleNamespace(compression=4, size=1, rank=0)
    stage.arena.key[2].fill_(3)
    stage.arena.value[2].fill_(5)
    stage.arena.written[2].fill_(True)
    stage.arena.page_fields["index_keys"][2].fill_(7)
    original = boundary_snapshot(stage, 1)
    expected = {name: value.clone() for name, value in original.items()}
    for name in ("key", "value", "written"):
        expected[name][:, :, 2:] = 0
    expected["page:index_keys"].zero_()
    assert _segment_snapshot(stage, 0, 2) == {
        name: tensor_digest(value) for name, value in expected.items()
    }
    compare_snapshot(original, boundary_snapshot(stage, 1))
    stage.arena.states["history"][2, 0] += 1
    assert _segment_snapshot(stage, 0, 2)["state:history"] != tensor_digest(
        expected["state:history"])


@pytest.mark.parametrize("compression", [2, 4, 8])
@pytest.mark.parametrize("parts", [1, 2, 4, 8])
def test_segment_snapshot_retains_only_completed_cyclic_owner_groups(compression, parts):
    from cubic_serve.guardrail.compare import tensor_digest
    from cubic_serve.guardrail.quantized_graph.text_cross_m import _segment_snapshot
    from cubic_serve.models.parallel_weights import IndexerTP

    block = 64
    for rank in range(parts):
        stage = _stage()
        stage.indexer_plan = IndexerTP(4, 8, compression, parts, rank)
        local_groups = stage.indexer_plan.cache_page_groups(block)
        values = torch.arange(1, local_groups * 8 + 1, dtype=torch.float32).reshape(
            1, local_groups, 8).bfloat16()
        stage.arena.page_fields["index_keys"] = values.expand(3, -1, -1).clone()
        stage.arena.layout = _native_layout(
            block, (SimpleNamespace(field_id="index_keys", shape=(local_groups, 8)),),
        )
        for consumed in sorted({0, 1, compression - 1, compression,
                                compression * (rank + 1) - 1, compression * (rank + 1),
                                block - 1, block}):
            expected = values.clone()
            for local in range(local_groups):
                # Enumerate global group ownership independently of the
                # candidate's closed-form local count.
                global_group = rank + local * parts
                if (global_group + 1) * compression > consumed:
                    expected[:, local] = 0
            assert _segment_snapshot(stage, 0, consumed)["page:index_keys"] == tensor_digest(
                expected), (compression, parts, rank, consumed)


def test_cross_m_rebuilds_the_declared_provider_input_transform(monkeypatch):
    from contextlib import nullcontext

    from cubic_serve.guardrail.quantized_graph import text_cross_m as audit

    original_weights = object()
    source = SimpleNamespace(
        captured_m=4, config=SimpleNamespace(text=SimpleNamespace(
            hyperconnection=SimpleNamespace(stream_count=3))),
        weights=SimpleNamespace(original=original_weights), topology=object(),
        buffers=SimpleNamespace(page_table=torch.zeros(4, 2)), arena=SimpleNamespace(
            layout=_native_layout(16)),
        operators=SimpleNamespace(config=object()), recurrent_state_dtype=torch.bfloat16,
        mrope_tables=object(),
    )
    calls = []
    stage = SimpleNamespace(launch=lambda: None, capture=lambda: None)
    def build(weights, topology, **options):
        calls.append((weights, topology, options))
        return stage
    monkeypatch.setattr(audit, "_observe", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(audit, "FlashNextDecodeStage", lambda *args, **kwargs:
                        pytest.fail("draft must retain its provider input transform"))
    result = audit._temporary(source, [], {}, draft_provider=SimpleNamespace(build_stage=build))
    assert result is stage
    assert calls[0][0] is original_weights
    assert calls[0][1] is source.topology
    assert calls[0][2]["captured_m"] == 4
    assert calls[0][2]["arena"] is source.arena


@pytest.mark.parametrize("recurrence,method,feedback", [
    (True, "mtp", (torch.zeros(1, 1, 3),) * 4),
    (False, "mtp", ()),
    (False, "other", (torch.zeros(1, 1, 3),) * 4),
])
def test_draft_cross_m_rejects_unlocked_feedback_provenance(recurrence, method, feedback):
    from cubic_serve.guardrail.quantized_graph.text_cross_m import audit_cross_m

    source = SimpleNamespace(
        topology=SimpleNamespace(pp_size=1),
        arena=SimpleNamespace(layout=SimpleNamespace(block_size_tokens=16)),
        config=SimpleNamespace(text=SimpleNamespace(qsa=SimpleNamespace(compression_ratio=4),
                                                  vocabulary_size=100)),
    )
    report = dict(draft_method=method, feedback_recurrence=recurrence)
    provider = SimpleNamespace(spec=SimpleNamespace(method="mtp"))
    with pytest.raises(ValueError, match="same fixed feedback trace"):
        audit_cross_m({1: SimpleNamespace(body=source)}, None, {}, report,
                      draft_provider=provider, feedback_rows=feedback)


def test_wide_review_capture_borrows_disjoint_cold_scratch():
    from cubic_serve.runtime.text_qualification import prepare_rows

    capacity, persistent = 8, 2
    mapping = torch.zeros(2, 1, capacity, dtype=torch.int32)
    arena = SimpleNamespace(
        device="cpu", spec=SimpleNamespace(page_capacity=8, state_slot_capacity=persistent),
        states={"state": torch.ones(persistent + capacity, 2)},
        key=torch.ones(16, 2), value=torch.ones(16, 2), written=torch.ones(16, 2),
        page_fields={},
    )
    arena.reset_state_rows = lambda ids: arena.states["state"].index_fill_(0, ids, 0)
    arena.reset_page_rows = lambda ids: [v.index_fill_(0, ids, 0) for v in
                                      (arena.key, arena.value, arena.written)]
    b = SimpleNamespace(
        state_slot_maps=(mapping,), row_activity=torch.zeros(capacity),
        segment_lengths=torch.zeros(capacity), page_source=torch.zeros(capacity),
        page_destination=torch.zeros(capacity, dtype=torch.int64),
        page_table=torch.zeros(capacity, 1),
    )
    def scratch():
        mapping[:] = torch.arange(persistent, persistent + capacity)[None, None]
        b.page_destination[:] = torch.arange(8, 16)
    stage = SimpleNamespace(
        captured_m=capacity, arena=arena, buffers=b, prepare_scratch=scratch,
        config=SimpleNamespace(graph=SimpleNamespace(text=SimpleNamespace(vocabulary_size=100))),
        fill_token_ids=lambda tokens: None, hidden=torch.zeros(capacity, 2),
        set_active_rows=lambda m: None,
    )
    stage.config.text_config = stage.config.graph.text
    prepare_rows(stage, 6, poison=True)
    assert mapping[1, 0].tolist() == [0, 1, 4, 5, 6, 7, 8, 9]
    assert b.page_destination.tolist() == [0, 1, 2, 3, 4, 5, 14, 15]
    assert torch.isfinite(arena.states["state"][:8]).all()
    assert torch.isnan(arena.states["state"][8:]).all()
    prepare_rows(stage, capacity)
    assert mapping[1, 0].unique().numel() == capacity
    assert torch.isfinite(arena.states["state"][mapping[1, 0].long()]).all()


def _ownership_executors(monkeypatch, *, corrupt=None):
    """CPU ownership model with capacity-dependent arithmetic and exact copies."""
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    block = 4
    arena = SimpleNamespace(
        device="cpu", layout=SimpleNamespace(block_size_tokens=block),
        spec=SimpleNamespace(state_slot_capacity=4, page_capacity=6),
        key=torch.zeros(6, block), value=torch.zeros(6, block),
        written=torch.zeros(6, block, dtype=torch.bool),
        states={"history": torch.zeros(4, 1, dtype=torch.int64),
                "recurrent": torch.zeros(4, 1)},
        page_fields={"aux": torch.zeros(6, block)},
    )

    def reset_state(owners):
        for value in arena.states.values():
            value[owners] = 0

    def reset_pages(pages):
        for value in (arena.key, arena.value, arena.written, *arena.page_fields.values()):
            value[pages] = 0

    arena.reset_state_rows = reset_state
    arena.reset_page_rows = reset_pages
    schedules = []

    def executor(capacity):
        stage = SimpleNamespace(
            captured_m=capacity, arena=arena, synchronize=lambda: None,
            config=SimpleNamespace(graph=SimpleNamespace(text=SimpleNamespace(vocabulary_size=32000))),
            hidden=torch.zeros(capacity, 1), logits=torch.zeros(capacity, 1),
            buffers=SimpleNamespace(
                page_destination=torch.zeros(capacity, dtype=torch.int64),
                state_slot_maps=(torch.zeros(2, 1, capacity, dtype=torch.int64),),
            ),
        )

        stage.config.text_config = stage.config.graph.text

        def replay(rows):
            schedules.append((rows[0].request_id, capacity, tuple(r.position for r in rows)))
            for i, row in enumerate(rows):
                source, dest = row.state_source_slot, row.state_destination_slot
                page = row.page_destination
                if row.reset_state:
                    reset_state([dest])
                elif row.segment_length and source != dest:
                    for name, value in arena.states.items():
                        value[dest] = value[source]
                        if corrupt == "state" and name == "history":
                            value[dest] += 1
                reset_pages(list(row.new_pages))
                if row.page_source != page and not row.new_pages:
                    for value in (arena.key, arena.value, arena.written,
                                  *arena.page_fields.values()):
                        value[page] = value[row.page_source]
                    if corrupt == "page":
                        arena.page_fields["aux"][page, 0] += 1
                # Deliberately distinguish capacity plans; an owner-copy audit
                # must not impose numerical equality across different plans.
                arena.states["recurrent"][dest] += row.token_id + capacity / 16
                arena.states["history"][dest] += row.token_id
                offset = row.position % block
                result = arena.states["recurrent"][dest, 0]
                arena.key[page, offset] = result
                arena.value[page, offset] = result
                arena.written[page, offset] = True
                arena.page_fields["aux"][page, offset] = row.position
                stage.hidden[i, 0] = result
                stage.logits[i, 0] = result
                stage.buffers.page_destination[i] = page
                stage.buffers.state_slot_maps[0][:, 0, i] = dest
                if corrupt == "source" and source != dest and not row.reset_state:
                    arena.states["history"][source] += 1

        return SimpleNamespace(body=stage, replay_batch=replay,
                               slot=SimpleNamespace(synchronize=lambda: None))

    return {m: executor(m) for m in (1, 2, 4)}, schedules


def test_owner_migration_matches_arithmetic_schedule(monkeypatch):
    from cubic_serve.runtime.text_qualification import audit_owner_migration

    executors, schedule = _ownership_executors(monkeypatch)
    assert audit_owner_migration(executors, max_sequence_tokens=6)
    assert [m for _, m, _ in schedule] == [1] * 6 + [1, 2, 4] * 4


@pytest.mark.parametrize("corrupt", ["state", "page"])
def test_owner_migration_rejects_lossy_copies(monkeypatch, corrupt):
    from cubic_serve.runtime.text_qualification import audit_owner_migration

    executors, _ = _ownership_executors(monkeypatch, corrupt=corrupt)
    with pytest.raises(ValueError, match="typed owner identity"):
        audit_owner_migration(executors, max_sequence_tokens=6)


def test_prefix_forks_match_prefill_schedule(monkeypatch):
    from cubic_serve.runtime.text_qualification import audit_prefix_forks

    executors, schedule = _ownership_executors(monkeypatch)
    assert audit_prefix_forks(executors, max_sequence_tokens=6)
    prefixes = [(m, positions) for _, m, positions in schedule if len(positions) > 1]
    assert prefixes == [(4, (0, 1, 2, 3))] * 2


@pytest.mark.parametrize("corrupt", ["state", "source"])
def test_prefix_forks_reject_loss_or_source_mutation(monkeypatch, corrupt):
    from cubic_serve.runtime.text_qualification import audit_prefix_forks

    executors, _ = _ownership_executors(monkeypatch, corrupt=corrupt)
    with pytest.raises(ValueError, match="capture identity"):
        audit_prefix_forks(executors, max_sequence_tokens=6)


def _replicated_stage(device):
    n = 8
    labels = torch.arange(n, device=device).remainder(2)
    pages = torch.tensor([7, 1, 5, 3, 6, 0, 2, 4], device=device)
    owners = torch.tensor([2, 5, 1, 7, 0, 4, 6, 3], device=device)
    def rows(shape, dtype, indices=None):
        source = (labels + 1).to(dtype).view(n, *([1] * len(shape)))
        source = source.expand(n, *shape).contiguous()
        if indices is None:
            return source
        storage = torch.zeros((12, *shape), dtype=dtype, device=device)
        storage.index_copy_(0, indices, source)
        return storage
    maps = torch.stack((owners, owners)).view(2, 1, n)
    history = torch.zeros(12, 1, dtype=torch.int64, device=device)
    history.index_copy_(0, owners, labels[:, None] + 2**54 + 1)
    stage = SimpleNamespace(
        captured_m=n, rank=0, hidden=rows((4,), torch.bfloat16),
        logits=rows((7,), torch.bfloat16), draft_feedback=rows((3,), torch.bfloat16),
        buffers=SimpleNamespace(page_destination=pages, state_slot_maps=(maps, maps.clone())),
        arena=SimpleNamespace(
            key=rows((2, 4, 1, 2), torch.bfloat16, pages),
            value=rows((2, 4, 1, 2), torch.bfloat16, pages),
            written=rows((2, 4), torch.bool, pages),
            states={"history": history, "recurrent": rows((2, 2), torch.bfloat16, owners)},
            page_fields={"index": rows((2, 3), torch.bfloat16, pages)},
        ),
        config=SimpleNamespace(text=SimpleNamespace(moe=SimpleNamespace(expert_count=4))),
        topology=SimpleNamespace(ep_size=2, expert_coordinates=lambda _: (0, 0)),
    )
    stage.hidden[:, 0].zero_()
    route_ids = torch.stack((labels, 3 - labels), -1)
    routes = rows((2, 4), torch.bfloat16)
    routes[:, 1] = torch.arange(n, device=device)[:, None]
    events = [("layer/route_ids/0", route_ids), ("layer/routes/0", routes)]
    physical = (5, 0, 7, 2, 3, 6, 1, 4)
    return stage, events, physical, tuple(row % 2 for row in physical)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fault", [None, "signed_zero", "integer", "state", "page", "feedback",
                                  "logits", "owner_map", "owned_route", "unused_page"])
def test_replicated_rows_require_lossless_all_field_equality_before_oracle_reuse(device, fault):
    import os

    from cubic_serve.guardrail.quantized_graph.text_cross_m import (
        _boundary_witnesses,
        _replicated_witness_rows,
    )
    if device == "cuda" and os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1":
        pytest.skip("explicit CUDA audit required")
    stage, events, physical, labels = _replicated_stage(device)
    original = _boundary_witnesses(stage, physical, max_workers=1)
    selected, mapping = _replicated_witness_rows(stage, physical, labels, events)
    assert len(selected) == 2
    witnesses = _boundary_witnesses(stage, tuple(physical[i] for i in selected), max_workers=1)
    assert [witnesses[i] for i in mapping] == original
    row = physical[-1]  # An explicitly permuted nonrepresentative physical row.
    owner = int(stage.buffers.state_slot_maps[0][1, 0, row])
    page = int(stage.buffers.page_destination[row])
    if fault == "signed_zero":
        stage.hidden[row, 0] = -0.0
    elif fault == "integer":
        stage.arena.states["history"][owner, 0] += 1
    elif fault == "state":
        stage.arena.states["recurrent"][owner, 0, 0] += 1
    elif fault == "page":
        stage.arena.page_fields["index"][page, 0, 0] += 1
    elif fault == "feedback":
        stage.draft_feedback[row, 0] += 1
    elif fault == "logits":
        stage.logits[row, 0] += 1
    elif fault == "owner_map":
        stage.buffers.state_slot_maps[1][1, 0, row] = (owner + 1) % 8
    elif fault == "owned_route":
        events[1][1][row, 0, 0] += 1
    elif fault == "unused_page":
        # This helper has no padding authority. Existing protected write-set
        # checks still run independently; an unowned row has no corpus identity.
        stage.arena.key[11].fill_(float("nan"))
    selected, _ = _replicated_witness_rows(stage, physical, labels, events)
    if fault not in (None, "unused_page"):
        assert selected == tuple(range(len(physical)))  # Preserve original attribution.
    else:
        assert len(selected) == 2


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float32, torch.bfloat16])
def test_draft_snapshot_roles_remain_independent_across_logical_owner_counts(dtype):
    from cubic_serve.runtime.quantized_qualification import _snapshot_draft_fixture_fields

    capacity = 8
    source = torch.arange(24 * 12).reshape(24, 3, 4).to(dtype).transpose(1, 2)
    buffers = tuple({} for _ in range(4))
    pointers = None
    for count in (1, 2, 5, 8, 1):
        indices = torch.cat((torch.arange(count), torch.arange(8, 8 + 2 * count)))
        values, expected = [], []
        for role in range(4):
            source.fill_(role + 1)
            values.append(_snapshot_draft_fixture_fields(
                [("state", source, indices)], buffers=buffers[role],
                owner_count=count, owner_capacity=capacity,
            )["state"])
            expected.append(source.index_select(0, indices).contiguous())
        source.zero_()
        for actual, reference in zip(values, expected, strict=True):
            assert actual.shape == (3 * count, 4, 3)
            assert torch.equal(actual.contiguous().view(torch.uint8), reference.view(torch.uint8))
        current = tuple(slot["state"].data_ptr() for slot in buffers)
        assert len(set(current)) == 4
        if pointers is not None:
            assert current == pointers
        pointers = current


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_draft_snapshot_pool_preserves_every_storage_bit_with_determinism_active(dtype):
    from cubic_serve.runtime.quantized_qualification import (
        _restore_draft_fixture_fields,
        _snapshot_draft_fixture_fields,
    )

    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    try:
        raw = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(256, 256)
        source = raw.view(dtype)
        indices = torch.arange(256)
        values = _snapshot_draft_fixture_fields([("bits", source, indices)], buffers={})
        assert torch.equal(values["bits"].view(torch.int16), raw)
        target = torch.zeros_like(source)
        _restore_draft_fixture_fields([("bits", target, indices)], values)
        assert torch.equal(target.view(torch.int16), raw)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.utils.deterministic.fill_uninitialized_memory == fill
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


@pytest.mark.parametrize("order", [(1, 2, 3, 7), (3, 1, 2, 1, 7), (-1, -2, 0, 1), ()])
def test_draft_fixture_restore_preserves_order_and_every_unowned_row(order):
    from cubic_serve.runtime.quantized_qualification import _restore_draft_fixture_fields

    original = torch.arange(10 * 12, dtype=torch.int64).reshape(10, 3, 4)
    expected, actual = original.clone(), original.clone()
    values = torch.arange(len(order) * 12).reshape(len(order), 3, 4) + 1000
    for row, index in enumerate(order):
        expected[index].copy_(values[row])
    _restore_draft_fixture_fields(
        [("state", actual, torch.tensor(order, dtype=torch.int64))], {"state": values})
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
def test_draft_word_comparison_matches_byte_equality_including_every_tail(
    monkeypatch, offset, native,
):
    from cubic_serve.runtime import quantized_qualification as module
    from cubic_serve.runtime.quantized_qualification import _equal_draft_fixture_bytes

    if not native:
        monkeypatch.setattr(module, "_CPU_MEMCMP", None)
    for length in (*range(36), 255, 256, 257):
        left = torch.arange(length + offset + 1).to(torch.uint8)[offset:offset + length]
        right = left.clone()
        assert _equal_draft_fixture_bytes(left, right)
        for position in (range(length) if length <= 36 else (0, 7, 8, length - 1)):
            right[position] ^= 1
            assert not _equal_draft_fixture_bytes(left, right)
            right[position] ^= 1
        assert torch.equal(left, right)
    assert not _equal_draft_fixture_bytes(torch.zeros(2, 4, dtype=torch.uint8),
                                         torch.zeros(4, 2, dtype=torch.uint8))


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("owner_count", [1, 2, 3])
def test_draft_feedback_identity_observes_all_repair_slots_without_dead_final_scratch(
    depth, owner_count,
):
    from cubic_serve.runtime.quantized_qualification import (
        _compare_draft_fixture_fields,
        _draft_window_fixture_fields,
        _snapshot_draft_fixture_fields,
    )

    stride = depth + 1
    scratch = tuple(range(8, 8 + 4 * stride))

    def arena():
        return SimpleNamespace(
            key=torch.zeros(8, 2), value=torch.zeros(8, 2),
            written=torch.zeros(8, 2, dtype=torch.bool),
            page_fields={}, states={"state": torch.zeros(64, 2)},
        )

    target = SimpleNamespace(
        body=SimpleNamespace(arena=arena(), hidden=torch.zeros(4 * stride, 8)),
        _scratch_states=scratch,
    )
    provider = SimpleNamespace(
        body=SimpleNamespace(arena=arena()),
        _draft_binding=SimpleNamespace(executor=SimpleNamespace(_scratch_states=scratch)),
    )
    round_ = SimpleNamespace(
        target=target, align=provider, depth=depth, stride=stride,
        pool=SimpleNamespace(backup={}),
    )
    hidden = torch.arange(64 * 8, dtype=torch.float32).reshape(64, 8)
    fields, *_ = _draft_window_fixture_fields(round_, hidden, owner_count, 1)
    selected = next(indices for name, _, indices in fields if name == "feedback")
    live = list(range(owner_count)) + [
        scratch[owner * stride + accepted]
        for owner in range(owner_count) for accepted in range(depth)
    ]
    assert selected.tolist() == live
    expected = _snapshot_draft_fixture_fields(fields)

    # Reproduce the M4/K1/one-owner failure: padding in a different bucket may
    # change the unused final scratch, but never a commit/repair source.
    dead = [scratch[owner * stride + depth] for owner in range(owner_count)]
    hidden[dead] += 1000
    _compare_draft_fixture_fields(_snapshot_draft_fixture_fields(fields), expected)
    for slot in live:
        hidden[slot, 0] += 1
        with pytest.raises(ValueError, match="rounded boundary feedback"):
            _compare_draft_fixture_fields(_snapshot_draft_fixture_fields(fields), expected)
        hidden[slot, 0] -= 1


def test_draft_snapshot_buffer_schema_change_fails_closed():
    from cubic_serve.runtime.quantized_qualification import _snapshot_draft_fixture_fields

    fields = [("state", torch.ones(3, 4), torch.tensor([0, 1]))]
    with pytest.raises(ValueError, match="owner capacity"):
        _snapshot_draft_fixture_fields(fields, buffers={}, owner_count=3, owner_capacity=4)
    with pytest.raises(ValueError, match="schema changed"):
        _snapshot_draft_fixture_fields(fields, buffers={"state": torch.ones(2, 4).to(torch.int64)})


@pytest.mark.parametrize("failed", [False, True])
def test_draft_original_is_saved_once_per_bucket_and_released_after_restore(monkeypatch, failed):
    from cubic_serve.runtime import quantized_qualification as module

    rounds = {(2 * count, 1): SimpleNamespace(
        C=count, depth=1, stride=2, target=SimpleNamespace(_block_size=16),
        _resident_window=SimpleNamespace(requires_greedy=True),
        _candidate_window=SimpleNamespace(requires_greedy=False),
    ) for count in (1, 4)}
    source = torch.arange(32, dtype=torch.int64).reshape(8, 4)
    before = source.clone()
    retained, calls = [], []

    def qualify(capacity, round_, window, providers, targets, hidden, admission, order, **kwargs):
        cache = kwargs["original_cache"]
        assert "snapshot_buffers" not in kwargs
        if not cache:
            assert torch.equal(source, before)
            original = module._DraftBucketSnapshot(
                [("state", source, torch.arange(round_.C))],
                byte_budget=kwargs["witness_byte_budget"])
            cache["receipt"] = original
            retained.append(original)
        source[:round_.C].add_(1)
        calls.append((capacity, order))
        if failed and len(calls) == 4:
            raise RuntimeError("fixture interrupted")
        return capacity, order, "unchanged-fixture"

    monkeypatch.setattr(module, "_qualify_draft_window_case", qualify)
    if failed:
        with pytest.raises(RuntimeError, match="fixture interrupted"):
            _qualify_mock_rounds(module, rounds)
    else:
        _qualify_mock_rounds(module, rounds)
        assert len(calls) == 3 * sum(len(module._draft_window_audit_owner_maps(count))
                                    for count in (1, 4))
        assert [image.total_bytes for image in retained] == [32, 128]
        assert all(round_._resident_window.qualified and round_._candidate_window.qualified
                   for round_ in rounds.values())
    assert torch.equal(source, before)
    assert retained and all(image.closed and image.memory is None for image in retained)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_draft_snapshot_pool_roundtrips_cuda_noncontiguous_state_and_duplicate_writes():
    from cubic_serve.executor.runner import configure_reference_math
    from cubic_serve.runtime.quantized_qualification import (
        _restore_draft_fixture_fields,
        _snapshot_draft_fixture_fields,
    )

    configure_reference_math()
    raw = torch.arange(8 * 12, dtype=torch.int64).reshape(8, 3, 4)
    source = raw.to("cuda").transpose(1, 2)
    order = (0, 1, 4, 5, 4, 7)
    indices = torch.tensor(order, device="cuda")
    values = _snapshot_draft_fixture_fields([("state", source, indices)], buffers={})
    assert torch.equal(values["state"], raw.transpose(1, 2).index_select(0, indices.cpu()))
    updates = torch.arange(len(order) * 12).reshape(len(order), 4, 3) + 1000
    expected = raw.transpose(1, 2).clone()
    for row, index in enumerate(order):
        expected[index].copy_(updates[row])
    _restore_draft_fixture_fields([("state", source, indices)], {"state": updates})
    assert torch.equal(source.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_draft_snapshot_pinned_roles_keep_every_storage_pattern_and_release(dtype):
    from cubic_serve.executor.runner import configure_reference_math
    from cubic_serve.runtime.quantized_qualification import (
        _restore_draft_fixture_fields,
        _snapshot_draft_fixture_fields,
    )

    configure_reference_math()
    torch.cuda.synchronize()
    torch.accelerator.empty_host_cache()
    before = torch.cuda.memory.host_memory_stats()
    raw = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(256, 256)
    source = raw.view(dtype).to("cuda")
    indices = torch.arange(255, -1, -1, device="cuda")
    pools, snapshots = tuple({} for _ in range(4)), []
    try:
        for pool in pools:
            snapshots.append(_snapshot_draft_fixture_fields(
                [("state", source, indices)], buffers=pool,
                owner_count=256, owner_capacity=512))
            assert snapshots[-1]["state"].is_pinned()
            assert torch.equal(snapshots[-1]["state"].view(torch.int16), raw.flip(0))
        assert len({snapshot["state"].data_ptr() for snapshot in snapshots}) == 4
        source.zero_()
        _restore_draft_fixture_fields([("state", source, indices)], snapshots[2])
        assert torch.equal(source.cpu().view(torch.int16), raw)
    finally:
        for snapshot in snapshots:
            snapshot.clear()
        for pool in pools:
            pool.clear()
        torch.cuda.synchronize()
        torch.accelerator.empty_host_cache()
    after = torch.cuda.memory.host_memory_stats()
    assert after["active_bytes.current"] == before["active_bytes.current"]
    assert after["allocated_bytes.current"] == before["allocated_bytes.current"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("failed", [False, True])
def test_cuda_bounded_original_releases_staging_and_restores_on_failure(monkeypatch, failed):
    from cubic_serve.runtime import quantized_qualification as module

    before = torch.cuda.memory.host_memory_stats()
    source = torch.arange(64, device="cuda").reshape(16, 4)
    expected = source.cpu()
    rounds = {(2, 1): SimpleNamespace(C=1, depth=1, stride=2,
        target=SimpleNamespace(_block_size=16),
        _resident_window=SimpleNamespace(requires_greedy=True))}
    images = []

    def qualify(*args, **kwargs):
        cache = kwargs["original_cache"]
        if not cache:
            cache["receipt"] = module._DraftBucketSnapshot(
                [("state", source, torch.arange(16, device="cuda"))], byte_budget=4096)
            images.append(cache["receipt"])
        source.add_(1)
        if failed:
            raise RuntimeError("fixture interrupted")
        return "complete"

    monkeypatch.setattr(module, "_qualify_draft_window_case", qualify)
    if failed:
        with pytest.raises(RuntimeError, match="fixture interrupted"):
            _qualify_mock_rounds(module, rounds)
    else:
        _qualify_mock_rounds(module, rounds)
    assert torch.equal(source.cpu(), expected)
    assert all(image.closed for image in images)
    after = torch.cuda.memory.host_memory_stats()
    assert after["active_bytes.current"] == before.get("active_bytes.current", 0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_draft_byte_comparison_keeps_every_floating_storage_pattern(dtype):
    from cubic_serve.runtime.quantized_qualification import _equal_draft_fixture_bytes

    raw = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    left = raw.view(dtype).view(torch.uint8)
    right = left.clone()
    assert _equal_draft_fixture_bytes(left, right)
    for position in (0, 1, 65535, left.numel() - 1):
        right[position] ^= 1
        assert not _equal_draft_fixture_bytes(left, right)
        right[position] ^= 1
    assert _equal_draft_fixture_bytes(left, right)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_draft_parallel_field_receipts_preserve_all_storage_bits_and_error_order(workers):
    from concurrent.futures import ThreadPoolExecutor

    from cubic_serve.runtime.quantized_qualification import _compare_draft_fixture_fields

    raw = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    expected = {"bf16": raw.view(torch.bfloat16), "fp16": raw.view(torch.float16),
                "integer": torch.arange(17, dtype=torch.int64)}
    actual = {name: value.clone() for name, value in expected.items()}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        _compare_draft_fixture_fields(actual, expected, pool=pool)
        actual["bf16"].view(torch.int16)[100] ^= 1
        actual["fp16"].view(torch.int16)[101] ^= 1
        with pytest.raises(ValueError, match="rounded boundary bf16$"):
            _compare_draft_fixture_fields(actual, expected, pool=pool)
        with pytest.raises(ValueError, match="rounded boundary bf16$"):
            _compare_draft_fixture_fields(actual, expected)


def test_parallel_draft_comparison_finishes_all_reads_before_first_failure(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from cubic_serve.runtime import quantized_qualification as module

    second_finished = threading.Event()
    done = set()
    expected = {name: torch.tensor([marker], dtype=torch.uint8)
                for marker, name in enumerate(("slow_first", "fast_second", "last"))}
    actual = {name: value.clone() for name, value in expected.items()}

    def same(left, right):
        marker = int(left[0])
        if marker == 0:
            assert second_finished.wait(2)
        elif marker == 1:
            second_finished.set()
        done.add(marker)
        return marker == 2

    monkeypatch.setattr(module, "_equal_draft_fixture_bytes", same)
    with ThreadPoolExecutor(max_workers=2) as pool:
        with pytest.raises(ValueError, match="rounded boundary slow_first$"):
            module._compare_draft_fixture_fields(actual, expected, pool=pool)
        assert done == {0, 1, 2}


def test_bounded_original_restore_failure_still_closes_backing_and_cannot_qualify(monkeypatch):
    from cubic_serve.runtime import quantized_qualification as module

    window = SimpleNamespace(requires_greedy=True)
    rounds = {(2, 1): SimpleNamespace(C=1, depth=1, stride=2,
        target=SimpleNamespace(_block_size=16), _resident_window=window)}
    closed = []
    def restore():
        raise OSError("restore failed")
    def qualify(*args, **kwargs):
        kwargs["original_cache"]["receipt"] = SimpleNamespace(
            restore=restore, close=lambda: closed.append(True))
        return "complete"
    monkeypatch.setattr(module, "_qualify_draft_window_case", qualify)
    with pytest.raises(OSError, match="restore failed"):
        _qualify_mock_rounds(module, rounds)
    assert closed == [True]
    assert not getattr(window, "qualified", False)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
def test_original_fixture_receipt_refreshes_unselected_padding_and_exact_storage(device, dtype):
    import os

    if device == "cuda" and os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1":
        pytest.skip("explicit CUDA audit required")
    from cubic_serve.runtime.quantized_qualification import (
        _DraftOriginalSnapshot,
        _snapshot_draft_fixture_fields,
    )

    tensor = torch.empty((20, 4), dtype=dtype, device=device)
    raw = tensor.view(torch.uint8).reshape(20, -1)
    pattern = torch.arange(raw.numel(), device=device).remainder(256).to(torch.uint8)
    raw.copy_(pattern.reshape_as(raw))
    if dtype != torch.int64:
        tensor[0].view(torch.int16).copy_(torch.tensor(
            [0, -32768, 0x7F81, 0x7FFF], dtype=torch.int16, device=device))
    full = [("state", tensor, torch.arange(20, device=device))]
    receipt = _DraftOriginalSnapshot(full, buffers={}, owner_count=20, owner_capacity=20)
    sequences = ([0], [0, 1], [1, 0], [0, 1, 2, 3, 4], [4, 3, 2, 1, 0],
                 [0], list(range(10)), [-1, -2, -1], list(range(20)))
    for step, rows in enumerate(sequences):
        fields = [("state", tensor, torch.tensor(rows, device=device))]
        expected = _snapshot_draft_fixture_fields(fields)
        receipt.prepare(fields)
        # A graph may change private padding outside the current logical rows.
        # It must remain untouched by restore and become a fresh original later.
        raw.fill_(step + 17)
        protected = raw.clone()
        receipt.restore(fields)
        actual = _snapshot_draft_fixture_fields(fields)
        assert torch.equal(actual["state"].view(torch.uint8), expected["state"].view(torch.uint8))
        for row in set(range(20)) - {value % 20 for value in rows}:
            assert torch.equal(raw[row], protected[row])
        assert not receipt.prepared


def test_fixture_span_reuse_preserves_index_alias_writes_and_next_call_changes():
    from cubic_serve.runtime.quantized_qualification import (
        _restore_draft_fixture_fields,
        _snapshot_draft_fixture_fields,
    )

    indices = torch.tensor([0, 1])
    first, second = torch.tensor([1, 0]), torch.tensor([11, 22])
    fields = [("first", first, indices), ("second", second, indices)]
    buffers = {"first": indices, "second": torch.empty(2, dtype=torch.int64)}
    values = _snapshot_draft_fixture_fields(fields, buffers=buffers)
    assert values["first"].tolist() == [1, 0]
    assert values["second"].tolist() == [22, 11]
    indices.copy_(torch.tensor([0, 1]))
    target = torch.zeros(2, dtype=torch.int64)
    _restore_draft_fixture_fields(
        [("indices", indices, torch.tensor([0, 1])), ("target", target, indices)],
        {"indices": torch.tensor([1, 0]), "target": torch.tensor([7, 8])})
    assert target.tolist() == [8, 7]
    indices.copy_(torch.tensor([0, 1]))
    snapshot = _snapshot_draft_fixture_fields([("target", target, indices)])
    assert snapshot["target"].tolist() == [8, 7]


def test_streaming_fixture_comparison_waits_for_completed_fields_and_error_reads():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Timer

    from cubic_serve.runtime.quantized_qualification import (
        _compare_draft_fixture_fields,
        _equal_draft_fixture_bytes,
        _snapshot_draft_fixture_fields,
    )

    expected = {"first": torch.arange(17, dtype=torch.uint8),
                "second": torch.arange(23, dtype=torch.uint8)}
    fields = [(name, value.clone(), torch.arange(value.numel()))
              for name, value in expected.items()]
    first_started, second_ready = Event(), Event()
    checks = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        def compare(name, value):
            if name == "first":
                first_started.set()
                assert second_ready.wait(2)
            return _equal_draft_fixture_bytes(value, expected[name])

        def ready(name, value):
            assert torch.equal(value, expected[name]), "field must be complete before its consumer"
            if name == "second":
                assert first_started.wait(2)
                second_ready.set()
            checks.append((name, pool.submit(compare, name, value)))

        actual = _snapshot_draft_fixture_fields(fields, buffers={}, field_ready=ready)
        _compare_draft_fixture_fields(actual, expected, pool=pool, receipts=checks)
        assert all(future.done() for _, future in checks)
        for value in actual.values():
            value.fill_(255)
        release, other_finished = Event(), Event()

        def failed():
            raise RuntimeError("injected comparison failure")

        def pending():
            assert release.wait(2)
            other_finished.set()
            return True

        timer = Timer(.02, release.set)
        timer.start()
        try:
            with pytest.raises(RuntimeError, match="injected comparison failure"):
                _compare_draft_fixture_fields({}, {}, pool=pool, receipts=[
                    ("first", pool.submit(failed)), ("second", pool.submit(pending))])
            assert other_finished.is_set(), "failure must not release storage with active readers"
        finally:
            release.set()
            timer.join()


def test_streaming_fixture_comparison_rejects_writable_field_aliases():
    from cubic_serve.runtime.quantized_qualification import _snapshot_draft_fixture_fields

    shared = torch.empty(2, dtype=torch.int64)
    fields = [(name, torch.ones(2, dtype=torch.int64), torch.arange(2))
              for name in ("first", "second")]
    with pytest.raises(ValueError, match="share writable field storage"):
        _snapshot_draft_fixture_fields(fields, buffers={name: shared for name, _, _ in fields},
                                      field_ready=lambda *_: None)


def test_owner_migration_binds_feedback_after_round_capture_poison(monkeypatch):
    from cubic_serve.runtime.text_qualification import audit_owner_migration

    arena = _stage().arena
    arena.layout = SimpleNamespace(block_size_tokens=4)
    arena.spec = SimpleNamespace(state_slot_capacity=3, page_capacity=3)
    observations = []
    executors = {}
    for capacity in (1, 2):
        stage = _stage()
        stage.captured_m, stage.arena = capacity, arena
        stage.config = object()
        stage.buffers.target_hidden = torch.full((capacity, 8), float("nan"), dtype=torch.bfloat16)
        stage.synchronize = lambda: None

        class Executor:
            def __init__(self, body):
                self.body = body

            def replay_batch(self, rows):
                row, body = rows[0], self.body
                feedback = body.buffers.target_hidden[0]
                observations.append((body.captured_m, row.position, feedback.clone()))
                value = feedback.float().sum()
                body.hidden[0].copy_(feedback)
                body.logits[0].fill_(value)
                body.buffers.page_destination[0] = row.page_destination
                body.buffers.state_slot_maps[0][1, 0, 0] = row.state_destination_slot
                for tensor in (arena.key, arena.value, arena.written, *arena.page_fields.values()):
                    tensor[row.page_destination].fill_(value)
                for tensor in arena.states.values():
                    tensor[row.state_destination_slot].fill_(row.position)

        executors[capacity] = Executor(stage)
    monkeypatch.setattr("torch.distributed.barrier", lambda: None)
    monkeypatch.setattr(
        "cubic_serve.models.builtin.builtin_text_config",
        lambda _: SimpleNamespace(vocabulary_size=64),
    )
    audit_owner_migration(executors, max_sequence_tokens=2)
    assert len(observations) == 6
    for _, position, feedback in observations:
        assert torch.isfinite(feedback).all()
        assert torch.equal(feedback, observations[position][2])
    assert torch.isnan(executors[2].body.buffers.target_hidden[1]).all()


def test_conditioning_corpus_tracks_logical_positions_across_m_and_row_permutations():
    from cubic_serve.runtime.text_qualification import prepare_conditioning_inputs
    wide, narrow = _stage(), _stage()
    wide.captured_m, narrow.captured_m = 4, 1
    wide.buffers.target_hidden = torch.full((4, 8), float('nan'), dtype=torch.bfloat16)
    narrow.buffers.target_hidden = torch.full((1, 8), float('nan'), dtype=torch.bfloat16)
    prepare_conditioning_inputs(wide, 2, positions=(2, 8))
    before = wide.buffers.target_hidden[:2].clone()
    for physical, position in enumerate((2, 8)):
        prepare_conditioning_inputs(narrow, 1, position=position)
        assert torch.equal(narrow.buffers.target_hidden[0], before[physical])
    prepare_conditioning_inputs(wide, 2, positions=(8, 2))
    assert torch.equal(wide.buffers.target_hidden[:2], before.flip(0))
    assert torch.isnan(wide.buffers.target_hidden[2:]).all()

def _seal_distributed(monkeypatch):
    monkeypatch.setattr(_seal_audit.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(
        _seal_audit.dist, "all_gather_object", lambda out, value, group: out.__setitem__(0, value)
    )
    monkeypatch.setattr(_seal_audit.dist, "barrier", lambda group: None)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("corrupt_closed", [False, True])
def test_run_crosses_pages_and_protects_completed_cache(monkeypatch, reverse, corrupt_closed):
    _seal_distributed(monkeypatch)
    cap, logical, durable, pages, block = 4, 2, 4, 8, 2
    states = {"carry": torch.zeros(durable + cap, 1)}
    arena = SimpleNamespace(
        device="cpu",
        layout=SimpleNamespace(block_size_tokens=block),
        spec=SimpleNamespace(page_capacity=pages, state_slot_capacity=durable),
        states=states,
        state_specs={"carry": SimpleNamespace(initial_value=0)},
        key=torch.zeros(pages + cap, 1, block, 1),
        value=torch.zeros(pages + cap, 1, block, 1),
        written=torch.zeros(pages + cap, 1, block, dtype=torch.bool),
        page_fields={},
    )
    arena.reset_state_rows = lambda ids: states["carry"].index_fill_(0, ids, 0)
    arena.reset_page_rows = lambda ids: [
        v.index_fill_(0, ids, 0) for v in (arena.key, arena.value, arena.written)
    ]
    b = SimpleNamespace(
        token_ids=torch.zeros(cap, dtype=torch.int64),
        row_activity=torch.zeros(cap),
        segment_lengths=torch.zeros(cap),
        page_source=torch.zeros(cap, dtype=torch.int64),
        page_destination=torch.zeros(cap, dtype=torch.int64),
        page_table=torch.zeros(cap, 3, dtype=torch.int32),
        state_slot_maps=(torch.zeros(2, 1, cap, dtype=torch.int32),),
        position_ids=torch.zeros(3, cap, dtype=torch.int64),
        sequence_lengths=torch.zeros(cap, dtype=torch.int64),
        token_offsets=torch.zeros(cap, dtype=torch.int64),
    )

    def prepare():
        b.row_activity.zero_()
        b.segment_lengths.zero_()
        b.state_slot_maps[0][:] = torch.arange(durable, durable + cap)[None, None]
        b.page_destination[:] = torch.arange(pages, pages + cap)

    def replay():
        for row in torch.nonzero(b.row_activity).flatten().tolist():
            page, offset = int(b.page_destination[row]), int(b.token_offsets[row])
            position = int(b.position_ids[0, row])
            assert int(b.page_table[row, position // block]) == page
            arena.key[page, 0, offset, 0] = b.token_ids[row]
            arena.value[page, 0, offset, 0] = -b.token_ids[row]
            arena.written[page, 0, offset] = True
            owner = int(b.state_slot_maps[0][1, 0, row])
            arena.states["carry"][owner] = position + 1
            if corrupt_closed and position == 2:
                first = int(b.page_table[row, 0])
                arena.key[first, 0, 0, 0] += 1

    stage = SimpleNamespace(
        arena=arena,
        buffers=b,
        captured_m=cap,
        hidden=torch.zeros(cap, 1),
        prepare_scratch=prepare,
        fill_token_ids=lambda value: b.token_ids.copy_(value),
        replay=replay,
        synchronize=lambda: None,
        stream=SimpleNamespace(wait_stream=lambda _: None),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    monkeypatch.setattr(
        _seal_audit,
        "_replicated_witness_rows",
        lambda s, rows, *args: (tuple(range(len(rows))), tuple(range(len(rows)))),
    )
    monkeypatch.setattr(_seal_audit, "_host_events", lambda *args: [])
    monkeypatch.setattr(
        _seal_audit, "_boundary_witnesses", lambda stage, rows, **kwargs: [{} for _ in rows]
    )
    monkeypatch.setattr(_seal_audit, "_intermediate_witness", lambda *args, **kwargs: ())
    kwargs = dict(
        stage=stage,
        logical=logical,
        corpora=((17, 18, 19, 20, 21), (31, 32, 33, 34, 35)),
        oracle=[[{}] * 5, [{}] * 5],
        groups={"tp": None},
        reverse=reverse,
    )
    if corrupt_closed:
        with pytest.raises(RuntimeError, match="closed page mutation"):
            _seal_audit._run(**kwargs)
    else:
        actual = _seal_audit._run(**kwargs)
        assert len(actual) == 5
        assert arena.written[:6].sum().item() == 10
        assert not arena.written[6:pages].any()


def _fake_seal_cross_m(monkeypatch, dtype="int6", feedback=False):
    from cubic_serve.runtime import quantized_qualification as q
    from cubic_serve.runtime.cache.quantization import (
        CacheDType,
        CacheGroupAxis,
        CacheStorageSpec,
        KVCacheStorageSpec,
    )

    _seal_distributed(monkeypatch)
    # CPU control-flow stand-ins have no reusable CUDA observation storage.
    monkeypatch.setattr(q, "_draft_reference_headroom", lambda device: 0)
    code = CacheDType.BF16 if dtype == "bf16" else CacheDType(dtype)
    storage = KVCacheStorageSpec(
        CacheStorageSpec(code, CacheGroupAxis.TOKENS),
        CacheStorageSpec(code, CacheGroupAxis.CHANNELS),
    )
    arena = SimpleNamespace(
        device="cpu",
        layout=SimpleNamespace(block_size_tokens=32, storage=storage),
        spec=SimpleNamespace(state_slot_capacity=4, page_capacity=17),
        reset_state_rows=lambda ids: None,
        reset_page_rows=lambda ids: None,
    )

    def body(m):
        return SimpleNamespace(
            arena=arena,
            config=SimpleNamespace(
                text=SimpleNamespace(
                    qsa=SimpleNamespace(compression_ratio=4), vocabulary_size=200000
                )
            ),
            topology=SimpleNamespace(pp_size=1, ep_size=1),
            weights=SimpleNamespace(original={}) if feedback else {},
            operators=object(),
            captured_m=m,
            rank=0,
        )

    executors = {m: SimpleNamespace(body=body(m)) for m in (1, 2, 4)}
    monkeypatch.setattr(
        _seal_audit,
        "row_arithmetic_plan",
        lambda ops, m: {"projection": "base" if m == 1 else "large"},
    )

    class Scope:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(_seal_audit, "row_arithmetic_scope", lambda plan: Scope())
    monkeypatch.setattr(
        _seal_audit,
        "_temporary",
        lambda body, events, groups, **kwargs: SimpleNamespace(
            close=lambda: None, helper=SimpleNamespace(_buffers=())
        ),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    runs = []
    references = []

    def run(stage, logical, corpora, oracle, groups, **kwargs):
        runs.append(
            (
                logical,
                len(corpora),
                len(corpora[0]),
                kwargs.get("reverse", False),
                len(kwargs.get("feedback_rows", ())),
            )
        )
        assert all(len(row) >= len(corpora[0]) for row in oracle)
        offset = kwargs.get("offset", 0)
        return [
            [(oracle[(r + offset) % len(corpora)][p], ()) for r in range(logical)]
            for p in range(len(corpora[0]))
        ]

    monkeypatch.setattr(_seal_audit, "_run", run)
    monkeypatch.setattr(_seal_audit, "_run_segments", lambda *args, **kwargs: None)

    def original(body, admission, groups, **kwargs):
        tokens = kwargs["token_ids"]
        references.append((len(tokens), len(kwargs.get("feedback_rows", ()))))
        return dict(
            cross_step_error_propagation_qualified=True,
            steps=[{"boundary_digests": {}} for _ in tokens],
        )

    monkeypatch.setattr(_seal_audit, "audit_original", original)
    report = dict(
        steps=[{"boundary_digests": {}} for _ in range(33)],
        draft_method="mtp" if feedback else None,
        feedback_recurrence=False,
    )
    provider = (
        SimpleNamespace(spec=SimpleNamespace(method="mtp"), reference_columns=())
        if feedback
        else None
    )
    feedback_rows = (torch.zeros(1, 1, 3),) * 33 if feedback else ()
    return (
        executors,
        SimpleNamespace(max_sequence_tokens=4096,
                        parallel_plan=SimpleNamespace(bindings_for_rank=lambda rank: ())),
        report,
        provider,
        feedback_rows,
        runs,
        references,
    )


@pytest.mark.parametrize("feedback", [False, True])
def test_seal_reuses_completed_m1_and_extends_only_first_new_arithmetic_corpus(
    monkeypatch, feedback
):
    executors, admission, report, provider, rows, runs, references = _fake_seal_cross_m(
        monkeypatch, feedback=feedback
    )
    result = _seal_audit.audit_cross_m(
        executors, admission, {"tp": None}, report, draft_provider=provider, feedback_rows=rows
    )
    assert sorted(n for n, _ in references) == [4] * 6 + [33]
    assert all(n == f for n, f in references) if feedback else all(f == 0 for _, f in references)
    for cap in (1, 2, 4):
        cases = [r for r in result["cases"] if r["capacity"] == cap and "segment_depth" not in r]
        assert cases and all(r["column_seal_steps"] == 33 for r in cases)
    assert any(n == 33 and logical == 3 and reverse for logical, _, n, reverse, _ in runs)
    assert (
        all(f == 33 for _, _, n, _, f in runs if n == 33)
        if feedback
        else all(f == 0 for *_, f in runs)
    )


def test_attention_partition_changes_get_their_own_complete_cross_m_witness(monkeypatch):
    from cubic_serve.ops.cuda.attention import paged_decode

    executors, admission, report, _, _, _, _ = _fake_seal_cross_m(monkeypatch)
    monkeypatch.setattr(_seal_audit, "row_arithmetic_plan", lambda *args: {"same": "recipe"})
    for capacity, executor in executors.items():
        splits = 32 if capacity == 1 else 16
        executor.body.helper = SimpleNamespace(_buffers={
            ("qsa.attention_partial", (capacity, 12, splits, 256), torch.float32): None,
        })

    def normal_planner(*args, **kwargs):
        return 7

    monkeypatch.setattr(paged_decode, "paged_gqa_kv_splits", normal_planner)
    original, temporary = _seal_audit.audit_original, _seal_audit._temporary
    reference_bindings, anchor_bindings = [], []

    def audit(body, *args, **kwargs):
        reference_bindings.append((body.captured_m, paged_decode.paged_gqa_kv_splits(1)))
        return original(body, *args, **kwargs)

    def anchor(body, *args, **kwargs):
        anchor_bindings.append((body.captured_m, paged_decode.paged_gqa_kv_splits(1)))
        return temporary(body, *args, **kwargs)

    monkeypatch.setattr(_seal_audit, "audit_original", audit)
    monkeypatch.setattr(_seal_audit, "_temporary", anchor)
    result = _seal_audit.audit_cross_m(executors, admission, {"tp": None}, report)
    assert [cohort["source_capacity"] for cohort in result["arithmetic_cohorts"]] == [1, 2]
    assert any(capacity == 2 for capacity, _ in reference_bindings)
    assert all(splits == 16 for capacity, splits in reference_bindings if capacity == 2)
    assert (1, 32) in anchor_bindings and (1, 16) in anchor_bindings
    assert paged_decode.paged_gqa_kv_splits is normal_planner


def test_native_cache_does_not_add_long_reference_or_graph_cases(monkeypatch):
    executors, admission, report, provider, rows, runs, references = _fake_seal_cross_m(
        monkeypatch, dtype="bf16"
    )
    result = _seal_audit.audit_cross_m(executors, admission, {"tp": None}, report)
    # One short preparation cannot replace the complete four-step corpora.
    assert sum(n == 1 for n, _ in references) == 1
    assert all(n in (1, 4) for n, _ in references)
    assert all(n == 4 for _, _, n, _, _ in runs)
    assert all(
        case["column_seal_steps"] == 4 for case in result["cases"] if "segment_depth" not in case
    )


@pytest.mark.parametrize("deferred", [False, True])
def test_cross_m_keeps_identical_corpora_adjacent_across_every_bound_plan(monkeypatch, deferred):
    from collections import Counter
    from itertools import groupby

    executors, admission, report, _, _, _, references = _fake_seal_cross_m(
        monkeypatch, dtype="bf16")
    monkeypatch.setattr(_seal_audit, "row_arithmetic_plan",
                        lambda ops, m: {"projection": str(m)})
    original = _seal_audit.audit_original
    calls, captured = [], []

    def audited(body, *args, **kwargs):
        calls.append((kwargs["token_ids"], body.captured_m))
        if deferred:
            assert captured == [4, 2, 1]
        return original(body, *args, **kwargs)

    def capture(capacity):
        captured.append(capacity)
        return executors[capacity]

    monkeypatch.setattr(_seal_audit, "audit_original", audited)
    options = (dict(capacities=tuple(executors), capture_bucket=capture) if deferred else {})
    result = _seal_audit.audit_cross_m(executors, admission, {"tp": None}, report, **options)
    grouped = [tuple(items) for _, items in groupby(calls, key=lambda row: row[0])]
    assert len(grouped) == 5
    assert Counter(capacity for _, capacity in calls) == {1: 4, 2: 5, 4: 5}
    assert {capacity for _, capacity in grouped[0]} == {2, 4}
    assert all({capacity for _, capacity in group} == {1, 2, 4} for group in grouped[1:])
    assert len(references) == 14
    assert sum(length == 1 for length, _ in references) == 2
    assert all(length in (1, 4) for length, _ in references)
    assert {cohort["source_capacity"] for cohort in result["arithmetic_cohorts"]} == {1, 2, 4}
    assert all(len(cohort["original_weight_reports"]) == 4
               for cohort in result["arithmetic_cohorts"])


def test_truncated_independent_m1_is_rejected_before_graphs(monkeypatch):
    executors, admission, report, provider, rows, runs, references = _fake_seal_cross_m(monkeypatch)
    report["steps"] = report["steps"][:4]
    with pytest.raises(ValueError, match="column sealing"):
        _seal_audit.audit_cross_m(executors, admission, {"tp": None}, report)
    assert not runs and not references


@pytest.mark.parametrize('fault', [None, 'state', 'protected', 'seed'])
@pytest.mark.parametrize('budget', [False, True])
def test_identical_window_reference_reuses_byte_identities_and_rejects_changed_seed(monkeypatch, fault, budget):
    import numpy as np
    from cubic_serve.runtime import performance
    from cubic_serve.runtime import quantized_qualification as q

    state = torch.full((5, 1), 19, dtype=torch.int64)
    indices = torch.tensor([0, 2, 3])
    arena = SimpleNamespace(spec=SimpleNamespace(page_capacity=2), written=torch.zeros(2, 1))
    pool = SimpleNamespace(metadata=[None], tokens=torch.zeros(1, 1, dtype=torch.int64),
                           scores=torch.zeros(1, 1, 1), ids=torch.zeros(1, 1, 1, dtype=torch.int64))
    sampler = SimpleNamespace(invalidate_metadata=lambda: None)
    round_ = SimpleNamespace(C=1, depth=1, stride=2, pool=pool,
        target=SimpleNamespace(body=SimpleNamespace(arena=arena, hidden=torch.zeros(1)),
                               _block_size=16, _sampling=SimpleNamespace(constraint_bound=False)),
        align=SimpleNamespace(body=SimpleNamespace(arena=arena)),
        head=SimpleNamespace(_sampling=SimpleNamespace(constraint_bound=False)),
        host_result=torch.tensor([[0, 7]]), head_greedy=sampler, target_greedy=sampler)
    round_.budget = SimpleNamespace(bind=lambda values: None)
    fields = [('state', state, indices)]
    calls, priming = [], [1]
    def advance(amount=1):
        state.index_add_(0, indices, torch.full((3, 1), amount, dtype=state.dtype))
        value = int(state[0])
        pool.metadata[0] = ('a', value, 7, None, value, None, (8,))
        pool.tokens.fill_(value);pool.scores.fill_(value);pool.ids.fill_(7)
    def ordinary(command, *args, **kwargs):
        calls.append(command.batch_id)
        advance(priming[0] if command.batch_id == 0 else 1)
        return SimpleNamespace(draft_ingress=False, step_ms=3.)
    def initialize(*args, **kwargs):state.index_fill_(0, indices, 0)
    monkeypatch.setattr(q, '_draft_window_fixture_fields', lambda *a: (fields, indices[:1], indices, indices))
    monkeypatch.setattr(q, '_initialize_draft_window_fixture', initialize)
    monkeypatch.setattr(q, '_draft_window_audit_rows', lambda *a, **kw: ())
    monkeypatch.setattr(performance, '_execute_pipelined_draft_batch', ordinary)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: None)
    monkeypatch.setattr(q.dist, 'get_world_size', lambda: 1)
    monkeypatch.setattr(q.dist, 'all_gather_object', lambda out, value: out.__setitem__(0, value))
    active_fault = [None]
    def replay():
        advance();advance()
        if active_fault[0] == 'state':state[0] += 1
        if active_fault[0] == 'protected':arena.written[1] = 1
    window = SimpleNamespace(host_flags=np.zeros(3, dtype='int32'), bind=lambda **kw: None,
        replay=replay, consume=lambda: (None, None), requires_budget=budget, requires_greedy=True,
        receipt=lambda *a, **kw: SimpleNamespace(segment_results=(SimpleNamespace(output_token_ids=(7,)),)))
    cache = {}
    def run():
        return q._qualify_draft_window_case(2, round_, window, {1: None}, {2: round_.target}, None, None, (0,),
            rank=0, sampler_rank=0, reference_cache=cache, witness_byte_budget=4096)
    before=state.clone();assert run();assert calls == [0, 1, 2, 0]
    assert torch.equal(state, before)
    first = next(iter(cache.values()))
    seed, expected = first[0]['state'], first[6]['state']
    assert isinstance(seed, str) and seed.startswith("b2s256:")
    assert isinstance(expected, str) and expected.startswith("b2s256:")
    # The already-required ingress is the seed on a hit. Altered state must be
    # rejected before the candidate, never repaired with a full cached image.
    priming[0] = 77 if fault == 'seed' else 1
    active_fault[0] = fault
    calls.clear()
    if fault:
        with pytest.raises(ValueError, match='rounded boundary|protected page activity'):
            run()
    else:
        assert run()
        assert round_._startup_round_costs[(2, True)] == [3., 3.]
        if budget:
            assert len(round_._startup_constraint_round_costs[(2, True, 2)]) == 2
    assert torch.equal(state, before)
    assert calls == [0]
    assert first[0]['state'] == seed and first[6]['state'] == expected


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32,
                                   torch.int64, torch.uint8, torch.bool])
@pytest.mark.parametrize("budget", [0, 64, 4096])
def test_owned_draft_reference_retains_every_storage_bit(device, dtype, budget):
    from cubic_serve.guardrail.compare import tensor_digest
    from cubic_serve.runtime.quantized_qualification import _DraftFixtureReference

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    base = torch.arange(9 * 34, device=device).reshape(9, 34).to(dtype)
    value = base[:, ::2]
    if value.is_floating_point():
        value[0, 0] = -0.
        value[2, 2] = float("nan")
    indices = torch.tensor([0, 2, 3, 7, 7], device=device)
    fields = (("state", value, indices), ("tail", value[:, 1:2], indices))
    with torch.inference_mode():
        reference = _DraftFixtureReference(fields, byte_budget=1024, device_byte_budget=budget)
    assert reference.bytes <= budget
    assert reference.identities == {
        name: tensor_digest(tensor.index_select(0, rows).cpu())
        for name, tensor, rows in fields
    }
    reference.compare(fields)
    original = value[0, 0].clone()
    # Includes +0 versus -0; floating equality is insufficient here.
    value[0, 0] = (0. if value.is_floating_point() else
                   not bool(original) if dtype == torch.bool else int(original) + 1)
    with pytest.raises(ValueError, match="rounded boundary"):
        reference.compare(fields)
    value[0, 0].copy_(original)
    reference.compare(fields)
    changed_indices = indices.clone()
    changed_indices[0] = 1
    with pytest.raises(ValueError, match="binding"):
        reference.compare((("state", value, changed_indices), fields[1]))
    with pytest.raises(ValueError, match="schema"):
        reference.compare(fields[:1])
    if reference.images:
        images = next(iter(reference.images.values()))
        images[0][0].view(torch.uint8).zero_()
        with pytest.raises(ValueError, match="immutable"):
            reference.compare(fields)


def test_mtp_composition_progress_is_useful_only_when_completed_count_advances():
    from cubic_serve.runtime.performance import (
        _LoadProgress, _StartupProgress, Qwen3_5TP2PP2ServingError)
    progress = _StartupProgress(2)
    assert not progress.update(_LoadProgress(0, 'mtp_composition', 0, 4, 'witnesses'))
    assert progress.update(_LoadProgress(0, 'mtp_composition', 1, 4, 'witnesses'))
    assert not progress.update(_LoadProgress(0, 'mtp_composition', 1, 4, 'witnesses'))
    with pytest.raises(Qwen3_5TP2PP2ServingError):
        progress.update(_LoadProgress(0, 'mtp_composition', 0, 4, 'witnesses'))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32,
                                   torch.int64, torch.uint8, torch.bool])
def test_zero_storage_proof_covers_strided_bits_with_bounded_chunks(monkeypatch, device, dtype):
    from cubic_serve.runtime import quantized_qualification as q

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    base = torch.zeros((11, 8, 18), dtype=dtype, device=device)
    # Unselected storage is outside this view's proof and must not be read.
    base[:, :, 1::2] = 1
    value = base[:, :, ::2].transpose(1, 2)
    original = q._draft_tensor_chunks
    sizes = []

    def bounded(tensor, limit):
        for chunk in original(tensor, limit):
            sizes.append(chunk.numel() * chunk.element_size())
            assert sizes[-1] <= limit
            yield chunk

    monkeypatch.setattr(q, "_draft_tensor_chunks", bounded)
    assert q._storage_is_zero(value, 64)
    assert len(sizes) > 1 and max(sizes) <= 64
    assert q._storage_is_zero(value[:0], 64)
    changes = ((-0., float("nan"), 1.) if value.is_floating_point()
               else ((1 << 54) + 1,) if dtype == torch.int64 else (1,))
    for changed in changes:
        value[-1, -1, -1] = changed
        assert not q._storage_is_zero(value, 64)
        value[-1, -1, -1] = 0
    assert q._storage_is_zero(value, 64)


def test_device_reference_admission_preserves_comparison_headroom(monkeypatch):
    from cubic_serve.runtime import quantized_qualification as q
    from cubic_serve.runtime.quantized_qualification import _draft_reference_device_budget

    field = SimpleNamespace(device=torch.device("cuda:0"))
    for free in (0, 1024, 1 << 30, 40 << 30):
        monkeypatch.setattr(q, "_draft_reference_headroom", lambda device, free=free: free)
        budget = _draft_reference_device_budget((("x", field, None),), 64 << 20)
        assert budget >= 0
        if budget:
            assert 2 * budget + (512 << 20) <= free
        if free < 512 << 20:
            assert budget == 0


def test_device_reference_headroom_obeys_driver_and_allocator_limits(monkeypatch):
    from cubic_serve.runtime.quantized_qualification import _draft_reference_headroom

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (12 << 30, 80 << 30))
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda device: .5)
    for reserved, expected in ((20, 12), (35, 5), (40, 0), (41, 0)):
        monkeypatch.setattr(torch.cuda, "memory_reserved",
                            lambda device, reserved=reserved: reserved << 30)
        assert _draft_reference_headroom(torch.device("cuda:0")) == expected << 30


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_device_reference_chunk_views_share_one_owned_field_allocation():
    from cubic_serve.runtime.quantized_qualification import _DraftFixtureReference

    value = torch.arange(8192, device="cuda", dtype=torch.int64).reshape(64, 128)
    fields = (("state", value, torch.arange(64, device="cuda")),)
    reference = _DraftFixtureReference(fields, byte_budget=4096, device_byte_budget=1 << 20)
    chunks = reference.images["state"]
    assert len(chunks) > 1
    assert len({image.untyped_storage().data_ptr() for image, _ in chunks}) == 1
    assert chunks[0][0].untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
    reference.compare(fields)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_exact_device_references_do_not_copy_or_hash_unchanged_images(monkeypatch):
    from cubic_serve.guardrail.compare import tensor_digest
    from cubic_serve.runtime import quantized_qualification as q

    value = torch.arange(99, dtype=torch.int32, device="cuda").reshape(9, 11)
    indices = torch.tensor([0, 2, 3, 7, 7], device="cuda")
    fields = (("state", value, indices),)
    wanted = tensor_digest(value.index_select(0, indices).cpu())
    copies = []
    original = q._draft_chunk_bytes
    monkeypatch.setattr(q, "_draft_chunk_bytes", lambda chunk: (
        copies.append(chunk.numel()), original(chunk))[1])
    reference = q._DraftFixtureReference(fields, byte_budget=1024, device_byte_budget=4096)
    reference.compare(fields)
    assert not copies
    value[0, 0] = -99
    # The optional canonical identity reads the frozen image, never live state.
    assert reference["state"] == wanted
    assert copies
    with pytest.raises(ValueError, match="rounded boundary"):
        reference.compare(fields)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype,word_dtype,pattern", [
    (torch.bfloat16, torch.int16, 0x7FC1),
    (torch.float16, torch.int16, 0x7E01),
    (torch.float32, torch.int32, 0x7FC00001),
])
def test_device_reference_rejects_a_changed_nan_payload(dtype, word_dtype, pattern):
    from cubic_serve.runtime.quantized_qualification import _DraftFixtureReference

    words = torch.tensor([[pattern, pattern, pattern]], dtype=word_dtype, device="cuda")
    value = words.view(dtype)
    fields = (("nan-tail", value, torch.tensor([0], device="cuda")),)
    reference = _DraftFixtureReference(fields, byte_budget=1024, device_byte_budget=4096)
    reference.compare(fields)
    words[0, -1].bitwise_xor_(1)
    assert bool(torch.isnan(value).all())
    with pytest.raises(ValueError, match="rounded boundary"):
        reference.compare(fields)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
@pytest.mark.parametrize("order", [(0, 1, 2, 3), (3, 0, 2, 0, -1), ()])
def test_streamed_fixture_identity_matches_independent_canonical_bytes(dtype, order):
    from cubic_serve.guardrail.compare import tensor_digest
    from cubic_serve.runtime.quantized_qualification import (
        _draft_fixture_identities, _compare_draft_fixture_identities,
    )

    raw = torch.arange(4 * 64 * 16, dtype=torch.int64).reshape(4, 64, 16)
    tensor = raw if dtype == torch.int64 else raw.to(torch.int16).view(dtype)
    tensor = tensor.transpose(1, 2)
    indices = torch.tensor(order, dtype=torch.int64)
    fields = [("state", tensor, indices)]
    selected = torch.stack([tensor[row] for row in order]) if order else tensor[:0]
    actual = _draft_fixture_identities(fields, 1024)
    expected = {"state": tensor_digest(selected)}
    assert actual == expected
    _compare_draft_fixture_identities(actual, expected)
    if order:
        tensor[order[0], 0, 0].reshape(1).view(torch.uint8)[0] ^= 1
        with pytest.raises(ValueError, match="rounded boundary state"):
            _compare_draft_fixture_identities(_draft_fixture_identities(fields, 1024), expected)


@pytest.mark.parametrize("backing", ["memory", "disk"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
def test_bounded_original_restores_every_bit_noncontiguous_and_duplicate_rows(tmp_path, backing, dtype):
    from cubic_serve.runtime.quantized_qualification import _DraftBucketSnapshot

    raw = torch.arange(65536, dtype=torch.int64).reshape(256, 256)
    source = raw.clone() if dtype == torch.int64 else raw.to(torch.int16).view(dtype)
    source = source.reshape(256, 16, 16).transpose(1, 2)
    before = source.clone()
    rows = [*range(256), 1, -1, 1]
    indices = torch.tensor(rows)
    budget = 8 << 20 if backing == "memory" else 4096
    image = _DraftBucketSnapshot([("state", source, indices)], byte_budget=budget, directory=tmp_path)
    descriptor = None if image.file is None else image.file.fileno()
    try:
        assert (image.memory is not None) == (backing == "memory")
        assert image.chunk_bytes <= budget // 8
        source.view(torch.int64 if dtype == torch.int64 else torch.int16).fill_(37)
        image.restore()
        assert torch.equal(source.contiguous().view(torch.uint8), before.contiguous().view(torch.uint8))
    finally:
        image.close()
    assert image.closed and image.memory is None
    if descriptor is not None:
        import os
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert not tuple(tmp_path.iterdir())


def test_bounded_original_short_read_fails_without_leaking_storage(tmp_path):
    from cubic_serve.runtime.quantized_qualification import _DraftBucketSnapshot

    source = torch.arange(4096, dtype=torch.int64).reshape(64, 64)
    image = _DraftBucketSnapshot([("state", source, torch.arange(64))],
                                 byte_budget=4096, directory=tmp_path)
    try:
        image.file.truncate(1)
        with pytest.raises(OSError, match="short read"):
            image.restore()
    finally:
        image.close()
    assert image.closed and not tuple(tmp_path.iterdir())


def test_witness_budget_observes_host_cgroup_and_rank_headroom(monkeypatch):
    from pathlib import Path
    from cubic_serve.runtime.quantized_qualification import _draft_witness_budget

    available = [256 << 20]
    def read(path):
        return {"/proc/meminfo": "MemAvailable: 8388608 kB\n",
                "/proc/self/cgroup": "0::/consumer\n",
                "/sys/fs/cgroup/memory.max": "max",
                "/sys/fs/cgroup/consumer/memory.max": str(512 << 20),
                "/sys/fs/cgroup/consumer/memory.current": str((512 << 20) - available[0])}[str(path)]
    monkeypatch.setattr(Path, "read_text", read)
    assert _draft_witness_budget(1) == 32 << 20
    assert _draft_witness_budget(4) == 8 << 20
    available[0] = 1 << 10
    with pytest.raises(MemoryError, match="host headroom"):
        _draft_witness_budget(1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_streamed_graph_witness_preserves_all_storage_patterns(tmp_path, dtype):
    from cubic_serve.guardrail.compare import tensor_digest
    from cubic_serve.runtime.quantized_qualification import (
        _DraftBucketSnapshot, _draft_fixture_identities,
    )

    words = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(256, 256)
    source = words.view(dtype).to("cuda")
    indices = torch.arange(256, device="cuda")
    fields = [("state", source, indices)]
    expected = {"state": tensor_digest(words.view(dtype))}
    before = torch.cuda.memory.host_memory_stats()["active_bytes.current"]
    image = _DraftBucketSnapshot(fields, byte_budget=4096, directory=tmp_path)
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            source.view(torch.int16).bitwise_xor_(1)
        # Capture itself runs the body once. Restore the exact initial image,
        # then compare one real graph replay to independent CPU integer bytes.
        image.restore()
        graph.replay()
        actual = _draft_fixture_identities(fields, 4096)
        reference = {"state": tensor_digest(words.bitwise_xor(1).view(dtype))}
        assert actual == reference and actual != expected
        image.restore()
        assert _draft_fixture_identities(fields, 4096) == expected
    finally:
        image.close()
    assert image.closed and not tuple(tmp_path.iterdir())
    assert torch.cuda.memory.host_memory_stats()["active_bytes.current"] == before


def test_bounded_original_freezes_row_identity_and_detects_disk_bit_corruption(tmp_path):
    from cubic_serve.runtime.quantized_qualification import _DraftBucketSnapshot

    source = torch.arange(4096, dtype=torch.int64).reshape(64, 64)
    before = source.clone()
    indices = torch.tensor([0, 63, 0, 63, 0])
    image = _DraftBucketSnapshot([("state", source, indices)], byte_budget=4096, directory=tmp_path)
    try:
        indices.fill_(1)
        source[0].zero_()
        source[63].zero_()
        image.restore()
        assert torch.equal(source, before)
        image.file.seek(0)
        image.file.write(b"\xff")
        with pytest.raises(ValueError, match="original state storage identity"):
            image.restore()
    finally:
        image.close()


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_streamed_fields_keep_canonical_identity_and_bounded_chunks(monkeypatch, workers):
    from cubic_serve.guardrail.compare import tensor_digest
    from cubic_serve.runtime import quantized_qualification as q

    monkeypatch.setattr(q.os, "sched_getaffinity", lambda _: set(range(workers)))
    fields, expected = [], {}
    for index, dtype in enumerate((torch.float16, torch.bfloat16, torch.int64, torch.float32)):
        source = torch.arange(16 * 32 * 8, dtype=torch.int64).reshape(16, 32, 8)
        if dtype != torch.int64:
            source = source.to(torch.int16 if torch.empty((), dtype=dtype).element_size() == 2 else torch.int32).view(dtype)
        source = source.transpose(1, 2)
        indices = torch.tensor([0, 1, 2, 15, 1])
        name = f"field{index}"
        fields.append((name, source, indices))
        expected[name] = tensor_digest(source.index_select(0, indices))
    sizes = []
    copy = q._draft_chunk_bytes
    def copied(chunk):
        sizes.append(chunk.numel() * chunk.element_size())
        return copy(chunk)
    monkeypatch.setattr(q, "_draft_chunk_bytes", copied)
    assert q._draft_fixture_identities(fields, 4096) == expected
    assert max(sizes) <= 4096 // (8 * workers)
    with pytest.raises(ValueError, match="duplicate draft witness"):
        q._draft_fixture_identities((fields[0], fields[0]), 4096)


@pytest.mark.parametrize("budget", [8192, 1 << 20])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_original_zero_runs_restore_storage_bits_without_copying_zero_chunks(
    monkeypatch, tmp_path, budget, dtype, device,
):
    from cubic_serve.runtime import quantized_qualification as q

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    words = torch.zeros((7, 1024), dtype=torch.int64 if dtype == torch.int64 else torch.int16)
    words[2, 0] = -(1 << 63) if dtype == torch.int64 else -32768
    words[4, -1] = (1 << 62) if dtype == torch.int64 else 0x7FC1
    source = words if dtype == torch.int64 else words.view(dtype)
    source = source.reshape(7, 32, 32).transpose(1, 2)
    before = source.clone()
    source = source.to(device)
    copied = []
    original = q._draft_chunk_bytes

    def copy(chunk):
        assert bool(chunk.contiguous().view(torch.uint8).any())
        copied.append(chunk.numel() * chunk.element_size())
        return original(chunk)

    monkeypatch.setattr(q, "_draft_chunk_bytes", copy)
    image = q._DraftBucketSnapshot(
        [("state", source, torch.tensor([*range(7), 0, -1]))],
        byte_budget=budget, directory=tmp_path,
    )
    try:
        assert image.zero_bytes > 0 and copied
        assert sum(copied) + image.zero_bytes == image.total_bytes
        source.fill_(37)
        image.restore()
        assert torch.equal(source.cpu().contiguous().view(torch.uint8),
                           before.contiguous().view(torch.uint8))
    finally:
        image.close()


def test_original_zero_run_metadata_is_bounded_and_tampering_fails_closed(tmp_path):
    from cubic_serve.runtime.quantized_qualification import _DraftBucketSnapshot

    source = torch.zeros(32, 128, dtype=torch.int64)
    source[1::2] = 17
    before = source.clone()
    image = _DraftBucketSnapshot([("state", source, torch.arange(32))],
                                 byte_budget=4096, directory=tmp_path)
    try:
        assert len(image.zero_runs) == 2  # Later zero chunks retain ordinary bytes.
        source.fill_(31)
        image.restore()
        assert torch.equal(source, before)
        image.zero_runs = []
        with pytest.raises(ValueError, match="original state storage identity"):
            image.restore()
    finally:
        image.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_zero_original_restore_completes_its_stream_before_returning(tmp_path):
    from cubic_serve.runtime.quantized_qualification import _DraftBucketSnapshot

    source = torch.zeros((8, 128), dtype=torch.int64, device="cuda")
    image = _DraftBucketSnapshot([("state", source, torch.arange(8))],
                                 byte_budget=1 << 20, directory=tmp_path)
    try:
        source.fill_(17)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(20000000)
            image.restore()
        # The independent default-stream read must not race a deferred restore.
        assert not source.cpu().view(torch.uint8).any()
    finally:
        image.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('workers', [2, 4])
@pytest.mark.parametrize('byte_budget', [4096, 1 << 20])
def test_mixed_field_sizes_keep_exact_bytes_input_order_and_readonly_ownership(
        monkeypatch, workers, byte_budget, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    from cubic_serve.guardrail.compare import tensor_digest
    from cubic_serve.runtime import quantized_qualification as q

    monkeypatch.setattr(q.os, 'sched_getaffinity', lambda _: set(range(workers)))
    fields, expected, original = [], {}, []
    for index, (width, dtype) in enumerate([
            (1, torch.bfloat16), (257, torch.int64), (9, torch.float32),
            (65, torch.float16), (3, torch.uint8)]):
        raw = torch.arange(8 * width * 3, dtype=torch.int64).reshape(8, width, 3)
        if dtype in (torch.bfloat16, torch.float16):
            raw = raw.to(torch.int16)
            raw[0, 0, 0] = 0x7FC1 if dtype == torch.bfloat16 else 0x7E01
            raw[1, 0, 0] = -32768
            source = raw.view(dtype)
        else:
            source = raw.to(dtype)
        source = source.transpose(1, 2).to(device)
        indices = torch.tensor([7, 0, 3, 3, 1, -1], device=device)
        name = f'field{index}'
        fields.append((name, source, indices))
        expected[name] = tensor_digest(torch.stack([source[i] for i in indices.tolist()]))
        before = source.contiguous().reshape(-1).view(torch.uint8).clone()
        original.append((before, indices.clone()))
    empty = torch.empty((0, 5), dtype=torch.float32, device=device)
    fields.append(('empty', empty, torch.empty(0, dtype=torch.int64, device=device)))
    expected['empty'] = tensor_digest(empty)

    actual = q._draft_fixture_identities(fields, byte_budget)
    assert actual == expected
    assert tuple(actual) == tuple(name for name, _, _ in fields)
    for (_, source, indices), (before, old_indices) in zip(fields[:-1], original, strict=True):
        assert torch.equal(source.contiguous().reshape(-1).view(torch.uint8), before)
        assert torch.equal(indices, old_indices)


@pytest.mark.parametrize("backing", ["memory", "disk"])
def test_bounded_original_mixed_field_byte_offsets_restore_exactly(tmp_path, backing):
    from cubic_serve.runtime.quantized_qualification import _DraftBucketSnapshot

    # Three bytes before int64/complex128 deliberately break typed alignment.
    fields = [("tag", torch.tensor([[3, 17, 255]], dtype=torch.uint8), torch.arange(1)),
              ("counter", torch.arange(14, dtype=torch.int64).reshape(2, 7), torch.arange(2)),
              ("wide", torch.arange(12, dtype=torch.float64).view(torch.complex128).reshape(2, 3),
               torch.arange(2))]
    originals = [tensor.clone() for _, tensor, _ in fields]
    image = _DraftBucketSnapshot(fields, byte_budget=4096 if backing == "memory" else 128,
                                 directory=tmp_path)
    try:
        assert (image.memory is not None) == (backing == "memory")
        for _, tensor, _ in fields:
            tensor.zero_()
        image.restore()
        for (_, tensor, _), original in zip(fields, originals, strict=True):
            assert torch.equal(tensor.view(torch.uint8), original.view(torch.uint8))
    finally:
        image.close()
    assert not tuple(tmp_path.iterdir())


def _cold_draft_owner_fixture():
    arena = SimpleNamespace(_scratch_active={}, _page_pool=SimpleNamespace(_holds=[set(), set()]),
        _state_pools={"state": SimpleNamespace(_holds=[set(), set()])},
        _require_healthy=lambda: None, spec=SimpleNamespace(page_capacity=4))
    round_ = SimpleNamespace(C=1, stride=4, depth=3,
        target=SimpleNamespace(_block_size=16,
            body=SimpleNamespace(arena=arena, hidden=torch.empty(1)),
            _sampling=SimpleNamespace(constraint_bound=False)),
        align=SimpleNamespace(body=SimpleNamespace(arena=arena)),
        head=SimpleNamespace(_sampling=SimpleNamespace(constraint_bound=False)),
        pool=SimpleNamespace(metadata=[None]))
    return round_, arena


@pytest.mark.parametrize("bad", ["admission", "pending", "window", "page", "state", "scratch", "capacity"])
def test_cold_draft_fixture_rejects_ownership_before_mutating_fields(monkeypatch, bad):
    from cubic_serve.runtime import quantized_qualification as q
    round_, arena = _cold_draft_owner_fixture()
    admission = SimpleNamespace(startup_qualification=True)
    if bad == "admission": admission.startup_qualification = False
    elif bad == "pending": round_.pool.metadata[0] = ("request",)
    elif bad == "window": round_._resident_window = SimpleNamespace(inflight=True)
    elif bad == "page": arena._page_pool._holds[1].add(1)
    elif bad == "state": arena._state_pools["state"]._holds[0].add(1)
    elif bad == "scratch": arena._scratch_active["lease"] = object()
    elif bad == "capacity": arena.spec.page_capacity = 0
    monkeypatch.setattr(q, "_draft_window_fixture_fields",
                        lambda *a: pytest.fail("ownership must precede field allocation/writes"))
    with pytest.raises(RuntimeError, match="ownership prerequisite"):
        q._prepare_cold_draft_fixture(round_, None, admission)


def test_cold_draft_fixture_initializes_only_declared_rows(monkeypatch):
    from cubic_serve.runtime import quantized_qualification as q
    round_, arena = _cold_draft_owner_fixture()
    storage = torch.arange(12).reshape(3, 4)
    before, indices = storage.clone(), torch.tensor([1])
    fields = [("state", storage, indices)]
    monkeypatch.setattr(q, "_draft_window_fixture_fields",
                        lambda *a: (fields, indices, indices, indices))
    def initialize(fields, *a, **kw):
        for _, tensor, rows in fields: tensor.index_fill_(0, rows, 0)
    monkeypatch.setattr(q, "_initialize_draft_window_fixture", initialize)
    q._prepare_cold_draft_fixture(round_, None, SimpleNamespace(startup_qualification=True))
    assert torch.equal(storage[[0, 2]], before[[0, 2]])
    assert not bool(storage[1].any())


def test_cold_draft_initialization_precedes_protected_row_baselines(monkeypatch):
    from cubic_serve.runtime import quantized_qualification as q
    round_, arena = _cold_draft_owner_fixture()
    order = []
    monkeypatch.setattr(q, "_prepare_cold_draft_fixture", lambda *a: order.append("cold"))
    def fields(*a):
        assert order == ["cold"]
        raise RuntimeError("stop before protected baseline allocation")
    monkeypatch.setattr(q, "_draft_window_fixture_fields", fields)
    with pytest.raises(RuntimeError, match="stop before protected baseline"):
        q._qualify_draft_window_case(4, round_, None, {}, {}, None, None, (0,),
            rank=0, sampler_rank=0, original_cache={}, initialize_cold_fixture=True)


def test_recording_draft_qualification_declares_cold_fixture_policy(monkeypatch):
    from cubic_serve.runtime import quantized_qualification as q
    session = q.StartupQualification.__new__(q.StartupQualification)
    session.run = lambda label, identity, action: action()
    seen = []
    def qualify(*args, **kwargs):
        seen.append(kwargs)
        return "complete-composition-witness"
    monkeypatch.setattr(q, "qualify_draft_windows", qualify)
    monkeypatch.setattr(q, "_draft_composition_receipt_key", lambda *a: "exact-key")
    assert session.draft_rounds({}, {}, {}, None, None, rank=0, sampler_rank=0) == "complete-composition-witness"
    assert seen[0]["initialize_cold_fixture"] is True
    assert seen[0]["reuse_receipts"] is False


def test_cached_native_composition_still_runs_independent_numerics(monkeypatch):
    from cubic_serve.runtime import quantized_qualification as q
    from cubic_serve.guardrail.quantized_graph import speculative
    round_ = SimpleNamespace(C=1, M=2, depth=1, stride=2)
    rounds = {(2, 1): round_}
    admission = SimpleNamespace(startup_numerical_audit=True, dynamic_speculation=False,
                                num_speculative_tokens=1)
    provider = SimpleNamespace(spec=SimpleNamespace(max_depth=1))
    calls, progress = [], []
    session = q.StartupQualification.__new__(q.StartupQualification)
    def run(label, identity, action, *, reuse=True):
        calls.append((label, reuse))
        if label == "draft-composition": return "cached-native", {(2, 1): ({}, ())}
        assert reuse is False
        return action()
    session.run = run
    monkeypatch.setattr(q, "_draft_window_case_plan", lambda *args: ("native-case",))
    monkeypatch.setattr(q, "_draft_composition_receipt_key", lambda *args: "bound-native-key")
    def numerical(*args, **kwargs):
        assert kwargs["draft_provider"] is provider and kwargs["groups"] == {"tp": "actual"}
        assert len(kwargs["cases"]) == 3
        for step in range(1, 10): kwargs["completed_round"](step, 9)
        return ("fresh-independent-witness",)
    monkeypatch.setattr(speculative, "audit_draft_portfolio_numerics", numerical)
    result = session.draft_rounds(rounds, {}, {}, None, admission, draft_provider=provider,
        groups={"tp": "actual"}, rank=0, sampler_rank=0,
        progress=lambda c, t: progress.append((c, t)))
    assert isinstance(result, str)
    assert calls == [("draft-composition", True), ("draft-independent-numerics", False)]
    assert progress == [(step, 10) for step in range(1, 11)]


@pytest.mark.parametrize('failed', [False, True])
def test_typed_numerical_progress_counts_only_completed_obligations(monkeypatch, failed):
    executors, admission, report, _, _, _, _ = _fake_seal_cross_m(monkeypatch, dtype='bf16')
    updates = []
    if failed:
        def original(*args, **kwargs):
            raise RuntimeError('incomplete original witness')
        monkeypatch.setattr(_seal_audit, 'audit_original', original)
        with pytest.raises(RuntimeError, match='incomplete original witness'):
            _seal_audit.audit_cross_m(executors, admission, {'tp': None}, report,
                completed_witness=lambda done, total: updates.append((done, total)))
        assert updates == [(0, 12)]
    else:
        _seal_audit.audit_cross_m(executors, admission, {'tp': None}, report,
            completed_witness=lambda done, total: updates.append((done, total)))
        assert updates == [(0, 12), (1, 12), (3, 12), (5, 12), (7, 12),
                           (9, 12), (10, 12), (11, 12), (12, 12)]


@pytest.mark.parametrize("device_bytes", [0, 4096])
def test_retired_reference_cache_releases_unused_blocks_after_dropping_witnesses(monkeypatch, device_bytes):
    import weakref
    from cubic_serve.runtime import quantized_qualification as q

    class Witness:
        def __init__(self):
            self.bytes = device_bytes

    cache = {"old": tuple(Witness() if i in (0, 6) else None for i in range(11))}
    witnesses = [weakref.ref(cache["old"][i]) for i in (0, 6)]
    releases = []

    def release():
        assert not cache
        assert all(witness() is None for witness in witnesses)
        releases.append(True)

    monkeypatch.setattr(torch.cuda, "empty_cache", release)
    q._retire_draft_fixture_references(cache)
    assert releases == [True]
    # Empty transitions do not add allocator synchronization.
    q._retire_draft_fixture_references(cache)
    assert releases == [True]


def test_startup_draft_corpus_is_independent_of_physical_pool_capacity(monkeypatch):
    from cubic_serve.runtime.preparation import warm_draft_costs
    from cubic_serve.runtime import performance
    observed=[]
    current=[]
    def execute(command,*args,**kwargs):
        current.append(tuple((r.request_id,r.token_id,r.position,r.page_table,r.sample_ordinal,r.sampling)
                             for r in command.rows))
        for i in range(0,len(command.rows),4):
            r=command.rows[i]
            pool.metadata[r.state_source_slot]=(r.request_id,r.position+1,77,None,r.sample_ordinal+1,None,(101,102,103))
        return SimpleNamespace(step_ms=1.)
    monkeypatch.setattr(performance,'_execute_pipelined_draft_batch',execute)
    for capacity in (32,64):
        arena=SimpleNamespace(shared_storage=SimpleNamespace(state_capacity=capacity),
            reset_state_rows=lambda indices:None,reset_page_rows=lambda indices:None)
        body=SimpleNamespace(hidden=torch.zeros(8,1),arena=arena)
        target=SimpleNamespace(body=body,vocab_size=128,_block_size=16,
                               _sampling=SimpleNamespace(parameters=SimpleNamespace(max_top_k=20)))
        pool=SimpleNamespace(metadata=[None]*capacity,backup={'x':torch.zeros(capacity,1)})
        round_=SimpleNamespace(target=target,align=SimpleNamespace(body=body),C=2,stride=4,
                               pool=pool,_startup_round_costs={})
        warm_draft_costs({(8,3):round_},{},{8:target},torch.zeros(capacity,1),SimpleNamespace(),rank=0,sampler_rank=0)
        observed.append(current[:]);current.clear()
    assert observed[0]==observed[1]
    assert [observed[0][0][i][0] for i in (0,4)]==['startup:0','startup:1']
    assert [observed[0][0][i][1] for i in (0,4)]==[17,18]


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_shared_draft_fixture_uses_disjoint_owned_checkpoints_and_preserves_padding(depth, order):
    from cubic_serve.runtime import quantized_qualification as q
    from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout, StateSlotSpec
    from cubic_serve.runtime.cache.shared_pool import SharedCacheField, SharedCacheStorage

    schemas = tuple((CacheArenaSpec(str(i), 32, 16,
        (StateSlotSpec("matrix", torch.bfloat16, (2, 3)),), scratch_capacity=16),
        KVLayout((i,), 4, 1, 8)) for i in range(2))
    storage = SharedCacheStorage(schemas, budget_bytes=16384, device="cpu",
        extra_state_fields=(SharedCacheField("feedback", torch.bfloat16, (4,)),))
    arenas = tuple(CacheArena(spec, layout, device="cpu", shared_storage=storage,
                              shared_index=i, rank_id=0)
                   for i, (spec, layout) in enumerate(zip(storage.arena_specs,
                                                        (v[1] for v in schemas))))
    hidden = storage.extra_states["feedback"]
    target = SimpleNamespace(body=SimpleNamespace(arena=arenas[0], hidden=torch.zeros(16, 4)),
        vocab_size=256, _block_size=4, _scratch_states=(storage.state_capacity,) * 16)
    provider = SimpleNamespace(body=SimpleNamespace(arena=arenas[1]))
    round_ = SimpleNamespace(C=2, depth=depth, stride=depth + 1, target=target, align=provider,
                             pool=SimpleNamespace(backup={}))
    fields, owners, target_states, provider_states = q._draft_window_fixture_fields(
        round_, hidden, 2, 4)
    pending = [None] * storage.state_capacity
    rows = q._draft_window_audit_rows(round_, pending, 2, 4, owner_order=order)
    assert len(set(target_states.tolist())) == 2 * (depth + 1)
    assert torch.equal(target_states, provider_states)
    assert storage.state_capacity not in target_states.tolist()
    for physical, logical in enumerate(order):
        segment = rows[physical * (depth + 1):(physical + 1) * (depth + 1)]
        assert {r.state_source_slot for r in segment} == {int(owners[logical])}
        assert segment[-1].state_destination_slot == int(owners[logical])
        assert len({r.state_destination_slot for r in segment}) == depth + 1
        assert segment[0].token_id == 101 + logical
        assert segment[0].page_table == tuple(range(logical * 4, (logical + 1) * 4))
    # Initializing the real fixture must neither change active KV payloads nor
    # write the constant padding state shared by all inactive physical rows.
    for arena in arenas:
        arena.key[:8].fill_(19)
        arena.states["matrix"][storage.state_capacity].fill_(23)
    for name, tensor, indices in fields:
        if name.endswith(".matrix"):
            tensor.index_fill_(0, indices, 7)
    for arena in arenas:
        assert bool((arena.key[:8] == 19).all())
        assert bool((arena.states["matrix"][storage.state_capacity] == 23).all())
    protected = q._draft_fixture_protected_pages(arenas[0], 8, target_states, device="cpu")
    # Independently enumerate byte intersections: every returned page is both
    # outside the active page set and disjoint from every owned state bundle.
    pool = storage.allocator
    state_bytes = set()
    for state in target_states.tolist():
        state_bytes.update(range(pool.state_offset_bytes + state * pool.state_bytes,
                                 pool.state_offset_bytes + (state + 1) * pool.state_bytes))
    expected = [page for page in range(8, pool.page_capacity)
                if not state_bytes.intersection(range(page * pool.page_bytes,
                                                       (page + 1) * pool.page_bytes))]
    assert protected.tolist() == expected


def test_shared_draft_fixture_rejects_overlapping_cold_state_and_pages():
    from cubic_serve.runtime import quantized_qualification as q
    from cubic_serve.runtime.cache.shared_pool import SharedCachePool
    pool = SharedCachePool(pool_id="audit", budget_bytes=1024, page_bytes=32, state_bytes=256)
    storage = SimpleNamespace(allocator=pool, state_capacity=pool.state_capacity)
    arena = SimpleNamespace(shared_storage=storage)
    round_ = SimpleNamespace(C=2, depth=1,
        target=SimpleNamespace(body=SimpleNamespace(arena=arena)),
        align=SimpleNamespace(body=SimpleNamespace(arena=arena)))
    with pytest.raises(ValueError, match="disjoint shared cache resources"):
        q._draft_fixture_state_map(round_, 1, 1)
