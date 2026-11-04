"""Startup restores plans or builds missing resources without engineering tests."""
from types import SimpleNamespace

import pytest
import torch

from cubic_serve.executor.readiness import execution_ready, prepare_execution
from cubic_serve.ops.cuda import cache_identity
from cubic_serve.ops.cuda.startup import startup_weight_identity
from cubic_serve.runtime.preparation import StartupPreparation


@pytest.fixture
def session(monkeypatch, tmp_path):
    monkeypatch.setenv("CUBIC_SERVE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache_identity, "gpu_model", lambda d: ("Model A",))
    return StartupPreparation(0, torch.device("cuda", 0), consensus=lambda value: value)


def forbidden(*a, **kw):
    pytest.fail("production startup entered an engineering fixture or repeated a warm cost")


def memory_measurement():
    return dict(graph_bytes=4096, transient_bytes=128, allocator_slack=256, preparation_bytes=1024,
                extra_state_fields=(("feedback", "bfloat16", (32,)),),
                overlap_plans={4: ((2, 2), False, None)}, round_keys=((4, 1),))


def test_warm_memory_plan_skips_measurement_and_tracks_only_used_plans(session, tmp_path):
    from cubic_serve.ops.cuda.launch_cost import _read_startup_calibration
    dependency = tmp_path / "selected.json"
    unrelated = tmp_path / "other.json"
    dependency.write_text("initial selected plan")
    calls = []

    def measure():
        calls.append(1)
        _read_startup_calibration(dependency, "test")
        return memory_measurement()

    expected = session.memory_plan((1024, "int4", "bf16"), measure, implementation="sources")
    unrelated.write_text("unrelated model plan")
    assert session.memory_plan((1024, "int4", "bf16"), forbidden, implementation="sources") == expected
    assert session.memory_plan_reused
    dependency.write_text("new selected plan")
    assert session.memory_plan((1024, "int4", "bf16"), measure, implementation="sources") == expected
    assert len(calls) == 2
    assert not session.memory_plan_reused
    session.memory_plan((512, "int4", "bf16"), measure, implementation="sources")
    session.memory_plan((1024, "bf16", "bf16"), measure, implementation="sources")
    session.memory_plan((1024, "int4", "bf16"), measure, implementation="changed sources")
    assert len(calls) == 5


def test_memory_default_invalidates_when_missing_selected_plan_appears(session, tmp_path):
    from cubic_serve.ops.cuda.launch_cost import _read_startup_calibration
    dependency = tmp_path / "initially-missing.json"
    calls = []

    def measure():
        calls.append(1)
        _read_startup_calibration(dependency, "test")
        return memory_measurement()

    session.memory_plan(1024, measure, implementation="sources")
    session.memory_plan(1024, forbidden, implementation="sources")
    dependency.write_text("installed choice")
    session.memory_plan(1024, measure, implementation="sources")
    assert len(calls) == 2


def test_one_rank_memory_miss_rebuilds_collectively(session):
    session.memory_plan(1024, memory_measurement, implementation="sources")
    flags = iter((False, True))
    session._consensus = lambda value: next(flags)
    assert session.memory_plan(1024, memory_measurement, implementation="sources") == memory_measurement()
    assert session.misses == 2


def test_probe_plan_dependencies_also_invalidate_memory_estimate(session, tmp_path):
    from cubic_serve.ops.cuda.launch_cost import _plan_file_digest
    probe = tmp_path / "probe-fit.json"
    probe.write_text("coefficients used by every graph shape")
    session.memory_plan(1024, memory_measurement, implementation="sources",
                        dependencies={str(probe): _plan_file_digest(probe)})
    session.memory_plan(1024, forbidden, implementation="sources")
    probe.write_text("new fit with same M1 choice but different large-M choice")
    session.memory_plan(1024, memory_measurement, implementation="sources")
    assert session.misses == 2


@pytest.mark.parametrize("name,value", [("graph_bytes", -1), ("transient_bytes", True),
    ("extra_state_fields", (("feedback", "missing_dtype", (32,)),)),
    ("overlap_plans", {4: ((-1, 2), False, None)}), ("round_keys", ((4, 0),))])
def test_invalid_memory_plan_never_publishes(session, tmp_path, name, value):
    result = dict(memory_measurement(), **{name: value})
    with pytest.raises(ValueError, match="invalid graph memory"):
        session.memory_plan(1024, lambda: result, implementation="sources")
    assert not tuple(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("fail_state_capture", [False, True])
def test_packed_mtp_reports_only_complete_live_capture(session, monkeypatch, fail_state_capture):
    from contextlib import contextmanager, nullcontext
    from cubic_serve.executor import mtp_runner

    stage = mtp_runner.PackedMTPStage.__new__(mtp_runner.PackedMTPStage)
    stage._graph, stage._state_graph = None, None
    stage._closed, stage._captured_once = False, False
    stage.captured_m, stage.hidden = 1, torch.zeros(1)
    stage.arena = SimpleNamespace(device=torch.device("cpu"))
    stage.stream = SimpleNamespace(wait_stream=lambda _: None, synchronize=lambda: None)
    stage.prepare_scratch = lambda: None
    stage.launch = lambda **_: None
    graphs = []

    @contextmanager
    def capture(graph, **_):
        graphs.append(graph)
        if fail_state_capture and len(graphs) == 2:
            raise RuntimeError("state capture failed")
        yield

    monkeypatch.setattr(mtp_runner, "CUDAGraph", object)
    monkeypatch.setattr(mtp_runner, "capture_graph", capture)
    monkeypatch.setattr(mtp_runner.torch.cuda, "current_stream", lambda _: None)
    monkeypatch.setattr(mtp_runner.torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(mtp_runner, "barrier", lambda: None)
    with pytest.raises(RuntimeError, match="has not captured"):
        session.capture(stage)
    with pytest.raises(ValueError, match="no live"):
        _ = stage.graph
    if fail_state_capture:
        with pytest.raises(RuntimeError, match="state capture failed"):
            stage.capture()
        assert not stage._captured_once
        with pytest.raises(RuntimeError, match="has not captured"):
            session.capture(stage)
    else:
        stage.capture()
        assert stage._captured_once
        assert stage.graph is graphs[0] and stage._state_graph is graphs[1]
        session.capture(stage)
        assert execution_ready(stage)
        with pytest.raises(ValueError, match="recapture"):
            stage.capture()
        stage._closed = True
        with pytest.raises(ValueError, match="no live"):
            _ = stage.graph


def test_unscoped_startup_cannot_search_candidates():
    from cubic_serve.ops.cuda.startup import calibration_allowed, require_calibration_permission
    assert not calibration_allowed()
    with pytest.raises(RuntimeError):
        require_calibration_permission("unscoped input encoder")
    with startup_weight_identity(None, allow_calibration=True):
        assert calibration_allowed()
        require_calibration_permission("explicit qualification")
    assert not calibration_allowed()


def test_missing_and_corrupt_cost_entries_rebuild_and_then_reuse(session, tmp_path):
    calls = []
    def prepare():
        calls.append(1)
        return {(4, True): [1.25, 1.5]}
    expected = session.run("target-costs", (4,), prepare)
    assert session.misses == 1
    assert session.run("target-costs", (4,), forbidden) == expected
    assert session.hits == 1 and len(calls) == 1
    path = next(tmp_path.rglob("*.json"))
    path.write_text("{partial")
    assert session.run("target-costs", (4,), prepare) == expected
    assert len(calls) == 2


def test_one_rank_miss_rebuilds_this_entry_collectively(session):
    session.run("target-costs", (4,), lambda: 1.)
    flags = iter((False, True))
    session._consensus = lambda value: next(flags)
    assert session.run("target-costs", (4,), lambda: 2.) == 2.


def test_structurally_valid_but_wrong_cost_entry_rebuilds(session):
    session.run("draft-costs", (4, 1), lambda: {(2, True): [1.]})
    expected = {(4, True): [1.]}
    validate = lambda value: isinstance(value, dict) and set(value) == {(4, True)}
    assert session.run("draft-costs", (4, 1), lambda: expected, validate=validate) == expected
    assert session.run("draft-costs", (4, 1), forbidden, validate=validate) == expected


def test_adding_a_bucket_only_prices_the_missing_bucket(session, monkeypatch):
    from cubic_serve.runtime import quantized_qualification
    monkeypatch.setattr(quantized_qualification, "_bound_composition_plan",
                        lambda body, **kwargs: body.image)
    calls = []
    def price(portfolio):
        calls.extend(portfolio)
        return tuple((capacity, float(capacity)) for capacity in portfolio)
    portfolio = {capacity: SimpleNamespace(body=SimpleNamespace(image=f"binary-{capacity}"))
                 for capacity in (1, 4, 8)}
    assert session.costs("target-costs", {k: portfolio[k] for k in (1, 4)}, price) == ((1, 1.), (4, 4.))
    assert session.costs("target-costs", portfolio, price) == ((1, 1.), (4, 4.), (8, 8.))
    assert calls == [1, 4, 8]
    portfolio[4].body.image = "updated-kernel"
    session.costs("target-costs", portfolio, price)
    assert calls == [1, 4, 8, 4]


def test_failed_preparation_never_publishes_a_result(session, tmp_path):
    def fail():
        raise ValueError("preparation failed")
    with pytest.raises(ValueError):
        session.run("target-costs", (4,), fail)
    assert not tuple(tmp_path.rglob("*.json"))
    with pytest.raises(RuntimeError, match="failed"):
        session.finish()


@pytest.mark.parametrize("fail", [False, True])
def test_offline_cost_publication_is_complete_and_waits_for_success(session, monkeypatch, tmp_path, fail):
    from cubic_serve.runtime import quantized_qualification as q
    monkeypatch.setattr(q.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(q.dist, "all_gather_object", lambda out, value: out.__setitem__(0, value))
    monkeypatch.setattr(q, "_bound_composition_plan", lambda body, **kw: body.image)
    monkeypatch.setattr(q, "_qualification_program_identity", lambda resource: "complete-test-program")
    offline = q.StartupQualification("prices", 0, recording=True, root=tmp_path/'offline')
    offline.preparation = session
    portfolio = {n: SimpleNamespace(body=SimpleNamespace(image=f"binary-{n}")) for n in (4, 8)}
    values = ((4, 2, 1.25), (4, 3, 1.5), (8, 2, 2.))
    assert offline.costs("depth", portfolio, lambda _: values, entry=4) == values
    assert not (tmp_path/'operators').exists()
    if fail:
        with pytest.raises(ValueError):
            offline.run("later-check", (), lambda: (_ for _ in ()).throw(ValueError()))
        with pytest.raises(RuntimeError):
            offline.finish()
        assert not (tmp_path/'operators').exists()
    else:
        offline.finish()
        assert session.costs("depth", portfolio, forbidden, entry=4) == values
        assert session.hits == 2


def test_complete_draft_prices_preserve_partial_owner_and_constraint_domains(session, monkeypatch):
    from cubic_serve.runtime import quantized_qualification as q
    from cubic_serve.runtime.preparation import valid_draft_costs
    monkeypatch.setattr(q, "_bound_composition_plan", lambda stage, **kw: getattr(stage, "image", "sampler"))
    stage = SimpleNamespace(body=SimpleNamespace(image="model-binary"),
        vocab_size=128, _block_size=16,
        _sampling=SimpleNamespace(parameters=SimpleNamespace(max_top_k=20)))
    round_ = SimpleNamespace(M=16, C=4, stride=3, target=stage, align=stage, head=stage,
        target_greedy=object(), head_greedy=object(), target_masked=object(), head_masked=object(),
        budget=None, qualified=False,
        _startup_round_costs={(r, g): [1.25, 1.5] for r in (9, 12) for g in (False, True)},
        _startup_constraint_round_costs={(r, g, d): [2., 2.25]
            for r in (9, 12) for g in (False, True) for d in (1, 2)})
    expected = round_._startup_round_costs, round_._startup_constraint_round_costs
    assert valid_draft_costs(expected, round_)
    session.publish_draft_costs({(16, 2): round_})
    identity, images = session.draft_cost_identity((16, 2), round_)
    restored = session.run("draft-costs", identity, forbidden, kernels=images,
                           validate=lambda v: valid_draft_costs(v, round_))
    assert restored == expected
    assert len(restored[0]) == 4 and len(restored[1]) == 8
    assert round_.qualified is False and not hasattr(round_, "_execution_capabilities")
    for key in ((6, True), (10, True), (9, 1), (9., True)):
        bad = {k: v for k, v in expected[0].items() if k != key}
        bad[key] = [1.]
        assert not valid_draft_costs((bad, expected[1]), round_)
    assert not valid_draft_costs((expected[0], {(9, True, 3): [1.]}), round_)


def test_component_binding_does_not_execute_or_fabricate_audit_flags(session):
    owner = SimpleNamespace(qualified=False, evidence={}, qualify=forbidden,
                            graphs=[object()], calls=[])
    session.component(owner, ())
    assert owner.qualified is False and owner.evidence == {}
    assert execution_ready(owner)
    owner._closed = True
    assert not execution_ready(owner)


def test_missing_graph_or_changed_operand_fails_before_runtime_admission():
    for owner, resources in ((SimpleNamespace(), {"graphs": (None,)}),
                             (SimpleNamespace(), {}),
                             (SimpleNamespace(), {"modules": (SimpleNamespace(),)}),
                             (SimpleNamespace(), {"calls": (SimpleNamespace(
                                 tensors=(object(),), bindings=()),)})):
        with pytest.raises(RuntimeError):
            prepare_execution(owner, **resources)
        assert not execution_ready(owner)


def test_normal_cold_launch_selection_does_not_search_or_fabricate_timings(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from cubic_serve.ops.cuda import launch_cost
    from cubic_serve.ops.cuda import startup
    monkeypatch.setenv("CUBIC_SERVE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache_identity, "gpu_model", lambda d: ("Model A",))
    monkeypatch.setattr(torch.cuda, "device", lambda d: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(startup, "startup_cuda_lock", lambda d: nullcontext())
    bound = []
    with startup_weight_identity(("checkpoint",), allow_calibration=False):
        selected = launch_cost.select_launch((2, 4), device=torch.device("cuda", 0),
            bind=bound.append, launch=forbidden, snapshot=forbidden,
            cache_key=("operator", 128), kernels=("binary-A",))
        assert selected == 2 and bound == [2]
        body = launch_cost._read_startup_calibration(next(tmp_path.rglob("*.json")),
                                                    "cubic.launch-calibration.v1")
        assert body["basis"] == "retained_default" and "scores" not in body


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA resource binding check")
def test_prepared_native_sampler_keeps_runtime_guards_and_audit_state_separate(tmp_path):
    import torch.distributed as dist
    from cubic_serve.generation.tp_sampling import (
        ShardedSamplingParameters, ShardedSamplingWorkspace, TensorParallelSamplingError)
    from cubic_serve.models.parallel_weights import AxisSpan
    from cubic_serve.ops.cuda.sampling.greedy import CapturedGreedySampler

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    plan = None
    try:
        capacity, width, candidates = 2, 64, 20
        parameters = ShardedSamplingParameters(torch.zeros(capacity, device=device),
            torch.full((capacity,), candidates, dtype=torch.int64, device=device),
            torch.full((capacity,), .95, device=device), candidates,
            torch.ones(capacity, dtype=torch.int64, device=device),
            torch.zeros(capacity, dtype=torch.int64, device=device))
        logits = torch.arange(capacity * width, device=device, dtype=torch.bfloat16).reshape(capacity, width)
        workspace = ShardedSamplingWorkspace(
            torch.empty((capacity, candidates), dtype=torch.bfloat16, device=device),
            torch.empty((capacity, candidates), dtype=torch.int64, device=device),
            torch.empty((1, capacity, candidates), dtype=torch.bfloat16, device=device),
            torch.empty((1, capacity, candidates), dtype=torch.int64, device=device))
        plan = CapturedGreedySampler(local_logits=logits, local_vocab_span=AxisSpan(0, width),
            final_ranks=(0,), final_vocab_spans=(AxisSpan(0, width),), true_vocab_size=width,
            token_ids=torch.empty(capacity, dtype=torch.int64, device=device),
            parameters=parameters, workspace=workspace, final_tp_group=dist.group.WORLD,
            world_group=dist.group.WORLD, root_valid=torch.empty(capacity, dtype=torch.bool, device=device),
            capture_stream=torch.cuda.Stream(device=device))
        session = StartupPreparation(0, device, consensus=lambda x: x)
        session.component(plan, ())
        assert plan.qualified is False and plan.evidence == {}
        assert plan.replay(capacity, False).cpu().tolist() == [width - 1] * capacity
        parameters.top_ps[0] = 0
        with pytest.raises(TensorParallelSamplingError, match="invalid"):
            plan.replay(capacity, False)
        assert plan.qualified is False
        plan.close()
        assert not execution_ready(plan)
    finally:
        if plan is not None:
            plan.close()
        dist.destroy_process_group()
