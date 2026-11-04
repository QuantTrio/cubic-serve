"""Stream dependencies preserve each lane's rounded recurrence across replays."""

import os
from functools import partial
from types import MethodType, SimpleNamespace

import pytest
import torch

from cubic_serve.executor.decode_overlap import DecodeInterleave, decode_calibration_prepare


@pytest.mark.parametrize("times,expected", [
    ((14.0, 11.0, 11.0, 10.0), (10.0, 11.0)),
    ((10.0, 8.0, 8.2, 10.0), (10.0, 8.2)),
    ((10.0, 8.0, 10.5, 10.0), (10.0, 10.5)),
])
def test_decode_comparison_rejects_stale_baseline_and_one_lucky_sample(
    monkeypatch, times, expected,
):
    from cubic_serve.executor import decode_overlap as module

    control, candidate, prepare, stream = (object() for _ in range(4))
    observations = iter(times)
    calls = []

    def measure(graph, reset, target_stream, samples):
        assert reset is prepare and target_stream is stream and samples == 3
        calls.append(graph)
        return next(observations)

    monkeypatch.setattr(module, "measure_decode_graph", measure)
    assert module.compare_decode_graphs(control, candidate, prepare, stream, 3) == expected
    assert calls == [control, candidate, candidate, control]


@pytest.mark.parametrize("quantized", [False, True])
def test_calibration_restores_embedding_ownership_after_every_reset(quantized):
    from cubic_serve.executor.model_runner import Qwen3_5TP2PP2DecodeBody
    from cubic_serve.executor.quantized_model_runner import QuantizedDecodeStage

    rng = torch.random.get_rng_state().clone()
    originals = []
    for start in (0, 128):
        body = SimpleNamespace(
            captured_m=64, _vocab_size=256, _is_first_stage=True,
            hidden=torch.empty(64, 1),
            buffers=SimpleNamespace(token_ids=torch.zeros(64, dtype=torch.int64)),
            embedding_span=SimpleNamespace(start=start, stop=start + 128),
            embedding_mask=torch.zeros(64, 1), feature_rows=torch.ones(64, 1),
        )
        if quantized:
            body.first = True
            body._vocab_span = body.embedding_span
            body.body = SimpleNamespace(buffers=body.buffers)
            bind_tokens = MethodType(QuantizedDecodeStage.bind_tokens, body)
        else:
            bind_tokens = MethodType(Qwen3_5TP2PP2DecodeBody.fill_token_ids, body)
        resets = []

        def reset(resource, body=body, resets=resets):
            resets.append(resource)
            body.buffers.token_ids.zero_()
            body.embedding_mask.zero_()

        body.prepare_scratch = reset
        resource = object()
        prepare = decode_calibration_prepare(
            rows=body.captured_m, vocabulary_size=body._vocab_size,
            device=body.hidden.device, reset=partial(body.prepare_scratch, resource),
            bind_tokens=bind_tokens,
        )
        prepare()
        ids, mask = body.buffers.token_ids.clone(), body.embedding_mask.clone()
        assert 0 < int(mask.sum()) < body.captured_m
        assert ids[mask[:, 0].bool()].unique().numel() > 1
        if not quantized:
            assert not body.feature_rows.any()
        body.buffers.token_ids.fill_(999)
        body.embedding_mask.fill_(-1)
        prepare()
        assert resets == [resource, resource]
        assert torch.equal(body.buffers.token_ids, ids)
        assert torch.equal(body.embedding_mask, mask)
        originals.append((ids + start, mask[:, 0].bool()))
    # Every rank receives the same corpus, with exactly one embedding owner.
    assert torch.equal(originals[0][1], ~originals[1][1])
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.skipif(
    os.environ.get("CUBIC_RUN_CUDA_GRAPH_TESTS") != "1", reason="requires CUDA graph audit"
)
@pytest.mark.parametrize("lanes", [2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_interleaved_recurrences_match_sequential_graph(lanes, dtype):
    device = torch.device("cuda", 0)
    inputs = [torch.full((4096,), 0.25, dtype=dtype, device=device) for _ in range(lanes)]
    states = [torch.empty_like(x) for x in inputs]
    outputs = [torch.empty_like(x) for x in inputs]
    schedule = DecodeInterleave(lanes, device)
    stream = torch.cuda.Stream(device=device)

    def steps(index):
        state = states[index]
        state.copy_(inputs[index])
        # Different sequence lengths exercise lane retirement and the final
        # compute/communication join, with a dtype rounding boundary per step.
        for _ in range(index + 2):
            state.mul_(0.5)
            yield partial(state.add_, 0.125)
        outputs[index].copy_(state)

    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        schedule.launch([steps(i) for i in range(lanes)])
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        schedule.launch([steps(i) for i in range(lanes)])
    for value in (0.375, -0.625, 0.75):
        for x in inputs:
            x.fill_(value)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        for index, output in enumerate(outputs):
            expected = inputs[index].clone()
            for _ in range(index + 2):
                expected.mul_(0.5).add_(0.125)
            assert torch.equal(output, expected)
    stream.synchronize()
    graph.reset()


def test_overlap_resource_candidates_preserve_explicit_policy_and_arithmetic():
    from cubic_serve.ops.strategies import (
        OperatorConfig,
        collective_parameters,
        collective_thread_config,
        collective_thread_scope,
        operator_strategy_scope,
        overlap_collective_threads,
    )

    config = OperatorConfig(activation_dtype="fp32", attention_dtype="auto", softmax_dtype="fp32")
    with operator_strategy_scope(config, device="arbitrary-device", model="arbitrary-model"):
        assert overlap_collective_threads() == (None, 256)
        with collective_thread_scope(256):
            assert collective_parameters().host_sum_threads == 256
            assert overlap_collective_threads() == (None,)
        assert collective_parameters().host_sum_threads == 128
    changed = collective_thread_config(config, 64)
    assert changed.activation_dtype == config.activation_dtype
    assert changed.softmax_dtype == config.softmax_dtype
    assert changed.attention_dtype == config.attention_dtype
    assert changed.profiles == config.profiles
    with operator_strategy_scope(changed, device="other", model="other"):
        assert overlap_collective_threads() == (None,)
    assert collective_thread_config(config, None) is config


def test_overlap_plan_rebind_preserves_quantized_child_configuration():
    from cubic_serve.executor.decode_overlap import DecodeOverlapPlan
    from cubic_serve.executor.quantized_model_runner import QuantizedDecodeStage
    from cubic_serve.ops.strategies import OperatorConfig

    owner = object.__new__(QuantizedDecodeStage)
    original = OperatorConfig(overrides=(("collective", (("transport_calibration", False),)),))
    arena = object()
    owner._overlap_factory = dict(operator_config=original, arena=arena)
    plan = DecodeOverlapPlan((32, 32), collective_threads=256)
    factory = owner._overlap_child_factory(plan)
    bound = factory["operator_config"].resolve("collective", device="any", model="any")
    assert bound.host_sum_threads == 256
    assert not bound.transport_calibration
    assert factory["arena"] is arena
    assert owner._overlap_factory["operator_config"] is original
    assert original.resolve("collective", device="any", model="any").host_sum_threads == 128
    assert owner._overlap_child_factory(DecodeOverlapPlan())["operator_config"] is original


@pytest.mark.parametrize("parts,threads", [
    ((32,), None), ((32, 0), None), ((), 256), ((32, 32), 192),
])
def test_overlap_plan_rejects_incomplete_resource_receipts(parts, threads):
    from cubic_serve.executor.decode_overlap import DecodeOverlapPlan

    with pytest.raises(ValueError):
        DecodeOverlapPlan(parts, collective_threads=threads)


def test_overlap_search_releases_previous_winner_before_next_resource_candidate(monkeypatch):
    from cubic_serve.executor import quantized_model_runner as module
    from cubic_serve.executor.decode_overlap import DecodeOverlapPlan
    from cubic_serve.ops.strategies import OperatorConfig

    stage = object.__new__(module.QuantizedDecodeStage)
    allocated = [0]
    observed = []
    def noop(*args, **kwargs):
        pass

    class Projection:
        operation = "linear"
        row_tile_count = 2

    class Graph:
        def __init__(self, size):
            self.size = size

        def reset(self):
            allocated[0] -= self.size

    def child_factory(*, captured_m, operator_config):
        allocated[0] += 40
        return SimpleNamespace(
            captured_m=captured_m, prepare_scratch=noop, launch=noop,
            threads=operator_config.resolve("collective", device="", model="").host_sum_threads,
        )

    def capture(children):
        observed.append(tuple(c.threads for c in children))
        return Graph(40 * len(children)), children, None, None

    stage._graph = object()
    stage._overlap_resources = None
    stage.overlap_measurements = []
    stage.overlap_plan = DecodeOverlapPlan()
    stage.topology = SimpleNamespace(tp_size=4, pp_size=1)
    stage.body = SimpleNamespace(_packed_moe=None, projection=Projection())
    stage.arena = SimpleNamespace(device=torch.device("cpu"))
    stage._overlap_policy = SimpleNamespace(
        overlap_partitions=lambda m, tiles: ((32, 32),),
        decode_overlap_samples=3, decode_overlap_min_gain_percent=5,
        decode_overlap_max_bytes=100,
    )
    stage._overlap_thread_candidates = (None, 256)
    stage._overlap_factory = dict(operator_config=OperatorConfig())
    stage._private_bytes = 40
    stage._vocab_span = SimpleNamespace(size=100)
    stage.captured_m, stage.rank, stage.stream = 64, 0, object()
    stage.prepare_scratch = stage.bind_tokens = stage.synchronize = noop
    stage._capture_interleaved = capture
    monkeypatch.setattr(module, "PackedProjectionDispatch", Projection)
    monkeypatch.setattr(module, "QuantizedDecodeStage", child_factory)
    monkeypatch.setattr(module, "measure_decode_graph", lambda *args: 10.0)
    timings = iter(((10.0, 8.0), (10.0, 7.0)))
    monkeypatch.setattr(module, "compare_decode_graphs", lambda *args: next(timings))
    monkeypatch.setattr(module.dist, "get_world_size", lambda group=None: 4)
    monkeypatch.setattr(module.dist, "all_reduce", noop)
    monkeypatch.setattr(module.dist, "barrier", noop)
    monkeypatch.setattr(module.torch.cuda, "synchronize", noop)
    monkeypatch.setattr(module.torch.cuda, "memory_allocated", lambda *args: allocated[0])
    monkeypatch.setattr(module.torch.cuda, "mem_get_info", lambda *args: (10000, 10000))

    stage.prepare_decode_overlap()
    assert observed == [(128, 128), (256, 256), (256, 256)]
    assert allocated[0] == 80
    assert stage.overlap_plan == DecodeOverlapPlan((32, 32), collective_threads=256)


def test_overlap_resource_candidates_use_topology_intersection(monkeypatch):
    from cubic_serve.executor import decode_overlap as module

    def restrict_to_other_rank(mask, op):
        assert op == torch.distributed.ReduceOp.MIN
        mask.mul_(torch.tensor([1, 0, 0]))

    monkeypatch.setattr(module.dist, "all_reduce", restrict_to_other_rank)
    assert module.common_overlap_thread_candidates((None, 256), torch.device("cpu")) == (None,)
    assert module.common_overlap_thread_candidates((None, 64), torch.device("cpu")) == (None, 64)
