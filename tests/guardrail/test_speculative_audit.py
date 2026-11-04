"""Owned live numerical prefixes used by independent speculative composition."""
from dataclasses import dataclass
import gc

import pytest
import torch

from cubic_serve.guardrail.quantized_graph.continuation import AuditPrefixStore


def _numerical_portfolio():
    from types import SimpleNamespace as S
    rounds = {(2 if depth == 1 else 4, depth):
              S(C=1, M=2 if depth == 1 else 4, depth=depth, stride=depth + 1)
              for depth in (1, 2, 3)}
    return rounds, S(dynamic_speculation=True, dynamic_max_speculative_tokens=3,
                     num_speculative_tokens=1), S(spec=S(max_depth=3))


def test_numerical_portfolio_declares_every_depth_and_fixed_sampling_policy():
    from cubic_serve.guardrail.quantized_graph.speculative import draft_numerical_case_plan
    rounds, admission, provider = _numerical_portfolio()
    cases = draft_numerical_case_plan(rounds, admission, provider)
    assert [(r.depth, None if p is None else p.seed) for _, r, p in cases] == [
        (1, None), (1, 0x123456789), (1, 0x12345678b),
        (2, None), (2, 0x123456789), (3, None), (3, 0x123456789)]
    admission.dynamic_speculation, admission.num_speculative_tokens = False, 3
    assert len(draft_numerical_case_plan({(4, 3): rounds[4, 3]}, admission, provider)) == 2


@pytest.mark.parametrize("fault", ["missing", "duplicate", "geometry", "provider"])
def test_numerical_portfolio_rejects_missing_or_misbound_units(fault):
    from types import SimpleNamespace as S
    from cubic_serve.guardrail.quantized_graph.speculative import draft_numerical_case_plan
    rounds, admission, provider = _numerical_portfolio()
    if fault == "missing": del rounds[4, 3]
    elif fault == "duplicate": rounds[8, 3] = S(C=1, M=8, depth=3, stride=4)
    elif fault == "geometry": rounds[4, 3].stride = 3
    else: provider = None
    with pytest.raises(ValueError):
        draft_numerical_case_plan(rounds, admission, provider)


@pytest.mark.parametrize("fault", [None, "transition", "progress", "execution"])
def test_numerical_portfolio_shares_only_math_and_closes_on_failed_unit(monkeypatch, fault):
    from cubic_serve.guardrail.quantized_graph import speculative as module
    from cubic_serve.guardrail import quantization
    rounds, admission, provider = _numerical_portfolio()
    memo, called, progress, cleared = object(), [], [], []
    class Memo:
        def clear(self): cleared.append(self)
    monkeypatch.setattr(quantization, "ReferenceMathMemo", Memo)
    def audit(key, round_, *args, **kwargs):
        nonlocal memo
        if not called: memo = kwargs["reference_math_memo"]
        assert kwargs["reference_math_memo"] is memo
        called.append((key, kwargs["sampling"]))
        if fault == "execution": raise RuntimeError("original oracle failed")
        for step in (1, 2, 3): kwargs["completed_round"](step, 4 if fault == "progress" else 3)
        return dict(rounds=[dict(accepted=0)] * 3,
            transitions=[dict(report=dict(candidate_boundary_continuation_state_and_logits_exact=
                fault != "transition", independent_trajectory_diagnostic=dict(route_branch_schedule_exact=True)))],
            sampling_policy="greedy" if kwargs["sampling"] is None else "stochastic",
            sampling_parameters=None, total_seconds=1.)
    monkeypatch.setattr(module, "audit_carried_draft_numerics", audit)
    def run():
        return module.audit_draft_portfolio_numerics(
            rounds, {}, {}, None, admission, draft_provider=provider, groups={},
            rank=0, sampler_rank=0, completed_round=lambda c, t: progress.append((c, t)))
    if fault is None:
        result = run()
        assert len(result) == len(called) == 7
        assert progress == [(n, 21) for n in range(1, 22)]
    else:
        with pytest.raises(RuntimeError): run()
        assert len(called) == 1
    assert cleared == [memo]

def test_sampling_reference_uses_fixed_proposal_and_verification_rng_coordinates():
    from cubic_serve.generation.sampler import SamplingParameters
    from cubic_serve.guardrail.quantized_graph.speculative import SamplingReference, philox_uniform

    sampling = SamplingParameters(0.6, 0.95, 2, 0x123456789)
    position = (1 << 33) + 7
    oracle = SamplingReference(sampling, position=position)
    logits = torch.tensor([[[0., 1., -2.]]], dtype=torch.bfloat16)
    proposal = oracle.propose(logits, 0)
    ordinal = position + 1
    assert oracle.events[-1]["uniform"] == philox_uniform(
        sampling.seed ^ (sampling.seed >> 32) ^ (ordinal >> 32), ordinal)
    decision = oracle.verify((proposal,), (logits,), (logits, logits))
    assert decision.accepted_draft_tokens == 1
    assert oracle.position == position + 2
    event = oracle.events[-1]
    key = (sampling.seed ^ (sampling.seed >> 32) ^ 0x9E3779B9) & 0xFFFFFFFF
    assert event["acceptance_uniform"] == philox_uniform(key, ordinal * 2)
    assert event["sample_uniform"] == philox_uniform(key ^ 0x85EBCA6B, ordinal * 2 + 1)
    assert event["q"] == event["p"]
    assert event["accepted"] is True
    assert decision.next_root_token_id == oracle.categorical(event["p"], event["sample_uniform"])


def test_sampling_reference_rejection_uses_target_ordered_positive_residual():
    from cubic_serve.generation.sampler import SamplingParameters
    from cubic_serve.guardrail.quantized_graph.speculative import SamplingReference

    oracle = SamplingReference(SamplingParameters(1., 1., 2, 17))
    q = torch.tensor([[[100., 0., -1.]]])
    p = torch.tensor([[[0., 100., -1.]]])
    decision = oracle.verify((0,), (q,), (p, p))
    assert decision.accepted_draft_tokens == 0
    assert decision.next_root_token_id == 1
    assert oracle.position == 1
    assert oracle.events[-1]["accepted"] is False
    with pytest.raises(ValueError, match="producing q"):
        oracle.verify((2,), (q,), (p, p))
    with pytest.raises(ValueError, match="complete fixed"):
        oracle.verify((0,), (), (p, p))


def test_sampling_reference_stable_ties_top_p_and_greedy_chain():
    from cubic_serve.generation.sampler import GREEDY_SAMPLING, SamplingParameters
    from cubic_serve.guardrail.quantized_graph.speculative import SamplingReference

    logits = torch.tensor([[[2., 2., 1.]]])
    oracle = SamplingReference(SamplingParameters(1., 0.5, 2, 0))
    assert oracle.distribution(logits) == {0: 0.5, 1: 0.5}
    greedy = SamplingReference(GREEDY_SAMPLING)
    assert greedy.propose(logits, 0) == 0
    decision = greedy.verify((0, 1), (logits, torch.flip(logits, (-1,))),
                             (logits, logits, logits))
    assert (decision.accepted_draft_tokens, decision.next_root_token_id) == (1, 0)
    for invalid in (torch.tensor([float("nan")]), torch.tensor([float("inf")])):
        with pytest.raises(ValueError, match="finite"):
            oracle.distribution(invalid)


def test_speculative_vocabulary_closes_exact_shards_before_decisions():
    from cubic_serve.guardrail.quantized_graph.speculative import _assemble_logits
    left = torch.tensor([[0., -0.]], dtype=torch.bfloat16)
    right = torch.tensor([[3., 4., 5.]], dtype=torch.bfloat16)
    actual = _assemble_logits([(2, 5, right), (0, 2, left)], 5)
    assert torch.equal(actual.view(torch.int16), torch.cat((left, right), -1).reshape(1, 1, 5).view(torch.int16))
    for parts, size in [([(0, 2, left), (3, 6, right)], 6),
                        ([(0, 2, left), (1, 4, right)], 4),
                        ([(0, 2, left), (2, 5, right.float())], 5),
                        ([(0, 2, left), (2, 5, right.flatten())], 5),
                        ([(0, 2, left)], 5), ([], 5)]:
        with pytest.raises(RuntimeError, match="vocabulary"):
            _assemble_logits(parts, size)


def test_speculative_prefix_schema_uses_each_roles_declared_state_inventory():
    from types import SimpleNamespace as S
    from cubic_serve.guardrail.quantized_graph.speculative import _native_prefix_state
    state, key, value, written, page_field = [torch.arange(4) + i * 10 for i in range(5)]
    arena = S(states={"recurrent": None}, page_fields={"epoch": None})
    typed = dict(native=dict(key=key, value=value, written=written, states={"recurrent": state}))
    positional = dict(storage=(key, value, written, page_field, state[None]))
    for payload in (typed, positional):
        result = _native_prefix_state(payload, arena)
        assert torch.equal(result["states"]["recurrent"], state)
        assert result["key"] is key and result["written"] is written
    provider = S(states={}, page_fields={})
    assert _native_prefix_state(dict(native=dict(key=key, value=value, written=written)), provider)["states"] == {}
    for payload in (dict(native=dict(key=key, states={})), dict(storage=(key, value, written))):
        with pytest.raises(RuntimeError, match="inventory"):
            _native_prefix_state(payload, arena)


@pytest.mark.parametrize("capture_fails", [False, True])
def test_pooled_packed_oracle_owns_its_capture_and_preserves_physical_math(monkeypatch, capture_fails):
    from types import SimpleNamespace as S
    from cubic_serve.contracts.execution import RankReplayResources
    from cubic_serve.guardrail.quantized_graph.speculative import _pooled_packed_target
    from cubic_serve.guardrail.quantized_graph import text

    arena, created = S(layout=S(block_size_tokens=16, storage=object())), []
    def private(source, pages, *, scratch_capacity):
        assert pages == 3 and scratch_capacity == 4
        return arena
    monkeypatch.setattr(text, "_audit_cache_arena", private)
    class Stage:
        def __init__(self, config, weights, topology, **options):
            self.options, self.closed = options, False
            created.append(self)
        def capture(self):
            if capture_fails:
                raise RuntimeError("capture fixture failure")
        def close(self):
            self.closed = True
    source = object.__new__(Stage)
    source.config, source.weights, source.topology = object(), object(), object()
    source.captured_m, source.arena, source.mrope_tables = 4, S(layout=arena.layout), object()
    source.operators = S(config=object())
    source.resource = RankReplayResources(0, "g", "s", "b", "w", "scratch:1")
    source._overlap_factory = dict(config=source.config, weights=source.weights,
        topology=source.topology, cache_storage=source.arena.layout.storage,
        operator_config=source.operators.config, model_factory=object())
    if capture_fails:
        with pytest.raises(RuntimeError, match="capture fixture"):
            _pooled_packed_target(source, 2)
        assert created[0].closed
    else:
        stage = _pooled_packed_target(source, 2)
        assert stage is created[0] and not stage.closed
    options = created[0].options
    assert options["captured_m"] == source.captured_m
    assert options["state_slots"] == 1
    assert options["resource"].scratch_arena_id == source.resource.scratch_arena_id
    assert options["resource"].graph_instance_id != source.resource.graph_instance_id
    assert options["arena"] is arena
    assert options["arena"] is not source.arena
    assert options["cache_storage"] is source.arena.layout.storage
    assert options["model_factory"] is source._overlap_factory["model_factory"]


def test_packed_mtp_keeps_global_token_domain_and_local_math_shapes(monkeypatch):
    from types import SimpleNamespace as S
    from cubic_serve.models.qwen3_5.model import Qwen3_5ReferenceModel
    from cubic_serve.guardrail.quantized_graph.replay import reference_model_for_token_domain

    @dataclass(frozen=True)
    class Text:
        vocabulary_size: int = 2
        dtype: str = "bfloat16"
        hidden_size: int = 1
        local_heads: int = 3
    @dataclass(frozen=True)
    class Graph:
        text: Text
    @dataclass(frozen=True)
    class Config:
        graph: Graph
        identity: object
    operators = object()
    base = Qwen3_5ReferenceModel(config=Config(Graph(Text()), S(model_type="qwen3_5")),
                               weights=S(device=torch.device("cpu")), operators=operators,
                               recurrent_state_dtype=torch.bfloat16)
    clone = reference_model_for_token_domain(base, 8)
    assert clone.config.graph.text.local_heads == base.config.graph.text.local_heads == 3
    assert base.config.graph.text.vocabulary_size == 2
    assert clone.config.graph.text.vocabulary_size == 8
    assert clone.weights is base.weights and clone.default_operators is operators
    assert clone.recurrent_state_dtype is torch.bfloat16
    monkeypatch.setattr(Qwen3_5ReferenceModel, "empty_mtp_cache", lambda *a: None)
    monkeypatch.setattr(Qwen3_5ReferenceModel, "_validate_mtp_cache", lambda *a: None)
    def embedding(*a, **kw):
        raise RuntimeError("global token reached the sharded embedding boundary")
    monkeypatch.setattr(Qwen3_5ReferenceModel, "embed_input_ids", embedding)
    feedback, positions = torch.zeros(1, 1, 1, dtype=torch.bfloat16), torch.zeros(3, 1, 1, dtype=torch.int64)
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        base.mtp_forward(torch.tensor([[7]]), feedback, position_ids=positions)
    with pytest.raises(RuntimeError, match="sharded embedding"):
        clone.mtp_forward(torch.tensor([[7]]), feedback, position_ids=positions)
    for token in (-1, 8):
        with pytest.raises(ValueError, match="out-of-vocabulary"):
            clone.mtp_forward(torch.tensor([[token]]), feedback, position_ids=positions)


@dataclass(frozen=True)
class ReferenceState:
    length: int
    value: torch.Tensor


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int64, torch.bool])
def test_live_prefix_forks_own_native_causal_and_free_states(dtype):
    base = torch.arange(24).reshape(4, 6).to(dtype)
    store, binding = AuditPrefixStore(4096), (object(), "wna8", "BF16")
    with torch.inference_mode():
        prefix = store.freeze(binding, dict(position=3, native=base[:, ::2],
            causal=ReferenceState(3, base + 1), independent=ReferenceState(3, base + 2)))
    expected = base[:, ::2].clone()
    base.zero_()
    first = prefix.restore(store, binding, "cpu")
    second = prefix.restore(store, binding, "cpu")
    assert torch.equal(first["native"], expected)
    first["native"].fill_(1)
    first["causal"].value.zero_()
    assert torch.equal(second["native"], expected)
    assert torch.equal(second["causal"].value, prefix.restore(store, binding, "cpu")["causal"].value)
    assert first["causal"].value.data_ptr() != first["independent"].value.data_ptr()
    assert first["causal"].value.data_ptr() != second["causal"].value.data_ptr()
    assert store.bytes == prefix.bytes == store.peak_bytes
    prefix.close()
    assert store.bytes == 0
    with pytest.raises(ValueError, match="lifetime"):
        prefix.restore(store, binding, "cpu")


def test_live_prefix_checks_budget_before_clone_and_retires_on_last_owner(monkeypatch):
    store = AuditPrefixStore(64)
    binding = object()
    prefix = store.freeze(binding, torch.ones(8))
    assert store.bytes == 32
    with pytest.raises(MemoryError, match="admitted"):
        store.freeze(binding, torch.empty(9))
    assert store.bytes == 32
    del prefix
    gc.collect()
    assert store.bytes == 0
    prefix = store.freeze(binding, torch.empty(16))
    assert store.bytes == store.peak_bytes == 64
    prefix.close()
    prefix.close()
    assert store.bytes == 0


def test_live_prefix_rejects_source_store_mutation_and_unsupported_schema():
    store, binding = AuditPrefixStore(128), object()
    prefix = store.freeze(binding, {"native": torch.tensor([-0., float("nan")])})
    for other_store, other_binding in [(store, object()), (AuditPrefixStore(128), binding)]:
        with pytest.raises(ValueError, match="source"):
            prefix.restore(other_store, other_binding, "cpu")
    restored = prefix.restore(store, binding, "cpu")["native"]
    assert restored.view(torch.int32).tolist() == [-2147483648, 2143289344]
    prefix._tensors[0][0].zero_()
    with pytest.raises(ValueError, match="modified"):
        prefix.restore(store, binding, "cpu")
    prefix.close()
    with pytest.raises(TypeError, match="schema"):
        store.freeze(binding, object())
    assert store.bytes == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA prefix transfer")
def test_live_prefix_restores_bf16_state_before_output_and_next_transition():
    # The durable rounding boundary survives CPU retention, branching and restore.
    store, binding = AuditPrefixStore(4096), object()
    with torch.inference_mode():
        exact = torch.tensor([1.00390625, 1.01171875], device="cuda")
        rounded = exact.bfloat16()
        prefix = store.freeze(binding, ReferenceState(1, rounded))
    restored = prefix.restore(store, binding, "cuda")
    assert restored.value.dtype is torch.bfloat16
    assert torch.equal(restored.value.view(torch.int16), rounded.view(torch.int16))
    assert not torch.equal(restored.value.float(), exact)
    prefix.close()


def test_speculative_reference_preserves_free_feedback_provenance_through_alignment():
    from cubic_serve.executor.draft import DraftSpec
    from cubic_serve.guardrail.quantized_graph.speculative import SpeculativeReference
    calls = []
    def transition(tokens, state, axes, feedback, independent_feedback):
        calls.append((int(tokens.item()), state, feedback, independent_feedback))
        ordinal = len(calls)
        return dict(position=ordinal - 1, token=int(tokens.item()), prefix=(ordinal, state),
            logits=torch.zeros(1, 1, 7),
            boundaries=dict(draft_feedback=torch.full((1, 3), ordinal, dtype=torch.bfloat16)),
            independent_feedback=torch.full((1, 1, 3), ordinal + 100, dtype=torch.bfloat16),
            report=dict(candidate_boundary_continuation_state_and_logits_exact=True,
                        independent_trajectory_diagnostic=dict(route_branch_schedule_exact=True)))
    ref = SpeculativeReference(DraftSpec('audit.hidden', 7, 3, 3),
        target_transition=transition, provider_transition=lambda *a: transition(*a), device='cpu')
    axes = torch.zeros(3, 1, 1, dtype=torch.int64)
    token = torch.ones(1, 1, dtype=torch.int64)
    initial = ref.external_feedback(torch.zeros(1, 1, 3, dtype=torch.bfloat16))
    proposal = ref.forward(token, initial, position_ids=axes)
    target = ref.target(token, None, axes)
    # Realignment starts from the pending root's state and TARGET feedback.
    aligned = ref.forward(token, target.feedback, proposal.state, position_ids=axes)
    assert calls[-1][1] is proposal.state
    assert calls[-1][2].flatten().tolist() == [2] * 3
    assert calls[-1][3].flatten().tolist() == [102] * 3
    # Autoregressive proposals then carry PROVIDER feedback independently.
    ref.forward(token, aligned.feedback, aligned.state, position_ids=axes)
    assert calls[-1][2].flatten().tolist() == [3] * 3
    assert calls[-1][3].flatten().tolist() == [103] * 3
    with pytest.raises(ValueError, match='no witnessed producer'):
        ref.forward(token, target.feedback.clone(), proposal.state, position_ids=axes)


def test_live_prefix_rejects_changed_discrete_prefix_or_replaced_tensor():
    store, binding = AuditPrefixStore(128), object()
    for change in ('position', 'tensor'):
        prefix = store.freeze(binding, dict(position=1, native=torch.ones(3)))
        if change == 'position':
            prefix._value['position'] = 2
        else:
            prefix._value['native'] = prefix._value['native'].clone()
        with pytest.raises(ValueError, match='schema was modified'):
            prefix.restore(store, binding, 'cpu')
        prefix.close()
    assert store.bytes == 0


def test_speculative_reference_rejects_mutation_after_feedback_witness():
    from cubic_serve.executor.draft import DraftSpec
    from cubic_serve.guardrail.quantized_graph.speculative import SpeculativeReference
    ref = SpeculativeReference(DraftSpec('audit.hidden', 7, 3, 1),
        target_transition=lambda *a: pytest.fail('transition must not run'),
        provider_transition=lambda *a: pytest.fail('transition must not run'), device='cpu')
    value = ref.external_feedback(torch.zeros(1, 1, 3, dtype=torch.bfloat16))
    value.add_(1)
    with pytest.raises(ValueError, match='feedback was modified'):
        ref.forward(torch.ones(1, 1, dtype=torch.int64), value,
                    position_ids=torch.zeros(3, 1, 1, dtype=torch.int64))


def test_observation_capture_pool_accounts_exclusive_lifetime_and_bounded_eviction():
    from types import SimpleNamespace
    from cubic_serve.guardrail.quantized_graph.capture_session import AuditCaptures
    retired = []
    def payload(name):
        return dict(stage=SimpleNamespace(close=lambda: retired.append(name), launch=lambda: None),
                    clone=torch.ones(4))
    pool = AuditCaptures(64)
    first = payload('first')
    assert pool.acquire('first') is None
    assert pool.retain('first', first, 48)
    with pytest.raises(RuntimeError, match='in-use'):
        pool.clear()
    with pytest.raises(RuntimeError, match='already exclusively'):
        pool.acquire('first')
    assert not pool.retain('second', payload('not-owned'), 32)
    pool.release('first')
    assert pool.acquire('first') is first
    pool.release('first')
    second = payload('second')
    assert pool.retain('second', second, 32)
    assert retired == ['first'] and not first
    pool.release('second')
    assert pool.bytes == 32 and pool.peak_bytes == 48
    pool.clear()
    assert pool.bytes == 0 and not second and retired == ['first', 'second']
    assert not pool.retain('oversized', payload('oversized'), 65)
    assert pool.hits == 1 and pool.misses == 1


@pytest.mark.parametrize('override', [
    dict(token_ids=None), dict(token_ids=(True,)), dict(token_ids=(-1,)),
    dict(token_ids=(8,)), dict(token_ids=(1, 2)), dict(position_axes=()),
    dict(position_axes=((0, 0),)), dict(position_axes=((0, -1, 0),)),
    dict(position_axes=((0, 128, 0),)), dict(prefix_capacity_tokens=None),
    dict(prefix_capacity_tokens=True), dict(prefix_capacity_tokens=129),
    dict(logical_m=2), dict(steps=2),
])
def test_dense_live_transition_rejects_incomplete_or_unadmitted_fixed_trace(override):
    from types import SimpleNamespace as S
    from cubic_serve.guardrail.quantized_graph import audit_original
    storage = S(quantized=False, column_group_alignment=1, identity='BF16')
    stage = S(captured_m=4, arena=S(layout=S(storage=storage)),
              hidden=torch.empty(1, 3, dtype=torch.bfloat16))
    admission = S(max_sequence_tokens=128,
        parallel_plan=S(topology=S(pp_size=1, ep_size=1)),
        config=S(graph=S(text=S(moe=None, vocabulary_size=8))))
    opts = dict(logical_m=1, prefix_length=0, token_ids=(1,),
        position_axes=((0, 0, 0),), prefix_capacity_tokens=16,
        prefix_store=AuditPrefixStore(4096), completed_transition=lambda v: None)
    opts.update(override)
    with pytest.raises(ValueError):
        audit_original(stage, admission, {}, **opts)


def test_projection_observer_owns_inputs_scales_and_activation_parent_chain():
    from cubic_serve.guardrail.quantized_graph.linear import Recorder
    recorder = Recorder()
    weight = torch.ones(3, 4)
    inputs = torch.arange(8, dtype=torch.int8).reshape(2, 4)
    scales = torch.tensor([.5, .25])
    gate, up = torch.ones(2, 3), torch.full((2, 3), 2.)
    recorder.record_projection(weight, inputs, gate, int8_token_scales=scales)
    recorder.record_projection(weight, inputs, up, int8_token_scales=scales)
    active = gate * up
    recorder.record_activation(gate, up, active)
    down = torch.ones(2, 4)
    recorder.record_projection(weight, active, down)
    inputs.zero_(); scales.zero_(); gate.zero_(); up.zero_()
    recorder.freeze()
    assert recorder.events[id(weight)][0][1].flatten().tolist() == list(range(8))
    assert recorder.encoded_inputs[id(weight), 0].tolist() == [.5, .25]
    parents = recorder.activation_sources[id(weight), 2]
    assert [key for key, _ in parents] == [(id(weight), 0), (id(weight), 1)]
    assert parents[0][1].flatten().tolist() == [1.] * 6
    assert parents[1][1].flatten().tolist() == [2.] * 6


@pytest.mark.parametrize('damage', [None, 'linear', 'embedding', 'attention', 'nonlinear', 'new_event'])
def test_provider_consumption_keeps_attention_at_its_own_proof_layer(damage):
    from types import SimpleNamespace as S
    from cubic_serve.guardrail.quantized_graph.packed_mtp import (
        _check_event_consumption, ProviderAuditHarnessError,
    )
    recorder = S(events={'weights':[1,2,3]}, embeddings=[1], attention_events=[1,2])
    nonlinear = S(indices={'zero':1, 'attention':0},
                  r=S(events={'zero':[1], 'attention':[1,2]}))
    causal = S(used={1,2,3}, embedding_used=1, attention_used=2)
    if damage == 'linear': causal.used.remove(3)
    elif damage == 'embedding': causal.embedding_used = 0
    elif damage == 'attention': causal.attention_used = 1
    elif damage == 'nonlinear': nonlinear.indices['zero'] = 0
    elif damage == 'new_event':
        nonlinear.indices['new'] = 0
        nonlinear.r.events['new'] = [1]
    if damage is None:
        _check_event_consumption(causal, nonlinear, recorder)
    else:
        with pytest.raises(ProviderAuditHarnessError):
            _check_event_consumption(causal, nonlinear, recorder)


@pytest.mark.parametrize('kind,exception', [('harness',RuntimeError),
                                          ('execution',RuntimeError),
                                          ('numerical',AssertionError)])
def test_provider_harness_and_execution_failures_are_not_float_verdicts(kind,exception):
    from cubic_serve.guardrail.quantized_graph.packed_mtp import _raise_errors
    with pytest.raises(exception, match=kind+' failure'):
        _raise_errors([None,(kind,'specific cause')], 'completed boundary')
