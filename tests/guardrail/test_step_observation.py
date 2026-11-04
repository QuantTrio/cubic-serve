"""Exact current-step inputs/ownership; these tests issue no numerical seal."""
import copy
from types import SimpleNamespace as S

import pytest
import torch

from cubic_serve.guardrail.quantized_graph.capture_session import OBSERVATION_FIELDS
from cubic_serve.guardrail.quantized_graph.step_observation import (
    ObservationUnavailable, agree_schedules, attention_witnesses, compare_values,
    producing_values, weight_bindings,
)


def fixture():
    weight = torch.zeros(2)
    names = {id(weight): 'norm.weight'}
    key = torch.zeros((1, 1, 4, 1, 2), dtype=torch.bfloat16)
    stage = S(captured_m=1, arena=S(layout=S(storage=S(quantized=False)),
        states={'recurrent': torch.zeros(1, 2)}, key=key, value=key.clone(),
        written=torch.zeros((1, 1, 4), dtype=torch.bool), page_fields={}),
        logits=torch.zeros(1, 3), draft_feedback=torch.zeros(1, 2))
    payload = {name: [] for name in OBSERVATION_FIELDS}
    payload.update(stage=stage, recorder=S(embeddings=[], linear=[], int8_scales={}, shared={}),
        ordered_events=[('norm', 'norm.weight')],
        normalizations=[(torch.ones(1, 2), weight, torch.ones(1, 2), 1e-6, None)],
        index_query_norms=set(), index_key_norms={0: torch.tensor([0])}, moe_events={})
    return payload, names


def view(payload, names, rank=0, position=0):
    return producing_values(payload, names, position=position, tensor_rank=rank,
                            tensor_parts=4, compression=4)


def test_complete_current_values_same_and_missing_new_inventory_fail_closed():
    payload, names = fixture()
    assert compare_values(view(payload, names), view(payload, names))['exact']
    payload['unmapped_future_producer'] = []
    with pytest.raises(ObservationUnavailable, match='inventory differs'):
        view(payload, names)


@pytest.mark.parametrize('kind', ['dtype', 'shape', 'signed-zero', 'nan-payload', 'state', 'input'])
def test_every_semantic_storage_bit_and_typed_boundary_is_required(kind):
    first, names = fixture()
    second = copy.copy(first)
    second['stage'] = copy.copy(first['stage'])
    first['stage'].logits = torch.tensor([0., float('nan')])
    second['stage'].logits = first['stage'].logits.clone()
    if kind == 'dtype':second['stage'].logits = second['stage'].logits.double()
    elif kind == 'shape':second['stage'].logits = second['stage'].logits.reshape(1, 2)
    elif kind == 'signed-zero':second['stage'].logits[0] = -0.
    elif kind == 'nan-payload':second['stage'].logits.view(torch.int32)[1] += 1
    elif kind == 'state':
        second['stage'].arena = copy.copy(first['stage'].arena)
        second['stage'].arena.states = {'recurrent': torch.ones(1, 2)}
    else:
        _, w, y, eps, group = second['normalizations'][0]
        second['normalizations'] = [(torch.zeros(1, 2), w, y, eps, group)]
    assert not compare_values(view(first, names), view(second, names))['exact']


def test_only_declared_unowned_normalization_outputs_have_no_producing_identity():
    first, names = fixture()
    second = copy.copy(first)
    x, w, _, eps, group = first['normalizations'][0]
    second['normalizations'] = [(x, w, torch.full((1, 2), 99.), eps, group)]
    assert compare_values(view(first, names, rank=1), view(second, names, rank=1))['exact']
    assert not compare_values(view(first, names, rank=0), view(second, names, rank=0))['exact']
    second['index_key_norms'] = {0: torch.tensor([4])}
    assert not compare_values(view(first, names, rank=1), view(second, names, rank=1))['exact']


def test_unused_selector_order_is_not_a_boundary_but_consumed_order_remains_exact():
    first, names = fixture()
    second = copy.copy(first)
    inputs = (1, 2, torch.ones(1), torch.ones(1), torch.zeros(1, dtype=torch.int32),
              torch.ones(1, dtype=torch.int32), torch.ones(1, dtype=torch.int32),
              torch.zeros(1, dtype=torch.int32), torch.ones(1, dtype=torch.int32))
    first['selector_events'] = [(*inputs, torch.zeros(1, dtype=torch.int64))]
    second['selector_events'] = [(*inputs, torch.ones(1, dtype=torch.int64))]
    assert compare_values(view(first, names), view(second, names))['exact']
    assert not compare_values(view(first, names, position=3), view(second, names, position=3))['exact']


def test_actual_selected_weight_versions_are_bound_and_unversioned_weights_cannot_reuse():
    payload, names = fixture()
    weight = payload['normalizations'][0][1]
    old = weight_bindings(payload, names)
    weight.add_(1)
    assert weight_bindings(payload, names) != old
    with torch.inference_mode():frozen = torch.zeros(2)
    x, _, y, eps, group = payload['normalizations'][0]
    payload['normalizations'] = [(x, frozen, y, eps, group)]
    with pytest.raises(ObservationUnavailable, match='version binding'):
        weight_bindings(payload, {id(frozen): 'norm.weight'})


def test_attention_partition_is_separate_but_each_actual_enclosure_is_evaluated():
    payload, _ = fixture()
    key = payload['stage'].arena.key
    value = torch.ones_like(key)
    query = torch.ones((1, 1, 2), dtype=torch.bfloat16)
    output = torch.ones_like(query)
    for splits in (16, 32):
        payload['ordered_events'] = [('attention', 0, splits)]
        payload['attention_events'] = [(0, .5, query, key, value,
            torch.zeros((1, 1), dtype=torch.int32), torch.ones(1, dtype=torch.int32),
            output, torch.zeros((1, 1), dtype=torch.int32), splits)]
        proof = attention_witnesses(payload, position=0, attention_dtype='auto')[0]
        assert proof['kv_splits'] == splits and proof['outside_enclosure'] == 0
    payload['attention_events'][0][8].fill_(2)
    with pytest.raises(RuntimeError, match='selected-position ownership'):
        attention_witnesses(payload, position=0, attention_dtype='auto')


def test_cross_rank_ordered_schedule_agreement_precedes_data_inspection(monkeypatch):
    from cubic_serve.guardrail.quantized_graph import step_observation as module
    payload, _ = fixture()
    monkeypatch.setattr(module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(module.dist, 'get_world_size', lambda group: 2)
    def gather(output, value, group):output[:] = [value, (('different',),)]
    monkeypatch.setattr(module.dist, 'all_gather_object', gather)
    with pytest.raises(RuntimeError, match='rank event schedules differ'):
        agree_schedules((payload,), None)


def session_fixture(monkeypatch, *, wrong_second=False, mutate_weight=False):
    from cubic_serve.executor.text_serving import TextServingBuffers
    from cubic_serve.guardrail.quantized_graph.capture_session import AuditCaptures
    from cubic_serve.guardrail.quantized_graph.step_observation import CurrentPlanWitnesses
    from cubic_serve.runtime import text_qualification
    first, names = fixture()
    second, _ = fixture()
    # Both graphs bind one checkpoint; their actual states stay private.
    second['normalizations'] = first['normalizations']
    root = object()
    for payload in (first, second):
        stage = payload['stage']
        stage.weights, stage.config = root, S(model='same original rule')
        stage.operators, stage.recurrent_state_dtype = S(config=S(attention_dtype='auto')), torch.bfloat16
        stage.helper, stage.topology, stage.tensor_rank = S(local_tp_routers=()), S(tp_size=4), 0
        stage.indexer_plan = S(compression=4)
        stage.arena.spec = S(page_capacity=1)
        stage.arena.reset_page_rows = lambda indices: None
        stage.feature_rows, stage.embedding_mask = torch.zeros(1, 2), torch.ones(1, dtype=torch.bool)
        ids = [torch.zeros(1, dtype=torch.int64) for _ in range(9)]
        stage.buffers = TextServingBuffers(*ids[:5], torch.zeros(1, 1, dtype=torch.int64),
            *ids[5:8], (torch.zeros(2, 1, 1, dtype=torch.int32),),
            torch.zeros(1, 2), torch.zeros(1, 2))
        stage.hidden = stage.buffers.hidden
        stage.close, stage.synchronize = lambda: None, lambda: None
    calls = []
    def replay():
        calls.append(1)
        second['stage'].arena.states['recurrent'].add_(2 if wrong_second and len(calls)==2 else 1)
        if mutate_weight:first['normalizations'][0][1].add_(1)
    second['stage'].replay = replay
    monkeypatch.setattr(text_qualification, 'prepare_rows', lambda stage, n: None)
    pool = AuditCaptures(1024)
    assert pool.retain('candidate', second, 1)
    pool.release('candidate')
    session = CurrentPlanWitnesses(pool, {'M16': 'candidate'}, names, None)
    return first, second, pool, session


@pytest.mark.parametrize('wrong_second', [False, True])
def test_current_native_prefix_induction_and_mismatch_requires_complete_original(monkeypatch, wrong_second):
    first, _, pool, session = session_fixture(monkeypatch, wrong_second=wrong_second)
    session.start(first['stage'], first)
    original = {'scope': 'independently completed original math', 'steps': []}
    try:
        for position in range(2):
            session.before_step(first['stage'], position)
            first['stage'].arena.states['recurrent'].add_(1)
            report = dict(position=position, candidate_boundary_continuation_state_and_logits_exact=True,
                independent_trajectory_diagnostic={'route_branch_schedule_exact': True},
                attention_reduction_proofs=[])
            original['steps'].append(report)
            session.completed_step(position, first, report)
    finally:
        session.finish()
    proof = session.report('M16', original)
    if wrong_second:
        assert proof is None and 'M16' in session.unavailable
    else:
        assert proof['cross_step_error_propagation_qualified']
        assert len(proof['steps'])==2 and all(
            step['current_operator_math_transport']['actual_weight_versions_exact'] for step in proof['steps'])
    assert not pool.borrowed
    pool.clear()


def test_current_witness_rejects_weight_mutation_and_releases_its_capture(monkeypatch):
    first, _, pool, session = session_fixture(monkeypatch, mutate_weight=True)
    session.start(first['stage'], first)
    session.before_step(first['stage'], 0)
    first['stage'].arena.states['recurrent'].add_(1)
    try:
        with pytest.raises(RuntimeError, match='weight version binding changed'):
            session.completed_step(0, first, {})
    finally:
        session.finish()
    assert not pool.borrowed
    pool.clear()


def test_unnamed_derived_tensor_requires_full_typed_content_and_own_version(monkeypatch):
    first, second, pool, session = session_fixture(monkeypatch)
    left, right = torch.arange(4.), torch.arange(4.)
    for payload, weight in ((first, left), (second, right)):
        payload['recorder'].embeddings = [(weight, torch.tensor([0]), torch.ones(1, 4))]
    session.start(first['stage'], first)
    session.before_step(first['stage'], 0)
    first['stage'].arena.states['recurrent'].add_(1)
    try:
        session.completed_step(0, first, dict(position=0, attention_reduction_proofs=[],
            candidate_boundary_continuation_state_and_logits_exact=True,
            independent_trajectory_diagnostic={'route_branch_schedule_exact': True}))
        assert len(session.steps['M16']) == 1
        right[0] = -99
        assert not compare_values(view(first, session.names), view(second, session.names))['exact']
    finally:
        session.finish()
        pool.clear()


def test_discrete_input_write_is_an_execution_failure_even_when_outputs_match(monkeypatch):
    first, second, pool, session = session_fixture(monkeypatch)
    replay = second['stage'].replay
    def corrupt():
        replay()
        second['stage'].buffers.page_destination.add_(1)
    second['stage'].replay = corrupt
    session.start(first['stage'], first)
    session.before_step(first['stage'], 0)
    first['stage'].arena.states['recurrent'].add_(1)
    try:
        with pytest.raises(RuntimeError, match='protected discrete input metadata'):
            session.completed_step(0, first, {})
    finally:
        session.finish()
        pool.clear()


def test_input_witness_budget_never_issues_a_partial_original_pass(monkeypatch):
    first, _, pool, session = session_fixture(monkeypatch)
    session.input_byte_budget = 0
    session.start(first['stage'], first)
    session.before_step(first['stage'], 0)
    assert not session.active and session.report('M16', {'steps': [1,2]}) is None
    session.finish()
    pool.clear()
