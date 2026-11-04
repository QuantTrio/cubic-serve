"""Draft transactions preserve independent feedback and accepted state prefixes."""

import pytest
import torch

from cubic_serve.executor.draft import DraftSpec, DraftStep, reference_draft_round
from cubic_serve.generation.tp_sampling import SpeculativeSamplingDecision


def test_dense_provider_reuses_original_math_and_explicit_prefix_state():
    from cubic_serve.executor.runner import ReferenceExecutionContext
    from cubic_serve.models.builtin import builtin_draft_provider
    from tests.models.qwen3_5.test_model_multimodal import _tiny_model

    model = _tiny_model()
    provider = builtin_draft_provider(model.config, method="mtp", max_depth=7)
    draft = provider.build_reference(model)
    width = model.config.graph.text.hidden_size
    feedback = torch.ones(1, 1, width, dtype=torch.bfloat16)
    tokens = torch.tensor([[2]], dtype=torch.int64)
    with ReferenceExecutionContext():
        first = draft.forward(tokens, feedback)
        prefix_key = first.state.layer.key.clone()
        actual = draft.forward(tokens, feedback, first.state)
        expected = model.mtp_forward(
            tokens, feedback, cache=first.state,
            position_ids=torch.ones((3, 1, 1), dtype=torch.int64),
        )
    assert torch.equal(actual.logits, expected.logits)
    assert torch.equal(actual.feedback, expected.hidden_states)
    assert torch.equal(first.state.layer.key, prefix_key)
    assert actual.state.sequence_length == 2
    assert provider.spec.feedback_width == width
    assert provider.spec.max_depth == 7


@pytest.mark.parametrize("depth", [1, 2, 3, 7])
def test_all_rejection_boundaries_and_full_acceptance_catchup(depth):
    class Provider:
        spec = DraftSpec("mtp", 64, 2, 7)
        def __init__(self):
            self.calls = []

        def forward(self, tokens, feedback, state=None, *, position_ids=None):
            assert feedback.shape == (1, 1, 2)
            state = () if state is None else state
            token = int(tokens.item())
            next_state = (*state, (token, tuple(feedback.flatten().tolist())))
            next_feedback = feedback + torch.tensor([[[1., 10.]]])
            logits = torch.zeros(1, 1, 64)
            logits[..., token + 1] = 1
            self.calls.append((token, feedback.clone(), position_ids.clone()))
            return DraftStep(logits, next_feedback, next_state)

    def target(tokens, state, positions):
        previous = () if state is None else state
        token = int(tokens.item())
        p = int(positions[0].item())
        return DraftStep(torch.zeros(1, 1, 64), torch.tensor([[[float(p), -float(p)]]]),
                         (*previous, token))

    for accepted in range(depth + 1):
        provider = Provider()
        provider.calls = []
        initial_feedback = torch.tensor([[[101., 202.]]])

        def verify(proposals, q, p, accepted=accepted):
            assert proposals == tuple(range(5, 5 + depth))
            assert len(q) == depth and len(p) == depth + 1
            return SpeculativeSamplingDecision(accepted, 31)

        result = reference_draft_round(
            provider=provider, target=target, root_token=torch.tensor([[4]]),
            feedback=initial_feedback, target_state=(), draft_state=(),
            position=12, depth=depth,
            propose=lambda logits, _: int(logits.argmax(-1).item()), verify=verify,
        )
        assert result.target_state == tuple(range(4, 5 + accepted))
        assert tuple(x[0] for x in result.draft_state) == tuple(range(4, 5 + accepted))
        assert result.output_tokens == (*range(5, 5 + accepted), 31)
        assert len(provider.calls) == depth + int(accepted == depth)
        assert torch.equal(initial_feedback, torch.tensor([[[101., 202.]]]))
        if accepted == depth:
            # Target feedback from the preceding target row, never the final
            # draft feedback or the collapsed logits representation.
            assert torch.equal(provider.calls[-1][1], result.target_steps[-2].feedback)
            assert int(provider.calls[-1][2][0].item()) == 12 + depth
            assert result.catchup is not None
        else:
            assert result.catchup is None
            assert result.draft_state == result.proposal_steps[accepted].state


@pytest.mark.parametrize("kwargs", [
    dict(method="", vocabulary_size=32, feedback_width=2, max_depth=3),
    dict(method="mtp", vocabulary_size=32, feedback_width=-1, max_depth=3),
    dict(method="mtp", vocabulary_size=32, feedback_width=2, max_depth=True),
])
def test_draft_capability_rejects_invalid_capacity(kwargs):
    with pytest.raises(ValueError, match="capacity"):
        DraftSpec(**kwargs)


def test_unconditioned_provider_uses_same_transaction():
    class Provider:
        spec = DraftSpec("external", 32, 0, 2)

        def forward(self, tokens, feedback, state=None, *, position_ids=None):
            assert feedback is None
            state = () if state is None else state
            return DraftStep(torch.zeros(1, 1, 32), None, (*state, int(tokens.item())))

    result = reference_draft_round(
        provider=Provider(), target=lambda tokens, state, pos: DraftStep(
            torch.zeros(1, 1, 32), None, (*(state or ()), int(tokens.item()))),
        root_token=torch.tensor([[4]]), feedback=None, target_state=None, draft_state=None,
        position=0, depth=2, propose=lambda logits, index: 5 + index,
        verify=lambda tokens, q, p: SpeculativeSamplingDecision(1, 9),
    )
    assert result.target_state == result.draft_state == (4, 5)
    assert result.output_tokens == (5, 9)


@pytest.mark.parametrize("depth", [1, 2, 3, 7])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_production_ragged_commit_matches_independent_locked_round(
    monkeypatch, depth, reverse, dynamic
):
    """Audit driver ownership against the same frozen proposal/decision trace.

    Sampling math is audited separately. These scalar model rules make every
    feedback consumer and typed state write observable without a shared body.
    """
    from types import SimpleNamespace

    from cubic_serve.runtime import performance as serving
    from cubic_serve.runtime.continuous_batch import BatchCommand, BatchRow, batch_segments

    monkeypatch.setattr(torch.distributed, "broadcast", lambda *args, **kwargs: None)
    capacities = (1, 2, 4, 8, 16, 32)
    size = 40

    def rounded(value):
        return float(torch.tensor(value, dtype=torch.bfloat16))

    class Provider:
        spec = DraftSpec("mtp", 128, 2, 7)

        def forward(self, tokens, feedback, state=None, *, position_ids=None):
            token, position = int(tokens.item()), int(position_ids[0].item())
            count, history, raw, _ = state
            next_state = (count + 1, history * 3 + token,
                          rounded(raw + float(feedback.sum())), position)
            return DraftStep(torch.zeros(1, 1, 128), feedback + 1, next_state)

    def target(tokens, state, positions):
        token, position = int(tokens.item()), int(positions[0].item())
        count, history, raw, _ = state
        return DraftStep(torch.zeros(1, 1, 128),
                         torch.tensor([[[float(token), float(position)]]], dtype=torch.bfloat16),
                         (count + 1, history * 3 + token, rounded(raw + token), position))

    def state_fields():
        return {"count": torch.full((size,), 2, dtype=torch.int64),
                "history": torch.full((size,), 13, dtype=torch.int64),
                "raw_keys": torch.full((size,), 5., dtype=torch.bfloat16),
                "positions": torch.full((size,), 11, dtype=torch.int64)}

    def read(fields, owner):
        return tuple(fields[name][owner].item() for name in
                     ("count", "history", "raw_keys", "positions"))

    def write(fields, owner, values):
        for name, value in zip(("count", "history", "raw_keys", "positions"), values, strict=True):
            fields[name][owner] = value

    for accepted in range(depth + 1):
        target_state, draft_state = state_fields(), state_fields()
        hidden = torch.full((size, 2), 7., dtype=torch.bfloat16)
        owners = (1, 0) if reverse else (0, 1)
        depths = {0: depth, 1: 1}
        accepted_by_owner = {0: accepted, 1: 1}
        rows = []
        for owner in owners:
            k = depths[owner]
            start = len(rows)
            for ordinal in range(k + 1):
                rows.append(BatchRow(
                    str(owner), 4 + owner if ordinal == 0 else 0, 12 + ordinal,
                    owner, owner if ordinal == k else 8 + start + ordinal,
                    owner, owner, (owner,), segment_length=k + 1 if ordinal == 0 else 0,
                    is_decode=True, mtp_position=12 + ordinal,
                    mtp_page_destination=owner, mtp_page_table=(owner,),
                ))
        rows = tuple(rows)
        initial_target = {owner: read(target_state, owner) for owner in owners}
        initial_draft = {owner: read(draft_state, owner) for owner in owners}
        initial_feedback = {owner: hidden[owner].clone()[None, None] for owner in owners}

        class Target:
            body = SimpleNamespace(arena=SimpleNamespace(states=target_state))
            initial = initial_target
            feedback = hidden

            def replay_batch(self, bound, *, commit_hidden):
                assert commit_hidden
                self.rows = bound
                for start, stop in batch_segments(bound):
                    state = self.initial[int(bound[start].request_id)]
                    for row in bound[start:stop]:
                        step = target(torch.tensor([[row.token_id]]), state,
                                      torch.full((3, 1, 1), row.position, dtype=torch.int64))
                        state = step.state
                        write(self.body.arena.states, row.state_destination_slot, state)
                        self.feedback[row.state_destination_slot] = step.feedback[0, 0]

            def gather_candidates(self, bound):
                pass

            def replay_elapsed_ms(self):
                return 1.

        class Draft:
            state_fields = draft_state
            accepts = accepted_by_owner
            vocab_size = 128
            captured_m = 32
            _max_draft_tokens = 7
            chain_tokens = serving.CapturedDraftRank.chain_tokens

            class CostEvent:
                def __init__(self, owner):
                    self.owner, self.recorded = owner, False

                def record(self):
                    self.recorded = True

                def elapsed_time(self, end):
                    assert self.owner.materialized and self.recorded and end.recorded
                    return 2.5

            def __init__(self):
                self._token_ids = torch.zeros(32, dtype=torch.int64)
                self._proposal_input_tokens = torch.zeros_like(self._token_ids)
                self._verification_draft_tokens = torch.zeros(7, 32, dtype=torch.int64)
                self._verification_validity = torch.zeros(7, dtype=torch.bool)
                self.materialized = False
                self._proposal_cost_events = tuple(
                    (self.CostEvent(self), self.CostEvent(self)) for _ in range(7)
                )

            def materialize_proposals(self, counts):
                result = serving.CapturedDraftRank.materialize_proposals(self, counts)
                self.materialized = True
                return result

            def reserve_state_slots(self, slots, rows=rows):
                assert set(slots) == {slot for row in rows for slot in
                                     (row.state_source_slot, row.state_destination_slot)}

            def replay(self, bound, feedback, *, sample=True, input_tokens=None,
                       validity_out=None):
                from dataclasses import replace

                if input_tokens is not None:
                    bound = tuple(replace(row, token_id=int(input_tokens[index]))
                                  for index, row in enumerate(bound))
                output_feedback = []
                for index, row in enumerate(bound):
                    step = Provider().forward(torch.tensor([[row.token_id]]),
                        feedback[index:index + 1, None],
                        read(self.state_fields, row.state_source_slot),
                        position_ids=torch.full((3, 1, 1), row.position, dtype=torch.int64))
                    write(self.state_fields, row.state_destination_slot, step.state)
                    output_feedback.append(step.feedback[0, 0])
                self.feedback = torch.stack(output_feedback)
                if not sample:
                    return ()
                tokens = tuple(30 + row.position for row in bound)
                self._token_ids[:len(tokens)] = torch.tensor(tokens)
                if validity_out is not None:
                    validity_out.fill_(True)
                    return ()
                return tokens

            def stash_proposal(self, depth, count, *, source):
                assert self._proposal_cost_events[depth][1].recorded == dynamic
                self._verification_draft_tokens[depth, :count].copy_(source._token_ids[:count])

            def chain_hidden(self, indices):
                return self.feedback[list(indices)].clone()

            def launch_verification(self, target_executor, physical_rows_by_depth):
                decisions = torch.full((depth, 2, 2), -1, dtype=torch.int64)
                for ordinal, level in enumerate(physical_rows_by_depth):
                    for index, physical in enumerate(level):
                        owner = int(target_executor.rows[physical].request_id)
                        decisions[ordinal, index] = torch.tensor(
                            (int(ordinal < self.accepts[owner]), 91 + ordinal))
                return decisions

            def flush_verification_profile(self):
                pass

        draft = Draft()
        result = serving.execute_draft_batch(
            BatchCommand(1, next(m for m in capacities if m >= len(rows)), rows),
            Target(), {m: draft for m in capacities}, hidden,
            SimpleNamespace(num_speculative_tokens=1, max_speculative_tokens=7,
                            dynamic_speculation=dynamic),
            capacities, rank=0, sampler_rank=0,
        )
        assert result.proposal_samples == (
            tuple((2 if ordinal == 0 else 1, 2.5) for ordinal in range(depth))
            if dynamic else ()
        )
        assert all(start.recorded == end.recorded == (dynamic and ordinal < depth)
                   for ordinal, (start, end) in enumerate(draft._proposal_cost_events))
        for owner in owners:
            k, r = depths[owner], accepted_by_owner[owner]
            expected = reference_draft_round(
                provider=Provider(), target=target, root_token=torch.tensor([[4 + owner]]),
                feedback=initial_feedback[owner], target_state=initial_target[owner],
                draft_state=initial_draft[owner], position=12, depth=k,
                propose=lambda logits, ordinal: 42 + ordinal,
                verify=lambda proposals, q, p, r=r, k=k:
                    SpeculativeSamplingDecision(r, 91 + min(r, k - 1)),
            )
            assert read(target_state, owner) == expected.target_state
            assert read(draft_state, owner) == expected.draft_state
            assert torch.equal(hidden[owner], expected.feedback[0, 0])
            segment = next(s for s in result.segment_results if s.request_id == str(owner))
            assert segment.output_token_ids == expected.output_tokens
            assert segment.committed_row_count == r + 1
        # Foreign persistent owners are untouched. Active transaction scratch
        # remains private and can contain uncommitted future checkpoints.
        for fields in (target_state, draft_state):
            assert read(fields, 2) == (2, 13, 5., 11)
        assert torch.equal(hidden[2], torch.tensor([7., 7.], dtype=torch.bfloat16))


def test_device_proposal_inputs_are_bounded_without_repairing_history():
    from types import SimpleNamespace

    from cubic_serve.runtime.performance import CapturedDraftRank, Qwen3_5TP2PP2ServingError

    owner = SimpleNamespace(
        _token_ids=torch.empty(4, dtype=torch.int64),
        _proposal_input_tokens=torch.empty(4, dtype=torch.int64),
        captured_m=4, vocab_size=128, _max_draft_tokens=2,
        _verification_validity=torch.tensor([True, False]),
        _verification_draft_tokens=torch.tensor([[7, 11, 13, -1], [-1, 128, 3, -1]]),
    )
    original = owner._verification_draft_tokens.clone()
    result = CapturedDraftRank.chain_tokens(owner, original[1], (1, 0, 2))
    assert result.tolist() == [127, 0, 3, 0]
    assert torch.equal(owner._verification_draft_tokens, original)
    with pytest.raises(Qwen3_5TP2PP2ServingError, match="sampling guard"):
        CapturedDraftRank.materialize_proposals(owner, (3, 3))
    owner._verification_validity.fill_(True)
    with pytest.raises(Qwen3_5TP2PP2ServingError, match="token range guard"):
        CapturedDraftRank.materialize_proposals(owner, (3, 3))
    assert CapturedDraftRank.materialize_proposals(owner, (3,)) == ((7, 11, 13),)


def test_draft_verifier_without_startup_witness_fails_closed():
    from cubic_serve.runtime.performance import CapturedDraftRank, Qwen3_5TP2PP2ServingError

    draft = CapturedDraftRank.__new__(CapturedDraftRank)
    draft._verifier_qualified = False
    with pytest.raises(Qwen3_5TP2PP2ServingError, match="startup qualification"):
        draft.launch_verification(None, ((0,),))


@pytest.mark.parametrize("depth", [1, 2, 3, 7])
def test_carried_round_realigns_with_target_feedback_and_preserves_root_checkpoint(depth):
    from cubic_serve.executor.draft import reference_pending_draft, reference_pipelined_draft_round

    class Provider:
        spec = DraftSpec("any-hidden-provider", 128, 2, 7)

        def forward(self, tokens, feedback, state=None, *, position_ids=None):
            # Deliberately make target and autoregressive feedback different.
            # Every durable BF16 step is rounded before subsequent consumption.
            raw = (feedback.float() * 1.03125 + int(tokens.item())).to(torch.bfloat16)
            previous = () if state is None else state
            checkpoint = (*previous, (int(tokens.item()), int(position_ids[0].item()),
                                      tuple(feedback.flatten().tolist())))
            return DraftStep(raw, raw, checkpoint)

    def target(tokens, state, positions):
        value, position = int(tokens.item()), int(positions[0].item())
        feedback = torch.tensor([[[position * .125, value * 1.0625]]], dtype=torch.bfloat16)
        return DraftStep(torch.zeros(1, 1, 128), feedback, (*(state or ()), value))

    provider = Provider()
    initial = torch.tensor([[[7., 11.]]], dtype=torch.bfloat16)
    for accepted in range(depth + 1):
        pending = reference_pending_draft(
            provider=provider, root_token=torch.tensor([[4]]), feedback=initial,
            draft_state=(), position=12, depth=depth, propose=lambda logits, i: 20 + i,
        )
        original_state = pending.root_step.state
        original_feedback = pending.root_step.feedback.clone()
        result = reference_pipelined_draft_round(
            provider=provider, target=target, pending=pending, target_state=(),
            device=torch.device("cpu"), next_depth=depth,
            propose=lambda logits, i: 40 + i,
            verify=lambda proposals, q, p: SpeculativeSamplingDecision(accepted, 9),
        )
        assert result.output_tokens == (*range(20, 20 + accepted), 9)
        assert result.target_state == (4, *range(20, 20 + accepted))
        assert tuple(x[0] for x in result.draft_state) == result.target_state
        assert tuple(x[0] for x in result.pending.root_step.state) == (*result.target_state, 9)
        assert len(result.alignment_steps) == accepted + 1
        assert len(result.pending.proposal_steps[-1].state) == len(result.target_state) + depth
        assert result.pending.position == 12 + accepted + 1
        for i, step in enumerate(result.alignment_steps):
            witness = step.state[-1]
            assert witness[1] == 13 + i
            assert witness[2] == tuple(result.target_steps[i].feedback.flatten().tolist())
        assert pending.root_step.state == original_state
        assert torch.equal(pending.root_step.feedback, original_feedback)
        assert torch.equal(initial, torch.tensor([[[7., 11.]]], dtype=torch.bfloat16))
        # A second round consumes the carried root exactly once and excludes
        # rejected autoregressive suffixes from its durable provider state.
        second = reference_pipelined_draft_round(
            provider=provider, target=target, pending=result.pending,
            target_state=result.target_state, device=torch.device("cpu"), next_depth=1,
            propose=lambda logits, i: 60,
            verify=lambda proposals, q, p: SpeculativeSamplingDecision(0, 10),
        )
        assert second.target_state == (*result.target_state, 9)
        assert tuple(x[0] for x in second.draft_state) == second.target_state
        assert second.pending.root_step.state[-1][0] == 10


@pytest.mark.parametrize("kind", ["dense", "flash"])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_pipelined_reference_composes_original_model_math_with_target_alignment(kind, depth):
    from copy import deepcopy
    from dataclasses import fields, is_dataclass
    from cubic_serve.executor.draft import reference_pending_draft, reference_pipelined_draft_round
    from cubic_serve.executor.runner import ReferenceExecutionContext

    if kind == "dense":
        from cubic_serve.models.builtin import builtin_draft_provider
        from tests.models.qwen3_5.test_model_multimodal import _tiny_model
        model = _tiny_model()
        draft = builtin_draft_provider(model.config, method="mtp", max_depth=3).build_reference(model)

        def target(tokens, state, positions):
            value = model.forward(tokens, cache=state, position_ids=positions)
            return DraftStep(value.logits, value.hidden_states, value.cache)
    else:
        from tests.models.qwen4_exp.test_mtp import tiny_mtp
        draft = tiny_mtp()

        def target(tokens, state, positions):
            logits, state, feedback = draft.target.forward_with_feedback(
                tokens, state, position_ids=positions)
            return DraftStep(logits, feedback, state)

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor) and a.dtype == b.dtype
            assert torch.equal(a, b)
        elif is_dataclass(a):
            assert type(a) is type(b)
            for field in fields(a):
                equal(getattr(a, field.name), getattr(b, field.name))
        elif isinstance(a, tuple):
            assert len(a) == len(b)
            for left, right in zip(a, b, strict=True):
                equal(left, right)
        else:
            assert a == b

    def axes(position):
        return torch.full((3, 1, 1), position, dtype=torch.int64)

    with ReferenceExecutionContext():
        ts = ds = None
        feedback = torch.zeros(1, 1, draft.spec.feedback_width, dtype=torch.bfloat16)
        for position, token in enumerate((2, 3)):
            value = torch.tensor([[token]])
            dstep = draft.forward(value, feedback, ds, position_ids=axes(position))
            tstep = target(value, ts, axes(position))
            ts, ds, feedback = tstep.state, dstep.state, tstep.feedback
        before_target, before_draft = deepcopy(ts), deepcopy(ds)
        pending = reference_pending_draft(
            provider=draft, root_token=torch.tensor([[4]]), feedback=feedback,
            draft_state=ds, position=2, depth=depth, propose=lambda logits, i: 2 + i,
        )
        for accepted in range(depth + 1):
            result = reference_pipelined_draft_round(
                provider=draft, target=target, pending=pending, target_state=ts,
                device=torch.device("cpu"), next_depth=depth,
                propose=lambda logits, i: 6 + i,
                verify=lambda proposals, q, p: SpeculativeSamplingDecision(accepted, 5),
            )
            # Independent sequential evaluation of the consumed target prefix,
            # then accepted draft prefix plus bonus, with immutable checkpoints.
            expected_target = ts
            expected_steps = []
            for i, token in enumerate((4, *pending.proposal_tokens[:accepted])):
                step = target(torch.tensor([[token]]), expected_target, axes(2 + i))
                expected_steps.append(step)
                expected_target = step.state
            expected_draft = pending.root_step.state
            for i, token in enumerate(pending.proposal_tokens[:accepted]):
                step = draft.forward(torch.tensor([[token]]), expected_steps[i].feedback,
                                     expected_draft, position_ids=axes(3 + i))
                expected_draft = step.state
            equal(result.target_state, expected_target)
            equal(result.draft_state, expected_draft)
            bonus = draft.forward(torch.tensor([[5]]), expected_steps[-1].feedback,
                                  expected_draft, position_ids=axes(3 + accepted))
            equal(result.pending.root_step, bonus)
            equal(before_target, ts)
            equal(before_draft, ds)


def test_pipeline_keeps_cache_position_separate_from_three_axis_continuation():
    from cubic_serve.executor.draft import reference_pending_draft, reference_pipelined_draft_round

    class Provider:
        spec = DraftSpec("unconditioned", 32, 0, 3)

        def forward(self, tokens, feedback, state=None, *, position_ids=None):
            assert feedback is None
            entry = (int(tokens.item()), tuple(position_ids.flatten().tolist()))
            return DraftStep(torch.zeros(1, 1, 32), None, (*(state or ()), entry))

    provider = Provider()
    pending = reference_pending_draft(
        provider=provider, root_token=torch.tensor([[4]]), feedback=None,
        draft_state=(), position=5, depth=3, propose=lambda logits, i: 6 + i,
        position_axes=(11, 17, 23), position_step=(0, 1, 2),
    )
    assert pending.proposal_steps[-1].state[-1][1] == (11, 19, 27)

    def target(tokens, state, positions):
        return DraftStep(torch.zeros(1, 1, 32), None,
                         (*(state or ()), tuple(positions.flatten().tolist())))

    result = reference_pipelined_draft_round(
        provider=provider, target=target, pending=pending, target_state=(),
        device=torch.device("cpu"), next_depth=1, propose=lambda logits, i: 12,
        verify=lambda proposals, q, p: SpeculativeSamplingDecision(2, 10),
    )
    assert result.target_state == ((11, 17, 23), (11, 18, 25), (11, 19, 27))
    assert result.pending.position == 8
    assert result.pending.position_axes == (11, 20, 29)
    assert result.pending.root_step.state[-1] == (10, (11, 20, 29))
    assert result.pending.root_step.feedback is None
