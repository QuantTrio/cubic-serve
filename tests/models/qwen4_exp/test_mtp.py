"""Original-checkpoint MTP math, independent feedback and immutable state."""

from copy import deepcopy
from dataclasses import fields, replace

import pytest
import torch

from cubic_serve.executor.draft import DraftSpec, DraftStep, reference_draft_round
from cubic_serve.executor.runner import ReferenceExecutionContext
from cubic_serve.generation.tp_sampling import SpeculativeSamplingDecision
from cubic_serve.models.qwen4_exp.model import Qwen4ExpMTPReference
from cubic_serve.models.qwen4_exp.weights import _expected_headers
from tests.models.qwen4_exp.test_reference import tiny_model


def test_feedback_traversal_preserves_every_producer_and_skips_only_logits():
    from cubic_serve.models.qwen4_exp.model import bind_qwen4_exp_text_graph

    config = tiny_model().config
    calls = []

    class Binder:
        def bind(self, operation, weight_prefix, *, layer_id):
            return lambda: calls.append((operation, weight_prefix, layer_id))

    program = bind_qwen4_exp_text_graph(config, Binder())
    program.launch()
    full = tuple(calls)
    calls.clear()
    program.launch_feedback()
    assert tuple(calls) == full[:-1]
    assert full[-1][0] == "lm_head"


def tiny_mtp():
    target = tiny_model()
    torch.manual_seed(191)
    intermediate = target.config.text.moe.expert_intermediate_size
    for name, h in _expected_headers(target.config).items():
        if h.partition != "mtp":
            continue
        value = (torch.randn(h.shape) * 0.1).bfloat16()
        if name.endswith(".experts.gate_up_proj"):
            for expert in range(h.shape[0]):
                root = name.rsplit(".", 1)[0] + f".{expert}"
                target.weights[root + ".gate_proj.weight"] = value[expert, :intermediate]
                target.weights[root + ".up_proj.weight"] = value[expert, intermediate:]
        elif name.endswith(".experts.down_proj"):
            for expert in range(h.shape[0]):
                root = name.rsplit(".", 1)[0] + f".{expert}"
                target.weights[root + ".down_proj.weight"] = value[expert]
        else:
            target.weights[name] = value
    return Qwen4ExpMTPReference(target, spec=DraftSpec("mtp", target.config.text.vocabulary_size,
                            target.config.text.hidden_size *
                            target.config.text.hyperconnection.stream_count, 7))


def test_mtp_chunking_matches_independently_advanced_state_and_does_not_mutate_prefix():
    draft = tiny_mtp()
    tokens = torch.tensor([[1, 5, 9, 2, 8, 3]])
    feedback = torch.randn(1, 6, draft.spec.feedback_width).bfloat16()
    with ReferenceExecutionContext():
        whole = draft.forward(tokens, feedback)
        first = draft.forward(tokens[:, :3], feedback[:, :3])
        snapshot = {f.name: getattr(first.state.layers[0], f.name).clone()
                    for f in fields(first.state.layers[0])
                    if getattr(first.state.layers[0], f.name) is not None}
        resumed = draft.forward(tokens[:, 3:], feedback[:, 3:], first.state)
        outputs, carries, state = [], [], None
        for i in range(6):
            step = draft.forward(tokens[:, i:i+1], feedback[:, i:i+1], state)
            outputs.append(step.logits)
            carries.append(step.feedback)
            state = step.state
    assert torch.equal(torch.cat(outputs, 1), whole.logits)
    assert torch.equal(torch.cat(carries, 1), whole.feedback)
    assert torch.equal(torch.cat((first.logits, resumed.logits), 1), whole.logits)
    assert whole.feedback.shape[-1] == 2 * draft.target.config.text.hidden_size
    for f in fields(state.layers[0]):
        a, b = getattr(state.layers[0], f.name), getattr(whole.state.layers[0], f.name)
        assert (a is None and b is None) or torch.equal(a, b)
    for name, value in snapshot.items():
        assert torch.equal(value, getattr(first.state.layers[0], name))


def test_mtp_feedback_and_position_contracts_fail_explicitly():
    draft = tiny_mtp()
    token = torch.tensor([[1]])
    feedback = torch.randn(1, 1, draft.spec.feedback_width).bfloat16()
    with ReferenceExecutionContext():
        with pytest.raises(ValueError, match="multi-stream"):
            draft.forward(token, feedback[..., :16])
        with pytest.raises(ValueError, match="position axes"):
            draft.forward(token, feedback, position_ids=torch.full((3, 1, 1), -1))
        first = draft.forward(token, feedback, position_ids=torch.zeros(3, 1, 1, dtype=torch.int64))
        with pytest.raises(ValueError, match="explicit axes"):
            draft.forward(token, feedback, first.state)
    text = draft.target.config.text
    with pytest.raises(ValueError, match="feedback or layer"):
        bad = replace(draft.target.config, text=replace(text, mtp=replace(
            text.mtp, hidden_state_source_layer=0)))
        draft.target.config = bad
        Qwen4ExpMTPReference(draft.target, spec=draft.spec)


def test_mtp_consumes_the_shared_input_embedding_boundary():
    draft = tiny_mtp()
    tokens = torch.tensor([[2, 5, 9]])
    feedback = torch.randn(1, 3, draft.spec.feedback_width).bfloat16()
    embeddings = draft.target.operators.embedding(
        tokens, draft.target.weights["model.language_model.embed_tokens.weight"],
        dtype=torch.bfloat16,
    )
    with ReferenceExecutionContext():
        lexical = draft.forward(tokens, feedback)
        supplied = draft.forward(tokens, feedback, input_embeddings=embeddings)
        assert torch.equal(lexical.logits, supplied.logits)
        assert torch.equal(lexical.feedback, supplied.feedback)
        changed = embeddings.clone()
        changed[:, 1] += .25
        visual = draft.forward(tokens, feedback, input_embeddings=changed)
        assert not torch.equal(visual.feedback, lexical.feedback)
        with pytest.raises(ValueError, match="input feature boundary"):
            draft.forward(tokens, feedback, input_embeddings=changed[..., :-1])


@pytest.mark.parametrize("depth", [1, 3])
def test_shared_transaction_with_real_model_math_preserves_every_accepted_state(depth):
    draft = tiny_mtp()
    target = draft.target

    def target_step(tokens, state, positions):
        logits, next_state, feedback = target.forward_with_feedback(
            tokens, state, position_ids=positions)
        return DraftStep(logits, feedback, next_state)

    def equal_state(a, b):
        assert a.length == b.length
        for left, right in zip(a.layers, b.layers, strict=True):
            for field in fields(left):
                x, y = getattr(left, field.name), getattr(right, field.name)
                assert (x is None and y is None) or torch.equal(x, y)

    with ReferenceExecutionContext():
        target_state = draft_state = None
        feedback = torch.zeros(1, 1, draft.spec.feedback_width).bfloat16()
        # Prefix maintenance consumes the token with the preceding target HC
        # feedback. Draft's proposal feedback has an independent lifetime.
        for position, token in enumerate((2, 9, 4)):
            tokens = torch.tensor([[token]])
            axes = torch.full((3, 1, 1), position, dtype=torch.int64)
            maintenance = draft.forward(tokens, feedback, draft_state, position_ids=axes)
            step = target_step(tokens, target_state, axes)
            target_state, draft_state, feedback = step.state, maintenance.state, step.feedback
        frozen_target, frozen_draft = deepcopy(target_state), deepcopy(draft_state)
        for accepted in range(depth + 1):
            result = reference_draft_round(
                provider=draft, target=target_step, root_token=torch.tensor([[6]]),
                feedback=feedback, target_state=target_state, draft_state=draft_state,
                position=3, depth=depth,
                propose=lambda logits, _: int(logits.argmax(-1).item()),
                verify=lambda tokens, q, p, accepted=accepted:
                    SpeculativeSamplingDecision(accepted, 7),
            )
            expected = target_state
            for ordinal, token in enumerate((6, *result.proposal_tokens[:accepted])):
                step = target_step(torch.tensor([[token]]), expected,
                                   torch.full((3, 1, 1), 3 + ordinal, dtype=torch.int64))
                expected = step.state
            equal_state(result.target_state, expected)
            assert result.draft_state.length == result.target_state.length == 4 + accepted
            assert torch.equal(result.feedback, step.feedback)
            equal_state(frozen_target, target_state)
            equal_state(frozen_draft, draft_state)
