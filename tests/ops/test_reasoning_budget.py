"""Independent CPU matcher witnesses for the integer sampling budget operands."""

import os
import random

import pytest
import torch

from cubic_serve.contracts.request import ReasoningControl
from cubic_serve.generation.constraints import ConstraintSpec, create_logit_constraint
from cubic_serve.ops.cuda.sampling.reasoning_budget import ReasoningBudgetProgram


pytestmark = pytest.mark.skipif(
    os.environ.get("CUBIC_RUN_REASONING_BUDGET_TESTS") != "1",
    reason="explicit CUDA reasoning-budget audit",
)


def _make(control):
    return create_logit_constraint(spec=ConstraintSpec.identity(), request_id="oracle",
                                   reasoning_control=control)


def _matcher(constraint):
    return constraint._matcher


@pytest.mark.parametrize("boundaries", [((2,), (7,)), ((2, 3), (7, 8), (5, 6)),
                                         ((2, 3, 2), (7, 8, 7), (5, 6, 5)),
                                         ((2, 3), (7, 8), (7, 6))])
@pytest.mark.parametrize("budget", [0, 1, 4, 2**100])
def test_native_ledger_and_graph_capture_match_independent_parser(boundaries, budget):
    device = torch.device("cuda:0")
    mask = torch.empty((4, 1), dtype=torch.int32, device=device)
    activity = torch.empty(4, dtype=torch.bool, device=device)
    program = ReasoningBudgetProgram(boundaries, mask, activity)
    controls = [ReasoningControl(boundaries[0], boundaries[1], budget, active,
                                implicit_ends=boundaries[2:]) for active in (True, False)]
    constraints = [_make(c) for c in controls]
    program.bind([_matcher(c).integer_snapshot() for c in constraints])
    ctx = torch.zeros(1 + 2 * 3, dtype=torch.int32, device=device)
    ctx[0] = 2
    decisions = torch.zeros((2, 2), dtype=torch.int64, device=device)
    prefix = program.prefix(ctx, decisions, owners=2, depth=0, vocabulary=16, head_step=0)
    commit = program.commit(ctx, decisions, owners=2, depth=0, vocabulary=16)
    stream = torch.cuda.Stream(device=device)
    program.stage()
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        prefix.launch()
    rng = random.Random(13)
    for index in range(80):
        graph.replay()
        actual_masks = mask[:2].cpu()
        tokens = []
        for row, constraint in enumerate(constraints):
            m = constraint.mask()
            logits = torch.zeros(16)
            constraint.apply(logits, m)
            allowed = set(torch.isfinite(logits).nonzero().flatten().tolist())
            native = {t for t in range(16) if int(actual_masks[row, 0]) & (1 << t)}
            assert native == allowed
            # Exercise opening, partial markers, mismatches and natural ends.
            pattern = (*boundaries[0], 4, *boundaries[1], 9, *boundaries[0], 4)
            desired = pattern[index % len(pattern)] if index % 3 else rng.randrange(2, 10)
            token = desired if desired in allowed else min(allowed)
            constraint.accept(token, m)
            tokens.append(token)
        decisions[:, 1].copy_(torch.tensor(tokens, device=device))
        commit.launch()
        expected = [_matcher(c).integer_snapshot() for c in constraints]
        actual = program.state[:2].cpu()
        for row, (_control, phase, used, tail) in enumerate(expected):
            assert tuple(actual[row, :4].tolist()) == (
                phase, used, min(budget, 2**63 - 1), len(tail))
            assert tuple(actual[row, 5:5 + len(tail)].tolist()) == tail
    graph.reset()


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("owners", [1, 3, 8])
def test_every_prefix_and_acceptance_boundary_uses_private_rows(depth, owners):
    device = torch.device("cuda:0")
    rows = owners * (depth + 1)
    masks = torch.empty((rows, 2), dtype=torch.int32, device=device)
    activity = torch.empty(rows, dtype=torch.bool, device=device)
    boundaries = ((2, 3), (31, 32), (5, 6))
    program = ReasoningBudgetProgram(boundaries, masks, activity)
    controls = [ReasoningControl(boundaries[0], boundaries[1], r % 4, True,
                                implicit_ends=boundaries[2:]) for r in range(owners)]
    constraints = [_make(c) for c in controls]
    snapshots = [_matcher(c).integer_snapshot() for c in constraints]
    ctx = torch.zeros(1 + owners * (2 + depth), dtype=torch.int32, device=device)
    ctx[0] = owners
    proposals = torch.zeros((depth, owners), dtype=torch.int64, device=device)
    decisions = torch.zeros((owners, 2), dtype=torch.int64, device=device)
    proposal_prefixes = []
    for row, constraint in enumerate(constraints):
        branch = constraint.fork("proposal")
        tokens = []
        for d in range(depth):
            mask = branch.mask()
            logits = torch.zeros(64)
            branch.apply(logits, mask)
            allowed = torch.isfinite(logits)
            token = 9 if allowed[9] else int(allowed.nonzero()[0])
            tokens.append(token); branch.accept(token, mask)
            ctx[1 + row * (2 + depth) + 2 + d] = token
            proposals[d, row] = token
        proposal_prefixes.append(tokens)
    prefix = program.prefix(ctx, proposals, owners=owners, depth=depth, vocabulary=64)
    commit = program.commit(ctx, decisions, owners=owners, depth=depth, vocabulary=64)
    for accepted in range(depth + 1):
        program.bind(snapshots); program.stage(); prefix.launch()
        actual = masks.cpu()
        expected_snapshots = []
        for row, parent in enumerate(constraints):
            branch = parent.fork("target")
            for d in range(depth + 1):
                mask = branch.mask(); logits = torch.zeros(64); branch.apply(logits, mask)
                allowed = set(torch.isfinite(logits).nonzero().flatten().tolist())
                native = {t for t in range(64)
                          if int(actual[row * (depth + 1) + d, t // 32]) & (1 << (t % 32))}
                assert native == allowed
                if d == accepted:
                    bonus = 10 if 10 in allowed else min(allowed)
                    decisions[row] = torch.tensor([accepted, bonus], device=device)
                    committed = parent.fork("commit")
                    for token in (*proposal_prefixes[row][:accepted], bonus):
                        settled = committed.mask()
                        committed.apply(torch.zeros(64), settled)
                        committed.accept(token, settled)
                    expected_snapshots.append(_matcher(committed).integer_snapshot())
                if d < depth:
                    branch.accept(proposal_prefixes[row][d], mask)
        commit.launch()
        actual_state = program.state[:owners].cpu()
        for row, (_c, phase, used, tail) in enumerate(expected_snapshots):
            assert tuple(actual_state[row, :4].tolist()) == (
                phase, used, controls[row].token_budget, len(tail))
            assert tuple(actual_state[row, 5:5 + len(tail)].tolist()) == tail


def test_eos_is_absorbing_only_for_the_owning_request():
    device = torch.device("cuda:0")
    masks = torch.empty((4, 1), dtype=torch.int32, device=device)
    activity = torch.empty(4, dtype=torch.bool, device=device)
    boundaries = ((2,), (7,))
    program = ReasoningBudgetProgram(boundaries, masks, activity, stop_token_ids=(0, 1))
    control = ReasoningControl((2,), (7,), 20, False)
    program.bind([_matcher(_make(control)).integer_snapshot()] * 2,
                 stop_enabled=[True, False])
    ctx = torch.zeros(7, dtype=torch.int32, device=device); ctx[0] = 2
    decisions = torch.tensor([[0, 0], [0, 0]], dtype=torch.int64, device=device)
    prefix = program.prefix(ctx, decisions, owners=2, depth=0, vocabulary=16, head_step=0)
    commit = program.commit(ctx, decisions, owners=2, depth=0, vocabulary=16)
    program.stage(); commit.launch(); prefix.launch()
    actual = masks.cpu()
    assert {t for t in range(16) if int(actual[0, 0]) & (1 << t)} == {0, 1}
    assert int(actual[1, 0]) == -1
