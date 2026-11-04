"""Provider-independent draft contracts and independent transaction reference.

MTP is a hidden-conditioned draft provider. The transaction, accepted prefix,
feedback lifetime and verification rule belong to this layer; checkpoint math
belongs to the provider. This reference is never a performance-mode fallback.
"""

from __future__ import annotations

from typing_extensions import Generic, TypeVar

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from cubic_serve.generation.sampler import SamplingParameters
from cubic_serve.generation.tp_sampling import SpeculativeSamplingDecision


@dataclass(frozen=True, slots=True)
class DraftReplayRow:
    """Provider cache ownership supplied by the common transaction planner."""

    request_id: str
    token_id: int
    position: int
    page_destination: int
    page_table: tuple[int, ...]
    state_source_slot: int
    sampling: SamplingParameters
    sample_ordinal: int
    state_destination_slot: int | None = None
    position_axes: tuple[int, int, int] | None = None
    input_feature: bytes = b""
    segment_length: int = 1


class DraftReplayBinding(Protocol):
    @property
    def feedback(self) -> Tensor: ...

    @property
    def state_fields(self) -> Mapping[str, Tensor]: ...

    def bind(self, rows: tuple[DraftReplayRow, ...], feedback: Tensor) -> None: ...

    def replay(self, *, sample: bool = True) -> None: ...

    def reserve_state_slots(self, slots: tuple[int, ...]) -> None: ...


class TransactionalDraftBinding:
    """Reuse the target row/cache binder for an independently owned draft graph.

    The scheduler supplies source/destination checkpoint slots at each proposal
    depth. This class never assigns request identity to graph padding, derives
    ownership from segment starts, or captures a graph at request time.
    """

    def __init__(self, executor) -> None:
        self.executor = executor

    @property
    def feedback(self) -> Tensor:
        return self.executor.body.draft_feedback

    @property
    def state_fields(self) -> Mapping[str, Tensor]:
        return self.executor.body.arena.states

    def replay(self, *, sample: bool = True) -> None:
        self.executor.complete_bound(replay=True, state_only=not sample)

    def reserve_state_slots(self, slots: tuple[int, ...]) -> None:
        """Protect every checkpoint until the complete leased round commits."""
        self.executor._reserved_scratch_states = frozenset(slots)
        # Inline rounds write sparse physical rows without using the host
        # binder. Its cached prefix no longer bounds those live rows: the next
        # ordinary bind must refresh padding across the complete bucket.
        self.executor._bound_active_rows = self.executor.captured_m

    def bind(self, rows: tuple[DraftReplayRow, ...], feedback: Tensor) -> None:
        from cubic_serve.runtime.continuous_batch import BatchRow

        body = self.executor.body
        target = body.buffers.target_hidden
        if target is None or (
            feedback.shape != (len(rows), target.shape[1])
            or feedback.dtype != target.dtype or feedback.device != target.device
        ):
            raise ValueError("draft feedback differs from its bound graph input")
        page_size = body.arena.layout.block_size_tokens
        bound = tuple(BatchRow(
            row.request_id, row.token_id, row.position,
            row.state_source_slot,
            row.state_source_slot if row.state_destination_slot is None
            else row.state_destination_slot,
            row.page_destination, row.page_destination, row.page_table,
            reset_state=row.position == 0,
            new_pages=(row.page_destination,) if row.position % page_size == 0 else (),
            sampling=row.sampling, sample_ordinal=row.sample_ordinal,
            position_axes=row.position_axes, input_feature=row.input_feature,
            segment_length=row.segment_length,
        ) for row in rows)
        self.executor.bind_batch(bound)
        target[:len(rows)].copy_(feedback)
        target[len(rows):].zero_()


@dataclass(frozen=True, slots=True)
class DraftSpec:
    method: str
    vocabulary_size: int
    feedback_width: int
    max_depth: int

    def __post_init__(self) -> None:
        if (
            not self.method
            or any(type(x) is not int for x in (
                self.vocabulary_size, self.feedback_width, self.max_depth
            ))
            or min(self.vocabulary_size, self.max_depth) <= 0
            or self.feedback_width < 0
        ):
            raise ValueError("invalid draft provider capacity contract")


@dataclass(frozen=True, slots=True)
class DraftProvider:
    """Model adapter's construction binding, independent of engine policy."""

    spec: DraftSpec
    build_stage: Callable[..., object]
    build_reference: Callable[..., object]
    reference_columns: tuple[str, ...] = ()


class HiddenConditionedReference:
    """Adapt an existing independent MTP entrypoint without duplicating math."""

    def __init__(self, target, *, spec: DraftSpec):
        self.target, self.spec = target, spec

    def forward(self, tokens, feedback, state=None, *, position_ids=None):
        if position_ids is None:
            start = 0 if state is None else state.sequence_length
            positions = torch.arange(start, start + tokens.shape[1], device=tokens.device)
            position_ids = positions[None, None].expand(3, tokens.shape[0], -1)
        result = self.target.mtp_forward(tokens, feedback, position_ids=position_ids, cache=state)
        return DraftStep(result.logits, result.hidden_states, result.cache)


State = TypeVar("State")
TargetState = TypeVar("TargetState")
ProviderState = TypeVar("ProviderState")


@dataclass(frozen=True, slots=True)
class DraftStep(Generic[State]):
    """Logits and next-step feedback are independent mathematical boundaries."""

    logits: Tensor
    feedback: Tensor | None
    state: State


class DraftModel(Protocol[State]):
    spec: DraftSpec

    def forward(
        self, tokens: Tensor, feedback: Tensor | None, state: State | None = None,
        *, position_ids: Tensor | None = None,
    ) -> DraftStep[State]:
        """Advance immutable input state; return separately owned output state."""
        ...


@dataclass(frozen=True, slots=True)
class DraftRound(Generic[TargetState, ProviderState]):
    proposal_tokens: tuple[int, ...]
    proposal_steps: tuple[DraftStep[ProviderState], ...]
    target_steps: tuple[DraftStep[TargetState], ...]
    decision: SpeculativeSamplingDecision
    output_tokens: tuple[int, ...]
    target_state: TargetState
    draft_state: ProviderState
    feedback: Tensor | None
    catchup: DraftStep[ProviderState] | None


def reference_draft_round(
    *, provider: DraftModel[ProviderState],
    target: Callable[[Tensor, TargetState | None, Tensor], DraftStep[TargetState]],
    root_token: Tensor, feedback: Tensor | None,
    target_state: TargetState | None, draft_state: ProviderState | None,
    position: int, depth: int,
    propose: Callable[[Tensor, int], int],
    verify: Callable[[tuple[int, ...], tuple[Tensor, ...], tuple[Tensor, ...]],
                     SpeculativeSamplingDecision],
) -> DraftRound[TargetState, ProviderState]:
    """Independent single-owner propose/review/repair transaction.

    The root is already sampled and not yet consumed by the target. Target row
    r commits the root plus r accepted proposals; the recovered/bonus token is
    the next unconsumed root. A fully accepted chain needs one draft catch-up
    step using the *target* feedback preceding the final consumed token.
    Sampling/RNG and processor state are supplied explicitly by the caller.
    """
    spec = provider.spec
    if (
        root_token.shape != (1, 1) or root_token.dtype is not torch.int64
        or type(position) is not int or position < 0
        or type(depth) is not int or not 1 <= depth <= spec.max_depth
    ):
        raise ValueError("draft reference round exceeds its declared contract")
    if spec.feedback_width and (
        feedback is None or feedback.shape != (1, 1, spec.feedback_width)
        or feedback.device != root_token.device
    ):
        raise ValueError("draft feedback differs from its provider contract")

    def positions(offset: int) -> Tensor:
        return torch.full((3, 1, 1), position + offset,
                          dtype=torch.int64, device=root_token.device)

    proposals: list[int] = []
    draft_steps: list[DraftStep[ProviderState]] = []
    token, carry, current_draft = root_token, feedback, draft_state
    for ordinal in range(depth):
        step = provider.forward(token, carry, current_draft, position_ids=positions(ordinal))
        proposal = propose(step.logits, ordinal)
        if type(proposal) is not int or not 0 <= proposal < spec.vocabulary_size:
            raise ValueError("draft sampler returned a token outside its vocabulary")
        proposals.append(proposal)
        draft_steps.append(step)
        token = torch.full_like(root_token, proposal)
        carry, current_draft = step.feedback, step.state

    target_steps: list[DraftStep[TargetState]] = []
    current_target = target_state
    for ordinal, token_id in enumerate((int(root_token.item()), *proposals)):
        step = target(torch.full_like(root_token, token_id), current_target, positions(ordinal))
        target_steps.append(step)
        current_target = step.state
    decision = verify(tuple(proposals), tuple(x.logits for x in draft_steps),
                      tuple(x.logits for x in target_steps))
    accepted, next_token = decision.accepted_draft_tokens, decision.next_root_token_id
    if (
        type(accepted) is not int or not 0 <= accepted <= depth
        or type(next_token) is not int or not 0 <= next_token < spec.vocabulary_size
    ):
        raise ValueError("draft verifier returned an invalid prefix decision")
    catchup = None
    if accepted == depth:
        catchup = provider.forward(
            torch.full_like(root_token, proposals[-1]), target_steps[-2].feedback,
            draft_steps[-1].state, position_ids=positions(depth),
        )
        committed_draft = catchup.state
    else:
        committed_draft = draft_steps[accepted].state
    committed = target_steps[accepted]
    return DraftRound(tuple(proposals), tuple(draft_steps), tuple(target_steps), decision,
                      (*proposals[:accepted], next_token), committed.state, committed_draft,
                      committed.feedback, catchup)


@dataclass(frozen=True, slots=True)
class PendingDraft(Generic[State]):
    """One request's next review, with a separately retained root checkpoint.

    ``root_step`` has consumed the pending root; later autoregressive steps
    must not replace that checkpoint. A review aligns its accepted prefix from
    this checkpoint with target feedback before producing the following root.
    """

    root_token_id: int
    position: int
    proposal_tokens: tuple[int, ...]
    proposal_steps: tuple[DraftStep[State], ...]
    root_step: DraftStep[State]
    position_axes: tuple[int, int, int] | None = None
    position_step: tuple[int, int, int] = (1, 1, 1)


@dataclass(frozen=True, slots=True)
class PipelinedDraftRound(Generic[TargetState, ProviderState]):
    target_steps: tuple[DraftStep[TargetState], ...]
    decision: SpeculativeSamplingDecision
    output_tokens: tuple[int, ...]
    target_state: TargetState
    draft_state: ProviderState
    feedback: Tensor | None
    alignment_steps: tuple[DraftStep[ProviderState], ...]
    pending: PendingDraft[ProviderState]


def _reference_propose(
    provider: DraftModel[State], root: DraftStep[State], *, root_token_id: int,
    position: int, depth: int, device: torch.device,
    propose: Callable[[Tensor, int], int],
    position_axes: tuple[int, int, int] | None = None,
    position_step: tuple[int, int, int] = (1, 1, 1),
) -> PendingDraft[State]:
    """The root head supplies q0; only subsequent proposals advance the body."""
    if type(depth) is not int or not 1 <= depth <= provider.spec.max_depth:
        raise ValueError("pending draft depth exceeds provider capacity")
    step = root
    tokens, steps = [], []
    for ordinal in range(depth):
        token = propose(step.logits, ordinal)
        if type(token) is not int or not 0 <= token < provider.spec.vocabulary_size:
            raise ValueError("pending draft token lies outside its vocabulary")
        tokens.append(token)
        steps.append(step)
        if ordinal + 1 < depth:
            step = provider.forward(
                torch.tensor([[token]], dtype=torch.int64, device=device),
                step.feedback, step.state,
                position_ids=_draft_positions(position, position_axes, position_step,
                                              ordinal + 1, device),
            )
    return PendingDraft(root_token_id, position, tuple(tokens), tuple(steps), root,
                        position_axes, position_step)


def _draft_positions(position, axes, step, offset, device) -> Tensor:
    first = (position, position, position) if axes is None else axes
    if (type(first) is not tuple or type(step) is not tuple
        or len(first) != 3 or len(step) != 3
        or any(type(value) is not int or value < 0 for value in (*first, *step))):
        raise ValueError("draft position axes differ from the continuation contract")
    return torch.tensor([base + offset * delta for base, delta in zip(first, step, strict=True)],
                        dtype=torch.int64, device=device).reshape(3, 1, 1)


def reference_pending_draft(
    *, provider: DraftModel[State], root_token: Tensor, feedback: Tensor | None,
    draft_state: State | None, position: int, depth: int,
    propose: Callable[[Tensor, int], int],
    position_axes: tuple[int, int, int] | None = None,
    position_step: tuple[int, int, int] = (1, 1, 1),
) -> PendingDraft[State]:
    """Bootstrap the same carried-proposal algorithm used by whole-round graphs."""
    if (root_token.shape != (1, 1) or root_token.dtype is not torch.int64
        or type(position) is not int or position < 0):
        raise ValueError("pending draft root differs from its contract")
    token_id = int(root_token.item())
    if not 0 <= token_id < provider.spec.vocabulary_size:
        raise ValueError("pending draft root lies outside its vocabulary")
    if provider.spec.feedback_width and (
        feedback is None or feedback.shape != (1, 1, provider.spec.feedback_width)
        or feedback.device != root_token.device
    ):
        raise ValueError("pending draft feedback differs from its contract")
    if type(depth) is not int or not 1 <= depth <= provider.spec.max_depth:
        raise ValueError("pending draft depth exceeds provider capacity")
    root = provider.forward(
        root_token, feedback, draft_state,
        position_ids=_draft_positions(position, position_axes, position_step, 0,
                                     root_token.device),
    )
    return _reference_propose(provider, root, root_token_id=token_id, position=position,
                              depth=depth, device=root_token.device, propose=propose,
                              position_axes=position_axes, position_step=position_step)


def reference_pipelined_draft_round(
    *, provider: DraftModel[ProviderState],
    target: Callable[[Tensor, TargetState | None, Tensor], DraftStep[TargetState]],
    pending: PendingDraft[ProviderState], target_state: TargetState | None,
    device: torch.device, next_depth: int,
    propose: Callable[[Tensor, int], int],
    verify: Callable[[tuple[int, ...], tuple[Tensor, ...], tuple[Tensor, ...]],
                     SpeculativeSamplingDecision],
) -> PipelinedDraftRound[TargetState, ProviderState]:
    """Review, align accepted tokens plus bonus root, then carry the next q/P.

    This independent reference deliberately differs from root-first speculative
    execution: every accepted draft token is realigned with preceding *target*
    feedback, and the bonus root is consumed before returning. The durable draft
    state stops before that bonus; its root-consumed state belongs to ``pending``.
    Neither state is reconstructed from an autoregressive proposal endpoint.
    Sampling callbacks bind their own fixed RNG/ordinal trace. This function is
    pre-readiness/audit execution only, never a serving fallback.
    """
    depth = len(pending.proposal_tokens)
    spec = provider.spec
    if (not 1 <= depth <= spec.max_depth
        or len(pending.proposal_steps) != depth
        or pending.proposal_steps[0] is not pending.root_step
        or type(pending.position) is not int or pending.position < 0
        or any(type(token) is not int or not 0 <= token < spec.vocabulary_size
               for token in (pending.root_token_id, *pending.proposal_tokens))
        or type(next_depth) is not int or not 1 <= next_depth <= spec.max_depth):
        raise ValueError("carried draft differs from its review contract")

    def token(value: int) -> Tensor:
        return torch.tensor([[value]], dtype=torch.int64, device=device)

    def positions(offset: int) -> Tensor:
        return _draft_positions(pending.position, pending.position_axes,
                                pending.position_step, offset, device)

    target_steps: list[DraftStep[TargetState]] = []
    state = target_state
    for offset, value in enumerate((pending.root_token_id, *pending.proposal_tokens)):
        result = target(token(value), state, positions(offset))
        target_steps.append(result)
        state = result.state
    decision = verify(pending.proposal_tokens,
                      tuple(step.logits for step in pending.proposal_steps),
                      tuple(step.logits for step in target_steps))
    accepted, bonus = decision.accepted_draft_tokens, decision.next_root_token_id
    if (type(accepted) is not int or not 0 <= accepted <= depth
        or type(bonus) is not int or not 0 <= bonus < spec.vocabulary_size):
        raise ValueError("carried draft verifier returned an invalid prefix")
    alignment = []
    draft_state = pending.root_step.state
    for offset in range(accepted):
        step = provider.forward(token(pending.proposal_tokens[offset]),
                                target_steps[offset].feedback, draft_state,
                                position_ids=positions(offset + 1))
        alignment.append(step)
        draft_state = step.state
    committed = target_steps[accepted]
    root = provider.forward(token(bonus), committed.feedback, draft_state,
                            position_ids=positions(accepted + 1))
    alignment.append(root)
    following = _reference_propose(
        provider, root, root_token_id=bonus, position=pending.position + accepted + 1,
        depth=next_depth, device=device, propose=propose,
        position_axes=tuple(int(value) for value in positions(accepted + 1).flatten().tolist()),
        position_step=pending.position_step,
    )
    return PipelinedDraftRound(
        tuple(target_steps), decision, (*pending.proposal_tokens[:accepted], bonus),
        committed.state, draft_state, committed.feedback, tuple(alignment), following,
    )
