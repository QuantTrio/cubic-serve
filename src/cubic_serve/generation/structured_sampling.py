"""Request-private grammar transactions, shared by target and draft sampling.

Target masks follow each reachable proposal prefix. Draft sampling may share
those masks or propose unconstrained tokens; rejection always uses the actual
draft distribution. Durable parser state advances only after output is committed.
"""

from __future__ import annotations

import json
import time
from math import isfinite
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from cubic_serve.contracts.request import (
    CanonicalChatRequest,
    ReasoningControl,
    StructuredOutputKind,
    StructuredOutputMeta,
)
from cubic_serve.generation.constraints import (
    BackendTokenMask,
    ConstraintCompilationError,
    ConstraintFactory,
    ConstraintLifecycleError,
    ConstraintMatcher,
    ConstraintSpec,
    ForcedTokenMask,
    RewindableConstraintMatcher,
    create_constraint_matcher,
)


@dataclass(frozen=True, slots=True)
class ConstraintBinding:
    request_id: str
    meta: StructuredOutputMeta
    revision: int
    remaining: int
    reasoning_end: tuple[int, ...] = ()
    reasoning_control: ReasoningControl | None = None
    stop_token_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (
            type(self.request_id) is not str
            or not self.request_id
            or self.request_id != self.request_id.strip()
            or not isinstance(self.meta, StructuredOutputMeta)
            or (self.meta.kind is StructuredOutputKind.IDENTITY and self.reasoning_control is None)
            or type(self.revision) is not int
            or self.revision < 0
            or type(self.remaining) is not int
            or self.remaining < 1
            or type(self.reasoning_end) is not tuple
            or any(type(t) is not int or t < 0 for t in self.reasoning_end)
        ):
            raise ValueError("invalid structured sampling binding")
        if self.reasoning_control is not None and not isinstance(
            self.reasoning_control, ReasoningControl
        ):
            raise TypeError("invalid reasoning sampling control")
        if self.stop_token_ids is not None and (
            type(self.stop_token_ids) is not tuple
            or any(type(token) is not int or token < 0 for token in self.stop_token_ids)
        ):
            raise ValueError("invalid request stop tokens")


def implicit_reasoning_ends(
    meta: StructuredOutputMeta, source: object
) -> tuple[tuple[int, ...], ...]:
    """The grammar decides which declared implicit boundaries can begin content."""
    return tuple(getattr(source, "reasoning_implicit_end_token_ids", ()))


class DelayedConstraintMatcher:
    """The same explicit reasoning boundary in reference and speculative execution."""

    def __init__(
        self, matcher: ConstraintMatcher, end: tuple[int, ...], stops: tuple[int, ...],
        implicit: tuple[tuple[int, ...], ...] = (),
    ):
        self.base, self.end, self.stops = matcher, end, stops
        self.implicit = implicit
        self.boundaries = tuple(sorted(
            ((end, False), *((t, True) for t in implicit)),
            key=lambda item: len(item[0]), reverse=True,
        ))
        self.tail_capacity = len(self.boundaries[0][0])
        self.reasoning = bool(end)
        self.tail: tuple[int, ...] = ()
        self.early_eos = False
        blocked = []
        for marker in implicit:
            probe = matcher.fork()
            if not all(probe.accept_token(token) for token in marker):
                blocked.append(marker)
        self._blocked_implicit = tuple(blocked)
        self._static_blocks = frozenset(marker[0] for marker in blocked if len(marker) == 1)
        self._reasoning_full_mask = self._reasoning_static_mask = None
        if blocked:
            if any(len(marker) > len(end) and marker[-len(end):] == end for marker in blocked):
                raise ConstraintCompilationError("reasoning boundaries conflict with the grammar")
            payload = matcher.fill_token_mask().payload
            if not isinstance(payload, torch.Tensor) or payload.dtype != torch.int32:
                raise ConstraintLifecycleError("grammar returned an invalid CPU mask ABI")
            self._reasoning_full_mask = torch.full_like(payload, -1)
            self._reasoning_static_mask = self._reasoning_full_mask.clone()
            self._clear_tokens(self._reasoning_static_mask, self._static_blocks)

    @staticmethod
    def _clear_tokens(mask: torch.Tensor, tokens) -> None:
        for token in tokens:
            word = (int(mask[0, token // 32]) & 0xFFFFFFFF) & ~(1 << (token % 32))
            mask[0, token // 32] = word if word < 2**31 else word - 2**32

    @property
    def constraint_digest(self) -> str:
        return self.base.constraint_digest

    def fill_token_mask(self) -> BackendTokenMask:
        if not self.reasoning:
            return self.base.fill_token_mask()
        blocked = frozenset(
            marker[-1] for marker in self._blocked_implicit
            if (len(marker) == 1 or self.tail[-len(marker) + 1:] == marker[:-1])
            and (*self.tail, marker[-1])[-len(self.end):] != self.end
        )
        if not blocked:
            return BackendTokenMask(None, False)
        if blocked == self._static_blocks:
            return BackendTokenMask(self._reasoning_static_mask, True)
        mask = self._reasoning_full_mask.clone()
        self._clear_tokens(mask, blocked)
        return BackendTokenMask(mask, True)

    @property
    def batch_mask_source(self):
        return None if self.reasoning else getattr(self.base, "batch_mask_source", None)

    def apply_token_mask(self, logits: torch.Tensor, payload: object) -> None:
        self.base.apply_token_mask(logits, payload)

    def accept_token(self, token: int) -> bool:
        if not self.reasoning:
            return self.base.accept_token(token)
        self.tail = (*self.tail, token)[-self.tail_capacity :]
        for marker, retain in self.boundaries:
            if self.tail[-len(marker):] == marker:
                self.reasoning, self.tail = False, ()
                if retain:
                    for prefix_token in marker:
                        if not self.base.accept_token(prefix_token):
                            return False
                break
        if token in self.stops:
            self.early_eos = True
        return True

    def is_terminated(self) -> bool:
        return self.early_eos or (not self.reasoning and self.base.is_terminated())

    def fork(self) -> DelayedConstraintMatcher:
        from copy import copy

        child = copy(self)
        child.base = self.base.fork()
        return child

    def checkpoint(self) -> object | None:
        if not isinstance(self.base, RewindableConstraintMatcher):
            return None
        base = self.base.checkpoint()
        return (None if base is None else
                (self, base, self.reasoning, self.tail, self.early_eos))

    def restore_checkpoint(self, checkpoint: object) -> None:
        if type(checkpoint) is not tuple or len(checkpoint) != 5 or checkpoint[0] is not self:
            raise ConstraintLifecycleError("reasoning checkpoint belongs to another matcher")
        self.base.restore_checkpoint(checkpoint[1])
        self.reasoning, self.tail, self.early_eos = checkpoint[2:]


@dataclass(frozen=True, slots=True)
class DelayedConstraintFactory:
    base: ConstraintFactory
    end: tuple[int, ...]
    stops: tuple[int, ...]
    implicit: tuple[tuple[int, ...], ...] = ()

    @property
    def backend_id(self) -> str:
        return self.base.backend_id + ":after-reasoning"

    def require_ready(self, spec: ConstraintSpec) -> None:
        self.base.require_ready(spec)

    def create_matcher(self, spec: ConstraintSpec) -> ConstraintMatcher:
        return DelayedConstraintMatcher(self.base.create_matcher(spec), self.end, self.stops, self.implicit)


def reasoning_control_from_prompt(
    request: CanonicalChatRequest, codec: object, prompt: tuple[int, ...]
) -> ReasoningControl | None:
    """Seed a budget from the actual open thought, not a model or context-size guess."""
    if request.thinking_token_budget is None:
        return None
    start = tuple(getattr(codec, "reasoning_start_token_ids", ()))
    end = tuple(getattr(codec, "reasoning_end_token_ids", ()))
    implicit = implicit_reasoning_ends(request.structured_output, codec)
    if not start or not end:
        raise ValueError("thinking_token_budget requires declared reasoning token boundaries")

    def last(marker: tuple[int, ...]) -> int:
        return next((index for index in range(len(prompt) - len(marker), -1, -1)
                     if prompt[index:index + len(marker)] == marker), -1)

    begin = last(start)
    closed = max(last(marker) for marker in (end, *implicit))
    active = begin > closed
    used = len(prompt) - begin - len(start) if active else 0
    tail: tuple[int, ...] = ()
    if active:
        for marker in (end, *implicit):
            for count in range(1, min(len(marker), used + 1)):
                if prompt[-count:] == marker[:count] and count > len(tail):
                    tail = marker[:count]
    return ReasoningControl(start, end, request.thinking_token_budget, active, used, implicit, tail)


def compact_json_constraint(meta: StructuredOutputMeta) -> StructuredOutputMeta:
    """Apply the startup JSON policy to canonical metadata on every rank."""
    if meta.kind is StructuredOutputKind.JSON_SCHEMA:
        return StructuredOutputMeta.json_schema(
            json.loads(meta.canonical_payload),
            options={**json.loads(meta.canonical_options), "disable_any_whitespace": True},
        )
    if meta.kind is StructuredOutputKind.TOOL_SCHEMA:
        payload = json.loads(meta.canonical_payload)
        response = payload.get("response")
        if response and response["kind"] == StructuredOutputKind.JSON_SCHEMA.value:
            nested = compact_json_constraint(StructuredOutputMeta(
                StructuredOutputKind.JSON_SCHEMA, response["payload"], response["options"]
            ))
            return StructuredOutputMeta.tool_schema(
                payload["tools"], choice=payload["choice"],
                parallel_tool_calls=payload.get("parallel_tool_calls", True),
                format=payload.get("format", "native"), response=nested,
            )
    return meta


class ReasoningBudgetMatcher:
    """Budget restriction over the existing matcher, forked with each proposal prefix.

    The output phase parser is reused as a token-only ledger. No text is decoded,
    no model math changes, and the forced mask stays compact until the established
    reference/graph mask boundary consumes it.
    """

    def __init__(self, base: ConstraintMatcher, control: ReasoningControl) -> None:
        if not isinstance(control, ReasoningControl):
            raise TypeError("reasoning matcher requires ReasoningControl")
        self.base, self.control = base, control
        self._seed = control.initial_reasoning_tokens
        self._start_tail: tuple[int, ...] = ()
        self._phase = self._new_phase(control.initial_tail) if control.initially_reasoning else None
        token = self._forced_token()
        if token is not None:
            pending = tuple(value for value, _ in self._phase.pending)
            offset = len(pending) if pending == control.end[:len(pending)] else 0
            try:
                probe = self.base.fork()
                allowed = all(probe.accept_token(value) for value in control.end[offset:])
            except Exception as error:
                raise ConstraintCompilationError(
                    "reasoning budget conflicts with the active grammar"
                ) from error
            if not allowed:
                raise ConstraintCompilationError(
                    "reasoning budget conflicts with the active grammar"
                )

    @staticmethod
    def _grammar_allows(mask: BackendTokenMask, token: int) -> bool:
        if not mask.needs_apply:
            return True
        payload = mask.payload
        if (
            not isinstance(payload, torch.Tensor) or payload.device.type != "cpu"
            or payload.dtype != torch.int32 or payload.ndim != 2 or payload.shape[0] != 1
            or not 0 <= token < payload.numel() * 32
        ):
            raise ConstraintLifecycleError("grammar returned an invalid CPU mask ABI")
        return bool(int(payload[0, token // 32]) & (1 << (token % 32)))

    def _new_phase(self, prefix: tuple[int, ...] = ()):
        from cubic_serve.generation.output import TokenPhaseOutputParser

        return TokenPhaseOutputParser(
            None, reasoning_start=self.control.start if not prefix else (),
            reasoning_end=self.control.end, reasoning_implicit_end=self.control.implicit_ends,
            reasoning_prefix=prefix,
        )

    @property
    def constraint_digest(self) -> str:
        return self.base.constraint_digest

    def integer_snapshot(self):
        """Encode the current rule state without advancing any proposal branch."""
        phase = self._phase
        if phase is None:
            return self.control, 0, 0, self._start_tail
        return (
            self.control, 1 if phase.phase == "reasoning_start" else 2,
            self._seed + phase.reasoning_token_count + phase.pending_reasoning_token_count,
            tuple(token for token, _ in phase.pending),
        )

    @property
    def reasoning(self) -> bool:
        return bool(getattr(self.base, "reasoning", False))

    def _forced_token(self) -> int | None:
        if self._phase is None:
            return None
        phase = self._phase
        pending = tuple(token for token, _ in phase.pending)
        unconfirmed = phase.pending_reasoning_token_count
        if self._seed + phase.reasoning_token_count + unconfirmed < self.control.token_budget:
            return None
        # A partial natural end must be completed, rather than restarted.
        end = self.control.end
        offset = len(pending) if pending and pending == end[:len(pending)] else 0
        return end[offset]

    def fill_token_mask(self) -> BackendTokenMask:
        mask = self.base.fill_token_mask()
        token = self._forced_token()
        return mask if token is None else BackendTokenMask(ForcedTokenMask(token, mask), True)

    @property
    def batch_mask_source(self):
        # A compact forced boundary is a wrapper rule, not the base grammar.
        return (None if self._forced_token() is not None
                else getattr(self.base, "batch_mask_source", None))

    def apply_token_mask(self, logits: torch.Tensor, payload: object) -> None:
        if not isinstance(payload, ForcedTokenMask):
            self.base.apply_token_mask(logits, payload)
            return
        if not 0 <= payload.token_id < logits.numel():
            raise ConstraintLifecycleError("reasoning end exceeds the sampler vocabulary")
        if payload.base.needs_apply:
            self.base.apply_token_mask(logits, payload.base.payload)
        value = logits[payload.token_id].clone()
        if not bool(torch.isfinite(value)):
            raise ConstraintLifecycleError("reasoning budget conflicts with the active grammar")
        logits.fill_(-float("inf"))
        logits[payload.token_id] = value

    def accept_token(self, token: int) -> bool:
        forced = self._forced_token()
        if forced is not None and token != forced:
            return False
        if self.base.accept_token(token) is not True:
            return False
        if self._phase is not None:
            self._phase.push(token, "")
            if self._phase.phase == "answer":
                self._phase = None
                self._seed = 0
                self._start_tail = ()
        else:
            self._start_tail = (*self._start_tail, token)[-len(self.control.start):]
            if self._start_tail == self.control.start:
                self._phase = self._new_phase()
                # The generated start sequence was already observed in waiting state.
                self._phase.phase = "reasoning"
                self._seed = 0
                self._start_tail = ()
        return True

    def is_terminated(self) -> bool:
        return self.base.is_terminated()

    def fork(self) -> ReasoningBudgetMatcher:
        from copy import copy

        child = copy(self)
        child.base = self.base.fork()
        if self._phase is not None:
            child._phase = copy(self._phase)
            child._phase.pending = list(self._phase.pending)
        return child

    def checkpoint(self) -> object | None:
        from copy import copy

        if not isinstance(self.base, RewindableConstraintMatcher):
            return None
        base = self.base.checkpoint()
        if base is None:
            return None
        phase = copy(self._phase)
        if phase is not None:
            phase.pending = list(phase.pending)
        return self, base, self._seed, self._start_tail, phase

    def restore_checkpoint(self, checkpoint: object) -> None:
        from copy import copy

        if type(checkpoint) is not tuple or len(checkpoint) != 5 or checkpoint[0] is not self:
            raise ConstraintLifecycleError("budget checkpoint belongs to another matcher")
        self.base.restore_checkpoint(checkpoint[1])
        self._seed, self._start_tail = checkpoint[2:4]
        self._phase = copy(checkpoint[4])
        if self._phase is not None:
            self._phase.pending = list(self._phase.pending)


class _Session:
    def __init__(
        self,
        binding: ConstraintBinding,
        matcher: ConstraintMatcher,
        stop_ids: tuple[int, ...],
        stop_mask: torch.Tensor,
    ) -> None:
        self.binding, self.matcher = binding, matcher
        self.stop_ids, self.stop_mask = stop_ids, stop_mask
        self.revision = 0
        self.terminated = False
        self._rewindable = matcher if isinstance(matcher, RewindableConstraintMatcher) else None

    @property
    def reasoning(self) -> bool:
        return bool(getattr(self.matcher, "reasoning", False))

    def fork(self) -> _Session:
        child = _Session(self.binding, self.matcher.fork(), self.stop_ids, self.stop_mask)
        child.revision, child.terminated = self.revision, self.terminated
        return child

    def checkpoint(self) -> object | None:
        native = None if self._rewindable is None else self._rewindable.checkpoint()
        return None if native is None else (self, self.revision, self.terminated, native)

    def restore_checkpoint(self, checkpoint: object) -> None:
        if (type(checkpoint) is not tuple or len(checkpoint) != 4
                or checkpoint[0] is not self or self._rewindable is None
                or type(checkpoint[1]) is not int or not 0 <= checkpoint[1] <= self.revision
                or type(checkpoint[2]) is not bool):
            raise ConstraintLifecycleError("session checkpoint differs from its private prefix")
        self._rewindable.restore_checkpoint(checkpoint[3])
        self.revision, self.terminated = checkpoint[1:3]

    @property
    def batch_mask_source(self):
        return None if self.terminated else getattr(self.matcher, "batch_mask_source", None)

    def mask(self, prepared=None) -> torch.Tensor | None:
        if prepared is not None and (prepared[0] is None
                                     or prepared[0] is not self.batch_mask_source):
            raise ConstraintLifecycleError("prepared grammar mask changed its private prefix owner")
        if self.terminated:
            # Speculative lookahead beyond EOS is an absorbing private state.
            # Durable commit stops at the first terminator, as does publication.
            return self.stop_mask
        result = self.matcher.fill_token_mask() if prepared is None else prepared[1]
        if not result.needs_apply:
            return None
        if isinstance(result.payload, ForcedTokenMask):
            token, base = result.payload.token_id, result.payload.base
            if not 0 <= token < self.stop_mask.numel() * 32:
                raise ConstraintLifecycleError("reasoning end exceeds the grammar vocabulary")
            if not ReasoningBudgetMatcher._grammar_allows(base, token):
                raise ConstraintLifecycleError("reasoning budget conflicts with the active grammar")
            mask = torch.zeros_like(self.stop_mask)
            word = 1 << (token % 32)
            mask[token // 32] = word if word < 2**31 else word - 2**32
            return mask
        if (
            not isinstance(result.payload, torch.Tensor)
            or result.payload.device.type != "cpu"
            or result.payload.dtype != torch.int32
            or result.payload.shape != (1, self.stop_mask.numel())
        ):
            raise ConstraintLifecycleError("grammar returned an invalid CPU mask ABI")
        return result.payload[0].clone()

    def accept(self, token: int) -> None:
        if type(token) is not int or token < 0:
            raise ConstraintLifecycleError("grammar requires an integer token")
        if self.terminated:
            if token not in self.stop_ids:
                raise ConstraintLifecycleError("nonterminal token follows grammar EOS")
        else:
            if self.matcher.accept_token(token) is not True:
                raise ConstraintLifecycleError("grammar rejected sampled token")
            self.terminated = self.matcher.is_terminated() or token in self.stop_ids
        self.revision += 1


class GrammarPool:
    """One worker's durable request states; no model/GPU/topology-specific rules."""

    def __init__(
        self, factory: ConstraintFactory | None, *, vocab_size: int, stop_token_ids: tuple[int, ...],
        batch_mask_workers: int = 1,
    ) -> None:
        if (
            type(vocab_size) is not int
            or vocab_size < 1 or type(batch_mask_workers) is not int or batch_mask_workers < 1
            or not stop_token_ids
            or any(type(t) is not int or not 0 <= t < vocab_size for t in stop_token_ids)
        ):
            raise ValueError("invalid grammar vocabulary or stop tokens")
        self.factory, self.vocab_size = factory, vocab_size
        self.batch_mask_workers = batch_mask_workers
        self.stop_ids = stop_token_ids
        self.stop_mask = torch.zeros((vocab_size + 31) // 32, dtype=torch.int32, device="cpu")
        for token in stop_token_ids:
            value = (int(self.stop_mask[token // 32]) & 0xFFFFFFFF) | (1 << (token % 32))
            self.stop_mask[token // 32] = value if value < 2**31 else value - 2**32
        self.sessions: dict[str, _Session] = {}
        # An old durable state becomes private only after its successor commits.
        # It trails that successor by exactly one immutable output journal.
        self._standby: dict[str, tuple[_Session, tuple[int, ...], _Session]] = {}

    def release(self, request_ids: tuple[str, ...]) -> None:
        for request_id in request_ids:
            self.sessions.pop(request_id, None)
            self._standby.pop(request_id, None)

    def begin(self, bindings: tuple[ConstraintBinding, ...]) -> GrammarRound:
        return GrammarRound(self, bindings)


@dataclass(frozen=True, slots=True)
class GrammarRoundCost:
    """CPU work witness; fork_ms includes private checkpoint restoration."""

    owners: tuple[tuple[str, int], ...]
    begin_ms: float
    mask_ms: float
    first_mask_ms: float
    advance_ms: float
    fork_ms: float
    accept_ms: float
    publish_ms: float
    mask_calls: int
    output_tokens: int

    def valid(self) -> bool:
        return (
            type(self.owners) is tuple and bool(self.owners)
            and all(type(v) is tuple and len(v) == 2 and type(v[0]) is str
                    and bool(v[0]) and type(v[1]) is int and v[1] >= 0 for v in self.owners)
            and len({v[0] for v in self.owners}) == len(self.owners)
            and all(type(v) in (int, float) and isfinite(v) and v >= 0 for v in (
                self.begin_ms, self.mask_ms, self.first_mask_ms, self.advance_ms,
                self.fork_ms, self.accept_ms, self.publish_ms,
            ))
            and self.first_mask_ms <= self.mask_ms
            and type(self.mask_calls) is int and self.mask_calls >= 0
            and type(self.output_tokens) is int and self.output_tokens >= len(self.owners)
        )


class GrammarRound:
    def __init__(self, pool: GrammarPool, bindings: tuple[ConstraintBinding, ...]) -> None:
        started = time.monotonic_ns()
        self._mask_ns = self._first_mask_ns = self._advance_ns = 0
        self._fork_ns = self._accept_ns = self._publish_ns = 0
        self._mask_calls = self._output_tokens = 0
        if len({b.request_id for b in bindings}) != len(bindings):
            raise ConstraintLifecycleError("duplicate request in grammar round")
        self.pool = pool
        self.bindings = {b.request_id: b for b in bindings}
        self.branches: dict[str, _Session] = {}
        self.masks: dict[str, list[torch.Tensor | None]] = {}
        self._checkpoints: dict[str, object | None] = {}
        self._proposals: dict[str, list[int]] = {}
        self.closed = False
        self._committed = False
        for binding in bindings:
            state = pool.sessions.get(binding.request_id)
            if state is None:
                if binding.revision != 0:
                    raise ConstraintLifecycleError("grammar request lost its durable prefix")
                spec = ConstraintSpec(binding.meta)
                if binding.reasoning_control is not None and any(
                    token >= pool.vocab_size
                    for marker in (binding.reasoning_control.start, binding.reasoning_control.end,
                                   *binding.reasoning_control.implicit_ends)
                    for token in marker
                ):
                    raise ConstraintLifecycleError("reasoning boundary exceeds the vocabulary")
                matcher = create_constraint_matcher(spec, pool.factory)
                if binding.reasoning_end and binding.meta.kind is not StructuredOutputKind.IDENTITY:
                    matcher = DelayedConstraintMatcher(
                        matcher, binding.reasoning_end, pool.stop_ids,
                        implicit_reasoning_ends(binding.meta, pool.factory),
                    )
                if binding.reasoning_control is not None:
                    matcher = ReasoningBudgetMatcher(matcher, binding.reasoning_control)
                stop_ids = (
                    pool.stop_ids if binding.stop_token_ids is None else binding.stop_token_ids
                )
                state = _Session(binding, matcher, stop_ids, pool.stop_mask)
                pool.sessions[binding.request_id] = state
            if (
                state.revision != binding.revision
                or state.binding.meta != binding.meta
                or state.binding.reasoning_end != binding.reasoning_end
                or state.binding.reasoning_control != binding.reasoning_control
                or state.binding.stop_token_ids != binding.stop_token_ids
            ):
                raise ConstraintLifecycleError("grammar binding differs from its committed prefix")
            standby = pool._standby.pop(binding.request_id, None)
            if standby is None:
                branch = state.fork()
            else:
                branch, journal, successor = standby
                if (successor is not state
                        or branch.binding != state.binding
                        or branch.matcher.constraint_digest != state.matcher.constraint_digest
                        or branch.revision + len(journal) != state.revision):
                    raise ConstraintLifecycleError("private grammar state lost its committed successor")
                for token in journal:
                    branch.accept(token)
                if branch.revision != state.revision or branch.terminated != state.terminated:
                    raise ConstraintLifecycleError("private grammar state differs from its durable prefix")
            self.branches[binding.request_id] = branch
            self._checkpoints[binding.request_id] = branch.checkpoint()
            self.masks[binding.request_id] = []
            self._proposals[binding.request_id] = []
        self._begin_ns = time.monotonic_ns() - started

    def cost_witness(self) -> GrammarRoundCost:
        if not self._committed:
            raise ConstraintLifecycleError("grammar cost precedes durable commit")
        return GrammarRoundCost(
            tuple((b.request_id, b.revision) for b in self.bindings.values()),
            *(v / 1_000_000 for v in (
                self._begin_ns, self._mask_ns, self._first_mask_ns,
                self._advance_ns, self._fork_ns, self._accept_ns, self._publish_ns,
            )), self._mask_calls, self._output_tokens,
        )

    def mask(self, request_id: str, depth: int, *, _prepared=None) -> torch.Tensor | None:
        if self.closed or request_id not in self.bindings:
            raise ConstraintLifecycleError("grammar round is closed or has no such request")
        history = self.masks[request_id]
        if type(depth) is not int or not 0 <= depth <= len(history):
            raise ConstraintLifecycleError("grammar mask depth is not contiguous")
        if depth == len(history):
            branch = self.branches[request_id]
            if branch.revision != self.bindings[request_id].revision + depth:
                raise ConstraintLifecycleError("grammar mask precedes its proposal prefix")
            started = time.monotonic_ns()
            history.append(branch.mask(_prepared))
            elapsed = time.monotonic_ns() - started
            self._mask_ns += elapsed
            self._first_mask_ns += elapsed if depth == 0 else 0
            self._mask_calls += 1
        return history[depth]

    def propose(self, request_id: str, token: int, depth: int) -> None:
        if (
            self.closed
            or request_id not in self.bindings
            or len(self.masks[request_id]) != depth + 1
        ):
            raise ConstraintLifecycleError("proposal requires its current grammar mask")
        branch = self.branches[request_id]
        if branch.revision != self.bindings[request_id].revision + depth:
            raise ConstraintLifecycleError("grammar proposal was already consumed")
        if type(token) is not int or not 0 <= token < self.pool.vocab_size:
            raise ConstraintLifecycleError("proposal token exceeds the grammar vocabulary")
        started = time.monotonic_ns()
        branch.accept(token)
        self._proposals[request_id].append(token)
        self._advance_ns += time.monotonic_ns() - started

    def target_prefix_masks(
        self, request_id: str, proposals: tuple[int, ...],
    ) -> tuple[tuple[torch.Tensor | None, ...], int]:
        """Mask only reachable p prefixes; an invalid q is an ordinary rejection.

        The first disallowed q has exactly zero target mass. Later p rows are
        unreachable after prefix merge and use permissive private masks. Keep
        the valid-prefix length as a host witness before any durable commit.
        """
        masks = []
        for depth, token in enumerate(proposals):
            mask = self.mask(request_id, depth)
            masks.append(mask)
            if type(token) is not int or not 0 <= token < self.pool.vocab_size:
                raise ConstraintLifecycleError("proposal token exceeds the grammar vocabulary")
            if mask is not None and not (int(mask[token // 32]) & (1 << (token % 32))):
                return tuple((*masks, *((None,) * (len(proposals) - depth)))), depth
            self.propose(request_id, token, depth)
        masks.append(self.mask(request_id, len(proposals)))
        return tuple(masks), len(proposals)

    def target_prefix_masks_many(self, proposals, *, workers=1):
        """Independent owners commute; each owner's provisional prefix is ordered."""
        if type(workers) is not int or not 1 <= workers <= self.pool.batch_mask_workers:
            raise ValueError("grammar cohort exceeds its CPU worker budget")
        if self.closed:
            raise ConstraintLifecycleError("grammar round is closed")
        if not proposals:
            return {}
        if any(
            request not in self.bindings or type(tokens) is not tuple
            or any(type(token) is not int or not 0 <= token < self.pool.vocab_size
                   for token in tokens) for request, tokens in proposals.items()
        ):
            raise ConstraintLifecycleError("target grammar prefixes differ from their request binding")
        masks = {request: [None] * (len(tokens) + 1) for request, tokens in proposals.items()}
        valid = {request: len(tokens) for request, tokens in proposals.items()}
        active = set(proposals)
        for depth in range(max(map(len, proposals.values())) + 1):
            owned = tuple(request for request, tokens in proposals.items()
                          if request in active and depth <= len(tokens))
            groups, prepared = {}, {}
            if workers > 1:
                for request in owned:
                    branch = self.branches[request]
                    if (len(self.masks[request]) != depth
                            or branch.revision != self.bindings[request].revision + depth):
                        raise ConstraintLifecycleError("grammar cohort precedes its proposal prefix")
                    source = branch.batch_mask_source
                    if source is not None:
                        groups.setdefault(source.batch_mask_key, []).append((request, source))
                for group in groups.values():
                    if len(group) < workers:
                        continue
                    sources = tuple(source for _, source in group)
                    started = time.monotonic_ns()
                    results = sources[0].fill_mask_cohort(sources, workers)
                    elapsed = time.monotonic_ns() - started
                    self._mask_ns += elapsed
                    self._first_mask_ns += elapsed if depth == 0 else 0
                    for (request, source), result in zip(group, results, strict=True):
                        prepared[request] = source, result
            for request in owned:
                mask = self.mask(request, depth, _prepared=prepared.get(request))
                masks[request][depth] = mask
                if depth == len(proposals[request]):
                    continue
                token = proposals[request][depth]
                if mask is not None and not (int(mask[token // 32]) & (1 << (token % 32))):
                    valid[request] = depth
                    active.remove(request)
                else:
                    self.propose(request, token, depth)
        return {request: (tuple(masks[request]), valid[request]) for request in proposals}

    def fork_after(
        self, outputs: Mapping[str, tuple[int, ...]], *, _reuse_private: bool = False,
    ) -> dict[str, _Session]:
        """Prepare the accepted prefix privately; durable state is still unchanged."""
        if self.closed or set(outputs) != set(self.bindings):
            raise ConstraintLifecycleError("grammar commit differs from its request ownership")
        updates = {}
        for request_id, tokens in outputs.items():
            binding = self.bindings[request_id]
            state = self.pool.sessions.get(request_id)
            if state is None or state.revision != binding.revision:
                raise ConstraintLifecycleError("grammar commit is stale")
            if not tokens or type(tokens) is not tuple:
                raise ConstraintLifecycleError("grammar commit requires immutable output tokens")
            started = time.monotonic_ns()
            checkpoint = self._checkpoints[request_id] if _reuse_private else None
            if checkpoint is None:
                child = state.fork()
            else:
                child = self.branches[request_id]
                child.restore_checkpoint(checkpoint)
            self._fork_ns += time.monotonic_ns() - started
            started = time.monotonic_ns()
            for token in tokens[: binding.remaining]:
                if type(token) is not int or not 0 <= token < self.pool.vocab_size:
                    raise ConstraintLifecycleError("committed token exceeds the grammar vocabulary")
                child.accept(token)
                if token in child.stop_ids:
                    break
            updates[request_id] = child
            self._accept_ns += time.monotonic_ns() - started
        return updates

    def commit(self, outputs: Mapping[str, tuple[int, ...]]) -> None:
        if self.closed:
            raise ConstraintLifecycleError("grammar round was already committed or closed")
        try:
            children = self.fork_after(outputs, _reuse_private=True)
        except BaseException:
            # Restore the original private proposals as well as leaving every
            # durable owner untouched. Validation failure remains retryable.
            try:
                for request_id, checkpoint in self._checkpoints.items():
                    if checkpoint is not None:
                        branch = self.branches[request_id]
                        branch.restore_checkpoint(checkpoint)
                        for token in self._proposals[request_id]:
                            branch.accept(token)
            except BaseException:
                self.closed = True
                raise
            raise
        started = time.monotonic_ns()
        updates = {}
        for request_id, child in children.items():
            binding = self.bindings[request_id]
            tokens = outputs[request_id]
            self._output_tokens += child.revision - binding.revision
            updates[request_id] = (
                None
                if child.terminated or tokens[-1] in child.stop_ids
                or child.revision - binding.revision >= binding.remaining
                else child
            )
        for request_id, state in updates.items():
            if state is None:
                self.pool.sessions.pop(request_id)
                self.pool._standby.pop(request_id, None)
            else:
                previous = self.pool.sessions[request_id]
                if self._checkpoints[request_id] is not None:
                    applied = outputs[request_id][:state.revision - self.bindings[request_id].revision]
                    self.pool._standby[request_id] = previous, applied, state
                self.pool.sessions[request_id] = state
        self.closed = True
        self._committed = True
        self._publish_ns += time.monotonic_ns() - started
