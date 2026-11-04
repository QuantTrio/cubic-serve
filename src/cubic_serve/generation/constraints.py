"""Fail-closed, request-local structured-generation boundary.

Factories may share compiled grammars; matchers may not share request state.
The core owns ``mask -> apply -> accept``. Optional backends such as xgrammar
only implement compilation and matcher operations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import NoReturn, Protocol, TypeVar, runtime_checkable

import torch

from cubic_serve.contracts.request import (
    ReasoningControl,
    StructuredOutputKind,
    StructuredOutputMeta,
)


class ConstraintReadinessError(RuntimeError):
    """A requested constraint has no ready backend and cannot be downgraded."""


class ConstraintCompilationError(ConstraintReadinessError):
    """A request supplied an invalid or unsupported constraint specification."""


class ConstraintLifecycleError(RuntimeError):
    """A constraint request violated its incremental state machine."""


ConstraintKind = StructuredOutputKind


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    """Generation view over the request's sole canonical structured metadata."""

    meta: StructuredOutputMeta

    def __post_init__(self) -> None:
        if not isinstance(self.meta, StructuredOutputMeta):
            raise TypeError("constraint spec requires StructuredOutputMeta")

    @property
    def kind(self) -> ConstraintKind:
        return self.meta.kind

    @property
    def canonical_payload(self) -> str:
        return self.meta.canonical_payload

    @property
    def digest(self) -> str:
        return self.meta.digest

    @classmethod
    def identity(cls) -> ConstraintSpec:
        return cls(StructuredOutputMeta.identity())

    @classmethod
    def json_schema(cls, schema: object, *, options: object = None) -> ConstraintSpec:
        return cls(StructuredOutputMeta.json_schema(schema, options=options))

    @classmethod
    def regex(cls, source: str) -> ConstraintSpec:
        return cls(StructuredOutputMeta.regex(source))

    @classmethod
    def choice(cls, choices: object) -> ConstraintSpec:
        return cls(StructuredOutputMeta.choice(choices))

    @classmethod
    def structural_tag(cls, value: object) -> ConstraintSpec:
        return cls(StructuredOutputMeta.structural_tag(value))

    @classmethod
    def tool_schema(
        cls,
        schema: object,
        *,
        choice: str = "auto",
        parallel_tool_calls: bool = True,
    ) -> ConstraintSpec:
        return cls(
            StructuredOutputMeta.tool_schema(
                schema,
                choice=choice,
                parallel_tool_calls=parallel_tool_calls,
            )
        )

    @classmethod
    def grammar(
        cls,
        source: str,
        *,
        dialect: str = "xgrammar-ebnf",
        root_rule: str = "root",
    ) -> ConstraintSpec:
        return cls(
            StructuredOutputMeta.grammar(
                source,
                dialect=dialect,
                root_rule=root_rule,
            )
        )


@dataclass(frozen=True, slots=True)
class ConstraintSnapshot:
    """Request-independent state rebuilt by replaying accepted tokens."""

    spec: ConstraintSpec
    accepted_token_ids: tuple[int, ...]
    matcher_terminated: bool
    reasoning_control: ReasoningControl | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ConstraintSpec) or type(self.matcher_terminated) is not bool:
            raise TypeError("constraint snapshot fields have invalid types")
        if any(type(token) is not int or token < 0 for token in self.accepted_token_ids):
            raise ValueError("constraint snapshot tokens must be non-negative integers")
        if self.reasoning_control is not None and not isinstance(
            self.reasoning_control, ReasoningControl
        ):
            raise TypeError("constraint snapshot reasoning control is invalid")


@dataclass(frozen=True, slots=True)
class BackendTokenMask:
    payload: object = field(repr=False, compare=False)
    needs_apply: bool


@dataclass(frozen=True, slots=True)
class ForcedTokenMask:
    """A compact intersection, materialized only at the existing mask boundary."""

    token_id: int
    base: BackendTokenMask


@dataclass(frozen=True, slots=True)
class TokenMask:
    revision: int
    needs_apply: bool
    _payload: object = field(repr=False, compare=False)
    _owner: object = field(repr=False, compare=False)


class ConstraintMatcher(Protocol):
    """Backend adapter for exactly one request."""

    @property
    def constraint_digest(self) -> str: ...

    def fill_token_mask(self) -> BackendTokenMask: ...

    def apply_token_mask(self, logits: torch.Tensor, payload: object) -> None: ...

    def accept_token(self, token_id: int) -> bool: ...

    def is_terminated(self) -> bool: ...

    def fork(self) -> ConstraintMatcher: ...


class ConstraintFactory(Protocol):
    """Shared compiler/cache that creates request-private matchers."""

    @property
    def backend_id(self) -> str: ...

    def require_ready(self, spec: ConstraintSpec) -> None: ...

    def create_matcher(self, spec: ConstraintSpec) -> ConstraintMatcher: ...


@runtime_checkable
class RewindableConstraintMatcher(Protocol):
    """Optional request-private checkpoints; a checkpoint never belongs to a fork."""

    def checkpoint(self) -> object | None: ...

    def restore_checkpoint(self, checkpoint: object) -> None: ...


class LogitConstraint(Protocol):
    @property
    def request_id(self) -> str: ...

    @property
    def spec(self) -> ConstraintSpec: ...

    @property
    def accepted_token_ids(self) -> tuple[int, ...]: ...

    @property
    def finished(self) -> bool: ...

    @property
    def failed(self) -> bool: ...

    def mask(self) -> TokenMask: ...

    def apply(self, logits: torch.Tensor, mask: TokenMask) -> None: ...

    def accept(self, token_id: int, mask: TokenMask) -> None: ...

    def snapshot(self) -> ConstraintSnapshot: ...

    def fork(self, request_id: str) -> LogitConstraint: ...

    def finish(self, *, truncated: bool = False) -> ConstraintSnapshot: ...


class _IdentityMatcher:
    def __init__(self, digest: str) -> None:
        self.constraint_digest = digest

    def fill_token_mask(self) -> BackendTokenMask:
        return BackendTokenMask(None, False)

    def apply_token_mask(self, logits: torch.Tensor, payload: object) -> None:
        raise AssertionError("identity mask is never applied")

    def accept_token(self, token_id: int) -> bool:
        return True

    def is_terminated(self) -> bool:
        return False

    def fork(self) -> ConstraintMatcher:
        return _IdentityMatcher(self.constraint_digest)

    def checkpoint(self) -> object:
        return self

    def restore_checkpoint(self, checkpoint: object) -> None:
        if checkpoint is not self:
            raise ConstraintLifecycleError("identity checkpoint belongs to another matcher")


_T = TypeVar("_T")


class _RequestConstraint:
    def __init__(
        self,
        request_id: str,
        spec: ConstraintSpec,
        matcher: ConstraintMatcher,
        accepted: tuple[int, ...] = (),
        reasoning_control: ReasoningControl | None = None,
    ) -> None:
        self._request_id = _text(request_id, "constraint request id")
        self._spec = spec
        self._reasoning_control = reasoning_control
        if matcher.constraint_digest != spec.digest:
            raise ConstraintReadinessError("constraint matcher identity differs from its spec")
        self._matcher = matcher
        self._accepted = list(accepted)
        self._owner = object()
        self._pending: TokenMask | None = None
        self._applied = self._finished = self._failed = False

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def spec(self) -> ConstraintSpec:
        return self._spec

    @property
    def accepted_token_ids(self) -> tuple[int, ...]:
        return tuple(self._accepted)

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def failed(self) -> bool:
        return self._failed

    def _die(self, message: str) -> NoReturn:
        self._failed = True
        raise ConstraintLifecycleError(message)

    def _active(self) -> None:
        if self._failed or self._finished:
            state = "failed" if self._failed else "finished"
            raise ConstraintLifecycleError(f"constraint session is {state}")

    def _backend(self, action: str, call: Callable[[], _T]) -> _T:
        try:
            return call()
        except Exception as error:
            self._failed = True
            raise ConstraintLifecycleError(f"constraint backend failed to {action}") from error

    def _terminated(self) -> bool:
        value = self._backend("check termination", self._matcher.is_terminated)
        if type(value) is not bool:
            self._die("constraint backend returned invalid termination state")
        return value

    def mask(self) -> TokenMask:
        self._active()
        if self._pending is not None:
            self._die("constraint already has an unconsumed token mask")
        if self._spec.kind is not ConstraintKind.IDENTITY and self._terminated():
            self._die("terminated constraint must finish before another token step")
        backend_mask = self._backend("fill mask", self._matcher.fill_token_mask)
        if not isinstance(backend_mask, BackendTokenMask):
            self._die("constraint backend returned an invalid token mask")
        self._pending = TokenMask(
            len(self._accepted), backend_mask.needs_apply, backend_mask.payload, self._owner
        )
        self._applied = False
        return self._pending

    def _check_mask(self, mask: TokenMask) -> None:
        if not isinstance(mask, TokenMask) or mask._owner is not self._owner:
            self._die("token mask belongs to another constraint session")
        if mask is not self._pending or mask.revision != len(self._accepted):
            self._die("token mask is stale or not pending")

    def apply(self, logits: torch.Tensor, mask: TokenMask) -> None:
        self._active()
        self._check_mask(mask)
        if self._applied:
            self._die("token mask may be applied only once")
        if logits.ndim != 1 or logits.numel() == 0 or not logits.is_floating_point():
            self._die("constraint apply requires non-empty rank-1 floating logits")
        if mask.needs_apply:
            self._backend(
                "apply mask", lambda: self._matcher.apply_token_mask(logits, mask._payload)
            )
        self._applied = True

    def accept(self, token_id: int, mask: TokenMask) -> None:
        self._active()
        self._check_mask(mask)
        if not self._applied:
            self._die("sampled token cannot be accepted before its mask is applied")
        if type(token_id) is not int or token_id < 0:
            self._die("accepted token id must be a non-negative integer")
        accepted = self._backend("accept token", lambda: self._matcher.accept_token(token_id))
        if accepted is not True:
            self._die(f"constraint backend rejected sampled token {token_id}")
        self._accepted.append(token_id)
        self._pending = None
        self._applied = False

    def snapshot(self) -> ConstraintSnapshot:
        if self._failed:
            raise ConstraintLifecycleError("failed constraint state cannot be snapshotted")
        if self._pending is not None:
            self._die("constraint snapshot requires a settled token step")
        return ConstraintSnapshot(
            self._spec, tuple(self._accepted), self._terminated(), self._reasoning_control
        )

    def fork(self, request_id: str) -> LogitConstraint:
        self._active()
        child_id = _text(request_id, "fork request id")
        if child_id == self._request_id or self._pending is not None:
            self._die("constraint fork requires a distinct id and settled token step")
        terminated = self._terminated()
        matcher = self._backend("fork", self._matcher.fork)
        if matcher is self._matcher:
            self._die("constraint backend fork reused request-owned matcher")
        child = _RequestConstraint(
            child_id, self._spec, matcher, tuple(self._accepted), self._reasoning_control
        )
        if child._terminated() != terminated:
            self._die("constraint fork did not preserve matcher state")
        return child

    def finish(self, *, truncated: bool = False) -> ConstraintSnapshot:
        self._active()
        if type(truncated) is not bool:
            self._die("constraint truncation marker must be boolean")
        snapshot = self.snapshot()
        if (
            not truncated
            and self._spec.kind is not ConstraintKind.IDENTITY
            and not snapshot.matcher_terminated
        ):
            self._die("structured output is incomplete at finish")
        self._finished = True
        return snapshot

    def replay(self, snapshot: ConstraintSnapshot) -> None:
        for token in snapshot.accepted_token_ids:
            if self._spec.kind is not ConstraintKind.IDENTITY and self._terminated():
                self._die("constraint replay has tokens after grammar termination")
            accepted = self._backend("replay token", partial(self._matcher.accept_token, token))
            if accepted is not True:
                self._die(f"constraint replay rejected token {token}")
            self._accepted.append(token)
        if self._terminated() != snapshot.matcher_terminated:
            self._die("constraint replay did not reproduce matcher state")


@dataclass(frozen=True, slots=True)
class UnavailableConstraintFactory:
    """Explicit result when an optional backend, for example xgrammar, is absent."""

    backend_id: str
    reason: str

    def require_ready(self, spec: ConstraintSpec) -> None:
        raise ConstraintReadinessError(f"{self.backend_id} is not ready: {self.reason}")

    def create_matcher(self, spec: ConstraintSpec) -> ConstraintMatcher:
        raise ConstraintReadinessError(f"{self.backend_id} is unavailable")


def create_constraint_matcher(
    spec: ConstraintSpec,
    factory: ConstraintFactory | None = None,
) -> ConstraintMatcher:
    if not isinstance(spec, ConstraintSpec):
        raise TypeError("constraint request requires a ConstraintSpec")
    if spec.kind is ConstraintKind.IDENTITY:
        return _IdentityMatcher(spec.digest)
    if factory is None:
        raise ConstraintReadinessError(
            f"{spec.kind.value} constraint has no backend; request was not downgraded to identity"
        )
    backend_id = _text(factory.backend_id, "constraint backend id")
    try:
        factory.require_ready(spec)
        matcher = factory.create_matcher(spec)
        if matcher.constraint_digest != spec.digest:
            raise ConstraintReadinessError("constraint matcher identity differs from its spec")
        return matcher
    except ConstraintCompilationError:
        raise
    except ConstraintReadinessError as error:
        raise ConstraintReadinessError(
            f"{backend_id} is not ready: {error}; request was not downgraded to identity"
        ) from error
    except Exception as error:
        raise ConstraintReadinessError(f"{backend_id} failed constraint admission") from error


def _create(
    spec: ConstraintSpec,
    request_id: str,
    factory: ConstraintFactory | None,
    reasoning_control: ReasoningControl | None = None,
) -> _RequestConstraint:
    matcher = create_constraint_matcher(spec, factory)
    if reasoning_control is not None:
        from cubic_serve.generation.structured_sampling import ReasoningBudgetMatcher

        matcher = ReasoningBudgetMatcher(matcher, reasoning_control)
    return _RequestConstraint(
        request_id, spec, matcher, reasoning_control=reasoning_control
    )


def create_logit_constraint(
    *,
    spec: ConstraintSpec,
    request_id: str,
    factory: ConstraintFactory | None = None,
    reasoning_control: ReasoningControl | None = None,
) -> LogitConstraint:
    """Create explicit identity or reject an unready constrained request."""

    return _create(spec, request_id, factory, reasoning_control)


def constraint_spec_from_meta(meta: StructuredOutputMeta) -> ConstraintSpec:
    """Project frozen request metadata onto the existing generation contract."""

    if not isinstance(meta, StructuredOutputMeta):
        raise TypeError("constraint metadata must be StructuredOutputMeta")
    return ConstraintSpec(meta)


def replay_logit_constraint(
    *,
    snapshot: ConstraintSnapshot,
    request_id: str,
    factory: ConstraintFactory | None = None,
) -> LogitConstraint:
    """Rebuild request-private matcher state from accepted tokens."""

    if not isinstance(snapshot, ConstraintSnapshot):
        raise TypeError("constraint replay requires a ConstraintSnapshot")
    constraint = _create(snapshot.spec, request_id, factory, snapshot.reasoning_control)
    constraint.replay(snapshot)
    return constraint


__all__ = (
    "BackendTokenMask",
    "ConstraintCompilationError",
    "ConstraintFactory",
    "ConstraintKind",
    "ConstraintLifecycleError",
    "ConstraintMatcher",
    "ConstraintReadinessError",
    "ConstraintSnapshot",
    "ConstraintSpec",
    "ForcedTokenMask",
    "LogitConstraint",
    "TokenMask",
    "UnavailableConstraintFactory",
    "constraint_spec_from_meta",
    "create_constraint_matcher",
    "create_logit_constraint",
    "replay_logit_constraint",
)
