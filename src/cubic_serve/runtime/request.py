"""Mutable request state owned by the continuous scheduler."""

from __future__ import annotations

from dataclasses import dataclass

from cubic_serve.contracts.execution import BatchPhase
from cubic_serve.contracts.request import InputPlan


@dataclass(frozen=True, slots=True)
class InflightSegment:
    """The one plan reservation currently outstanding for a request."""

    plan_id: str
    phase: BatchPhase
    token_start: int
    token_count: int
    target_revision: int


@dataclass(slots=True)
class RequestState:
    """Explicit lifecycle state; no tensor or cache storage lives here."""

    input_plan: InputPlan
    execution_input_digest: str
    max_new_tokens: int
    prefill_cursor: int = 0
    generated_token_ids: tuple[int, ...] = ()
    decode_pending: bool = False
    cancelled: bool = False
    finished: bool = False
    state_revision: int = 0
    inflight: InflightSegment | None = None

    def __post_init__(self) -> None:
        algorithm, separator, digest = self.execution_input_digest.partition(":")
        if algorithm != "b2s256" or separator != ":" or len(digest) != 64:
            raise ValueError("execution input digest must use b2s256:<64 hex chars>")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError("execution input digest contains non-hex characters") from error
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not 0 <= self.prefill_cursor <= len(self.input_plan.token_ids):
            raise ValueError("prefill cursor must lie within the realized input")
        if any(token_id < 0 for token_id in self.generated_token_ids):
            raise ValueError("generated token ids must be non-negative")
        if self.state_revision < 0:
            raise ValueError("state revision must be non-negative")
        if self.cancelled and self.finished:
            raise ValueError("a request cannot be both cancelled and finished")
        if self.decode_pending and (
            self.prefill_cursor != len(self.input_plan.token_ids)
            or not self.generated_token_ids
            or self.cancelled
            or self.finished
        ):
            raise ValueError("decode can only be pending for an active, fully-prefilled request")
        if len(self.generated_token_ids) >= self.max_new_tokens and not self.finished:
            raise ValueError("a request at its output limit must be marked finished")

    @property
    def request_id(self) -> str:
        return self.input_plan.request_id

    @property
    def prompt_token_count(self) -> int:
        return len(self.input_plan.token_ids)

    @property
    def active(self) -> bool:
        return not self.cancelled and not self.finished

    @property
    def schedulable(self) -> bool:
        if not self.active or self.inflight is not None:
            return False
        return self.prefill_cursor < self.prompt_token_count or self.decode_pending
