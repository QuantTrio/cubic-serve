"""Model-program ABI kept independent from cache and operator implementations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

from cubic_serve.contracts.execution import BatchPhase, OpSignature, PlanSegment
from cubic_serve.contracts.request import InputPlan, TokenPosition
from cubic_serve.inputs import EncodedFeatureBundle, FeatureReader


def _require_content_digest(value: str, field: str) -> None:
    algorithm, separator, digest = value.partition(":")
    if algorithm != "b2s256" or separator != ":" or len(digest) != 64:
        raise ValueError(f"{field} must use b2s256:<64 hex chars>")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError(f"{field} contains non-hexadecimal characters") from error


@dataclass(frozen=True, slots=True)
class ProgramExecutionRecipe:
    """Explicit stable choices and source artifacts behind one program instance.

    The adapter is trusted code and must expose every environment-dependent
    semantic branch here. ``artifact_digests`` are trusted declarations for
    helper sources outside the path-loaded adapter file: core validates their
    canonical digest syntax and binds them into replay/prefix identities, but
    does not independently discover or attest those external artifacts.
    """

    recipe_id: str
    revision: str
    settings: tuple[tuple[str, str], ...] = ()
    artifact_digests: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.recipe_id
            or self.recipe_id != self.recipe_id.strip()
            or not self.revision
            or self.revision != self.revision.strip()
        ):
            raise ValueError("program recipe identity and revision are required")
        for field, values in (
            ("settings", self.settings),
            ("artifact_digests", self.artifact_digests),
        ):
            if type(values) is not tuple or any(
                type(item) is not tuple
                or len(item) != 2
                or any(
                    type(value) is not str or not value or value != value.strip() for value in item
                )
                for item in values
            ):
                raise TypeError(f"program recipe {field} must be canonical string pairs")
            keys = tuple(key for key, _ in values)
            if keys != tuple(sorted(set(keys))):
                raise ValueError(f"program recipe {field} keys must be unique and sorted")
        for name, digest in self.artifact_digests:
            _require_content_digest(digest, f"program artifact {name!r} digest")


@dataclass(frozen=True, slots=True)
class ModelBuildContext:
    """Explicit local-rank resources available while constructing a model program."""

    device: torch.device
    rank_id: int
    world_size: int
    owned_layer_ids: tuple[int, ...]
    model_root: Path | None = None

    def __post_init__(self) -> None:
        if self.rank_id < 0 or self.world_size <= 0 or self.rank_id >= self.world_size:
            raise ValueError("model build rank must lie within its positive world size")
        if tuple(sorted(set(self.owned_layer_ids))) != self.owned_layer_ids:
            raise ValueError("owned model layers must be non-negative, unique, and sorted")
        if self.owned_layer_ids and self.owned_layer_ids[0] < 0:
            raise ValueError("owned model layers must be non-negative")
        if self.model_root is not None and not self.model_root.is_dir():
            raise ValueError("model root must name an existing directory")


@dataclass(frozen=True, slots=True)
class PreparedInput:
    """One realized input plan and its exact optional encoded-feature payload."""

    input_plan: InputPlan
    feature_bundle: EncodedFeatureBundle | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input_plan, InputPlan):
            raise TypeError("prepared input requires an InputPlan")
        requires_features = bool(self.input_plan.encoder_jobs)
        if requires_features != (self.feature_bundle is not None):
            raise ValueError(
                "non-text InputPlan requires exactly one encoded feature bundle; "
                "text-only InputPlan cannot carry one"
            )
        if self.feature_bundle is not None:
            self.feature_bundle.verify_for_input_plan(self.input_plan)

    @property
    def semantic_digest(self) -> str:
        feature_digest = (
            None if self.feature_bundle is None else self.feature_bundle.semantic_digest
        )
        payload = f"{self.input_plan.semantic_digest}\0{feature_digest or '-'}".encode()
        return f"b2s256:{hashlib.blake2s(payload).hexdigest()}"


@dataclass(frozen=True, slots=True)
class StateInitialization:
    state_id: str
    value: torch.Tensor
    position: int = 0
    valid: bool = True

    def __post_init__(self) -> None:
        if not self.state_id or self.position < 0:
            raise ValueError("state initialization requires an id and non-negative position")
        if not isinstance(self.value, torch.Tensor) or self.value.layout is not torch.strided:
            raise TypeError("state initialization must contain a strided tensor")
        if self.value.numel() == 0:
            raise ValueError("state initialization cannot be empty")
        if (
            self.value.device.type != "meta"
            and (self.value.is_floating_point() or self.value.is_complex())
            and not bool(torch.isfinite(self.value).all().item())
        ):
            raise ValueError("state initialization must contain only finite values")


@dataclass(frozen=True, slots=True)
class ModelSegment:
    """Least-authority mathematical input minted from one runtime plan segment.

    Request identity, full-input digests, output boundaries, and physical cache
    placement are deliberately absent so they cannot change a prefix transition.
    """

    phase: BatchPhase
    token_start: int
    token_ids: tuple[int, ...]
    positions: tuple[TokenPosition, ...]

    @classmethod
    def from_plan(cls, segment: PlanSegment) -> ModelSegment:
        return cls(
            phase=segment.phase,
            token_start=segment.token_start,
            token_ids=segment.token_ids,
            positions=segment.positions,
        )

    def __post_init__(self) -> None:
        if self.phase not in (BatchPhase.PREFILL, BatchPhase.DECODE):
            raise ValueError("model segment must be prefill or decode")
        if self.token_start < 0 or not self.token_ids:
            raise ValueError("model segment requires a non-empty non-negative token span")
        if len(self.token_ids) != len(self.positions):
            raise ValueError("model segment requires one position per token")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("model segment token ids must be non-negative")

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class ModelStepResult:
    """Last-row logits for a transition; runtime decides whether they are observable."""

    logits: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.logits, torch.Tensor):
            raise TypeError("model step logits must be a tensor")
        if self.logits.ndim != 1 or self.logits.numel() == 0:
            raise ValueError("model step logits must be a non-empty rank-1 tensor")
        if not self.logits.is_floating_point():
            raise TypeError("model step logits must be floating point")


@runtime_checkable
class SemanticOpExecutor(Protocol):
    """A sealed operator dispatcher; model code supplies semantics, never a kernel."""

    def call(self, signature: OpSignature, /, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class ModelStateIO(Protocol):
    """Logical-token state access; physical page ids stay inside the runtime."""

    def read_kv(
        self, layer_id: int, token_ordinals: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def write_kv(
        self,
        layer_id: int,
        token_ordinals: tuple[int, ...],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None: ...

    def read_recurrent_state(self, state_id: str) -> torch.Tensor: ...

    def write_recurrent_state(
        self,
        state_id: str,
        value: torch.Tensor,
        *,
        target_position: int,
        valid: bool = True,
    ) -> None: ...


@runtime_checkable
class ModelProgram(Protocol):
    """Executable model graph with no authority to select device kernels."""

    def named_weight_tensors(self) -> tuple[tuple[str, torch.Tensor], ...]:
        """Enumerate the exact local executable weight inventory."""

        ...

    def execution_recipe(self) -> ProgramExecutionRecipe:
        """Declare all stable semantic choices selected while building this instance."""

        ...

    def initialize_request(self) -> tuple[StateInitialization, ...]: ...

    def execute_segment(
        self,
        segment: ModelSegment,
        ops: SemanticOpExecutor,
        state: ModelStateIO,
        features: FeatureReader,
    ) -> ModelStepResult: ...
