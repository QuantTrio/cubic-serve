"""Instance-local operator catalogs and fail-closed reference-plan sealing."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from cubic_serve.contracts.execution import (
    ExecutionMode,
    ExecutionPlan,
    OpBinding,
    OpSignature,
)

ReferenceCallable = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class ReferenceImplementation:
    semantic_op: str
    implementation_id: str
    function: ReferenceCallable

    def __post_init__(self) -> None:
        if not self.semantic_op or not self.implementation_id:
            raise ValueError("reference semantic and implementation identity are required")
        if not callable(self.function):
            raise TypeError("reference implementation must be callable")


@dataclass(frozen=True, slots=True)
class ReferenceCatalog:
    """An immutable catalog assembled explicitly for one engine instance."""

    implementations: tuple[ReferenceImplementation, ...]

    def __post_init__(self) -> None:
        semantics = tuple(item.semantic_op for item in self.implementations)
        implementation_ids = tuple(item.implementation_id for item in self.implementations)
        if not semantics:
            raise ValueError("a reference catalog cannot be empty")
        if len(set(semantics)) != len(semantics):
            raise ValueError("a semantic op cannot have multiple reference implementations")
        if len(set(implementation_ids)) != len(implementation_ids):
            raise ValueError("reference implementation ids must be unique")

    def resolve(self, semantic_op: str) -> ReferenceImplementation:
        for implementation in self.implementations:
            if implementation.semantic_op == semantic_op:
                return implementation
        raise LookupError(f"missing reference implementation for {semantic_op!r}")

    def extend(self, additions: Iterable[ReferenceImplementation]) -> ReferenceCatalog:
        return ReferenceCatalog(self.implementations + tuple(additions))


def builtin_reference_catalog() -> ReferenceCatalog:
    """Construct the builtin catalog without registration side effects."""

    from cubic_serve.ops.eager.attention import full_attention
    from cubic_serve.ops.eager.linear import linear

    return ReferenceCatalog(
        (
            ReferenceImplementation(
                semantic_op="attention.full",
                implementation_id="cubic.reference.attention.full.v1",
                function=full_attention,
            ),
            ReferenceImplementation(
                semantic_op="linear",
                implementation_id="cubic.reference.linear.v1",
                function=linear,
            ),
        )
    )


def seal_reference_plan(
    execution_plan_id: str,
    signatures: Iterable[OpSignature],
    catalog: ReferenceCatalog,
) -> ExecutionPlan:
    """Resolve every required semantic op internally or reject readiness."""

    frozen_signatures = tuple(signatures)
    if not frozen_signatures:
        raise ValueError("reference readiness requires at least one op signature")
    if len(set(frozen_signatures)) != len(frozen_signatures):
        raise ValueError("reference readiness cannot contain duplicate signatures")

    bindings = tuple(
        OpBinding(
            signature=signature,
            reference_implementation_id=catalog.resolve(signature.semantic_op).implementation_id,
        )
        for signature in frozen_signatures
    )
    return ExecutionPlan(
        execution_plan_id=execution_plan_id,
        mode=ExecutionMode.REFERENCE,
        bindings=bindings,
    )


def resolve_reference_binding(
    binding: OpBinding,
    catalog: ReferenceCatalog,
) -> ReferenceCallable:
    if binding.reference_implementation_id is None or binding.candidate is not None:
        raise ValueError("binding is not a sealed reference binding")
    implementation = catalog.resolve(binding.signature.semantic_op)
    if implementation.implementation_id != binding.reference_implementation_id:
        raise RuntimeError("reference binding identity does not match the active catalog")
    return implementation.function
