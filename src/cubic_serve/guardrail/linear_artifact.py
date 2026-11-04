"""Structural admission for one content-addressed BF16 vendor launch."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from cubic_serve.guardrail.compare import stable_digest, tensor_digest
from cubic_serve.guardrail.compiled_artifact import (
    CompiledArtifactManifest,
    CompiledArtifactRejected,
    ReleaseArtifactApproval,
    resolve_release_artifact,
)
from cubic_serve.ops.cuda.compiled_artifact import receipt_payload
from cubic_serve.ops.cuda.linear.kernel import (
    BF16_LINEAR_OPERATOR_ID,
    BF16LinearArtifactIdentity,
    BF16LinearLaunchReceipt,
    BF16LinearPrimitive,
    launch_bf16_linear,
    project_bf16_linear_artifact,
    validate_bf16_linear_bound_operands,
    validate_bf16_linear_receipt,
)

LINEAR_ARTIFACT_SEMANTICS = (
    ("k_domain", "[0,K)-exactly-once"),
    ("accumulator", "fp32-mma"),
    ("store", "bf16-rne"),
    ("dispatch", "aten.mm.out/torch-cublas"),
    ("selected_cubin", "not-exposed-by-vendor-abi"),
)
_WITNESS_TOKEN = object()


LinearArtifactRejected = CompiledArtifactRejected
ReleaseBF16LinearArtifact = ReleaseArtifactApproval
BF16LinearArtifactManifest = CompiledArtifactManifest


def _resolve_release_artifact(
    manifest: BF16LinearArtifactManifest,
    artifact: BF16LinearArtifactIdentity,
) -> ReleaseBF16LinearArtifact:
    """Non-virtual release admission used by the production verifier."""

    return resolve_release_artifact(
        manifest,
        artifact,
        expected_operator_id=BF16_LINEAR_OPERATOR_ID,
        expected_semantic_facts=LINEAR_ARTIFACT_SEMANTICS,
        required_flags=(
            ("allow_bf16_reduced_precision_reduction", "false"),
            ("allow_tf32", "false"),
        ),
    )


@dataclass(frozen=True, slots=True)
class StructuralLinearArtifactWitness:
    """Artifact/launch closure only; this is not numerical evidence."""

    artifact_digest: str
    approval_digest: str
    operand_binding_digest: str
    launch_digest: str
    output_digest: str
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _WITNESS_TOKEN:
            raise TypeError("structural linear witnesses are verifier-issued only")

    @property
    def witness_digest(self) -> str:
        return stable_digest(
            {
                "kind": "structural-linear-artifact-v1",
                "artifact": self.artifact_digest,
                "approval": self.approval_digest,
                "operands": self.operand_binding_digest,
                "launch": self.launch_digest,
                "output": self.output_digest,
            }
        )


class _LinearArtifactObservation:
    """Private, single-consume bridge from one launch to arithmetic checking."""

    __slots__ = (
        "actual",
        "consumed",
        "output",
        "primitive",
        "receipt",
        "tensors",
        "versions",
        "witness",
    )

    def __init__(
        self, *, primitive: BF16LinearPrimitive, receipt: BF16LinearLaunchReceipt,
        witness: StructuralLinearArtifactWitness, actual: Tensor, x: Tensor,
        weight: Tensor, output: Tensor,
    ) -> None:
        self.primitive, self.receipt, self.witness, self.actual = (
            primitive,
            receipt,
            witness,
            actual,
        )
        self.tensors = (x, weight, output)
        self.output = output
        self.versions = tuple(tensor._version for tensor in self.tensors)
        self.consumed = False

    def consume(self) -> tuple[StructuralLinearArtifactWitness, Tensor]:
        if self.consumed:
            raise LinearArtifactRejected("linear launch observation was already consumed")
        self.consumed = True
        try:
            validate_bf16_linear_receipt(self.primitive, self.receipt)
            validate_bf16_linear_bound_operands(self.primitive, *self.tensors)
            if tuple(tensor._version for tensor in self.tensors) != self.versions:
                raise LinearArtifactRejected("linear operands changed after durable collection")
            artifact = project_bf16_linear_artifact(self.primitive)
            if artifact.artifact_digest != self.witness.artifact_digest:
                raise LinearArtifactRejected("linear artifact changed before observation consume")
        except LinearArtifactRejected:
            raise
        except Exception as error:
            raise LinearArtifactRejected(str(error)) from error
        return self.witness, self.actual


def verify_bf16_linear_artifact_launch(
    *, primitive: BF16LinearPrimitive, x: Tensor, weight: Tensor, output: Tensor,
    manifest: BF16LinearArtifactManifest,
) -> StructuralLinearArtifactWitness:
    """Own sentinel fill, direct launch, synchronization, and output collection."""

    observation = _observe_bf16_linear_artifact_launch(
        primitive=primitive, x=x, weight=weight, output=output, manifest=manifest
    )
    witness, _actual = observation.consume()
    return witness


def _observe_bf16_linear_artifact_launch(
    *, primitive: BF16LinearPrimitive, x: Tensor, weight: Tensor, output: Tensor,
    manifest: BF16LinearArtifactManifest,
) -> _LinearArtifactObservation:
    if type(primitive) is not BF16LinearPrimitive:
        raise LinearArtifactRejected("verification requires the production linear primitive")
    if type(manifest) is not BF16LinearArtifactManifest:
        raise LinearArtifactRejected("verification requires the exact typed release manifest")
    try:
        validate_bf16_linear_bound_operands(primitive, x, weight, output)
        artifact = project_bf16_linear_artifact(primitive)
        approval = _resolve_release_artifact(manifest, artifact)
        output.fill_(float("nan"))
        receipt = launch_bf16_linear(primitive, x, weight, output)
        torch.cuda.synchronize(x.device)
        validate_bf16_linear_receipt(primitive, receipt)
        validate_bf16_linear_bound_operands(primitive, x, weight, output)
        if project_bf16_linear_artifact(primitive) != artifact:
            raise LinearArtifactRejected("compiled artifact changed across its launch")
        collected = output.detach().to(device="cpu", copy=True).contiguous()
        if bool(torch.isnan(collected).any().item()):
            raise LinearArtifactRejected(
                "linear launch left sentinel output bytes stale or nonfinite"
            )
    except LinearArtifactRejected:
        raise
    except Exception as error:
        raise LinearArtifactRejected(str(error)) from error
    launch = receipt_payload(receipt)
    witness = StructuralLinearArtifactWitness(
        artifact.artifact_digest,
        approval.approval_digest,
        stable_digest(launch["bindings"]),
        stable_digest(launch),
        tensor_digest(collected),
        _WITNESS_TOKEN,
    )
    return _LinearArtifactObservation(
        primitive=primitive,
        receipt=receipt,
        witness=witness,
        actual=collected,
        x=x,
        weight=weight,
        output=output,
    )
__all__ = (
    "LINEAR_ARTIFACT_SEMANTICS", "BF16LinearArtifactManifest", "LinearArtifactRejected",
    "ReleaseBF16LinearArtifact", "StructuralLinearArtifactWitness",
    "verify_bf16_linear_artifact_launch",
)
