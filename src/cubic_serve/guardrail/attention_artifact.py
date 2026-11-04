"""Structural admission for one retained BF16 paged-GQA artifact and launch."""

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
from cubic_serve.ops.cuda.attention.paged_decode import (
    BF16_PAGED_GQA_OPERATOR_ID,
    BF16PagedGQAArtifactIdentity,
    BF16PagedGQADecodePrimitive,
    BF16PagedGQALaunchReceipt,
    launch_bf16_paged_gqa,
    project_bf16_paged_gqa_artifact,
    validate_bf16_paged_gqa_bound_operands,
    validate_bf16_paged_gqa_receipt,
)
from cubic_serve.ops.cuda.compiled_artifact import receipt_payload

PAGED_GQA_ARTIFACT_SEMANTICS = (
    ("active_token_domain", "each position in [0,seq_len) exactly once through page_table"),
    ("gqa_head_map", "query_head//(query_heads//kv_heads)"),
    ("layer", "exact compile-time layer"),
    ("scale", "exact compile-time positive-finite-float"),
    ("mask", "position<seq_len and dimension<head_dim"),
    ("inputs", "query,key,value=bf16;page_table,seq_lens=int32"),
    ("accumulator", "fp32-online-softmax"),
    ("store", "bf16-rne"),
    ("padding", "no-request-state; isolation requires graph-padding evidence"),
)
_WITNESS_TOKEN = object()


AttentionArtifactRejected = CompiledArtifactRejected
ReleaseBF16PagedGQAArtifact = ReleaseArtifactApproval
BF16PagedGQAArtifactManifest = CompiledArtifactManifest


def _resolve_release_artifact(
    manifest: BF16PagedGQAArtifactManifest,
    artifact: BF16PagedGQAArtifactIdentity,
) -> ReleaseBF16PagedGQAArtifact:
    """Non-virtual release admission used by the production verifier."""

    return resolve_release_artifact(
        manifest,
        artifact,
        expected_operator_id=BF16_PAGED_GQA_OPERATOR_ID,
        expected_semantic_facts=PAGED_GQA_ARTIFACT_SEMANTICS,
    )


@dataclass(frozen=True, slots=True)
class StructuralAttentionArtifactWitness:
    """Artifact/launch closure only; never arithmetic or padding evidence."""

    artifact_digest: str
    approval_digest: str
    operand_binding_digest: str
    launch_digest: str
    domain_digest: str
    output_digest: str
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _WITNESS_TOKEN:
            raise TypeError("structural attention witnesses are verifier-issued only")

    @property
    def witness_digest(self) -> str:
        return stable_digest(
            {
                "kind": "structural-paged-gqa-artifact-v1",
                "artifact": self.artifact_digest,
                "approval": self.approval_digest,
                "operands": self.operand_binding_digest,
                "launch": self.launch_digest,
                "domain": self.domain_digest,
                "output": self.output_digest,
            }
        )


class _AttentionArtifactObservation:
    """Private, single-consume bridge from a live launch to later checks."""

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
        self, *, primitive: BF16PagedGQADecodePrimitive, receipt: BF16PagedGQALaunchReceipt,
        witness: StructuralAttentionArtifactWitness, actual: Tensor, tensors: tuple[Tensor, ...],
    ) -> None:
        self.primitive, self.receipt, self.witness, self.actual = (
            primitive,
            receipt,
            witness,
            actual,
        )
        self.tensors, self.output = tensors, tensors[-1]
        self.versions = tuple(tensor._version for tensor in tensors)
        self.consumed = False

    def consume(self) -> tuple[StructuralAttentionArtifactWitness, Tensor]:
        if self.consumed:
            raise AttentionArtifactRejected("paged-GQA launch observation was already consumed")
        self.consumed = True
        try:
            validate_bf16_paged_gqa_receipt(self.primitive, self.receipt)
            validate_bf16_paged_gqa_bound_operands(self.primitive, *self.tensors)
            if tuple(tensor._version for tensor in self.tensors) != self.versions:
                raise AttentionArtifactRejected("paged-GQA operands changed after collection")
            artifact = project_bf16_paged_gqa_artifact(self.primitive)
            if artifact.artifact_digest != self.witness.artifact_digest:
                raise AttentionArtifactRejected("paged-GQA artifact changed before consume")
        except AttentionArtifactRejected:
            raise
        except Exception as error:
            raise AttentionArtifactRejected(str(error)) from error
        return self.witness, self.actual


def verify_bf16_paged_gqa_artifact_launch(
    *, primitive: BF16PagedGQADecodePrimitive, query: Tensor, key: Tensor, value: Tensor,
    page_table: Tensor, seq_lens: Tensor, output: Tensor, active_rows: tuple[int, ...],
    layer: int, scale: float, manifest: BF16PagedGQAArtifactManifest,
) -> StructuralAttentionArtifactWitness:
    """Own preflight, sentinel fill, retained launch, sync, and collection."""

    observation = _observe_bf16_paged_gqa_artifact_launch(
        primitive=primitive,
        query=query,
        key=key,
        value=value,
        page_table=page_table,
        seq_lens=seq_lens,
        output=output,
        active_rows=active_rows,
        layer=layer,
        scale=scale,
        manifest=manifest,
    )
    witness, _actual = observation.consume()
    return witness


def _observe_bf16_paged_gqa_artifact_launch(
    *, primitive: BF16PagedGQADecodePrimitive, query: Tensor, key: Tensor, value: Tensor,
    page_table: Tensor, seq_lens: Tensor, output: Tensor, active_rows: tuple[int, ...],
    layer: int, scale: float, manifest: BF16PagedGQAArtifactManifest,
) -> _AttentionArtifactObservation:
    tensors = query, key, value, page_table, seq_lens, output
    if type(primitive) is not BF16PagedGQADecodePrimitive:
        raise AttentionArtifactRejected("verification requires the production paged-GQA primitive")
    if type(manifest) is not BF16PagedGQAArtifactManifest:
        raise AttentionArtifactRejected("verification requires the exact typed release manifest")
    try:
        validate_bf16_paged_gqa_bound_operands(primitive, *tensors)
        if type(layer) is not int or layer != primitive.meta.layer:
            raise AttentionArtifactRejected("paged-GQA layer differs from the bound specialization")
        if (
            type(scale) is not float
            or primitive.meta.scale is None
            or scale.hex() != primitive.meta.scale.hex()
        ):
            raise AttentionArtifactRejected("paged-GQA scale differs from the bound specialization")
        artifact = project_bf16_paged_gqa_artifact(primitive)
        approval = _resolve_release_artifact(
            manifest, artifact
        )  # before sentinel fill or any launch
        domain_digest = _preflight_domain(key, page_table, seq_lens, active_rows)
        output.fill_(float("nan"))
        receipt = launch_bf16_paged_gqa(primitive, *tensors)
        torch.cuda.synchronize(query.device)
        validate_bf16_paged_gqa_receipt(primitive, receipt)
        validate_bf16_paged_gqa_bound_operands(primitive, *tensors)
        if project_bf16_paged_gqa_artifact(primitive) != artifact:
            raise AttentionArtifactRejected("compiled paged-GQA artifact changed across launch")
        collected = output.detach().to(device="cpu", copy=True).contiguous()
        if bool(torch.isnan(collected).any().item()):
            raise AttentionArtifactRejected("paged-GQA launch left sentinel output bytes stale")
    except AttentionArtifactRejected:
        raise
    except Exception as error:
        raise AttentionArtifactRejected(str(error)) from error
    launch = {
        **receipt_payload(receipt),
        "active_rows": active_rows,
        "padding_rows": tuple(row for row in range(query.shape[0]) if row not in active_rows),
        "layer": layer,
        "scale": scale.hex(),
    }
    witness = StructuralAttentionArtifactWitness(
        artifact.artifact_digest,
        approval.approval_digest,
        stable_digest(launch["bindings"]),
        stable_digest(launch),
        domain_digest,
        tensor_digest(collected),
        _WITNESS_TOKEN,
    )
    return _AttentionArtifactObservation(
        primitive=primitive,
        receipt=receipt,
        witness=witness,
        actual=collected,
        tensors=tensors,
    )


def _preflight_domain(
    key: Tensor, page_table: Tensor, seq_lens: Tensor, active_rows: tuple[int, ...],
) -> str:
    capacity, table_pages = page_table.shape
    pages, _, page_block, _, _ = key.shape
    if (
        type(active_rows) is not tuple
        or not active_rows
        or len(active_rows) != len(set(active_rows))
        or any(type(row) is not int or not 0 <= row < capacity for row in active_rows)
    ):
        raise AttentionArtifactRejected("active rows must be unique physical row indices")
    lengths = tuple(int(item) for item in seq_lens.detach().cpu().tolist())
    table = page_table.detach().cpu()
    active = set(active_rows)
    traversals: list[tuple[int, int, tuple[int, ...]]] = []
    for row, length in enumerate(lengths):
        if not 0 < length <= table_pages * page_block:
            raise AttentionArtifactRejected("paged-GQA sequence length exceeds table capacity")
        if row not in active and length != 1:
            raise AttentionArtifactRejected("padding rows require one legal scratch token")
        used_pages = (length + page_block - 1) // page_block
        page_ids = tuple(int(item) for item in table[row, :used_pages].tolist())
        if any(not 0 <= page < pages for page in page_ids):
            raise AttentionArtifactRejected("paged-GQA page traversal is out of range")
        traversals.append((row, length, page_ids))
    return stable_digest(
        {
            "active_rows": active_rows,
            "padding_rows": tuple(row for row in range(capacity) if row not in active),
            "page_block": page_block,
            "traversals": tuple(traversals),
            "padding_isolation": "external-graph-guardrail-required",
        }
    )


__all__ = (
    "PAGED_GQA_ARTIFACT_SEMANTICS", "AttentionArtifactRejected",
    "BF16PagedGQAArtifactManifest", "ReleaseBF16PagedGQAArtifact",
    "StructuralAttentionArtifactWitness", "verify_bf16_paged_gqa_artifact_launch",
)
