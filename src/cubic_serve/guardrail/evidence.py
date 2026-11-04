"""Reproducible, atomic guardrail evidence artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from cubic_serve._compat import StrEnum
from pathlib import Path
from typing import Any

from cubic_serve.guardrail.compare import ComparisonResult


class EvidenceProfile(StrEnum):
    STARTUP = "startup"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class QualificationFingerprint:
    """Everything whose change invalidates candidate correctness evidence."""

    contract_digest: str
    source_digest: str
    binary_digest: str
    op_signature_digest: str
    implementation_id: str
    device_capability: str
    driver_version: str
    runtime_version: str
    topology_digest: str

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not value for value in values.values()):
            raise ValueError("qualification fingerprint fields cannot be empty")

    @property
    def key(self) -> str:
        return _digest_json(asdict(self))


@dataclass(frozen=True, slots=True)
class ComparisonRecord:
    case_id: str
    semantic_op: str
    stage: int
    rank: int
    candidate_id: str
    launch_signature: str
    result: ComparisonResult

    def __post_init__(self) -> None:
        if not all((self.case_id, self.semantic_op, self.candidate_id, self.launch_signature)):
            raise ValueError("comparison record identity is required")
        if self.stage < 0 or self.rank < 0:
            raise ValueError("comparison stage and rank must be non-negative")


@dataclass(frozen=True, slots=True)
class GuardrailEvidence:
    profile: EvidenceProfile
    fingerprint: QualificationFingerprint
    records: tuple[ComparisonRecord, ...]
    created_at: str

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("guardrail evidence requires at least one comparison")
        try:
            parsed = datetime.fromisoformat(self.created_at)
        except ValueError as error:
            raise ValueError("created_at must be an ISO-8601 timestamp") from error
        if parsed.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        case_ids = tuple(record.case_id for record in self.records)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("guardrail case ids must be unique")

    @classmethod
    def now(
        cls,
        *,
        profile: EvidenceProfile,
        fingerprint: QualificationFingerprint,
        records: tuple[ComparisonRecord, ...],
    ) -> GuardrailEvidence:
        return cls(
            profile=profile,
            fingerprint=fingerprint,
            records=records,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @property
    def passed(self) -> bool:
        return all(record.result.passed for record in self.records)

    @property
    def artifact_id(self) -> str:
        return _digest_json(self.to_dict(include_artifact_id=False))

    def to_dict(self, *, include_artifact_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "profile": self.profile.value,
            "qualification_key": self.fingerprint.key,
            "fingerprint": asdict(self.fingerprint),
            "passed": self.passed,
            "created_at": self.created_at,
            "records": [
                {
                    "case_id": record.case_id,
                    "semantic_op": record.semantic_op,
                    "stage": record.stage,
                    "rank": record.rank,
                    "candidate_id": record.candidate_id,
                    "launch_signature": record.launch_signature,
                    "result": {
                        **asdict(record.result),
                        "difference_class": record.result.difference_class.value,
                    },
                }
                for record in self.records
            ],
        }
        if include_artifact_id:
            payload["artifact_id"] = self.artifact_id
        return payload

    def write_atomic(self, path: Path) -> None:
        """Write a canonical JSON artifact without ever exposing a partial pass."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        encoded = _canonical_json(self.to_dict()).encode("utf-8") + b"\n"
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest_json(value: Any) -> str:
    return "b2s256:" + hashlib.blake2s(_canonical_json(value).encode("utf-8")).hexdigest()
