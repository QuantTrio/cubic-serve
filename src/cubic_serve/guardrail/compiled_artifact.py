from __future__ import annotations

from dataclasses import asdict, dataclass

from cubic_serve.ops.cuda.compiled_artifact import (
    CompiledArtifactIdentity,
    require_content_digest,
    stable_digest,
)


class CompiledArtifactRejected(ValueError):
    """A release manifest or compiled identity failed closed."""


@dataclass(frozen=True, slots=True)
class ReleaseArtifactApproval:
    release_id: str
    operator_id: str
    artifact_digest: str
    launch_specialization_digest: str
    semantic_facts: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if any(type(value) is not str or not value.strip()
               for value in (self.release_id, self.operator_id)):
            raise CompiledArtifactRejected("release approval requires release and operator IDs")
        try:
            require_content_digest(self.artifact_digest, "release artifact")
            require_content_digest(self.launch_specialization_digest, "release specialization")
        except ValueError as error:
            raise CompiledArtifactRejected(str(error)) from error
        if (
            type(self.semantic_facts) is not tuple or not self.semantic_facts
            or any(
                type(item) is not tuple or len(item) != 2
                or any(type(value) is not str or not value for value in item)
                for item in self.semantic_facts
            )
            or len({name for name, _ in self.semantic_facts}) != len(self.semantic_facts)
        ):
            raise CompiledArtifactRejected("release approval requires canonical semantic facts")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("release artifact approvals cannot be subclassed")

    @property
    def approval_digest(self) -> str:
        return stable_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class CompiledArtifactManifest:
    release_id: str
    entries: tuple[ReleaseArtifactApproval, ...]

    def __post_init__(self) -> None:
        if (
            type(self.release_id) is not str or not self.release_id.strip()
            or type(self.entries) is not tuple or not self.entries
        ):
            raise CompiledArtifactRejected("artifact manifest must be explicit and non-empty")
        for entry in self.entries:
            if type(entry) is not ReleaseArtifactApproval:
                raise CompiledArtifactRejected("compiled artifact manifest requires exact entries")
            ReleaseArtifactApproval.__post_init__(entry)
            if entry.release_id != self.release_id:
                raise CompiledArtifactRejected("compiled artifact manifest mixes releases")
        identities = tuple((entry.operator_id, entry.artifact_digest) for entry in self.entries)
        if len(set(identities)) != len(identities):
            raise CompiledArtifactRejected("artifact manifest contains duplicate identities")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("compiled artifact manifests cannot be subclassed")


def resolve_release_artifact(
    manifest: object, artifact: object, *, expected_operator_id: str,
    expected_semantic_facts: tuple[tuple[str, str], ...],
    required_flags: tuple[tuple[str, str], ...] = (),
) -> ReleaseArtifactApproval:
    """Non-virtual, consumption-time admission over explicit policy inputs."""

    try:
        if type(manifest) is not CompiledArtifactManifest:
            raise CompiledArtifactRejected("exact compiled artifact manifest is required")
        if type(artifact) is not CompiledArtifactIdentity:
            raise CompiledArtifactRejected("exact compiled artifact identity is required")
        CompiledArtifactManifest.__post_init__(manifest)
        CompiledArtifactIdentity.__post_init__(artifact)
        if artifact.operator_id != expected_operator_id:
            raise CompiledArtifactRejected("compiled artifact operator identity differs")
        matches = tuple(entry for entry in manifest.entries
                        if entry.operator_id == expected_operator_id
                        and entry.artifact_digest == artifact.artifact_digest)
        if len(matches) != 1:
            raise CompiledArtifactRejected("compiled artifact is not release-approved")
        approval = matches[0]
        ReleaseArtifactApproval.__post_init__(approval)
        if approval.semantic_facts != expected_semantic_facts:
            raise CompiledArtifactRejected("release approval omits required operator semantics")
        if approval.launch_specialization_digest != artifact.launch_specialization_digest:
            raise CompiledArtifactRejected("compiled launch specialization differs from approval")
        flags = dict(artifact.compiler_flags)
        if any(flags.get(name) != value for name, value in required_flags):
            raise CompiledArtifactRejected("compiled artifact flags differ from approval policy")
        return approval
    except CompiledArtifactRejected:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise CompiledArtifactRejected(str(error)) from error
