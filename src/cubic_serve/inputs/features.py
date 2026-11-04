"""Core-owned encoded-feature identity and fail-closed consumption ABI.

This module does not encode images or videos. Encoders live outside this
boundary and publish already-realized BF16 feature rows for a canonical
``InputPlan``. The store owns its copy and authorizes every model read against
the request, encoder job, row span, consumer layer, and semantic digest.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from cubic_serve._compat import StrEnum
from threading import RLock
from typing import Protocol, runtime_checkable

import torch

from cubic_serve.contracts.execution import BatchPhase, PlanSegment
from cubic_serve.contracts.request import InputPlan, TokenPlacement


class FeatureStoreError(RuntimeError):
    """Base failure for encoded-feature publication or consumption."""


class FeatureAdmissionError(FeatureStoreError):
    """A feature bundle does not match its canonical input plan."""


class FeatureAccessError(FeatureStoreError):
    """A model attempted an unauthorized or ambiguous feature read."""


class FeatureIntegrityError(FeatureStoreError):
    """Feature metadata or tensor bits no longer match their sealed identity."""


class FeatureTensorLayout(StrEnum):
    """Supported logical feature tensor layouts."""

    ROWS_HIDDEN_CONTIGUOUS = "rows_hidden.contiguous"


def _require_digest(value: str, *, field_name: str) -> None:
    algorithm, separator, hexadecimal = value.partition(":")
    if algorithm != "b2s256" or separator != ":" or len(hexadecimal) != 64:
        raise ValueError(f"{field_name} must use b2s256:<64 hex chars>")
    try:
        int(hexadecimal, 16)
    except ValueError as error:
        raise ValueError(f"{field_name} contains non-hexadecimal characters") from error


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"b2s256:{hashlib.blake2s(encoded).hexdigest()}"


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    with torch.no_grad():
        raw = tensor.detach().contiguous().view(torch.uint16).cpu().numpy().tobytes(order="C")
    return f"b2s256:{hashlib.blake2s(raw).hexdigest()}"


@dataclass(frozen=True, slots=True)
class EncodedFeature:
    """One encoder job's realized, device-resident BF16 feature matrix."""

    job_id: str
    content_part_ordinal: int
    encoder_id: str
    processor_id: str
    source_media_digest: str
    tensor: torch.Tensor = field(compare=False, repr=False)
    layout: FeatureTensorLayout = FeatureTensorLayout.ROWS_HIDDEN_CONTIGUOUS
    tensor_content_digest: str = field(init=False)
    semantic_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not all((self.job_id, self.encoder_id, self.processor_id)):
            raise ValueError("encoded feature job, encoder, and processor identities are required")
        if self.content_part_ordinal < 0:
            raise ValueError("encoded feature content ordinal must be non-negative")
        _require_digest(self.source_media_digest, field_name="source media digest")
        if not isinstance(self.layout, FeatureTensorLayout):
            raise TypeError("encoded feature layout must be a FeatureTensorLayout")
        self._validate_tensor()
        tensor_digest = _tensor_content_digest(self.tensor)
        object.__setattr__(self, "tensor_content_digest", tensor_digest)
        object.__setattr__(self, "semantic_digest", self._compute_semantic_digest(tensor_digest))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def feature_rows(self) -> int:
        return int(self.tensor.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.tensor.shape[1])

    def _validate_tensor(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("encoded feature payload must be a torch.Tensor")
        if self.tensor.dtype is not torch.bfloat16:
            raise TypeError("encoded feature tensor must be BF16")
        if self.tensor.layout is not torch.strided or not self.tensor.is_contiguous():
            raise ValueError("encoded feature tensor must use contiguous strided layout")
        if self.tensor.ndim != 2 or any(dimension <= 0 for dimension in self.tensor.shape):
            raise ValueError("encoded feature tensor must have non-empty [rows, hidden] shape")
        if self.tensor.device.type == "meta":
            raise ValueError("encoded feature tensor must have realized device storage")
        if self.tensor.requires_grad:
            raise ValueError("encoded feature tensor cannot require gradients")
        if not bool(torch.isfinite(self.tensor).all().item()):
            raise ValueError("encoded feature tensor must contain only finite values")

    def _compute_semantic_digest(self, tensor_digest: str) -> str:
        return _canonical_digest(
            {
                "content_part_ordinal": self.content_part_ordinal,
                "device": str(self.tensor.device),
                "dtype": "bfloat16",
                "encoder_id": self.encoder_id,
                "hidden_size": self.hidden_size,
                "job_id": self.job_id,
                "layout": self.layout.value,
                "processor_id": self.processor_id,
                "rows": self.feature_rows,
                "source_media_digest": self.source_media_digest,
                "tensor_content_digest": tensor_digest,
            }
        )

    def verify_integrity(self) -> None:
        """Reject mutation of shape, metadata, or tensor bits after sealing."""

        try:
            self._validate_tensor()
            tensor_digest = _tensor_content_digest(self.tensor)
            semantic_digest = self._compute_semantic_digest(tensor_digest)
        except (TypeError, ValueError) as error:
            raise FeatureIntegrityError(f"encoded feature {self.job_id!r} is invalid") from error
        if tensor_digest != self.tensor_content_digest or semantic_digest != self.semantic_digest:
            raise FeatureIntegrityError(
                f"encoded feature {self.job_id!r} no longer matches its sealed digest"
            )


class EncodedFeatureCache:
    """Small byte-bounded LRU for request-independent encoder results."""

    def __init__(self, max_bytes: int) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("encoded feature cache size must be a non-negative integer")
        self._max_bytes = max_bytes
        self._resident_bytes = 0
        self._entries: OrderedDict[str, EncodedFeature] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> EncodedFeature | None:
        if not key:
            raise ValueError("encoded feature cache key is required")
        with self._lock:
            feature = self._entries.pop(key, None)
            if feature is not None:
                self._entries[key] = feature
            return feature

    def put(self, key: str, feature: EncodedFeature) -> EncodedFeature:
        if not key or not isinstance(feature, EncodedFeature):
            raise TypeError("encoded feature cache requires a key and EncodedFeature")
        size = feature.tensor.numel() * feature.tensor.element_size()
        with self._lock:
            previous = self._entries.get(key)
            if previous is not None:
                if previous.semantic_digest != feature.semantic_digest:
                    raise FeatureIntegrityError("encoded feature cache key changed identity")
                self._entries.move_to_end(key)
                return previous
            if size > self._max_bytes:
                return feature
            while self._entries and self._resident_bytes + size > self._max_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._resident_bytes -= evicted.tensor.numel() * evicted.tensor.element_size()
            self._entries[key] = feature
            self._resident_bytes += size
            return feature


@dataclass(frozen=True, slots=True)
class FeatureConsumerBinding:
    """Bind one encoded feature to its authorized decoder placement and layers."""

    feature: EncodedFeature
    placement: TokenPlacement

    def __post_init__(self) -> None:
        if self.feature.content_part_ordinal != self.placement.content_part_ordinal:
            raise ValueError("feature and placement content ordinals must match")
        end = self.placement.feature_row_start + self.placement.feature_row_count
        if end > self.feature.feature_rows:
            raise ValueError("feature placement exceeds the encoded feature row count")

    def semantic_payload(self) -> dict[str, object]:
        return {
            "consumer_layers": self.placement.consumer_layers,
            "consumer_site": self.placement.consumer_site,
            "content_part_ordinal": self.placement.content_part_ordinal,
            "feature_digest": self.feature.semantic_digest,
            "feature_row_count": self.placement.feature_row_count,
            "feature_row_start": self.placement.feature_row_start,
            "fusion": self.placement.fusion.value,
            "job_id": self.feature.job_id,
            "token_count": self.placement.token_count,
            "token_start": self.placement.token_start,
        }


@dataclass(frozen=True, slots=True)
class EncodedFeatureBundle:
    """Canonical set of feature/consumer bindings for one request."""

    request_id: str
    bindings: tuple[FeatureConsumerBinding, ...]
    semantic_digest: str = field(init=False)

    @classmethod
    def from_input_plan(
        cls,
        input_plan: InputPlan,
        features: tuple[EncodedFeature, ...],
    ) -> EncodedFeatureBundle:
        """Seal external encoder results against one fully realized input plan."""

        return cls(input_plan.request_id, _bindings_for_input_plan(input_plan, features))

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("feature bundle request identity is required")
        if not self.bindings:
            raise ValueError("feature bundle requires at least one encoded feature")
        keys = tuple(
            (
                binding.feature.content_part_ordinal,
                binding.feature.job_id,
                binding.placement.token_start,
                binding.placement.feature_row_start,
            )
            for binding in self.bindings
        )
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("feature bundle bindings must be unique and canonical")
        feature_digest_by_job: dict[str, str] = {}
        for binding in self.bindings:
            job_id = binding.feature.job_id
            known_digest = feature_digest_by_job.setdefault(
                job_id,
                binding.feature.semantic_digest,
            )
            if known_digest != binding.feature.semantic_digest:
                raise ValueError("one encoder job cannot bind different encoded features")
        object.__setattr__(self, "semantic_digest", self._compute_semantic_digest())

    def _compute_semantic_digest(self) -> str:
        return _canonical_digest(
            {
                "bindings": [binding.semantic_payload() for binding in self.bindings],
            }
        )

    def verify_integrity(self) -> None:
        for binding in self.bindings:
            binding.feature.verify_integrity()
        if self._compute_semantic_digest() != self.semantic_digest:
            raise FeatureIntegrityError(
                f"feature bundle {self.request_id!r} no longer matches its sealed digest"
            )

    def verify_for_input_plan(self, input_plan: InputPlan) -> None:
        """Reject request, job, placement, media, or tensor drift from the plan."""

        self.verify_integrity()
        if self.request_id != input_plan.request_id:
            raise FeatureAdmissionError("feature bundle request identity differs from InputPlan")
        features_by_job = {
            binding.feature.job_id: binding.feature for binding in self.bindings
        }
        expected = EncodedFeatureBundle.from_input_plan(
            input_plan,
            tuple(features_by_job[job_id] for job_id in sorted(features_by_job)),
        )
        if expected.semantic_digest != self.semantic_digest:
            raise FeatureAdmissionError("feature bundle semantic identity differs from InputPlan")

    def semantic_digest_for_parts(self, content_part_ordinals: tuple[int, ...]) -> str:
        """Derive a request-independent digest for the selected media parts."""

        if tuple(sorted(set(content_part_ordinals))) != content_part_ordinals:
            raise ValueError("feature digest part ordinals must be unique and sorted")
        requested = set(content_part_ordinals)
        selected = tuple(
            binding
            for binding in self.bindings
            if binding.feature.content_part_ordinal in requested
        )
        if {binding.feature.content_part_ordinal for binding in selected} != requested:
            raise ValueError("feature digest references a part absent from the bundle")
        return _canonical_digest({"bindings": [binding.semantic_payload() for binding in selected]})


def _bindings_for_input_plan(
    input_plan: InputPlan,
    features: tuple[EncodedFeature, ...],
) -> tuple[FeatureConsumerBinding, ...]:
    feature_job_ids = tuple(feature.job_id for feature in features)
    if len(set(feature_job_ids)) != len(feature_job_ids):
        raise FeatureAdmissionError("cannot publish a duplicate encoder job")
    jobs_by_id = {job.job_id: job for job in input_plan.encoder_jobs}
    if set(feature_job_ids) != set(jobs_by_id):
        raise FeatureAdmissionError(
            "published feature jobs must exactly match the input plan encoder jobs"
        )
    placements_by_ordinal: dict[int, list[TokenPlacement]] = {}
    for placement in input_plan.placements:
        placements_by_ordinal.setdefault(placement.content_part_ordinal, []).append(placement)
    parts_by_ordinal = {part.ordinal: part for part in input_plan.content_parts}

    bindings: list[FeatureConsumerBinding] = []
    for feature in features:
        feature.verify_integrity()
        job = jobs_by_id[feature.job_id]
        if (
            feature.content_part_ordinal != job.content_part_ordinal
            or feature.encoder_id != job.encoder_id
            or feature.processor_id != job.processor_id
        ):
            raise FeatureAdmissionError(
                f"encoded feature {feature.job_id!r} does not match its encoder job"
            )
        part = parts_by_ordinal[job.content_part_ordinal]
        if part.media is None or part.media.content_digest != feature.source_media_digest:
            raise FeatureAdmissionError(
                f"encoded feature {feature.job_id!r} has the wrong source media digest"
            )
        for placement in placements_by_ordinal[job.content_part_ordinal]:
            try:
                binding = FeatureConsumerBinding(feature, placement)
            except ValueError as error:
                raise FeatureAdmissionError(
                    f"encoded feature {feature.job_id!r} does not fit its placement"
                ) from error
            bindings.append(binding)
    bindings.sort(
        key=lambda binding: (
            binding.feature.content_part_ordinal,
            binding.feature.job_id,
            binding.placement.token_start,
            binding.placement.feature_row_start,
        )
    )
    return tuple(bindings)


@dataclass(frozen=True, slots=True)
class FeatureSpan:
    """Current-segment feature rows for one static model consumer site."""

    token_offset: int
    rows: torch.Tensor = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.token_offset < 0:
            raise ValueError("feature span token offset must be non-negative")
        if (
            not isinstance(self.rows, torch.Tensor)
            or self.rows.dtype is not torch.bfloat16
            or self.rows.layout is not torch.strided
            or not self.rows.is_contiguous()
            or self.rows.ndim != 2
            or min(self.rows.shape) <= 0
        ):
            raise ValueError("feature span rows must be a non-empty contiguous BF16 matrix")

    @property
    def token_count(self) -> int:
        return int(self.rows.shape[0])


@runtime_checkable
class FeatureReader(Protocol):
    """Segment-scoped read capability presented to model code.

    The capability is already bound to a request and an immutable
    ``PlanSegment``. Callers therefore cannot select another request, phase,
    token span, or feature job outside that segment's placement overlap.
    """

    def spans(self, consumer_site: str, consumer_layer: int) -> tuple[FeatureSpan, ...]: ...


@dataclass(frozen=True, slots=True)
class _ScopedFeatureRows:
    job_id: str
    semantic_digest: str
    token_start: int
    feature_row_start: int
    consumer_site: str
    consumer_layers: tuple[int, ...]
    tensor: torch.Tensor = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.job_id or not self.consumer_site:
            raise ValueError("scoped feature rows require job and consumer identities")
        _require_digest(self.semantic_digest, field_name="scoped feature digest")
        if min(self.token_start, self.feature_row_start) < 0:
            raise ValueError("scoped feature starts must be non-negative")
        if not self.consumer_layers:
            raise ValueError("scoped feature rows require consumer layers")
        if self.tensor.ndim != 2 or self.tensor.shape[0] <= 0:
            raise ValueError("scoped feature tensor must be a non-empty row matrix")


class _SegmentFeatureReader:
    """Capability containing only rows authorized for one segment.

    It deliberately has no reference to ``FeatureStore``. Even accidental
    inspection of private fields cannot reveal another segment's feature rows.
    """

    __slots__ = (
        "_active",
        "_phase",
        "_request_id",
        "_rows",
        "_token_count",
        "_token_start",
    )

    def __init__(
        self,
        *,
        request_id: str,
        phase: BatchPhase,
        token_start: int,
        token_count: int,
        rows: tuple[_ScopedFeatureRows, ...],
    ) -> None:
        self._request_id = request_id
        self._phase = phase
        self._token_start = token_start
        self._token_count = token_count
        self._rows = rows
        self._active = True

    def revoke(self) -> None:
        """Expire this one-call capability and drop its scoped tensor handles."""

        self._active = False
        self._rows = ()

    def spans(self, consumer_site: str, consumer_layer: int) -> tuple[FeatureSpan, ...]:
        if not self._active:
            raise FeatureAccessError("segment feature capability has expired")
        if not consumer_site or consumer_site != consumer_site.strip():
            raise FeatureAccessError("feature consumer site must be non-empty and canonical")
        if consumer_layer < 0:
            raise FeatureAccessError("feature consumer layer must be non-negative")
        selected = tuple(
            row
            for row in self._rows
            if row.consumer_site == consumer_site and consumer_layer in row.consumer_layers
        )
        with torch.no_grad():
            return tuple(
                FeatureSpan(
                    token_offset=row.token_start - self._token_start,
                    rows=row.tensor.detach().clone(memory_format=torch.contiguous_format),
                )
                for row in selected
            )


class FeatureStore:
    """Core-owned in-memory store for immutable encoded feature snapshots."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._rows: dict[tuple[str, str], EncodedFeature] = {}
        self._bundles: dict[str, EncodedFeatureBundle] = {}

    def publish(
        self,
        input_plan: InputPlan,
        bundle: EncodedFeatureBundle,
    ) -> EncodedFeatureBundle:
        """Atomically admit all encoder results declared by one input plan."""

        bundle.verify_for_input_plan(input_plan)
        owned_features: dict[str, EncodedFeature] = {}
        for binding in bundle.bindings:
            feature = binding.feature
            if feature.job_id in owned_features:
                continue
            with torch.no_grad():
                owned_tensor = feature.tensor.detach().clone(memory_format=torch.contiguous_format)
            owned_feature = EncodedFeature(
                job_id=feature.job_id,
                content_part_ordinal=feature.content_part_ordinal,
                encoder_id=feature.encoder_id,
                processor_id=feature.processor_id,
                source_media_digest=feature.source_media_digest,
                tensor=owned_tensor,
                layout=feature.layout,
            )
            if owned_feature.semantic_digest != feature.semantic_digest:
                raise FeatureIntegrityError(
                    f"encoded feature {feature.job_id!r} changed while being admitted"
                )
            owned_features[feature.job_id] = owned_feature

        owned_bundle = EncodedFeatureBundle.from_input_plan(
            input_plan,
            tuple(owned_features[job_id] for job_id in sorted(owned_features)),
        )
        if owned_bundle.semantic_digest != bundle.semantic_digest:
            raise FeatureIntegrityError(
                "owned feature bundle identity changed while being admitted"
            )

        with self._lock:
            if input_plan.request_id in self._bundles:
                raise FeatureAdmissionError(
                    f"request {input_plan.request_id!r} already has a feature bundle"
                )
            keys = tuple((input_plan.request_id, job_id) for job_id in sorted(owned_features))
            if any(key in self._rows for key in keys):
                raise FeatureAdmissionError("cannot overwrite an existing encoded feature job")
            self._bundles[input_plan.request_id] = owned_bundle
            self._rows.update(
                (key, owned_features[key[1]]) for key in keys
            )
        return owned_bundle

    def reader_for_segment(
        self,
        input_plan: InputPlan,
        segment: PlanSegment,
    ) -> _SegmentFeatureReader:
        """Mint a least-authority reader for exactly one immutable plan segment."""

        if segment.request_id != input_plan.request_id:
            raise FeatureAccessError("segment and input plan request identities differ")
        if segment.input_plan_digest != input_plan.semantic_digest:
            raise FeatureAccessError("segment and input plan semantic identities differ")
        if segment.phase not in (BatchPhase.PREFILL, BatchPhase.DECODE):
            raise FeatureAccessError("feature capability requires a prefill or decode segment")
        segment_end = segment.token_start + segment.token_count
        if segment.phase is BatchPhase.PREFILL and segment_end > len(input_plan.token_ids):
            raise FeatureAccessError("prefill segment exceeds the realized input token span")
        if segment.phase is BatchPhase.DECODE and segment.token_start < len(input_plan.token_ids):
            raise FeatureAccessError("decode segment overlaps the realized prompt token span")
        with self._lock:
            bundle = self._bundles.get(input_plan.request_id)
            if bundle is None:
                if input_plan.encoder_jobs:
                    raise FeatureAccessError("input plan media has no published feature bundle")
                bindings: tuple[FeatureConsumerBinding, ...] = ()
            else:
                bundle.verify_integrity()
                bindings = bundle.bindings

            feature_digest = None if bundle is None else bundle.semantic_digest
            payload = f"{input_plan.semantic_digest}\0{feature_digest or '-'}".encode()
            execution_input_digest = f"b2s256:{hashlib.blake2s(payload).hexdigest()}"
            if segment.execution_input_digest != execution_input_digest:
                raise FeatureAccessError("segment and prepared input identities differ")

            scoped_rows: list[_ScopedFeatureRows] = []
            if segment.phase is BatchPhase.PREFILL:
                for binding in bindings:
                    placement = binding.placement
                    overlap_start = max(segment.token_start, placement.token_start)
                    overlap_end = min(
                        segment_end,
                        placement.token_start + placement.token_count,
                    )
                    if overlap_start >= overlap_end:
                        continue
                    feature = binding.feature
                    feature.verify_integrity()
                    row_start = placement.feature_row_start + overlap_start - placement.token_start
                    row_count = overlap_end - overlap_start
                    with torch.no_grad():
                        tensor = (
                            feature.tensor.narrow(0, row_start, row_count)
                            .detach()
                            .clone(memory_format=torch.contiguous_format)
                        )
                    scoped_rows.append(
                        _ScopedFeatureRows(
                            job_id=feature.job_id,
                            semantic_digest=feature.semantic_digest,
                            token_start=overlap_start,
                            feature_row_start=row_start,
                            consumer_site=placement.consumer_site,
                            consumer_layers=placement.consumer_layers,
                            tensor=tensor,
                        )
                    )
        return _SegmentFeatureReader(
            request_id=input_plan.request_id,
            phase=segment.phase,
            token_start=segment.token_start,
            token_count=segment.token_count,
            rows=tuple(scoped_rows),
        )

    def bundle_digest(self, request_id: str) -> str:
        with self._lock:
            bundle = self._bundles.get(request_id)
            if bundle is None:
                raise FeatureAccessError(f"request {request_id!r} has no feature bundle")
            bundle.verify_integrity()
            return bundle.semantic_digest

    def release(self, request_id: str) -> None:
        with self._lock:
            bundle = self._bundles.pop(request_id, None)
            if bundle is None:
                raise FeatureAccessError(f"request {request_id!r} has no feature bundle")
            for job_id in {binding.feature.job_id for binding in bundle.bindings}:
                del self._rows[(request_id, job_id)]

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)
