"""Closed physical payload bindings for plain and encoded logical weights."""

from __future__ import annotations

import hashlib
import json
import os
import time
from bisect import bisect_left
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor

from cubic_serve.models.checkpoint import (
    CheckpointFileEvidence,
    CheckpointInventory,
    _file_stamp,
    _hash_file,
    _stamp_digest,
    inspect_checkpoint,
)
from cubic_serve.models.parallel_weights import AxisSpan, SafeTensorReader, confined_checkpoint_path
from cubic_serve.models.quantization import (
    QuantizationConfig,
    QuantizationError,
    QuantizationScheme,
)
from cubic_serve.quantization.registry import get_extension


@lru_cache(maxsize=1)
def _configure_host_allocator() -> None:
    """Bound cold page preparation by available CPUs, preserving user settings."""
    config = torch._C._accelerator_getAllocatorSettings()
    if "pinned_use_cuda_host_register:" in config or "pinned_num_register_threads:" in config:
        return
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    budget = min(8, max(1, cpus // world))
    threads = 1 << (budget.bit_length() - 1)
    # One/two-thread registration did not improve cold allocations. Keep the
    # ordinary allocator when the host cannot supply enough parallel page work.
    if threads < 4:
        return
    from cubic_serve.ops.cuda.embedding.immutable import host_registration_supported

    # The host allocator is process-wide, so its default must work on every
    # visible device, including a later consumer on a heterogeneous topology.
    if not all(host_registration_supported(i) for i in range(torch.cuda.device_count())):
        return
    settings = f"pinned_use_cuda_host_register:True,pinned_num_register_threads:{threads}"
    torch._C._accelerator_setAllocatorSettings(f"{config},{settings}" if config else settings)


def pin_host_tensor(tensor: Tensor) -> Tensor:
    """Make one immutable owned pinned copy using the common host allocator."""
    _configure_host_allocator()
    return tensor.pin_memory()


def use_pinned_host_storage(device: torch.device) -> bool:
    """Resolve declared host placement once per execution device, before loading.

    CPU reference execution needs neither pinning nor a CUDA context. Only an
    unsupported CUDA mapping capability falls back; query/allocation errors
    retain their original failure behavior.
    """
    if device.type != "cuda":
        return False
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return _pinned_host_storage_supported(device)


@lru_cache(maxsize=128)
def _pinned_host_storage_supported(device: torch.device) -> bool:
    from cubic_serve.ops.cuda.embedding.immutable import mapped_host_memory_supported

    if mapped_host_memory_supported(device):
        return True
    print(
        f"WARNING: CPU embedding storage is unsupported on {device}: "
        "mapped host memory is unavailable; falling back to GPU storage.",
        flush=True,
    )
    return False


@dataclass(frozen=True, slots=True)
class StackedMatrixSlice:
    """Model-declared matrix view of a checkpoint [E,N,K] tensor.

    This is serialization geometry, not runtime expert assignment. Admission
    requires all such views to partition the physical rows exactly once.
    """

    name: str
    shape: tuple[int, int, int]
    index: int
    rows: AxisSpan
    output_partitions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.name
            or len(self.shape) != 3
            or min(self.shape) <= 0
            or not 0 <= self.index < self.shape[0]
            or self.rows.stop > self.shape[1]
            or (
                self.output_partitions
                and (
                    min(self.output_partitions) <= 0 or sum(self.output_partitions) != self.shape[1]
                )
            )
        ):
            raise QuantizationError("invalid stacked matrix source geometry")

    def selection(self, axis: int, span: AxisSpan) -> tuple[slice, ...]:
        rows = slice(self.rows.start, self.rows.stop)
        columns = slice(None)
        if axis == 0:
            rows = slice(self.rows.start + span.start, self.rows.start + span.stop)
        else:
            columns = slice(span.start, span.stop)
        return slice(self.index, self.index + 1), rows, columns


@dataclass(frozen=True, slots=True)
class LogicalWeight:
    name: str
    shape: tuple[int, ...]
    dtype: str
    partition: str
    operation: str = "tensor"
    aliases: tuple[str, ...] = ()
    output_partitions: tuple[int, ...] = ()
    stacked_source: StackedMatrixSlice | None = None
    # Execution placement only: checkpoint identity and numerical semantics stay unchanged.
    storage: str = "device"

    def __post_init__(self) -> None:
        if not self.name or any(type(n) is not int or n <= 0 for n in self.shape):
            raise QuantizationError("logical weight identity and positive shape are required")
        if self.operation not in ("tensor", "linear", "embedding"):
            raise QuantizationError("unknown logical weight consumer")
        if self.operation != "tensor" and len(self.shape) != 2:
            raise QuantizationError("linear and embedding weights require [N,K]")
        if self.storage not in ("device", "pinned_host") or (
            self.storage != "device" and self.operation != "embedding"
        ):
            raise QuantizationError("host storage requires an explicitly declared embedding")
        if self.output_partitions and sum(self.output_partitions) != self.shape[0]:
            raise QuantizationError("logical output partitions differ from weight rows")
        if self.stacked_source is not None and self.shape != (
            self.stacked_source.rows.size,
            self.stacked_source.shape[-1],
        ):
            raise QuantizationError("stacked source differs from logical matrix shape")


@dataclass(frozen=True, slots=True)
class EncodedWeight:
    logical: LogicalWeight
    encoding: QuantizationScheme | None
    # (format-local suffix, original checkpoint name, original shard)
    payload: tuple[tuple[str, str, str], ...]
    stacked_source: StackedMatrixSlice | None = None


@dataclass(frozen=True, slots=True)
class CheckpointSourceReceipt:
    """Worker file evidence from a complete admission, without unused tensor headers.

    This is a payload-loading view, not a new closed checkpoint inventory.
    The original inventory and semantic digests remain the global identity.
    """
    model_root: Path
    config_file: CheckpointFileEvidence
    index_file: CheckpointFileEvidence | None
    shard_files: tuple[CheckpointFileEvidence, ...]
    inventory_digest: str
    payload_verified: bool
    total_tensor_bytes: int


@dataclass(frozen=True, slots=True)
class CheckpointWeights:
    inventory: CheckpointInventory | CheckpointSourceReceipt
    quantization: QuantizationConfig | None
    weights: tuple[EncodedWeight, ...]
    semantic_digest: str
    declared_total_size: int | None

    def _open_reader(self, path: Path) -> AbstractContextManager[Any]:
        """Container boundary; ownership, slicing and loading stay shared."""
        return safe_open(path, framework="pt", device="cpu")

    @classmethod
    def admit(
        cls,
        root: str | Path,
        logical_weights: Iterable[LogicalWeight],
        *,
        verify_payloads: bool = True,
        require_quantization: bool = True,
    ) -> CheckpointWeights:
        # Exporters may retain the pre-conversion size summary. Physical names,
        # shapes, byte ranges and shard ownership remain closed and attested.
        inventory = inspect_checkpoint(
            root, verify_payloads=verify_payloads, require_index_size=False
        )
        declared_size = None
        if inventory.index_file is not None:
            index = json.loads(
                (inventory.model_root / inventory.index_file.relative_path).read_text()
            )
            declared_size = index.get("metadata", {}).get("total_size")
        raw = json.loads((inventory.model_root / "config.json").read_text())
        actual_config_digest = (
            "b2s256:"
            + hashlib.blake2s((inventory.model_root / "config.json").read_bytes()).hexdigest()
        )
        if actual_config_digest != inventory.config_file.content_digest:
            raise QuantizationError("checkpoint config changed during quantized admission")
        config = QuantizationConfig.from_model_config(raw)
        if config is None and require_quantization:
            raise QuantizationError("quantized admission requires declared quantization")
        logical = tuple(sorted(logical_weights, key=lambda item: item.name))
        if not logical or len({w.name for w in logical}) != len(logical):
            raise QuantizationError("logical weight plan must be nonempty and unique")
        records = {record.name: record for record in inventory.tensors}
        consumed: set[str] = set()
        stacked_rows: dict[str, list[tuple[int, int]]] = {}
        stacked_totals: dict[str, int] = {}
        bindings: list[EncodedWeight] = []
        for weight in logical:
            module = weight.name.removesuffix(".weight")
            encoding = None
            if config is not None and weight.operation != "tensor":
                encoding = config.resolve(
                    module, aliases=weight.aliases, operation=weight.operation
                )
            if encoding is not None:
                admit_encoding = getattr(
                    get_extension(encoding.format).module("layout"), "admit_encoding", None
                )
                if admit_encoding is not None:
                    encoding = admit_encoding(weight, encoding, records)
            expected = _payload_headers(weight, encoding)
            names = [
                weight.name if encoding is None else module + "." + suffix
                for suffix, _, _ in expected
            ]
            source = None
            if not any(name in records for name in names) and weight.stacked_source is not None:
                source = weight.stacked_source
                if encoding is None:
                    expected = (("weight", weight.dtype, source.shape),)
                    names = [source.name]
                else:
                    layout = get_extension(encoding.format).module("layout")
                    naming = getattr(layout, "stacked_payload_name", None)
                    if naming is None:
                        raise QuantizationError(
                            f"{encoding.format} does not declare stacked payload serialization"
                        )
                    full = replace(
                        weight,
                        shape=source.shape[1:],
                        stacked_source=None,
                        output_partitions=source.output_partitions,
                    )
                    expected = tuple(
                        (key, dtype, (source.shape[0], *shape))
                        for key, dtype, shape in _payload_headers(full, encoding)
                    )
                    names = [naming(source.name, key) for key, _, _ in expected]
            bound = []
            for (suffix, dtype, shape), name in zip(expected, names, strict=True):
                record = records.get(name)
                if record is None:
                    raise QuantizationError(f"missing payload {name!r} for {weight.name!r}")
                plain_float = (
                    encoding is None and dtype == "BF16" and record.dtype in ("BF16", "F16", "F32")
                )
                if (record.dtype != dtype and not plain_float) or record.shape != shape:
                    raise QuantizationError(
                        f"payload {name!r}: expected {dtype}{shape}, "
                        f"got {record.dtype}{record.shape}"
                    )
                if name in consumed and (source is None or name not in stacked_rows):
                    raise QuantizationError(
                        f"physical payload {name!r} has multiple logical owners"
                    )
                consumed.add(name)
                if source is not None:
                    total = source.shape[0] * source.shape[1]
                    if stacked_totals.setdefault(name, total) != total:
                        raise QuantizationError("inconsistent stacked payload logical geometry")
                    start = source.index * source.shape[1] + source.rows.start
                    stacked_rows.setdefault(name, []).append((start, start + source.rows.size))
                bound.append((suffix, name, record.shard_path))
            bindings.append(EncodedWeight(weight, encoding, tuple(bound), source))
        for name, intervals in stacked_rows.items():
            cursor = 0
            for start, stop in sorted(intervals):
                if start != cursor:
                    raise QuantizationError(
                        f"stacked payload {name!r} has overlapping or missing owners"
                    )
                cursor = stop
            if cursor != stacked_totals[name]:
                raise QuantizationError(f"stacked payload {name!r} has unowned rows")
        unknown = sorted(set(records) - consumed)
        if unknown:
            raise QuantizationError(f"checkpoint has {len(unknown)} unowned tensors: {unknown[:8]}")
        canonical = [
            (
                w.logical.name,
                w.logical.shape,
                w.logical.dtype,
                w.logical.partition,
                w.logical.operation,
                w.logical.output_partitions,
                repr(w.encoding),
                w.payload,
                repr(w.stacked_source),
            )
            for w in bindings
        ]
        digest = (
            "b2s256:"
            + hashlib.blake2s(
                json.dumps(
                    [inventory.inventory_digest, config.digest if config else None, canonical],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
        return cls(inventory, config, tuple(bindings), digest, declared_size)

    def validate_for(self, root: str | Path, logical_weights: Iterable[LogicalWeight]) -> None:
        """Reuse this startup admission without repeating header/rule discovery."""
        if not isinstance(self.inventory, CheckpointInventory):
            raise QuantizationError("payload view cannot replace complete checkpoint admission")
        if Path(root).resolve(strict=True) != self.inventory.model_root:
            raise QuantizationError("checkpoint admission belongs to another model root")
        logical = tuple(sorted(logical_weights, key=lambda item: item.name))
        if logical != tuple(weight.logical for weight in self.weights):
            raise QuantizationError("checkpoint admission differs from the logical weight plan")
        for evidence in (self.inventory.config_file, self.inventory.index_file):
            if evidence is None:
                continue
            path = confined_checkpoint_path(self.inventory.model_root, evidence.relative_path)
            if path.stat().st_size != evidence.byte_count or _hash_file(path)[0] != evidence.content_digest:
                raise QuantizationError("checkpoint metadata changed after admission")
        for evidence in self.inventory.shard_files:
            self.validate_shard(evidence.relative_path)

    def payload_view(self, *, partitions: tuple[str, ...] | None = None) -> CheckpointWeights:
        """Transport admitted payload bindings and their original file witnesses."""
        weights = self.weights if partitions is None else tuple(
            weight for weight in self.weights if weight.logical.partition in partitions)
        if not weights:
            raise QuantizationError("payload view has no admitted logical weights")
        shards = {shard for weight in weights for _, _, shard in weight.payload}
        source = self.inventory
        receipt = CheckpointSourceReceipt(
            source.model_root, source.config_file, source.index_file,
            tuple(item for item in source.shard_files if item.relative_path in shards),
            source.inventory_digest, source.payload_verified, source.total_tensor_bytes)
        return replace(self, inventory=receipt, weights=weights)

    def read(
        self, weight: EncodedWeight, *, device: torch.device | str = "cpu"
    ) -> dict[str, Tensor]:
        position = bisect_left(self.weights, weight.logical.name, key=lambda w: w.logical.name)
        if position == len(self.weights) or self.weights[position] is not weight:
            raise QuantizationError("payload binding belongs to another checkpoint plan")
        result: dict[str, Tensor] = {}
        for shard in sorted({s for _, _, s in weight.payload}):
            path = confined_checkpoint_path(self.inventory.model_root, shard)
            self.validate_shard(shard)
            before = _file_stamp(path)
            with self._open_reader(path) as reader:
                for suffix, name, owner in weight.payload:
                    if owner == shard:
                        value = (
                            reader.get_tensor(name)
                            if weight.stacked_source is None
                            else reader.get_slice(name)[
                                stacked_payload_selection(
                                    weight, suffix, 0, AxisSpan(0, weight.logical.shape[0])
                                )
                            ].squeeze(0)
                        )
                        result[suffix] = value.to(device=device)
            if _file_stamp(path) != before:
                raise QuantizationError("checkpoint shard changed while reading its payload")
        return result

    def validate_shard(self, shard: str) -> None:
        """Recheck the existing inventory attestation before materialization."""
        evidence = next((f for f in self.inventory.shard_files if f.relative_path == shard), None)
        if evidence is None:
            raise QuantizationError("payload shard is outside the admitted inventory")
        path = confined_checkpoint_path(self.inventory.model_root, shard)
        stamp = _file_stamp(path)
        actual = (
            _hash_file(path)[0] if self.inventory.payload_verified else _stamp_digest(shard, stamp)
        )
        if actual != evidence.content_digest or stamp.byte_count != evidence.byte_count:
            raise QuantizationError("checkpoint shard changed after quantized admission")

    def materialize(
        self,
        weight: EncodedWeight,
        *,
        device: torch.device | str = "cpu",
        compute_dtype: torch.dtype = torch.bfloat16,
        axis: int = 0,
        spans: tuple[AxisSpan, ...] | None = None,
        _readers: Mapping[str, SafeTensorReader] | None = None,
        _execution_device: torch.device | str | None = None,
    ) -> Tensor | QuantizedMatrix:
        """Read logical owner spans directly from original serialized payloads.

        Placement is supplied by the model's ownership plan. This loader neither
        assigns experts nor changes activation-scale or collective semantics.
        Only unencoded execution weights follow compute_dtype.
        """
        position = bisect_left(self.weights, weight.logical.name, key=lambda w: w.logical.name)
        if position == len(self.weights) or self.weights[position] is not weight:
            raise QuantizationError("payload binding belongs to another checkpoint plan")
        if compute_dtype not in (torch.float16, torch.bfloat16):
            raise QuantizationError("plain execution weights require FP16 or BF16")
        logical, encoding = weight.logical, weight.encoding
        host = logical.storage == "pinned_host" and use_pinned_host_storage(
            torch.device(device if _execution_device is None else _execution_device)
        )
        if host:
            device = "cpu"
        if type(axis) is not int or not 0 <= axis < len(logical.shape):
            raise QuantizationError("logical slice axis is out of bounds")
        spans = (AxisSpan(0, logical.shape[axis]),) if spans is None else spans
        if not spans or any(span.stop > logical.shape[axis] for span in spans):
            raise QuantizationError("logical owner spans are empty or out of bounds")
        if any(left.stop > right.start for left, right in pairwise(spans)):
            raise QuantizationError("logical owner spans must retain disjoint checkpoint order")
        if encoding is not None and axis == 1 and len(spans) != 1:
            raise QuantizationError("packed input ownership requires one contiguous span")
        if encoding is not None:
            materialize = getattr(
                get_extension(encoding.format).module("layout"), "materialize", None
            )
            if materialize is not None:
                matrix = materialize(self, weight, axis, spans, device, _readers)
                if host:
                    matrix = QuantizedMatrix(
                        {name: pin_host_tensor(tensor) for name, tensor in matrix.payload.items()},
                        matrix.encoding, matrix.shape, matrix.partitions,
                    )
                return matrix
        # Validate every format boundary before reading or allocating any payload.
        selections = {}
        partitions: tuple[int, ...] = ()
        for suffix, _, _ in weight.payload:
            if encoding is None:
                indices = []
                for span in spans:
                    if weight.stacked_source is not None:
                        indices.append(weight.stacked_source.selection(axis, span))
                    else:
                        index = [slice(None)] * len(logical.shape)
                        index[axis] = slice(span.start, span.stop)
                        indices.append(tuple(index))
                selections[suffix] = (indices, axis)
            else:
                indices = []
                selected_parts: list[int] = []
                for span in spans:
                    packed_index, parts = _packed_slice_index(
                        logical.shape,
                        logical.output_partitions,
                        encoding,
                        suffix,
                        axis,
                        span.start,
                        span.stop,
                    )
                    indices.append(
                        stacked_payload_selection(weight, suffix, axis, span)
                        if weight.stacked_source is not None
                        else packed_index
                    )
                    selected_parts.extend(parts)
                partitions = tuple(selected_parts)
                # Tensor-global scales have no output-row axis.
                layout = get_extension(encoding.format).module("layout")
                scalar = suffix in layout.SCALAR_PAYLOADS
                selections[suffix] = (
                    indices[:1] if scalar else indices,
                    layout.CONCAT_AXIS,
                )
        payload = {}
        for shard in sorted({s for _, _, s in weight.payload}):
            if _readers is None:
                path = confined_checkpoint_path(self.inventory.model_root, shard)
                self.validate_shard(shard)
                before = _file_stamp(path)
                context = self._open_reader(path)
            else:
                # Private readers belong to materialize_many's source-checked
                # transaction; its final check precedes any returned weights.
                context = nullcontext(_readers[shard])
            with context as reader:
                for suffix, name, owner in weight.payload:
                    if owner != shard:
                        continue
                    indices, cat_axis = selections[suffix]
                    source = reader.get_slice(name)
                    pieces = [source[index] for index in indices]
                    if weight.stacked_source is not None:
                        pieces = [piece.squeeze(0) for piece in pieces]
                    value = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=cat_axis)
                    dtype = (
                        compute_dtype
                        if encoding is None and logical.dtype == "BF16"
                        else value.dtype
                    )
                    payload[suffix] = (
                        pin_host_tensor(value.to(dtype=dtype).contiguous()) if host else
                        value.to(device=device, dtype=dtype,
                                 copy=torch.device(device).type == "cpu").contiguous()
                    )
            if _readers is None and _file_stamp(path) != before:
                raise QuantizationError("checkpoint shard changed while reading owner spans")
        if encoding is None:
            return payload["weight"]
        shape = list(logical.shape)
        shape[axis] = sum(span.size for span in spans)
        return QuantizedMatrix(payload, encoding, (shape[0], shape[1]), partitions)

    def materialize_many(
        self,
        requests: tuple[tuple[EncodedWeight, int, tuple[AxisSpan, ...]], ...],
        *,
        device: torch.device | str,
        compute_dtype: torch.dtype,
        packed_device: torch.device | str | None = None,
        progress: Callable[[int, int], None] | None = None,
        _request_order: Iterable[tuple[EncodedWeight, int, tuple[AxisSpan, ...]]] | None = None,
        _pending_weight: Callable[[str, Tensor | QuantizedMatrix], None] | None = None,
    ) -> dict[str, Tensor | QuantizedMatrix]:
        """Reuse attested readers for one source-checked loading transaction.

        Private pending operands may feed construction, never model execution.
        The caller must join this transaction, including its final source checks,
        before executing or publishing any constructed binding.
        """
        names = [weight.logical.name for weight, _, _ in requests]
        if len(set(names)) != len(names):
            raise QuantizationError("materialization requests have duplicate logical owners")
        shards = sorted({s for weight, _, _ in requests for _, _, s in weight.payload})
        if progress is not None:
            progress(0, len(requests))
        reported_at = time.monotonic()
        with ExitStack() as stack:
            readers = {}
            stamps = {}
            for shard in shards:
                self.validate_shard(shard)
                path = confined_checkpoint_path(self.inventory.model_root, shard)
                stamps[shard] = _file_stamp(path)
                readers[shard] = stack.enter_context(self._open_reader(path))
            result = {}
            remaining = {weight.logical.name: (weight, axis, spans)
                         for weight, axis, spans in requests}
            order = requests if _request_order is None else _request_order
            for completed, (weight, axis, spans) in enumerate(order, 1):
                expected = remaining.pop(weight.logical.name, None)
                if (expected is None or expected[0] is not weight
                        or expected[1:] != (axis, spans)):
                    raise QuantizationError("pending load order differs from admitted owners")
                destination = (packed_device if weight.encoding is not None
                               and packed_device is not None else device)
                cuda_work = (_pending_weight is not None and torch.device(device).type == "cuda"
                             and (torch.device(destination).type == "cuda"
                                  or weight.logical.storage == "pinned_host"))
                if cuda_work:
                    from cubic_serve.ops.cuda.startup import startup_cuda_lock
                    guard = startup_cuda_lock(device)
                else:
                    guard = nullcontext()
                with guard:
                    result[weight.logical.name] = self.materialize(
                        weight, axis=axis, spans=spans, device=destination,
                        compute_dtype=compute_dtype, _readers=readers,
                        _execution_device=device,
                    )
                    if _pending_weight is not None:
                        _pending_weight(weight.logical.name, result[weight.logical.name])
                    if cuda_work:
                        torch.cuda.current_stream(device).synchronize()
                if progress is not None and completed < len(requests):
                    now = time.monotonic()
                    if now - reported_at >= 0.1:
                        progress(completed, len(requests))
                        reported_at = now
            if remaining:
                raise QuantizationError("pending load order omitted admitted owners")
            for shard in shards:
                if (
                    _file_stamp(confined_checkpoint_path(self.inventory.model_root, shard))
                    != stamps[shard]
                ):
                    raise QuantizationError("checkpoint shard changed during rank materialization")
        if progress is not None:
            progress(len(requests), len(requests))
        return result


# Existing packed callers keep their admission contract. The common binding
# implementation also admits plain payloads when the model execution path asks
# for them explicitly; it never invents a quantization declaration.
QuantizedCheckpoint = CheckpointWeights


def _payload_headers(
    weight: LogicalWeight, encoding: QuantizationScheme | None
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    if encoding is None:
        return (("weight", weight.dtype, weight.shape),)
    return get_extension(encoding.format).module("layout").payload_headers(weight, encoding)


@dataclass(frozen=True, slots=True)
class QuantizedMatrix:
    """Immutable packed matrix binding; numerical execution is owned by operators."""

    payload: Mapping[str, Tensor]
    encoding: QuantizationScheme
    shape: tuple[int, int]
    partitions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        logical = LogicalWeight(
            "matrix.weight",
            self.shape,
            "BF16",
            "language",
            "linear",
            output_partitions=self.partitions,
        )
        expected = _payload_headers(logical, self.encoding)
        dtypes = {
            "I64": torch.int64,
            "BF16": torch.bfloat16,
            "I32": torch.int32,
            "F16": torch.float16,
            "F32": torch.float32,
            "F8_E4M3": torch.float8_e4m3fn,
            "U8": torch.uint8,
            "I8": torch.int8,
            "U16": torch.uint16,
        }
        if set(self.payload) != {key for key, _, _ in expected}:
            raise QuantizationError("packed matrix payload does not close its declared encoding")
        devices = set()
        for key, dtype, shape in expected:
            value = self.payload[key]
            if (
                not isinstance(value, Tensor)
                or value.dtype != dtypes[dtype]
                or tuple(value.shape) != shape
                or not value.is_contiguous()
            ):
                raise QuantizationError(
                    f"packed matrix payload {key!r} has an invalid physical ABI"
                )
            devices.add(value.device)
        if len(devices) != 1:
            raise QuantizationError("packed matrix payload must share one device")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def to(self, device: torch.device | str) -> QuantizedMatrix:
        """Copy encoded storage without changing any payload dtype or semantics."""
        target = torch.device(device)
        if next(iter(self.payload.values())).device == target:
            return self
        return QuantizedMatrix(
            {name: tensor.to(target) for name, tensor in self.payload.items()},
            self.encoding,
            self.shape,
            self.partitions,
        )

    def rows(self, start: int, end: int) -> QuantizedMatrix:
        return self._slice(0, start, end)

    def columns(self, start: int, end: int) -> QuantizedMatrix:
        """Preserve codes/groups; the consumer still owns global activation scales."""
        return self._slice(1, start, end)

    def _slice(self, axis: int, start: int, end: int) -> QuantizedMatrix:
        if axis not in (0, 1) or not 0 <= start < end <= self.shape[axis]:
            raise QuantizationError("quantized slice is out of bounds")
        slicer = getattr(get_extension(self.encoding.format).module("layout"), "slice_matrix", None)
        if slicer is not None:
            return slicer(self, axis, start, end)
        payload = {}
        parts: tuple[int, ...] = ()
        for key, tensor in self.payload.items():
            index, parts = _packed_slice_index(
                self.shape, self.partitions, self.encoding, key, axis, start, end
            )
            payload[key] = tensor[index].contiguous()
        shape = list(self.shape)
        shape[axis] = end - start
        return QuantizedMatrix(payload, self.encoding, (shape[0], shape[1]), parts)


def _packed_slice_index(
    shape: tuple[int, ...],
    partitions: tuple[int, ...],
    encoding: QuantizationScheme,
    key: str,
    axis: int,
    start: int,
    end: int,
) -> tuple[tuple[slice, ...], tuple[int, ...]]:
    if axis not in (0, 1) or not 0 <= start < end <= shape[axis]:
        raise QuantizationError("quantized slice is out of bounds")
    return (
        get_extension(encoding.format)
        .module("layout")
        .slice_index(shape, partitions, encoding, key, axis, start, end)
    )


def stacked_payload_selection(
    weight: EncodedWeight, key: str, axis: int, span: AxisSpan
) -> tuple[slice, ...]:
    """Compose expert/projection ownership with format-owned TP coordinates."""
    source = weight.stacked_source
    assert source is not None
    if weight.encoding is None:
        return source.selection(axis, span)
    outer, _ = _packed_slice_index(
        source.shape[1:],
        source.output_partitions,
        weight.encoding,
        key,
        0,
        source.rows.start,
        source.rows.stop,
    )
    inner, _ = _packed_slice_index(
        weight.logical.shape,
        weight.logical.output_partitions,
        weight.encoding,
        key,
        axis,
        span.start,
        span.stop,
    )
    composed = []
    for parent, child in zip(outer, inner, strict=True):
        offset = parent.start or 0
        composed.append(
            slice(
                offset + (child.start or 0),
                parent.stop if child.stop is None else offset + child.stop,
            )
        )
    return (slice(source.index, source.index + 1), *composed)
