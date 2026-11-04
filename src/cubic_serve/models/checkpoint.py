"""Read-only, core-issued inventory for local safetensors checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from decimal import Decimal
from cubic_serve._compat import StrEnum
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Protocol, cast

from safetensors import SafetensorError, safe_open

from cubic_serve import cache_root

_INVENTORY_SCHEMA = "cubic.checkpoint.inventory"
_SCHEMA_VERSION = 1
_ISSUER_TOKEN = object()
_HASH_CHUNK_BYTES = 4 * 1024 * 1024
_CACHE_SCHEMA = "cubic.checkpoint.inventory-cache"
# Older readers assume every cached index had an exact total_size. Keep them
# from consuming a header-only inventory issued under the relaxed size policy.
_CACHE_VERSION = 2
_MAX_CACHE_BYTES = 64 * 1024 * 1024

# safetensors 0.8 exposes physical ordering through offset_keys() but does not
# expose numeric data_offsets. This evidence identifier makes that boundary
# explicit instead of implying byte-offset evidence.
SAFETENSORS_OFFSET_EVIDENCE = "safetensors.safe_open.offset_keys:order-only"

_DTYPE_BYTES: dict[str, int] = {
    "BF16": 2,
    "BOOL": 1,
    "C64": 8,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
}


class CheckpointInventoryError(ValueError):
    """The local checkpoint cannot produce a closed, trustworthy inventory."""


class CheckpointLayout(StrEnum):
    """Supported Hugging Face safetensors layouts."""

    SINGLE_FILE = "single-file"
    SHARDED_INDEX = "sharded-index"


class _SafeSlice(Protocol):
    def get_dtype(self) -> str: ...

    def get_shape(self) -> list[int]: ...


class _SafeReader(Protocol):
    def __enter__(self) -> _SafeReader: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None: ...

    def keys(self) -> list[str]: ...

    def offset_keys(self) -> list[str]: ...

    def get_slice(self, name: str) -> _SafeSlice: ...


@dataclass(frozen=True, slots=True)
class _FileStamp:
    device: int
    inode: int
    byte_count: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True, init=False)
class CheckpointFileEvidence:
    """content digest identity of one file read while issuing the inventory."""

    relative_path: str
    byte_count: int
    content_digest: str

    def __init__(
        self,
        *,
        relative_path: str,
        byte_count: int,
        content_digest: str,
        _token: object | None = None,
    ) -> None:
        if _token is not _ISSUER_TOKEN:
            raise TypeError("checkpoint file evidence may only be issued from a local file")
        _validate_relative_path(relative_path, field="checkpoint evidence path")
        if type(byte_count) is not int or byte_count <= 0:
            raise CheckpointInventoryError("checkpoint evidence file must be non-empty")
        _require_content_digest(content_digest, field="checkpoint file content_digest")
        object.__setattr__(self, "relative_path", relative_path)
        object.__setattr__(self, "byte_count", byte_count)
        object.__setattr__(self, "content_digest", content_digest)


@dataclass(frozen=True, slots=True, init=False)
class CheckpointTensor:
    """Header-derived identity of one tensor without materializing its data.

    offset_ordinal is scoped to one shard and comes from the official
    safe_open.offset_keys() ordering. Numeric byte offsets are deliberately
    absent because safetensors 0.8 does not expose them through its public API.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    shard_path: str
    offset_ordinal: int
    numel: int
    byte_count: int

    def __init__(
        self,
        *,
        name: str,
        dtype: str,
        shape: tuple[int, ...],
        shard_path: str,
        offset_ordinal: int,
        numel: int,
        byte_count: int,
        _token: object | None = None,
    ) -> None:
        if _token is not _ISSUER_TOKEN:
            raise TypeError(
                "checkpoint tensor records may only be issued from a safetensors header"
            )
        _validate_tensor_name(name)
        if type(dtype) is not str or dtype not in _DTYPE_BYTES:
            raise CheckpointInventoryError(f"unsupported safetensors dtype: {dtype!r}")
        if type(shape) is not tuple or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise CheckpointInventoryError("checkpoint tensor shape must be non-negative integers")
        _validate_relative_path(shard_path, field="tensor shard path")
        if type(offset_ordinal) is not int or offset_ordinal < 0:
            raise CheckpointInventoryError("tensor offset ordinal must be non-negative")
        if type(numel) is not int or numel <= 0 or math.prod(shape) != numel:
            raise CheckpointInventoryError("checkpoint tensors must have a non-empty valid shape")
        if type(byte_count) is not int or byte_count != numel * _DTYPE_BYTES[dtype]:
            raise CheckpointInventoryError("tensor byte count differs from its dtype and shape")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "shard_path", shard_path)
        object.__setattr__(self, "offset_ordinal", offset_ordinal)
        object.__setattr__(self, "numel", numel)
        object.__setattr__(self, "byte_count", byte_count)


@dataclass(frozen=True, slots=True, init=False)
class CheckpointInventory:
    """Closed checkpoint inventory issued from local files, not model claims.

    expected_names is the exact input for a later FullBF16WeightPolicy.
    Classification of narrowly allowed FP32 and INT64 names remains the
    architecture policy's responsibility.
    """

    model_root: Path
    layout: CheckpointLayout
    config_file: CheckpointFileEvidence
    index_file: CheckpointFileEvidence | None
    shard_files: tuple[CheckpointFileEvidence, ...]
    tensors: tuple[CheckpointTensor, ...]
    expected_names: tuple[str, ...]
    tensor_count: int
    total_numel: int
    total_tensor_bytes: int
    offset_evidence: str
    inventory_digest: str
    payload_verified: bool

    def __init__(
        self,
        *,
        model_root: Path,
        layout: CheckpointLayout,
        config_file: CheckpointFileEvidence,
        index_file: CheckpointFileEvidence | None,
        shard_files: tuple[CheckpointFileEvidence, ...],
        tensors: tuple[CheckpointTensor, ...],
        expected_names: tuple[str, ...],
        tensor_count: int,
        total_numel: int,
        total_tensor_bytes: int,
        offset_evidence: str,
        inventory_digest: str,
        payload_verified: bool,
        _token: object | None = None,
    ) -> None:
        if _token is not _ISSUER_TOKEN:
            raise TypeError("checkpoint inventory may only be issued by checkpoint inspection")
        if not model_root.is_absolute() or not model_root.is_dir():
            raise CheckpointInventoryError("checkpoint inventory requires an absolute model root")
        if not isinstance(layout, CheckpointLayout):
            raise TypeError("layout must be a CheckpointLayout")
        if type(payload_verified) is not bool:
            raise TypeError("payload_verified must be bool")
        if layout is CheckpointLayout.SINGLE_FILE and index_file is not None:
            raise CheckpointInventoryError("single-file inventory cannot contain an index")
        if layout is CheckpointLayout.SHARDED_INDEX and index_file is None:
            raise CheckpointInventoryError("sharded inventory requires an index")
        if not shard_files:
            raise CheckpointInventoryError("checkpoint inventory requires at least one shard")
        shard_names = tuple(item.relative_path for item in shard_files)
        if shard_names != tuple(sorted(set(shard_names))):
            raise CheckpointInventoryError("checkpoint shard evidence must be unique and sorted")
        names = tuple(tensor.name for tensor in tensors)
        if names != tuple(sorted(set(names))) or expected_names != names:
            raise CheckpointInventoryError(
                "checkpoint tensor inventory must be exact, unique, and sorted"
            )
        if tensor_count != len(tensors) or tensor_count <= 0:
            raise CheckpointInventoryError("checkpoint tensor count is inconsistent")
        if total_numel != sum(tensor.numel for tensor in tensors):
            raise CheckpointInventoryError("checkpoint numel total is inconsistent")
        if total_tensor_bytes != sum(tensor.byte_count for tensor in tensors):
            raise CheckpointInventoryError("checkpoint tensor byte total is inconsistent")
        if offset_evidence != SAFETENSORS_OFFSET_EVIDENCE:
            raise CheckpointInventoryError("checkpoint offset evidence boundary is unsupported")
        _require_content_digest(inventory_digest, field="checkpoint inventory digest")
        expected_digest = _inventory_digest(
            layout=layout,
            config_file=config_file,
            index_file=index_file,
            shard_files=shard_files,
            tensors=tensors,
            payload_verified=payload_verified,
        )
        if inventory_digest != expected_digest:
            raise CheckpointInventoryError("checkpoint inventory digest differs from its evidence")
        object.__setattr__(self, "model_root", model_root)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "config_file", config_file)
        object.__setattr__(self, "index_file", index_file)
        object.__setattr__(self, "shard_files", shard_files)
        object.__setattr__(self, "tensors", tensors)
        object.__setattr__(self, "expected_names", expected_names)
        object.__setattr__(self, "tensor_count", tensor_count)
        object.__setattr__(self, "total_numel", total_numel)
        object.__setattr__(self, "total_tensor_bytes", total_tensor_bytes)
        object.__setattr__(self, "offset_evidence", offset_evidence)
        object.__setattr__(self, "inventory_digest", inventory_digest)
        object.__setattr__(self, "payload_verified", payload_verified)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _content_digest_payload(value: Any) -> str:
    digest = hashlib.blake2s(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"b2s256:{digest}"


def _require_content_digest(value: str, *, field: str) -> None:
    algorithm, separator, digest = value.partition(":")
    if algorithm != "b2s256" or separator != ":" or len(digest) != 64:
        raise CheckpointInventoryError(f"{field} must use b2s256:<64 hex chars>")
    try:
        int(digest, 16)
    except ValueError as error:
        raise CheckpointInventoryError(f"{field} contains non-hexadecimal characters") from error


def _validate_tensor_name(name: str) -> None:
    if type(name) is not str or not name or name != name.strip():
        raise CheckpointInventoryError("tensor names must be canonical non-empty strings")


def _validate_relative_path(value: str, *, field: str) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise CheckpointInventoryError(f"{field} must be a canonical POSIX relative path")
    return _canonical_relative_path(value, field=field)


@lru_cache(maxsize=4096)
def _canonical_relative_path(value: str, *, field: str) -> PurePosixPath:
    """Reuse only immutable path syntax; confinement and file stamps stay live."""
    relative = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        relative.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise CheckpointInventoryError(f"{field} must be a canonical POSIX relative path")
    return relative


def _file_stamp(path: Path) -> _FileStamp:
    try:
        details = path.stat()
    except OSError as error:
        raise CheckpointInventoryError(f"checkpoint file is unavailable: {path}") from error
    if not stat.S_ISREG(details.st_mode):
        raise CheckpointInventoryError(f"checkpoint path is not a regular file: {path}")
    return _FileStamp(
        device=details.st_dev,
        inode=details.st_ino,
        byte_count=details.st_size,
        modified_ns=details.st_mtime_ns,
        changed_ns=details.st_ctime_ns,
    )


def _confined_file(model_root: Path, relative_path: str) -> Path:
    relative = _validate_relative_path(relative_path, field="checkpoint path")
    candidate = model_root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise CheckpointInventoryError(
            f"checkpoint file is unavailable: {relative_path}"
        ) from error
    if not resolved.is_relative_to(model_root):
        raise CheckpointInventoryError(f"checkpoint path escapes model_root: {relative_path!r}")
    _file_stamp(resolved)
    return resolved


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.blake2s()
    byte_count = 0
    try:
        with path.open("rb") as source:
            while chunk := source.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise CheckpointInventoryError(f"checkpoint file cannot be read: {path}") from error
    return f"b2s256:{digest.hexdigest()}", byte_count


def _reject_nonfinite_json(value: str) -> object:
    raise CheckpointInventoryError(f"JSON constants must be finite: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointInventoryError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_json_object(
    model_root: Path,
    relative_path: str,
    *,
    label: str,
) -> tuple[dict[str, object], CheckpointFileEvidence]:
    path = _confined_file(model_root, relative_path)
    before = _file_stamp(path)
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        parsed = cast(
            object,
            json.loads(
                text,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_nonfinite_json,
                parse_float=Decimal,
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointInventoryError(f"{label} must be valid strict UTF-8 JSON") from error
    after = _file_stamp(path)
    if before != after or len(raw) != before.byte_count:
        raise CheckpointInventoryError(f"{label} changed during inspection")
    if type(parsed) is not dict:
        raise CheckpointInventoryError(f"{label} root must be a JSON object")
    evidence = CheckpointFileEvidence(
        relative_path=relative_path,
        byte_count=len(raw),
        content_digest=f"b2s256:{hashlib.blake2s(raw).hexdigest()}",
        _token=_ISSUER_TOKEN,
    )
    return cast(dict[str, object], parsed), evidence


def _index_weight_map(index: dict[str, object]) -> tuple[dict[str, str], int]:
    if set(index) != {"metadata", "weight_map"}:
        raise CheckpointInventoryError(
            "checkpoint index must contain exactly metadata and weight_map"
        )
    metadata = index["metadata"]
    if type(metadata) is not dict or set(metadata) != {"total_size"}:
        raise CheckpointInventoryError("checkpoint index metadata must contain only total_size")
    raw_total_size = cast(dict[str, object], metadata)["total_size"]
    if type(raw_total_size) is int:
        total_size = raw_total_size
    elif type(raw_total_size) is Decimal:
        decimal_total_size = raw_total_size
        if (
            not decimal_total_size.is_finite()
            or decimal_total_size != decimal_total_size.to_integral_value()
        ):
            raise CheckpointInventoryError(
                "checkpoint index metadata.total_size must be a non-negative integral number"
            )
        total_size = int(decimal_total_size)
    else:
        raise CheckpointInventoryError(
            "checkpoint index metadata.total_size must be a non-negative integral number"
        )
    if total_size < 0:
        raise CheckpointInventoryError(
            "checkpoint index metadata.total_size must be a non-negative integral number"
        )
    weight_map = index["weight_map"]
    if type(weight_map) is not dict or not weight_map:
        raise CheckpointInventoryError("checkpoint index weight_map must be a non-empty object")
    typed_map: dict[str, str] = {}
    for name, shard_path in cast(dict[str, object], weight_map).items():
        _validate_tensor_name(name)
        if type(shard_path) is not str or not shard_path.endswith(".safetensors"):
            raise CheckpointInventoryError(
                f"checkpoint index shard for {name!r} must be a .safetensors path"
            )
        _validate_relative_path(shard_path, field=f"checkpoint shard for {name!r}")
        typed_map[name] = shard_path
    return typed_map, total_size


def _discover_safetensors(model_root: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    resolved_paths: dict[Path, str] = {}
    try:
        candidates = tuple(model_root.rglob("*.safetensors"))
    except OSError as error:
        raise CheckpointInventoryError("model_root cannot be scanned") from error
    for candidate in candidates:
        relative_path = candidate.relative_to(model_root).as_posix()
        resolved = _confined_file(model_root, relative_path)
        previous = resolved_paths.get(resolved)
        if previous is not None:
            raise CheckpointInventoryError(
                f"duplicate shard paths resolve to one file: {previous!r}, {relative_path!r}"
            )
        resolved_paths[resolved] = relative_path
        discovered[relative_path] = resolved
    return discovered


def _read_shard_tensors(
    model_root: Path,
    shard_path: str,
    *,
    verify_payloads: bool,
) -> tuple[tuple[CheckpointTensor, ...], CheckpointFileEvidence]:
    path = _confined_file(model_root, shard_path)
    before = _file_stamp(path)
    try:
        reader = cast(
            _SafeReader,
            safe_open(str(path), framework="pt", device="cpu", backend="pread"),
        )
        with reader:
            names = reader.keys()
            offset_names = reader.offset_keys()
            if (
                len(names) != len(set(names))
                or len(offset_names) != len(set(offset_names))
                or set(names) != set(offset_names)
            ):
                raise CheckpointInventoryError(
                    f"safetensors header has non-closed offset keys: {shard_path}"
                )
            tensors: list[CheckpointTensor] = []
            for ordinal, name in enumerate(offset_names):
                _validate_tensor_name(name)
                tensor_slice = reader.get_slice(name)
                dtype = tensor_slice.get_dtype()
                raw_shape = tensor_slice.get_shape()
                if type(dtype) is not str or type(raw_shape) is not list:
                    raise CheckpointInventoryError(
                        f"safetensors returned invalid header metadata for {name!r}"
                    )
                shape = tuple(raw_shape)
                if any(type(dimension) is not int or dimension < 0 for dimension in shape):
                    raise CheckpointInventoryError(
                        f"safetensors returned an invalid shape for {name!r}"
                    )
                try:
                    dtype_bytes = _DTYPE_BYTES[dtype]
                except KeyError as error:
                    raise CheckpointInventoryError(
                        f"unsupported safetensors dtype for {name!r}: {dtype!r}"
                    ) from error
                numel = math.prod(shape)
                tensors.append(
                    CheckpointTensor(
                        name=name,
                        dtype=dtype,
                        shape=shape,
                        shard_path=shard_path,
                        offset_ordinal=ordinal,
                        numel=numel,
                        byte_count=numel * dtype_bytes,
                        _token=_ISSUER_TOKEN,
                    )
                )
    except CheckpointInventoryError:
        raise
    except (OSError, SafetensorError, RuntimeError) as error:
        raise CheckpointInventoryError(f"invalid safetensors shard: {shard_path}") from error
    if verify_payloads:
        digest, byte_count = _hash_file(path)
    else:
        digest = _content_digest_payload(
            {
                "schema": "cubic.checkpoint.file-stamp",
                "version": 1,
                "relative_path": shard_path,
                "device": before.device,
                "inode": before.inode,
                "byte_count": before.byte_count,
                "modified_ns": before.modified_ns,
                "changed_ns": before.changed_ns,
            }
        )
        byte_count = before.byte_count
    after = _file_stamp(path)
    if before != after or byte_count != before.byte_count:
        raise CheckpointInventoryError(f"checkpoint shard changed during inspection: {shard_path}")
    if not tensors:
        raise CheckpointInventoryError(f"safetensors shard contains no tensors: {shard_path}")
    evidence = CheckpointFileEvidence(
        relative_path=shard_path,
        byte_count=byte_count,
        content_digest=digest,
        _token=_ISSUER_TOKEN,
    )
    return tuple(tensors), evidence


def inspect_checkpoint_file(
    model_root: str | Path, relative_path: str, *, verify_payload: bool = False,
) -> CheckpointFileEvidence:
    """Issue physical-file evidence for a container-owned checkpoint reader.

    This attests one confined file, not a tensor inventory or numerical result.
    Header-only admission retains the same stamp identity used by safetensors;
    small semantic/config files may request a complete content digest.
    """
    root = Path(model_root).resolve(strict=True)
    path = _confined_file(root, relative_path)
    before = _file_stamp(path)
    digest, size = (_hash_file(path) if verify_payload
                    else (_stamp_digest(relative_path, before), before.byte_count))
    if _file_stamp(path) != before or size != before.byte_count:
        raise CheckpointInventoryError("checkpoint file changed during attestation")
    return CheckpointFileEvidence(relative_path=relative_path, byte_count=size,
                                  content_digest=digest, _token=_ISSUER_TOKEN)


def _file_payload(evidence: CheckpointFileEvidence) -> dict[str, object]:
    return {
        "relative_path": evidence.relative_path,
        "byte_count": evidence.byte_count,
        "content_digest": evidence.content_digest,
    }


def _tensor_payload(tensor: CheckpointTensor) -> dict[str, object]:
    return {
        "name": tensor.name,
        "dtype": tensor.dtype,
        "shape": tensor.shape,
        "shard_path": tensor.shard_path,
        "offset_ordinal": tensor.offset_ordinal,
        "numel": tensor.numel,
        "byte_count": tensor.byte_count,
    }


def _inventory_digest(
    *,
    layout: CheckpointLayout,
    config_file: CheckpointFileEvidence,
    index_file: CheckpointFileEvidence | None,
    shard_files: tuple[CheckpointFileEvidence, ...],
    tensors: tuple[CheckpointTensor, ...],
    payload_verified: bool,
) -> str:
    payload: dict[str, object] = {
        "schema": _INVENTORY_SCHEMA,
        "version": _SCHEMA_VERSION,
        "layout": layout.value,
        "config_file": _file_payload(config_file),
        "index_file": None if index_file is None else _file_payload(index_file),
        "shard_files": tuple(_file_payload(item) for item in shard_files),
        "offset_evidence": SAFETENSORS_OFFSET_EVIDENCE,
        "tensors": tuple(_tensor_payload(tensor) for tensor in tensors),
    }
    if not payload_verified:
        payload["payload_verification"] = "header-and-file-stamp"
    return _content_digest_payload(payload)


def _require_exact_shards(model_root: Path, expected_paths: set[str]) -> None:
    discovered = set(_discover_safetensors(model_root))
    missing = sorted(expected_paths - discovered)
    unknown = sorted(discovered - expected_paths)
    if missing or unknown:
        raise CheckpointInventoryError(
            f"checkpoint shard set is not closed: missing={missing}, unknown={unknown}"
        )


def _stamp_digest(relative_path: str, stamp: _FileStamp) -> str:
    return _content_digest_payload(
        {
            "schema": "cubic.checkpoint.file-stamp",
            "version": 1,
            "relative_path": relative_path,
            "device": stamp.device,
            "inode": stamp.inode,
            "byte_count": stamp.byte_count,
            "modified_ns": stamp.modified_ns,
            "changed_ns": stamp.changed_ns,
        }
    )


def _inventory_cache_path(model_root: Path) -> Path:
    root_digest = hashlib.blake2s(os.fsencode(model_root)).hexdigest()
    return cache_root() / "checkpoints" / f"inventory-{root_digest}.json"


def _cache_file_evidence(raw: object) -> CheckpointFileEvidence:
    if type(raw) is not dict or set(raw) != {"relative_path", "byte_count", "content_digest"}:
        raise CheckpointInventoryError("checkpoint cache contains invalid file evidence")
    item = cast(dict[str, object], raw)
    return CheckpointFileEvidence(
        relative_path=cast(str, item["relative_path"]),
        byte_count=cast(int, item["byte_count"]),
        content_digest=cast(str, item["content_digest"]),
        _token=_ISSUER_TOKEN,
    )


def _cache_tensor(raw: object) -> CheckpointTensor:
    required = {
        "name",
        "dtype",
        "shape",
        "shard_path",
        "offset_ordinal",
        "numel",
        "byte_count",
    }
    if type(raw) is not dict or set(raw) != required:
        raise CheckpointInventoryError("checkpoint cache contains invalid tensor evidence")
    item = cast(dict[str, object], raw)
    shape = item["shape"]
    if type(shape) is not list:
        raise CheckpointInventoryError("checkpoint cache tensor shape must be a list")
    return CheckpointTensor(
        name=cast(str, item["name"]),
        dtype=cast(str, item["dtype"]),
        shape=tuple(cast(list[int], shape)),
        shard_path=cast(str, item["shard_path"]),
        offset_ordinal=cast(int, item["offset_ordinal"]),
        numel=cast(int, item["numel"]),
        byte_count=cast(int, item["byte_count"]),
        _token=_ISSUER_TOKEN,
    )


def _source_evidence(inventory: CheckpointInventory) -> tuple[CheckpointFileEvidence, ...]:
    prefix = (inventory.config_file,)
    if inventory.index_file is not None:
        prefix += (inventory.index_file,)
    return (*prefix, *inventory.shard_files)


def _validate_cached_sources(
    model_root: Path,
    inventory: CheckpointInventory,
    raw_stamps: object,
) -> None:
    if type(raw_stamps) is not dict:
        raise CheckpointInventoryError("checkpoint cache file stamps must be an object")
    stamps = cast(dict[str, object], raw_stamps)
    evidence = _source_evidence(inventory)
    if set(stamps) != {item.relative_path for item in evidence}:
        raise CheckpointInventoryError("checkpoint cache source set is not closed")
    shard_paths = {item.relative_path for item in inventory.shard_files}
    for item in evidence:
        raw_stamp = stamps[item.relative_path]
        if type(raw_stamp) is not list or len(raw_stamp) != 5:
            raise CheckpointInventoryError("checkpoint cache contains an invalid file stamp")
        stamp = _file_stamp(_confined_file(model_root, item.relative_path))
        if [
            stamp.device,
            stamp.inode,
            stamp.byte_count,
            stamp.modified_ns,
            stamp.changed_ns,
        ] != raw_stamp:
            raise CheckpointInventoryError("checkpoint source changed after inventory caching")
        if item.relative_path in shard_paths:
            if item.content_digest != _stamp_digest(item.relative_path, stamp):
                raise CheckpointInventoryError(
                    "checkpoint shard stamp differs from cached evidence"
                )
        else:
            digest, byte_count = _hash_file(model_root / item.relative_path)
            if digest != item.content_digest or byte_count != item.byte_count:
                raise CheckpointInventoryError("checkpoint metadata differs from cached evidence")
    _require_exact_shards(model_root, shard_paths)


def _load_inventory_cache(model_root: Path) -> CheckpointInventory | None:
    path = _inventory_cache_path(model_root)
    try:
        if _file_stamp(path).byte_count > _MAX_CACHE_BYTES:
            return None
        raw = json.loads(
            path.read_bytes(),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
        if type(raw) is not dict:
            return None
        document = cast(dict[str, object], raw)
        required = {
            "schema",
            "version",
            "model_root",
            "layout",
            "config_file",
            "index_file",
            "shard_files",
            "tensors",
            "offset_evidence",
            "inventory_digest",
            "file_stamps",
        }
        if (
            set(document) != required
            or document["schema"] != _CACHE_SCHEMA
            or document["version"] != _CACHE_VERSION
            or document["model_root"] != os.fspath(model_root)
        ):
            return None
        raw_shards = document["shard_files"]
        raw_tensors = document["tensors"]
        if type(raw_shards) is not list or type(raw_tensors) is not list:
            return None
        raw_index = document["index_file"]
        tensors = tuple(_cache_tensor(item) for item in raw_tensors)
        inventory = CheckpointInventory(
            model_root=model_root,
            layout=CheckpointLayout(cast(str, document["layout"])),
            config_file=_cache_file_evidence(document["config_file"]),
            index_file=None if raw_index is None else _cache_file_evidence(raw_index),
            shard_files=tuple(_cache_file_evidence(item) for item in raw_shards),
            tensors=tensors,
            expected_names=tuple(item.name for item in tensors),
            tensor_count=len(tensors),
            total_numel=sum(item.numel for item in tensors),
            total_tensor_bytes=sum(item.byte_count for item in tensors),
            offset_evidence=cast(str, document["offset_evidence"]),
            inventory_digest=cast(str, document["inventory_digest"]),
            payload_verified=False,
            _token=_ISSUER_TOKEN,
        )
        _validate_cached_sources(model_root, inventory, document["file_stamps"])
        return inventory
    except (CheckpointInventoryError, OSError, TypeError, ValueError, KeyError):
        return None


def _store_inventory_cache(inventory: CheckpointInventory) -> None:
    evidence = _source_evidence(inventory)
    stamps: dict[str, list[int]] = {}
    shard_paths = {item.relative_path for item in inventory.shard_files}
    for item in evidence:
        stamp = _file_stamp(_confined_file(inventory.model_root, item.relative_path))
        if item.relative_path in shard_paths and item.content_digest != _stamp_digest(
            item.relative_path, stamp
        ):
            return
        stamps[item.relative_path] = [
            stamp.device,
            stamp.inode,
            stamp.byte_count,
            stamp.modified_ns,
            stamp.changed_ns,
        ]
    document = {
        "schema": _CACHE_SCHEMA,
        "version": _CACHE_VERSION,
        "model_root": os.fspath(inventory.model_root),
        "layout": inventory.layout.value,
        "config_file": _file_payload(inventory.config_file),
        "index_file": (
            None if inventory.index_file is None else _file_payload(inventory.index_file)
        ),
        "shard_files": [_file_payload(item) for item in inventory.shard_files],
        "tensors": [_tensor_payload(item) for item in inventory.tensors],
        "offset_evidence": inventory.offset_evidence,
        "inventory_digest": inventory.inventory_digest,
        "file_stamps": stamps,
    }
    path = _inventory_cache_path(inventory.model_root)
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as target:
                target.write(_canonical_json(document))
                target.flush()
                os.fsync(target.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)
    except OSError:
        return


def _inspect_checkpoint_uncached(
    model_root: str | os.PathLike[str],
    *,
    verify_payloads: bool = True,
    require_index_size: bool = True,
) -> CheckpointInventory:
    """Inspect a local checkpoint without loading tensor payloads.

    Tensor name, dtype, shape, and physical offset ordering are read through
    safetensors' public lazy API. Full shard content digest is optional so production
    serving can keep payload attestation in the explicit audit path.
    """

    if type(verify_payloads) is not bool:
        raise TypeError("verify_payloads must be bool")

    try:
        root = Path(model_root).resolve(strict=True)
    except (OSError, TypeError) as error:
        raise CheckpointInventoryError("model_root must be an existing directory") from error
    if not root.is_dir():
        raise CheckpointInventoryError("model_root must be an existing directory")

    _, config_file = _read_json_object(root, "config.json", label="config.json")

    single_exists = (root / "model.safetensors").exists()
    index_exists = (root / "model.safetensors.index.json").exists()
    if single_exists and index_exists:
        raise CheckpointInventoryError(
            "checkpoint layout is ambiguous: single file and index coexist"
        )
    if not single_exists and not index_exists:
        raise CheckpointInventoryError(
            "checkpoint requires model.safetensors or its standard index"
        )

    index_file: CheckpointFileEvidence | None
    expected_total_size: int | None
    shard_paths: tuple[str, ...]
    if single_exists:
        layout = CheckpointLayout.SINGLE_FILE
        index_file = None
        expected_total_size = None
        weight_map: dict[str, str] | None = None
        shard_paths = ("model.safetensors",)
    else:
        layout = CheckpointLayout.SHARDED_INDEX
        index, index_file = _read_json_object(
            root,
            "model.safetensors.index.json",
            label="model.safetensors.index.json",
        )
        weight_map, expected_total_size = _index_weight_map(index)
        shard_paths = tuple(sorted(set(weight_map.values())))

    expected_shards = set(shard_paths)
    for shard_path in shard_paths:
        _confined_file(root, shard_path)
    _require_exact_shards(root, expected_shards)

    observed: dict[str, CheckpointTensor] = {}
    shard_evidence: dict[str, CheckpointFileEvidence] = {}
    for shard_path in shard_paths:
        shard_tensors, evidence = _read_shard_tensors(
            root,
            shard_path,
            verify_payloads=verify_payloads,
        )
        shard_evidence[shard_path] = evidence
        for tensor in shard_tensors:
            if tensor.name in observed:
                raise CheckpointInventoryError(
                    f"duplicate tensor name across checkpoint shards: {tensor.name!r}"
                )
            if weight_map is not None:
                mapped_shard = weight_map.get(tensor.name)
                if mapped_shard is None:
                    raise CheckpointInventoryError(
                        f"tensor {tensor.name!r} is missing from checkpoint index"
                    )
                if mapped_shard != shard_path:
                    raise CheckpointInventoryError(
                        f"tensor {tensor.name!r} maps to {mapped_shard!r}, "
                        f"but its header is in {shard_path!r}"
                    )
            observed[tensor.name] = tensor

    if weight_map is not None:
        indexed_names = set(weight_map)
        observed_names = set(observed)
        missing_from_headers = sorted(indexed_names - observed_names)
        missing_from_index = sorted(observed_names - indexed_names)
        if missing_from_headers or missing_from_index:
            raise CheckpointInventoryError(
                "checkpoint index/header tensor mapping is not closed: "
                f"missing_from_headers={missing_from_headers}, "
                f"missing_from_index={missing_from_index}"
            )

    tensors = tuple(observed[name] for name in sorted(observed))
    if not tensors:
        raise CheckpointInventoryError("checkpoint contains no tensors")
    total_tensor_bytes = sum(tensor.byte_count for tensor in tensors)
    if (
        require_index_size
        and expected_total_size is not None
        and expected_total_size != total_tensor_bytes
    ):
        raise CheckpointInventoryError(
            "checkpoint index metadata.total_size differs from header-derived tensor bytes"
        )

    shard_files = tuple(shard_evidence[path] for path in shard_paths)
    # Detect ordinary additions/removals or path swaps during the complete scan.
    _require_exact_shards(root, expected_shards)
    inventory_digest = _inventory_digest(
        layout=layout,
        config_file=config_file,
        index_file=index_file,
        shard_files=shard_files,
        tensors=tensors,
        payload_verified=verify_payloads,
    )
    names = tuple(tensor.name for tensor in tensors)
    return CheckpointInventory(
        model_root=root,
        layout=layout,
        config_file=config_file,
        index_file=index_file,
        shard_files=shard_files,
        tensors=tensors,
        expected_names=names,
        tensor_count=len(tensors),
        total_numel=sum(tensor.numel for tensor in tensors),
        total_tensor_bytes=total_tensor_bytes,
        offset_evidence=SAFETENSORS_OFFSET_EVIDENCE,
        inventory_digest=inventory_digest,
        payload_verified=verify_payloads,
        _token=_ISSUER_TOKEN,
    )


def inspect_checkpoint(
    model_root: str | os.PathLike[str],
    *,
    verify_payloads: bool = True,
    require_index_size: bool = True,
) -> CheckpointInventory:
    """Inspect a local checkpoint, reusing stat-validated header evidence in serving."""

    if type(verify_payloads) is not bool:
        raise TypeError("verify_payloads must be bool")
    if type(require_index_size) is not bool:
        raise TypeError("require_index_size must be bool")
    try:
        root = Path(model_root).resolve(strict=True)
    except (OSError, TypeError) as error:
        raise CheckpointInventoryError("model_root must be an existing directory") from error
    if not root.is_dir():
        raise CheckpointInventoryError("model_root must be an existing directory")
    if not verify_payloads:
        cached = _load_inventory_cache(root)
        if cached is not None:
            # Cache identity describes the actual headers, independently of
            # whether an exporter retained a stale advisory total_size. A later
            # strict caller must still enforce its own index-size requirement.
            if require_index_size and cached.index_file is not None:
                index, evidence = _read_json_object(
                    root, cached.index_file.relative_path, label="checkpoint index"
                )
                if evidence != cached.index_file:
                    cached = None
                elif _index_weight_map(index)[1] != cached.total_tensor_bytes:
                    raise CheckpointInventoryError(
                        "checkpoint index metadata.total_size differs from header-derived tensor bytes"
                    )
        if cached is not None:
            return cached
    inventory = _inspect_checkpoint_uncached(
        root, verify_payloads=verify_payloads, require_index_size=require_index_size
    )
    if not verify_payloads:
        _store_inventory_cache(inventory)
    return inventory
