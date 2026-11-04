"""GGUF physical storage, independent of model names and compute backends.

The upstream reader owns the GGUF grammar. This layer closes split ownership,
physical ranges and file identity, and reads original encoded rows. It neither
dequantizes weights nor qualifies an execution plan. Model packages translate
the resulting names/metadata; quantization extensions own their arithmetic.
"""

from __future__ import annotations

import hashlib
import math
import re
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cubic_serve.models.checkpoint import CheckpointInventoryError

_INVENTORY_ISSUER = object()


@dataclass(frozen=True, slots=True)
class GGUFFile:
    path: Path
    stamp: tuple[int, int, int, int, int]
    header_digest: str
    data_offset: int
    metadata: tuple[tuple[str, Any], ...]

    def field(self, name: str, default: Any = None) -> Any:
        return next((value for key, value in self.metadata if key == name), default)


@dataclass(frozen=True, slots=True)
class GGUFTensor:
    name: str
    shape: tuple[int, ...]
    type_id: int
    file_index: int
    offset: int
    byte_count: int
    row_bytes: int
    block_size: int
    block_bytes: int


@dataclass(frozen=True, slots=True, init=False)
class GGUFInventory:
    files: tuple[GGUFFile, ...]
    tensors: tuple[GGUFTensor, ...]

    def __init__(self, files: tuple[GGUFFile, ...], tensors: tuple[GGUFTensor, ...], *,
                 _token: object | None = None) -> None:
        if _token is not _INVENTORY_ISSUER:
            raise TypeError("GGUF inventories must be issued by physical inspection")
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "tensors", tensors)

    @property
    def total_tensor_bytes(self) -> int:
        return sum(tensor.byte_count for tensor in self.tensors)

    def tensor(self, name: str) -> GGUFTensor:
        position = bisect_left(self.tensors, name, key=lambda item: item.name)
        if position < len(self.tensors) and self.tensors[position].name == name:
            return self.tensors[position]
        raise CheckpointInventoryError(f"unknown GGUF tensor {name!r}")

    def read_rows(
        self, name: str, start: int, end: int, *, column_start: int = 0,
        column_end: int | None = None,
    ) -> np.ndarray:
        """Copy an encoded [row, block-byte] slice without requantization.

        Leading dimensions are flattened in original checkpoint order. An
        expert's rows therefore remain contiguous, without any expert remap.
        Column slices must preserve complete original quantization blocks.
        """
        tensor = self.tensor(name)
        columns = tensor.shape[-1]
        rows = math.prod(tensor.shape[:-1])
        column_end = columns if column_end is None else column_end
        if any(type(v) is not int for v in (start, end, column_start, column_end)):
            raise CheckpointInventoryError("GGUF row/column indices must be integers")
        if not (0 <= start < end <= rows and 0 <= column_start < column_end <= columns):
            raise CheckpointInventoryError("GGUF slice is outside the logical tensor")
        if column_start % tensor.block_size or column_end % tensor.block_size:
            raise CheckpointInventoryError("GGUF slicing must preserve original scale blocks")
        file = self.files[tensor.file_index]
        if _stamp(file.path) != file.stamp:
            raise CheckpointInventoryError("GGUF shard changed after inspection")
        first = column_start // tensor.block_size * tensor.block_bytes
        last = column_end // tensor.block_size * tensor.block_bytes
        mapped = np.memmap(file.path, mode="r", dtype=np.uint8,
                           offset=tensor.offset + start * tensor.row_bytes,
                           shape=(end - start, tensor.row_bytes))
        result = np.array(mapped[:, first:last], copy=True, order="C")
        del mapped
        if _stamp(file.path) != file.stamp:
            raise CheckpointInventoryError("GGUF shard changed during row read")
        return result


def _stamp(path: Path) -> tuple[int, int, int, int, int]:
    info = path.stat()
    if not path.is_file():
        raise CheckpointInventoryError("GGUF shard must be a regular file")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _freeze(value: Any) -> Any:
    return tuple(_freeze(v) for v in value) if isinstance(value, list) else value


def _read_file(path: Path, file_index: int) -> tuple[GGUFFile, tuple[GGUFTensor, ...]]:
    from gguf import GGML_QUANT_SIZES, GGUFEndian, GGUFReader

    before = _stamp(path)
    try:
        reader = GGUFReader(path, mode="r")
        # Native CUDA transport consumes little-endian GGML blocks. Reject a
        # swapped file rather than silently byte-swapping only its metadata.
        if reader.endianess != GGUFEndian.LITTLE:
            raise CheckpointInventoryError("big-endian GGUF payload is not supported")
        data_offset = int(reader.data_offset)
        if data_offset > before[2]:
            raise CheckpointInventoryError("GGUF tensor-data offset exceeds the file")
        metadata = tuple((key, _freeze(field.contents())) for key, field in reader.fields.items())
        records = []
        occupied = []
        names: set[str] = set()
        for tensor in reader.tensors:
            shape = tuple(int(v) for v in reversed(tensor.shape))
            if not tensor.name or tensor.name in names:
                raise CheckpointInventoryError("GGUF tensor names must be nonempty and unique")
            names.add(tensor.name)
            if not 1 <= len(shape) <= 4 or any(v <= 0 for v in shape):
                raise CheckpointInventoryError("GGUF tensors require positive logical dimensions")
            block, size = GGML_QUANT_SIZES[tensor.tensor_type]
            if shape[-1] % block:
                raise CheckpointInventoryError("GGUF row is not a complete encoding block")
            row_bytes = shape[-1] // block * size
            byte_count = math.prod(shape[:-1]) * row_bytes
            offset = int(tensor.data_offset)
            if (byte_count != tensor.n_bytes or offset < data_offset
                    or (offset - data_offset) % reader.alignment
                    or offset + byte_count > before[2]):
                raise CheckpointInventoryError("GGUF tensor has an invalid physical range")
            occupied.append((offset, offset + byte_count))
            records.append(GGUFTensor(tensor.name, shape, int(tensor.tensor_type), file_index,
                                      offset, byte_count, row_bytes, block, size))
        previous = data_offset
        for start, end in sorted(occupied):
            if start < previous:
                raise CheckpointInventoryError("GGUF tensor byte ranges overlap")
            previous = end
        digest = "b2s256:" + hashlib.blake2s(reader.data[:data_offset]).hexdigest()
        del reader
    except CheckpointInventoryError:
        raise
    except (ValueError, KeyError, IndexError, OverflowError, OSError) as error:
        raise CheckpointInventoryError(f"invalid GGUF shard {path.name!r}") from error
    if _stamp(path) != before:
        raise CheckpointInventoryError("GGUF shard changed during header inspection")
    return GGUFFile(path, before, digest, data_offset, metadata), tuple(records)


def inspect_gguf(path: str | Path) -> GGUFInventory:
    """Inspect one GGUF bundle, discovering only its explicitly named splits.

    A directory must identify one bundle. Multimodal/MTP sidecars are separate
    bundles and must be supplied explicitly by the model's checkpoint mapping.
    An empty metadata-only first split is valid when later splits own tensors.
    """
    source = Path(path).resolve(strict=True)
    if source.is_dir():
        candidates = sorted(source.glob("*.gguf"))
        heads = [p for p in candidates if not re.search(r"-\d{5}-of-\d{5}\.gguf$", p.name)
                 or re.search(r"-00001-of-\d{5}\.gguf$", p.name)]
        if len(heads) != 1:
            raise CheckpointInventoryError("GGUF directory must identify one unambiguous bundle")
        source = heads[0]
    first, first_tensors = _read_file(source, 0)
    count = first.field("split.count", 1)
    if type(count) is not int or count < 1:
        raise CheckpointInventoryError("invalid GGUF split count")
    if count == 1:
        if first.field("split.no", 0) != 0:
            raise CheckpointInventoryError("invalid single-file GGUF split number")
        files, tensors = [first], list(first_tensors)
    else:
        match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", source.name)
        if match is None or int(match[3]) != count or first.field("split.no") != int(match[2]) - 1:
            raise CheckpointInventoryError("GGUF split metadata disagrees with its filename")
        files, tensors = [], []
        for number in range(count):
            sibling = source.with_name(f"{match[1]}-{number + 1:05d}-of-{count:05d}.gguf")
            if not sibling.is_file() or sibling.resolve().parent != source.parent:
                raise CheckpointInventoryError("GGUF bundle has a missing or escaped split")
            file, records = (
                (first, first_tensors) if sibling == source else _read_file(sibling, number)
            )
            # The caller may start from any split, not just the first one.
            if sibling == source and number != 0:
                from dataclasses import replace

                records = tuple(replace(record, file_index=number) for record in records)
            if file.field("split.count") != count or file.field("split.no") != number:
                raise CheckpointInventoryError("GGUF split ownership is inconsistent")
            files.append(file)
            tensors.extend(records)
        totals = {file.field("split.tensors.count") for file in files}
        if totals != {len(tensors)}:
            raise CheckpointInventoryError("GGUF split tensor total is inconsistent")
    if not tensors or len({tensor.name for tensor in tensors}) != len(tensors):
        raise CheckpointInventoryError("GGUF bundle tensor ownership is empty or duplicated")
    # Header metadata may live only in split zero. Any repeats must agree.
    shared: dict[str, Any] = {}
    for file in files:
        for key, value in file.metadata:
            if key.startswith(("GGUF.", "split.")):
                continue
            if key in shared and shared[key] != value:
                raise CheckpointInventoryError(f"conflicting GGUF metadata field {key!r}")
            shared[key] = value
    return GGUFInventory(tuple(files), tuple(sorted(tensors, key=lambda record: record.name)),
                         _token=_INVENTORY_ISSUER)
