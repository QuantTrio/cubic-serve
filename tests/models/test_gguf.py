"""Physical GGUF admission tests; these do not qualify model execution."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from cubic_serve.models.checkpoint import CheckpointInventoryError
from cubic_serve.models.gguf import inspect_gguf


def _string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _write(path: Path, tensors=(), metadata=(), *, duplicate_offsets=False):
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for name, value in metadata:
        kind, payload = (8, _string(value)) if isinstance(value, str) else (
            4, struct.pack("<I", value)
        )
        header += _string(name) + struct.pack("<I", kind) + payload
    data = b""
    for name, shape, type_id, payload in tensors:
        data += b"\0" * (-len(data) % 32)
        offset = 0 if duplicate_offsets else len(data)
        header += (_string(name) + struct.pack("<I", len(shape))
                   + struct.pack("<" + "Q" * len(shape), *reversed(shape))
                   + struct.pack("<IQ", type_id, offset))
        data += payload
    path.write_bytes(header + b"\0" * (-len(header) % 32) + data)


def test_reads_original_codes_and_block_aligned_columns(tmp_path):
    # IQ4_NL has 18 bytes per original group of 32, including FP16 scale.
    raw = np.arange(4 * 36, dtype=np.uint8).reshape(4, 36)
    file = tmp_path / "weights.gguf"
    _write(file, [("opaque.weight", (2, 2, 64), 20, raw.tobytes())])
    inventory = inspect_gguf(file)
    assert inventory.tensor("opaque.weight").shape == (2, 2, 64)
    np.testing.assert_array_equal(inventory.read_rows("opaque.weight", 1, 3), raw[1:3])
    np.testing.assert_array_equal(
        inventory.read_rows("opaque.weight", 2, 4, column_start=32), raw[2:4, 18:]
    )
    with pytest.raises(CheckpointInventoryError, match="scale blocks"):
        inventory.read_rows("opaque.weight", 0, 1, column_start=1)


def test_metadata_only_first_split_and_arbitrary_entry(tmp_path):
    metadata = (("split.count", 2), ("split.tensors.count", 1))
    first = tmp_path / "example-00001-of-00002.gguf"
    second = tmp_path / "example-00002-of-00002.gguf"
    _write(first, metadata=(*metadata, ("split.no", 0), ("general.architecture", "example")))
    _write(second, [("x", (1, 2), 0, struct.pack("<ff", 1, 2))],
           (*metadata, ("split.no", 1)))
    for entry in (first, second, tmp_path):
        inventory = inspect_gguf(entry)
        assert inventory.tensor("x").file_index == 1
        assert inventory.files[0].field("general.architecture") == "example"
        assert inventory.read_rows("x", 0, 1).view("<f4").tolist() == [[1, 2]]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "total", "metadata"])
def test_rejects_incomplete_or_conflicting_split_ownership(tmp_path, failure):
    metadata = (("split.count", 2), ("split.tensors.count", 3 if failure == "total" else 2))
    for split in range(2):
        if failure == "missing" and split == 1:
            continue
        name = "same" if failure == "duplicate" else f"w{split}"
        arch = str(split) if failure == "metadata" else "fixture"
        _write(tmp_path / f"weights-{split + 1:05d}-of-00002.gguf",
               [(name, (1, 1), 0, struct.pack("<f", 1))],
               (*metadata, ("split.no", split), ("general.architecture", arch)))
    with pytest.raises(CheckpointInventoryError):
        inspect_gguf(tmp_path)


def test_rejects_overlap_truncation_and_later_mutation(tmp_path):
    file = tmp_path / "weights.gguf"
    tensors = [("a", (1, 2), 0, b"\0" * 8), ("b", (1, 2), 0, b"\0" * 8)]
    _write(file, tensors, duplicate_offsets=True)
    with pytest.raises(CheckpointInventoryError, match="overlap"):
        inspect_gguf(file)
    _write(file, tensors)
    inventory = inspect_gguf(file)
    file.write_bytes(file.read_bytes()[:-1])
    with pytest.raises(CheckpointInventoryError):
        inspect_gguf(file)
    with pytest.raises(CheckpointInventoryError, match="changed"):
        inventory.read_rows("a", 0, 1)


def test_ambiguous_directory_requires_explicit_bundle(tmp_path):
    for name in ("first", "second"):
        _write(tmp_path / f"{name}.gguf", [("x", (1,), 0, b"\0" * 4)])
    with pytest.raises(CheckpointInventoryError, match="unambiguous"):
        inspect_gguf(tmp_path)
