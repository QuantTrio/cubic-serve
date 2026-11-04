"""Load-time GGUF views of existing logical weights.

Model packages declare names and inverse exporter transforms. This component
only slices original storage and joins the declared parts. Encoded operands
remain original GGML blocks; neither the model nor its forward sees a container.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from bisect import bisect_left
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from cubic_serve.models.checkpoint import inspect_checkpoint_file
from cubic_serve.models.gguf import GGUFInventory
from cubic_serve.models.parallel_weights import AxisSpan
from cubic_serve.models.quantization import QuantizationConfig, QuantizationError
from cubic_serve.models.quantized_checkpoint import (
    CheckpointSourceReceipt,
    CheckpointWeights,
    EncodedWeight,
    LogicalWeight,
    QuantizedMatrix,
)

_PLAIN_DTYPES = {0: "F32", 1: "F16", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
                 28: "F64", 30: "BF16"}
_NUMPY_DTYPES = {0: "<f4", 1: "<f2", 24: "i1", 25: "<i2", 26: "<i4", 27: "<i8", 28: "<f8"}


@dataclass(frozen=True, slots=True)
class GGUFPart:
    """A rectangle in flattened original row order, never expert ownership."""

    inventory: GGUFInventory
    name: str
    rows: AxisSpan | None = None
    columns: AxisSpan | None = None
    reshape: tuple[int, ...] = ()

    @property
    def region(self) -> tuple[AxisSpan, AxisSpan]:
        tensor = self.inventory.tensor(self.name)
        rows = self.rows or AxisSpan(0, math.prod(tensor.shape[:-1]))
        columns = self.columns or AxisSpan(0, tensor.shape[-1])
        if rows.stop > math.prod(tensor.shape[:-1]) or columns.stop > tensor.shape[-1]:
            raise QuantizationError("GGUF source rectangle exceeds its original tensor")
        if columns.start % tensor.block_size or columns.stop % tensor.block_size:
            raise QuantizationError("GGUF source rectangle splits an original scale block")
        return rows, columns

    @property
    def shape(self) -> tuple[int, ...]:
        rows, columns = self.region
        if self.reshape:
            if any(n <= 0 for n in self.reshape) or math.prod(self.reshape) != (
                rows.size * columns.size
            ):
                raise QuantizationError("GGUF source reshape changes the element count")
            return self.reshape
        return rows.size, columns.size


@dataclass(frozen=True, slots=True)
class GGUFMetadataTensor:
    """An integer tensor exported into an attested GGUF header."""

    inventory: GGUFInventory
    key: str

    def read(self) -> Tensor:
        values = self.inventory.files[0].field(self.key)
        if not isinstance(values, tuple) or any(type(n) is not int for n in values):
            raise QuantizationError("GGUF integer metadata tensor is absent or not integral")
        return torch.tensor(values, dtype=torch.int64)


def _permutation(order: tuple[int, ...], size: int, *, label: str) -> None:
    if order and (len(order) != size or sorted(order) != list(range(size))):
        raise QuantizationError(f"GGUF {label} must be a complete coordinate permutation")


@dataclass(frozen=True, slots=True)
class GGUFWeightView:
    """Lazy serialization view with the existing model's logical identity."""

    logical: LogicalWeight
    parts: tuple[GGUFPart, ...] = ()
    join_axis: int = 0
    row_order: tuple[int, ...] = ()
    column_order: tuple[int, ...] = ()
    transform: Callable[[Tensor], Tensor] | None = None
    metadata: GGUFMetadataTensor | None = None

    @property
    def type_id(self) -> int | None:
        if self.metadata is not None:
            return None
        types = {part.inventory.tensor(part.name).type_id for part in self.parts}
        if len(types) != 1:
            raise QuantizationError("joined GGUF weights must preserve one original encoding")
        return types.pop()

    @property
    def encoded(self) -> bool:
        return self.type_id is not None and self.type_id not in _PLAIN_DTYPES

    def validate(self) -> None:
        if self.metadata is not None:
            if self.parts or self.row_order or self.column_order or self.transform is not None:
                raise QuantizationError("GGUF metadata and payload views are mutually exclusive")
            value = self.metadata.read()
            if self.logical.dtype != "I64" or tuple(value.shape) != self.logical.shape:
                raise QuantizationError("GGUF metadata differs from the logical integer tensor")
            return
        if not self.parts:
            raise QuantizationError("GGUF logical weight has no source")
        shapes = [part.shape for part in self.parts]
        rank = len(shapes[0])
        if not 0 <= self.join_axis < rank or any(len(s) != rank for s in shapes):
            raise QuantizationError("GGUF source join has incompatible dimensions")
        shape = list(shapes[0])
        for other in shapes[1:]:
            if any(shape[i] != other[i] for i in range(rank) if i != self.join_axis):
                raise QuantizationError("GGUF source join changes non-concatenated dimensions")
            shape[self.join_axis] += other[self.join_axis]
        if math.prod(shape) != math.prod(self.logical.shape):
            raise QuantizationError("GGUF view differs from the logical element count")
        if self.encoded and (
            self.logical.operation == "tensor" or len(self.logical.shape) != 2
            or tuple(shape) != self.logical.shape or self.join_axis != 0
            or self.transform is not None
        ):
            raise QuantizationError("encoded GGUF views must remain original matrix blocks")
        if not self.encoded:
            dtype = _PLAIN_DTYPES[self.parts[0].inventory.tensor(self.parts[0].name).type_id]
            compatible = (self.logical.dtype == "BF16" and dtype in ("BF16", "F16", "F32"))
            if dtype != self.logical.dtype and not compatible:
                raise QuantizationError(
                    "GGUF plain storage differs from its logical dtype contract")
        _permutation(self.row_order, shape[0], label="row order")
        _permutation(self.column_order, shape[-1], label="column order")
        if self.column_order:
            block = self.parts[0].inventory.tensor(self.parts[0].name).block_size
            columns = np.asarray(self.column_order).reshape(-1, block)
            if (np.any(columns[:, 0] % block)
                    or np.any(columns - columns[:, :1] != np.arange(block))):
                raise QuantizationError("GGUF column permutation splits original scale blocks")

    @torch.inference_mode(False)
    def read(self, selection: tuple[slice, ...] | None = None) -> Tensor:
        """Read plain logical coordinates or encoded [row, byte] coordinates.

        Production admission validates the view once before calling this method.
        Reads retain the source inventory's before/after file-identity checks.
        Large encoded embeddings are sliced before any host allocation.
        Checkpoint tensors retain mutation counters even if their caller is
        running inference. This preserves ordinary immutable-weight tracking
        and lets the independent audit reuse exactly witnessed pure math.
        """
        if not self.encoded:
            if self.metadata is not None:
                result = self.metadata.read()
            else:
                pieces = [self._plain(part) for part in self.parts]
                result = pieces[0] if len(pieces) == 1 else torch.cat(pieces, self.join_axis)
                if self.row_order:
                    result = result[list(self.row_order)]
                if self.column_order:
                    result = result[..., list(self.column_order)]
                if self.transform is not None:
                    result = self.transform(result)
                result = result.reshape(self.logical.shape)
            return result if selection is None else result[selection]

        rows, columns = self.logical.shape
        tensor = self.parts[0].inventory.tensor(self.parts[0].name)
        row_bytes = columns // tensor.block_size * tensor.block_bytes
        selection = selection or (slice(None), slice(None))
        if len(selection) != 2 or any(not isinstance(s, slice) for s in selection):
            raise QuantizationError("GGUF encoded reads require rectangular row/byte slices")
        first, last, step = selection[0].indices(rows)
        begin, end, byte_step = selection[1].indices(row_bytes)
        if (first >= last or begin >= end or step != 1 or byte_step != 1
                or begin % tensor.block_bytes or end % tensor.block_bytes):
            raise QuantizationError("GGUF read must retain complete contiguous original blocks")
        row_ids = np.arange(first, last)
        if self.row_order:
            row_ids = np.asarray(self.row_order)[row_ids]
        first_block, last_block = begin // tensor.block_bytes, end // tensor.block_bytes
        blocks = np.arange(first_block, last_block)
        if self.column_order:
            blocks = np.asarray(self.column_order).reshape(-1, tensor.block_size)[blocks, 0]
            blocks = blocks // tensor.block_size
        encoded_result = np.empty((len(row_ids), end - begin), dtype=np.uint8)
        offset = 0
        for part in self.parts:
            source_rows, source_columns = part.region
            positions = np.flatnonzero((row_ids >= offset) & (row_ids < offset + source_rows.size))
            if positions.size:
                original = row_ids[positions] - offset + source_rows.start
                # Coalesce consecutive rows, including permutations of whole heads.
                breaks = np.flatnonzero(np.diff(original) != 1) + 1
                for run in np.split(np.arange(len(original)), breaks):
                    raw = part.inventory.read_rows(
                        part.name, int(original[run[0]]), int(original[run[-1]]) + 1,
                        column_start=source_columns.start, column_end=source_columns.stop)
                    if self.column_order:
                        raw = raw.reshape(len(run), -1, tensor.block_bytes)[:, blocks].reshape(
                            len(run), -1)
                    else:
                        raw = raw[:, begin:end]
                    encoded_result[positions[run]] = raw
            offset += source_rows.size
        return torch.from_numpy(encoded_result)

    @staticmethod
    def _plain(part: GGUFPart) -> Tensor:
        record = part.inventory.tensor(part.name)
        rows, columns = part.region
        raw = part.inventory.read_rows(part.name, rows.start, rows.stop,
                                       column_start=columns.start, column_end=columns.stop)
        if record.type_id == 30:
            value = torch.from_numpy(raw.view("<u2").copy()).view(torch.bfloat16)
        elif record.type_id in _NUMPY_DTYPES:
            value = torch.from_numpy(raw.view(_NUMPY_DTYPES[record.type_id]).copy())
        else:
            raise QuantizationError("an encoded operand cannot be materialized as a plain tensor")
        return value.reshape(part.shape)


def validate_gguf_coverage(
    inventories: tuple[GGUFInventory, ...], views: tuple[GGUFWeightView, ...],
) -> None:
    """Every original element has exactly one load-time logical owner."""
    if len({v.logical.name for v in views}) != len(views):
        raise QuantizationError("GGUF logical weight declarations are duplicated")
    declared = {id(i): i for i in inventories}
    regions: dict[tuple[int, str], list[tuple[AxisSpan, AxisSpan]]] = {}
    for view in views:
        view.validate()
        if view.metadata is not None and id(view.metadata.inventory) not in declared:
            raise QuantizationError("GGUF metadata source is outside the admitted bundles")
        for part in view.parts:
            if id(part.inventory) not in declared:
                raise QuantizationError("GGUF payload source is outside the admitted bundles")
            regions.setdefault((id(part.inventory), part.name), []).append(part.region)
    for inventory in inventories:
        for tensor in inventory.tensors:
            rectangles = regions.get((id(inventory), tensor.name), [])
            events: dict[int, list[tuple[int, AxisSpan]]] = {}
            for number, (rows, columns) in enumerate(rectangles):
                events.setdefault(rows.start, []).append((number, columns))
                events.setdefault(rows.stop, []).append((number, columns))
            active: dict[int, AxisSpan] = {}
            cursor = 0
            for boundary, changes in sorted(events.items()):
                if boundary > cursor:
                    column = 0
                    for span in sorted(active.values()):
                        if span.start != column:
                            raise QuantizationError("GGUF source has missing or overlapping owners")
                        column = span.stop
                    if column != tensor.shape[-1]:
                        raise QuantizationError("GGUF source has unowned columns")
                for number, span in changes:
                    if active.pop(number, None) is None:
                        active[number] = span
                cursor = boundary
            if cursor != math.prod(tensor.shape[:-1]) or active:
                raise QuantizationError("GGUF source has unowned rows")


class _Slice:
    def __init__(self, view: GGUFWeightView):
        self.view = view

    def __getitem__(self, selection: tuple[slice, ...]) -> Tensor:
        return self.view.read(selection)


class _Reader(AbstractContextManager["_Reader"]):
    def __init__(self, views: tuple[GGUFWeightView, ...]):
        self.views = views

    def __enter__(self) -> _Reader:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def _view(self, name: str) -> GGUFWeightView:
        position = bisect_left(self.views, name, key=lambda item: item.logical.name)
        if position == len(self.views) or self.views[position].logical.name != name:
            raise QuantizationError("GGUF payload is outside the admitted logical views")
        return self.views[position]

    def get_slice(self, name: str) -> _Slice:
        return _Slice(self._view(name))

    def get_tensor(self, name: str) -> Tensor:
        return self._view(name).read()


@dataclass(frozen=True, slots=True)
class GGUFCheckpointWeights(CheckpointWeights):
    """One container reader plugged into the shared owner/materialization path.

    Its inventory is file evidence, not a fabricated safetensors inventory.
    Physical tensor closure is proved by validate_gguf_coverage at admission.
    """

    views: tuple[GGUFWeightView, ...]
    _complete: bool = True

    @classmethod
    def from_views(
        cls, root: str | Path, inventories: tuple[GGUFInventory, ...],
        views: tuple[GGUFWeightView, ...],
    ) -> GGUFCheckpointWeights:
        from cubic_serve.quantization.extensions.ggml.checkpoint import scheme

        root = Path(root).resolve(strict=True)
        validate_gguf_coverage(inventories, views)
        views = tuple(sorted(views, key=lambda item: item.logical.name))
        sources = {}
        for inventory in inventories:
            for file in inventory.files:
                relative = str(file.path.relative_to(root))
                evidence = inspect_checkpoint_file(root, relative)
                # Inspection and file evidence must describe the same source.
                from cubic_serve.models.gguf import _stamp

                if _stamp(file.path) != file.stamp:
                    raise QuantizationError("GGUF file changed before checkpoint admission")
                sources[relative] = evidence
        config = inspect_checkpoint_file(root, "config.json", verify_payload=True)
        index = (inspect_checkpoint_file(root, "gguf.index.json", verify_payload=True)
                 if (root / "gguf.index.json").is_file() else None)
        physical = tuple((str(f.path.relative_to(root)), f.header_digest, f.stamp)
                         for inv in inventories for f in inv.files)
        inventory_digest = "b2s256:" + hashlib.blake2s(repr(physical).encode()).hexdigest()
        bindings = []
        canonical = []
        for view in views:
            type_id = view.type_id
            encoding = scheme(type_id) if view.encoded and type_id is not None else None
            if view.parts:
                inventory = view.parts[0].inventory
            elif view.metadata is not None:
                inventory = view.metadata.inventory
            else:
                raise QuantizationError("GGUF source is empty after admission")
            file = (inventory.files[inventory.tensor(view.parts[0].name).file_index]
                    if view.parts else inventory.files[0])
            shard = str(file.path.relative_to(root))
            suffix = "weight_packed" if encoding is not None else "weight"
            bindings.append(EncodedWeight(view.logical, encoding,
                                           ((suffix, view.logical.name, shard),)))
            canonical.append((view.logical, repr(encoding),
                tuple((str(p.inventory.files[p.inventory.tensor(p.name).file_index].path.relative_to(
                    root)), p.name, p.rows, p.columns, p.reshape) for p in view.parts),
                view.join_axis, view.row_order, view.column_order,
                inspect.getsource(view.transform) if view.transform is not None else None,
                view.metadata.key if view.metadata is not None else None))
        semantic = "b2s256:" + hashlib.blake2s(repr((inventory_digest, config.content_digest,
            index.content_digest if index is not None else None, canonical)).encode()).hexdigest()
        receipt = CheckpointSourceReceipt(root, config, index, tuple(sources[k] for k in sorted(
            sources)), inventory_digest, False, sum(i.total_tensor_bytes for i in inventories))
        quantization = QuantizationConfig.from_model_config(
            json.loads((root / "config.json").read_bytes()))
        return cls(receipt, quantization, tuple(bindings), semantic,
                   receipt.total_tensor_bytes, views)

    def _open_reader(self, path: Path) -> _Reader:
        return _Reader(self.views)

    def validate_for(self, root: str | Path, logical_weights: Iterable[LogicalWeight]) -> None:
        if not self._complete:
            raise QuantizationError("GGUF payload view cannot replace complete admission")
        if Path(root).resolve(strict=True) != self.inventory.model_root:
            raise QuantizationError("GGUF admission belongs to another model root")
        if tuple(sorted(logical_weights, key=lambda w: w.name)) != tuple(
            w.logical for w in self.weights
        ):
            raise QuantizationError("GGUF admission differs from the logical weight plan")
        for evidence in (self.inventory.config_file, self.inventory.index_file):
            if evidence is not None and inspect_checkpoint_file(
                self.inventory.model_root, evidence.relative_path, verify_payload=True
            ) != evidence:
                raise QuantizationError("GGUF configuration changed after admission")
        for evidence in self.inventory.shard_files:
            self.validate_shard(evidence.relative_path)

    def payload_view(self, *, partitions: tuple[str, ...] | None = None) -> GGUFCheckpointWeights:
        partial = CheckpointWeights.payload_view(self, partitions=partitions)
        names = {weight.logical.name for weight in partial.weights}
        # Retain all physical witnesses: a joined view can read a non-anchor file.
        return replace(self, weights=partial.weights, _complete=False,
                       views=tuple(v for v in self.views if v.logical.name in names))

    def materialize_many(
        self, requests: tuple[tuple[EncodedWeight, int, tuple[AxisSpan, ...]], ...], *,
        device: torch.device | str, compute_dtype: torch.dtype,
        packed_device: torch.device | str | None = None,
        progress: Callable[[int, int], None] | None = None,
        _request_order: Iterable[tuple[EncodedWeight, int, tuple[AxisSpan, ...]]] | None = None,
        _pending_weight: Callable[[str, Tensor | QuantizedMatrix], None] | None = None,
    ) -> dict[str, Tensor | QuantizedMatrix]:
        # Joined tensors may read more than their anchor shard. Keep the shared
        # transaction, and close every physical source before publishing it.
        for evidence in self.inventory.shard_files:
            self.validate_shard(evidence.relative_path)
        result = CheckpointWeights.materialize_many(
            self, requests, device=device, compute_dtype=compute_dtype, packed_device=packed_device,
            progress=progress, _request_order=_request_order, _pending_weight=_pending_weight)
        for evidence in self.inventory.shard_files:
            self.validate_shard(evidence.relative_path)
        return result


def admit_gguf_checkpoint(
    root: str | Path, config: Any, logical: tuple[LogicalWeight, ...],
    declaration: Callable[..., tuple[GGUFWeightView, ...]],
) -> GGUFCheckpointWeights:
    """Container dispatch independent of architecture names and model execution."""
    from cubic_serve.models.gguf import inspect_gguf
    from cubic_serve.models.parallel_weights import confined_checkpoint_path

    root = Path(root).resolve(strict=True)
    index = root / "gguf.index.json"
    if index.is_file():
        raw = json.loads(index.read_bytes())
        if (not isinstance(raw, dict) or set(raw) != {"version", "bundles"}
                or type(raw["version"]) is not int or raw["version"] != 1
                or not isinstance(raw["bundles"], dict) or not raw["bundles"]
                or any(not isinstance(k, str) or not k or not isinstance(v, str)
                       or not v.endswith(".gguf") for k, v in raw["bundles"].items())):
            raise QuantizationError("GGUF index requires a supported version and explicit bundles")
        bundles = {key: inspect_gguf(confined_checkpoint_path(root, path))
                   for key, path in raw["bundles"].items()}
    else:
        bundles = {"main": inspect_gguf(root)}
    views = declaration(config, logical, bundles=bundles)
    if (type(views) is not tuple or any(not isinstance(v, GGUFWeightView) for v in views)
            or tuple(sorted((v.logical for v in views), key=lambda w: w.name))
            != tuple(sorted(logical, key=lambda w: w.name))):
        raise QuantizationError("GGUF declarations must retain the complete existing logical plan")
    # Register only the format descriptor; admission still proves every physical
    # header/type and never grants a numerical or graph qualification receipt.
    from cubic_serve.quantization.extensions.ggml.extension import EXTENSION
    from cubic_serve.quantization.registry import extensions, register

    if not any(e.format_id == EXTENSION.format_id for e in extensions()):
        register(EXTENSION)
    return GGUFCheckpointWeights.from_views(root, tuple(bundles.values()), views)
