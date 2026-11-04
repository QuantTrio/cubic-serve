"""Byte-exact GGML block slicing, with original scale groups kept intact."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from gguf import GGML_QUANT_SIZES
from torch import Tensor

from cubic_serve.models.quantization import QuantizationError

from .checkpoint import column_offset, kind, scheme

if TYPE_CHECKING:
    from cubic_serve.models.parallel_weights import AxisSpan, SafeTensorReader
    from cubic_serve.models.quantization import QuantizationScheme
    from cubic_serve.models.quantized_checkpoint import (
        CheckpointWeights,
        EncodedWeight,
        LogicalWeight,
        QuantizedMatrix,
    )

ROW_ALIGNMENT = 1
CONCAT_AXIS = 0
SCALAR_PAYLOADS = ()


def payload_headers(weight: LogicalWeight, encoding: QuantizationScheme
                    ) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    n, k = weight.shape
    block, size = GGML_QUANT_SIZES[kind(encoding)]
    return (("weight_packed", "U8", (n, math.ceil((k + column_offset(encoding)) / block) * size)),)


def slice_index(shape: tuple[int, ...], partitions: tuple[int, ...], encoding: QuantizationScheme,
                key: str, axis: int, start: int, end: int,
                ) -> tuple[tuple[slice, ...], tuple[int, ...]]:
    if key != "weight_packed" or axis not in (0, 1):
        raise QuantizationError("unknown GGML payload or slice axis")
    if axis == 0:
        return (slice(start, end), slice(None)), ()
    block, size = GGML_QUANT_SIZES[kind(encoding)]
    if start % block or end % block:
        raise QuantizationError("GGML TP ownership must preserve original scale blocks")
    return (slice(None), slice(start // block * size, end // block * size)), partitions


def stacked_payload_name(source: str, key: str) -> str:
    if key != "weight_packed":
        raise QuantizationError("unknown stacked GGML payload")
    return source + ".weight_packed"


def _slice(
    read: Callable[[tuple[slice, ...]], Tensor], shape: tuple[int, ...],
    encoding: QuantizationScheme,
    axis: int, start: int, end: int, partitions: tuple[int, ...],
) -> QuantizedMatrix:
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix

    block, size = GGML_QUANT_SIZES[kind(encoding)]
    offset = column_offset(encoding)
    if axis == 0:
        index = (slice(start, end), slice(None))
        logical_shape = (end - start, shape[1])
        parts, position = [], 0
        for width in partitions:
            overlap = min(end, position + width) - max(start, position)
            if overlap > 0:
                parts.append(overlap)
            position += width
        partitions = tuple(parts)
    else:
        first, last = (offset + start) // block, math.ceil((offset + end) / block)
        index = (slice(None), slice(first * size, last * size))
        logical_shape = (shape[0], end - start)
        offset = (offset + start) % block
    return QuantizedMatrix({"weight_packed": read(index).contiguous()},
                           scheme(int(kind(encoding)), offset), logical_shape, partitions)


def slice_matrix(matrix: QuantizedMatrix, axis: int, start: int, end: int) -> QuantizedMatrix:
    return _slice(lambda index: matrix.payload["weight_packed"][index], matrix.shape,
                  matrix.encoding, axis, start, end, matrix.partitions)


def materialize(
    checkpoint: CheckpointWeights, weight: EncodedWeight, axis: int, spans: tuple[AxisSpan, ...],
    device: torch.device | str, readers: Mapping[str, SafeTensorReader] | None,
) -> QuantizedMatrix:
    from cubic_serve.models.checkpoint import _file_stamp
    from cubic_serve.models.parallel_weights import confined_checkpoint_path
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix

    if len(weight.payload) != 1 or weight.stacked_source is not None:
        raise QuantizationError("GGML loading requires admitted original matrix views")
    if weight.encoding is None:
        raise QuantizationError("GGML materialization requires an admitted block encoding")
    _, name, shard = weight.payload[0]
    path = confined_checkpoint_path(checkpoint.inventory.model_root, shard)
    before = _file_stamp(path)
    if readers is None:
        checkpoint.validate_shard(shard)
    context = checkpoint._open_reader(path) if readers is None else nullcontext(readers[shard])
    with context as reader:
        source = reader.get_slice(name)
        pieces = [_slice(lambda index: source[index], weight.logical.shape, weight.encoding,
                         axis, span.start, span.stop, weight.logical.output_partitions)
                  for span in spans]
        packed = torch.cat([p.payload["weight_packed"] for p in pieces], dim=axis) if len(
            pieces) > 1 else pieces[0].payload["weight_packed"]
        shape = list(weight.logical.shape)
        shape[axis] = sum(span.size for span in spans)
        parts = (tuple(n for p in pieces for n in p.partitions)
                 if axis == 0 else pieces[0].partitions)
        result = QuantizedMatrix({"weight_packed": packed.to(device)}, pieces[0].encoding,
                                  (shape[0], shape[1]), parts)
    if _file_stamp(path) != before:
        raise QuantizationError("GGUF source changed during original-block materialization")
    return result
