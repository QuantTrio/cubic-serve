"""Independent eager decoding and activation arithmetic for packed checkpoints.

These functions define reference math. They do not call an optimized quantized
operator, import vLLM, or claim CUDA Graph/performance qualification.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from cubic_serve.models.quantization import (
    ActivationFormat,
    QuantizationError,
    QuantizationScheme,
)
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.quantization.common.reference import (
    ActivationQuantization as ActivationQuantization,
)
from cubic_serve.quantization.common.reference import _positive_finite as _positive_finite
from cubic_serve.quantization.common.reference import _tensor as _tensor
from cubic_serve.quantization.extensions.awq.reference import decode_awq as decode_awq
from cubic_serve.quantization.extensions.cubic.reference import cubic_metadata as cubic_metadata
from cubic_serve.quantization.extensions.cubic.reference import decode_cubic as decode_cubic
from cubic_serve.quantization.extensions.cubic.reference import unpack_cubic as unpack_cubic
from cubic_serve.quantization.extensions.fp8.reference import decode_fp8 as decode_fp8
from cubic_serve.quantization.extensions.nvfp4.reference import decode_nvfp4 as decode_nvfp4
from cubic_serve.quantization.extensions.nvfp4.reference import unpack_fp4 as unpack_fp4
from cubic_serve.quantization.registry import get_extension


def decode_weight(
    payload: Mapping[str, Tensor],
    encoding: QuantizationScheme,
    shape: tuple[int, int],
    *,
    partitions: tuple[int, ...] | None = None,
) -> Tensor:
    if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise QuantizationError("quantized logical weight must be a positive [N,K] matrix")
    devices = {value.device for value in payload.values()}
    if len(devices) != 1:
        raise QuantizationError("quantized weight payload must share one device")
    return (
        get_extension(encoding.format)
        .module("reference")
        .decode(payload, encoding, shape, partitions)
    )


def quantize_activation(
    inputs: Tensor, encoding: QuantizationScheme, *, global_inverse: Tensor | None = None
) -> ActivationQuantization:
    values = inputs.float()
    if not bool(torch.isfinite(values).all()):
        raise QuantizationError("activation quantization requires finite operands")
    if encoding.activation is ActivationFormat.IDENTITY:
        return ActivationQuantization(inputs, values.new_ones(()), values)
    return (
        get_extension(encoding.format)
        .module("reference")
        .quantize(inputs, encoding, global_inverse)
    )


def quantized_linear(
    inputs: Tensor,
    payload: Mapping[str, Tensor],
    encoding: QuantizationScheme,
    shape: tuple[int, int],
    *,
    partitions: tuple[int, ...] | None = None,
    output_dtype: torch.dtype | None = None,
) -> Tensor:
    activation = activation_from_payload(inputs, payload, encoding)
    weight = decode_weight(payload, encoding, shape, partitions=partitions)
    if encoding.activation is ActivationFormat.IDENTITY:
        weight = weight.to(inputs.dtype).float()
    if inputs.shape[-1] != shape[1]:
        raise QuantizationError("quantized linear input differs from logical K")
    from cubic_serve.ops.eager.linear import linear

    return linear(activation.dequantized, weight, output_dtype=output_dtype or inputs.dtype)


def decode_matrix(matrix: QuantizedMatrix) -> Tensor:
    return decode_weight(
        matrix.payload, matrix.encoding, matrix.shape, partitions=matrix.partitions or None
    )


def activation_from_payload(inputs, payload, encoding):
    reference = get_extension(encoding.format).module("reference")
    resolver = getattr(reference, "activation_from_payload", None)
    if resolver is not None:
        return resolver(inputs, payload, encoding)
    return quantize_activation(inputs, encoding, global_inverse=payload.get("input_global_scale"))


def linear_matrix(inputs: Tensor, matrix: QuantizedMatrix, *, output_rows: int = 256,
                  compute_mode: str = "wna16", decode_rows: int | None = None) -> Tensor:
    from cubic_serve.ops.eager.linear import linear

    if compute_mode == "wna8":
        from cubic_serve.quantization.registry import select_compute_mode

        compute_mode = select_compute_mode(
            matrix.encoding, "linear", compute_mode,
            m=inputs.reshape(-1, inputs.shape[-1]).shape[0],
            n=matrix.shape[0], k=matrix.shape[1],
            dtype=inputs.dtype, device=inputs.device,
        )

    if compute_mode != "wna8":
        # Only tiny format-owned activation metadata moves to the operand
        # device; the packed weight remains streamed by output rows below.
        activation_payload = {
            name: value.to(inputs.device) if value.numel() == 1 else value
            for name, value in matrix.payload.items()
        }
        activation = activation_from_payload(
            inputs, activation_payload, matrix.encoding
        ).dequantized
    alignment = math.lcm(8, matrix.encoding.group_out)
    tile = max(alignment, output_rows // alignment * alignment)
    # Weight transport/decoding is independent of the canonical dot geometry.
    # Decode a bounded group of output tiles together, then retain the original
    # per-tile GEMM shape and row order exactly. WNA8 keeps its format-owned
    # integer reduction unit unchanged.
    decode_tile = tile
    row_batch = None
    prepared_linear = None
    if compute_mode == "wna8":
        reference = get_extension(matrix.encoding.format).module("reference")
        prepare = getattr(reference, "prepare_wna8_linear", None)
        if prepare is not None:
            prepared = prepare(inputs, matrix, output_rows=tile, byte_budget=256 << 20)
            if prepared is not None:
                row_batch, prepared_linear = prepared
                if not callable(prepared_linear):
                    raise ValueError("prepared reference linear must be callable")
        row_batch_size = getattr(reference, "wna8_output_row_batch_size", None)
        if row_batch is None and row_batch_size is not None:
            row_batch = row_batch_size(inputs, matrix, byte_budget=256 << 20)
        if row_batch is not None and (type(row_batch) is not int or row_batch <= 0):
            raise ValueError("reference row batch size must be a positive integer")
    if row_batch is not None:
        if decode_rows is not None and (type(decode_rows) is not int or decode_rows <= 0):
            raise ValueError("reference decode rows must be a positive integer")
        requested = row_batch if decode_rows is None else min(decode_rows, row_batch)
        batch_alignment = tile if prepared_linear is not None else alignment
        decode_tile = max(batch_alignment, requested // batch_alignment * batch_alignment)
    elif compute_mode != "wna8":
        if decode_rows is None:
            decode_rows = max(tile, (256 << 20) // (64 * matrix.shape[1]))
        elif type(decode_rows) is not int or decode_rows <= 0:
            raise ValueError("reference decode rows must be a positive integer")
        decode_tile = max(tile, decode_rows // tile * tile)
    outputs = []
    for start in range(0, matrix.shape[0], decode_tile):
        part = matrix.rows(start, min(start + decode_tile, matrix.shape[0])).to(inputs.device)
        if prepared_linear is not None:
            outputs.append(prepared_linear(part))
            continue
        if compute_mode == "wna8":
            reference = get_extension(matrix.encoding.format).module("reference")
            integer_linear = getattr(reference, "wna8_linear", None)
            if integer_linear is not None:
                outputs.append(integer_linear(inputs, part))
                continue
            # Activation grouping, scale storage and rounding belong to the
            # format, just as WNA16's quantize/decode boundaries do.
            activation = reference.wna8_operand(inputs)
            weight = reference.wna8_weight(part)
        else:
            weight = decode_matrix(part)
        if compute_mode == "wna16" and matrix.encoding.activation is ActivationFormat.IDENTITY:
            weight = weight.to(inputs.dtype).float()
        for offset in range(0, weight.shape[0], tile):
            outputs.append(linear(
                activation, weight[offset:offset + tile], output_dtype=inputs.dtype))
    return torch.cat(outputs, dim=-1)


def embedding_matrix(input_ids: Tensor, matrix: QuantizedMatrix, *, dtype: torch.dtype) -> Tensor:
    values = []
    group = math.lcm(
        matrix.encoding.group_out,
        get_extension(matrix.encoding.format).module("layout").ROW_ALIGNMENT,
    )
    for index in input_ids.reshape(-1).tolist():
        start = index // group * group
        values.append(
            decode_matrix(matrix.rows(start, start + group).to(input_ids.device))[index - start].to(
                dtype
            )
        )
    return torch.stack(values).reshape(*input_ids.shape, matrix.shape[1])


class QuantizedReferenceOperators:
    """Bounded reference workspace applies equally to packed and plain linear."""

    def __init__(self, output_rows: int = 256, *, compute_mode: str | None = None):
        if type(output_rows) is not int or output_rows <= 0:
            raise ValueError("reference output tile rows must be positive")
        self.output_rows = output_rows
        if compute_mode is None:
            from cubic_serve.ops.strategies import compute_mode as scoped_mode
            compute_mode = scoped_mode()
        if compute_mode not in ("wna16", "wna8"):
            raise ValueError("compute_mode must be wna16 or wna8")
        self.compute_mode = compute_mode

    def linear(self, inputs: Tensor, weight: Tensor | QuantizedMatrix) -> Tensor:
        from cubic_serve.ops.eager.linear import linear

        if isinstance(weight, QuantizedMatrix):
            return linear_matrix(inputs, weight, output_rows=self.output_rows,
                                 compute_mode=self.compute_mode)
        return torch.cat(
            [
                linear(inputs, weight[start : start + self.output_rows])
                for start in range(0, weight.shape[0], self.output_rows)
            ],
            dim=-1,
        )

    def embedding(
        self, input_ids: Tensor, weight: Tensor | QuantizedMatrix, *, dtype: torch.dtype
    ) -> Tensor:
        if isinstance(weight, QuantizedMatrix):
            return embedding_matrix(input_ids, weight, dtype=dtype)
        if weight.device.type == "cpu" and input_ids.device.type == "cuda":
            # Independent lookup copies only selected original rows, never the table.
            selected = weight.index_select(0, input_ids.flatten().cpu())
            return selected.reshape(*input_ids.shape, weight.shape[1]).to(input_ids.device)
        return torch.nn.functional.embedding(input_ids, weight)

    def full_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        query_ordinals: Tensor,
        key_ordinals: Tensor,
    ) -> Tensor:
        from cubic_serve.ops.eager.attention import full_attention

        return full_attention(
            query,
            key,
            value,
            query_ordinals=query_ordinals,
            key_ordinals=key_ordinals,
            output_dtype=query.dtype,
        )
