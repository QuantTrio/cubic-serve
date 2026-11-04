"""Lossless Q8_0 transport into the existing affine integer compute contract.

Original signed codes and FP16 scale values are preserved. Metadata is widened
exactly to FP32 so the shared decoder evaluates scale * code in FP32 before the
original A16 operand rounding; it must not introduce an FP16 product boundary.
This prepares packed bytes, never a decoded floating-point weight matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cubic_serve.models.quantization import QuantizationScheme
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.cuda.compiled_artifact import bind_tensor

from .checkpoint import column_offset, kind

if TYPE_CHECKING:
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators


def compatible(matrix: QuantizedMatrix) -> bool:
    n, k = matrix.shape
    return int(kind(matrix.encoding)) == 8 and column_offset(matrix.encoding) == 0 and (
        n % 64 == 0 and k % 32 == 0)


def prepare(operators: PackedLinearOperators, matrix: QuantizedMatrix) -> QuantizedMatrix | None:
    if not compatible(matrix):
        raise ValueError("Q8 affine transport requires complete original scale groups")
    original = matrix.payload["weight_packed"]
    key = ("ggml.q8.original-affine.v1", id(matrix), bind_tensor(original))
    cached = operators.materialized_weights.get(key)
    if cached is not None and cached[0] is matrix:
        return cached[1]
    n, k = matrix.shape
    blocks = original.reshape(n, k // 32, 34)
    scale = blocks[:, :, :2].contiguous().view(torch.float16).reshape(n, k // 32).float()
    # INT owns a positive-scale invariant. Valid GGML zero/signed scales keep
    # their original native decoder instead of changing that shared contract.
    if not bool((torch.isfinite(scale) & (scale > 0)).all()):
        return None
    # INT's serialized words are offset binary. XOR changes the byte layout,
    # not the signed code, and preserves all 256 original code values.
    packed = blocks[:, :, 2:].bitwise_xor(128).reshape(n, k).contiguous().view(torch.int32)
    result = QuantizedMatrix({
        "weight_packed": packed,
        "weight_scale": scale,
        "weight_shape": torch.tensor((n, k), dtype=torch.int64, device=original.device),
    }, QuantizationScheme("int-pack-v1", 8, 32, metadata="dense:sym:F32"),
        matrix.shape, matrix.partitions)
    operators.materialized_weights[key] = (matrix, result)
    return result
