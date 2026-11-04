"""INT's offset-code transport and scale/zero arithmetic for shared row kernels."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators

import torch

from cubic_serve.ops.cuda.compiled_artifact import bind_tensor
from cubic_serve.quantization.common.small_rows import SmallRowsWeight

from .checkpoint import fields


def supported(matrix: QuantizedMatrix) -> bool:
    packing, _, _ = fields(matrix.encoding)
    return (matrix.encoding.group_out == 1 and matrix.encoding.activation == "identity"
            and 2 <= matrix.encoding.bits <= 8
            and (packing == "dense" or (packing == "legacy" and 32 % matrix.encoding.bits == 0)))


def probe(matrix: QuantizedMatrix, n: int, device: torch.device) -> QuantizedMatrix:
    """Private exact-zero affine fixture, including the original zero-point axis."""
    import math

    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix

    enc, (_, symmetry, _) = matrix.encoding, fields(matrix.encoding)
    k = matrix.shape[1]
    gi = min(k, enc.group_in)
    pattern = []
    for word in range(enc.bits):
        value = sum(1 << bit for bit in range(32)
                    if (word * 32 + bit) % enc.bits == enc.bits - 1)
        pattern.append(value if value < 2**31 else value - 2**32)
    template = torch.tensor(pattern, device=device, dtype=torch.int32)
    payload = {
        "weight_packed": template.repeat(n, k // 32),
        "weight_scale": torch.full((n, k // gi), 1 / 512, device=device,
                                   dtype=matrix.payload["weight_scale"].dtype),
        "weight_shape": torch.tensor((n, k), device=device, dtype=torch.int64),
    }
    if symmetry == "asym":
        count = math.ceil(n * enc.bits / 32)
        words = template.repeat(math.ceil(count / enc.bits))[:count]
        payload["weight_zero_point"] = words[:, None].expand(-1, k // gi).contiguous()
    return QuantizedMatrix(payload, enc, (n, k), (n,))


def prepare(
    operators: PackedLinearOperators, matrix: QuantizedMatrix, device: torch.device,
) -> SmallRowsWeight:
    enc, payload = matrix.encoding, matrix.payload
    packing, symmetry, _ = fields(enc)
    n, k = matrix.shape
    key = ("int.small_rows.v1", device, enc, matrix.shape,
           tuple((name, bind_tensor(t)) for name, t in sorted(payload.items())))
    if key in operators.prepared_weights:
        codes = operators.prepared_weights[(*key, "codes")][1]
        metadata = operators.prepared_weights[key][1]
    else:
        raw = payload["weight_packed"].to(device)
        if packing == "dense" or (packing == "legacy" and 32 % enc.bits == 0):
            codes = raw.contiguous().view(torch.uint8)
            if codes.data_ptr() % 16:
                codes = codes.clone()
        else:
            raise ValueError("INT small-row transport requires dense original words")
        group = min(k, enc.group_in)
        scales = payload["weight_scale"].to(device)
        if scales.shape != (n, k // group):
            raise ValueError("INT scales differ from original group geometry")
        arrays = [scales.contiguous().view(torch.uint8).reshape(-1)]
        zero_offset = arrays[0].numel()
        if symmetry == "asym":
            # Pad only byte alignment between planes; packed zero-point codes and
            # their N-axis ownership remain in the original serialization.
            arrays.append(torch.zeros((-zero_offset) % 4, device=device, dtype=torch.uint8))
            zero_offset += arrays[-1].numel()
            arrays.append(payload["weight_zero_point"].to(device).contiguous().view(torch.uint8).reshape(-1))
        metadata = torch.cat(arrays)
        operators.prepared_weights[key] = matrix, metadata
        operators.prepared_weights[(*key, "codes")] = matrix, codes
    group = min(k, enc.group_in)
    scales = payload["weight_scale"]
    zero_offset = scales.numel() * scales.element_size()
    if symmetry == "asym":
        zero_offset += (-zero_offset) % 4
    kind = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[scales.dtype]
    result = SmallRowsWeight(codes, metadata, (n, k), enc.bits, group,
        Path(__file__).with_suffix(".cuh").read_bytes(),
        (("SCALE_KIND", str(kind)), ("ASYMMETRIC", str(int(symmetry == "asym"))),
         ("ZERO_OFFSET", str(zero_offset)), ("LEGACY", str(int(packing == "legacy"))),
         ("FORMAT_RAW_BYTES", str(scales.element_size()))))
    return result


def prepare_mma(
    operators: PackedLinearOperators, matrix: QuantizedMatrix, device: torch.device,
) -> SmallRowsWeight | None:
    """Use the same original-code and scale/zero contract in shared MMA staging."""
    return prepare(operators, matrix, device) if supported(matrix) else None
