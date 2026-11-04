"""Cubic's packed transport and polynomial metadata for the shared row kernel."""
from __future__ import annotations

import struct
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators

import torch

from cubic_serve.ops.cuda.compiled_artifact import bind_tensor
from cubic_serve.quantization.common.small_rows import SmallRowsWeight


def supported(matrix: QuantizedMatrix) -> bool:
    enc = matrix.encoding
    return (enc.group_out == 1 and 2 <= enc.bits <= 8 and enc.activation == "identity"
            and enc.metadata in ("float32-scale-float16-ab", "int8-scale-int4-ab-global-fp32",
                                 "e5m9-scale-curve2-tensor-fp16"))


def probe(matrix: QuantizedMatrix, n: int, device: torch.device) -> QuantizedMatrix:
    """Private zero-weight finite fixture; never copies checkpoint values."""
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix

    original_n, k = matrix.shape
    enc = matrix.encoding
    original_parts = matrix.partitions or (original_n,)
    boundaries = [0]
    for i in range(1, len(original_parts)):
        boundaries.append(max(boundaries[-1] + 16,
            min(n - 16 * (len(original_parts) - i),
                int(sum(original_parts[:i]) * n / original_n) // 16 * 16)))
    boundaries.append(n)
    parts = tuple(b - a for a, b in pairwise(boundaries))
    p = {}
    for name, original in matrix.payload.items():
        shape = list(original.shape)
        if name == "weight_packed":
            shape = [n, k * enc.bits // 8]
        elif original.ndim == 2 and name not in ("weight_curve_a", "weight_curve_b"):
            shape = [n, k // enc.group_in]
        p[name] = torch.zeros(shape, device=device, dtype=original.dtype)
        if name == "weight_scale":
            p[name].fill_(1 if original.dtype == torch.int8 else 1 / 512)
        elif name == "weight_scale_global":
            p[name].fill_(1 / 512)
        elif name in ("weight_a", "weight_curve_a"):
            p[name].fill_(1)
    return QuantizedMatrix(p, enc, (n, k), parts)


def prepare(
    operators: PackedLinearOperators, matrix: QuantizedMatrix, device: torch.device,
) -> SmallRowsWeight:
    n, k = matrix.shape
    enc, payload = matrix.encoding, matrix.payload
    key = ("cubic.small_rows.v1", device, enc, matrix.shape, matrix.partitions,
           tuple((name, bind_tensor(t)) for name, t in sorted(payload.items())))
    if key in operators.prepared_weights:
        codes = operators.prepared_weights[(*key, "codes")][1]
        metadata = operators.prepared_weights[key][1]
    else:
        count = k // enc.group_in
        sizes = matrix.partitions or (n,)
        if sum(sizes) != n or any(size <= 0 for size in sizes):
            raise ValueError("Cubic output partitions must cover the original rows")
        variant = {"float32-scale-float16-ab": 0, "int8-scale-int4-ab-global-fp32": 1,
                   "e5m9-scale-curve2-tensor-fp16": 2}[enc.metadata]
        def plane(name: str, width: int) -> torch.Tensor:
            value = payload[name]
            if value.shape != (n, count):
                raise ValueError("Cubic group metadata differs from the logical geometry")
            return value.to(device).contiguous().view(torch.uint8).reshape(n, count, width)
        if variant == 1:
            raw = torch.cat((plane("weight_scale", 1), plane("weight_ab", 1)), -1)
        elif variant == 2:
            raw = plane("weight_metadata", 2)
        else:
            raw = torch.cat(
                (plane("weight_scale", 4), plane("weight_a", 2), plane("weight_b", 2)), -1,
            )
        records, end = [], 0
        for part, size in enumerate(sizes):
            end += size
            record = struct.pack("<I", end)
            if variant == 1:
                record += struct.pack("<3f", *(float(payload[name].reshape(-1)[part]) for name in
                    ("weight_a_global", "weight_b_global", "weight_scale_global"))) + bytes(16)
            elif variant == 2:
                record += bytes(12)
                for name in ("weight_curve_a", "weight_curve_b"):
                    value = payload[name].reshape(-1, 4)[part].contiguous()
                    record += bytes(value.view(torch.uint8).cpu().tolist())
            else:
                record += bytes(28)
            records.append(record)
        descriptor = torch.tensor(list(b"".join(records)), dtype=torch.uint8, device=device)
        metadata = torch.cat((descriptor, raw.reshape(-1)))

        codes = payload["weight_packed"].to(device).contiguous()
        if codes.data_ptr() % 16:
            codes = codes.clone()
        operators.prepared_weights[key] = matrix, metadata
        operators.prepared_weights[(*key, "codes")] = matrix, codes
    sizes = matrix.partitions or (n,)
    variant = {"float32-scale-float16-ab": 0, "int8-scale-int4-ab-global-fp32": 1,
               "e5m9-scale-curve2-tensor-fp16": 2}[enc.metadata]
    reciprocal = 1.0 / ((1 << (enc.bits - 1)) - 1)
    high = struct.unpack("f", struct.pack("f", reciprocal))[0]
    low = struct.unpack("f", struct.pack("f", reciprocal - high))[0]
    elements = 8 if enc.bits <= 4 else 4 if enc.group_in >= 128 else 2
    result = SmallRowsWeight(
        codes, metadata, (n, k), enc.bits,
        enc.group_in, Path(__file__).with_suffix(".cuh").read_bytes(),
        (("RECIP_HI", high.hex() + "f"), ("RECIP_LO", low.hex() + "f"),
         ("METADATA_VARIANT", str(variant)), ("DESCRIPTOR_COUNT", str(len(sizes))),
         ("RAW_OFFSET", str(32 * len(sizes))), ("FORMAT_RAW_BYTES", str(2 if variant else 8))),
        elements,
    )
    return result


def prepare_mma(
    operators: PackedLinearOperators, matrix: QuantizedMatrix, device: torch.device,
) -> SmallRowsWeight | None:
    """Attach finite FP32 normalized curves; retain original per-group metadata.

    Scale multiplication and activation-storage rounding still occur inside the
    CTA. The table size depends on finite curve/code domains and original output
    partitions, never on N*K or the number of checkpoint groups.
    """
    enc = matrix.encoding
    if not supported(matrix) or enc.metadata == "float32-scale-float16-ab":
        return None
    base = prepare(operators, matrix, device)
    key = ("cubic.small_mma.curves.v1", bind_tensor(base.metadata), enc.bits)
    cached = operators.prepared_weights.get(key)
    if cached is None:
        payload = matrix.payload
        parts = matrix.partitions or (matrix.shape[0],)
        codes = 1 << enc.bits
        half = codes // 2
        q = torch.arange(-half, half, dtype=torch.int32)
        q[0] = 0  # Cubic's most-negative packed code is its zero sentinel.
        t = (q.abs().double() / (half - 1)).float()
        tables = []
        for part in range(len(parts)):
            if enc.metadata == "int8-scale-int4-ab-global-fp32":
                curve = torch.arange(256, dtype=torch.int32)
                ac, bc = curve & 15, curve >> 4
                ac, bc = torch.where(ac >= 8, ac - 16, ac), torch.where(bc >= 8, bc - 16, bc)
                ag = payload["weight_a_global"].reshape(-1)[part].cpu().float()
                bg = payload["weight_b_global"].reshape(-1)[part].cpu().float()
                a = (1 + ac.float() * ag).half().float()
                b = (bc.float() * bg).half().float()
            else:
                a = payload["weight_curve_a"].reshape(-1, 4)[part].cpu().float()
                b = payload["weight_curve_b"].reshape(-1, 4)[part].cpu().float()
            a, b = a[:, None], b[:, None]
            value = t * (a + t * (b + t * ((1 - a) - b)))
            tables.append(torch.where(q < 0, -value, value).reshape(-1))
        suffix = torch.cat(tables).contiguous().view(torch.uint8).to(device)
        metadata = torch.cat((base.metadata, suffix))
        operators.prepared_weights[key] = matrix, metadata
    else:
        metadata = cached[1]
    return replace(base, metadata=metadata,
                   decoder=Path(__file__).with_name("small_mma.cuh").read_bytes())
