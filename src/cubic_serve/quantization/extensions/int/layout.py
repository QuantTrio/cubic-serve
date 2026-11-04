"""Original word layouts and bounded owner slicing; no floating weight materialization."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

from torch import Tensor

if TYPE_CHECKING:
    from cubic_serve.models.checkpoint import CheckpointTensor
    from cubic_serve.models.parallel_weights import AxisSpan, SafeTensorReader
    from cubic_serve.models.quantized_checkpoint import (
        EncodedWeight,
        LogicalWeight,
        QuantizedCheckpoint,
        QuantizedMatrix,
    )
from dataclasses import replace
from functools import lru_cache

import torch
from safetensors import safe_open

from cubic_serve.models.quantization import QuantizationError, QuantizationScheme

from .checkpoint import fields
from .reference import unpack

ROW_ALIGNMENT = 1
CONCAT_AXIS = 0
SCALAR_PAYLOADS = ("weight_shape",)


def words(count: int, bits: int, packing: str) -> int:
    return math.ceil(count * bits / 32) if packing == "dense" else math.ceil(count / (32 // bits))


def payload_headers(
    weight: LogicalWeight, encoding: QuantizationScheme
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    packing, symmetry, dtype = fields(encoding)
    if packing == "auto" or dtype == "auto":
        raise QuantizationError("INT physical encoding requires header admission")
    n, k = weight.shape
    groups = math.ceil(k / min(k, encoding.group_in))
    result = [
        ("weight_packed", "I32", (n, words(k, encoding.bits, packing))),
        ("weight_scale", dtype, (n, groups)),
        ("weight_shape", "I64", (2,)),
    ]
    if symmetry == "asym":
        result.append(("weight_zero_point", "I32", (words(n, encoding.bits, packing), groups)))
    return tuple(result)


@lru_cache(maxsize=256)
def _resolved_encoding(encoding: QuantizationScheme, metadata: str) -> QuantizationScheme:
    """Reuse an immutable declaration; every tensor still proves its headers."""
    return encoding if encoding.metadata == metadata else replace(encoding, metadata=metadata)


def admit_encoding(
    weight: LogicalWeight, encoding: QuantizationScheme, records: Mapping[str, CheckpointTensor]
) -> QuantizationScheme:
    prefix = weight.name.removesuffix(".weight") + "."
    packing, symmetry, dtype = fields(encoding)
    if dtype == "auto":
        scale = records.get(prefix + "weight_scale")
        if scale is None:
            raise QuantizationError(f"missing INT scale: {prefix}")
        dtype = scale.dtype
    candidates = []
    for mode in ("dense", "legacy") if packing == "auto" else (packing,):
        candidate = _resolved_encoding(encoding, f"{mode}:{symmetry}:{dtype}")
        if all(
            prefix + s in records
            and records[prefix + s].dtype == d
            and records[prefix + s].shape == sh
            for s, d, sh in payload_headers(weight, candidate)
        ):
            candidates.append(candidate)
    if not candidates:
        raise QuantizationError(f"INT physical headers disagree with declaration: {prefix}")
    if len(candidates) == 2 and encoding.bits not in (2, 4, 8):
        raise QuantizationError(
            f"ambiguous INT packing; declare weights.packing dense or legacy: {prefix}"
        )
    return candidates[0]


def pack(codes: Tensor, bits: int, packing: str = "dense", axis: int = 1) -> Tensor:
    source = codes.transpose(0, 1) if axis == 0 else codes
    rows, count = source.shape
    unsigned = source.to(torch.int64) + (1 << (bits - 1))
    if bool(((unsigned < 0) | (unsigned >= 1 << bits)).any()):
        raise QuantizationError("INT code outside declared signed range")
    out = torch.zeros(rows, words(count, bits, packing), dtype=torch.int64, device=source.device)
    pos = torch.arange(count, device=source.device)
    bit = pos * bits if packing == "dense" else pos // (32 // bits) * 32 + pos % (32 // bits) * bits
    idx, off = bit // 32, bit % 32
    out.scatter_add_(1, idx.expand(rows, -1), (unsigned << off) & 0xFFFFFFFF)
    spill = off + bits > 32
    if bool(spill.any()):
        out.scatter_add_(
            1, (idx[spill] + 1).expand(rows, -1), unsigned[:, spill] >> (32 - off[spill])
        )
    out = out.to(torch.int32)
    return out.transpose(0, 1).contiguous() if axis == 0 else out



def _rebase_legacy_words(raw: Tensor, bits: int, start: int, count: int, axis: int) -> Tensor:
    """Rebase complete code slots without expanding them or changing source words."""
    source = raw.T if axis == 0 else raw
    codes_per_word = 32 // bits
    word_count = words(count, bits, "legacy")
    used_bits = codes_per_word * bits
    mask = (1 << used_bits) - 1
    shift = start % codes_per_word * bits
    rebased = (source[:, :word_count].to(torch.int64) & mask) >> shift
    if shift:
        high = source[:, 1:word_count + 1].to(torch.int64) & mask
        rebased[:, :high.shape[1]] |= high << (used_bits - shift)
    rebased &= mask
    tail = count % codes_per_word
    if tail:
        rebased[:, -1] &= (1 << (tail * bits)) - 1
    result = rebased.to(torch.int32)
    return result.T.contiguous() if axis == 0 else result


def _slice(
    read: Callable[[str, tuple[slice, ...]], Tensor],
    shape: tuple[int, int],
    encoding: QuantizationScheme,
    axis: int,
    start: int,
    end: int,
    partitions: tuple[int, ...] = (),
) -> QuantizedMatrix:
    """Read only word covers and scale groups; normalize partial groups by repetition."""
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix

    packing, symmetry, dtype = fields(encoding)
    n, k = shape
    r0, r1, c0, c1 = (start, end, 0, k) if axis == 0 else (0, n, start, end)
    group = min(k, encoding.group_in)
    whole_k = c0 == 0 and c1 == k
    local_group = group if whole_k else math.gcd(group, math.gcd(c0, c1 - c0))
    positions = torch.arange(c0, c1, local_group) // group
    g0, g1 = c0 // group, math.ceil(c1 / group)

    def codes(key: str, dim: int, lo: int, hi: int, other: slice) -> Tensor:
        first = lo * encoding.bits // 32 if packing == "dense" else lo // (32 // encoding.bits)
        last = words(hi, encoding.bits, packing)
        index = (slice(first, last), other) if dim == 0 else (other, slice(first, last))
        raw = read(key, index)
        origin = first * (32 // encoding.bits) if packing == "legacy" else 0
        if packing == "legacy":
            return unpack(raw, encoding.bits, hi - lo, dim, packing, lo - origin)
        # Rebase dense bit offsets without assuming a word starts at a code boundary.
        source = raw.T if dim == 0 else raw
        bit = torch.arange(lo, hi, device=source.device) * encoding.bits - first * 32
        word, shift = bit // 32, bit % 32
        low = source[:, word].long() & 0xFFFFFFFF
        high = source[:, (word + 1).clamp(max=source.shape[1] - 1)].long() & 0xFFFFFFFF
        q = (
            ((low >> shift) | torch.where(shift + encoding.bits > 32, high << (32 - shift), 0))
            & ((1 << encoding.bits) - 1)
        ) - (1 << (encoding.bits - 1))
        return q.T.to(torch.int8).contiguous() if dim == 0 else q.to(torch.int8)

    # Whole K rows preserve both serialized layouts verbatim. In particular,
    # vocabulary ownership must not unpack a multi-gigabyte PLE table just to
    # select its rows. Only packed zero points need a small boundary repack.
    codes_per_word = 32 // encoding.bits
    aligned_legacy = (
        packing == "legacy" and c0 % codes_per_word == c1 % codes_per_word == 0
    )
    mode = packing if whole_k or packing == "legacy" else "dense"
    if whole_k:
        packed = read("weight_packed", (slice(r0, r1), slice(None))).contiguous()
    elif packing == "dense" and c0 * encoding.bits % 32 == c1 * encoding.bits % 32 == 0:
        # Whole-word TP spans already have the exact local dense packing.
        # Keep the scale-group normalization below, but avoid unpack/repack.
        packed = read(
            "weight_packed",
            (slice(r0, r1), slice(c0 * encoding.bits // 32, c1 * encoding.bits // 32)),
        ).contiguous()
    elif aligned_legacy:
        # Legacy words contain floor(32 / bits) complete codes and unused high
        # bits. A word-aligned owner span keeps this ABI; converting it to dense
        # would needlessly expand every code and repack every expert on the CPU.
        packed = read(
            "weight_packed",
            (slice(r0, r1), slice(c0 // codes_per_word, c1 // codes_per_word)),
        ).contiguous()
    elif packing == "legacy":
        # Rebase whole legacy words, not individual codes. Skip the unused
        # high bits between words before joining the next word's low codes.
        raw = read(
            "weight_packed",
            (slice(r0, r1), slice(c0 // codes_per_word, words(c1, encoding.bits, packing))),
        )
        packed = _rebase_legacy_words(raw, encoding.bits, c0, c1 - c0, axis=1)
    else:
        packed = pack(codes("weight_packed", 1, c0, c1, slice(r0, r1)), encoding.bits)
    scale = read("weight_scale", (slice(r0, r1), slice(g0, g1)))
    scale = (scale if whole_k else scale[:, (positions - g0).to(scale.device)]).contiguous()
    newshape = (r1 - r0, c1 - c0)
    payload = {
        "weight_packed": packed,
        "weight_scale": scale,
        "weight_shape": torch.tensor(newshape, dtype=torch.int64, device=packed.device),
    }
    if symmetry == "asym":
        zero_word_span = (
            (packing == "dense" and r0 * encoding.bits % 32 == r1 * encoding.bits % 32 == 0)
            or (packing == "legacy" and r0 % codes_per_word == r1 % codes_per_word == 0)
        )
        if zero_word_span:
            first = r0 * encoding.bits // 32 if packing == "dense" else r0 // codes_per_word
            last = r1 * encoding.bits // 32 if packing == "dense" else r1 // codes_per_word
            z = read("weight_zero_point", (slice(first, last), slice(g0, g1)))
            z = (z if whole_k else z[:, (positions - g0).to(z.device)]).contiguous()
            if packing == "legacy" and codes_per_word * encoding.bits < 32:
                z = z & ((1 << (codes_per_word * encoding.bits)) - 1)
            payload["weight_zero_point"] = z
        elif packing == "legacy":
            raw = read(
                "weight_zero_point",
                (slice(r0 // codes_per_word, words(r1, encoding.bits, packing)), slice(g0, g1)),
            )
            z = _rebase_legacy_words(raw, encoding.bits, r0, r1 - r0, axis=0)
            payload["weight_zero_point"] = (
                z if whole_k else z[:, (positions - g0).to(z.device)]
            ).contiguous()
        else:
            z = codes("weight_zero_point", 0, r0, r1, slice(g0, g1))
            z = z[:, (positions - g0).to(z.device)]
            payload["weight_zero_point"] = pack(z, encoding.bits, mode, axis=0)
    newparts = partitions if axis == 1 else ()
    return QuantizedMatrix(
        payload,
        replace(encoding, group_in=local_group, metadata=f"{mode}:{symmetry}:{dtype}"),
        newshape,
        newparts,
    )


def slice_matrix(matrix: QuantizedMatrix, axis: int, start: int, end: int) -> QuantizedMatrix:
    return _slice(
        lambda key, index: matrix.payload[key][index],
        matrix.shape,
        matrix.encoding,
        axis,
        start,
        end,
        matrix.partitions,
    )


def materialize(
    checkpoint: QuantizedCheckpoint,
    weight: EncodedWeight,
    axis: int,
    spans: tuple[AxisSpan, ...],
    device: torch.device | str,
    readers: Mapping[str, SafeTensorReader] | None,
) -> QuantizedMatrix:
    from contextlib import ExitStack

    from cubic_serve.models.checkpoint import _file_stamp
    from cubic_serve.models.parallel_weights import confined_checkpoint_path
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix

    assert weight.encoding is not None
    logical_shape = (weight.logical.shape[0], weight.logical.shape[1])
    bykey = {s: (name, shard) for s, name, shard in weight.payload}
    with ExitStack() as stack:
        handles = {}
        stamps = {}
        for shard in sorted({s for _, s in bykey.values()}):
            if readers is not None:
                # The shared loader owns these opened readers and checks every
                # source before opening and again before publishing the batch.
                handles[shard] = readers[shard]
                continue
            path = confined_checkpoint_path(checkpoint.inventory.model_root, shard)
            checkpoint.validate_shard(shard)
            stamps[path] = _file_stamp(path)
            handles[shard] = stack.enter_context(
                safe_open(path, framework="pt", device="cpu")
            )

        def read(key: str, index: tuple[slice, ...]) -> Tensor:
            name, shard = bykey[key]
            return cast(Tensor, handles[shard].get_slice(name)[index])

        if tuple(read("weight_shape", (slice(None),)).tolist()) != weight.logical.shape:
            raise QuantizationError("INT weight_shape values disagree with logical model shape")
        pieces = [
            _slice(
                read,
                logical_shape,
                weight.encoding,
                axis,
                s.start,
                s.stop,
                weight.logical.output_partitions,
            )
            for s in spans
        ]
        if any(_file_stamp(p) != stamp for p, stamp in stamps.items()):
            raise QuantizationError("checkpoint changed during INT owner read")
    if len(pieces) == 1:
        return pieces[0].to(device)
    shape = (sum(p.shape[0] for p in pieces), pieces[0].shape[1])
    e = pieces[0].encoding
    payload = {
        key: torch.cat([p.payload[key] for p in pieces], dim=0)
        for key in ("weight_packed", "weight_scale")
    }
    payload["weight_shape"] = torch.tensor(shape, dtype=torch.int64)
    if fields(e)[1] == "asym":
        z = torch.cat(
            [
                unpack(p.payload["weight_zero_point"], e.bits, p.shape[0], 0, fields(e)[0])
                for p in pieces
            ]
        )
        payload["weight_zero_point"] = pack(z, e.bits, fields(e)[0], axis=0)
    return QuantizedMatrix(payload, e, shape).to(device)
