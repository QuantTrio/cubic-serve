"""Grouped cache encoding boundaries; attention consumes the packed views.

The format writer is prepared before capture. It never selects a language-model
execution mode, allocates during replay, or materializes historical KV to BF16.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
import triton
import triton.language as tl
from torch import Tensor

from cubic_serve.ops.cuda.compiled_artifact import bind_tensor
from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.runtime.cache.quantization import (
    CacheDType, CacheGroupAxis, GroupedCacheLayout,
    slot_storage_contiguous, storage_regions_overlap,
)

if TYPE_CHECKING:
    from cubic_serve.ops.cuda.native import NativeLaunch
    from cubic_serve.runtime.cache.pages import CacheArena


@dataclass(frozen=True, slots=True)
class PagedKVEncoding:
    """Storage ABI for already sealed pages; tail publication is a separate step."""

    key: GroupedCacheLayout
    value: GroupedCacheLayout
    key_scales: Tensor | None = field(default=None, repr=False, compare=False)
    value_scales: Tensor | None = field(default=None, repr=False, compare=False)
    enable_rotation: bool = False
    shared_storage: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.enable_rotation) is not bool:
            raise TypeError("encoded cache rotation requires a boolean")
        if self.key.logical_shape != self.value.logical_shape:
            raise ValueError("K and V must describe the same logical page geometry")
        if not (self.key.spec.dtype.quantized or self.value.spec.dtype.quantized):
            raise ValueError("ordinary KV uses the existing unencoded page ABI")

    @property
    def rotation_contract(self) -> dict[str, object] | None:
        if not self.enable_rotation:
            return None
        dim = self.key.head_dim
        return {
            "algorithm": "per-head-block-hadamard-fp32-rn-v1",
            "block": dim & -dim,
            "key": self.key.spec.dtype.quantized,
            "value": self.value.spec.dtype.quantized,
            "forward": "FP32 butterfly and normalization, then native RNE",
            "inverse": "normalized FP32 attention, inverse, then native RNE",
        }

    def validate(self, key: Tensor, value: Tensor) -> tuple[int, int, int, int, int]:
        if key.ndim < 2 or value.ndim < 2 or key.shape[:2] != value.shape[:2]:
            raise ValueError("encoded K and V require matching page/layer extents")
        pages, layers = key.shape[:2]
        if pages <= 0 or layers <= 0 or key.device != value.device:
            raise ValueError("encoded pages require positive extents on one device")
        tensors = [key, value]
        for data, scales, layout in (
            (key, self.key_scales, self.key),
            (value, self.value_scales, self.value),
        ):
            if layout.spec.dtype in (
                CacheDType.FP8, CacheDType.INT8, CacheDType.INT6, CacheDType.INT4,
            ):
                if (
                    tuple(data.shape)
                    != (pages, layers, layout.group_count, layout.packed_group_bytes)
                    or data.dtype is not torch.uint8
                    or scales is None
                    or tuple(scales.shape) != (pages, layers, layout.group_count)
                    or scales.dtype is not torch.float32
                    or scales.device != data.device
                    or not slot_storage_contiguous(scales)
                ):
                    raise ValueError("encoded page codes/scales differ from their declared layout")
                tensors.append(scales)
            elif layout.spec.dtype in (CacheDType.BF16, CacheDType.FP16):
                expected = torch.bfloat16 if layout.spec.dtype is CacheDType.BF16 else torch.float16
                if (
                    tuple(data.shape) != (pages, layers, *layout.logical_shape)
                    or data.dtype is not expected
                    or scales is not None
                ):
                    raise ValueError("plain page differs from its declared storage layout")
            else:
                raise ValueError(
                    "encoded attention has no native implementation for this storage format"
                )
            if not slot_storage_contiguous(data):
                raise ValueError("encoded page storage requires contiguous inner page payloads")
        if any(storage_regions_overlap(left, right) for i, left in enumerate(tensors)
               for right in tensors[i + 1:]):
            raise ValueError("page codes and scales must occupy disjoint fields")
        return pages, layers, *self.key.logical_shape


@dataclass(frozen=True, slots=True)
class PagedKVReadWindow:
    """Native tail and current span over an immutable encoded prefix.

    start_positions and start_rows identify the incoming segment for each row.
    sources names its immutable pre-forward state slot. Column tails store native
    values at absolute_position % group_size; row groups need no historical tail.
    These are startup-bound buffers, filled by the existing topology lease.
    """

    key: Tensor
    value: Tensor
    start_positions: Tensor
    start_rows: Tensor
    sources: Tensor
    key_tail: Tensor | None = None
    value_tail: Tensor | None = None
    activity: Tensor | None = None
    producer_query: Tensor | None = None
    producer_key: Tensor | None = None
    producer_value: Tensor | None = None

    @property
    def tensors(self) -> tuple[Tensor, ...]:
        return (
            self.key,
            self.value,
            self.start_positions,
            self.start_rows,
            self.sources,
            *(t for t in (self.key_tail, self.value_tail) if t is not None),
            *((self.activity,) if self.activity is not None else ()),
            *(t for t in (self.producer_query, self.producer_key, self.producer_value)
              if t is not None),
        )

    def validate(self, query: Tensor, encoding: PagedKVEncoding, *, layers: int) -> None:
        rows, _, _ = query.shape
        heads, dim = encoding.key.logical_shape[1:]
        if any(
            tensor.device != query.device or not (
                slot_storage_contiguous(tensor) if tensor is self.key_tail or tensor is self.value_tail
                else tensor.is_contiguous()) for tensor in self.tensors
        ):
            raise ValueError("native cache window must be contiguous on the query device")
        if (
            self.key.shape != (rows, heads, dim)
            or self.value.shape != self.key.shape
            or self.key.dtype is not query.dtype
            or self.value.dtype is not query.dtype
        ):
            raise ValueError("incoming K/V span differs from native query geometry")
        for metadata, dtype in (
            (self.start_positions, torch.int64),
            (self.start_rows, torch.int32),
            (self.sources, torch.int64),
        ):
            if metadata.shape != (rows,) or metadata.dtype is not dtype:
                raise ValueError("native cache window requires exact row metadata")
        if self.activity is not None and (
            self.activity.shape != (rows,) or self.activity.dtype is not torch.int32
        ):
            raise ValueError("native cache window requires explicit INT32 row activity")
        for source, expected in (
            (self.producer_key, self.key), (self.producer_value, self.value),
        ):
            if source is not None and (
                source.shape != expected.shape or source.dtype != query.dtype
            ):
                raise ValueError("cache rotation producer differs from its native boundary")
        if self.producer_query is not None and (
            self.producer_query.ndim != 3 or self.producer_query.shape[0] != rows
            or self.producer_query.shape[1] < 1 or self.producer_query.shape[-1] != dim
            or self.producer_query.dtype != query.dtype
        ):
            raise ValueError("query rotation producer differs from its native head geometry")
        for tail, layout in ((self.key_tail, encoding.key), (self.value_tail, encoding.value)):
            required = layout.spec.dtype.quantized and layout.spec.axis is CacheGroupAxis.TOKENS
            if required:
                if (
                    tail is None
                    or tail.ndim != 5
                    or tail.shape[0] <= 0
                    or tail.shape[1:] != (layers, layout.spec.group_size, heads, dim)
                    or tail.dtype is not query.dtype
                ):
                    raise ValueError("column cache requires native per-owner tail storage")
            elif tail is not None:
                raise ValueError("row and plain cache have no historical column tail")


def paged_cache_encoding(arena: CacheArena) -> PagedKVEncoding | None:
    """Resolve format-owned operands from a shared arena, independently of model."""
    if not arena.layout.storage.quantized:
        return None
    return PagedKVEncoding(
        arena.layout.key_encoding, arena.layout.value_encoding,
        arena.page_fields.get("kv_key_scales"), arena.page_fields.get("kv_value_scales"),
        arena.layout.storage.enable_rotation, getattr(arena,"shared_storage",None),
    )


class PagedKVGraphBinding:
    """Shared segmented ownership and native append boundaries for a graph.

    The engine supplies the actual state-map owner stride. Formats own their
    codes/scales; all models use the same source-tail and destination-checkpoint
    protocol. Construction allocates scratch, never durable owner state.
    """

    def __init__(
        self, arena: CacheArena, *, positions: Tensor, segments: Tensor, activity: Tensor,
        source_slots: Tensor, destination_slots: Tensor, owner_stride: int = 1,
        positions_are_lengths: bool = False,
    ) -> None:
        encoding = paged_cache_encoding(arena)
        if encoding is None:
            raise ValueError("native cache binding requires an explicitly encoded arena")
        self.arena, self.encoding = arena, encoding
        self.segments, self.activity = segments, activity
        self._lengths = positions if positions_are_lengths else None
        self.positions = (torch.empty_like(positions, dtype=torch.int64)
                          if positions_are_lengths else positions)
        self.starts = torch.empty_like(self.positions)
        self.roots = torch.empty_like(segments)
        self.sources = torch.empty_like(self.positions)
        self.destinations = torch.empty_like(self.positions)
        self._metadata = bind_cache_window_metadata(
            segments, activity, self.positions, source_slots, self.starts, self.roots,
            self.sources, owner_stride=owner_stride,
            destination_slots=destination_slots, destinations=self.destinations,
        )
        self._rotation_scratch: dict[tuple, Tensor] = {}
        self._rotation_plans: dict[tuple, NativeLaunch] = {}

    @property
    def scratch(self) -> tuple[Tensor, ...]:
        return (self.starts, self.roots, self.sources, self.destinations,
                *((self.positions,) if self._lengths is not None else ()),
                *self._rotation_scratch.values())

    def _rotation_output(self, role: str, source: Tensor) -> Tensor:
        identity = role, tuple(source.shape), source.dtype, source.device
        if identity not in self._rotation_scratch:
            self._rotation_scratch[identity] = torch.empty_like(source)
        return self._rotation_scratch[identity]

    def query_for(self, query: Tensor) -> Tensor:
        if not (self.encoding.enable_rotation and self.encoding.key.spec.dtype.quantized):
            return query
        identity = "query", tuple(query.shape), query.dtype, query.device
        if identity not in self._rotation_scratch:
            raise ValueError("paired query rotation must be bound before attention")
        return self._rotation_scratch[identity]

    def prepare_metadata(self) -> None:
        if self._lengths is not None:
            torch.sub(self._lengths, 1, out=self.positions)
        self._metadata.launch()

    def bind_layer(
        self, key: Tensor, value: Tensor, *, layer: int, pages: Tensor,
        query: Tensor | None = None,
    ) -> tuple[PagedKVReadWindow, PackedOperatorBinding, PackedOperatorBinding]:
        rotation = None
        producer_query = producer_key = producer_value = None
        if self.encoding.enable_rotation:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("cache rotation must be bound before capture")
            from cubic_serve.ops.cuda.cache_rotation_native import bind_cache_rotation

            rotate_key = self.encoding.key.spec.dtype.quantized
            rotate_value = self.encoding.value.spec.dtype.quantized
            if rotate_key and query is None:
                raise ValueError("encoded key rotation requires its native query producer")
            query_out = self._rotation_output("query", query) if rotate_key else None
            key_out = self._rotation_output("key", key) if rotate_key else None
            value_out = self._rotation_output("value", value) if rotate_value else None
            identity = tuple(None if t is None else bind_tensor(t) for t in (query, key, value))
            if identity not in self._rotation_plans:
                self._rotation_plans[identity] = bind_cache_rotation(
                    query, key, value, query_output=query_out, key_output=key_out,
                    value_output=value_out, rotate_key=rotate_key, rotate_value=rotate_value,
                )
            rotation = self._rotation_plans[identity]
            producer_query = query if rotate_key else None
            producer_key = key if rotate_key else None
            producer_value = value if rotate_value else None
            key = key if key_out is None else key_out
            value = value if value_out is None else value_out
        window = PagedKVReadWindow(
            key, value, self.starts, self.roots, self.sources,
            self.arena.states.get("kv_key_tail"), self.arena.states.get("kv_value_tail"),
            self.activity,
            producer_query, producer_key, producer_value,
        )
        options = dict(
            layer=layer, positions=self.positions, pages=pages, destinations=self.destinations,
            segments=self.segments, activity=self.activity,
        )
        prepare = bind_cache_append(
            self.arena.key, self.arena.value, self.arena.written, self.encoding, window,
            phase="prepare", **options,
        )
        if rotation is not None:
            prepare = PackedOperatorBinding(
                prepare.output, (rotation, *prepare._launches), scratch=prepare.scratch,
            )
        checkpoint = bind_cache_append(
            self.arena.key, self.arena.value, self.arena.written, self.encoding, window,
            phase="checkpoint", **options,
        )
        return window, prepare, checkpoint


@triton.jit
def cache_group_indices(
    token,
    head,
    channel,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    COLUMN: tl.constexpr,
):
    if COLUMN:
        groups = tl.cdiv(TOKENS, GROUP)
        return (head * DIM + channel) * groups + token // GROUP, token % GROUP
    else:
        groups = tl.cdiv(DIM, GROUP)
        return (token * HEADS + head) * groups + channel // GROUP, channel % GROUP


@triton.jit
def load_cache_group(
    payload,
    scales,
    group,
    member,
    mask,
    GROUP: tl.constexpr,
    BITS: tl.constexpr,
    FP8: tl.constexpr,
    OUTPUT: tl.constexpr,
):
    """Local decode at the attention load boundary, with no intermediate slab."""
    if FP8:
        code = tl.load(payload + group * GROUP + member, mask, other=0)
        value = _decode_e4m3fn(code)
    else:
        group_bytes: tl.constexpr = triton.cdiv(GROUP * BITS, 8)
        byte = member * BITS // 8
        shift = member * BITS % 8
        code = tl.load(payload + group * group_bytes + byte, mask, other=0).to(tl.int32)
        if BITS == 6:
            spill = tl.load(
                payload + group * group_bytes + byte + 1, mask & (shift + BITS > 8), other=0
            ).to(tl.int32)
            code |= spill << 8
        code = (code >> shift) & ((1 << BITS) - 1)
        value = tl.where(code >= (1 << (BITS - 1)), code - (1 << BITS), code).to(tl.float32)
    scale = tl.load(scales + group, mask, other=1)
    return (value * scale).to(OUTPUT)


@triton.jit
def _decode_e4m3fn(code):
    # Decode the storage format with baseline integer/FP32 instructions. E4M3FN
    # is storage, not a requirement for native FP8 instructions on the device.
    bits = code.to(tl.uint32)
    sign = (bits & 0x80) << 24
    exponent, mantissa = (bits >> 3) & 15, bits & 7
    normal = ((exponent + 120) << 23) | (mantissa << 20)
    subnormal = (mantissa.to(tl.float32) * 0.001953125).to(tl.uint32, bitcast=True)
    magnitude = tl.where(exponent == 0, subnormal, normal)
    magnitude = tl.where((exponent == 15) & (mantissa == 7), 0x7FC00000, magnitude)
    return (magnitude | sign).to(tl.float32, bitcast=True)


@triton.jit
def _fp8_scale(maximum):
    return tl.where(
        maximum == 0, 1.0, tl.maximum(tl.div_rn(maximum, 448.0), 1.1754943508222875e-38)
    )


@triton.jit
def _fp8_codes(values, scale):
    normalized = tl.minimum(tl.maximum(tl.div_rn(values.to(tl.float32), scale), -448.0), 448.0)
    bits = normalized.to(tl.uint32, bitcast=True)
    magnitude = bits & 0x7FFFFFFF
    # FP32 has twenty more fraction bits. Half minus one plus the retained
    # parity bit implements RNE, including carries into the next exponent.
    rounded = magnitude + 0x7FFFF + ((magnitude >> 20) & 1)
    normal = (rounded >> 20) - 960  # (127 - E4M3 bias 7) * 8.
    subnormal = tl.inline_asm_elementwise(
        "cvt.rni.u32.f32 $0, $1;",
        "=r,f",
        [(magnitude.to(tl.float32, bitcast=True) * 512.0)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    code = tl.where(magnitude < 0x3C800000, subnormal, normal)
    return (code | ((bits >> 24) & 0x80)).to(tl.uint8)


@triton.jit
def _write_fp8_groups(
    values,
    payload,
    scales,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    COLUMN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    members = tl.arange(0, BLOCK)
    if COLUMN:
        count: tl.constexpr = triton.cdiv(TOKENS, GROUP)
        channel = group // count % DIM
        head = group // (count * DIM)
        token = group % count * GROUP + members
        valid = (members < GROUP) & (token < TOKENS)
    else:
        count: tl.constexpr = triton.cdiv(DIM, GROUP)
        channel = group % count * GROUP + members
        head = group // count % HEADS
        token = group // (count * HEADS)
        valid = (members < GROUP) & (channel < DIM)
    offsets = (token * HEADS + head) * DIM + channel
    x = tl.load(values + offsets, valid, other=0).to(tl.float32)
    maximum = tl.max(tl.abs(x), 0)
    scale = _fp8_scale(maximum)
    encoded = _fp8_codes(x, scale)
    tl.store(payload + group * GROUP + members, encoded, members < GROUP)
    tl.store(scales + group, scale)


@triton.jit
def _append_column_groups(
    incoming,
    payload,
    scales,
    tail,
    sources,
    destinations,
    segments,
    activity,
    positions,
    pages,
    written,
    ROWS: tl.constexpr,
    LAYERS: tl.constexpr,
    LAYER: tl.constexpr,
    PAGE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    GB: tl.constexpr,
    DB: tl.constexpr,
    STORE_GROUPS: tl.constexpr,
    STORE_TAIL: tl.constexpr,
):
    root = tl.program_id(0)
    length = tl.load(segments + root)
    if (length <= 0) | (tl.load(activity + root) == 0):
        return
    flat = tl.program_id(1) * DB + tl.arange(0, DB)
    member = tl.arange(0, GB)
    valid = (member[:, None] < GROUP) & (flat[None, :] < HEADS * DIM)
    source = tl.load(sources + root)
    base = ((source * LAYERS + LAYER) * GROUP + member[:, None]) * HEADS * DIM
    history = tl.load(tail + base + flat[None, :], valid & (source >= 0), other=0)
    first_position = tl.load(positions + root)
    for step in range(tl.minimum(length, ROWS - root)):
        row = root + step
        if tl.load(activity + row) != 0:
            position = first_position + step
            current = tl.load(incoming + row * HEADS * DIM + flat, flat < HEADS * DIM, 0)
            history = tl.where(member[:, None] == position % GROUP, current[None, :], history)
            if (position + 1) % GROUP == 0:
                if STORE_GROUPS:
                    x = history.to(tl.float32)
                    maximum = tl.max(tl.where(valid, tl.abs(x), 0), 0)
                    scale = _fp8_scale(maximum)
                    codes = _fp8_codes(x, scale[None, :])
                    page = tl.load(pages + row).to(tl.int64)
                    group = (
                        (page * LAYERS + LAYER) * HEADS * DIM * (PAGE // GROUP)
                        + flat * (PAGE // GROUP)
                        + (position % PAGE) // GROUP
                    )
                    tl.store(payload + group[None, :] * GROUP + member[:, None], codes, valid)
                    tl.store(scales + group, scale, flat < HEADS * DIM)
                # A sealed group's native copy is dead. Canonical zeroing keeps
                # checkpoints independent of rejected/lookahead ring contents.
                history = tl.full((GB, DB), 0, incoming.dtype.element_ty)
            if STORE_TAIL:
                destination = tl.load(destinations + row).to(tl.int64)
                out = ((destination * LAYERS + LAYER) * GROUP + member[:, None]) * HEADS * DIM
                tl.store(tail + out + flat[None, :], history, valid & (destination >= 0))
            if STORE_GROUPS and tl.program_id(1) == 0:
                page = tl.load(pages + row).to(tl.int64)
                tl.store(written + (page * LAYERS + LAYER) * PAGE + position % PAGE, 1)


@triton.jit
def _append_row_groups(
    incoming,
    payload,
    scales,
    positions,
    pages,
    activity,
    written,
    LAYERS: tl.constexpr,
    LAYER: tl.constexpr,
    PAGE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    GB: tl.constexpr,
    QUANTIZED: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(activity + row) == 0:
        return
    group = tl.program_id(1)
    groups: tl.constexpr = triton.cdiv(DIM, GROUP)
    head = group // groups
    member = tl.arange(0, GB)
    channel = (group % groups) * GROUP + member
    valid = (member < GROUP) & (channel < DIM)
    x = tl.load(incoming + (row * HEADS + head) * DIM + channel, valid, 0)
    position = tl.load(positions + row)
    page = tl.load(pages + row).to(tl.int64)
    token = position % PAGE
    if QUANTIZED:
        maximum = tl.max(tl.abs(x.to(tl.float32)), 0)
        scale = _fp8_scale(maximum)
        codes = _fp8_codes(x, scale)
        destination = ((page * LAYERS + LAYER) * PAGE + token) * HEADS * groups + group
        tl.store(payload + destination * GROUP + member, codes, member < GROUP)
        tl.store(scales + destination, scale)
    else:
        destination = (((page * LAYERS + LAYER) * PAGE + token) * HEADS + head) * DIM + channel
        tl.store(payload + destination, x, valid)
    if group == 0:
        tl.store(written + (page * LAYERS + LAYER) * PAGE + token, 1)


def bind_cache_append(
    key: Tensor,
    value: Tensor,
    written: Tensor,
    encoding: PagedKVEncoding,
    window: PagedKVReadWindow,
    *,
    layer: int,
    positions: Tensor,
    pages: Tensor,
    destinations: Tensor,
    segments: Tensor,
    activity: Tensor,
    phase: Literal["complete", "prepare", "checkpoint"] = "complete",
    implementation: Literal["cuda", "triton"] = "cuda",
) -> PackedOperatorBinding:
    """Prepare visible groups before attention; checkpoint native tails after it.

    Sources are pre-forward native checkpoints; each segment owns its destination
    snapshots and writable pages. A speculative caller still owns accepted-prefix
    selection, page COW and rollback. This binding cannot authorize publication.
    ``complete`` is useful only when attention is absent; an attention graph must
    use prepare/checkpoint so an in-place owner tail survives all its consumers.
    """
    page_count, layers, page, heads, dim = encoding.validate(key, value)
    if any(
        layout.spec.dtype.quantized and layout.spec.axis is CacheGroupAxis.TOKENS
        and page % layout.spec.group_size
        for layout in (encoding.key, encoding.value)
    ):
        raise ValueError("column cache groups must divide their page")
    rows = window.key.shape[0]
    if phase not in ("complete", "prepare", "checkpoint"):
        raise ValueError("cache append requires a declared preparation/checkpoint phase")
    if implementation not in ("cuda", "triton"):
        raise ValueError("unknown cache encoding implementation")
    # Window validation needs only the original compute dtype and row/head shape.
    window.validate(window.key, encoding, layers=layers)
    if (
        key.device.type != "cuda"
        or type(layer) is not int
        or not 0 <= layer < layers
        or written.shape != (page_count, layers, page)
        or written.dtype is not torch.bool
    ):
        raise ValueError("cache append differs from its paged layer geometry")
    for tensor, dtype in (
        (positions, torch.int64),
        (pages, torch.int32),
        (segments, torch.int32),
        (activity, torch.int32),
    ):
        if tensor.shape != (rows,) or tensor.dtype is not dtype:
            raise ValueError("cache append requires exact logical row metadata")
    if destinations.shape != (rows,) or destinations.dtype not in (torch.int32, torch.int64):
        raise ValueError("cache append requires exact destination owner slots")
    all_tensors = (
        *window.tensors,
        key,
        value,
        written,
        positions,
        pages,
        destinations,
        segments,
        activity,
        *(t for t in (encoding.key_scales, encoding.value_scales) if t is not None),
    )
    fields = (key, value, written, window.key_tail, window.value_tail,
              encoding.key_scales, encoding.value_scales)
    if any(t.device != key.device or not (
            slot_storage_contiguous(t) if any(t is field for field in fields)
            else t.is_contiguous()) for t in all_tensors):
        raise ValueError("cache append operands must be contiguous on the cache device")
    writable = (
        key,
        value,
        written,
        window.key_tail,
        window.value_tail,
        encoding.key_scales,
        encoding.value_scales,
    )
    for output in (t for t in writable if t is not None):
        if any(
            t is not output
            and storage_regions_overlap(t, output)
            and not (encoding.shared_storage is not None
                     and encoding.shared_storage.allows_domain_overlap(t, output))
            for t in all_tensors
        ):
            raise ValueError("cache append output may not alias another operand")
    if implementation == "triton" and any(
        layout.spec.dtype.quantized and layout.spec.dtype is not CacheDType.FP8
        for layout in (encoding.key, encoding.value)
    ):
        raise ValueError("Triton cache compatibility supports only FP8 encoding")
    calls = []
    for incoming, payload, scales, tail, layout in (
        (window.key, key, encoding.key_scales, window.key_tail, encoding.key),
        (window.value, value, encoding.value_scales, window.value_tail, encoding.value),
    ):
        group = layout.spec.group_size
        quantized = layout.spec.dtype.quantized
        if implementation == "cuda":
            if phase == "checkpoint" and not (
                quantized and layout.spec.axis is CacheGroupAxis.TOKENS
            ):
                continue
            from cubic_serve.ops.cuda.cache_quantization_native import bind_cache_launch

            calls.append(bind_cache_launch(
                incoming, payload, scales, layout, tail=tail, sources=window.sources,
                destinations=destinations, segments=segments, activity=activity,
                positions=positions, pages=pages, written=written, layers=layers,
                layer=layer, store_groups=phase != "checkpoint", store_tail=phase != "prepare",
            ))
            continue
        if quantized and layout.spec.axis is CacheGroupAxis.TOKENS:
            assert tail is not None and scales is not None
            calls.append(
                _PackedLaunch.compile(
                    _append_column_groups,
                    (
                        incoming,
                        payload,
                        scales,
                        tail,
                        window.sources,
                        destinations,
                        segments,
                        activity,
                        positions,
                        pages,
                        written,
                    ),
                    (
                        rows,
                        layers,
                        layer,
                        page,
                        heads,
                        dim,
                        group,
                        triton.next_power_of_2(group),
                        32,
                        phase != "checkpoint",
                        phase != "prepare",
                    ),
                    (rows, triton.cdiv(heads * dim, 32)),
                    {"num_warps": 4, "enable_fp_fusion": False},
                )
            )
        elif phase != "checkpoint":
            calls.append(
                _PackedLaunch.compile(
                    _append_row_groups,
                    (
                        incoming,
                        payload,
                        written if scales is None else scales,
                        positions,
                        pages,
                        activity,
                        written,
                    ),
                    (
                        layers,
                        layer,
                        page,
                        heads,
                        dim,
                        group,
                        triton.next_power_of_2(group),
                        quantized,
                    ),
                    (rows, heads * triton.cdiv(dim, group)),
                    {"num_warps": 4, "enable_fp_fusion": False},
                )
            )
    return PackedOperatorBinding(written, tuple(calls))


@triton.jit
def _cache_window_metadata(
    segments,
    activity,
    positions,
    source_slots,
    start_positions,
    start_rows,
    sources,
    destination_slots,
    destinations,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    OWNER_STRIDE: tl.constexpr,
    HAS_DESTINATIONS: tl.constexpr,
):
    root = tl.program_id(0)
    if HAS_DESTINATIONS:
        destination = tl.load(destination_slots + root).to(tl.int64)
        tl.store(destinations + root, destination // OWNER_STRIDE)
    if tl.load(activity + root) == 0:
        # Padding keeps its caller-assigned private source; no live segment
        # inherits this row and zero continuation is never used as activity.
        tl.store(start_positions + root, 0)
        tl.store(start_rows + root, root)
        tl.store(sources + root, tl.load(source_slots + root).to(tl.int64) // OWNER_STRIDE)
        return
    length = tl.load(segments + root)
    if length > 0:
        length = tl.minimum(length, ROWS - root)
        first = tl.load(positions + root)
        source = tl.load(source_slots + root).to(tl.int64) // OWNER_STRIDE
        for begin in range(tl.cdiv(length, BLOCK)):
            offset = begin * BLOCK + tl.arange(0, BLOCK)
            row = root + offset
            valid = offset < length
            live = tl.load(activity + row, valid, 0) != 0
            tl.store(start_positions + row, first, valid & live)
            tl.store(start_rows + row, root, valid & live)
            tl.store(sources + row, source, valid & live)


def bind_cache_window_metadata(
    segments: Tensor,
    activity: Tensor,
    positions: Tensor,
    source_slots: Tensor,
    start_positions: Tensor,
    start_rows: Tensor,
    sources: Tensor,
    *,
    owner_stride: int = 1,
    destination_slots: Tensor | None = None,
    destinations: Tensor | None = None,
) -> PackedOperatorBinding:
    """Project existing segmented row ownership into native cache read windows."""
    rows = segments.numel()
    has_destinations = destination_slots is not None
    if (type(owner_stride) is not int or owner_stride < 1
            or has_destinations != (destinations is not None)):
        raise ValueError("cache owner stride and destination metadata must be explicit")
    tensors = (segments, activity, positions, source_slots, start_positions, start_rows, sources)
    if has_destinations:
        assert destination_slots is not None and destinations is not None
        tensors += (destination_slots, destinations)
    if (
        rows <= 0
        or segments.device.type != "cuda"
        or any(
            t.shape != (rows,) or t.device != segments.device or not t.is_contiguous()
            for t in tensors
        )
    ):
        raise ValueError("cache window metadata requires contiguous same-device row vectors")
    if (
        segments.dtype is not torch.int32
        or activity.dtype is not torch.int32
        or positions.dtype is not torch.int64
        or source_slots.dtype not in (torch.int32, torch.int64)
        or start_positions.dtype is not torch.int64
        or start_rows.dtype is not torch.int32
        or sources.dtype is not torch.int64
        or (has_destinations and (destination_slots.dtype not in (torch.int32, torch.int64)
                                 or destinations.dtype is not torch.int64))
    ):
        raise ValueError("cache window metadata differs from its integer ownership ABI")
    outputs = (start_positions, start_rows, sources)
    if has_destinations:
        outputs += (destinations,)
    for output in outputs:
        if any(
            t is not output
            and t.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
            for t in tensors
        ):
            raise ValueError("cache window output may not alias another metadata operand")
    call = _PackedLaunch.compile(
        _cache_window_metadata,
        (*tensors[:7], source_slots if destination_slots is None else destination_slots,
         sources if destinations is None else destinations),
        (rows, 128, owner_stride, has_destinations),
        (rows,),
        {"num_warps": 4},
    )
    return PackedOperatorBinding(start_positions, (call,))


class CacheGroupWritePrimitive(PackedOperatorBinding):
    """Fixed-address format writer for a sealed [token, head, channel] group span.

    Sealing/tail policy is owned by the cache transaction, not inferred here.
    Unsupported formats fail before compilation instead of silently using FP8.
    """

    def __init__(
        self, values: Tensor, payload: Tensor, scales: Tensor, layout: GroupedCacheLayout,
        *, implementation: Literal["cuda", "triton"] = "cuda",
    ) -> None:
        if layout.spec.dtype not in (
            CacheDType.FP8, CacheDType.INT8, CacheDType.INT6, CacheDType.INT4,
        ):
            raise ValueError("this cache writer has no native implementation for the encoding")
        if (
            values.device.type != "cuda"
            or values.dtype not in (torch.bfloat16, torch.float16)
            or values.shape != layout.logical_shape
            or not values.is_contiguous()
            or payload.shape != (layout.group_count, layout.packed_group_bytes)
            or payload.dtype is not torch.uint8
            or not payload.is_contiguous()
            or scales.shape != (layout.group_count,)
            or scales.dtype is not torch.float32
            or not scales.is_contiguous()
            or not (values.device == payload.device == scales.device)
            or len({t.untyped_storage().data_ptr() for t in (values, payload, scales)}) != 3
        ):
            raise ValueError(
                "cache writer requires separate fixed-address values, codes and scales"
            )
        self.layout = layout
        if implementation == "cuda":
            from cubic_serve.ops.cuda.cache_quantization_native import bind_cache_launch

            call = bind_cache_launch(values, payload, scales, layout)
            super().__init__(payload, (call,))
            return
        if implementation != "triton":
            raise ValueError("unknown cache encoding implementation")
        if layout.spec.dtype is not CacheDType.FP8:
            raise ValueError("Triton cache compatibility supports only FP8 encoding")
        call = _PackedLaunch.compile(
            _write_fp8_groups,
            (values, payload, scales),
            (
                layout.tokens,
                layout.heads,
                layout.head_dim,
                layout.spec.group_size,
                layout.spec.axis is CacheGroupAxis.TOKENS,
                triton.next_power_of_2(layout.spec.group_size),
            ),
            (layout.group_count,),
            {"num_warps": 4, "enable_fp_fusion": False},
        )
        super().__init__(payload, (call,))
