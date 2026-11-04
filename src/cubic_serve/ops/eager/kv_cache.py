"""Independent tensor reference for symmetric grouped KV encoding.

No model, native performance kernel or CUDA Graph is called here. The production
writer/reader must be audited against this rule before a format can serve.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.functional import pad

from cubic_serve.executor.runner import require_reference_execution
from cubic_serve.runtime.cache.quantization import (
    CacheDType,
    CacheGroupAxis,
    CacheStorageSpec,
    GroupedCacheLayout,
    KVCacheStorageSpec,
)


@dataclass(frozen=True, slots=True)
class EncodedCache:
    layout: GroupedCacheLayout
    payload: Tensor
    scales: Tensor | None


def cache_page_reference(values: Tensor, layout: GroupedCacheLayout) -> EncodedCache:
    """Canonical physical page image from independent native producer history.

    Unwritten rows and unsealed column groups have zero payload and zero scales;
    their native values belong to the separate owner checkpoint. This routine
    is qualification math, never a serving writer or a candidate-state decoder.
    """
    require_reference_execution()
    if (values.ndim != 3 or values.shape[1:] != layout.logical_shape[1:]
            or not 0 <= values.shape[0] <= layout.tokens):
        raise ValueError("cache page reference requires its native logical token prefix")
    count = values.shape[0]
    original = torch.zeros(layout.logical_shape, dtype=values.dtype, device=values.device)
    original[:count] = values
    encoded = encode_cache_reference(original, layout)
    if not layout.spec.dtype.quantized:
        return encoded
    ids = torch.arange(layout.group_count, device=values.device)
    if layout.spec.axis is CacheGroupAxis.TOKENS:
        if layout.tokens % layout.spec.group_size:
            raise ValueError("physical column-cache pages must align with complete groups")
        complete = ids % layout.group_shape[-1] < count // layout.spec.group_size
    else:
        complete = ids // (layout.heads * layout.group_shape[-1]) < count
    return EncodedCache(layout, torch.where(complete[:, None], encoded.payload, 0),
                        torch.where(complete, encoded.scales, 0))


class ReferenceCacheOperators:
    """Compose an existing reference operator set with generic cache semantics."""

    def __init__(self, operators, storage: KVCacheStorageSpec):
        self.operators, self.storage = operators, storage

    def __getattr__(self, name):
        return getattr(self.operators, name)

    def full_attention(self, query, key, value, *, query_ordinals, key_ordinals):
        return cache_attention_reference(
            query, key, value, storage=self.storage, query_ordinals=query_ordinals,
            key_ordinals=key_ordinals,
        )

    def selected_attention(self, query, key, value, indices):
        batch, _, tokens, _ = query.shape
        key_tokens = key.shape[2]
        query_ordinals = torch.arange(key_tokens - tokens, key_tokens, device=query.device,
                                      dtype=torch.int64)[None].expand(batch, -1)
        key_ordinals = torch.arange(key_tokens, device=query.device,
                                    dtype=torch.int64)[None].expand(batch, -1)
        return cache_attention_reference(
            query, key, value, storage=self.storage, query_ordinals=query_ordinals,
            key_ordinals=key_ordinals, selected_indices=indices,
        )


@dataclass(frozen=True, slots=True)
class ReferenceCacheStream:
    """Independent use-before-seal oracle, never a serving implementation.

    Completed column groups are immutable. An incomplete column group remains
    in the original compute dtype. Row groups do not depend on future tokens.
    Only an accepted append seals groups; a review itself cannot modify history.
    """

    spec: CacheStorageSpec
    sealed: tuple[EncodedCache, ...]
    tail: Tensor

    @property
    def tokens(self) -> int:
        return sum(chunk.layout.tokens for chunk in self.sealed) + self.tail.shape[0]

    def materialize(self) -> Tensor:
        require_reference_execution()
        return torch.cat(
            [decode_cache_reference(chunk, output_dtype=self.tail.dtype) for chunk in self.sealed]
            + [self.tail],
            dim=0,
        )

    def commit(self, incoming: Tensor, accepted_tokens: int) -> ReferenceCacheStream:
        require_reference_execution()
        if (
            incoming.ndim != 3
            or incoming.shape[1:] != self.tail.shape[1:]
            or incoming.dtype is not self.tail.dtype
            or incoming.device != self.tail.device
            or not bool(torch.isfinite(incoming).all())
        ):
            raise ValueError("cache append differs from its native token geometry")
        if type(accepted_tokens) is not int or not 0 <= accepted_tokens <= incoming.shape[0]:
            raise ValueError("accepted cache prefix is outside the reviewed token span")
        if accepted_tokens == 0:
            return self
        available = torch.cat((self.tail, incoming[:accepted_tokens]), dim=0)
        seal = available.shape[0]
        if self.spec.dtype.quantized and self.spec.axis is CacheGroupAxis.TOKENS:
            seal -= seal % self.spec.group_size
        chunks = self.sealed
        if seal:
            layout = GroupedCacheLayout(self.spec, seal, *available.shape[1:])
            chunks += (encode_cache_reference(available[:seal], layout),)
        return ReferenceCacheStream(self.spec, chunks, available[seal:].clone())


@dataclass(frozen=True, slots=True)
class ReferenceKVCache:
    key: ReferenceCacheStream
    value: ReferenceCacheStream
    enable_rotation: bool = False

    @classmethod
    def empty(
        cls,
        storage: KVCacheStorageSpec,
        *,
        heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
    ) -> ReferenceKVCache:
        require_reference_execution()
        resolved = storage.resolve(dtype)
        if heads <= 0 or head_dim <= 0 or dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("cache reference requires positive native token geometry")
        return cls(
            ReferenceCacheStream(
                resolved.key, (), torch.empty((0, heads, head_dim), dtype=dtype, device=device)
            ),
            ReferenceCacheStream(
                resolved.value, (), torch.empty((0, heads, head_dim), dtype=dtype, device=device)
            ),
            resolved.enable_rotation,
        )

    def review(self, key: Tensor, value: Tensor) -> ReferenceKVReview:
        require_reference_execution()
        if key.shape != value.shape or key.ndim != 3 or key.shape[0] <= 0:
            raise ValueError("review requires matching nonempty native K/V token spans")
        for incoming, stream in ((key, self.key), (value, self.value)):
            if (
                incoming.shape[1:] != stream.tail.shape[1:]
                or incoming.dtype is not stream.tail.dtype
                or incoming.device != stream.tail.device
                or not bool(torch.isfinite(incoming).all())
            ):
                raise ValueError("review requires finite native reference operands on one device")
        if self.enable_rotation:
            from cubic_serve.ops.eager.rotation import rotate_cache_reference

            key = rotate_cache_reference(key) if self.key.spec.dtype.quantized else key
            value = rotate_cache_reference(value) if self.value.spec.dtype.quantized else value
        return ReferenceKVReview(self, key.clone(), value.clone())


@dataclass(frozen=True, slots=True)
class ReferenceKVReview:
    parent: ReferenceKVCache
    key: Tensor
    value: Tensor

    def visible(self, incoming_tokens: int) -> tuple[Tensor, Tensor]:
        """Observe the same rounded history as token-at-a-time execution.

        The current token is native input. Earlier incoming tokens have already
        reached their mathematical cache commit boundary, even inside a prefill
        or MTP review. Future tokens cannot set any visible group's scale.
        """
        require_reference_execution()
        if type(incoming_tokens) is not int or not 1 <= incoming_tokens <= self.key.shape[0]:
            raise ValueError("query visibility is outside the incoming token span")
        previous = self.commit(incoming_tokens - 1)
        return (
            torch.cat(
                (previous.key.materialize(), self.key[incoming_tokens - 1 : incoming_tokens])
            ),
            torch.cat(
                (previous.value.materialize(), self.value[incoming_tokens - 1 : incoming_tokens])
            ),
        )

    def commit(self, accepted_tokens: int) -> ReferenceKVCache:
        require_reference_execution()
        return ReferenceKVCache(
            self.parent.key.commit(self.key, accepted_tokens),
            self.parent.value.commit(self.value, accepted_tokens),
            self.parent.enable_rotation,
        )


def cache_attention_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    storage: KVCacheStorageSpec,
    query_ordinals: Tensor,
    key_ordinals: Tensor,
    selected_indices: Tensor | None = None,
) -> Tensor:
    """Independent causal attention over retained native producer history.

    Reference models retain their own original K/V producers. Reconstructing a
    closed group from those unchanged operands reproduces its immutable rounding
    boundary. Each query observes only earlier committed groups plus its native
    current token; even multi-token review cannot borrow a future token's scale.
    No candidate state, performance operator or graph is an oracle input.
    """
    require_reference_execution()
    from cubic_serve.ops.eager.attention import full_attention
    from cubic_serve.ops.eager.sparse_indexer import attend_selected

    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("cache attention reference requires BHSD query and native K/V history")
    batch, heads, queries, dim = query.shape
    if (key.shape[0] != batch or key.shape[-1] != dim or key.shape[1] < 1
            or heads % key.shape[1] or min(batch, heads, queries, dim, key.shape[2]) < 1
            or query.dtype not in (torch.bfloat16, torch.float16)
            or key.dtype != query.dtype or value.dtype != query.dtype
            or key.device != query.device or value.device != query.device):
        raise ValueError("cache reference native operands differ in geometry, dtype or device")
    if (query_ordinals.shape != (batch, queries)
            or key_ordinals.shape != (batch, key.shape[2])
            or query_ordinals.dtype != torch.int64 or key_ordinals.dtype != torch.int64
            or query_ordinals.device != query.device or key_ordinals.device != query.device):
        raise ValueError("cache reference requires explicit INT64 logical token ordinals")
    expected = torch.arange(key.shape[2], dtype=torch.int64, device=query.device)
    if not torch.equal(key_ordinals, expected[None].expand_as(key_ordinals)):
        raise ValueError("cache reference requires complete native producer history from zero")
    if (bool((query_ordinals < 0).any()) or bool((query_ordinals >= key.shape[2]).any())
            or bool((query_ordinals[:, 1:] <= query_ordinals[:, :-1]).any())):
        raise ValueError("cache reference query ordinals must increase within native history")
    if selected_indices is not None and (
        selected_indices.ndim != 3 or selected_indices.shape[:2] != (batch, queries)
        or selected_indices.device != query.device
        or selected_indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("cache reference selected indices must cover the logical queries")
    output = torch.empty_like(query)
    rotate_key = storage.enable_rotation and storage.key.dtype.quantized
    rotate_value = storage.enable_rotation and storage.value.dtype.quantized
    if rotate_key or rotate_value:
        from cubic_serve.ops.eager.rotation import rotate_cache_reference

    for b in range(batch):
        stream = ReferenceKVCache.empty(
            storage, heads=key.shape[1], head_dim=dim, dtype=query.dtype, device=query.device,
        )
        native_key, native_value = key[b].transpose(0, 1), value[b].transpose(0, 1)
        for q in range(queries):
            position = int(query_ordinals[b, q])
            if stream.key.tokens < position:
                begin = stream.key.tokens
                stream = stream.review(native_key[begin:position], native_value[begin:position]
                                       ).commit(position - begin)
            review = stream.review(native_key[position:position + 1],
                                   native_value[position:position + 1])
            visible = tuple(t.transpose(0, 1)[None] for t in review.visible(1))
            row = query[b:b + 1, :, q:q + 1]
            if rotate_key:
                row = rotate_cache_reference(row)
            result_dtype = torch.float32 if rotate_value else query.dtype
            if selected_indices is None:
                result = full_attention(
                    row, *visible, query_ordinals=query_ordinals[b:b + 1, q:q + 1],
                    key_ordinals=key_ordinals[b:b + 1, :position + 1], output_dtype=result_dtype,
                )
            else:
                indices = selected_indices[b:b + 1, q:q + 1]
                if bool((indices < -1).any()) or bool((indices > position).any()):
                    raise ValueError("cache reference selection crosses its causal boundary")
                result = attend_selected(row, *visible, indices, output_dtype=result_dtype)
            if rotate_value:
                result = rotate_cache_reference(result, output_dtype=query.dtype)
            output[b:b + 1, :, q:q + 1] = result
            stream = review.commit(1)
    return output


def _groups(values: Tensor, layout: GroupedCacheLayout) -> Tensor:
    ordered = values.permute(1, 2, 0) if layout.spec.axis is CacheGroupAxis.TOKENS else values
    extra = (-ordered.shape[-1]) % layout.spec.group_size
    return pad(ordered, (0, extra)).reshape(-1, layout.spec.group_size)


def _ungroup(groups: Tensor, layout: GroupedCacheLayout) -> Tensor:
    if layout.spec.axis is CacheGroupAxis.TOKENS:
        return groups.reshape(layout.heads, layout.head_dim, -1)[..., : layout.tokens].permute(
            2, 0, 1
        )
    return groups.reshape(layout.tokens, layout.heads, -1)[..., : layout.head_dim]


def encode_cache_reference(values: Tensor, layout: GroupedCacheLayout) -> EncodedCache:
    require_reference_execution()
    if tuple(values.shape) != layout.logical_shape or values.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        raise ValueError("cache input must match its logical shape and floating dtype")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("cache encoding requires finite inputs")
    dtype = layout.spec.dtype
    if not dtype.quantized:
        stored = torch.bfloat16 if dtype is CacheDType.BF16 else torch.float16
        payload = values.to(stored).clone().contiguous()
        if not bool(torch.isfinite(payload).all()):
            raise ValueError("cache input is outside its storage dtype range")
        return EncodedCache(layout, payload, None)

    groups = _groups(values.float(), layout)
    maximum = groups.abs().amax(-1)
    # A zero group has the canonical finite scale 1. Nonzero scales retain
    # FP32 normal range, avoiding backend-dependent denormal flushing.
    # CUDA tensor/scalar division may use a rounded reciprocal multiplication.
    # Evaluate the ratio of exact FP32 operands in FP64, then round once to
    # FP32, independently witnessing the format's correctly rounded division.
    # This is reference math only; native writers retain __fdiv_rn.
    quotient = (maximum.double() / dtype.quant_max).float()
    scales = torch.where(
        maximum == 0, 1.0, quotient.clamp_min(torch.finfo(torch.float32).tiny)
    )
    normalized = (groups.double() / scales[:, None].double()).float().clamp(
        -dtype.quant_max, dtype.quant_max)
    if dtype is CacheDType.FP8:
        payload = normalized.to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        bits = dtype.bits
        codes = torch.round(normalized).to(torch.int32) & ((1 << bits) - 1)
        # Scalar bit placement is intentionally independent of a vectorized
        # performance packer. INT6 straddles bytes and has no padded INT8 path.
        payload = torch.zeros(
            (layout.group_count, layout.packed_group_bytes), dtype=torch.uint8, device=values.device
        )
        for member in range(layout.spec.group_size):
            offset = member * bits
            byte, shift = divmod(offset, 8)
            payload[:, byte] |= ((codes[:, member] << shift) & 255).to(torch.uint8)
            if shift + bits > 8:
                payload[:, byte + 1] |= (codes[:, member] >> (8 - shift)).to(torch.uint8)
    return EncodedCache(layout, payload.contiguous(), scales.contiguous())


def decode_cache_reference(encoded: EncodedCache, *, output_dtype: torch.dtype) -> Tensor:
    require_reference_execution()
    if output_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("decoded cache requires a floating output dtype")
    layout, payload, scales = encoded.layout, encoded.payload, encoded.scales
    dtype = layout.spec.dtype
    if not dtype.quantized:
        expected = torch.bfloat16 if dtype is CacheDType.BF16 else torch.float16
        if (
            scales is not None
            or tuple(payload.shape) != layout.logical_shape
            or payload.dtype is not expected
        ):
            raise ValueError("unquantized cache payload differs from its storage contract")
        return payload.to(output_dtype).clone()
    if (
        payload.dtype is not torch.uint8
        or tuple(payload.shape) != (layout.group_count, layout.packed_group_bytes)
        or not payload.is_contiguous()
        or scales is None
        or scales.dtype is not torch.float32
        or scales.shape != (layout.group_count,)
        or scales.device != payload.device
        or not bool(torch.isfinite(scales).all())
        or not bool((scales > 0).all())
    ):
        raise ValueError("encoded cache payload or scales differ from the storage contract")
    if dtype is CacheDType.FP8:
        groups = payload.view(torch.float8_e4m3fn).float()
        if not bool(torch.isfinite(groups).all()):
            raise ValueError("FP8 cache contains a nonfinite code")
    else:
        bits = dtype.bits
        groups = torch.empty(
            (layout.group_count, layout.spec.group_size), dtype=torch.float32, device=payload.device
        )
        for member in range(layout.spec.group_size):
            byte, shift = divmod(member * bits, 8)
            code = payload[:, byte].to(torch.int32) >> shift
            if shift + bits > 8:
                code |= payload[:, byte + 1].to(torch.int32) << (8 - shift)
            code &= (1 << bits) - 1
            signed = torch.where(code >= 1 << (bits - 1), code - (1 << bits), code)
            if bool((signed == -(1 << (bits - 1))).any()):
                raise ValueError("symmetric cache contains the reserved negative endpoint")
            groups[:, member] = signed.float()
    return _ungroup(groups * scales[:, None], layout).to(output_dtype).contiguous()


def witness_cache_query(native_query, actual, *, encoding, window):
    """Witness the paired query on its actual, already proved raw producer."""
    require_reference_execution()
    native = native_query.reshape_as(actual)
    expected = native
    if getattr(encoding, "enable_rotation", False) and encoding.key.spec.dtype.quantized:
        from cubic_serve.ops.eager.rotation import rotate_cache_reference

        source = None if window is None else getattr(window, "producer_query", None)
        if source is None:
            raise AssertionError("rotated query lacks its native producer snapshot")
        source = source[:native.shape[0]]
        if (source.shape != native.shape or source.dtype != native.dtype
                or not torch.equal(source.contiguous().view(torch.uint8),
                                   native.contiguous().view(torch.uint8))):
            raise AssertionError("unproved native query rotation producer")
        expected = rotate_cache_reference(source)
    if (actual.dtype != expected.dtype or not torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )):
        raise AssertionError("unproved paired native query boundary")


def witness_cache_read(
    native_key, native_value, *, key_pages, value_pages, encoding, window, table, layer, row=0
):
    """Prove encoded reads on independently witnessed actual native producers.

    This is an operator-local witness after forward completion, never an oracle
    for the complete model. Histories contain this logical query's prefix only.
    Other prepared future groups and borrowed page bytes are outside the read
    set; exact ownership/write-set witnesses remain separate obligations.
    """
    require_reference_execution()
    if (
        native_key.ndim != 3
        or native_value.shape != native_key.shape
        or native_key.dtype != native_value.dtype
        or native_key.device != native_value.device
        or native_key.shape[0] < 1
    ):
        raise ValueError("cache witness requires matched native token/head/channel histories")
    length = native_key.shape[0]
    start = int(window.start_positions[row])
    root = int(window.start_rows[row])
    if not (0 <= start < length and 0 <= root <= row and start + row - root == length - 1):
        raise AssertionError("cache read window differs from the native causal prefix")
    if not torch.equal(
        window.start_rows[root : row + 1], torch.full_like(window.start_rows[root : row + 1], root)
    ):
        raise AssertionError("cache read span crosses another logical owner")
    activity = getattr(window, "activity", None)
    if activity is not None and not torch.equal(
        activity[root:row + 1], torch.ones_like(activity[root:row + 1]),
    ):
        raise AssertionError("cache witness span contains a row without logical ownership")

    def identical(left, right):
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(
                left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
            )
        )

    rotation = getattr(encoding, "enable_rotation", False)
    for native, incoming, payload, scales, layout, tail, producer in (
        (native_key, window.key, key_pages, encoding.key_scales, encoding.key, window.key_tail,
         getattr(window, "producer_key", None)),
        (
            native_value,
            window.value,
            value_pages,
            encoding.value_scales,
            encoding.value,
            window.value_tail,
            getattr(window, "producer_value", None),
        ),
    ):
        if rotation and layout.spec.dtype.quantized:
            from cubic_serve.ops.eager.rotation import rotate_cache_reference

            if producer is None or not identical(native[start:], producer[root:row + 1]):
                raise AssertionError("unproved native cache rotation producer")
            native = rotate_cache_reference(native)
        if not identical(native[start:], incoming[root : row + 1]):
            raise AssertionError("unproved native cache producer")
        for page_index in range((length + layout.tokens - 1) // layout.tokens):
            values = native[page_index * layout.tokens : (page_index + 1) * layout.tokens]
            image = cache_page_reference(values, layout)
            physical = int(table[row, page_index])
            if layout.spec.dtype.quantized:
                groups = layout.groups_for_tokens(tuple(range(values.shape[0])), complete=True)
                ids = torch.tensor(groups, dtype=torch.int64, device=payload.device)
                if not identical(payload[physical, layer, ids], image.payload[ids]):
                    raise AssertionError("cache encoding differs on actual native producers")
                if not identical(scales[physical, layer, ids], image.scales[ids]):
                    raise AssertionError("cache scale differs on actual native producers")
            elif not identical(payload[physical, layer, : values.shape[0]], values):
                raise AssertionError("plain cache differs on actual native producers")
        if layout.spec.dtype.quantized and layout.spec.axis is CacheGroupAxis.TOKENS:
            if not 0 <= int(window.sources[row]) < tail.shape[0]:
                raise AssertionError("cache tail owner lies outside the snapshot")
            expected = torch.zeros_like(tail[0, layer])
            count = start % layout.spec.group_size
            if count:
                expected[:count] = native[start - count : start]
            if not identical(tail[int(window.sources[row]), layer], expected):
                raise AssertionError("unproved pre-attention native tail boundary")
    cache = ReferenceKVCache.empty(
        KVCacheStorageSpec(encoding.key.spec, encoding.value.spec, rotation),
        heads=native_key.shape[1],
        head_dim=native_key.shape[-1],
        dtype=native_key.dtype,
        device=native_key.device,
    )
    return cache.review(native_key, native_value).visible(length)
