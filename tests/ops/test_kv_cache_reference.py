"""Known-code and error-bound checks for the independent cache oracle."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cubic_serve.executor.runner import ReferenceExecutionContext
from cubic_serve.ops.eager.kv_cache import (
    ReferenceKVCache,
    witness_cache_read,
    cache_attention_reference,
    cache_page_reference,
    decode_cache_reference,
    encode_cache_reference,
)
from cubic_serve.runtime.cache.quantization import (
    CacheDType,
    CacheGroupAxis,
    CacheStorageSpec,
    GroupedCacheLayout,
    KVCacheStorageSpec,
)


@pytest.mark.parametrize("width", [1, 3, 7, 16, 32, 64, 96, 128, 192, 256, 384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cache_rotation_has_independent_orthogonal_and_native_rounding_witness(width, dtype):
    import math

    from cubic_serve.ops.eager.rotation import rotate_cache_reference

    generator = torch.Generator().manual_seed(7081 + width)
    # Transpose deliberately exercises a noncontiguous operand view.
    values = torch.randn((3, 2, width), generator=generator).to(dtype).transpose(0, 1)
    frozen = values.clone()
    block = width & -width
    transform = torch.tensor([
        [(-1.0) ** ((i & j).bit_count()) for j in range(block)]
        for i in range(block)
    ], dtype=torch.float64) / math.sqrt(block)
    dense = torch.block_diag(*([transform] * (width // block)))
    assert (dense @ dense - torch.eye(width, dtype=torch.float64)).abs().max() < 5e-15
    with ReferenceExecutionContext():
        actual = rotate_cache_reference(values)
        wide = rotate_cache_reference(values, output_dtype=torch.float32)
    assert torch.equal(values, frozen)
    assert actual.dtype is dtype
    assert torch.equal(actual, wide.to(dtype))
    ideal = values.double() @ dense
    steps = block.bit_length()
    gamma = steps * 2**-24 / (1 - steps * 2**-24)
    alpha = 1 / math.sqrt(block)
    rounded_alpha = float(torch.tensor(alpha, dtype=torch.float32))
    magnitude = values.double().reshape(2, 3, -1, block).abs().sum(-1, keepdim=True)
    magnitude = magnitude.expand(2, 3, -1, block).reshape_as(values)
    bound = magnitude * (abs(rounded_alpha) * gamma + abs(rounded_alpha - alpha))
    bound += (actual.double() - wide.double()).abs()
    assert torch.all((actual.double() - ideal).abs() <= bound + 1e-13)
    corrupted = actual.double().clone()
    corrupted[..., 0] += 1
    assert not torch.all((corrupted - ideal).abs() <= bound + 1e-13)


def test_cache_rotation_requires_reference_context_and_inverse_rounds_only_at_end():
    from cubic_serve.ops.eager.rotation import rotate_cache_reference

    values = torch.randn((3, 256), generator=torch.Generator().manual_seed(419)) * 0.123
    with pytest.raises(RuntimeError, match="reference"):
        rotate_cache_reference(values)
    with ReferenceExecutionContext():
        accumulator = rotate_cache_reference(values)
        native = rotate_cache_reference(values, output_dtype=torch.bfloat16)
        assert accumulator.dtype is torch.float32
        assert torch.equal(native, accumulator.bfloat16())
        prematurely_rounded = rotate_cache_reference(values.bfloat16())
        assert not torch.equal(native, prematurely_rounded)
        with pytest.raises(ValueError, match="finite"):
            rotate_cache_reference(torch.tensor([float("nan")]))


@pytest.mark.parametrize("dtype", [CacheDType.FP8, CacheDType.INT8,
                                  CacheDType.INT6, CacheDType.INT4])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("rotation", [False, True])
def test_batched_reference_attention_is_invariant_to_review_width_and_future_scale_outliers(
    dtype, sparse, rotation,
):
    generator = torch.Generator().manual_seed(343)
    width = 16 if rotation else 7
    key = torch.randn((2, 2, 9, width), generator=generator).bfloat16()
    value = torch.randn((2, 2, 9, width), generator=generator).bfloat16()
    query = torch.randn((2, 4, 9, width), generator=generator).bfloat16()
    ordinals = torch.arange(9, dtype=torch.int64)[None].expand(2, -1)
    storage = KVCacheStorageSpec(
        CacheStorageSpec(dtype, CacheGroupAxis.TOKENS, 4),
        CacheStorageSpec(dtype, CacheGroupAxis.CHANNELS, 4),
        enable_rotation=rotation,
    )
    # Explicit indices select only current and preceding token, with -1 padding.
    indices = torch.stack((torch.arange(9), torch.arange(9) - 1), dim=-1)
    selected = indices[None].expand(2, -1, -1).int() if sparse else None
    with ReferenceExecutionContext():
        full = cache_attention_reference(
            query, key, value, storage=storage, query_ordinals=ordinals,
            key_ordinals=ordinals, selected_indices=selected,
        )
        single = []
        for position in range(9):
            single.append(cache_attention_reference(
                query[:, :, position:position + 1], key[:, :, :position + 1],
                value[:, :, :position + 1], storage=storage,
                query_ordinals=ordinals[:, position:position + 1],
                key_ordinals=ordinals[:, :position + 1],
                selected_indices=None if selected is None else selected[:, position:position + 1],
            ))
        assert torch.equal(full, torch.cat(single, dim=2))
        changed_key, changed_value = key.clone(), value.clone()
        changed_key[:, :, 3] = 20000
        changed_value[:, :, 3] = -20000
        changed = cache_attention_reference(
            query, changed_key, changed_value, storage=storage,
            query_ordinals=ordinals, key_ordinals=ordinals, selected_indices=selected,
        )
        assert torch.equal(changed[:, :, :3], full[:, :, :3])
        assert not torch.equal(changed[:, :, 3], full[:, :, 3])


@pytest.mark.parametrize("fmt", [CacheDType.FP8, CacheDType.INT8,
                                CacheDType.INT6, CacheDType.INT4])
@pytest.mark.parametrize("rotated_sides", ["k", "v", "kv"])
@pytest.mark.parametrize("native", [torch.bfloat16, torch.float16])
def test_rotated_cache_review_and_partial_commit_apply_transform_exactly_once(
    fmt, rotated_sides, native,
):
    from cubic_serve.ops.eager.rotation import rotate_cache_reference

    native_format = CacheDType.BF16 if native is torch.bfloat16 else CacheDType.FP16
    storage = KVCacheStorageSpec(
        CacheStorageSpec(fmt if "k" in rotated_sides else native_format,
                         CacheGroupAxis.TOKENS, 4),
        CacheStorageSpec(fmt if "v" in rotated_sides else native_format,
                         CacheGroupAxis.CHANNELS, 4), enable_rotation=True,
    )
    generator = torch.Generator().manual_seed(871)
    raw = [torch.randn((9, 2, 16), generator=generator).to(native) for _ in range(2)]
    frozen = [x.clone() for x in raw]
    with ReferenceExecutionContext():
        empty = ReferenceKVCache.empty(storage, heads=2, head_dim=16, dtype=native)
        full = empty.review(*raw)
        encoded_inputs = [rotate_cache_reference(x) if side in rotated_sides else x
                          for x, side in zip(raw, ("k", "v"))]
        plain = ReferenceKVCache.empty(replace(storage, enable_rotation=False),
                                       heads=2, head_dim=16, dtype=native)
        independently_transformed = plain.review(*encoded_inputs)
        single = empty
        for i in range(9):
            step = single.review(*(x[i:i + 1] for x in raw))
            for a, b, c in zip(step.visible(1), full.visible(i + 1),
                               independently_transformed.visible(i + 1)):
                assert torch.equal(a, b) and torch.equal(b, c)
            single = step.commit(1)
            assert single.enable_rotation
        parent = full.commit(3)
        protected = [s.materialize().clone() for s in (parent.key, parent.value)]
        pending = parent.review(*(x[3:9] for x in raw))
        for accepted in (0, 1, 3, 6):
            child, expected = pending.commit(accepted), full.commit(3 + accepted)
            assert child.enable_rotation
            for a, b in ((child.key, expected.key), (child.value, expected.value)):
                assert torch.equal(a.materialize(), b.materialize())
            for snapshot, side in zip(protected, (parent.key, parent.value)):
                assert torch.equal(snapshot, side.materialize())
    assert all(torch.equal(x, y) for x, y in zip(raw, frozen))


@pytest.mark.parametrize("sparse", [False, True])
def test_rotated_attention_inverse_observes_fp32_output_before_native_rounding(sparse):
    from cubic_serve.ops.eager.attention import full_attention
    from cubic_serve.ops.eager.rotation import rotate_cache_reference
    from cubic_serve.ops.eager.sparse_indexer import attend_selected
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    generator = torch.Generator().manual_seed(7721)
    query, key, value = [torch.randn((1, 1, 3, 32), generator=generator).bfloat16()
                         for _ in range(3)]
    frozen = [x.clone() for x in (query, key, value)]
    storage = KVCacheStorageSpec(
        CacheStorageSpec(CacheDType.BF16, CacheGroupAxis.TOKENS, 4),
        CacheStorageSpec(CacheDType.INT6, CacheGroupAxis.CHANNELS, 4), True,
    )
    ordinals = torch.arange(3, dtype=torch.int64)[None]
    indices = torch.tensor([[[0, -1, -1], [0, 1, -1], [0, 1, 2]]], dtype=torch.int32)
    with ReferenceExecutionContext(), operator_strategy_scope(
        OperatorConfig(attention_dtype="auto", softmax_dtype="fp32"), device="", model="",
    ):
        actual = cache_attention_reference(
            query, key, value, storage=storage, query_ordinals=ordinals,
            key_ordinals=ordinals, selected_indices=indices if sparse else None,
        )
        stream = ReferenceKVCache.empty(storage, heads=1, head_dim=32)
        views = stream.review(key[0].transpose(0, 1), value[0].transpose(0, 1)).visible(3)
        visible = [x.transpose(0, 1)[None] for x in views]
        if sparse:
            core = attend_selected(query[:, :, -1:], *visible, indices[:, -1:],
                                   output_dtype=torch.float32)
        else:
            core = full_attention(query[:, :, -1:], *visible,
                                  query_ordinals=ordinals[:, -1:], key_ordinals=ordinals,
                                  output_dtype=torch.float32)
        expected = rotate_cache_reference(core, output_dtype=query.dtype)
        premature = rotate_cache_reference(core.to(query.dtype))
        assert core.dtype is torch.float32
        assert torch.equal(actual[:, :, -1:], expected)
        assert not torch.equal(expected, premature)
    assert all(torch.equal(x, y) for x, y in zip((query, key, value), frozen))


def test_reference_attention_rejects_noncausal_selected_ids_and_incomplete_history():
    values = torch.ones((1, 1, 4, 7), dtype=torch.float16)
    ordinals = torch.arange(4, dtype=torch.int64)[None]
    with ReferenceExecutionContext():
        with pytest.raises(ValueError, match="complete native"):
            cache_attention_reference(values, values, values, storage=KVCacheStorageSpec(),
                                      query_ordinals=ordinals, key_ordinals=ordinals + 1)
        with pytest.raises(ValueError, match="causal boundary"):
            cache_attention_reference(
                values, values, values, storage=KVCacheStorageSpec(),
                query_ordinals=ordinals, key_ordinals=ordinals,
                selected_indices=torch.full((1, 4, 1), 3, dtype=torch.int32),
            )


@pytest.mark.parametrize("dtype", [CacheDType.FP8, CacheDType.INT8,
                                  CacheDType.INT6, CacheDType.INT4])
@pytest.mark.parametrize("axis", list(CacheGroupAxis))
def test_reference_page_image_keeps_unsealed_and_unwritten_groups_canonical(dtype, axis):
    grouped = GroupedCacheLayout(CacheStorageSpec(dtype, axis, 4), 8, 2, 7)
    values = torch.randn((8, 2, 7), generator=torch.Generator().manual_seed(444)).half()
    with ReferenceExecutionContext():
        complete = encode_cache_reference(values, grouped)
        for count in (0, 1, 3, 4, 7, 8):
            page = cache_page_reference(values[:count], grouped)
            if axis is CacheGroupAxis.TOKENS:
                code_groups = count // 4
                mask = torch.arange(grouped.group_count) % grouped.group_shape[-1] < code_groups
            else:
                mask = (torch.arange(grouped.group_count)
                        < count * grouped.heads * grouped.group_shape[-1])
            assert torch.equal(page.payload[mask], complete.payload[mask])
            assert torch.equal(page.scales[mask], complete.scales[mask])
            assert not bool(page.payload[~mask].any())
            assert not bool(page.scales[~mask].any())
@pytest.mark.parametrize(
    "dtype", [CacheDType.FP8, CacheDType.INT8, CacheDType.INT6, CacheDType.INT4]
)
def test_current_queries_use_native_span_and_rejected_draft_cannot_change_history(dtype):
    storage = KVCacheStorageSpec(
        CacheStorageSpec(dtype, CacheGroupAxis.TOKENS, 4),
        CacheStorageSpec(dtype, CacheGroupAxis.CHANNELS, 4),
    )
    values = torch.randn((13, 2, 7), generator=torch.Generator().manual_seed(331)).bfloat16()
    with ReferenceExecutionContext():
        initial = ReferenceKVCache.empty(storage, heads=2, head_dim=7)
        # Prefill consumes the native incoming span before any group is sealed.
        prefill = initial.review(values[:7], -values[:7])
        assert torch.equal(prefill.visible(3)[0], values[:3])
        committed = prefill.commit(7)
        assert committed.key.tokens == committed.value.tokens == 7
        assert committed.key.tail.shape[0] == 3
        assert committed.value.tail.shape[0] == 0
        sealed_codes = committed.key.sealed[0].payload.clone()
        sealed_scales = committed.key.sealed[0].scales.clone()
        # One accepted token closes the tail. The rejected suffix contains an
        # extreme outlier that must never set the scale of this closed group.
        speculative = values[7:11].clone()
        speculative[1:] = 20000
        review = committed.review(speculative, -speculative)
        normal = committed.review(values[7:8], -values[7:8])
        for candidate, expected in zip(review.visible(1), normal.visible(1), strict=True):
            assert torch.equal(candidate, expected)
        accepted = review.commit(1)
        canonical = normal.commit(1)
        assert torch.equal(accepted.key.materialize(), canonical.key.materialize())
        assert torch.equal(accepted.value.materialize(), canonical.value.materialize())
        assert torch.equal(committed.key.sealed[0].payload, sealed_codes)
        assert torch.equal(committed.key.sealed[0].scales, sealed_scales)
        assert review.commit(0).key is committed.key
        assert committed.key.tokens == 7


def test_sealed_column_groups_are_independent_of_append_chunking_with_identical_commit_policy():
    storage = KVCacheStorageSpec(
        CacheStorageSpec(CacheDType.INT6, CacheGroupAxis.TOKENS, 4),
        CacheStorageSpec(CacheDType.INT4, CacheGroupAxis.CHANNELS, 4),
    )
    values = torch.randn((17, 3, 9), generator=torch.Generator().manual_seed(90)).half()
    with ReferenceExecutionContext():
        empty = ReferenceKVCache.empty(storage, heads=3, head_dim=9, dtype=torch.float16)
        bulk = empty.review(values, -values).commit(17)
        scalar = empty
        for token in values.split(1):
            scalar = scalar.review(token, -token).commit(1)
        assert torch.equal(bulk.key.materialize(), scalar.key.materialize())
        assert torch.equal(bulk.value.materialize(), scalar.value.materialize())
        assert bulk.key.tail.shape[0] == scalar.key.tail.shape[0] == 1
        # Reference materialization is an independent copy, never mutable storage.
        materialized = bulk.key.materialize()
        materialized.fill_(100)
        assert not torch.equal(materialized, bulk.key.materialize())


def test_each_query_observes_the_same_rounded_history_in_prefill_review_and_m1():
    storage = KVCacheStorageSpec(
        CacheStorageSpec(CacheDType.FP8, CacheGroupAxis.TOKENS, 4),
        CacheStorageSpec(CacheDType.FP8, CacheGroupAxis.CHANNELS, 4),
    )
    values = torch.randn((9, 2, 7), generator=torch.Generator().manual_seed(773)).bfloat16()
    with ReferenceExecutionContext():
        initial = ReferenceKVCache.empty(storage, heads=2, head_dim=7)
        bulk_review = initial.review(values, -values)
        scalar = initial
        for count in range(1, 10):
            step = scalar.review(values[count - 1 : count], -values[count - 1 : count])
            for bulk, single in zip(bulk_review.visible(count), step.visible(1), strict=True):
                assert torch.equal(bulk, single), count
            scalar = step.commit(1)


@pytest.mark.parametrize("dtype", [CacheDType.INT8, CacheDType.INT6, CacheDType.INT4])
@pytest.mark.parametrize("axis", list(CacheGroupAxis))
def test_integer_quantization_satisfies_per_group_rne_error_bound(dtype, axis):
    values = torch.randn((9, 3, 11), generator=torch.Generator().manual_seed(7731))
    values[:, 0, 0] *= 50
    layout = GroupedCacheLayout(CacheStorageSpec(dtype, axis, 7), *values.shape)
    with ReferenceExecutionContext():
        encoded = encode_cache_reference(values, layout)
        restored = decode_cache_reference(encoded, output_dtype=torch.float32)
    for t in range(9):
        for h in range(3):
            for d in range(11):
                group, _ = layout.group_location(t, h, d)
                scale = encoded.scales[group].item()
                bound = scale * (0.5 + 4 * torch.finfo(torch.float32).eps * dtype.quant_max)
                assert abs(values[t, h, d].item() - restored[t, h, d].item()) <= bound
    assert encoded.payload.numel() == layout.payload_bytes
    assert encoded.scales.numel() * 4 == layout.scale_bytes


@pytest.mark.parametrize(
    "dtype", [CacheDType.FP8, CacheDType.INT8, CacheDType.INT6, CacheDType.INT4]
)
def test_transposed_grouping_reuses_the_same_format_without_head_mixing(dtype):
    values = torch.randn((9, 3, 11), generator=torch.Generator().manual_seed(818))
    col = GroupedCacheLayout(CacheStorageSpec(dtype, CacheGroupAxis.TOKENS, 7), *values.shape)
    transposed = values.permute(2, 1, 0).contiguous()
    row = GroupedCacheLayout(CacheStorageSpec(dtype, CacheGroupAxis.CHANNELS, 7), *transposed.shape)
    with ReferenceExecutionContext():
        c = encode_cache_reference(values, col)
        r = encode_cache_reference(transposed, row)
        c_dec = decode_cache_reference(c, output_dtype=torch.float32)
        r_dec = decode_cache_reference(r, output_dtype=torch.float32)
    c_codes = c.payload.reshape(*col.group_shape, -1).permute(1, 0, 2, 3)
    assert torch.equal(c_codes.reshape_as(r.payload), r.payload)
    assert torch.equal(
        c.scales.reshape(col.group_shape).permute(1, 0, 2).reshape_as(r.scales), r.scales
    )
    assert torch.equal(c_dec.permute(2, 1, 0), r_dec)


@pytest.mark.parametrize("dtype", [CacheDType.INT8, CacheDType.INT6, CacheDType.INT4])
def test_signed_integer_codes_match_independent_little_endian_bitstream(dtype):
    top = int(dtype.quant_max)
    codes = [-top, -3, -1, 0, 1, 3, top]
    values = torch.tensor(codes, dtype=torch.float32).reshape(1, 1, -1)
    layout = GroupedCacheLayout(CacheStorageSpec(dtype, CacheGroupAxis.CHANNELS, 7), 1, 1, 7)
    with ReferenceExecutionContext():
        encoded = encode_cache_reference(values, layout)
        restored = decode_cache_reference(encoded, output_dtype=torch.float32)
    stream = sum(
        (value & ((1 << dtype.bits) - 1)) << (i * dtype.bits) for i, value in enumerate(codes)
    )
    assert bytes(encoded.payload.flatten().tolist()) == stream.to_bytes(
        layout.payload_bytes, "little"
    )
    assert encoded.scales.item() == 1
    assert torch.equal(restored, values)


def test_fp8_known_codes_are_e4m3fn_and_input_storage_is_not_aliased():
    values = torch.tensor([0.0, 1.0, -1.0, 448.0]).reshape(4, 1, 1)
    layout = GroupedCacheLayout(CacheStorageSpec(CacheDType.FP8, CacheGroupAxis.TOKENS, 4), 4, 1, 1)
    with ReferenceExecutionContext():
        encoded = encode_cache_reference(values, layout)
        restored = decode_cache_reference(encoded, output_dtype=torch.bfloat16)
    assert encoded.payload.tolist() == [[0x00, 0x38, 0xB8, 0x7E]]
    assert torch.equal(restored.float(), values)
    values.zero_()
    assert restored[-1].item() == 448


@pytest.mark.parametrize(
    "dtype", [CacheDType.FP8, CacheDType.INT8, CacheDType.INT6, CacheDType.INT4]
)
def test_zero_groups_and_padding_have_finite_canonical_scales(dtype):
    layout = GroupedCacheLayout(CacheStorageSpec(dtype, CacheGroupAxis.TOKENS, 7), 3, 2, 5)
    with ReferenceExecutionContext():
        encoded = encode_cache_reference(torch.zeros(layout.logical_shape), layout)
        assert torch.equal(encoded.scales, torch.ones(layout.group_count))
        assert not decode_cache_reference(encoded, output_dtype=torch.float32).any()
        with pytest.raises(ValueError, match="scales"):
            decode_cache_reference(
                replace(encoded, scales=encoded.scales * 0), output_dtype=torch.float32
            )


def test_reference_is_not_an_implicit_performance_fallback():
    layout = GroupedCacheLayout(
        CacheStorageSpec(CacheDType.INT4, CacheGroupAxis.TOKENS, 4), 4, 1, 1
    )
    with pytest.raises(RuntimeError, match="ReferenceExecutionContext"):
        encode_cache_reference(torch.zeros(layout.logical_shape), layout)


@pytest.mark.parametrize(
    "dtype", [CacheDType.FP8, CacheDType.INT8, CacheDType.INT6, CacheDType.INT4]
)
@pytest.mark.parametrize(
    "axes",
    [
        (CacheGroupAxis.TOKENS, CacheGroupAxis.CHANNELS),
        (CacheGroupAxis.CHANNELS, CacheGroupAxis.TOKENS),
    ],
)
@pytest.mark.parametrize("sides", ["k", "v", "kv"])
@pytest.mark.parametrize("rotation", [False, True])
def test_cache_read_witness_preserves_causality_and_detects_corrupt_boundaries(
    dtype, axes, sides, rotation,
):
    from cubic_serve.ops.eager.rotation import rotate_cache_reference

    generator = torch.Generator().manual_seed(541)
    dim = 16 if rotation else 7
    history = [torch.randn((20, 2, dim), generator=generator).bfloat16() for _ in range(2)]
    history[0][-1] *= 100
    history[1][-1] *= -100
    layouts = [
        GroupedCacheLayout(
            CacheStorageSpec(dtype if side in sides else CacheDType.BF16, axis, 4), 8, 2, dim
        )
        for side, axis in zip(("k", "v"), axes)
    ]
    order = torch.tensor([2, 0, 1])
    payloads = []
    scales = []
    tails = []
    incoming = []
    producers = []
    row = 5
    with ReferenceExecutionContext():
        for native, layout in zip(history, layouts):
            raw = torch.zeros((16, 2, dim), dtype=torch.bfloat16)
            raw[4:9] = native[15:20]
            producers.append(raw)
            if rotation and layout.spec.dtype.quantized:
                native = rotate_cache_reference(native)
            pages = [cache_page_reference(native[i * 8 : (i + 1) * 8], layout) for i in range(3)]
            data = torch.stack([p.payload for p in pages])
            physical = torch.empty_like(data)
            physical[order] = data
            payloads.append(physical[:, None])
            if layout.spec.dtype.quantized:
                scale = torch.stack([p.scales for p in pages])
                physical = torch.empty_like(scale)
                physical[order] = scale
                scales.append(physical[:, None])
            else:
                scales.append(None)
            if layout.spec.dtype.quantized and layout.spec.axis is CacheGroupAxis.TOKENS:
                tail = torch.zeros((6, 1, 4, 2, dim), dtype=torch.bfloat16)
                tail[3, 0, :3] = native[12:15]
                tails.append(tail)
            else:
                tails.append(None)
            values = torch.zeros((16, 2, dim), dtype=torch.bfloat16)
            values[4:9] = native[15:20]
            incoming.append(values)
        starts = torch.full((16,), 15, dtype=torch.int64)
        roots = torch.arange(16, dtype=torch.int32)
        roots[4:9] = 4
        sources = torch.full((16,), 3, dtype=torch.int64)
        encoding = SimpleNamespace(
            key=layouts[0], value=layouts[1], key_scales=scales[0], value_scales=scales[1],
            enable_rotation=rotation,
        )
        window = SimpleNamespace(
            key=incoming[0],
            value=incoming[1],
            start_positions=starts,
            start_rows=roots,
            sources=sources,
            key_tail=tails[0],
            value_tail=tails[1],
            producer_key=producers[0] if rotation and "k" in sides else None,
            producer_value=producers[1] if rotation and "v" in sides else None,
        )
        options = dict(
            key_pages=payloads[0],
            value_pages=payloads[1],
            encoding=encoding,
            window=window,
            table=order[None].expand(16, -1),
            layer=0,
            row=row,
        )
        actual = witness_cache_read(history[0][:17], history[1][:17], **options)
        stream = ReferenceKVCache.empty(
            KVCacheStorageSpec(*(x.spec for x in layouts), enable_rotation=rotation),
            heads=2, head_dim=dim,
        )
        expected = stream.review(history[0][:17], history[1][:17]).visible(17)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        # A corrupted earlier code is consumed even when a future group is native.
        quant_side = 0 if "k" in sides else 1
        saved = payloads[quant_side][int(order[0]), 0, 0, 0].clone()
        payloads[quant_side][int(order[0]), 0, 0, 0] ^= 1
        with pytest.raises(AssertionError, match="cache encoding"):
            witness_cache_read(history[0][:17], history[1][:17], **options)
        payloads[quant_side][int(order[0]), 0, 0, 0] = saved
        saved = scales[quant_side][int(order[0]), 0, 0].clone()
        scales[quant_side][int(order[0]), 0, 0] *= 2
        with pytest.raises(AssertionError, match="cache scale"):
            witness_cache_read(history[0][:17], history[1][:17], **options)
        scales[quant_side][int(order[0]), 0, 0] = saved
        incoming[0][row, 0, 0] += 1
        with pytest.raises(AssertionError, match="native cache producer"):
            witness_cache_read(history[0][:17], history[1][:17], **options)


def test_query_rotation_witness_requires_coherent_raw_source_and_native_output():
    from cubic_serve.ops.eager.kv_cache import witness_cache_query
    from cubic_serve.ops.eager.rotation import rotate_cache_reference

    native = torch.randn((2, 3, 16), generator=torch.Generator().manual_seed(915)).bfloat16()
    encoding = SimpleNamespace(enable_rotation=True, key=SimpleNamespace(
        spec=CacheStorageSpec(CacheDType.INT4, CacheGroupAxis.TOKENS),
    ))
    window = SimpleNamespace(producer_query=native.clone())
    with ReferenceExecutionContext():
        actual = rotate_cache_reference(native)
        witness_cache_query(native[:, :, None], actual, encoding=encoding, window=window)
        broken = actual.clone()
        broken.view(torch.int16)[0, 0, 0] ^= 1
        with pytest.raises(AssertionError, match="paired native"):
            witness_cache_query(native, broken, encoding=encoding, window=window)
        window.producer_query.view(torch.int16)[0, 0, 0] ^= 1
        with pytest.raises(AssertionError, match="rotation producer"):
            witness_cache_query(native, actual, encoding=encoding, window=window)
        window.producer_query = None
        with pytest.raises(AssertionError, match="lacks its native producer"):
            witness_cache_query(native, actual, encoding=encoding, window=window)
