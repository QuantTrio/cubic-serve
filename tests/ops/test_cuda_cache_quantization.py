"""CUDA encoding/load identity at the common grouping boundary, not serving TPS."""

import os

import pytest
import torch
import triton
import triton.language as tl

from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
from cubic_serve.ops.cuda.cache_quantization import (
    CacheGroupWritePrimitive,
    PagedKVEncoding,
    PagedKVReadWindow,
    _decode_e4m3fn,
    _fp8_codes,
    bind_cache_window_metadata,
    bind_cache_append,
    cache_group_indices,
    load_cache_group,
)
from cubic_serve.ops.eager.kv_cache import decode_cache_reference, encode_cache_reference
from cubic_serve.runtime.cache.quantization import (
    CacheDType,
    CacheGroupAxis,
    CacheStorageSpec,
    GroupedCacheLayout,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)


@pytest.mark.parametrize("owner_stride", [1, 3, 17])
@pytest.mark.parametrize("owner_dtype", [torch.int32, torch.int64])
def test_segmented_owner_normalization_preserves_continuations_and_private_padding(
    owner_stride, owner_dtype
):
    # Layer-strided state indices and owner indices are different address spaces.
    # Normalization depends only on the actual state-map stride, never model names.
    segments = torch.tensor([3, 0, 0, 0, 2, 0, 0, 1], device="cuda", dtype=torch.int32)
    activity = torch.tensor([1, 1, 1, 0, 1, 1, 0, 1], device="cuda", dtype=torch.int32)
    positions = torch.tensor([41, 42, 43, 0, 63, 64, 0, 101],
                             device="cuda", dtype=torch.int64)
    owner = torch.tensor([2, 0, 0, 13, 5, 0, 19, 7], device="cuda", dtype=owner_dtype)
    source = owner * owner_stride
    destination = torch.arange(8, device="cuda", dtype=owner_dtype) * owner_stride
    starts = torch.empty(8, device="cuda", dtype=torch.int64)
    roots = torch.empty(8, device="cuda", dtype=torch.int32)
    sources, destinations = torch.empty_like(starts), torch.empty_like(starts)
    binding = bind_cache_window_metadata(
        segments, activity, positions, source, starts, roots, sources,
        owner_stride=owner_stride, destination_slots=destination, destinations=destinations,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.launch()
    graph.replay()
    assert starts.cpu().tolist() == [41, 41, 41, 0, 63, 63, 0, 101]
    assert roots.cpu().tolist() == [0, 0, 0, 3, 4, 4, 6, 7]
    assert sources.cpu().tolist() == [2, 2, 2, 13, 5, 5, 19, 7]
    assert destinations.cpu().tolist() == list(range(8))
    # Replay must consume current leased values rather than constructor-time copies.
    source.add_(owner_stride)
    destination.add_(2 * owner_stride)
    graph.replay()
    assert sources.cpu().tolist() == [3, 3, 3, 14, 6, 6, 20, 8]
    assert destinations.cpu().tolist() == list(range(2, 10))


@pytest.mark.parametrize("owner_stride", [1, 3, 17])
@pytest.mark.parametrize("strided", [False, True, "pooled"])
def test_shared_cache_graph_binding_seals_and_checkpoints_with_layer_strided_owners(owner_stride, strided):
    from cubic_serve.guardrail.quantized_graph.attention_bounds import enclose
    from cubic_serve.ops.cuda.attention.paged_decode import (
        PageCOWCopyPrimitive, PagedGQADecodePrimitive,
    )
    from cubic_serve.ops.cuda.cache_quantization import PagedKVGraphBinding
    from cubic_serve.ops.eager.kv_cache import ReferenceKVCache, cache_page_reference
    from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout
    from cubic_serve.runtime.cache.quantization import KVCacheStorageSpec

    configure_reference_math()
    storage = KVCacheStorageSpec(
        CacheStorageSpec(CacheDType.FP8, CacheGroupAxis.TOKENS, 4),
        CacheStorageSpec(CacheDType.FP8, CacheGroupAxis.CHANNELS, 4),
    )
    layout = KVLayout((2, 9), 8, 1, 13, storage=storage)
    spec = CacheArenaSpec("common", 4, 6, layout.quantization_state_specs,
                          scratch_capacity=4)
    if strided=="pooled":
        from cubic_serve.runtime.cache.shared_pool import SharedCacheStorage
        shared=SharedCacheStorage(((spec,layout),),budget_bytes=8192,state_capacity=7,device="cuda")
        arena=CacheArena(shared.arena_specs[0],layout,"cuda",rank_id=0,shared_storage=shared)
        for index in (1,2,3,4,6):shared.allocator.reserve_at("state",index)
        for index in range(5):shared.allocator.reserve_at("page",index)
    else:
        arena = CacheArena(spec, layout, "cuda", rank_id=0)
    if strided is True:
        from cubic_serve.runtime.cache.shared_pool import SharedCacheField, SharedCacheLayout
        page_tensors = {"key": arena.key, "value": arena.value, "written": arena.written,
                        **arena.page_fields}
        page_layout = SharedCacheLayout(tuple(
            SharedCacheField(name, tensor.dtype, tuple(tensor.shape[1:]))
            for name, tensor in page_tensors.items())
            + (SharedCacheField("guard", torch.uint8, (7,)),))
        page_backing = torch.full((arena.key.shape[0] * page_layout.entry_bytes,), 0x5A,
                                  dtype=torch.uint8, device="cuda")
        page_views = page_layout.views(page_backing, arena.key.shape[0])
        for name, tensor in page_tensors.items():
            page_views[name].copy_(tensor)
        arena.key, arena.value, arena.written = (page_views[name] for name in ("key", "value", "written"))
        arena.page_fields = {name: page_views[name] for name in arena.page_fields}
        state_layout = SharedCacheLayout(tuple(
            SharedCacheField(name, tensor.dtype, tuple(tensor.shape[1:]))
            for name, tensor in arena.states.items())
            + (SharedCacheField("guard", torch.uint8, (7,)),))
        slots = next(iter(arena.states.values())).shape[0]
        state_backing = torch.full((slots * state_layout.entry_bytes,), 0x5A,
                                   dtype=torch.uint8, device="cuda")
        state_views = state_layout.views(state_backing, slots)
        for name, tensor in arena.states.items():
            state_views[name].copy_(tensor)
        arena.states = {name: state_views[name] for name in arena.states}
    generator = torch.Generator().manual_seed(1041)
    key, value = (torch.randn((10, 1, 13), generator=generator).bfloat16() for _ in range(2))
    with ReferenceExecutionContext():
        prefix = ReferenceKVCache.empty(storage, heads=1, head_dim=13)
        prefix = prefix.review(key[:7], value[:7]).commit(7)
        review = prefix.review(key[7:], value[7:])
        for side, native, grouped in (("key", key, layout.key_encoding),
                                      ("value", value, layout.value_encoding)):
            page = cache_page_reference(native[:7], grouped)
            getattr(arena, side)[0] = page.payload.cuda()[None]
            arena.page_fields[f"kv_{side}_scales"][0] = page.scales.cuda()[None]
    arena.written[0, :, :7] = True
    arena.states["kv_key_tail"][1, :, :3] = key[4:7].cuda()[None]
    protected = tuple(t.clone() for t in (arena.key[:2], arena.value[:2],
                                         *[v[:2] for v in arena.page_fields.values()],
                                         arena.states["kv_key_tail"][[1, 6]]))
    segments = torch.tensor([3, 0, 0, 0], device="cuda", dtype=torch.int32)
    activity = torch.tensor([1, 1, 1, 0], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([8, 9, 10, 1], device="cuda", dtype=torch.int32)
    owners = torch.tensor([[1, 0, 0, 6], [2, 3, 4, 6]],
                          device="cuda", dtype=torch.int32) * owner_stride
    metadata = PagedKVGraphBinding(
        arena, positions=lengths, positions_are_lengths=True, segments=segments,
        activity=activity, source_slots=owners[0], destination_slots=owners[1],
        owner_stride=owner_stride,
    )
    incoming = []
    for native in (key, value):
        current = torch.zeros((4, 1, 13), device="cuda", dtype=torch.bfloat16)
        current[:3] = native[7:].cuda()
        incoming.append(current)
    pages = torch.tensor([2, 3, 3, 4], device="cuda", dtype=torch.int32)
    source_pages = torch.tensor([0, 1, 3, 4], device="cuda", dtype=torch.int32)
    copy = PageCOWCopyPrimitive(
        arena.key, arena.value, arena.written, source_pages, pages,
        cache_encoding=metadata.encoding, page_fields=tuple(arena.page_fields.values()),
    )
    window, prepare, checkpoint = metadata.bind_layer(*incoming, layer=1, pages=pages)
    query = torch.randn((4, 3, 13), generator=generator).bfloat16().cuda()
    output = torch.empty_like(query)
    table = torch.tensor([[2, 3], [2, 3], [2, 3], [4, 4]],
                         device="cuda", dtype=torch.int32)
    attention = PagedGQADecodePrimitive(
        query, arena.key, arena.value, table, lengths, output, layer=1,
        cache_encoding=metadata.encoding, cache_window=window,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata.prepare_metadata()
        copy.launch(arena.key, arena.value, arena.written, source_pages, pages)
        prepare.launch()
        receipt = attention.launch(query, arena.key, arena.value, table, lengths, output)
        checkpoint.launch()
    attention.validate_receipt(receipt)
    graph.replay()
    for count in range(1, 4):
        with ReferenceExecutionContext():
            visible = review.visible(count)
            accepted = review.commit(count)
        proof = enclose(query[count - 1:count].cpu(), *(v[None] for v in visible),
                        output[count - 1:count].cpu(), scale=13 ** -0.5,
                        splits=attention._grid[2], probability_dtype=torch.bfloat16)
        assert proof["outside_enclosure"] == 0, proof
        expected = torch.zeros((4, 1, 13), dtype=torch.bfloat16)
        expected[:accepted.key.tail.shape[0]] = accepted.key.tail
        assert torch.equal(arena.states["kv_key_tail"][count + 1, 1].cpu(), expected)
    after = (arena.key[:2], arena.value[:2], *[v[:2] for v in arena.page_fields.values()],
             arena.states["kv_key_tail"][[1, 6]])
    assert all(torch.equal(a, b) for a, b in zip(protected, after, strict=True))
    if strided is True:
        assert (page_views["guard"] == 0x5A).all()
        assert (state_views["guard"] == 0x5A).all()


@pytest.mark.parametrize("axis", list(CacheGroupAxis))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_fp8_scales_match_cpu_and_gpu_independent_correctly_rounded_reference(axis, dtype):
    configure_reference_math()
    values = (torch.randn((32, 2, 256), generator=torch.Generator().manual_seed(414)) * 2).to(dtype)
    grouped = GroupedCacheLayout(CacheStorageSpec(CacheDType.FP8, axis, 32), *values.shape)
    with ReferenceExecutionContext():
        cpu = encode_cache_reference(values, grouped)
        gpu = encode_cache_reference(values.cuda(), grouped)
    payload, scales = torch.empty_like(gpu.payload), torch.empty_like(gpu.scales)
    binding = CacheGroupWritePrimitive(values.cuda(), payload, scales, grouped)
    binding.launch()
    assert torch.equal(cpu.scales, gpu.scales.cpu())
    assert torch.equal(cpu.payload, gpu.payload.cpu())
    assert torch.equal(cpu.scales, scales.cpu())
    assert torch.equal(cpu.payload, payload.cpu())


@triton.jit
def _codec_probe(values, codes, restored, ITEMS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(values + i, i < ITEMS, 0)
    code = _fp8_codes(value, 1.0)
    tl.store(codes + i, code, i < ITEMS)
    tl.store(restored + i, _decode_e4m3fn(code), i < ITEMS)


def test_portable_e4m3_codec_all_finite_codes_and_rne_midpoints():
    configure_reference_math()
    all_codes = torch.arange(256, dtype=torch.uint8)
    all_values = all_codes.view(torch.float8_e4m3fn).float()
    finite = all_values[torch.isfinite(all_values)]
    positive = all_values[:127]
    middle = (positive[:-1] + positive[1:]) / 2
    directions = (torch.full_like(middle, -float("inf")), torch.full_like(middle, float("inf")))
    boundary = torch.cat((middle, *(torch.nextafter(middle, d) for d in directions)))
    values = torch.cat((finite, boundary, -boundary)).contiguous()
    x = values.cuda()
    codes = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    restored = torch.empty_like(x)
    _codec_probe[(triton.cdiv(x.numel(), 128),)](x, codes, restored, x.numel(), 128)
    expected = values.to(torch.float8_e4m3fn)
    assert torch.equal(codes.cpu(), expected.view(torch.uint8))
    # Exact FP32 representation includes signed zero, not just numerical equality.
    assert torch.equal(restored.cpu().view(torch.int32), expected.float().view(torch.int32))


@triton.jit
def _read_probe(
    payload,
    scales,
    output,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    COLUMN: tl.constexpr,
    BLOCK: tl.constexpr,
    FP8: tl.constexpr = True,
    BITS: tl.constexpr = 8,
):
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = flat < TOKENS * HEADS * DIM
    token, head, channel = flat // (HEADS * DIM), flat // DIM % HEADS, flat % DIM
    group, member = cache_group_indices(token, head, channel, TOKENS, HEADS, DIM, GROUP, COLUMN)
    restored = load_cache_group(
        payload, scales, group, member, active, GROUP, BITS, FP8, output.dtype.element_ty
    )
    tl.store(output + flat, restored, active)


@pytest.mark.parametrize("axis", list(CacheGroupAxis))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group_size", [7, 32])
@pytest.mark.parametrize("implementation,cache_dtype", [
    ("cuda", CacheDType.FP8), ("triton", CacheDType.FP8),
    ("cuda", CacheDType.INT8), ("cuda", CacheDType.INT6), ("cuda", CacheDType.INT4),
])
def test_cache_capture_replay_matches_independent_cpu_codec(
    axis, dtype, group_size, implementation, cache_dtype
):
    configure_reference_math()
    layout = GroupedCacheLayout(CacheStorageSpec(cache_dtype, axis, group_size), 17, 3, 37)
    source = torch.randn(layout.logical_shape, generator=torch.Generator().manual_seed(951)).to(
        dtype
    )
    source[:, 0, 0] *= 25
    source[:2, 1, :] = 0
    values = source.cuda()
    payload = torch.empty(
        (layout.group_count, layout.packed_group_bytes), dtype=torch.uint8, device="cuda"
    )
    scales = torch.empty(layout.group_count, dtype=torch.float32, device="cuda")
    output = torch.empty_like(values)
    writer = CacheGroupWritePrimitive(
        values, payload, scales, layout, implementation=implementation
    )
    writer.launch()
    args = dict(
        TOKENS=layout.tokens,
        HEADS=layout.heads,
        DIM=layout.head_dim,
        GROUP=group_size,
        COLUMN=axis is CacheGroupAxis.TOKENS,
        BLOCK=128,
        FP8=cache_dtype is CacheDType.FP8,
        BITS=cache_dtype.bits,
    )
    grid = (triton.cdiv(source.numel(), 128),)
    _read_probe[grid](payload, scales, output, **args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        writer.launch()
        _read_probe[grid](payload, scales, output, **args)
    for multiplier in (1.0, -3.0, 0.0):
        current = (source * multiplier).to(dtype)
        values.copy_(current)
        graph.replay()
        with ReferenceExecutionContext():
            reference = encode_cache_reference(current, layout)
            restored = decode_cache_reference(reference, output_dtype=dtype)
        assert torch.equal(payload.cpu(), reference.payload)
        assert torch.equal(scales.cpu(), reference.scales)
        assert torch.equal(output.cpu(), restored)


@pytest.mark.parametrize("quantized_sides", ["k", "v", "kv"])
@pytest.mark.parametrize("precision", ["auto", "fp32"])
@pytest.mark.parametrize("implementation,cache_dtype", [
    ("cuda", CacheDType.FP8), ("triton", CacheDType.FP8),
    ("cuda", CacheDType.INT8), ("cuda", CacheDType.INT6), ("cuda", CacheDType.INT4),
])
@pytest.mark.parametrize("axes", [
    (CacheGroupAxis.TOKENS, CacheGroupAxis.CHANNELS),
    (CacheGroupAxis.CHANNELS, CacheGroupAxis.TOKENS),
])
@pytest.mark.parametrize("geometry", [
    (1, 12, 256, torch.bfloat16, 1, False),
    (2, 26, 37, torch.float16, 4, True),
    (2, 34, 80, torch.bfloat16, 32, False),
])
def test_sealed_encoded_pages_preserve_attention_tree_and_physical_row_mapping(
    quantized_sides,
    precision,
    implementation,
    cache_dtype,
    geometry,
    axes,
    *,
    token_group_size=8,
    strided=False,
    encoded_binding=True,
):
    from cubic_serve.guardrail.quantized_graph import attention_bounds
    from cubic_serve.ops.cuda.attention.paged_decode import (
        PagedGQADecodePrimitive,
        paged_gqa_fingerprint_payload,
    )
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    if implementation == "cuda" and precision != "auto":
        pytest.skip("the native MMA candidate requires native QKV operands")
    configure_reference_math()
    generator = torch.Generator().manual_seed(501)
    pages, layers, block = 4, 2, 16
    kv_heads, query_heads, dim, dtype, splits, selected = geometry
    key_format, value_format = (
        cache_dtype if isinstance(cache_dtype, tuple) else (cache_dtype, cache_dtype)
    )
    plain = CacheDType.BF16 if dtype is torch.bfloat16 else CacheDType.FP16
    source_k = torch.randn((pages, layers, block, kv_heads, dim), generator=generator).to(dtype)
    source_v = torch.randn(source_k.shape, generator=generator).to(dtype)
    specs = [
        CacheStorageSpec(
            key_format if "k" in quantized_sides else plain, axes[0],
            token_group_size if axes[0] is CacheGroupAxis.TOKENS else 32,
        ),
        CacheStorageSpec(
            value_format if "v" in quantized_sides else plain,
            axes[1],
            token_group_size if axes[1] is CacheGroupAxis.TOKENS else 32,
        ),
    ]
    layouts = [GroupedCacheLayout(spec, block, kv_heads, dim) for spec in specs]
    data, scales, decoded = [], [], []
    with ReferenceExecutionContext():
        for source, layout in zip((source_k, source_v), layouts, strict=True):
            encoded = [
                [encode_cache_reference(source[p, layer], layout) for layer in range(layers)]
                for p in range(pages)
            ]
            data.append(
                torch.stack([torch.stack([x.payload for x in row]) for row in encoded]).cuda()
            )
            scales.append(
                None
                if not layout.spec.dtype.quantized
                else torch.stack([torch.stack([x.scales for x in row]) for row in encoded]).cuda()
            )
            decoded.append(
                torch.stack(
                    [
                        torch.stack([decode_cache_reference(x, output_dtype=dtype) for x in row])
                        for row in encoded
                    ]
                )
            )
    if strided:
        from cubic_serve.runtime.cache.shared_pool import SharedCacheField, SharedCacheLayout
        tensors = {"key": data[0], "value": data[1],
                   **{f"scale{i}": tensor for i, tensor in enumerate(scales) if tensor is not None}}
        layout = SharedCacheLayout(tuple(SharedCacheField(name, tensor.dtype, tuple(tensor.shape[1:]))
                                         for name, tensor in tensors.items())
                                   + (SharedCacheField("guard", torch.int64, (7,)),))
        backing = torch.full((pages * layout.entry_bytes,), 0x5A, dtype=torch.uint8, device="cuda")
        fields = layout.views(backing, pages)
        for name, tensor in tensors.items():
            fields[name].copy_(tensor)
        data = [fields["key"], fields["value"]]
        scales = [fields.get(f"scale{i}") for i in range(2)]
        protected_backing = backing.clone()
    encoding = PagedKVEncoding(*layouts, *scales) if encoded_binding else None
    corpus = torch.randn((1, query_heads, dim), generator=generator).to(dtype)
    order = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device="cuda")
    protected = [x.clone() for x in (*data, *scales) if x is not None]
    with operator_strategy_scope(
        OperatorConfig(attention_dtype=precision, softmax_dtype="fp32"), device="", model=""
    ):
        previous = None
        for capacity in (1, 8):
            row = capacity - 1
            query = torch.full(
                (capacity, query_heads, dim), float("nan"), dtype=dtype, device="cuda"
            )
            query[row : row + 1].copy_(corpus)
            output = torch.empty_like(query)
            table = order[None].expand(capacity, -1).contiguous()
            counts = torch.ones(capacity, dtype=torch.int32, device="cuda")
            if selected == "unaligned":
                positions = torch.cat((torch.arange(5, 13), torch.arange(21, 29),
                                       torch.arange(37, 39)))
            elif selected == "page-boundary":
                positions = torch.cat((torch.arange(13, 21), torch.arange(29, 37),
                                       torch.arange(45, 47)))
            else:
                positions = torch.arange(0, 39, 2) if selected else torch.arange(39)
            counts[row] = positions.numel()
            indices = (positions.int().cuda()[None].expand(capacity, -1).contiguous()
                       if selected else None)
            scratch = {} if splits == 1 else dict(
                partial=torch.empty((capacity, query_heads, splits, dim), device="cuda"),
                statistics=torch.empty((capacity, query_heads, splits, 2), device="cuda"),
                arrivals=torch.zeros((capacity, query_heads), dtype=torch.int32, device="cuda"),
            )
            primitive = PagedGQADecodePrimitive(
                query, *data, table, counts, output, layer=1, cache_encoding=encoding if encoded_binding else None,
                selected_indices=indices, implementation=implementation, **scratch,
            )
            assert primitive.project_artifact().backend == (
                "cuda-driver" if implementation == "cuda" else "triton"
            )
            fingerprint = paged_gqa_fingerprint_payload(
                query, data[0], table, layer=1, scale=dim**-0.5,
                cache_encoding=encoding if encoded_binding else None, implementation=implementation,
            )
            if encoded_binding:
                assert fingerprint["launch"]["implementation"] == (
                    "native-mma16816" if implementation == "cuda" else "triton"
                )
            if implementation == "cuda":
                assert set(fingerprint["native_sources"]) == {
                    "paged_decode_native.py", "paged_decode_native.cu",
                    "mma.cuh", "activation_native.cuh",
                }
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                receipt = primitive.launch(query, *data, table, counts, output)
            primitive.validate_receipt(receipt)
            graph.replay()
            actual = output[row : row + 1].cpu()
            page_ids = order.cpu()[positions // block].long()
            selected_k = decoded[0][page_ids, 1, positions % block][None]
            selected_v = decoded[1][page_ids, 1, positions % block][None]
            proof = attention_bounds.enclose(
                corpus,
                selected_k,
                selected_v,
                actual,
                scale=dim**-0.5,
                splits=splits,
                probability_dtype=dtype if precision == "auto" else torch.float32,
            )
            assert proof["outside_enclosure"] == 0, proof
            if previous is not None:
                assert torch.equal(actual, previous)
            previous = actual
    for before, after in zip(
        protected, (x for x in (*data, *scales) if x is not None), strict=True
    ):
        assert torch.equal(before, after)
    if strided:
        assert torch.equal(backing, protected_backing)


@pytest.mark.parametrize("cache_dtype,quantized_sides", [
    (CacheDType.FP8, "kv"), (CacheDType.INT8, "kv"), (CacheDType.INT6, "kv"),
    (CacheDType.INT4, "kv"), (CacheDType.INT4, "k"), (CacheDType.INT4, "v"),
])
def test_shared_page_stride_encoded_attention(cache_dtype, quantized_sides):
    test_sealed_encoded_pages_preserve_attention_tree_and_physical_row_mapping(
        quantized_sides=quantized_sides, precision="auto", implementation="cuda",
        cache_dtype=cache_dtype, geometry=(2, 34, 80, torch.bfloat16, 32, False),
        axes=(CacheGroupAxis.TOKENS, CacheGroupAxis.CHANNELS), strided=True,
    )


@pytest.mark.parametrize(
    "group,axes,quantized_sides",
    [
        (8, (CacheGroupAxis.TOKENS, CacheGroupAxis.CHANNELS), "kv"),
        (7, (CacheGroupAxis.CHANNELS, CacheGroupAxis.TOKENS), "kv"),
        (7, (CacheGroupAxis.TOKENS, CacheGroupAxis.CHANNELS), "k"),
        (8, (CacheGroupAxis.CHANNELS, CacheGroupAxis.TOKENS), "v"),
    ],
)
@pytest.mark.parametrize("incoming_count", [1, 4])
@pytest.mark.parametrize("precision", ["auto", "fp32"])
@pytest.mark.parametrize("implementation,cache_dtype", [
    ("cuda", CacheDType.FP8), ("triton", CacheDType.FP8),
    ("cuda", CacheDType.INT8), ("cuda", CacheDType.INT6), ("cuda", CacheDType.INT4),
])
def test_native_tail_and_mtp_span_use_causal_views_without_reading_unsealed_codes(
    incoming_count,
    precision,
    implementation,
    group,
    axes,
    quantized_sides,
    cache_dtype,
):
    from cubic_serve.guardrail.quantized_graph import attention_bounds
    from cubic_serve.ops.cuda.attention.paged_decode import PagedGQADecodePrimitive
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    if implementation == "cuda" and precision != "auto":
        pytest.skip("the native MMA candidate requires native QKV operands")

    configure_reference_math()
    generator = torch.Generator().manual_seed(5001)
    layers, block, heads, dim, query_heads = 2, 2 * group, 2, 32, 4
    history = 5 * group - 1  # Review crosses the next immutable column-group boundary.
    native = [torch.randn((6 * group, heads, dim), generator=generator).bfloat16()
              for _ in range(2)]
    layouts = [
        GroupedCacheLayout(
            CacheStorageSpec(cache_dtype if side in quantized_sides else CacheDType.BF16,
                             axis, group), block, heads, dim
        )
        for side, axis in zip(("k", "v"), axes, strict=True)
    ]
    page_order = torch.tensor([2, 0, 1], dtype=torch.int32)
    data, scales, dequantized = [], [], []
    with ReferenceExecutionContext():
        for source, layout in zip(native, layouts, strict=True):
            encoded = [encode_cache_reference(part, layout) for part in source.split(block)]
            logical_data = torch.stack([x.payload for x in encoded])
            logical_scales = (torch.stack([x.scales for x in encoded])
                              if layout.spec.dtype.quantized else None)
            physical_data = torch.empty_like(logical_data)
            physical_data[page_order.long()] = logical_data
            if logical_scales is not None:
                physical_scales = torch.empty_like(logical_scales)
                physical_scales[page_order.long()] = logical_scales
            data.append(
                physical_data[:, None].expand(-1, layers, *physical_data.shape[1:])
                .contiguous().cuda()
            )
            scales.append(physical_scales[:, None].expand(-1, layers, -1).contiguous().cuda()
                          if logical_scales is not None else None)
            dequantized.append(
                torch.cat([decode_cache_reference(x, output_dtype=torch.bfloat16) for x in encoded])
            )
    # Unsealed columns and future row groups are poison, not plausible data.
    last_page = int(page_order[2])
    for payload, scale, layout in zip(data, scales, layouts, strict=True):
        if layout.spec.dtype.quantized:
            offset = (0 if layout.spec.axis is CacheGroupAxis.TOKENS
                      else (group - 1) * heads * ((dim + group - 1) // group))
            payload[last_page, :, offset:].fill_(255)
            scale[last_page, :, offset:].fill_(float("nan"))
        else:
            payload[last_page, :, group - 1:].fill_(float("nan"))
    encoding = PagedKVEncoding(*layouts, *scales)
    corpus = torch.randn((incoming_count, query_heads, dim), generator=generator).bfloat16()
    from cubic_serve.ops.eager.kv_cache import ReferenceKVCache
    from cubic_serve.runtime.cache.quantization import KVCacheStorageSpec

    with ReferenceExecutionContext():
        reference = ReferenceKVCache.empty(
            KVCacheStorageSpec(*(layout.spec for layout in layouts)), heads=heads, head_dim=dim
        )
        reference = reference.review(native[0][:history], native[1][:history]).commit(history)
        review = reference.review(
            native[0][history : history + incoming_count],
            native[1][history : history + incoming_count],
        )
    previous = None
    with operator_strategy_scope(
        OperatorConfig(attention_dtype=precision, softmax_dtype="fp32"), device="", model=""
    ):
        for capacity in (1, 8) if incoming_count == 1 else (4, 8):
            first = capacity - incoming_count
            query = torch.full(
                (capacity, query_heads, dim), float("nan"), dtype=torch.bfloat16, device="cuda"
            )
            query[first:].copy_(corpus)
            incoming = [
                torch.zeros((capacity, heads, dim), dtype=query.dtype, device="cuda")
                for _ in range(2)
            ]
            for values, source in zip(incoming, native, strict=True):
                values[first:].copy_(source[history : history + incoming_count])
            starts = torch.zeros(capacity, dtype=torch.int64, device="cuda")
            starts[first:] = history
            roots = torch.arange(capacity, dtype=torch.int32, device="cuda")
            roots[first:] = first
            sources = torch.arange(capacity, dtype=torch.int64, device="cuda")
            sources[first:] = first
            tails = []
            for values, layout in zip(native, layouts, strict=True):
                if layout.spec.dtype.quantized and layout.spec.axis is CacheGroupAxis.TOKENS:
                    tail = torch.zeros((capacity, layers, group, heads, dim),
                                       dtype=query.dtype, device="cuda")
                    tail[first, :, : group - 1].copy_(
                        values[4 * group:history][None].expand(layers, -1, -1, -1)
                    )
                    tails.append(tail)
                else:
                    tails.append(None)
            window = PagedKVReadWindow(*incoming, starts, roots, sources, *tails)
            positions = torch.zeros(capacity, dtype=torch.int64, device="cuda")
            positions[first:] = torch.arange(
                history, history + incoming_count, dtype=torch.int64, device="cuda"
            )
            write_pages = page_order.cuda()[positions // block]
            activity = torch.zeros(capacity, dtype=torch.int32, device="cuda")
            activity[first:] = 1
            segments = torch.zeros_like(activity)
            segments[first] = incoming_count
            written = torch.zeros((3, layers, block), dtype=torch.bool, device="cuda")
            preparation = bind_cache_append(
                *data,
                written,
                encoding,
                window,
                layer=1,
                positions=positions,
                pages=write_pages,
                destinations=sources,
                segments=segments,
                activity=activity,
                phase="prepare",
            )
            source_slots = torch.arange(capacity, dtype=torch.int64, device="cuda")
            metadata = bind_cache_window_metadata(
                segments,
                activity,
                positions,
                source_slots,
                starts,
                roots,
                sources,
            )
            output = torch.empty_like(query)
            table = page_order.cuda()[None].expand(capacity, -1).contiguous()
            lengths = torch.ones(capacity, dtype=torch.int32, device="cuda")
            lengths[first:] = torch.arange(
                history + 1, history + incoming_count + 1, dtype=torch.int32, device="cuda"
            )
            primitive = PagedGQADecodePrimitive(
                query,
                *data,
                table,
                lengths,
                output,
                layer=1,
                cache_encoding=encoding,
                cache_window=window,
                implementation=implementation,
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                metadata.launch()
                preparation.launch()
                receipt = primitive.launch(query, *data, table, lengths, output)
            primitive.validate_receipt(receipt)
            graph.replay()
            actual = output[first:].cpu()
            for offset in range(incoming_count):
                with ReferenceExecutionContext():
                    key_view, value_view = review.visible(offset + 1)
                proof = attention_bounds.enclose(
                    corpus[offset : offset + 1],
                    key_view[None],
                    value_view[None],
                    actual[offset : offset + 1],
                    scale=dim**-0.5,
                    splits=1,
                    probability_dtype=query.dtype if precision == "auto" else torch.float32,
                )
                assert proof["outside_enclosure"] == 0, proof
            if previous is not None:
                assert torch.equal(actual, previous)
            previous = actual


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("quantized_sides", ["k", "v", "kv"])
@pytest.mark.parametrize("position_offset", [0, 1 << 40, 1 << 62])
@pytest.mark.parametrize("cache_dtype", [
    CacheDType.FP8, CacheDType.INT8, CacheDType.INT6, CacheDType.INT4,
])
def test_append_checkpoints_and_page_cow_preserve_native_tail_and_protected_prefix(
    dtype,
    quantized_sides,
    position_offset,
    cache_dtype,
    *,
    group=8,
    retry_after_rejection=False,
):
    from cubic_serve.ops.cuda.attention.paged_decode import PageCOWCopyPrimitive
    from cubic_serve.ops.eager.kv_cache import ReferenceKVCache
    from cubic_serve.runtime.cache.quantization import KVCacheStorageSpec

    configure_reference_math()
    gen = torch.Generator().manual_seed(1255)
    block, heads, dim, layers, history, rows = 2 * group, 2, 37, 2, 5 * group - 3, 4
    plain = CacheDType.BF16 if dtype is torch.bfloat16 else CacheDType.FP16
    specs = (
        CacheStorageSpec(
            cache_dtype if "k" in quantized_sides else plain, CacheGroupAxis.TOKENS, group
        ),
        CacheStorageSpec(
            cache_dtype if "v" in quantized_sides else plain, CacheGroupAxis.CHANNELS, group
        ),
    )
    native = [torch.randn((6 * group, heads, dim), generator=gen).to(dtype) for _ in range(2)]
    incoming = [torch.randn((rows, heads, dim), generator=gen).to(dtype) for _ in range(2)]
    # The last review token is a large rejected lookahead on several branches.
    for tensor in incoming:
        tensor[-1] *= 100
    layouts = [GroupedCacheLayout(spec, block, heads, dim) for spec in specs]
    data, scales = [], []
    with ReferenceExecutionContext():
        for source, layout in zip(native, layouts, strict=True):
            encoded = [encode_cache_reference(part, layout) for part in source.split(block)]
            original = torch.stack([item.payload for item in encoded])
            physical = torch.cat((original, torch.zeros_like(original)))
            data.append(
                physical[:, None].expand(6, layers, *physical.shape[1:]).contiguous().cuda()
            )
            if layout.spec.dtype.quantized:
                original_scales = torch.stack([item.scales for item in encoded])
                physical_scales = torch.cat((original_scales, torch.zeros_like(original_scales)))
                scales.append(physical_scales[:, None].expand(6, layers, -1).contiguous().cuda())
            else:
                scales.append(None)
        reference = ReferenceKVCache.empty(
            KVCacheStorageSpec(*specs), heads=heads, head_dim=dim, dtype=dtype
        )
        reference = reference.review(native[0][:history], native[1][:history]).commit(history)
        review = reference.review(*incoming)
    encoding = PagedKVEncoding(*layouts, *scales)
    written = torch.zeros((6, layers, block), dtype=torch.bool, device="cuda")
    written[:2] = True
    written[2, :, : history % block] = True
    original_payloads = [t[:3].clone() for t in (*data, written, *scales) if t is not None]
    source_pages = torch.arange(3, dtype=torch.int32, device="cuda")
    target_pages = source_pages + 3
    copy = PageCOWCopyPrimitive(
        *data,
        written,
        source_pages,
        target_pages,
        page_fields=tuple(t for t in scales if t is not None),
        cache_encoding=encoding,
    )
    tails = []
    for side, spec in enumerate(specs):
        if spec.dtype.quantized and spec.axis is CacheGroupAxis.TOKENS:
            checkpoint_count = 2 * rows + 1 if retry_after_rejection else rows + 1
            tail = torch.zeros((checkpoint_count, layers, group, heads, dim),
                               dtype=dtype, device="cuda")
            tail[0, :, : history % group].copy_(
                native[side][history - history % group : history][None].expand(layers, -1, -1, -1)
            )
            tails.append(tail)
        else:
            tails.append(None)
    starts = torch.full((rows,), history + position_offset, dtype=torch.int64, device="cuda")
    roots = torch.zeros(rows, dtype=torch.int32, device="cuda")
    sources = torch.zeros(rows, dtype=torch.int64, device="cuda")
    window = PagedKVReadWindow(*(t.cuda() for t in incoming), starts, roots, sources, *tails)
    positions = torch.arange(
        history + position_offset, history + position_offset + rows,
        dtype=torch.int64, device="cuda",
    )
    pages = ((positions - position_offset) // block + 3).int()
    destinations = torch.arange(1, rows + 1, dtype=torch.int64, device="cuda")
    segments = torch.tensor([rows, 0, 0, 0], dtype=torch.int32, device="cuda")
    activity = torch.ones(rows, dtype=torch.int32, device="cuda")
    append = bind_cache_append(
        *data,
        written,
        encoding,
        window,
        layer=1,
        positions=positions,
        pages=pages,
        destinations=destinations,
        segments=segments,
        activity=activity,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        copy.launch(*data, written, source_pages, target_pages)
        receipts = append.launch()
    assert len(append.project_artifacts()) == len(receipts) == 2
    graph.replay()
    for count in range(1, rows + 1):
        with ReferenceExecutionContext():
            accepted = review.commit(count)
        for tail, stream in zip(tails, (accepted.key, accepted.value), strict=True):
            if tail is not None:
                expected = torch.zeros((group, heads, dim), dtype=dtype)
                expected[: stream.tail.shape[0]] = stream.tail
                assert torch.equal(tail[count, 1].cpu(), expected)
    # A complete K group is encoded from only its actual native values;
    # the large token in the following group cannot influence its scale.
    for side, (layout, payload, scale) in enumerate(zip(layouts, data, scales, strict=True)):
        page_start = history // block * block
        expected_native = native[side][page_start : page_start + block].clone()
        expected_native[history % block : history % block + rows] = incoming[side]
        with ReferenceExecutionContext():
            expected = encode_cache_reference(expected_native, layout)
        if layout.spec.dtype.quantized:
            selected = []
            for token in (
                range(group)
                if layout.spec.axis is CacheGroupAxis.TOKENS
                else range(history % block, history % block + rows)
            ):
                for head in range(heads):
                    for channel in range(dim):
                        selected.append(layout.group_location(token, head, channel)[0])
            indices = torch.tensor(sorted(set(selected)), dtype=torch.long)
            assert torch.equal(payload[5, 1].cpu()[indices], expected.payload[indices])
            assert torch.equal(scale[5, 1].cpu()[indices], expected.scales[indices])
        else:
            assert torch.equal(
                payload[5, 1, history % block : history % block + rows].cpu(), incoming[side]
            )
    for before, after in zip(
        original_payloads, (t[:3] for t in (*data, written, *scales) if t is not None), strict=True
    ):
        assert torch.equal(before.view(torch.uint8), after.view(torch.uint8))

    if not retry_after_rejection:
        return

    # Reject all but the first reviewed input. Retain its independent tail,
    # while raw codes/scales may still contain the rejected future seal.
    count = 1
    with ReferenceExecutionContext():
        committed = review.commit(count)
    new_incoming = [torch.randn((rows, heads, dim), generator=gen).to(dtype) for _ in range(2)]
    for x in new_incoming:
        x[0] *= 99
    with ReferenceExecutionContext():
        new_review = committed.review(*new_incoming)
    start = history + count
    new_starts = torch.full_like(starts, start + position_offset)
    new_sources = torch.ones_like(sources)
    new_window = PagedKVReadWindow(*(x.cuda() for x in new_incoming), new_starts,
                                  roots, new_sources, *tails)
    new_positions = torch.arange(start + position_offset, start + position_offset + rows,
                                 dtype=torch.int64, device='cuda')
    new_pages = ((new_positions - position_offset) // block + 3).int()
    new_destinations = torch.arange(rows + 1, 2 * rows + 1,
                                   dtype=torch.int64, device='cuda')
    new_append = bind_cache_append(*data, written, encoding, new_window, layer=1,
        positions=new_positions, pages=new_pages, destinations=new_destinations,
        segments=segments, activity=activity)
    old_future_payload = data[0][5, 1].clone()
    old_future_scale = scales[0][5, 1].clone()
    retry_protected = tuple(
        t[:5].clone() for t in (*data, written, *scales) if t is not None
    )
    source_tails = tuple((tail, tail[:rows + 1].clone()) for tail in tails if tail is not None)
    retry_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(retry_graph):
        new_append.launch()
    retry_graph.replay()
    for n in range(1, rows + 1):
        with ReferenceExecutionContext():
            accepted = new_review.commit(n)
        for tail, stream in zip(tails, (accepted.key, accepted.value), strict=True):
            if tail is not None:
                expected_tail = torch.zeros((group, heads, dim), dtype=dtype)
                expected_tail[:stream.tail.shape[0]] = stream.tail
                assert torch.equal(tail[rows + n, 1].cpu(), expected_tail), ('retry tail', n)
    assert not (torch.equal(old_future_payload, data[0][5, 1]) and
                torch.equal(old_future_scale, scales[0][5, 1])), 'future group not resealed'
    for side, (layout, payload, scale) in enumerate(zip(layouts, data, scales, strict=True)):
        page_start = history // block * block
        expected_native = native[side][page_start : page_start + block].clone()
        expected_native[history % block : history % block + count] = incoming[side][:count]
        expected_native[start % block : start % block + rows] = new_incoming[side]
        with ReferenceExecutionContext():
            expected = encode_cache_reference(expected_native, layout)
        tokens = (range(group) if layout.spec.axis is CacheGroupAxis.TOKENS
                  else range(history % block, start % block + rows))
        indices = torch.tensor(sorted({layout.group_location(token, head, channel)[0]
            for token in tokens for head in range(heads) for channel in range(dim)}))
        assert torch.equal(payload[5, 1].cpu()[indices], expected.payload[indices]), (
            "retry codes", side,
        )
        assert torch.equal(scale[5, 1].cpu()[indices], expected.scales[indices]), (
            "retry scales", side,
        )
    for old, current in zip(
        retry_protected,
        (t[:5] for t in (*data, written, *scales) if t is not None), strict=True,
    ):
        assert torch.equal(old.view(torch.uint8), current.view(torch.uint8)), (
            "protected prefix changed"
        )
    for tail, original in source_tails:
        assert torch.equal(tail[:rows + 1], original), "rejection checkpoint changed"


@pytest.mark.parametrize("cache_dtype,implementation", [
    (CacheDType.FP8, "cuda"), (CacheDType.FP8, "triton"),
    (CacheDType.INT8, "cuda"), (CacheDType.INT6, "cuda"), (CacheDType.INT4, "cuda"),
])
def test_append_rejects_column_groups_that_cross_physical_pages(cache_dtype, implementation):
    # A sealed read-only view may have a partial last group; mutable pages must
    # keep every immutable column group within a single physical owner.
    layout = GroupedCacheLayout(CacheStorageSpec(cache_dtype, CacheGroupAxis.TOKENS, 8),
                                15, 1, 8)
    payload = [torch.zeros((1, 1, layout.group_count, layout.packed_group_bytes),
                           dtype=torch.uint8, device="cuda") for _ in range(2)]
    scales = [torch.zeros((1, 1, layout.group_count), device="cuda") for _ in range(2)]
    encoding = PagedKVEncoding(layout, layout, *scales)
    incoming = [torch.zeros((1, 1, 8), dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    positions = torch.full((1,), 7, dtype=torch.int64, device="cuda")
    roots = torch.zeros(1, dtype=torch.int32, device="cuda")
    owners = torch.zeros(1, dtype=torch.int64, device="cuda")
    tails = [torch.zeros((1, 1, 8, 1, 8), dtype=torch.bfloat16, device="cuda")
             for _ in range(2)]
    window = PagedKVReadWindow(*incoming, positions, roots, owners, *tails)
    with pytest.raises(ValueError, match="column cache groups must divide their page"):
        bind_cache_append(
            *payload, torch.zeros((1, 1, 15), dtype=torch.bool, device="cuda"), encoding, window,
            layer=0, positions=positions, pages=roots, destinations=owners,
            segments=torch.ones_like(roots), activity=torch.ones_like(roots),
            implementation=implementation,
        )


@pytest.mark.parametrize("sides", ["k", "v", "kv"])
@pytest.mark.parametrize("strided", [False, True])
def test_new_page_reset_uses_each_side_byte_geometry_and_preserves_inactive_records(sides, strided):
    from cubic_serve.ops.cuda.pointwise.qwen_decode import update_resident_pages
    from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout
    from cubic_serve.runtime.cache.quantization import KVCacheStorageSpec

    configure_reference_math()
    layout = KVLayout(
        (0, 3), 32, 3, 37,
        storage=KVCacheStorageSpec(
            CacheStorageSpec(CacheDType.FP8 if "k" in sides else CacheDType.BF16,
                             CacheGroupAxis.TOKENS),
            CacheStorageSpec(CacheDType.FP8 if "v" in sides else CacheDType.BF16,
                             CacheGroupAxis.CHANNELS),
        ),
    )
    state_specs = layout.quantization_state_specs
    arena = CacheArena(CacheArenaSpec("reset", 3, 2 if state_specs else 0, state_specs),
                       layout, "cuda", rank_id=0)
    if strided:
        from cubic_serve.runtime.cache.shared_pool import SharedCacheField, SharedCacheLayout
        payloads = {"key": arena.key, "value": arena.value, "written": arena.written,
                    **arena.page_fields}
        packed = SharedCacheLayout(tuple(SharedCacheField(name, t.dtype, tuple(t.shape[1:]))
                                         for name, t in payloads.items())
                                   + (SharedCacheField("guard", torch.uint8, (7,)),))
        raw = torch.full((arena.key.shape[0] * packed.entry_bytes,), 0x5A,
                         dtype=torch.uint8, device="cuda")
        views = packed.views(raw, arena.key.shape[0])
        for name, tensor in payloads.items():
            views[name].copy_(tensor)
        arena.key, arena.value, arena.written = (views[name] for name in ("key", "value", "written"))
        arena.page_fields = {name: views[name] for name in arena.page_fields}
    fields = tuple(arena.page_fields.values())
    for tensor in (arena.key, arena.value, *fields):
        tensor.fill_(17)
    arena.written.fill_(True)
    metadata = [torch.full((2,), 19, device="cuda", dtype=torch.int32) for _ in range(3)]
    table = torch.full((2, 3), -91, device="cuda", dtype=torch.int32)
    updates = torch.tensor([[0, 1, 1], [-99999, -99999, -99999]], device="cuda", dtype=torch.int32)
    active = torch.tensor([True, False], device="cuda")
    tensors = (arena.key, arena.value, arena.written, *metadata, table)
    snapshot = [tensor.cpu() for tensor in (*tensors, *fields)]
    update_resident_pages(updates, *tensors, page_fields=fields,
                          update_activity=active, compile_only=True)
    assert all(torch.equal(before, after.cpu()) for before, after in
               zip(snapshot, (*tensors, *fields), strict=True))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        update_resident_pages(updates, *tensors, page_fields=fields, update_activity=active)
    graph.replay()
    expected = [tensor.clone() for tensor in snapshot]
    for tensor in (*expected[:3], *expected[7:]):
        tensor[1] = 0
    for tensor in expected[3:6]:
        tensor[0] = 0 if tensor is expected[3] else 1
    expected[6][0, 1] = 1
    assert all(torch.equal(before, after.cpu()) for before, after in
               zip(expected, (*tensors, *fields), strict=True))
    if strided:
        assert (views["guard"] == 0x5A).all()


@pytest.mark.parametrize("cache_dtype", [
    CacheDType.FP8, CacheDType.INT8, CacheDType.INT6, CacheDType.INT4,
])
@pytest.mark.parametrize("splits", [1, 32])
@pytest.mark.parametrize("implementation,precision", [
    ("cuda", "auto"), ("triton", "auto"), ("triton", "fp32"),
])
def test_encoded_attention_padding_never_reads_stale_context_or_native_span(
    cache_dtype, splits, implementation, precision,
):
    from cubic_serve.ops.cuda.attention.paged_decode import PagedGQADecodePrimitive
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    if implementation == "triton" and cache_dtype is not CacheDType.FP8:
        pytest.skip("integer cache attention uses the native CUDA contract")
    # A retired physical row retains a long context and poisoned page/owner
    # metadata. Explicit activity, rather than segment starts or those lengths,
    # decides whether attention may consume it. Adjacent backing rows are poison
    # so a tensor-extent violation is observable even in a caching allocation.
    rows, heads, dim, query_heads, page = 2, 1, 64, 19, 32
    layouts = tuple(GroupedCacheLayout(
        CacheStorageSpec(cache_dtype, axis, page), page, heads, dim,
    ) for axis in (CacheGroupAxis.TOKENS, CacheGroupAxis.CHANNELS))
    data = [torch.zeros((2, 1, g.group_count, g.packed_group_bytes),
                        device="cuda", dtype=torch.uint8) for g in layouts]
    scales = [torch.ones((2, 1, g.group_count), device="cuda") for g in layouts]
    encoding = PagedKVEncoding(*layouts, *scales)
    backing = [torch.full((rows + 64, heads, dim), float("nan"),
                          device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    incoming = [x[:rows] for x in backing]
    for x in incoming:
        x.fill_(1)
    query = torch.ones((rows, query_heads, dim), device="cuda", dtype=torch.bfloat16)
    tail = torch.zeros((rows, 1, page, heads, dim), device="cuda", dtype=query.dtype)
    starts = torch.tensor([32, 0], device="cuda", dtype=torch.int64)
    roots = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    sources = torch.arange(rows, device="cuda", dtype=torch.int64)
    activity = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    window = PagedKVReadWindow(*incoming, starts, roots, sources, tail, None, activity)
    table = torch.tensor([[0, 1], [0, 1]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([33, 33], device="cuda", dtype=torch.int32)
    output = torch.full_like(query, float("nan"))
    options = {}
    if splits == 32:
        options = dict(
            partial=torch.empty((rows, query_heads, splits, dim), device="cuda"),
            statistics=torch.empty((rows, query_heads, splits, 2), device="cuda"),
            arrivals=torch.zeros((rows, query_heads), device="cuda", dtype=torch.int32),
        )
    with operator_strategy_scope(
        OperatorConfig(attention_dtype=precision, softmax_dtype="fp32"), device="", model="",
    ):
        op = PagedGQADecodePrimitive(
            query, *data, table, lengths, output, layer=0,
            cache_encoding=encoding, cache_window=window,
            implementation=implementation, **options,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op.launch(query, *data, table, lengths, output)
        graph.replay()
        first = output.clone()
        assert torch.isfinite(first[0]).all()
        assert torch.equal(first[1], torch.zeros_like(first[1]))
        # Invalid identity metadata may never become addresses for padding.
        table[1].fill_(-2**30)
        sources[1] = 2**60
        lengths[1] = 2**30
        graph.replay()
        assert torch.equal(output, first)
        # Activity is replay-time data: retire row zero and admit row one using
        # the identical sealed graph, without reconstructing the primitive.
        activity.copy_(torch.tensor([0, 1], device="cuda", dtype=torch.int32))
        table[1].copy_(table[0])
        starts[1] = 32
        sources[1] = 1
        lengths[1] = 33
        table[0].fill_(-2**30)
        sources[0] = 2**60
        lengths[0] = 2**30
        graph.replay()
        assert torch.equal(output[0], torch.zeros_like(output[0]))
        assert torch.equal(output[1], first[0])
        if splits == 32:
            assert torch.equal(options["arrivals"], torch.zeros_like(options["arrivals"]))


@pytest.mark.parametrize("fmt", [CacheDType.FP8, CacheDType.INT8,
                                CacheDType.INT6, CacheDType.INT4])
@pytest.mark.parametrize("rotated_sides", ["k", "v", "kv"])
@pytest.mark.parametrize("native", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("dim", [32, 96, 256], ids=["d32", "d96", "d256"])
@pytest.mark.parametrize("splits", [1, 32], ids=["s1", "s32"])
def test_native_attention_restores_rotated_value_basis_before_output_rounding(
    fmt, rotated_sides, native, dim, splits,
):
    from cubic_serve.guardrail.quantized_graph.attention_bounds import enclose
    from cubic_serve.ops.cuda.attention.paged_decode import PagedGQADecodePrimitive
    from cubic_serve.ops.eager.rotation import rotate_cache_reference
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    generator = torch.Generator().manual_seed(4805)
    key_format, value_format = fmt if isinstance(fmt, tuple) else (fmt, fmt)
    native_format = CacheDType.BF16 if native is torch.bfloat16 else CacheDType.FP16
    heads, query_heads, tokens, page = 1, 19, 64, 32
    raw = [torch.randn((tokens, heads, dim), generator=generator).to(native) for _ in range(2)]
    original_query = torch.randn((1, query_heads, dim), generator=generator).to(native)
    layouts = tuple(GroupedCacheLayout(
        CacheStorageSpec(kind if side in rotated_sides else native_format, axis, 32),
        page, heads, dim,
    ) for side, axis, kind in zip(
        ("k", "v"), (CacheGroupAxis.TOKENS, CacheGroupAxis.CHANNELS),
        (key_format, value_format),
    ))
    with ReferenceExecutionContext():
        operands = [rotate_cache_reference(x) if side in rotated_sides else x
                    for x, side in zip(raw, ("k", "v"))]
        cpu_query = (rotate_cache_reference(original_query) if "k" in rotated_sides
                     else original_query)
        encoded = [[encode_cache_reference(x, layout) for x in values.split(page)]
                   for values, layout in zip(operands, layouts)]
        decoded = [torch.cat([decode_cache_reference(x, output_dtype=native) for x in side])
                   for side in encoded]
    payload = [torch.stack([x.payload for x in side])[:, None].contiguous().cuda()
               for side in encoded]
    scales = [(torch.stack([x.scales for x in side])[:, None].contiguous().cuda()
               if layout.spec.dtype.quantized else None) for side, layout in zip(encoded, layouts)]
    encoding = PagedKVEncoding(*layouts, *scales, enable_rotation=True)
    query = cpu_query.cuda()
    output = torch.full_like(query, float("nan"))
    table = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([tokens], device="cuda", dtype=torch.int32)
    options = {}
    if splits == 32:
        options = dict(
            partial=torch.empty((1, query_heads, splits, dim), device="cuda"),
            statistics=torch.empty((1, query_heads, splits, 2), device="cuda"),
            arrivals=torch.zeros((1, query_heads), device="cuda", dtype=torch.int32),
        )
    before = [x.clone() for x in (query, *payload, *(x for x in scales if x is not None))]
    with operator_strategy_scope(
        OperatorConfig(attention_dtype="auto", softmax_dtype="fp32"), device="", model="",
    ):
        primitive = PagedGQADecodePrimitive(
            query, *payload, table, lengths, output, layer=0, cache_encoding=encoding,
            implementation="cuda", **options,
        )
        primitive.launch(query, *payload, table, lengths, output)
        expected_capture = output.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            primitive.launch(query, *payload, table, lengths, output)
        output.fill_(float("nan"))
        graph.replay()
        assert torch.equal(output, expected_capture)
    proof = enclose(cpu_query, *(x[None] for x in decoded), output.cpu(),
                    scale=dim**-0.5, splits=splits, probability_dtype=native,
                    value_rotation="v" in rotated_sides)
    assert proof["outside_enclosure"] == 0, proof
    assert all(torch.equal(a, b) for a, b in zip(
        before, (query, *payload, *(x for x in scales if x is not None)),
    ))
    if splits == 32:
        assert torch.equal(options["arrivals"], torch.zeros_like(options["arrivals"]))


@pytest.mark.parametrize("native", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("dim", [7, 32, 96, 256])
@pytest.mark.parametrize("rows", [1, 17])
@pytest.mark.parametrize("rotated_sides", ["k", "v", "kv"])
def test_native_cache_rotation_preserves_raw_producers_and_native_query_dtype(
    native, dim, rows, rotated_sides,
):
    from cubic_serve.ops.cuda.cache_rotation_native import bind_cache_rotation
    from cubic_serve.ops.eager.rotation import rotate_cache_reference

    generator = torch.Generator().manual_seed(8317)
    cpu = [torch.randn((rows, heads, dim), generator=generator).to(native) for heads in (4, 1, 1)]
    inputs = [x.cuda() for x in cpu]
    frozen = [x.clone() for x in inputs]
    flags = ("k" in rotated_sides, "k" in rotated_sides, "v" in rotated_sides)
    outputs = [torch.full_like(x, float("nan")) if enabled else None
               for x, enabled in zip(inputs, flags)]
    launch = bind_cache_rotation(
        inputs[0] if flags[0] else None, *inputs[1:],
        query_output=outputs[0], key_output=outputs[1], value_output=outputs[2],
        rotate_key=flags[1], rotate_value=flags[2],
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch.launch()
    graph.replay()
    with ReferenceExecutionContext():
        for source, actual, enabled in zip(cpu, outputs, flags):
            if enabled:
                assert actual.dtype is native
                assert torch.equal(actual.cpu(), rotate_cache_reference(source))
    assert all(torch.equal(x, y) for x, y in zip(inputs, frozen))
    # Same sealed graph must consume newly filled native inputs.
    for x in inputs:
        x.mul_(0.5)
    graph.replay()
    with ReferenceExecutionContext():
        for source, actual, enabled in zip(inputs, outputs, flags):
            if enabled:
                assert torch.equal(actual.cpu(), rotate_cache_reference(source.cpu()))


def test_native_cache_rotation_rejects_aliasing_before_launch():
    from cubic_serve.ops.cuda.cache_rotation_native import bind_cache_rotation

    query = torch.ones((1, 4, 32), device="cuda", dtype=torch.bfloat16)
    key, value = [torch.ones((1, 1, 32), device="cuda", dtype=query.dtype) for _ in range(2)]
    with pytest.raises(ValueError, match="alias"):
        bind_cache_rotation(query, key, value, query_output=query, key_output=torch.empty_like(key),
                            value_output=None, rotate_key=True, rotate_value=False)


@pytest.mark.parametrize("fmt", [CacheDType.FP8, CacheDType.INT8,
                                CacheDType.INT6, CacheDType.INT4])
@pytest.mark.parametrize("rotated_sides", ["k", "v", "kv"])
def test_rotating_shared_cache_binding_seals_and_checkpoints_partial_review(fmt, rotated_sides):
    from cubic_serve.guardrail.quantized_graph.attention_bounds import enclose
    from cubic_serve.ops.cuda.attention.paged_decode import PageCOWCopyPrimitive, PagedGQADecodePrimitive
    from cubic_serve.ops.cuda.cache_quantization import PagedKVGraphBinding
    from cubic_serve.ops.eager.kv_cache import ReferenceKVCache, cache_page_reference
    from cubic_serve.ops.eager.rotation import rotate_cache_reference
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope
    from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout
    from cubic_serve.runtime.cache.quantization import KVCacheStorageSpec

    rows, dim, group, history, stride = 4, 16, 32, 31, 3
    storage = KVCacheStorageSpec(
        CacheStorageSpec(fmt if "k" in rotated_sides else CacheDType.BF16,
                         CacheGroupAxis.TOKENS, group),
        CacheStorageSpec(fmt if "v" in rotated_sides else CacheDType.BF16,
                         CacheGroupAxis.CHANNELS, group), True,
    )
    layout = KVLayout((7,), group, 1, dim, storage=storage)
    spec = CacheArenaSpec("rotating", 4, 6, layout.quantization_state_specs, scratch_capacity=4)
    arena = CacheArena(spec, layout, "cuda", rank_id=0)
    generator = torch.Generator().manual_seed(6207)
    raw = [torch.randn((history + 3, 1, dim), generator=generator).bfloat16() for _ in range(2)]
    corpus = torch.randn((3, 3, dim), generator=generator).bfloat16()
    with ReferenceExecutionContext():
        prefix = ReferenceKVCache.empty(storage, heads=1, head_dim=dim)
        prefix = prefix.review(*(x[:history] for x in raw)).commit(history)
        review = prefix.review(*(x[history:] for x in raw))
        producers = [rotate_cache_reference(x) if side in rotated_sides else x
                     for x, side in zip(raw, ("k", "v"))]
        expected_query = rotate_cache_reference(corpus) if "k" in rotated_sides else corpus
        for name, native, grouped in zip(("key", "value"), producers,
                                         (layout.key_encoding, layout.value_encoding)):
            image = cache_page_reference(native[:history], grouped)
            getattr(arena, name)[0, 0].copy_(image.payload.cuda())
            if image.scales is not None:
                arena.page_fields[f"kv_{name}_scales"][0, 0].copy_(image.scales.cuda())
    arena.written[0, 0, :history] = True
    if "kv_key_tail" in arena.states:
        arena.states["kv_key_tail"][1, 0, :history].copy_(producers[0][:history].cuda())
    protected = [x[:2].clone() for x in (arena.key, arena.value, *arena.page_fields.values())]
    protected_tail = (None if "kv_key_tail" not in arena.states
                      else arena.states["kv_key_tail"][[1, 6]].clone())
    incoming = [torch.full((rows, 1, dim), float("nan"), device="cuda", dtype=torch.bfloat16)
                for _ in range(2)]
    for target, source in zip(incoming, raw):
        target[:3].copy_(source[history:].cuda())
    query = torch.full((rows, 3, dim), float("nan"), device="cuda", dtype=torch.bfloat16)
    query[:3].copy_(corpus.cuda())
    segments = torch.tensor([3, 0, 0, 0], device="cuda", dtype=torch.int32)
    activity = torch.tensor([1, 1, 1, 0], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([32, 33, 34, 2**30], device="cuda", dtype=torch.int32)
    owners = torch.tensor([[1, 0, 0, 6], [2, 3, 4, 6]], device="cuda", dtype=torch.int32) * stride
    metadata = PagedKVGraphBinding(arena, positions=lengths, positions_are_lengths=True,
                                  segments=segments, activity=activity, source_slots=owners[0],
                                  destination_slots=owners[1], owner_stride=stride)
    pages = torch.tensor([2, 3, 3, 4], device="cuda", dtype=torch.int32)
    sources = torch.tensor([0, 1, 3, 4], device="cuda", dtype=torch.int32)
    copy = PageCOWCopyPrimitive(arena.key, arena.value, arena.written, sources, pages,
                               cache_encoding=metadata.encoding,
                               page_fields=tuple(arena.page_fields.values()))
    window, prepare, checkpoint = metadata.bind_layer(*incoming, query=query, layer=0, pages=pages)
    owned_bytes = sum(x.numel() * x.element_size() for x in metadata.scratch)
    # Serial layers reuse scratch and input bindings instead of multiplying them.
    metadata.bind_layer(*incoming, query=query, layer=0, pages=pages)
    assert owned_bytes == sum(x.numel() * x.element_size() for x in metadata.scratch)
    assert len(metadata._rotation_plans) == 1
    rotated_query = metadata.query_for(query)
    if "k" in rotated_sides:
        assert rotated_query.data_ptr() != query.data_ptr()
    else:
        assert rotated_query is query
    output = torch.full_like(query, float("nan"))
    table = torch.tensor([[2, 3], [2, 3], [2, 3], [-2**30, -2**30]],
                         device="cuda", dtype=torch.int32)
    with operator_strategy_scope(
        OperatorConfig(attention_dtype="auto", softmax_dtype="fp32"), device="", model="",
    ):
        attention = PagedGQADecodePrimitive(
            rotated_query, arena.key, arena.value, table, lengths, output,
            layer=0, cache_encoding=metadata.encoding, cache_window=window,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            metadata.prepare_metadata()
            copy.launch(arena.key, arena.value, arena.written, sources, pages)
            prepare.launch()
            attention.launch(rotated_query, arena.key, arena.value, table, lengths, output)
            checkpoint.launch()
        graph.replay()
    assert torch.equal(output[3], torch.zeros_like(output[3]))
    assert torch.equal(query[:3].cpu(), corpus)
    for source, original in zip(incoming, raw):
        assert torch.equal(source[:3].cpu(), original[history:])
    for count in range(1, 4):
        with ReferenceExecutionContext():
            keys, values = review.visible(count)
            accepted = review.commit(count)
        proof = enclose(expected_query[count - 1:count], keys[None], values[None],
                        output[count - 1:count].cpu(), scale=dim**-0.5,
                        splits=attention._grid[2], probability_dtype=query.dtype,
                        value_rotation="v" in rotated_sides)
        assert proof["outside_enclosure"] == 0, proof
        if "kv_key_tail" in arena.states:
            expected = torch.zeros((group, 1, dim), dtype=query.dtype)
            expected[:accepted.key.tail.shape[0]] = accepted.key.tail
            assert torch.equal(arena.states["kv_key_tail"][count + 1, 0].cpu(), expected)
    assert all(torch.equal(a, b[:2]) for a, b in zip(
        protected, (arena.key, arena.value, *arena.page_fields.values()),
    ))
    if protected_tail is not None:
        assert torch.equal(arena.states["kv_key_tail"][[1, 6]], protected_tail)
    with ReferenceExecutionContext():
        for name, native, grouped in zip(("key", "value"), producers,
                                         (layout.key_encoding, layout.value_encoding)):
            for physical, span in ((2, native[:32]), (3, native[32:])):
                image = cache_page_reference(span, grouped)
                assert torch.equal(getattr(arena, name)[physical, 0].cpu(), image.payload)
                if image.scales is not None:
                    assert torch.equal(arena.page_fields[f"kv_{name}_scales"][physical, 0].cpu(),
                                       image.scales)


@pytest.mark.parametrize("key_format,value_format", [
    (CacheDType.FP8, CacheDType.INT4),
    (CacheDType.INT4, CacheDType.FP8),
    (CacheDType.INT6, CacheDType.INT8),
    (CacheDType.INT8, CacheDType.INT6),
])
@pytest.mark.parametrize("geometry", [
    (1, 12, 256, torch.bfloat16, 1, False),
    (2, 26, 37, torch.float16, 4, True),
])
def test_independent_kv_formats_preserve_selected_and_padded_attention(
    key_format, value_format, geometry,
):
    # Exercise different packed widths and codebooks in the same native reader,
    # with independent CPU-decoded operands, reversed pages and row relocation.
    test_sealed_encoded_pages_preserve_attention_tree_and_physical_row_mapping(
        "kv", "auto", "cuda", (key_format, value_format), geometry,
        (CacheGroupAxis.TOKENS, CacheGroupAxis.CHANNELS),
    )


@pytest.mark.parametrize("key_format,value_format", [
    (CacheDType.FP8, CacheDType.INT4),
    (CacheDType.INT4, CacheDType.FP8),
    (CacheDType.INT6, CacheDType.INT8),
    (CacheDType.INT8, CacheDType.INT6),
])
@pytest.mark.parametrize("native,splits", [(torch.bfloat16, 32), (torch.float16, 1)])
def test_independent_kv_formats_preserve_rotated_attention(
    key_format, value_format, native, splits,
):
    # D96 consists of three complete D32 rotation groups. The independent
    # enclosure includes the FP32 inverse before native output rounding.
    test_native_attention_restores_rotated_value_basis_before_output_rounding(
        (key_format, value_format), "kv", native, 96, splits,
    )


@pytest.mark.parametrize("key_format,value_format,native,dim,splits", [
    (CacheDType.FP8, CacheDType.FP8, torch.bfloat16, 96, 1),
    (CacheDType.INT4, CacheDType.INT6, torch.bfloat16, 256, 32),
    (CacheDType.INT6, CacheDType.INT4, torch.float16, 96, 16),
    (CacheDType.INT8, CacheDType.FP8, torch.bfloat16, 80, 4),
])
def test_rotated_attention_preserves_concurrent_rows_across_buckets(
    key_format, value_format, native, dim, splits,
):
    from cubic_serve.guardrail.quantized_graph.attention_bounds import enclose
    from cubic_serve.ops.cuda.attention.paged_decode import PagedGQADecodePrimitive
    from cubic_serve.ops.eager.rotation import rotate_cache_reference
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    generator = torch.Generator().manual_seed(4819)
    pages, layers, page, heads, query_heads = 4, 2, 32, 2, 26
    layouts = [GroupedCacheLayout(CacheStorageSpec(fmt, axis, 32), page, heads, dim)
               for fmt, axis in ((key_format, CacheGroupAxis.TOKENS),
                                 (value_format, CacheGroupAxis.CHANNELS))]
    payload, scales, decoded = [], [], []
    with ReferenceExecutionContext():
        for layout in layouts:
            raw = torch.randn((pages, layers, page, heads, dim), generator=generator).to(native)
            encoded = [[encode_cache_reference(rotate_cache_reference(raw[p, layer]), layout)
                        for layer in range(layers)] for p in range(pages)]
            payload.append(torch.stack([torch.stack([x.payload for x in row])
                                        for row in encoded]).cuda())
            scales.append(torch.stack([torch.stack([x.scales for x in row])
                                       for row in encoded]).cuda())
            decoded.append(torch.stack([torch.stack([decode_cache_reference(x, output_dtype=native)
                                                      for x in row]) for row in encoded]))
        corpus = rotate_cache_reference(torch.randn((3, query_heads, dim),
                                                     generator=generator).to(native))
    encoding = PagedKVEncoding(*layouts, *scales, enable_rotation=True)
    order = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int32)
    selections = (torch.arange(57), torch.arange(5, 39), torch.arange(0, 39, 2))
    protected = [x.clone() for x in (*payload, *scales)]

    def execute(capacity, logical_rows, physical_rows):
        query = torch.full((capacity, query_heads, dim), float("nan"),
                           dtype=native, device="cuda")
        output = torch.empty_like(query)
        table = torch.full((capacity, pages), -1, device="cuda", dtype=torch.int32)
        counts = torch.zeros(capacity, device="cuda", dtype=torch.int32)
        indices = torch.full((capacity, page * pages), -1, device="cuda", dtype=torch.int32)
        for logical, physical in zip(logical_rows, physical_rows, strict=True):
            query[physical].copy_(corpus[logical])
            table[physical].copy_(order)
            counts[physical] = selections[logical].numel()
            indices[physical, :selections[logical].numel()].copy_(selections[logical])
        scratch = {} if splits == 1 else dict(
            partial=torch.empty((capacity, query_heads, splits, dim), device="cuda"),
            statistics=torch.empty((capacity, query_heads, splits, 2), device="cuda"),
            arrivals=torch.zeros((capacity, query_heads), device="cuda", dtype=torch.int32),
        )
        primitive = PagedGQADecodePrimitive(
            query, *payload, table, counts, output, layer=1, cache_encoding=encoding,
            selected_indices=indices, implementation="cuda", **scratch,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            receipt = primitive.launch(query, *payload, table, counts, output)
        primitive.validate_receipt(receipt)
        graph.replay()
        actual = output[list(physical_rows)].cpu()
        if splits != 1:
            assert torch.equal(scratch["arrivals"], torch.zeros_like(scratch["arrivals"]))
        return actual

    with operator_strategy_scope(
        OperatorConfig(attention_dtype="auto", softmax_dtype="fp32"), device="", model="",
    ):
        expected = torch.cat([execute(1, (logical,), (0,)) for logical in range(3)])
        for logical, positions in enumerate(selections):
            ids = order.cpu()[positions // page].long()
            operands = [x[ids, 1, positions % page][None] for x in decoded]
            proof = enclose(corpus[logical:logical + 1], *operands,
                            expected[logical:logical + 1], scale=dim**-0.5,
                            splits=splits, probability_dtype=native, value_rotation=True)
            assert proof["outside_enclosure"] == 0, proof
        for capacity in (4, 8, 16, 32, 64, 128, 256, 512):
            rows = (0, capacity // 2, capacity - 1)
            for permutation in (rows, rows[::-1]):
                assert torch.equal(execute(capacity, (0, 1, 2), permutation), expected)
    assert all(torch.equal(a, b) for a, b in zip(protected, (*payload, *scales), strict=True))


@pytest.mark.parametrize("cache_dtype", [
    CacheDType.INT4, CacheDType.INT6, CacheDType.INT8, CacheDType.FP8,
])
@pytest.mark.parametrize("native", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("splits", [1, 4])
@pytest.mark.parametrize("selection", ["unaligned", "page-boundary"])
def test_lowbit_column_reads_preserve_word_boundaries_and_sparse_gaps(
    cache_dtype, native, splits, selection,
):
    # Eight selected tokens start at member five of a sixteen-token group.
    # Packed integer/FP8 spans cross storage words; gaps and the final entries
    # exercise scalar fallback. CPU-decoded operands supply the independent
    # attention enclosure; page permutation, padding and storage protection
    # use the existing complete operator contract.
    test_sealed_encoded_pages_preserve_attention_tree_and_physical_row_mapping(
        "kv", "auto", "cuda", cache_dtype, (2, 26, 37, native, splits, selection),
        (CacheGroupAxis.TOKENS, CacheGroupAxis.CHANNELS), token_group_size=16,
    )


@pytest.mark.parametrize("cache_dtype", [
    CacheDType.FP8, CacheDType.INT8, CacheDType.INT6, CacheDType.INT4,
])
@pytest.mark.parametrize("group", [8, 32])
@pytest.mark.parametrize("native", [torch.bfloat16, torch.float16])
def test_rejected_lookahead_reseals_replaced_group_from_committed_tail(cache_dtype, group, native):
    test_append_checkpoints_and_page_cow_preserve_native_tail_and_protected_prefix(
        native, "kv", 0, cache_dtype, group=group, retry_after_rejection=True,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("splits,selected", [(1, False), (4, True)])
def test_plain_shared_page_strides_preserve_independent_attention_enclosure(
    dtype, splits, selected,
):
    # BF16/FP16 serving consumes real outer page strides without an encoding
    # object. Padding poison and neighbor bytes remain
    # covered by the same independent attention witness as encoded formats.
    test_sealed_encoded_pages_preserve_attention_tree_and_physical_row_mapping(
        "", "auto", "triton", CacheDType.BF16,
        (2, 6, 64, dtype, splits, selected),
        (CacheGroupAxis.CHANNELS, CacheGroupAxis.CHANNELS),
        strided=True, encoded_binding=False,
    )
