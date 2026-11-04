"""Storage accounting and logical-group ownership, without a CUDA context."""

import itertools

import pytest
import torch


def test_slot_field_overlap_uses_actual_strides_and_preserves_resource_aliases():
    from cubic_serve.runtime.cache.quantization import (
        slot_storage_contiguous, storage_regions_overlap,
    )
    raw = torch.zeros((8, 128), dtype=torch.uint8)
    first, second = raw[:, :32], raw[:, 64:96]
    assert slot_storage_contiguous(first)
    assert not storage_regions_overlap(first, second)
    assert not storage_regions_overlap(second, first)
    assert storage_regions_overlap(first, raw[:, 16:48])
    assert storage_regions_overlap(first, raw.view(4, 256)[:, :64])
    assert not slot_storage_contiguous(raw[:, ::2])
    assert not slot_storage_contiguous(torch.as_strided(raw, (8, 32), (1, 1)))
    assert not storage_regions_overlap(first, first.clone())

from cubic_serve.runtime.cache.quantization import (
    CacheDType,
    CacheGroupAxis,
    CacheStorageSpec,
    GroupedCacheLayout,
    KVCacheStorageSpec,
)


def test_independent_kv_defaults_preserve_model_precision_and_rotation_is_boolean():
    original = KVCacheStorageSpec()
    for model, dtype in ((torch.bfloat16, CacheDType.BF16), (torch.float16, CacheDType.FP16)):
        resolved = original.resolve(model)
        assert resolved.key.dtype is resolved.value.dtype is dtype
        assert resolved.key.axis is CacheGroupAxis.TOKENS
        assert resolved.value.axis is CacheGroupAxis.CHANNELS
        assert not resolved.enable_rotation
    assert original.key.dtype is CacheDType.AUTO
    mixed = KVCacheStorageSpec(CacheStorageSpec(CacheDType.INT6, CacheGroupAxis.TOKENS))
    assert mixed.resolve(torch.float16).key.dtype is CacheDType.INT6
    assert mixed.resolve(torch.float16).value.dtype is CacheDType.FP16
    with pytest.raises(TypeError, match="boolean"):
        KVCacheStorageSpec(enable_rotation="false")
    with pytest.raises(ValueError, match="model dtype"):
        original.resolve(torch.float32)


@pytest.mark.parametrize("axis", list(CacheGroupAxis))
@pytest.mark.parametrize(
    "bits,dtype",
    [(8, CacheDType.FP8), (8, CacheDType.INT8), (6, CacheDType.INT6), (4, CacheDType.INT4)],
)
def test_accounting_includes_group_padding_real_int6_packing_and_scales(axis, bits, dtype):
    # Odd extents expose accidental head mixing and silently padded INT6.
    layout = GroupedCacheLayout(CacheStorageSpec(dtype, axis, 7), 9, 3, 11)
    groups = 3 * 11 * 2 if axis is CacheGroupAxis.TOKENS else 9 * 3 * 2
    assert layout.group_count == groups
    assert layout.payload_bytes == groups * ((7 * bits + 7) // 8)
    assert layout.scale_bytes == groups * 4
    assert layout.storage_bytes == layout.payload_bytes + layout.scale_bytes


@pytest.mark.parametrize("axis", list(CacheGroupAxis))
def test_coordinate_mapping_has_exact_membership_and_never_crosses_heads(axis):
    layout = GroupedCacheLayout(CacheStorageSpec(CacheDType.INT4, axis, 4), 7, 3, 9)
    seen = set()
    owners = {}
    for t, h, d in itertools.product(range(7), range(3), range(9)):
        group, member = layout.group_location(t, h, d)
        assert (group, member) not in seen
        seen.add((group, member))
        identity = (h, d, t // 4) if axis is CacheGroupAxis.TOKENS else (t, h, d // 4)
        assert owners.setdefault(group, identity) == identity
    assert len(seen) == 7 * 3 * 9
    for index in ((-1, 0, 0), (7, 0, 0), (0, 3, 0), (0, 0, 9), (True, 0, 0)):
        with pytest.raises(IndexError):
            layout.group_location(*index)


@pytest.mark.parametrize("dtype", [CacheDType.BF16, CacheDType.FP16])
def test_unquantized_storage_has_no_scale_or_group_padding_charge(dtype):
    layout = GroupedCacheLayout(CacheStorageSpec(dtype, CacheGroupAxis.TOKENS, 32), 7, 3, 9)
    assert layout.storage_bytes == 7 * 3 * 9 * 2
    assert layout.scale_bytes == 0


def test_invalid_or_unresolved_storage_fails_before_allocation():
    with pytest.raises(ValueError, match="resolved"):
        GroupedCacheLayout(CacheStorageSpec(CacheDType.AUTO, CacheGroupAxis.TOKENS), 1, 1, 1)
    for size in (True, 0, -1, 3.5):
        with pytest.raises(ValueError):
            CacheStorageSpec(CacheDType.INT4, CacheGroupAxis.TOKENS, size)


@pytest.mark.parametrize("key_dtype", [CacheDType.BF16, CacheDType.FP8, CacheDType.INT6])
@pytest.mark.parametrize("value_dtype", [CacheDType.BF16, CacheDType.INT8, CacheDType.INT4])
def test_mixed_arena_allocations_match_capacity_formula_including_scales(key_dtype, value_dtype):
    from cubic_serve.runtime.cache import (
        CacheArena,
        CacheArenaSpec,
        KVLayout,
        StateSlotSpec,
        estimate_cache_arena_bytes,
    )

    layout = KVLayout(
        (0, 3),
        32,
        3,
        37,
        storage=KVCacheStorageSpec(
            CacheStorageSpec(key_dtype, CacheGroupAxis.TOKENS),
            CacheStorageSpec(value_dtype, CacheGroupAxis.CHANNELS),
        ),
    )
    # The same allocator owns scale fields, protected padding and named state.
    spec = CacheArenaSpec(
        "mixed",
        5,
        3,
        (StateSlotSpec("tail", torch.bfloat16, (2, 32, 3, 37)),),
        scratch_capacity=2,
        scratch_arena_ids=("graph:0", "graph:1"),
    )
    arena = CacheArena(layout=layout, spec=spec, rank_id=0, device="cpu")
    actual = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            arena.key,
            arena.value,
            arena.written,
            *arena.page_fields.values(),
            *arena.states.values(),
        )
    )
    assert estimate_cache_arena_bytes(layout, spec, allocation_alignment_bytes=1) == actual
    assert arena.backed_bytes == actual
    aligned = sum(
        ((tensor.numel() * tensor.element_size() + 255) // 256) * 256
        for tensor in (
            arena.key,
            arena.value,
            arena.written,
            *arena.page_fields.values(),
            *arena.states.values(),
        )
    )
    assert estimate_cache_arena_bytes(layout, spec, allocation_alignment_bytes=256) == aligned
    assert arena.key.shape == (9, *layout.storage_shape(key=True))
    assert arena.value.shape == (9, *layout.storage_shape(key=False))
    assert layout.page_shape == (2, 32, 3, 37)  # Logical geometry never becomes byte geometry.
    for key, dtype in ((True, key_dtype), (False, value_dtype)):
        name = "kv_key_scales" if key else "kv_value_scales"
        assert (name in arena.page_fields) == dtype.quantized


def test_column_group_page_boundary_is_explicit_and_layout_replace_is_idempotent():
    from dataclasses import replace

    from cubic_serve.runtime.cache import KVLayout

    storage = KVCacheStorageSpec(CacheStorageSpec(CacheDType.FP8, CacheGroupAxis.TOKENS))
    with pytest.raises(ValueError, match="divide"):
        KVLayout((0,), 16, 1, 8, storage=storage)
    layout = KVLayout((0,), 32, 1, 8, storage=storage)
    assert replace(layout) == layout


def test_column_tails_use_common_named_state_and_page_alignment_is_geometric():
    from cubic_serve.runtime.cache import KVLayout

    assert KVCacheStorageSpec().column_group_alignment == 1
    storage = KVCacheStorageSpec(
        CacheStorageSpec(CacheDType.FP8, CacheGroupAxis.TOKENS, 8),
        CacheStorageSpec(CacheDType.INT6, CacheGroupAxis.TOKENS, 12),
    )
    assert storage.column_group_alignment == 24
    layout = KVLayout((1, 5), 48, 3, 37, dtype=torch.float16, storage=storage)
    states = layout.quantization_state_specs
    assert [spec.state_id for spec in states] == ["kv_key_tail", "kv_value_tail"]
    assert [spec.shape for spec in states] == [(2, 8, 3, 37), (2, 12, 3, 37)]
    assert all(spec.dtype is torch.float16 and spec.initial_value == 0 for spec in states)


@pytest.mark.parametrize("axis", list(CacheGroupAxis))
def test_physical_group_authority_requires_all_column_members(axis):
    grouped = GroupedCacheLayout(CacheStorageSpec(CacheDType.INT6, axis, 4), 8, 2, 7)
    for tokens in ((3,), (0, 1, 3), (0, 1, 2, 3), (4, 5, 6, 7)):
        touched = {grouped.group_location(t, h, d)[0]
                   for t, h, d in itertools.product(tokens, range(2), range(7))}
        assert set(grouped.groups_for_tokens(tokens)) == touched
        complete = set(grouped.groups_for_tokens(tokens, complete=True))
        if axis is CacheGroupAxis.CHANNELS or len(tokens) == 4:
            assert complete == touched
        else:
            assert not complete
    for bad in ((1, 0), (0, 0), (True,), (-1,), (8,)):
        with pytest.raises(ValueError, match="offsets"):
            grouped.groups_for_tokens(bad)


@pytest.mark.parametrize("dtype", [CacheDType.FP8, CacheDType.INT8,
                                  CacheDType.INT6, CacheDType.INT4])
@pytest.mark.parametrize("axis", list(CacheGroupAxis))
def test_code_validity_is_independent_of_decode_and_rejects_reserved_and_padding(dtype, axis):
    from cubic_serve.executor.runner import ReferenceExecutionContext
    from cubic_serve.ops.eager.kv_cache import encode_cache_reference

    grouped = GroupedCacheLayout(CacheStorageSpec(dtype, axis, 4), 7, 2, 7)
    values = torch.randn(grouped.logical_shape, generator=torch.Generator().manual_seed(104))
    with ReferenceExecutionContext():
        encoded = encode_cache_reference(values, grouped)
    assert bool(grouped.valid_code_groups(encoded.payload).all())
    invalid = encoded.payload.clone()
    # First member starts at bit zero, independently of the format's packing width.
    endpoint = 127 if dtype is CacheDType.FP8 else 1 << (dtype.bits - 1)
    invalid[0, 0] = (int(invalid[0, 0]) & ~((1 << dtype.bits) - 1)) | endpoint
    assert not bool(grouped.valid_code_groups(invalid)[0])
    invalid = encoded.payload.clone()
    last_group = grouped.group_shape[-1] - 1
    member = 3  # Both token/channel extents have one storage-only padded member.
    byte, shift = divmod(member * dtype.bits, 8)
    invalid[last_group, byte] |= 1 << shift
    assert not bool(grouped.valid_code_groups(invalid)[last_group])


@pytest.mark.parametrize("dtype,group", [(CacheDType.INT6, 7), (CacheDType.INT4, 3)])
@pytest.mark.parametrize("axis", list(CacheGroupAxis))
def test_encoded_groups_reject_nonzero_unused_bits_in_final_byte(dtype, group, axis):
    grouped = GroupedCacheLayout(CacheStorageSpec(dtype, axis, group), 7, 2, 7)
    payload = torch.zeros((2, grouped.group_count, grouped.packed_group_bytes), dtype=torch.uint8)
    assert bool(grouped.valid_code_groups(payload).all())
    payload[1, 0, -1] |= 1 << (group * dtype.bits % 8)
    valid = grouped.valid_code_groups(payload)
    assert bool(valid[0].all())
    assert not bool(valid[1, 0])
    assert bool(valid[1, 1:].all())


def _encoded_prefix_fixture(dtype):
    from cubic_serve.runtime.cache import KVLayout
    from cubic_serve.runtime.cache.pages import KVPageSnapshot
    from cubic_serve.runtime.cache.prefix import seal_prefix_snapshot

    storage = KVCacheStorageSpec(
        CacheStorageSpec(dtype, CacheGroupAxis.TOKENS, 4),
        CacheStorageSpec(dtype, CacheGroupAxis.CHANNELS, 4),
    )
    layout = KVLayout((0,), 8, 1, 7, storage=storage)
    key = torch.zeros(layout.storage_shape(key=True), dtype=torch.uint8)
    value = torch.zeros(layout.storage_shape(key=False), dtype=torch.uint8)
    fields = {spec.field_id: torch.ones(spec.shape, dtype=spec.dtype)
              for spec in layout.page_fields}
    page = KVPageSnapshot(3, 0, "owner", 0, key, value,
                          torch.ones((1, 8), dtype=torch.bool), fields)
    snapshot = seal_prefix_snapshot(
        snapshot_id="prefix", prefix_key="same", source_rank=0,
        source_owner_id="owner", source_revision=0, kv_layout=layout,
        physical_block_boundary=8, arithmetic_state_commit_boundary=8,
        state_commit_alignment=1, replay_tokens=(), pages=(page,), recurrent_states=(),
    )
    return layout, snapshot


def test_prefix_identity_includes_format_even_for_identical_code_bytes_and_scales():
    _, fp8 = _encoded_prefix_fixture(CacheDType.FP8)
    _, int8 = _encoded_prefix_fixture(CacheDType.INT8)
    assert torch.equal(fp8.pages[0].key, int8.pages[0].key)
    assert fp8.compute_digest() != int8.compute_digest()
    assert fp8.compute_semantic_digest() != int8.compute_semantic_digest()
    assert fp8.clone().compute_semantic_digest() == fp8.compute_semantic_digest()


def test_prefix_replay_cannot_resume_inside_a_lossy_closed_column_group():
    from dataclasses import replace

    from cubic_serve.runtime.cache.prefix import seal_recurrent_state_snapshot

    _, snapshot = _encoded_prefix_fixture(CacheDType.FP8)
    state = seal_recurrent_state_snapshot(
        state_id="arithmetic", owner_rank=0, owner_id="owner", revision=0,
        position=3, value=torch.zeros(1), valid=True,
    )
    unaligned = replace(snapshot, arithmetic_state_commit_boundary=3,
                        replay_tokens=(1,) * 5, recurrent_states=(state,),
                        state_valid_mask=(True,))
    unaligned = replace(unaligned, content_digest=unaligned.compute_digest())
    with pytest.raises(ValueError, match="closed column"):
        unaligned.validate()
    state = replace(state, position=4)
    state = replace(state, content_digest=state.compute_digest())
    aligned = replace(unaligned, arithmetic_state_commit_boundary=4,
                      replay_tokens=(1,) * 4, recurrent_states=(state,))
    aligned = replace(aligned, content_digest=aligned.compute_digest())
    aligned.validate()


def test_physical_page_validation_distinguishes_native_tail_and_sealed_codes():
    layout, snapshot = _encoded_prefix_fixture(CacheDType.INT6)
    page = snapshot.pages[0].clone()
    page.written[:, 3:] = False
    page.page_fields["kv_key_scales"].zero_()  # Three K rows remain in the native tail.
    layout.validate_page_storage(page.key, page.value, page.written, page.page_fields, device="cpu")
    page.written[:, 3] = True
    with pytest.raises(ValueError, match="positive scales"):
        layout.validate_page_storage(page.key, page.value, page.written,
                                     page.page_fields, device="cpu")
    with pytest.raises(ValueError, match="fully written"):
        layout.validate_page_storage(page.key, page.value, page.written,
                                     page.page_fields, device="cpu", require_full=True)


def test_mixed_encoded_qualification_images_preserve_physical_storage_and_prefix_restore():
    from cubic_serve.runtime.cache import (CacheArena, CacheArenaSpec, CacheStateStore,
                                          QualificationOwnerImage)

    layout, snapshot = _encoded_prefix_fixture(CacheDType.INT6)
    page = snapshot.pages[0]
    spec = CacheArenaSpec("encoded", 4, 0, ())
    arena = CacheArena(spec, layout, "cpu", rank_id=0)
    store = CacheStateStore(layout, rank_id=0, arena=arena)
    image = QualificationOwnerImage(
        "owner", (3,), page.key[None], page.value[None], page.written[None], (),
        page_fields=tuple((name, value[None]) for name, value in page.page_fields.items()),
    )
    lease = store.provision_qualification_owner(image)
    actual = store.snapshot_qualification_owner(lease)
    assert torch.equal(actual.key, image.key)
    assert torch.equal(actual.value, image.value)
    with pytest.raises(ValueError, match="native-tail visibility"):
        store.read_kv("owner", 3, 0, (0,))
    # The logical write API must never treat packed bytes as native floating rows.
    region = store.kv_region("owner", 3, 0, (0,))
    txn = store.begin(store.plan("write", "owner", writes=(region,)))
    payload = torch.zeros((1, 1, 7), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="group storage authority"):
        txn.write_kv(3, 0, (0,), payload, payload)
    txn.abort()
    detached = CacheStateStore(layout, rank_id=0)
    restored = detached.restore_prefix(snapshot, expected_prefix_key="same", owner_id="copy")
    assert len(restored.source_to_restored_pages) == 1
