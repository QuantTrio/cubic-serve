"""Independent byte-occupancy and lifecycle witnesses for shared cache ownership."""

import os
import random

import pytest
import torch

from cubic_serve.runtime.cache.pages import (
    ArenaExhausted, ArenaSlotReleaseError, StaleArenaSlot,
)
from cubic_serve.runtime.cache.shared_pool import (
    SharedCacheField, SharedCacheLayout, SharedCachePool, SharedCacheFreeSlots,
    SharedCacheStorage, shared_reservation_fits,
)


def _pool(**geometry):
    return SharedCachePool(pool_id="test", **geometry)


def test_normal_target_and_provider_capture_binds_only_private_scratch(monkeypatch):
    from types import SimpleNamespace
    from cubic_serve.runtime.quantized_graph import _prepare_capture_inputs
    from cubic_serve.runtime import quantized_qualification
    def forbidden(*args, **kwargs):
        raise AssertionError("normal startup executed an engineering corpus")
    monkeypatch.setattr(quantized_qualification, "eager_snapshots", forbidden)
    calls = []
    for role in ("target", "provider"):
        stage = SimpleNamespace(captured_m=1024, resource=role,
                                prepare_scratch=lambda r: calls.append(r))
        assert _prepare_capture_inputs(stage, (1, 512, 1024),
                                       SimpleNamespace(recording=False)) is None
        assert stage.bucket_logical_ms[0] == 513
        assert stage.bucket_logical_ms[-1] == 1024
    assert calls == ["target", "provider"]


@pytest.mark.parametrize("geometry", [
    dict(budget_bytes=1009, page_bytes=31, state_bytes=71,
         rank_geometries=((1301, 37, 43),)),
    dict(budget_bytes=503, page_bytes=13, state_bytes=0,
         rank_geometries=((1009, 31, 71),)),
])
@pytest.mark.parametrize("states,pages", [(1, 1), (2, 2), (3, 1), (4, 3)])
def test_cold_owner_bound_matches_exact_all_rank_resource_grants(geometry, states, pages):
    pool = _pool(**geometry)
    limit = pool.cold_owner_capacity(state_slots=states, pages=pages)
    state_slots, page_slots = SharedCacheFreeSlots(pool, "state"), SharedCacheFreeSlots(pool, "page")
    for _ in range(limit * states):
        state_slots.allocate()
    for _ in range(limit * pages):
        page_slots.allocate()
    assert not shared_reservation_fits(((state_slots, states), (page_slots, pages)))
    for budget, page_bytes, state_bytes in pool.rank_geometries:
        owned = set()
        offset = budget - pool.state_capacity * state_bytes
        for i in range(pool.state_capacity - limit * states, pool.state_capacity):
            owned.update(range(offset + i * state_bytes, offset + (i + 1) * state_bytes))
        for i in range(limit * pages):
            page = set(range(i * page_bytes, (i + 1) * page_bytes))
            assert not page & owned
            owned |= page
        assert len(owned) <= budget


@pytest.mark.parametrize("owners,maximum", [(1, 4), (32, 128), (128, 512), (256, 1024)])
@pytest.mark.parametrize("stateful", [False, True])
def test_graph_bootstrap_keeps_real_owners_and_constant_padding_state(owners, maximum, stateful):
    from cubic_serve.runtime.quantized_graph import _bootstrap_cache_arenas
    from cubic_serve.runtime.cache import CacheArenaSpec, KVLayout, StateSlotSpec

    schemas = tuple((CacheArenaSpec(str(i), maximum + 1, owners,
        (StateSlotSpec("matrix", torch.bfloat16, (4, 8)),) if stateful else (),
        scratch_capacity=maximum * (i + 1)), KVLayout((i,), 2, 1, 4 + i))
        for i in range(2))
    storage, arenas = _bootstrap_cache_arenas(schemas, "cpu", 0)
    assert len(arenas) == 2
    for arena in arenas:
        assert arena.shared_storage is storage
        assert arena.spec.state_slot_capacity == owners
        assert set(arena.padding_state_slots(maximum, "default")) == {owners}
        if stateful:
            assert arena.states["matrix"].shape[0] == owners + 1
    page_bytes = storage.page_layout.entry_bytes
    state_bytes = 0 if storage.state_layout is None else storage.state_layout.entry_bytes
    # Independently locate owned pages and the real/padding state ranges.
    assert storage.allocator.state_offset_bytes == (maximum + 1) * page_bytes
    for i, arena in enumerate(arenas):
        arena.key[:maximum + 1].fill_(17 + i)
        if stateful:
            arena.states["matrix"].fill_(23 + i)
        assert (arena.key[:maximum + 1] == 17 + i).all()
    # No M * matrix_bytes padding slab remains. Only ordinary KV scratch rows
    # scale with the graph width; rounding is less than one page.
    upper = ((maximum + 1 + 2 * maximum) * page_bytes
             + (owners + int(stateful)) * state_bytes)
    assert storage.backing.numel() < upper + page_bytes
    assert storage.backing.numel() >= upper


def test_released_states_become_attention_capacity_without_moving_live_pages():
    pool = _pool(budget_bytes=1024, page_bytes=32, state_bytes=256)
    state = pool.allocate("state")
    pages = [pool.allocate("page") for _ in range(24)]
    with pytest.raises(ArenaExhausted):
        pool.allocate("page")
    pool.release(state)
    pages.extend(pool.allocate("page") for _ in range(8))
    assert [page.index for page in pages] == list(range(32))
    assert pool.live_bytes == 1024
    assert pool.unassigned_bytes == 0
    for page in pages:
        pool.validate(page)


def test_page_preference_observes_newly_unblocked_lower_indices():
    pool = _pool(budget_bytes=512, page_bytes=16, state_bytes=64)
    state = pool.reserve_at("state", 0)
    pages = SharedCacheFreeSlots(pool, "page")
    assert pages.allocate() == 4
    assert pages.allocate() == 5
    pool.release(state)
    assert pages.allocate() == 0
    assert pages.allocate() == 1
    pages.release(0)
    assert pages.allocate() == 0
    assert pages.allocate() == 2


def test_field_family_grants_match_independent_all_rank_intervals_after_holes_and_references():
    geometries = ((1009, 13, 71), (1101, 17, 43), (1403, 19, 59))
    pool = _pool(budget_bytes=1009, page_bytes=13, state_bytes=71,
                 rank_geometries=geometries[1:])
    families = (SharedCacheFreeSlots(pool, "page"), SharedCacheFreeSlots(pool, "page"),
                SharedCacheFreeSlots(pool, "state"))
    owned = [set(), set(), set()]
    capacities = (min(b // p for b, p, _ in geometries),
                  min(b // s for b, _, s in geometries))

    def eligible(family):
        is_page = family < 2
        candidates = range(capacities[0]) if is_page else range(capacities[1] - 1, -1, -1)
        other = owned[2] if is_page else owned[0] | owned[1]
        result = []
        for index in candidates:
            if index in owned[family]:
                continue
            safe = True
            for budget, page_bytes, state_bytes in geometries:
                origin = budget - capacities[1] * state_bytes
                lo = index * page_bytes if is_page else origin + index * state_bytes
                hi = lo + (page_bytes if is_page else state_bytes)
                for opposite in other:
                    start = origin + opposite * state_bytes if is_page else opposite * page_bytes
                    stop = start + (state_bytes if is_page else page_bytes)
                    safe &= hi <= start or stop <= lo
            if safe:
                result.append(index)
        return result

    rng = random.Random(7951)
    for _ in range(1500):
        family = rng.randrange(3)
        expected = eligible(family)
        assert len(families[family]) == len(expected)
        if owned[family] and rng.random() < .48:
            index = rng.choice(sorted(owned[family]))
            families[family].release(index)
            owned[family].remove(index)
        elif expected:
            index = families[family].allocate()
            assert index == expected[0]
            owned[family].add(index)
        else:
            with pytest.raises(ArenaExhausted):
                families[family].allocate()
        assert pool.live_bytes == len(owned[0] | owned[1]) * 13 + len(owned[2]) * 71
        for slots in families:
            for ticket in slots._tickets.values():
                pool.validate(ticket)


@pytest.mark.parametrize("page_bytes,state_bytes", [(13, 71), (64, 17), (7, 113)])
def test_random_transitions_match_independent_byte_ownership(page_bytes, state_bytes):
    rng = random.Random(1729)
    pool = _pool(budget_bytes=1009, page_bytes=page_bytes, state_bytes=state_bytes)
    live = []
    for _ in range(2000):
        if live and rng.random() < .45:
            slot = live.pop(rng.randrange(len(live)))
            pool.release(slot)
            with pytest.raises(StaleArenaSlot):
                pool.validate(slot)
        else:
            try:
                live.append(pool.allocate(rng.choice(("page", "state"))))
            except ArenaExhausted:
                pass
        owner = {}
        for slot in live:
            pool.validate(slot)
            size = page_bytes if slot.kind == "page" else state_bytes
            offset = pool.state_offset_bytes if slot.kind == "state" else 0
            for byte in range(offset + slot.index * size, offset + (slot.index + 1) * size):
                assert 0 <= byte < pool.budget_bytes
                assert byte not in owner
                owner[byte] = slot
        assert pool.live_bytes == len(owner)
        assert pool.unassigned_bytes >= 0
    for slot in live:
        pool.release(slot)
    assert pool.live_bytes == 0
    assert pool.unassigned_bytes == pool.budget_bytes


def test_attention_uses_remainder_when_no_state_is_owned():
    pool = _pool(budget_bytes=100, page_bytes=10, state_bytes=31)
    pages = [pool.allocate("page") for _ in range(10)]
    assert pool.live_bytes == 100
    with pytest.raises(ArenaExhausted):
        pool.allocate("state")
    for page in pages[-4:]:
        pool.release(page)
    assert pool.allocate("state").index == 2


def test_reuse_changes_generation_and_foreign_or_double_release_fails_closed():
    pool = _pool(budget_bytes=256, page_bytes=16, state_bytes=64)
    original = pool.allocate("state")
    pool.release(original)
    replacement = pool.allocate("state")
    assert replacement.index == original.index
    assert replacement.generation > original.generation
    with pytest.raises(ArenaSlotReleaseError):
        pool.release(original)
    other = SharedCachePool(pool_id="other", budget_bytes=256, page_bytes=16, state_bytes=64)
    with pytest.raises(StaleArenaSlot):
        other.validate(replacement)


@pytest.mark.parametrize("kind", ["page", "state"])
def test_shared_holds_reject_duplicate_release_without_retiring_another_family(kind):
    from copy import copy

    pool = _pool(budget_bytes=256, page_bytes=64, state_bytes=128)
    first = pool.reserve_at(kind, 0)
    second = pool.reserve_at(kind, 0)
    assert first.hold_id != second.hold_id
    pool.release(first)
    live, blocked = pool.live_bits.copy(), pool.blocked_bits.copy()
    with pytest.raises(StaleArenaSlot):
        pool.validate(first)
    with pytest.raises(ArenaSlotReleaseError):
        pool.release(copy(first))
    assert pool.live_bits == live and pool.blocked_bits == blocked
    pool.validate(second)
    opposite = "state" if kind == "page" else "page"
    with pytest.raises(ArenaExhausted):
        pool.reserve_at(opposite, 0)
    pool.release(second)
    assert pool.live_bytes == 0
    with pytest.raises(ArenaSlotReleaseError):
        pool.release(copy(second))
    pool.reserve_at(opposite, 0)


def test_interior_holes_are_reused_without_moving_live_storage():
    pool = _pool(budget_bytes=256, page_bytes=16, state_bytes=64)
    pages = [pool.allocate("page") for _ in range(4)]
    states = [pool.allocate("state") for _ in range(2)]
    pool.release(pages[1])
    pool.release(states[0])
    assert pool.allocate("page").index == pages[1].index
    assert pool.allocate("state").index == states[0].index
    pool.validate(pages[-1])
    pool.validate(states[-1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int64])
def test_fixed_views_preserve_live_fields_and_exact_integer_bits(dtype):
    pages = SharedCacheLayout((SharedCacheField("codes", torch.uint8, (13,)),
                               SharedCacheField("scales", torch.float32, (2,))))
    states = SharedCacheLayout((SharedCacheField("matrix", dtype, (3, 4)),
                                SharedCacheField("history", torch.int64, (2,))))
    raw = torch.zeros(4096, dtype=torch.uint8)
    pool = _pool(budget_bytes=raw.numel(), page_bytes=pages.entry_bytes,
                 state_bytes=states.entry_bytes)
    page_views = pages.views(raw, pool.page_capacity)
    state_views = states.views(raw, pool.state_capacity, offset=pool.state_offset_bytes)
    identities = {name: tensor.data_ptr() for name, tensor in state_views.items()}
    page = pool.allocate("page")
    state = pool.allocate("state")
    page_views["codes"][page.index].fill_(173)
    page_views["scales"][page.index].fill_(0.125)
    state_views["matrix"][state.index].fill_(7)
    state_views["history"][state.index].fill_(2**54 + 1)
    assert (page_views["codes"][page.index] == 173).all()
    assert (page_views["scales"][page.index] == 0.125).all()
    assert (state_views["matrix"][state.index] == 7).all()
    assert state_views["history"][state.index].tolist() == [2**54 + 1] * 2
    assert {name: tensor.data_ptr() for name, tensor in state_views.items()} == identities
    for tensor in (*page_views.values(), *state_views.values()):
        assert tensor[0].is_contiguous()
        assert tensor.untyped_storage().data_ptr() == raw.data_ptr()
    pool.release(state)
    while True:
        try:
            slot = pool.allocate("page")
        except ArenaExhausted:
            break
        page_views["codes"][slot.index].fill_(17)
    assert (page_views["codes"][page.index] == 173).all()
    assert (page_views["scales"][page.index] == 0.125).all()


def test_page_families_share_bundles_and_joint_admission_does_not_mutate():
    pool = _pool(budget_bytes=1024, page_bytes=64, state_bytes=256)
    target, provider = SharedCacheFreeSlots(pool, "page"), SharedCacheFreeSlots(pool, "page")
    states = SharedCacheFreeSlots(pool, "state")
    assert not shared_reservation_fits(((target, 12), (states, 2)))
    assert pool.live_bytes == 0
    assert target.candidates(3) == (0, 1, 2)
    assert states.candidates(2) == (3, 2)
    assert target.allocate() == provider.allocate() == 0
    assert pool.live_bytes == 64
    target.release(0)
    assert pool.live_bytes == 64
    provider.release(0)
    assert pool.live_bytes == 0
    leased = [states.allocate() for _ in range(3)]
    assert len(target) == 4
    for index in leased:
        states.release(index)
    assert len(target) == 16


def test_shared_backing_prices_schema_and_keeps_one_padding_state_per_arena():
    from cubic_serve.runtime.cache import CacheArenaSpec, KVLayout, StateSlotSpec
    schema = CacheArenaSpec("target", 2, 2, (
        StateSlotSpec("history", torch.int64, (2,), initial_value=31),
        StateSlotSpec("matrix", torch.bfloat16, (2, 4)),
    ), scratch_capacity=128)
    layout = KVLayout((0,), 2, 1, 4)
    storage = SharedCacheStorage(((schema, layout),), budget_bytes=4096, device="cpu")
    pages, states = storage.views(0)
    assert states["matrix"].shape[0] == storage.state_capacity + 1
    assert storage.arena_specs[0].state_scratch_capacity_per_arena == 1
    assert pages["key"].shape[0] == storage.page_scratch_start + 128
    assert (states["history"] == 31).all()
    assert states["history"].stride(0) * states["history"].element_size() == storage.state_layout.entry_bytes
    assert storage.backing.numel() == SharedCacheStorage.allocation_bytes(4096, ((schema, layout),))
    assert pages["key"].untyped_storage().data_ptr() == states["matrix"].untyped_storage().data_ptr()


@pytest.mark.parametrize("kind", ["int4", "int6", "int8", "fp8"])
def test_encoded_padding_pages_are_constant_and_exclusive_between_leases(kind):
    from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout
    from cubic_serve.runtime.cache.quantization import (
        CacheDType, CacheGroupAxis, CacheStorageSpec, KVCacheStorageSpec,
    )
    storage_spec = KVCacheStorageSpec(
        CacheStorageSpec(CacheDType(kind), CacheGroupAxis.TOKENS, 32),
        CacheStorageSpec(CacheDType(kind), CacheGroupAxis.CHANNELS, 32),
    )
    layout = KVLayout((0,), 32, 1, 32, storage=storage_spec)
    sizes = []
    for rows in (8, 1024):
        schema = CacheArenaSpec("encoded", 1, 1, layout.quantization_state_specs,
                                scratch_capacity=rows, scratch_arena_ids=("a", "b"))
        storage = SharedCacheStorage(((schema, layout),), budget_bytes=1 << 18, device="cpu")
        arena = CacheArena(storage.arena_specs[0], layout, "cpu", rank_id=0, shared_storage=storage)
        a, b = (arena.padding_page_slots(rows, lane) for lane in ("a", "b"))
        assert len(a) == rows and len(b) == rows
        assert len(set(a)) == len(set(b)) == 1 and set(a).isdisjoint(b)
        assert arena.key.shape[0] == storage.page_scratch_start + 2
        assert min(a) >= storage.page_capacity and max(b) < arena.key.shape[0]
        sizes.append(storage.backing.numel())
    assert sizes[0] == sizes[1]


@pytest.mark.skipif(os.environ.get('CUBIC_TEST_QUANT_CUDA') != '1',
                    reason='explicit compact-padding write-set proof')
@pytest.mark.parametrize("kind", ["int4", "int6", "int8", "fp8"])
def test_compact_encoded_padding_does_not_write_pages_or_state(kind):
    from cubic_serve.ops.cuda.cache_quantization import PagedKVGraphBinding
    from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout
    from cubic_serve.runtime.cache.quantization import (
        CacheDType, CacheGroupAxis, CacheStorageSpec, KVCacheStorageSpec,
    )
    rows, dim = 8, 32
    policy = KVCacheStorageSpec(
        CacheStorageSpec(CacheDType(kind), CacheGroupAxis.TOKENS, 32),
        CacheStorageSpec(CacheDType(kind), CacheGroupAxis.CHANNELS, 32),
    )
    layout = KVLayout((0,), 32, 1, dim, storage=policy)
    schema = CacheArenaSpec("encoded", 1, 1, layout.quantization_state_specs,
                            scratch_capacity=rows)
    storage = SharedCacheStorage(((schema, layout),), budget_bytes=1 << 18, device="cuda")
    arena = CacheArena(storage.arena_specs[0], layout, "cuda", rank_id=0, shared_storage=storage)
    owner = storage.allocator.allocate("page").index
    state_ids = [storage.allocator.allocate("state").index for _ in range(3)]
    padding = arena.padding_page_slots(rows, "default")[0]
    dummy_state = arena.padding_state_slots(rows, "default")[0]
    positions = torch.zeros(rows, device="cuda", dtype=torch.int64)
    positions[1] = 1
    segments = torch.zeros(rows, device="cuda", dtype=torch.int32)
    segments[0] = 2
    activity = torch.zeros_like(segments)
    sources = torch.full_like(segments, dummy_state)
    destinations = torch.full_like(segments, dummy_state)
    sources[:2] = state_ids[0]
    destinations[:2] = torch.tensor(state_ids[1:], device="cuda", dtype=torch.int32)
    pages = torch.full_like(segments, padding)
    pages[:2] = owner
    binding = PagedKVGraphBinding(arena, positions=positions, segments=segments,
                                  activity=activity, source_slots=sources,
                                  destination_slots=destinations)
    key = torch.full((rows, 1, dim), float('nan'), device="cuda", dtype=torch.bfloat16)
    value = torch.full_like(key, float('nan'))
    key[:2] = torch.arange(2 * dim, device="cuda").reshape(2, 1, dim).div(17).bfloat16()
    value[:2] = -key[:2]
    _, prepare, checkpoint = binding.bind_layer(key, value, layer=0, pages=pages)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.prepare_metadata()
        prepare.launch()
        checkpoint.launch()
    before = storage.backing.clone()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(before, storage.backing)
    # Positive control: a real segment includes its zero-start continuation.
    activity[:2] = 1
    graph.replay()
    torch.cuda.synchronize()
    allowed = torch.zeros(storage.backing.numel(), dtype=torch.bool)
    start = owner * storage.page_layout.entry_bytes
    allowed[start:start + storage.page_layout.entry_bytes] = True
    for slot in state_ids[1:]:
        start = storage.allocator.state_offset_bytes + slot * storage.state_layout.entry_bytes
        allowed[start:start + storage.state_layout.entry_bytes] = True
    actual = storage.backing.cpu()
    baseline = before.cpu()
    assert torch.equal(actual[~allowed], baseline[~allowed])
    assert not torch.equal(actual, baseline)
    assert arena.written[owner, 0, :2].cpu().tolist() == [True, True]
    expected = torch.zeros_like(arena.states['kv_key_tail'][state_ids[2], 0])
    expected[:2].copy_(key[:2])
    assert torch.equal(arena.states['kv_key_tail'][state_ids[2], 0], expected)
    graph.reset()


def test_shared_arena_reclaims_state_bytes_for_kv_with_stable_backing_and_generations():
    from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout, StateSlotSpec
    schema = CacheArenaSpec("target", 2, 2, (
        StateSlotSpec("history", torch.int64, (2,), initial_value=31),
        StateSlotSpec("matrix", torch.bfloat16, (16, 16)),
    ), scratch_capacity=128)
    layout = KVLayout((0,), 2, 1, 4)
    storage = SharedCacheStorage(((schema, layout),), budget_bytes=4096, device="cpu")
    arena = CacheArena(storage.arena_specs[0], layout, "cpu", rank_id=0,
                       shared_storage=storage)
    identity = arena.backing_identity
    matrix, value = arena.reserve_state("matrix")
    history, tokens = arena.reserve_state("history")
    assert matrix.index == history.index == storage.state_capacity - 1
    value.fill_(7)
    assert tokens.tolist() == [31, 31]
    pages = []
    while True:
        try:
            pages.append(arena.reserve_page(page_id=len(pages), owner_rank=0,
                                             owner_id="request", zero=True))
        except ArenaExhausted:
            break
    assert (value == 7).all()
    before = len(pages)
    arena.release_state("matrix", matrix)
    arena.release_state("history", history)
    with pytest.raises((StaleArenaSlot, ArenaSlotReleaseError)):
        arena.validate_state("matrix", matrix)
    while True:
        try:
            pages.append(arena.reserve_page(page_id=len(pages), owner_rank=0,
                                             owner_id="request", zero=True))
        except ArenaExhausted:
            break
    assert len(pages) > before
    assert len(pages) == storage.page_capacity
    assert arena.backing_identity == identity
    assert set(arena.padding_state_slots(128, "default")) == {storage.state_capacity}
    for page in pages:
        arena.release_page(page)


def test_topology_frontiers_obey_each_actual_rank_instead_of_maximum_shape_patches():
    pool = SharedCachePool(pool_id="topology", budget_bytes=1024, page_bytes=64,
                           state_bytes=64, rank_geometries=((1536, 32, 256),))
    assert pool.state_capacity == 6
    pages, states = SharedCacheFreeSlots(pool, "page"), SharedCacheFreeSlots(pool, "state")
    assert shared_reservation_fits(((pages, 8), (states, 5)))
    assert not shared_reservation_fits(((pages, 9), (states, 5)))
    for _ in range(5):
        states.allocate()
    assert len(pages) == 8


def test_interior_released_states_are_immediately_usable_by_attention():
    pool = _pool(budget_bytes=1024, page_bytes=32, state_bytes=256)
    states, pages = SharedCacheFreeSlots(pool, "state"), SharedCacheFreeSlots(pool, "page")
    owned = [states.allocate() for _ in range(4)]
    for index in owned[:3]:
        states.release(index)
    assert pool.unassigned_bytes == 768
    assert len(pages) == 24
    assert pages.candidates(24) == tuple(range(8, 32))
    assert shared_reservation_fits(((pages, 24),))
    for _ in range(24):
        pages.allocate()
    assert pool.live_bytes == 1024
    assert pool.live_bits["state"] == 1


@pytest.mark.parametrize("geometry", [(1009, 13, 71), (1009, 64, 17), (1013, 7, 113)])
def test_bitmap_admission_equals_independent_interval_oracle(geometry):
    budget, page_bytes, state_bytes = geometry
    pool = _pool(budget_bytes=budget, page_bytes=page_bytes, state_bytes=state_bytes)
    families = [SharedCacheFreeSlots(pool, "page"), SharedCacheFreeSlots(pool, "state")]
    rng = random.Random(8253)
    owned = [[], []]
    for _ in range(1000):
        which = rng.randrange(2)
        if owned[which] and rng.random() < .45:
            index = owned[which].pop(rng.randrange(len(owned[which])))
            families[which].release(index)
        elif len(families[which]):
            owned[which].append(families[which].allocate())
        for i, (size, offset) in enumerate(((page_bytes, 0), (state_bytes, pool.state_offset_bytes))):
            other_size = state_bytes if i == 0 else page_bytes
            other_offset = pool.state_offset_bytes if i == 0 else 0
            expected = []
            for j in range(pool.capacity(families[i].kind)):
                if j in owned[i]:
                    continue
                start, end = offset + j * size, offset + (j + 1) * size
                if not any(start < other_offset + (k + 1) * other_size and end > other_offset + k * other_size
                           for k in owned[1-i]):
                    expected.append(j)
            if i:
                expected.reverse()
            assert len(families[i]) == len(expected)
            assert families[i].candidates(len(expected)) == tuple(expected)


def test_page_journal_rolls_back_only_its_own_allocations_and_keeps_family_refs():
    from cubic_serve.runtime.cache.shared_pool import SharedCacheJournal
    pool=_pool(budget_bytes=1024,page_bytes=32,state_bytes=256)
    target,provider=SharedCacheFreeSlots(pool,'page'),SharedCacheFreeSlots(pool,'page')
    original=target.allocate()
    journal=SharedCacheJournal(target)
    temporary=journal.allocate()
    assert provider.allocate()==original
    assert provider.allocate()==temporary
    journal.finish(commit=False)
    assert pool.live_bytes==64
    assert target.candidates(1)==(temporary,)
    provider.release(temporary)
    assert pool.live_bytes==32
    kept=SharedCacheJournal(target);index=kept.allocate();kept.finish(commit=True)
    target.release(index);target.release(original);provider.release(original)
    assert pool.live_bytes==0


def test_worker_ownership_rejects_overlapping_bytes_and_unowned_rows_before_replay():
    from cubic_serve.runtime.cache.shared_pool import SharedCacheOwnership
    from cubic_serve.runtime.continuous_batch import BatchCommand,BatchRow
    pool=_pool(budget_bytes=1024,page_bytes=32,state_bytes=256)
    state=pool.allocate('state');page=pool.allocate('page')
    witness=SharedCacheOwnership(_pool(budget_bytes=1024,page_bytes=32,state_bytes=256))
    payload=SharedCacheOwnership.pack(b'command',pool)
    assert witness.unpack(payload)==b'command'
    row=BatchRow('r',17,0,state.index,state.index,page.index,page.index,(page.index,))
    witness.validate_command(BatchCommand(0,1,(row,)))
    from dataclasses import replace
    with pytest.raises(ValueError,match='unowned'):
        witness.validate_command(BatchCommand(0,1,(replace(row,state_source_slot=0),)))
    with pytest.raises(ValueError,match='unowned'):
        witness.validate_command(BatchCommand(0,1,(replace(row,page_table=(1,)),)))
    pool.live_bits['page']|=1<<31
    with pytest.raises(ValueError,match='overlap'):
        witness.unpack(SharedCacheOwnership.pack(b'command',pool))
    with pytest.raises(ValueError,match='lacks'):
        witness.unpack(b'command')


def test_mixed_rank_geometries_match_independent_byte_intersections_after_recycling():
    geometry=((1009,13,71),(1536,31,19),(512,11,0))
    pool=SharedCachePool(pool_id='ranks',budget_bytes=geometry[0][0],
        page_bytes=geometry[0][1],state_bytes=geometry[0][2],rank_geometries=geometry[1:])
    families=(SharedCacheFreeSlots(pool,'page'),SharedCacheFreeSlots(pool,'state'))
    live=[[],[]];rng=random.Random(1129)
    for _ in range(500):
        kind=rng.randrange(2)
        if live[kind] and rng.random()<.45:
            families[kind].release(live[kind].pop(rng.randrange(len(live[kind]))))
        elif len(families[kind]):live[kind].append(families[kind].allocate())
        for which,family in enumerate(families):
            expected=[]
            for index in range(pool.capacity(family.kind)):
                if index in live[which]:continue
                fits=True
                for budget,page,state in geometry:
                    origin=budget-pool.state_capacity*state
                    begin=(index*page if which==0 else origin+index*state)
                    end=begin+(page if which==0 else state)
                    if begin==end:continue
                    for other in live[1-which]:
                        a=(origin+other*state if which==0 else other*page)
                        b=a+(state if which==0 else page)
                        if a!=b and begin<b and end>a:fits=False
                if fits:expected.append(index)
            if which:expected.reverse()
            assert family.candidates(len(family))==tuple(expected)


def test_cache_ownership_persists_across_batch_and_resident_command_decode():
    from cubic_serve.runtime.cache.shared_pool import SharedCacheOwnership
    from cubic_serve.runtime.continuous_batch import BatchCommand,BatchRow,ContinueBatchCommand,serialize_worker_command,deserialize_worker_command
    pool=_pool(budget_bytes=1024,page_bytes=32,state_bytes=256)
    state=pool.allocate('state');page=pool.allocate('page')
    validator=SharedCacheOwnership(_pool(budget_bytes=1024,page_bytes=32,state_bytes=256))
    row=BatchRow('r',17,1,state.index,state.index,page.index,page.index,(page.index,),is_decode=True)
    commands=(BatchCommand(0,1,(row,)),ContinueBatchCommand(1,0,1,('r',)),BatchCommand(2,1,(row,)))
    for expected in commands:
        command=deserialize_worker_command(validator.unpack(SharedCacheOwnership.pack(serialize_worker_command(expected),pool)))
        validator.validate_command(command)
        assert command==expected
