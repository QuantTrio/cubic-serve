from dataclasses import replace

import pytest
import torch

from cubic_serve.runtime.cache import (
    CacheArena,
    CacheArenaSpec,
    CacheStateStore,
    KVLayout,
    QualificationOwnerImage,
    RankPrefixSource,
    estimate_cache_arena_bytes,
)
from cubic_serve.runtime.cache.pages import PageFieldSpec


def _fixture():
    layout = KVLayout(
        (0,),
        4,
        1,
        2,
        page_fields=(
            PageFieldSpec("index_keys", torch.bfloat16, (1, 2, 3)),
            PageFieldSpec("positions", torch.int64, (1, 2)),
        ),
    )
    spec = CacheArenaSpec("fields", 5, 0, (), scratch_capacity=2)
    arena = CacheArena(spec, layout, "cpu", rank_id=0)
    store = CacheStateStore(layout, rank_id=0, arena=arena)
    key = torch.ones(1, *layout.page_shape, dtype=torch.bfloat16)
    image = QualificationOwnerImage(
        "a",
        (7,),
        key,
        -key,
        torch.ones(1, 1, 4, dtype=torch.bool),
        (),
        page_fields=(
            ("index_keys", torch.arange(6).reshape(1, 1, 2, 3).to(torch.bfloat16)),
            ("positions", torch.full((1, 1, 2), 2**54 + 1, dtype=torch.int64)),
        ),
    )
    return arena, store, image


def _snapshot(store, owner, page):
    return store.capture_prefix(
        snapshot_id=f"snapshot:{owner}",
        prefix_key="same-prefix",
        owner_id=owner,
        page_ids=(page,),
        state_ids=(),
        physical_block_boundary=4,
        arithmetic_state_commit_boundary=4,
        state_commit_alignment=1,
        replay_tokens=(),
    )


def test_page_fields_budget_cow_prefix_restore_and_digest():
    arena, store, image = _fixture()
    tensors = (arena.key, arena.value, arena.written, *arena.page_fields.values())
    assert estimate_cache_arena_bytes(
        arena.layout, arena.spec, allocation_alignment_bytes=1
    ) == sum(t.numel() * t.element_size() for t in tensors)
    assert all(t.data_ptr() in arena.backing_identity for t in tensors)
    lease = store.provision_qualification_owner(image)
    for (_, actual), (_, expected) in zip(
        store.snapshot_qualification_owner(lease).page_fields, image.page_fields, strict=True
    ):
        assert torch.equal(actual, expected)
    snapshot = _snapshot(store, "a", 7)
    part = store.pin_prefix_part(RankPrefixSource(0, "a", 0, (7,), ()), token_boundary=4)
    restored = store.activate_prefix_restore(store.prepare_prefix_restore(part, "b"))
    other_page = restored.source_to_restored_pages[0][1]
    # Updating one KV row copies every auxiliary field to the new owned page.
    txn = store.begin(store.plan("update", "a", writes=(store.kv_region("a", 7, 0, (0,)),)))
    payload = torch.full((1, 1, 2), 9, dtype=torch.bfloat16)
    txn.write_kv(7, 0, (0,), payload, -payload)
    txn.verify()
    txn.commit()
    after = store.snapshot_qualification_owner(lease)
    assert torch.equal(after.value[0, 0, 0], -payload[0])
    for (_, actual), (_, expected) in zip(after.page_fields, image.page_fields, strict=True):
        assert torch.equal(actual, expected)
    assert (
        _snapshot(store, "b", other_page).compute_semantic_digest()
        == snapshot.compute_semantic_digest()
    )
    # Detached restoration also preserves fields and rejects schema/data tampering.
    detached = CacheStateStore(arena.layout, rank_id=0)
    restored = detached.restore_prefix(snapshot, expected_prefix_key="same-prefix", owner_id="c")
    page = restored.source_to_restored_pages[0][1]
    assert (
        _snapshot(detached, "c", page).compute_semantic_digest()
        == snapshot.compute_semantic_digest()
    )
    changed = snapshot.clone()
    changed.pages[0].page_fields["positions"][0, 0] += 1
    with pytest.raises(ValueError, match="digest mismatch"):
        detached.restore_prefix(changed, expected_prefix_key="same-prefix", owner_id="bad")
    missing = snapshot.clone()
    missing.pages[0].page_fields.pop("index_keys")
    with pytest.raises(ValueError, match="fields"):
        detached.restore_prefix(missing, expected_prefix_key="same-prefix", owner_id="missing")
    store.release_prefix_part(part)
    store.retire_qualification_owner(lease, expected_revision=1)
    # Reallocated physical storage must not leak prior compressed keys/positions.
    page = arena.reserve_page(page_id=99, owner_rank=0, owner_id="new", zero=True)
    assert all(not bool(value.any()) for value in page.page_fields.values())
    arena.release_page(page)


def test_page_field_schema_rejects_missing_wrong_dtype_and_nonfinite():
    _, store, image = _fixture()
    with pytest.raises(ValueError, match="fields"):
        store.provision_qualification_owner(replace(image, page_fields=()))
    for values in (
        (("index_keys", image.page_fields[0][1].float()), image.page_fields[1]),
        (
            ("index_keys", torch.full_like(image.page_fields[0][1], float("nan"))),
            image.page_fields[1],
        ),
    ):
        with pytest.raises(ValueError, match="page field"):
            store.provision_qualification_owner(replace(image, page_fields=values))


def test_new_owner_reset_preserves_schema_initial_values_and_other_owners():
    from cubic_serve.runtime.cache import StateSlotSpec

    layout = KVLayout((0,), 4, 1, 2, page_fields=(PageFieldSpec("index", torch.bfloat16, (2, 3)),))
    spec = CacheArenaSpec(
        "initial-state",
        3,
        3,
        (
            StateSlotSpec("history", torch.int64, (2,), initial_value=2**54 + 1),
            StateSlotSpec("recurrent", torch.float32, (2, 2)),
        ),
    )
    arena = CacheArena(spec, layout, "cpu", rank_id=0)
    assert (arena.states["history"] == 2**54 + 1).all()
    slot, history = arena.reserve_state("history")
    history.fill_(42)
    arena.release_state("history", slot)
    reused, history = arena.reserve_state("history")
    assert (history == 2**54 + 1).all()
    arena.release_state("history", reused)
    for tensor in (
        arena.key,
        arena.value,
        arena.written,
        *arena.page_fields.values(),
        *arena.states.values(),
    ):
        tensor.fill_(1)
    rows = torch.tensor([1], dtype=torch.int64)
    arena.reset_state_rows(rows)
    arena.reset_page_rows(rows)
    assert (arena.states["history"][1] == 2**54 + 1).all()
    assert (arena.states["recurrent"][1] == 0).all()
    for tensor in (arena.key, arena.value, arena.written, *arena.page_fields.values()):
        assert (tensor[1] == 0).all()
        assert (tensor[0] == 1).all() and (tensor[2:] == 1).all()
    for tensor in arena.states.values():
        assert (tensor[0] == 1).all() and (tensor[2:] == 1).all()
