"""Independent byte, ownership and capture witnesses for stable KV addresses."""

import gc
import os
import weakref

import pytest
import torch

from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout, StateSlotSpec
from cubic_serve.runtime.cache.pages import CacheStateError, PageFieldSpec


def test_capacity_admission_keeps_storage_and_rejects_live_or_stale_owners():
    layout = KVLayout((0,), 2, 1, 2)
    spec = CacheArenaSpec("test", 2, 2, (StateSlotSpec("state", torch.bfloat16, (2,)),),
                          scratch_capacity=2, reserved_page_capacity=16)
    arena = CacheArena(spec, layout, "cpu", rank_id=0)
    pointers = arena.backing_identity
    page = arena.reserve_page(page_id=0, owner_rank=0, owner_id="before")
    with pytest.raises(ValueError, match="live leases"):
        arena.finalize_page_capacity(8)
    arena.release_page(page)
    slot, _ = arena.reserve_state("state")
    with pytest.raises(ValueError, match="live leases"):
        arena.finalize_page_capacity(8)
    arena.release_state("state", slot)
    arena.finalize_page_capacity(8)
    assert arena.backing_identity == pointers
    assert arena.spec.page_capacity == 8
    assert arena.spec.storage_page_capacity == 16
    new = arena.reserve_page(page_id=1, owner_rank=0, owner_id="after")
    assert new.slot.index == page.slot.index and new.slot.generation > page.slot.generation
    with pytest.raises(CacheStateError):
        arena.validate_page_ref(page.slot)
    arena.release_page(new)
    with pytest.raises(ValueError):
        arena.finalize_page_capacity(9)


def test_virtual_extent_cannot_be_smaller_than_admitted_pages():
    with pytest.raises(ValueError):
        CacheArenaSpec("test", 3, 0, (), reserved_page_capacity=2)
    spec = CacheArenaSpec("test", 3, 0, ())
    assert spec.storage_page_capacity == spec.page_capacity == 3


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1",
                    reason="explicit CUDA storage audit required")
def test_graph_replays_and_permuted_rows_keep_exact_bytes_after_backing_growth():
    from cubic_serve.runtime.cache.virtual_memory import virtual_memory_supported

    device = torch.device("cuda", torch.cuda.current_device())
    if not virtual_memory_supported(device):
        pytest.skip("CUDA virtual memory is unavailable")
    layout = KVLayout((0,), 16, 2, 8,
                      page_fields=(PageFieldSpec("extra", torch.int32, (4,)),))
    spec = CacheArenaSpec("test", 2, 0, (), scratch_capacity=4,
                          reserved_page_capacity=32768)
    arena = CacheArena(spec, layout, device, rank_id=0)
    pointers = arena.backing_identity
    selected = torch.tensor([0, spec.storage_page_capacity], device=device)
    cpu = torch.arange(2 * 16 * 2 * 8, dtype=torch.int32).reshape(2, 1, 16, 2, 8)
    cpu = cpu.to(torch.bfloat16)
    arena.key[0].copy_(cpu[0])
    arena.key[spec.storage_page_capacity].copy_(cpu[1])
    output = torch.empty_like(cpu, device=device)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.index_select(arena.key, 0, selected, out=output)
    before = sum(owner.backed_bytes for owner, _ in arena._virtual_backings)
    additional = arena.additional_page_backing_bytes(4097)
    assert additional > 0
    arena.finalize_page_capacity(4097)
    assert arena.backing_identity == pointers
    assert sum(owner.backed_bytes for owner, _ in arena._virtual_backings) == before + additional
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(output.cpu().view(torch.uint8), cpu.view(torch.uint8))
    selected.copy_(selected.flip(0))
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(output.cpu().view(torch.uint8), cpu.flip(0).view(torch.uint8))
    for tensor in (arena.key, arena.value, arena.written, *arena.page_fields.values()):
        assert not bool(tensor[4096].contiguous().view(torch.uint8).any())
    graph.reset()


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1",
                    reason="explicit CUDA storage audit required")
def test_last_tensor_view_owns_mapping_and_sealed_storage_cannot_grow():
    from cubic_serve.runtime.cache.virtual_memory import VirtualAllocation, virtual_memory_supported

    device = torch.device("cuda", torch.cuda.current_device())
    if not virtual_memory_supported(device):
        pytest.skip("CUDA virtual memory is unavailable")
    owner = VirtualAllocation(16 << 20, device)
    owner.ensure_range(0, 1)
    tensor = owner.tensor((16 << 20,), torch.uint8)
    view = tensor[:1]
    lifetime = weakref.ref(owner)
    owner.seal()
    with pytest.raises(RuntimeError, match="unsealed startup"):
        owner.ensure_range(0, 4 << 20)
    del owner, tensor
    gc.collect()
    assert lifetime() is not None
    view.fill_(17)
    assert view.item() == 17
    del view
    gc.collect()
    assert lifetime() is None
