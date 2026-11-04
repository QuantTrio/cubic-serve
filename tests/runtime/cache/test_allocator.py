from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from cubic_serve.runtime.cache import (
    CacheStateStore,
    KVLayout,
    OwnershipError,
    RankLocalPageAllocator,
    UnwrittenKVError,
)


class RankLocalPageAllocatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = KVLayout(
            layer_ids=(0,),
            block_size_tokens=4,
            num_kv_heads=2,
            head_dim=3,
        )
        self.store = CacheStateStore(self.layout, rank_id=0)
        self.allocator = RankLocalPageAllocator(self.store)

    def _payload(self, token_count: int, value: float) -> torch.Tensor:
        return torch.full(
            self.layout.payload_shape(token_count),
            value,
            dtype=torch.bfloat16,
        )

    def _fill_pages(self, owner_id: str, page_ids: tuple[int, ...]) -> None:
        offsets = tuple(range(self.layout.block_size_tokens))
        writes = tuple(
            self.store.kv_region(owner_id, page_id, layer_id, offsets)
            for page_id in page_ids
            for layer_id in self.layout.layer_ids
        )
        txn = self.store.begin(
            self.store.plan(f"txn:fill:{owner_id}", owner_id, writes=writes)
        )
        for page_id in page_ids:
            for layer_id in self.layout.layer_ids:
                txn.write_kv(
                    page_id,
                    layer_id,
                    offsets,
                    self._payload(4, float(page_id + 1)),
                    self._payload(4, float(-(page_id + 1))),
                )
        txn.verify()
        txn.commit()

    def test_context_order_and_segment_write_pages_are_exact(self) -> None:
        first = self.allocator.allocate(
            request_id="request:a", token_start=0, token_count=3
        )
        page_zero = first.context_pages[0]
        self.assertEqual(first.cache_slots, (page_zero * 4, page_zero * 4 + 1, page_zero * 4 + 2))
        self.assertEqual(first.context_pages, (page_zero,))
        self.assertEqual(first.write_pages, (page_zero,))

        crossing = self.allocator.allocate(
            request_id="request:a", token_start=3, token_count=3
        )
        page_one = crossing.context_pages[1]
        self.assertEqual(crossing.context_pages, (page_zero, page_one))
        self.assertEqual(crossing.write_pages, (page_zero, page_one))
        self.assertEqual(
            crossing.cache_slots,
            (page_zero * 4 + 3, page_one * 4, page_one * 4 + 1),
        )

        within_page = self.allocator.allocate(
            request_id="request:a", token_start=6, token_count=2
        )
        self.assertEqual(within_page.context_pages, (page_zero, page_one))
        self.assertEqual(within_page.write_pages, (page_one,))
        self.assertEqual(within_page.cache_slots, (page_one * 4 + 2, page_one * 4 + 3))

    def test_exact_retry_is_idempotent_but_gaps_and_overlap_fail(self) -> None:
        original = self.allocator.allocate(
            request_id="request:a", token_start=0, token_count=5
        )
        retry = self.allocator.allocate(
            request_id="request:a", token_start=0, token_count=5
        )
        self.assertIs(retry, original)
        self.assertEqual(
            len(self.allocator.page_table("request:a").logical_to_physical), 2
        )

        with self.assertRaisesRegex(ValueError, "contiguous token frontier"):
            self.allocator.allocate(
                request_id="request:a", token_start=4, token_count=1
            )
        with self.assertRaisesRegex(ValueError, "contiguous token frontier"):
            self.allocator.allocate(
                request_id="request:a", token_start=6, token_count=1
            )
        with self.assertRaisesRegex(ValueError, "token zero"):
            self.allocator.allocate(
                request_id="request:fresh", token_start=4, token_count=1
            )

    def test_allocation_never_makes_unwritten_slots_readable(self) -> None:
        allocation = self.allocator.allocate(
            request_id="request:a", token_start=0, token_count=1
        )
        with self.assertRaises(UnwrittenKVError):
            self.store.read_kv(
                "request:a", allocation.context_pages[0], 0, (0,)
            )
        self.assertTrue(
            self.store.unwritten_storage_is_zero(
                "request:a", allocation.context_pages[0]
            )
        )

    def test_restored_prefix_binds_in_logical_order_and_extends_at_boundary(self) -> None:
        source_pages = (
            self.store.allocate_page("request:source"),
            self.store.allocate_page("request:source"),
        )
        self._fill_pages("request:source", source_pages)
        snapshot = self.store.capture_prefix(
            snapshot_id="snapshot:source",
            prefix_key="prefix:source",
            owner_id="request:source",
            page_ids=source_pages,
            state_ids=(),
            physical_block_boundary=8,
            arithmetic_state_commit_boundary=8,
            state_commit_alignment=1,
            replay_tokens=(),
        )
        restore = self.store.restore_prefix(
            snapshot,
            expected_prefix_key="prefix:source",
            owner_id="request:target",
        )

        table = self.allocator.bind_restored_prefix(
            request_id="request:target", restore=restore
        )
        restored_pages = tuple(target for _, target in restore.source_to_restored_pages)
        self.assertEqual(table.logical_to_physical, restored_pages)
        self.assertEqual(table.prefix_token_count, 8)
        self.assertEqual(table.reserved_token_count, 8)

        extension = self.allocator.allocate(
            request_id="request:target", token_start=8, token_count=2
        )
        new_page = extension.context_pages[-1]
        self.assertEqual(extension.context_pages, (*restored_pages, new_page))
        self.assertEqual(extension.write_pages, (new_page,))
        self.assertEqual(extension.cache_slots, (new_page * 4, new_page * 4 + 1))

    def test_prefix_binding_validates_owner_rank_revision_and_single_bind(self) -> None:
        source_page = self.store.allocate_page("request:source")
        self._fill_pages("request:source", (source_page,))
        snapshot = self.store.capture_prefix(
            snapshot_id="snapshot:source",
            prefix_key="prefix:source",
            owner_id="request:source",
            page_ids=(source_page,),
            state_ids=(),
            physical_block_boundary=4,
            arithmetic_state_commit_boundary=4,
            state_commit_alignment=1,
            replay_tokens=(),
        )
        restore = self.store.restore_prefix(
            snapshot,
            expected_prefix_key="prefix:source",
            owner_id="request:target",
        )

        with self.assertRaisesRegex(ValueError, "owner"):
            self.allocator.bind_restored_prefix(
                request_id="request:other", restore=restore
            )
        with self.assertRaisesRegex(ValueError, "different cache rank"):
            self.allocator.bind_restored_prefix(
                request_id="request:target",
                restore=replace(restore, target_rank=1),
            )
        self.allocator.bind_restored_prefix(
            request_id="request:target", restore=restore
        )
        with self.assertRaisesRegex(ValueError, "already has"):
            self.allocator.bind_restored_prefix(
                request_id="request:target", restore=restore
            )

    def test_release_is_revision_guarded_and_owner_storage_is_retired(self) -> None:
        allocation = self.allocator.allocate(
            request_id="request:a", token_start=0, token_count=1
        )
        page_id = allocation.context_pages[0]
        with self.assertRaisesRegex(ValueError, "release revision changed"):
            self.allocator.release("request:a", expected_revision=1)

        released = self.allocator.release("request:a", expected_revision=0)
        self.assertTrue(released.released)
        self.assertEqual(released.logical_to_physical, (page_id,))
        with self.assertRaisesRegex(RuntimeError, "released"):
            self.allocator.allocate(
                request_id="request:a", token_start=1, token_count=1
            )
        with self.assertRaisesRegex(RuntimeError, "released"):
            self.allocator.release("request:a", expected_revision=0)

        other = self.allocator.allocate(
            request_id="request:b", token_start=0, token_count=1
        )
        self.assertNotEqual(other.context_pages[0], page_id)
        with self.assertRaises(OwnershipError):
            self.store.unwritten_storage_is_zero("request:a", page_id)


if __name__ == "__main__":
    unittest.main()
