from __future__ import annotations

import unittest
from unittest import mock

import torch

from cubic_serve.contracts import execution as ec
from cubic_serve.runtime.cache import (
    AccessViolation,
    ArenaExhausted,
    ArenaPoisoned,
    CacheArena,
    CacheArenaSpec,
    CacheStateError,
    CacheStateStore,
    KVLayout,
    OwnershipError,
    PrefixCache,
    QualificationOwnerImage,
    QualificationOwnerLease,
    RecurrentStateProvision,
    RevisionConflict,
    ScratchArenaLeaseError,
    StaleArenaSlot,
    StateSlotSpec,
    TransactionStateError,
    UnwrittenKVError,
)


class CacheStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = KVLayout(
            layer_ids=(0, 2),
            block_size_tokens=4,
            num_kv_heads=2,
            head_dim=3,
        )
        self.store = CacheStateStore(self.layout, rank_id=0)

    def _payload(self, token_count: int, value: float) -> torch.Tensor:
        return torch.full(self.layout.payload_shape(token_count), value, dtype=torch.bfloat16)

    def _fill_pages_and_state(
        self,
        owner_id: str,
        page_ids: tuple[int, ...],
        state_id: str,
        *,
        state_position: int,
    ) -> None:
        offsets = tuple(range(self.layout.block_size_tokens))
        writes = (
            *(
                self.store.kv_region(owner_id, page_id, layer_id, offsets)
                for page_id in page_ids
                for layer_id in self.layout.layer_ids
            ),
            self.store.recurrent_region(owner_id, state_id),
        )
        txn = self.store.begin(
            self.store.plan(
                f"fill:{owner_id}",
                owner_id,
                writes=writes,
                target_position=state_position,
            )
        )
        for page_index, page_id in enumerate(page_ids):
            for layer_id in self.layout.layer_ids:
                base = float(10 * page_index + layer_id + 1)
                txn.write_kv(
                    page_id,
                    layer_id,
                    offsets,
                    self._payload(len(offsets), base),
                    self._payload(len(offsets), -base),
                )
        txn.write_recurrent_state(
            state_id,
            torch.full((2, 3), 7.0),
            target_position=state_position,
        )
        txn.verify()
        txn.commit()

    def test_layout_is_explicit_and_supports_half_precision(self) -> None:
        self.assertEqual(
            self.layout.axis_order,
            ("layer", "token_in_block", "kv_head", "head_dim"),
        )
        self.assertEqual(self.layout.page_shape, (2, 4, 2, 3))
        with self.assertRaisesRegex(ValueError, "FP16 or BF16"):
            KVLayout(
                layer_ids=(0,),
                block_size_tokens=4,
                num_kv_heads=1,
                head_dim=2,
                dtype=torch.float32,
            )

    def test_fresh_page_is_zeroed_but_unwritten_positions_are_unreadable(self) -> None:
        page_id = self.store.allocate_page("request:a")
        self.assertTrue(self.store.unwritten_storage_is_zero("request:a", page_id))
        self.assertFalse(bool(torch.any(self.store.page_written_mask("request:a", page_id)).item()))

        with self.assertRaisesRegex(UnwrittenKVError, "never readable"):
            self.store.read_kv("request:a", page_id, 0, (0,))

    def test_transaction_commits_kv_and_recurrent_state_atomically(self) -> None:
        page_id = self.store.allocate_page("request:a")
        self.store.register_recurrent_state(
            "request:a",
            "gdn:0",
            torch.zeros((2, 3), dtype=torch.float32),
            position=0,
        )
        kv_write = self.store.kv_region("request:a", page_id, 0, (0, 1))
        state_write = self.store.recurrent_region("request:a", "gdn:0")
        txn = self.store.begin(
            self.store.plan(
                "txn:1",
                "request:a",
                writes=(kv_write, state_write),
                target_position=2,
            )
        )

        self.assertEqual(txn.status, "open")
        txn.write_kv(
            page_id,
            0,
            (0, 1),
            self._payload(2, 3.0),
            self._payload(2, -3.0),
        )
        txn.write_recurrent_state(
            "gdn:0",
            torch.ones((2, 3), dtype=torch.float32),
            target_position=2,
        )
        self.assertEqual(txn.status, "tentative")

        # Tentative data is private until the verified commit.
        with self.assertRaises(UnwrittenKVError):
            self.store.read_kv("request:a", page_id, 0, (0,))
        self.assertTrue(
            torch.equal(
                self.store.read_recurrent_state("request:a", "gdn:0"),
                torch.zeros((2, 3)),
            )
        )

        txn.verify()
        self.assertEqual(txn.status, "verified")
        self.assertEqual(txn.commit(), 1)
        self.assertEqual(txn.status, "committed")

        key, value = self.store.read_kv("request:a", page_id, 0, (0, 1))
        self.assertTrue(torch.equal(key, self._payload(2, 3.0)))
        self.assertTrue(torch.equal(value, self._payload(2, -3.0)))
        self.assertTrue(
            torch.equal(
                self.store.read_recurrent_state("request:a", "gdn:0"),
                torch.ones((2, 3)),
            )
        )
        self.assertEqual(self.store.page_revision("request:a", page_id), 1)
        self.assertEqual(self.store.state_revision("request:a", "gdn:0"), 1)
        self.assertEqual(self.store.state_position("request:a", "gdn:0"), 2)

    def test_owner_provision_validates_all_state_before_publication(self) -> None:
        provisions = (
            RecurrentStateProvision("state:a", torch.zeros((2,), dtype=torch.float32)),
            RecurrentStateProvision(
                "state:b",
                torch.zeros((2,), dtype=torch.float32, device="meta"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "device"):
            self.store.provision_owner("request:provision", provisions)
        with self.assertRaises(OwnershipError):
            self.store.current_revision("request:provision")

    def test_nonfinite_initial_and_tentative_state_fail_closed(self) -> None:
        for nonfinite in (float("nan"), float("inf"), float("-inf")):
            with (
                self.subTest(boundary="provision", nonfinite=nonfinite),
                self.assertRaisesRegex(ValueError, "finite"),
            ):
                RecurrentStateProvision(
                    "state:bad",
                    torch.tensor((nonfinite,), dtype=torch.float32),
                )

        owner = "request:state-finite"
        self.store.register_recurrent_state(
            owner,
            "state:a",
            torch.zeros((2,), dtype=torch.float32),
            position=0,
        )
        transaction = self.store.begin(
            self.store.plan(
                "txn:state-nan",
                owner,
                writes=(self.store.recurrent_region(owner, "state:a"),),
                target_position=1,
            )
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            transaction.write_recurrent_state(
                "state:a",
                torch.tensor((1.0, float("nan"))),
                target_position=1,
            )
        self.assertEqual(self.store.current_revision(owner), 0)
        self.assertTrue(
            torch.equal(
                self.store.read_recurrent_state(owner, "state:a"),
                torch.zeros((2,), dtype=torch.float32),
            )
        )
        transaction.abort()

    def test_nonfinite_kv_write_fails_before_tentative_publication(self) -> None:
        owner = "request:kv-finite"
        page_id = self.store.allocate_page(owner)
        transaction = self.store.begin(
            self.store.plan(
                "txn:kv-nan",
                owner,
                writes=(self.store.kv_region(owner, page_id, 0, (0,)),),
            )
        )
        key = self._payload(1, 1.0)
        key.flatten()[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            transaction.write_kv(
                page_id,
                0,
                (0,),
                key,
                self._payload(1, 2.0),
            )
        self.assertEqual(transaction.status, "open")
        self.assertEqual(self.store.current_revision(owner), 0)
        self.assertFalse(bool(torch.any(self.store.page_written_mask(owner, page_id)).item()))
        transaction.abort()

    def test_exact_owner_provision_can_be_rolled_back_before_admission(self) -> None:
        owner = "request:rollback"
        self.store.provision_owner(
            owner,
            (RecurrentStateProvision("state:a", torch.zeros((2,), dtype=torch.float32)),),
        )
        page = self.store.allocate_page(owner)
        with self.assertRaisesRegex(OwnershipError, "pages differ"):
            self.store.rollback_owner_provision(owner, expected_revision=0)
        self.store.rollback_owner_provision(
            owner,
            expected_revision=0,
            expected_page_ids=(page,),
        )
        with self.assertRaises(OwnershipError):
            self.store.current_revision(owner)

    def test_commit_bundle_rejects_any_unverified_member_without_partial_publish(self) -> None:
        pages = {owner: self.store.allocate_page(owner) for owner in ("request:a", "request:b")}
        transactions = []
        for owner, value in (("request:a", 1.0), ("request:b", 2.0)):
            page_id = pages[owner]
            transaction = self.store.begin(
                self.store.plan(
                    f"txn:bundle:{owner}",
                    owner,
                    writes=(self.store.kv_region(owner, page_id, 0, (0,)),),
                )
            )
            transaction.write_kv(
                page_id,
                0,
                (0,),
                self._payload(1, value),
                self._payload(1, -value),
            )
            transactions.append(transaction)
        transactions[0].verify()

        with self.assertRaisesRegex(TransactionStateError, "every bundled"):
            self.store.commit_bundle(tuple(transactions))

        for owner, page_id in pages.items():
            self.assertEqual(self.store.current_revision(owner), 0)
            self.assertEqual(self.store.page_revision(owner, page_id), 0)
            self.assertFalse(bool(torch.any(self.store.page_written_mask(owner, page_id)).item()))
        for transaction in transactions:
            if transaction.status not in ("committed", "aborted"):
                transaction.abort()

    def test_commit_bundle_publishes_all_verified_owners_together(self) -> None:
        transactions = []
        pages = {}
        for owner, value in (("request:a", 1.0), ("request:b", 2.0)):
            page_id = self.store.allocate_page(owner)
            pages[owner] = page_id
            transaction = self.store.begin(
                self.store.plan(
                    f"txn:bundle:{owner}",
                    owner,
                    writes=(self.store.kv_region(owner, page_id, 0, (0,)),),
                )
            )
            transaction.write_kv(
                page_id,
                0,
                (0,),
                self._payload(1, value),
                self._payload(1, -value),
            )
            transaction.verify()
            transactions.append(transaction)

        self.assertEqual(self.store.commit_bundle(tuple(transactions)), (1, 1))
        self.assertEqual(
            tuple(transaction.status for transaction in transactions), ("committed",) * 2
        )
        for owner, expected in (("request:a", 1.0), ("request:b", 2.0)):
            key, _ = self.store.read_kv(owner, pages[owner], 0, (0,))
            self.assertTrue(torch.equal(key, self._payload(1, expected)))

    def test_commit_bundle_rechecks_payload_finiteness_without_partial_publish(self) -> None:
        for corruption in ("kv", "state"):
            with self.subTest(corruption=corruption):
                store = CacheStateStore(self.layout, rank_id=0)
                pages: dict[str, int] = {}
                transactions = []
                for owner, value in (("request:a", 1.0), ("request:b", 2.0)):
                    page_id = store.allocate_page(owner)
                    pages[owner] = page_id
                    store.register_recurrent_state(
                        owner,
                        "state:a",
                        torch.zeros((2,), dtype=torch.float32),
                        position=0,
                    )
                    transaction = store.begin(
                        store.plan(
                            f"txn:finite-recheck:{corruption}:{owner}",
                            owner,
                            writes=(
                                store.kv_region(owner, page_id, 0, (0,)),
                                store.recurrent_region(owner, "state:a"),
                            ),
                            target_position=1,
                        )
                    )
                    transaction.write_kv(
                        page_id,
                        0,
                        (0,),
                        self._payload(1, value),
                        self._payload(1, -value),
                    )
                    transaction.write_recurrent_state(
                        "state:a",
                        torch.full((2,), value),
                        target_position=1,
                    )
                    transaction.verify()
                    transactions.append(transaction)

                if corruption == "kv":
                    transactions[1]._staged_pages[pages["request:b"]].key.flatten()[0] = float(
                        "nan"
                    )
                else:
                    transactions[1]._staged_states["state:a"].value[0] = float("inf")

                with self.assertRaisesRegex(ValueError, "finite"):
                    store.commit_bundle(tuple(transactions))

                for owner, page_id in pages.items():
                    self.assertEqual(store.current_revision(owner), 0)
                    self.assertEqual(store.page_revision(owner, page_id), 0)
                    self.assertFalse(
                        bool(torch.any(store.page_written_mask(owner, page_id)).item())
                    )
                    self.assertTrue(
                        torch.equal(
                            store.read_recurrent_state(owner, "state:a"),
                            torch.zeros((2,), dtype=torch.float32),
                        )
                    )
                self.assertEqual(
                    tuple(transaction.status for transaction in transactions),
                    ("verified", "verified"),
                )
                for transaction in transactions:
                    transaction.abort()

    def test_abort_discards_copy_on_write_storage(self) -> None:
        page_id = self.store.allocate_page("request:a")
        region = self.store.kv_region("request:a", page_id, 0, (0,))
        txn = self.store.begin(self.store.plan("txn:abort", "request:a", writes=(region,)))
        txn.write_kv(page_id, 0, (0,), self._payload(1, 9.0), self._payload(1, -9.0))
        txn.abort()

        self.assertEqual(txn.status, "aborted")
        self.assertEqual(self.store.page_revision("request:a", page_id), 0)
        self.assertFalse(bool(self.store.page_written_mask("request:a", page_id)[0, 0].item()))
        self.assertTrue(self.store.unwritten_storage_is_zero("request:a", page_id))
        with self.assertRaises(TransactionStateError):
            txn.verify()

    def test_revision_conflict_aborts_without_partial_publication(self) -> None:
        page_id = self.store.allocate_page("request:a")
        self.store.register_recurrent_state(
            "request:a",
            "gdn:0",
            torch.zeros((2, 3), dtype=torch.float32),
            position=0,
        )
        winner_region = self.store.kv_region("request:a", page_id, 0, (0,))
        loser_kv = self.store.kv_region("request:a", page_id, 0, (1,))
        loser_state = self.store.recurrent_region("request:a", "gdn:0")
        winner = self.store.begin(
            self.store.plan("txn:winner", "request:a", writes=(winner_region,))
        )
        loser = self.store.begin(
            self.store.plan(
                "txn:loser",
                "request:a",
                writes=(loser_kv, loser_state),
                target_position=1,
            )
        )

        loser.write_kv(page_id, 0, (1,), self._payload(1, 8.0), self._payload(1, -8.0))
        loser.write_recurrent_state("gdn:0", torch.full((2, 3), 8.0), target_position=1)
        winner.write_kv(page_id, 0, (0,), self._payload(1, 1.0), self._payload(1, -1.0))
        winner.verify()
        winner.commit()

        with self.assertRaises(RevisionConflict):
            loser.verify()
        self.assertEqual(loser.status, "aborted")
        with self.assertRaises(UnwrittenKVError):
            self.store.read_kv("request:a", page_id, 0, (1,))
        self.assertTrue(
            torch.equal(
                self.store.read_recurrent_state("request:a", "gdn:0"),
                torch.zeros((2, 3)),
            )
        )

    def test_write_set_must_be_fulfilled_exactly(self) -> None:
        page_id = self.store.allocate_page("request:a")
        region = self.store.kv_region("request:a", page_id, 0, (0, 1))
        txn = self.store.begin(self.store.plan("txn:partial", "request:a", writes=(region,)))
        txn.write_kv(page_id, 0, (0,), self._payload(1, 1.0), self._payload(1, -1.0))
        with self.assertRaisesRegex(AccessViolation, "not fulfilled"):
            txn.verify()
        txn.abort()

    def test_recurrent_position_is_explicit_plan_authority_and_monotonic(self) -> None:
        owner_id = "request:position"
        self.store.register_recurrent_state(
            owner_id,
            "gdn:0",
            torch.zeros((2, 3), dtype=torch.float32),
            position=2,
        )
        region = self.store.recurrent_region(owner_id, "gdn:0")
        with self.assertRaisesRegex(ValueError, "target position"):
            self.store.plan("txn:no-position", owner_id, writes=(region,))

        backwards = self.store.begin(
            self.store.plan(
                "txn:backwards",
                owner_id,
                writes=(region,),
                target_position=1,
            )
        )
        with self.assertRaisesRegex(ValueError, "cannot move backwards"):
            backwards.write_recurrent_state("gdn:0", torch.ones((2, 3)), target_position=1)
        backwards.abort()

        planned = self.store.begin(
            self.store.plan(
                "txn:position-bound",
                owner_id,
                writes=(region,),
                target_position=3,
            )
        )
        with self.assertRaisesRegex(AccessViolation, "transaction target"):
            planned.write_recurrent_state("gdn:0", torch.ones((2, 3)), target_position=4)
        planned.write_recurrent_state("gdn:0", torch.ones((2, 3)), target_position=3)
        planned.verify()
        planned.commit()
        self.assertEqual(self.store.state_position(owner_id, "gdn:0"), 3)

    def test_access_dtype_shape_and_ownership_fail_closed(self) -> None:
        page_id = self.store.allocate_page("request:a")
        with self.assertRaises(OwnershipError):
            self.store.page_written_mask("request:b", page_id)

        region = self.store.kv_region("request:a", page_id, 0, (0,))
        txn = self.store.begin(self.store.plan("txn:bad", "request:a", writes=(region,)))
        with self.assertRaisesRegex(AccessViolation, "declared"):
            txn.read_kv(page_id, 0, (0,))
        with self.assertRaisesRegex(ValueError, "storage dtype"):
            txn.write_kv(
                page_id,
                0,
                (0,),
                torch.zeros((1, 2, 3), dtype=torch.float32),
                torch.zeros((1, 2, 3), dtype=torch.float32),
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            txn.write_kv(
                page_id,
                0,
                (0,),
                torch.zeros((1, 1, 3), dtype=torch.bfloat16),
                torch.zeros((1, 1, 3), dtype=torch.bfloat16),
            )
        txn.abort()

    def test_prefix_snapshot_distinguishes_physical_and_state_boundaries(self) -> None:
        owner_id = "request:source"
        page_ids = (
            self.store.allocate_page(owner_id),
            self.store.allocate_page(owner_id),
        )
        self.store.register_recurrent_state(
            owner_id,
            "gdn:0",
            torch.zeros((2, 3), dtype=torch.float32),
            position=0,
        )
        self._fill_pages_and_state(owner_id, page_ids, "gdn:0", state_position=6)

        snapshot = self.store.capture_prefix(
            snapshot_id="snapshot:1",
            prefix_key="b2s256:canonical-input-and-plan",
            owner_id=owner_id,
            page_ids=page_ids,
            state_ids=("gdn:0",),
            physical_block_boundary=8,
            arithmetic_state_commit_boundary=6,
            state_commit_alignment=3,
            replay_tokens=(106, 107),
        )
        self.assertEqual(snapshot.physical_block_boundary, 8)
        self.assertEqual(snapshot.arithmetic_state_commit_boundary, 6)
        self.assertEqual(snapshot.replay_tokens, (106, 107))
        self.assertEqual(snapshot.state_valid_mask, (True,))
        self.assertEqual(snapshot.recurrent_states[0].position, 6)

        prefix_cache = PrefixCache()
        self.assertTrue(prefix_cache.publish(snapshot))
        self.assertEqual(snapshot.source_rank, 0)
        lease = prefix_cache.acquire(snapshot.prefix_key, source_rank=0)
        self.assertIsNotNone(lease)
        assert lease is not None
        self.assertFalse(prefix_cache.evict(snapshot.prefix_key, source_rank=0))
        leased_snapshot = lease.snapshot
        leased_snapshot.pages[0].key.zero_()
        lease.release()

        # Lease consumers cannot mutate the cache's private immutable image.
        second_lease = prefix_cache.acquire(snapshot.prefix_key, source_rank=0)
        assert second_lease is not None
        clean_snapshot = second_lease.snapshot
        self.assertFalse(bool(torch.all(clean_snapshot.pages[0].key == 0).item()))
        second_lease.release()

        tampered = clean_snapshot.clone()
        tampered.pages[0].key[0, 0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.store.restore_prefix(
                tampered,
                expected_prefix_key=snapshot.prefix_key,
                owner_id="request:tampered",
            )

        nonfinite = clean_snapshot.clone()
        nonfinite.pages[0].key[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.store.restore_prefix(
                nonfinite,
                expected_prefix_key=snapshot.prefix_key,
                owner_id="request:nonfinite",
            )
        with self.assertRaises(OwnershipError):
            self.store.current_revision("request:nonfinite")

        restored = self.store.restore_prefix(
            clean_snapshot,
            expected_prefix_key=snapshot.prefix_key,
            owner_id="request:restored",
        )
        self.assertEqual(restored.replay_tokens, (106, 107))
        self.assertEqual(restored.state_valid_mask, (True,))
        self.assertEqual(restored.state_positions, (("gdn:0", 6),))
        self.assertEqual((restored.source_rank, restored.target_rank), (0, 0))
        restored_page_id = restored.source_to_restored_pages[0][1]
        restored_key, _ = self.store.read_kv("request:restored", restored_page_id, 0, (0, 1, 2, 3))
        source_key, _ = self.store.read_kv(owner_id, page_ids[0], 0, (0, 1, 2, 3))
        self.assertTrue(torch.equal(restored_key, source_key))
        self.assertTrue(
            torch.equal(
                self.store.read_recurrent_state("request:restored", "gdn:0"),
                torch.full((2, 3), 7.0),
            )
        )

        other_rank = CacheStateStore(self.layout, rank_id=1)
        with self.assertRaisesRegex(OwnershipError, "cannot restore"):
            other_rank.restore_prefix(
                clean_snapshot,
                expected_prefix_key=snapshot.prefix_key,
                owner_id="request:wrong-rank",
            )
        self.assertTrue(prefix_cache.evict(snapshot.prefix_key, source_rank=0))

    def test_semantic_prefix_collision_quarantines_future_reads(self) -> None:
        first_owner = "request:first-prefix"
        first_page = self.store.allocate_page(first_owner)
        self.store.register_recurrent_state(
            first_owner,
            "gdn:0",
            torch.zeros((2, 3), dtype=torch.float32),
            position=0,
        )
        self._fill_pages_and_state(first_owner, (first_page,), "gdn:0", state_position=4)
        first = self.store.capture_prefix(
            snapshot_id="snapshot:first",
            prefix_key="prefix:collision",
            owner_id=first_owner,
            page_ids=(first_page,),
            state_ids=("gdn:0",),
            physical_block_boundary=4,
            arithmetic_state_commit_boundary=4,
            state_commit_alignment=1,
            replay_tokens=(),
        )

        second_owner = "request:second-prefix"
        second_page = self.store.allocate_page(second_owner)
        self.store.register_recurrent_state(
            second_owner,
            "gdn:0",
            torch.zeros((2, 3), dtype=torch.float32),
            position=0,
        )
        self._fill_pages_and_state(second_owner, (second_page,), "gdn:0", state_position=4)
        rewrite = self.store.begin(
            self.store.plan(
                "txn:rewrite-second",
                second_owner,
                writes=(self.store.kv_region(second_owner, second_page, 0, (0,)),),
            )
        )
        rewrite.write_kv(
            second_page,
            0,
            (0,),
            self._payload(1, 99.0),
            self._payload(1, -99.0),
        )
        rewrite.verify()
        rewrite.commit()
        second = self.store.capture_prefix(
            snapshot_id="snapshot:second",
            prefix_key="prefix:collision",
            owner_id=second_owner,
            page_ids=(second_page,),
            state_ids=("gdn:0",),
            physical_block_boundary=4,
            arithmetic_state_commit_boundary=4,
            state_commit_alignment=1,
            replay_tokens=(),
        )

        prefix_cache = PrefixCache()
        self.assertTrue(prefix_cache.publish(first))
        with self.assertRaisesRegex(ValueError, "two state bundles"):
            prefix_cache.publish(second)
        self.assertIsNotNone(prefix_cache.quarantine_reason("prefix:collision", source_rank=0))
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            prefix_cache.acquire("prefix:collision", source_rank=0)

    def test_current_state_cannot_claim_an_earlier_arithmetic_boundary(self) -> None:
        owner_id = "request:advanced"
        page_id = self.store.allocate_page(owner_id)
        self.store.register_recurrent_state(
            owner_id,
            "gdn:0",
            torch.zeros((2, 3), dtype=torch.float32),
            position=0,
        )
        self._fill_pages_and_state(owner_id, (page_id,), "gdn:0", state_position=4)

        with self.assertRaisesRegex(ValueError, "position.*arithmetic boundary"):
            self.store.capture_prefix(
                snapshot_id="snapshot:false-boundary",
                prefix_key="prefix:false-boundary",
                owner_id=owner_id,
                page_ids=(page_id,),
                state_ids=("gdn:0",),
                physical_block_boundary=4,
                arithmetic_state_commit_boundary=3,
                state_commit_alignment=3,
                replay_tokens=(44,),
            )

    def test_earlier_recurrent_checkpoint_supports_exact_replay_tail(self) -> None:
        owner_id = "request:checkpoint"
        page_id = self.store.allocate_page(owner_id)
        self.store.register_recurrent_state(
            owner_id,
            "gdn:0",
            torch.zeros((2, 3), dtype=torch.float32),
            position=0,
        )

        prefix_offsets = (0, 1, 2)
        first_writes = (
            *(
                self.store.kv_region(owner_id, page_id, layer_id, prefix_offsets)
                for layer_id in self.layout.layer_ids
            ),
            self.store.recurrent_region(owner_id, "gdn:0"),
        )
        first = self.store.begin(
            self.store.plan(
                "txn:state-at-3",
                owner_id,
                writes=first_writes,
                target_position=3,
            )
        )
        for layer_id in self.layout.layer_ids:
            first.write_kv(
                page_id,
                layer_id,
                prefix_offsets,
                self._payload(3, 3.0),
                self._payload(3, -3.0),
            )
        first.write_recurrent_state("gdn:0", torch.full((2, 3), 3.0), target_position=3)
        first.verify()
        first.commit()
        checkpoint = self.store.checkpoint_recurrent_state(owner_id, "gdn:0")

        tail_offsets = (3,)
        second_writes = (
            *(
                self.store.kv_region(owner_id, page_id, layer_id, tail_offsets)
                for layer_id in self.layout.layer_ids
            ),
            self.store.recurrent_region(owner_id, "gdn:0"),
        )
        second = self.store.begin(
            self.store.plan(
                "txn:state-at-4",
                owner_id,
                writes=second_writes,
                target_position=4,
            )
        )
        for layer_id in self.layout.layer_ids:
            second.write_kv(
                page_id,
                layer_id,
                tail_offsets,
                self._payload(1, 4.0),
                self._payload(1, -4.0),
            )
        second.write_recurrent_state("gdn:0", torch.full((2, 3), 4.0), target_position=4)
        second.verify()
        second.commit()

        tampered_checkpoint = checkpoint.clone()
        tampered_checkpoint.value.add_(1)
        with self.assertRaisesRegex(ValueError, "checkpoint content digest mismatch"):
            self.store.capture_prefix(
                snapshot_id="snapshot:tampered-checkpoint",
                prefix_key="prefix:tampered-checkpoint",
                owner_id=owner_id,
                page_ids=(page_id,),
                state_ids=("gdn:0",),
                physical_block_boundary=4,
                arithmetic_state_commit_boundary=3,
                state_commit_alignment=3,
                replay_tokens=(44,),
                recurrent_checkpoints=(tampered_checkpoint,),
            )

        snapshot = self.store.capture_prefix(
            snapshot_id="snapshot:checkpoint",
            prefix_key="prefix:checkpoint",
            owner_id=owner_id,
            page_ids=(page_id,),
            state_ids=("gdn:0",),
            physical_block_boundary=4,
            arithmetic_state_commit_boundary=3,
            state_commit_alignment=3,
            replay_tokens=(44,),
            recurrent_checkpoints=(checkpoint,),
        )
        self.assertEqual(self.store.state_position(owner_id, "gdn:0"), 4)
        self.assertEqual(snapshot.recurrent_states[0].position, 3)
        self.assertEqual(snapshot.recurrent_states[0].revision, 1)
        self.assertTrue(torch.equal(snapshot.recurrent_states[0].value, torch.full((2, 3), 3.0)))

        restored = self.store.restore_prefix(
            snapshot,
            expected_prefix_key="prefix:checkpoint",
            owner_id="request:checkpoint-restored",
        )
        self.assertEqual(restored.state_positions, (("gdn:0", 3),))
        self.assertEqual(restored.replay_tokens, (44,))
        self.assertEqual(self.store.state_position("request:checkpoint-restored", "gdn:0"), 3)
        self.assertTrue(
            torch.equal(
                self.store.read_recurrent_state("request:checkpoint-restored", "gdn:0"),
                torch.full((2, 3), 3.0),
            )
        )

    def test_partial_or_pending_state_cannot_publish_as_prefix(self) -> None:
        owner_id = "request:a"
        page_id = self.store.allocate_page(owner_id)
        region = self.store.kv_region(owner_id, page_id, 0, (0,))
        txn = self.store.begin(self.store.plan("txn:pending", owner_id, writes=(region,)))
        with self.assertRaisesRegex(TransactionStateError, "tentative"):
            self.store.capture_prefix(
                snapshot_id="snapshot:bad",
                prefix_key="prefix:bad",
                owner_id=owner_id,
                page_ids=(page_id,),
                state_ids=(),
                physical_block_boundary=4,
                arithmetic_state_commit_boundary=4,
                state_commit_alignment=1,
                replay_tokens=(),
            )
        txn.abort()

        with self.assertRaisesRegex(ValueError, "fully written"):
            self.store.capture_prefix(
                snapshot_id="snapshot:partial",
                prefix_key="prefix:partial",
                owner_id=owner_id,
                page_ids=(page_id,),
                state_ids=(),
                physical_block_boundary=4,
                arithmetic_state_commit_boundary=4,
                state_commit_alignment=1,
                replay_tokens=(),
            )

    def test_invalid_recurrent_state_is_restored_with_explicit_mask(self) -> None:
        owner_id = "request:source"
        page_id = self.store.allocate_page(owner_id)
        self.store.register_recurrent_state(
            owner_id,
            "gdn:0",
            torch.zeros((2, 3), dtype=torch.float32),
            position=0,
            valid=False,
        )
        offsets = tuple(range(4))
        writes = tuple(
            self.store.kv_region(owner_id, page_id, layer_id, offsets)
            for layer_id in self.layout.layer_ids
        )
        txn = self.store.begin(self.store.plan("txn:fill", owner_id, writes=writes))
        for layer_id in self.layout.layer_ids:
            txn.write_kv(
                page_id,
                layer_id,
                offsets,
                self._payload(4, 1.0),
                self._payload(4, -1.0),
            )
        txn.verify()
        txn.commit()

        snapshot = self.store.capture_prefix(
            snapshot_id="snapshot:replay",
            prefix_key="prefix:replay",
            owner_id=owner_id,
            page_ids=(page_id,),
            state_ids=("gdn:0",),
            physical_block_boundary=4,
            arithmetic_state_commit_boundary=0,
            state_commit_alignment=8,
            replay_tokens=(1, 2, 3, 4),
        )
        self.assertEqual(snapshot.state_valid_mask, (False,))
        self.store.restore_prefix(
            snapshot,
            expected_prefix_key="prefix:replay",
            owner_id="request:restored",
        )
        with self.assertRaises(CacheStateError):
            self.store.read_recurrent_state("request:restored", "gdn:0")


class CacheArenaTests(unittest.TestCase):
    def setUp(self) -> None:
        self._lease_epoch = 0
        self.layout = KVLayout(
            layer_ids=(0,),
            block_size_tokens=2,
            num_kv_heads=1,
            head_dim=2,
        )
        self.spec = CacheArenaSpec(
            arena_id="arena:test",
            page_capacity=6,
            state_slot_capacity=6,
            state_specs=(
                StateSlotSpec("conv:0", torch.bfloat16, (2, 3)),
                StateSlotSpec("recurrent:0", torch.float32, (2, 2)),
            ),
            scratch_capacity=2,
            scratch_arena_ids=("lease:0", "lease:1"),
        )

    def _store(self) -> CacheStateStore:
        return CacheStateStore(self.layout, rank_id=0, arena_spec=self.spec)

    def _qualification_image(
        self, owner: str, page: int, *, corrupt_unwritten: bool = False
    ) -> QualificationOwnerImage:
        key = torch.zeros((1, *self.layout.page_shape), dtype=torch.bfloat16)
        value = torch.zeros_like(key)
        written = torch.zeros((1, 1, 2), dtype=torch.bool)
        key[0, 0, 0].fill_(3)
        value[0, 0, 0].fill_(-3)
        written[0, 0, 0] = True
        if corrupt_unwritten:
            key[0, 0, 1, 0, 0] = 9
        return QualificationOwnerImage(
            owner,
            (page,),
            key,
            value,
            written,
            (
                RecurrentStateProvision(
                    "conv:0", torch.ones((2, 3), dtype=torch.bfloat16), position=1
                ),
                RecurrentStateProvision(
                    "recurrent:0", torch.ones((2, 2), dtype=torch.float32), position=1
                ),
            ),
        )

    def test_qualification_owner_image_roundtrips_partial_pages_and_blocks_pending(self) -> None:
        store = self._store()
        image = self._qualification_image("qualification:a", 4)
        lease = store.provision_qualification_owner(image)
        observed = store.snapshot_qualification_owner(lease)
        self.assertTrue(torch.equal(observed.key, image.key))
        self.assertTrue(torch.equal(observed.value, image.value))
        self.assertTrue(torch.equal(observed.written, image.written))
        self.assertEqual(
            tuple(state.state_id for state in observed.recurrent_states),
            ("conv:0", "recurrent:0"),
        )

        transaction = self._transaction(store, 4, "qualification:pending", "qualification:a")
        with self.assertRaisesRegex(TransactionStateError, "tentative state"):
            store.snapshot_qualification_owner(lease)
        transaction.abort()
        store.retire_qualification_owner(lease, expected_revision=0)

    def test_qualification_owner_lease_rejects_unwritten_wrong_and_stale_images(self) -> None:
        store = self._store()
        with self.assertRaisesRegex(ValueError, "canonical arena ABI"):
            store.provision_qualification_owner(
                self._qualification_image("qualification:bad", 3, corrupt_unwritten=True)
            )

        lease = store.provision_qualification_owner(self._qualification_image("qualification:a", 4))
        forged = (
            QualificationOwnerLease("qualification:a", (4,), lease.state_ids, object()),
            QualificationOwnerLease("qualification:foreign", (4,), lease.state_ids, object()),
        )
        for capability in forged:
            with self.assertRaisesRegex(OwnershipError, "stale or foreign"):
                store.snapshot_qualification_owner(capability)
        store.retire_qualification_owner(lease, expected_revision=0)
        with self.assertRaisesRegex(OwnershipError, "stale or foreign"):
            store.snapshot_qualification_owner(lease)

        replacement = store.provision_qualification_owner(
            self._qualification_image("qualification:a", 4)
        )
        with self.assertRaisesRegex(OwnershipError, "stale or foreign"):
            store.snapshot_qualification_owner(lease)
        store.retire_qualification_owner(replacement, expected_revision=0)

    def _launch(
        self,
        scratch_arena_id: str = "lease:0",
        *,
        logical_m: int = 1,
        captured_m: int = 2,
        resource_domain: str = "program",
        epoch: int | None = None,
        rank: int = 0,
    ) -> ec.LaunchPlan:
        if epoch is None:
            self._lease_epoch += 1
            epoch = self._lease_epoch
        launch_id = f"launch:{resource_domain}:{scratch_arena_id}:{epoch}"
        bucket = ec.GraphBucket(
            bucket_id="bucket:cache-arena",
            template_id="template:cache-arena",
            scope_id="scope:cache-arena",
            replay_abi_id="abi:cache-arena",
            topology_id="topology:cache-arena",
            partition_id="partition:cache-arena",
            phase=ec.BatchPhase.DECODE,
            capacity_items=(("captured_m", captured_m),),
            logical_m_values=(1, captured_m),
            captured_m=captured_m,
            row_map_schema_id="rows:cache-arena",
            scratch_schema_id="scratch:cache-arena",
            graph_safety_contract_ids=("contract:padding",),
            evidence_id="evidence:cache-arena",
        )
        rows: tuple[ec.ActiveLaunchRow | ec.PaddingLaunchRow, ...] = (
            *(ec.ActiveLaunchRow(index) for index in range(logical_m)),
            *(ec.PaddingLaunchRow(index) for index in range(captured_m - logical_m)),
        )
        resource = ec.RankReplayResources(
            rank=rank,
            graph_instance_id=f"graph:{resource_domain}:{scratch_arena_id}",
            stream_lane_id=f"stream:{resource_domain}:{scratch_arena_id}",
            buffer_set_id=f"buffers:{resource_domain}:{scratch_arena_id}",
            workspace_arena_id=f"workspace:{resource_domain}:{scratch_arena_id}",
            scratch_arena_id=scratch_arena_id,
        )
        lease = ec.ReplayLease(
            lease_id=f"lease:{resource_domain}:{scratch_arena_id}:{epoch}",
            epoch=epoch,
            launch_plan_id=launch_id,
            execution_plan_id="execution:cache-arena",
            bucket_id=bucket.bucket_id,
            template_id=bucket.template_id,
            topology_id=bucket.topology_id,
            partition_id=bucket.partition_id,
            rank_resources=(resource,),
        )
        return ec.LaunchPlan(
            launch_plan_id=launch_id,
            execution_plan_id=lease.execution_plan_id,
            source_plan_id=f"batch:{launch_id}",
            logical_m=logical_m,
            bucket=bucket,
            rows=rows,
            replay_lease=lease,
        )

    def _provision(self, store: CacheStateStore, owner: str = "request:a") -> int:
        store.provision_owner(
            owner,
            (
                RecurrentStateProvision("conv:0", torch.zeros((2, 3), dtype=torch.bfloat16)),
                RecurrentStateProvision("recurrent:0", torch.zeros((2, 2), dtype=torch.float32)),
            ),
        )
        return store.allocate_page(owner)

    def _transaction(
        self,
        store: CacheStateStore,
        page: int,
        transaction_id: str,
        owner: str = "request:a",
    ):
        return store.begin(
            store.plan(
                transaction_id,
                owner,
                writes=(
                    store.kv_region(owner, page, 0, (0,)),
                    store.recurrent_region(owner, "conv:0"),
                    store.recurrent_region(owner, "recurrent:0"),
                ),
                target_position=1,
            )
        )

    def _write(self, transaction, page: int, value: float) -> None:
        shape = self.layout.payload_shape(1)
        transaction.write_kv(
            page,
            0,
            (0,),
            torch.full(shape, value, dtype=torch.bfloat16),
            torch.full(shape, -value, dtype=torch.bfloat16),
        )
        transaction.write_recurrent_state(
            "conv:0",
            torch.full((2, 3), value, dtype=torch.bfloat16),
            target_position=1,
        )
        transaction.write_recurrent_state(
            "recurrent:0",
            torch.full((2, 2), value, dtype=torch.float32),
            target_position=1,
        )

    @staticmethod
    def _emulate_graph(binding, page: int, value: float) -> None:
        page_slot = {slot.resource_id: slot for slot in binding.page_slots}[page]
        source, destination = page_slot.source_slot, page_slot.destination_slot
        binding.key[destination].copy_(binding.key[source])
        binding.value[destination].copy_(binding.value[source])
        binding.written[destination].copy_(binding.written[source])
        binding.key[destination, 0, 0].fill_(value)
        binding.value[destination, 0, 0].fill_(-value)
        binding.written[destination, 0, 0] = True
        states = dict(binding.state_tensors)
        for state_slot in binding.state_slots:
            slab = states[str(state_slot.resource_id)]
            slab[state_slot.destination_slot].copy_(slab[state_slot.source_slot])
            slab[state_slot.destination_slot].add_(value)

    def test_shared_rank_arena_gives_multiple_graph_slots_identical_base_pointers(self) -> None:
        spec = CacheArenaSpec(
            self.spec.arena_id,
            8,
            8,
            self.spec.state_specs,
            self.spec.scratch_capacity,
            self.spec.scratch_arena_ids,
        )
        arena = CacheArena(spec, self.layout, "cpu", rank_id=0)
        first = CacheStateStore(self.layout, rank_id=0, arena=arena)
        second = CacheStateStore(self.layout, rank_id=0, arena=arena)
        first_page = self._provision(first, "qualification:temp")
        second_page = self._provision(second, "production:request")
        first_txn = self._transaction(first, first_page, "txn:qualification", "qualification:temp")
        second_txn = self._transaction(second, second_page, "txn:production", "production:request")
        first_binding = first_txn.graph_binding(self._launch("lease:0"))
        second_binding = second_txn.graph_binding(self._launch("lease:1"))
        self.assertEqual(first_binding.key.data_ptr(), second_binding.key.data_ptr())
        self.assertEqual(first_binding.value.data_ptr(), second_binding.value.data_ptr())
        self.assertEqual(
            tuple(tensor.data_ptr() for _, tensor in first_binding.state_tensors),
            tuple(tensor.data_ptr() for _, tensor in second_binding.state_tensors),
        )
        self.assertNotEqual(first_binding.scratch_page_slots, second_binding.scratch_page_slots)
        self._emulate_graph(first_binding, first_page, 2.0)
        first_txn.accept_graph_writes()
        first_txn.verify()
        revision = first_txn.commit()
        first.read_kv("qualification:temp", first_page, 0, (0,))
        first.checkpoint_recurrent_state("qualification:temp", "recurrent:0")
        second_txn.abort()
        first.retire_owner(
            "qualification:temp", expected_revision=revision, expected_page_ids=(first_page,)
        )

    def test_fp16_kv_and_conv_preserve_bf16_recurrent_storage(self) -> None:
        layout = KVLayout((0,), 4, 2, 3, dtype=torch.float16)
        spec = CacheArenaSpec(
            "fp16", 2, 2,
            (StateSlotSpec("conv", torch.float16, (2, 3)),
             StateSlotSpec("recurrent", torch.bfloat16, (2, 3))),
        )
        arena = CacheArena(spec, layout, "cpu", rank_id=0)
        self.assertEqual(arena.key.dtype, torch.float16)
        self.assertEqual(arena.value.dtype, torch.float16)
        self.assertEqual(arena.states["conv"].dtype, torch.float16)
        self.assertEqual(arena.states["recurrent"].dtype, torch.bfloat16)

    def test_schema_preallocates_typed_slabs_and_lease_private_scratch(self) -> None:
        with self.assertRaisesRegex(ValueError, "state slabs require"):
            StateSlotSpec("bad", torch.float64, (1,))
        with self.assertRaisesRegex(ValueError, "unique sorted ids"):
            CacheArenaSpec(
                "bad-order",
                1,
                1,
                (
                    StateSlotSpec("z", torch.float32, (1,)),
                    StateSlotSpec("a", torch.bfloat16, (1,)),
                ),
            )
        with self.assertRaisesRegex(ValueError, "unique, and sorted"):
            CacheArenaSpec("bad-scratch", 1, 0, (), 1, ("z", "a"))

        store = self._store()
        page_a = self._provision(store)
        page_b = self._provision(store, "request:b")
        transaction_a = self._transaction(store, page_a, "txn:binding:a")
        transaction_b = self._transaction(store, page_b, "txn:binding:b", owner="request:b")
        with self.assertRaisesRegex(ScratchArenaLeaseError, "unknown scratch arena"):
            transaction_a.graph_binding(self._launch("unknown"))
        launch_a = self._launch("lease:0")
        binding_a = transaction_a.graph_binding(launch_a)
        with self.assertRaisesRegex(ScratchArenaLeaseError, "another replay lease"):
            transaction_a.graph_binding(self._launch("lease:1"))
        launch_b = self._launch("lease:1")
        binding_b = transaction_b.graph_binding(launch_b)

        self.assertEqual(binding_a.key.shape, (10, 1, 2, 1, 2))
        self.assertEqual(binding_a.key.dtype, torch.bfloat16)
        self.assertEqual(binding_a.value.dtype, torch.bfloat16)
        self.assertEqual(binding_a.written.dtype, torch.bool)
        self.assertTrue(binding_a.key.is_contiguous())
        self.assertTrue(binding_a.value.is_contiguous())
        self.assertTrue(binding_a.written.is_contiguous())
        states = dict(binding_a.state_tensors)
        self.assertEqual(states["conv:0"].dtype, torch.bfloat16)
        self.assertEqual(states["recurrent:0"].dtype, torch.float32)
        self.assertTrue(all(tensor.is_contiguous() for tensor in states.values()))
        self.assertEqual(binding_a.scratch_page_slots, (6, 7))
        self.assertEqual(binding_b.scratch_page_slots, (8, 9))
        self.assertEqual(
            dict(binding_a.scratch_state_slots),
            {"conv:0": (6, 7), "recurrent:0": (6, 7)},
        )
        self.assertEqual(
            dict(binding_b.scratch_state_slots),
            {"conv:0": (8, 9), "recurrent:0": (8, 9)},
        )
        self.assertTrue(set(binding_a.scratch_page_slots).isdisjoint(binding_b.scratch_page_slots))
        owner_page_slots = {
            physical
            for binding in (binding_a, binding_b)
            for slots in binding.page_slots
            for physical in (slots.source_slot, slots.destination_slot)
        }
        owner_state_slots = {
            physical
            for binding in (binding_a, binding_b)
            for slots in binding.state_slots
            for physical in (slots.source_slot, slots.destination_slot)
        }
        self.assertTrue(owner_page_slots < set(range(self.spec.page_capacity)))
        self.assertTrue(owner_state_slots < set(range(self.spec.state_slot_capacity)))
        for binding in (binding_a, binding_b):
            self.assertEqual(len(set(binding.scratch_page_slots)), self.spec.scratch_capacity)
            self.assertTrue(set(binding.scratch_page_slots).isdisjoint(owner_page_slots))
            for _, slots in binding.scratch_state_slots:
                self.assertEqual(len(set(slots)), self.spec.scratch_capacity)
                self.assertTrue(set(slots).isdisjoint(owner_state_slots))
        self.assertFalse(hasattr(binding_a, "owner_id"))
        transaction_a.abort()
        transaction_b.abort()

    def test_binding_exports_read_only_and_cow_generation_identity(self) -> None:
        store = self._store()
        write_page = self._provision(store)
        read_page = store.allocate_page("request:a")
        transaction = store.begin(
            store.plan(
                "txn:read-cow",
                "request:a",
                reads=(
                    store.kv_region("request:a", read_page, 0, (0,)),
                    store.kv_region("request:a", write_page, 0, (0,)),
                    store.recurrent_region("request:a", "conv:0"),
                    store.recurrent_region("request:a", "recurrent:0"),
                ),
                writes=(
                    store.kv_region("request:a", write_page, 0, (0,)),
                    store.recurrent_region("request:a", "conv:0"),
                ),
                target_position=1,
            )
        )
        launch = self._launch("lease:0")
        binding = transaction.graph_binding(launch)

        self.assertEqual(tuple(slot.resource_id for slot in binding.read_page_slots), (read_page,))
        self.assertEqual(
            tuple(slot.resource_id for slot in binding.read_state_slots),
            ("recurrent:0",),
        )
        committed_read = store._pages[read_page]
        read_slot = binding.read_page_slots[0]
        self.assertEqual(
            (read_slot.source_slot, read_slot.source_generation, read_slot.source_revision),
            (
                committed_read.slot.index,
                committed_read.slot.generation,
                committed_read.revision,
            ),
        )
        cow = {slot.resource_id: slot for slot in binding.page_slots}[write_page]
        source, destination = store._pages[write_page], transaction._staged_pages[write_page]
        self.assertEqual(
            (
                cow.source_slot,
                cow.source_generation,
                cow.source_revision,
                cow.destination_slot,
                cow.destination_generation,
                cow.destination_revision,
            ),
            (
                source.slot.index,
                source.slot.generation,
                source.revision,
                destination.slot.index,
                destination.slot.generation,
                source.revision + 1,
            ),
        )
        self.assertEqual(binding.scratch_lease_id, launch.replay_lease.lease_id)
        self.assertEqual(binding.scratch_lease_epoch, launch.replay_lease.epoch)
        self.assertEqual(binding.scratch_launch_plan_id, launch.launch_plan_id)
        transaction.abort()

    def test_scratch_claim_rejects_active_foreign_stale_and_oversize_launch(self) -> None:
        store = self._store()
        page_a = self._provision(store)
        page_b = self._provision(store, "request:b")
        transaction_a = self._transaction(store, page_a, "txn:scratch:a")
        transaction_b = self._transaction(store, page_b, "txn:scratch:b", owner="request:b")
        active_launch = self._launch("lease:0")
        transaction_a.graph_binding(active_launch)
        with self.assertRaisesRegex(ScratchArenaLeaseError, "active replay lease"):
            transaction_b.graph_binding(self._launch("lease:0"))
        transaction_a.abort()

        with self.assertRaisesRegex(ScratchArenaLeaseError, "foreign replay resource"):
            transaction_b.graph_binding(self._launch("lease:0", resource_domain="foreign"))
        with self.assertRaisesRegex(ScratchArenaLeaseError, "stale"):
            transaction_b.graph_binding(
                self._launch("lease:0", epoch=active_launch.replay_lease.epoch)
            )
        with self.assertRaisesRegex(ScratchArenaLeaseError, "exceeds"):
            transaction_b.graph_binding(self._launch("lease:1", logical_m=1, captured_m=4))
        with self.assertRaisesRegex(ScratchArenaLeaseError, "cache-store rank"):
            transaction_b.graph_binding(self._launch("lease:1", rank=1))
        transaction_b.graph_binding(self._launch("lease:0"))
        transaction_b.abort()

    def test_one_launch_shares_scratch_until_its_last_participant_finishes(self) -> None:
        store = self._store()
        pages = {
            owner: self._provision(store, owner)
            for owner in ("request:a", "request:b", "request:c")
        }
        launch = self._launch("lease:0")
        first = self._transaction(store, pages["request:a"], "txn:hold:a")
        second = self._transaction(store, pages["request:b"], "txn:hold:b", owner="request:b")
        waiting = self._transaction(store, pages["request:c"], "txn:hold:c", owner="request:c")
        first.graph_binding(launch)
        second.graph_binding(launch)
        first.abort()
        with self.assertRaisesRegex(ScratchArenaLeaseError, "active replay lease"):
            waiting.graph_binding(self._launch("lease:0"))
        second.abort()
        waiting.graph_binding(self._launch("lease:0"))
        waiting.abort()

    def test_arena_serializes_owner_and_pins_source_generation(self) -> None:
        store = self._store()
        page = self._provision(store)
        first = self._transaction(store, page, "txn:owner:first")
        binding = first.graph_binding(self._launch("lease:0"))
        source = binding.page_slots[0]
        with self.assertRaisesRegex(TransactionStateError, "one active mutation"):
            self._transaction(store, page, "txn:owner:overlap")

        self._emulate_graph(binding, page, 4.0)
        first.accept_graph_writes()
        first.verify()
        first.commit()
        second = self._transaction(store, page, "txn:owner:second")
        rebound = second.graph_binding(self._launch("lease:0")).page_slots[0]
        self.assertEqual(rebound.source_slot, source.destination_slot)
        self.assertEqual(rebound.source_generation, source.destination_generation)
        second.abort()

    def test_partial_arena_exhaustion_releases_every_reserved_slot(self) -> None:
        spec = CacheArenaSpec(
            "arena:partial",
            2,
            1,
            (StateSlotSpec("state:0", torch.float32, (1,)),),
            1,
            ("lease:0",),
        )
        store = CacheStateStore(self.layout, rank_id=0, arena_spec=spec)
        store.provision_owner(
            "request:a",
            (RecurrentStateProvision("state:0", torch.zeros((1,))),),
        )
        page = store.allocate_page("request:a")
        plan = store.plan(
            "txn:partial-exhaustion",
            "request:a",
            writes=(
                store.kv_region("request:a", page, 0, (0,)),
                store.recurrent_region("request:a", "state:0"),
            ),
            target_position=1,
        )
        with self.assertRaises(ArenaExhausted):
            store.begin(plan)
        retry = store.begin(
            store.plan(
                "txn:partial-exhaustion",
                "request:a",
                writes=(store.kv_region("request:a", page, 0, (0,)),),
            )
        )
        self.assertEqual(retry._staged_pages[page].slot.index, 1)
        self.assertEqual(retry._staged_pages[page].slot.generation, 2)
        retry.abort()

    def test_graph_commit_is_metadata_only_and_backing_addresses_stay_stable(self) -> None:
        store = self._store()
        page = self._provision(store)
        committed_page_slot = store._pages[page].slot
        committed_state_slots = {
            state_id: store._states[("request:a", state_id)].slot
            for state_id in ("conv:0", "recurrent:0")
        }
        with (
            mock.patch.object(torch.Tensor, "clone", side_effect=AssertionError("clone")),
            mock.patch.object(torch.Tensor, "copy_", side_effect=AssertionError("copy")),
            mock.patch.object(torch.Tensor, "zero_", side_effect=AssertionError("zero")),
            mock.patch.object(torch, "empty", side_effect=AssertionError("allocate")),
            mock.patch.object(torch, "zeros", side_effect=AssertionError("allocate")),
            mock.patch.object(torch, "tensor", side_effect=AssertionError("allocate")),
            mock.patch.object(torch, "empty_like", side_effect=AssertionError("allocate")),
            mock.patch.object(torch, "zeros_like", side_effect=AssertionError("allocate")),
        ):
            transaction = self._transaction(store, page, "txn:commit")
            binding = transaction.graph_binding(self._launch("lease:0"))
        pointers = (
            binding.key.data_ptr(),
            binding.value.data_ptr(),
            binding.written.data_ptr(),
            *(tensor.data_ptr() for _, tensor in binding.state_tensors),
        )
        page_slot = {slot.resource_id: slot for slot in binding.page_slots}[page]
        self.assertEqual(page_slot.source_slot, committed_page_slot.index)
        self._emulate_graph(binding, page, 3.0)

        with (
            mock.patch.object(torch.Tensor, "clone", side_effect=AssertionError("clone")),
            mock.patch.object(torch.Tensor, "copy_", side_effect=AssertionError("copy")),
            mock.patch.object(torch.Tensor, "zero_", side_effect=AssertionError("zero")),
            mock.patch.object(torch.Tensor, "resize_", side_effect=AssertionError("resize")),
            mock.patch.object(torch.Tensor, "set_", side_effect=AssertionError("set")),
            mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("sync")),
            mock.patch.object(torch, "isfinite", side_effect=AssertionError("scan")),
            mock.patch.object(torch, "empty", side_effect=AssertionError("allocate")),
            mock.patch.object(torch, "zeros", side_effect=AssertionError("allocate")),
            mock.patch.object(torch, "tensor", side_effect=AssertionError("allocate")),
            mock.patch.object(torch, "empty_like", side_effect=AssertionError("allocate")),
            mock.patch.object(torch, "zeros_like", side_effect=AssertionError("allocate")),
        ):
            transaction.accept_graph_writes()
            with self.assertRaisesRegex(TransactionStateError, "only for an open"):
                transaction.graph_binding(self._launch("lease:0"))
            transaction.verify()
            self.assertEqual(transaction.commit(), 1)

        self.assertEqual(
            pointers,
            (
                binding.key.data_ptr(),
                binding.value.data_ptr(),
                binding.written.data_ptr(),
                *(tensor.data_ptr() for _, tensor in binding.state_tensors),
            ),
        )
        self.assertEqual(store._pages[page].slot.index, page_slot.destination_slot)
        state_slots = {slot.resource_id: slot for slot in binding.state_slots}
        for state_id, old_slot in committed_state_slots.items():
            self.assertEqual(
                store._states[("request:a", state_id)].slot.index,
                state_slots[state_id].destination_slot,
            )
            self.assertNotEqual(store._states[("request:a", state_id)].slot, old_slot)
        key, value = store.read_kv("request:a", page, 0, (0,))
        self.assertTrue(torch.equal(key, torch.full_like(key, 3.0)))
        self.assertTrue(torch.equal(value, torch.full_like(value, -3.0)))
        self.assertEqual(store.read_recurrent_state("request:a", "conv:0").dtype, torch.bfloat16)
        self.assertEqual(
            store.read_recurrent_state("request:a", "recurrent:0").dtype,
            torch.float32,
        )
        self.assertEqual(store.state_position("request:a", "conv:0"), 1)
        self.assertTrue(
            torch.equal(
                store.read_recurrent_state("request:a", "conv:0"),
                torch.full((2, 3), 3.0, dtype=torch.bfloat16),
            )
        )
        self.assertTrue(
            torch.equal(
                store.read_recurrent_state("request:a", "recurrent:0"),
                torch.full((2, 2), 3.0, dtype=torch.float32),
            )
        )

        reused = self._transaction(store, page, "txn:reuse")
        reused_binding = reused.graph_binding(self._launch("lease:0"))
        reused_page_slot = {slot.resource_id: slot for slot in reused_binding.page_slots}[page]
        self.assertEqual(reused_page_slot.destination_slot, committed_page_slot.index)
        self.assertEqual(reused_binding.key.data_ptr(), binding.key.data_ptr())
        reused.abort()
        with self.assertRaisesRegex(TransactionStateError, "only for an open"):
            reused.graph_binding(self._launch("lease:0"))

    def test_arena_commit_bundle_is_all_or_none(self) -> None:
        def prepared(store: CacheStateStore, suffix: str):
            transactions = []
            pages = {}
            launch = self._launch("lease:0")
            for index, owner in enumerate(("request:a", "request:b")):
                page = self._provision(store, owner)
                pages[owner] = page
                transaction = self._transaction(
                    store, page, f"txn:bundle:{suffix}:{owner}", owner=owner
                )
                binding = transaction.graph_binding(launch)
                self._emulate_graph(binding, page, float(index + 1))
                transaction.accept_graph_writes()
                transaction.verify()
                transactions.append(transaction)
            return pages, transactions

        failing_store = self._store()
        pages, transactions = prepared(failing_store, "failure")
        corrupted = transactions[1]._staged_pages[pages["request:b"]]
        corrupted.owner_id = "request:foreign"
        with self.assertRaisesRegex(AccessViolation, "metadata changed"):
            failing_store.commit_bundle(tuple(transactions))
        self.assertEqual(tuple(failing_store.current_revision(owner) for owner in pages), (0, 0))
        self.assertEqual(
            tuple(transaction.status for transaction in transactions), ("verified",) * 2
        )
        corrupted.owner_id = "request:b"
        for transaction in transactions:
            transaction.abort()

        committed_store = self._store()
        pages, transactions = prepared(committed_store, "success")
        self.assertEqual(committed_store.commit_bundle(tuple(transactions)), (1, 1))
        self.assertEqual(tuple(committed_store.current_revision(owner) for owner in pages), (1, 1))

    def test_graph_commit_mutates_only_touched_metadata(self) -> None:
        class CountingDict(dict):
            def __init__(self, values):
                super().__init__(values)
                self.mutations = 0

            def __setitem__(self, key, value):
                self.mutations += 1
                return super().__setitem__(key, value)

            def __delitem__(self, key):
                self.mutations += 1
                return super().__delitem__(key)

            def pop(self, key, *default):
                self.mutations += 1
                return super().pop(key, *default)

            def copy(self):
                raise AssertionError("global mapping copy")

        store = CacheStateStore(
            self.layout,
            rank_id=0,
            arena_spec=CacheArenaSpec("arena:scale", 66, 0, (), 1, ("lease:0",)),
        )
        pages = {}
        for index in range(64):
            owner = f"request:{index}"
            store.provision_owner(owner)
            pages[owner] = store.allocate_page(owner)
        owner, page = "request:0", pages["request:0"]
        transaction = store.begin(
            store.plan(
                "txn:scale",
                owner,
                writes=(store.kv_region(owner, page, 0, (0,)),),
            )
        )
        binding = transaction.graph_binding(self._launch("lease:0"))
        self._emulate_graph(binding, page, 5.0)
        transaction.accept_graph_writes()
        transaction.verify()

        store._pages = CountingDict(store._pages)
        store._states = CountingDict(store._states)
        store._owner_revisions = CountingDict(store._owner_revisions)
        store._active_transactions = CountingDict(store._active_transactions)
        store._active_by_owner = CountingDict(store._active_by_owner)
        mapping_ids = tuple(
            id(mapping)
            for mapping in (
                store._pages,
                store._states,
                store._owner_revisions,
                store._active_transactions,
                store._active_by_owner,
            )
        )
        transaction.commit()
        self.assertEqual(
            mapping_ids,
            tuple(
                id(mapping)
                for mapping in (
                    store._pages,
                    store._states,
                    store._owner_revisions,
                    store._active_transactions,
                    store._active_by_owner,
                )
            ),
        )
        self.assertEqual(
            tuple(
                mapping.mutations
                for mapping in (
                    store._pages,
                    store._states,
                    store._owner_revisions,
                    store._active_transactions,
                    store._active_by_owner,
                )
            ),
            (1, 0, 1, 1, 1),
        )

    def test_abort_preserves_bytes_mapping_and_recycles_generation(self) -> None:
        store = self._store()
        page = self._provision(store)
        initial = self._transaction(store, page, "txn:initial")
        self._write(initial, page, 2.0)
        initial.verify()
        initial.commit()

        committed_slot = store._pages[page].slot
        committed_key = store.read_kv("request:a", page, 0, (0,))[0]
        aborted = self._transaction(store, page, "txn:abort")
        binding = aborted.graph_binding(self._launch("lease:1"))
        tentative = aborted._staged_pages[page].slot
        self._emulate_graph(binding, page, 9.0)
        with (
            mock.patch.object(torch.Tensor, "clone", side_effect=AssertionError("clone")),
            mock.patch.object(torch.Tensor, "copy_", side_effect=AssertionError("copy")),
            mock.patch.object(torch, "isfinite", side_effect=AssertionError("scan")),
        ):
            aborted.accept_graph_writes()
            aborted.abort()

        self.assertEqual(store._pages[page].slot, committed_slot)
        self.assertTrue(torch.equal(store.read_kv("request:a", page, 0, (0,))[0], committed_key))
        reused = self._transaction(store, page, "txn:after-abort")
        reused_slot = reused._staged_pages[page].slot
        self.assertEqual(reused_slot.index, tentative.index)
        self.assertGreater(reused_slot.generation, tentative.generation)
        self.assertEqual(
            reused.graph_binding(self._launch("lease:1")).key.data_ptr(),
            binding.key.data_ptr(),
        )
        reused.abort()
        with self.assertRaisesRegex(TransactionStateError, "only for an open"):
            aborted.graph_binding(self._launch("lease:1"))

    def test_exhaustion_stale_generation_and_double_release_fail_typed(self) -> None:
        page_only = CacheStateStore(
            self.layout,
            rank_id=0,
            arena_spec=CacheArenaSpec("arena:small", 1, 0, (), 2, ("lease:0",)),
        )
        page = page_only.allocate_page("request:a")
        with self.assertRaises(ArenaExhausted):
            page_only.allocate_page("request:b")
        with self.assertRaises(ArenaExhausted):
            page_only.begin(
                page_only.plan(
                    "txn:no-cow",
                    "request:a",
                    writes=(page_only.kv_region("request:a", page, 0, (0,)),),
                )
            )
        self.assertEqual(page_only.current_revision("request:a"), 0)

        store = self._store()
        mapped_page = store.allocate_page("request:generation")
        arena = store._arena
        old_page = store._pages[mapped_page]
        arena.release_page(old_page)
        replacement = arena.reserve_page(
            page_id=99,
            owner_rank=0,
            owner_id="request:replacement",
        )
        self.assertEqual(replacement.slot.index, old_page.slot.index)
        self.assertGreater(replacement.slot.generation, old_page.slot.generation)
        with self.assertRaises(StaleArenaSlot):
            store.page_revision("request:generation", mapped_page)
        with self.assertRaises(ArenaPoisoned):
            arena.release_page(old_page)
        with self.assertRaises(ArenaPoisoned):
            arena.reserve_page(page_id=100, owner_rank=0, owner_id="request:blocked")

        state_limited = CacheStateStore(
            self.layout,
            rank_id=0,
            arena_spec=CacheArenaSpec(
                "arena:state-small",
                1,
                1,
                (StateSlotSpec("recurrent:0", torch.float32, (1,)),),
                2,
                ("lease:0",),
            ),
        )
        state_limited.register_recurrent_state(
            "request:a", "recurrent:0", torch.zeros((1,)), position=0
        )
        with self.assertRaises(ArenaExhausted):
            state_limited.register_recurrent_state(
                "request:b", "recurrent:0", torch.zeros((1,)), position=0
            )

    def test_tentative_generation_mismatch_blocks_graph_binding(self) -> None:
        store = self._store()
        page = self._provision(store)
        transaction = self._transaction(store, page, "txn:stale-tentative")
        staged = transaction._staged_pages[page]
        arena = store._arena
        arena.release_page(staged)
        replacement = arena.reserve_page(
            page_id=page,
            owner_rank=0,
            owner_id="request:a",
        )
        self.assertEqual(replacement.slot.index, staged.slot.index)
        with self.assertRaises(StaleArenaSlot):
            transaction.graph_binding(self._launch("lease:0"))
        transaction._staged_pages[page] = replacement

        staged_state = transaction._staged_states["conv:0"]
        arena.release_state("conv:0", staged_state.slot)
        replacement_slot, _ = arena.reserve_state("conv:0")
        self.assertEqual(replacement_slot.index, staged_state.slot.index)
        self.assertGreater(replacement_slot.generation, staged_state.slot.generation)
        with self.assertRaises(StaleArenaSlot):
            transaction.graph_binding(self._launch("lease:0"))
        staged_state.slot = replacement_slot
        transaction.abort()

    def test_arena_schema_rejects_wrong_state_and_kv_dtypes(self) -> None:
        store = self._store()
        with self.assertRaisesRegex(ValueError, "arena schema"):
            store.register_recurrent_state("request:bad", "conv:0", torch.zeros((2, 3)), position=0)
        with self.assertRaisesRegex(ValueError, "absent"):
            store.register_recurrent_state(
                "request:bad", "unknown", torch.zeros((2, 2)), position=0
            )

        page = self._provision(store)
        transaction = self._transaction(store, page, "txn:bad-kv")
        with self.assertRaisesRegex(ValueError, "storage dtype"):
            transaction.write_kv(
                page,
                0,
                (0,),
                torch.zeros((1, 1, 2), dtype=torch.float16),
                torch.zeros((1, 1, 2), dtype=torch.float16),
            )
        transaction.abort()

        unbound = self._transaction(store, page, "txn:unbound")
        with self.assertRaisesRegex(TransactionStateError, "sealed graph binding"):
            unbound.accept_graph_writes()
        unbound.abort()


if __name__ == "__main__":
    unittest.main()
