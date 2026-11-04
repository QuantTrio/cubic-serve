from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

import torch

from cubic_serve.runtime.cache import (
    ArenaSlotRef,
    ArenaSlotReleaseError,
    CacheArena,
    CacheArenaSpec,
    CacheStateStore,
    KVLayout,
    OwnershipError,
    PhysicalPrefixConflict,
    PrefixBundleCache,
    PrefixBundleManifest,
    PrefixExtensionComponent,
    PrefixExtensionComponentSpec,
    PrefixExtensionSchema,
    RankPrefixExtension,
    RankPrefixExtensionPayload,
    RankPrefixSource,
    RecurrentStateProvision,
    StaleArenaSlot,
    StateSlotSpec,
    prefix_extension_content_digest,
)


class PhysicalPrefixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = KVLayout((0,), 2, 1, 2)
        self.spec = CacheArenaSpec(
            "arena:physical",
            4,
            4,
            (
                StateSlotSpec("conv:0", torch.bfloat16, (2, 2)),
                StateSlotSpec("recurrent:0", torch.float32, (2, 2)),
            ),
            1,
            ("graph:0", "graph:1"),
        )

    def _store(self, rank: int) -> tuple[CacheArena, CacheStateStore]:
        arena = CacheArena(self.spec, self.layout, "cpu", rank_id=rank)
        return arena, CacheStateStore(self.layout, rank_id=rank, arena=arena)

    def _source(
        self, store: CacheStateStore, owner: str, *, value: float | None = None
    ) -> RankPrefixSource:
        scalar = float(store.rank_id + 1) if value is None else value
        store.provision_owner(
            owner,
            (
                RecurrentStateProvision(
                    "conv:0",
                    torch.full((2, 2), scalar + 2, dtype=torch.bfloat16),
                    position=2,
                ),
                RecurrentStateProvision(
                    "recurrent:0",
                    torch.full((2, 2), scalar + 4, dtype=torch.float32),
                    position=2,
                ),
            ),
        )
        page = store.allocate_page(owner)
        txn = store.begin(
            store.plan(
                f"fill:{store.rank_id}:{owner}",
                owner,
                writes=(store.kv_region(owner, page, 0, (0, 1)),),
            )
        )
        payload = torch.full((2, 1, 2), scalar, dtype=torch.bfloat16)
        txn.write_kv(page, 0, (0, 1), payload, payload)
        txn.verify()
        txn.commit()
        return RankPrefixSource(
            store.rank_id,
            owner,
            store.current_revision(owner),
            (page,),
            ("conv:0", "recurrent:0"),
        )

    def _manifest(self) -> PrefixBundleManifest:
        return PrefixBundleManifest("prefix:x", "topology:2", (0, 1), 2)

    def _extension_schema(self, *, semantic: str = "a") -> PrefixExtensionSchema:
        return PrefixExtensionSchema(
            "mtp",
            "cubic.test.mtp.v1",
            f"b2s256:{semantic * 64}",
            (
                PrefixExtensionComponentSpec("draft_kv", "sequence", 1, 16),
                PrefixExtensionComponentSpec("target_hidden", "boundary", 0),
            ),
        )

    def _rank_extension(self, rank: int, *, boundary: int = 2) -> RankPrefixExtension:
        payload = self._rank_extension_payload(rank, boundary=boundary)
        return RankPrefixExtension(
            rank,
            boundary,
            (
                PrefixExtensionComponent(
                    "draft_kv",
                    "sequence",
                    boundary,
                    1,
                    boundary - 1,
                    0,
                    boundary - 1,
                    16,
                    (boundary - 2) % 16 + 1,
                    "private_immutable_blob",
                    prefix_extension_content_digest("draft_kv", payload.components[0][1]),
                ),
                PrefixExtensionComponent(
                    "target_hidden",
                    "boundary",
                    boundary,
                    0,
                    1,
                    boundary,
                    boundary + 1,
                    1,
                    1,
                    "private_immutable_blob",
                    prefix_extension_content_digest(
                        "target_hidden", payload.components[1][1]
                    ),
                ),
            ),
        )

    @staticmethod
    def _rank_extension_payload(
        rank: int, *, boundary: int = 2
    ) -> RankPrefixExtensionPayload:
        return RankPrefixExtensionPayload(
            rank,
            (
                (
                    "draft_kv",
                    (torch.arange(boundary - 1, dtype=torch.bfloat16) + rank,),
                ),
                (
                    "target_hidden",
                    (torch.full((2, 2), rank + 1, dtype=torch.bfloat16),),
                ),
            ),
        )

    def _mtp_payloads(
        self, manifest: PrefixBundleManifest
    ) -> tuple[RankPrefixExtensionPayload, ...]:
        return tuple(
            self._rank_extension_payload(rank, boundary=manifest.token_boundary)
            for rank in manifest.ranks
        )

    def _mtp_manifest(
        self, key: str = "prefix:mtp", *, semantic: str = "a"
    ) -> PrefixBundleManifest:
        schema = self._extension_schema(semantic=semantic)
        return PrefixBundleManifest(
            key,
            "topology:2",
            (0, 1),
            2,
            "mtp",
            (self._rank_extension(0), self._rank_extension(1)),
            schema,
        )

    def test_rank_global_arena_injection_preserves_every_backing_pointer(self) -> None:
        arena = CacheArena(self.spec, self.layout, "cpu", rank_id=0)
        first = CacheStateStore(self.layout, rank_id=0, arena=arena)
        second = CacheStateStore(self.layout, rank_id=0, arena=arena)
        self.assertEqual(first.arena_backing_identity, arena.backing_identity)
        self.assertEqual(second.arena_backing_identity, arena.backing_identity)
        with self.assertRaisesRegex(ValueError, "rank/layout/device"):
            CacheStateStore(self.layout, rank_id=1, arena=arena)

    def test_unique_hold_capabilities_reject_foreign_double_and_stale(self) -> None:
        arena = CacheArena(
            CacheArenaSpec("arena:holds", 1, 0, (), 1, ("graph:0",)),
            self.layout,
            "cpu",
            rank_id=0,
        )
        page = arena.reserve_page(page_id=0, owner_rank=0, owner_id="owner", zero=True)
        assert page.slot is not None
        retained = arena.retain_page(page.slot)
        foreign = ArenaSlotRef(retained.index, retained.generation, retained.hold_id + 999)
        with self.assertRaises(ArenaSlotReleaseError):
            arena.validate_page_ref(foreign)
        arena.release_page(page)
        arena.validate_page_ref(retained)
        arena.release_page_ref(retained)
        with self.assertRaises(ArenaSlotReleaseError):
            arena.release_page_ref(retained)

        fresh = CacheArena(
            CacheArenaSpec("arena:stale", 1, 0, (), 1, ("graph:0",)),
            self.layout,
            "cpu",
            rank_id=0,
        )
        old = fresh.reserve_page(page_id=0, owner_rank=0, owner_id="old", zero=True)
        assert old.slot is not None
        old_ref = old.slot
        fresh.release_page(old)
        replacement = fresh.reserve_page(page_id=1, owner_rank=0, owner_id="new", zero=True)
        with self.assertRaises(StaleArenaSlot):
            fresh.validate_page_ref(old_ref)
        fresh.release_page(replacement)

    def test_publish_failure_rolls_back_every_earlier_rank_and_is_a_miss(self) -> None:
        _, store0 = self._store(0)
        _, store1 = self._store(1)
        source0 = self._source(store0, "source:0")
        source1 = self._source(store1, "source:1")
        cache = PrefixBundleCache((store0, store1))
        incomplete = RankPrefixSource(
            source1.rank_id,
            source1.owner_id,
            source1.owner_revision,
            source1.page_ids,
            ("conv:0",),
        )
        with self.assertRaisesRegex(ValueError, "every arena state component"):
            cache.publish(self._manifest(), (source0, incomplete))
        self.assertIsNone(cache.acquire("prefix:x", topology_id="topology:2"))

        store0.retire_owner(
            source0.owner_id,
            expected_revision=source0.owner_revision,
            expected_page_ids=source0.page_ids,
        )
        store1.retire_owner(
            source1.owner_id,
            expected_revision=source1.owner_revision,
            expected_page_ids=source1.page_ids,
        )
        store0.allocate_page("fresh:0")
        store1.allocate_page("fresh:1")

    def test_digest_failure_precedes_retains_and_all_slots_are_reusable(self) -> None:
        _, store0 = self._store(0)
        _, store1 = self._store(1)
        sources = (self._source(store0, "source:0"), self._source(store1, "source:1"))
        cache = PrefixBundleCache((store0, store1))
        with (
            mock.patch.object(
                store0, "_physical_prefix_digest", side_effect=RuntimeError("digest injected")
            ),
            self.assertRaisesRegex(RuntimeError, "digest injected"),
        ):
            cache.publish(self._manifest(), sources)
        for store, source in zip((store0, store1), sources, strict=True):
            store.retire_owner(
                source.owner_id,
                expected_revision=source.owner_revision,
                expected_page_ids=source.page_ids,
            )
            self._source(store, f"reused:{store.rank_id}")

    def test_evict_with_lease_and_restored_owner_have_independent_holds(self) -> None:
        arenas_and_stores = (self._store(0), self._store(1))
        stores = tuple(store for _, store in arenas_and_stores)
        sources = tuple(self._source(store, f"source:{store.rank_id}") for store in stores)
        cache = PrefixBundleCache(stores)
        self.assertTrue(cache.publish(self._manifest(), sources))
        lease = cache.acquire("prefix:x", topology_id="topology:2")
        assert lease is not None
        self.assertTrue(cache.evict("prefix:x", topology_id="topology:2"))
        self.assertIsNone(cache.acquire("prefix:x", topology_id="topology:2"))
        restores = lease.restore(((0, "restored:0"), (1, "restored:1")))
        lease.release()

        for store, source in zip(stores, sources, strict=True):
            store.retire_owner(
                source.owner_id,
                expected_revision=source.owner_revision,
                expected_page_ids=source.page_ids,
            )
        for store, restore in zip(stores, restores, strict=True):
            page = restore.source_to_restored_pages[0][1]
            key, _ = store.read_kv(restore.owner_id, page, 0, (0, 1))
            self.assertTrue(torch.all(key == store.rank_id + 1))
            self.assertEqual(
                store.read_recurrent_state(restore.owner_id, "recurrent:0").dtype,
                torch.float32,
            )
            txn = store.begin(
                store.plan(
                    f"cow:{store.rank_id}",
                    restore.owner_id,
                    writes=(store.kv_region(restore.owner_id, page, 0, (0,)),),
                )
            )
            replacement = torch.full((1, 1, 2), 9, dtype=torch.bfloat16)
            txn.write_kv(page, 0, (0,), replacement, replacement)
            txn.verify()
            revision = txn.commit()
            store.retire_owner(
                restore.owner_id,
                expected_revision=revision,
                expected_page_ids=(page,),
            )
            with self.assertRaises(OwnershipError):
                store.current_revision(restore.owner_id)

    def test_cross_rank_restore_prepare_failure_has_no_visible_owner(self) -> None:
        _, store0 = self._store(0)
        _, store1 = self._store(1)
        sources = (self._source(store0, "source:0"), self._source(store1, "source:1"))
        cache = PrefixBundleCache((store0, store1))
        cache.publish(self._manifest(), sources)
        lease = cache.acquire("prefix:x", topology_id="topology:2")
        assert lease is not None
        with (
            mock.patch.object(
                store1, "prepare_prefix_restore", side_effect=RuntimeError("injected")
            ),
            self.assertRaisesRegex(RuntimeError, "injected"),
        ):
            lease.restore(((0, "failed:0"), (1, "failed:1")))
        for store, owner in ((store0, "failed:0"), (store1, "failed:1")):
            with self.assertRaises(OwnershipError):
                store.current_revision(owner)
        restores = lease.restore(((0, "restored:0"), (1, "restored:1")))
        lease.release()
        cache.evict("prefix:x", topology_id="topology:2")
        for store, restore in zip((store0, store1), restores, strict=True):
            store.retire_owner(
                restore.owner_id,
                expected_revision=restore.owner_revision,
                expected_page_ids=tuple(target for _, target in restore.source_to_restored_pages),
            )

    def test_cross_rank_activation_failure_rolls_back_already_active_rank(self) -> None:
        _, store0 = self._store(0)
        _, store1 = self._store(1)
        sources = (self._source(store0, "source:0"), self._source(store1, "source:1"))
        cache = PrefixBundleCache((store0, store1))
        cache.publish(self._manifest(), sources)
        lease = cache.acquire("prefix:x", topology_id="topology:2")
        assert lease is not None

        def fail_after_pages(prepared) -> None:
            store1._pages.update(prepared.pages)
            raise RuntimeError("activation injected")

        with (
            mock.patch.object(store1, "_publish_prepared_delta", side_effect=fail_after_pages),
            self.assertRaisesRegex(RuntimeError, "activation injected"),
        ):
            lease.restore(((0, "failed:0"), (1, "failed:1")))
        for store, owner in ((store0, "failed:0"), (store1, "failed:1")):
            with self.assertRaises(OwnershipError):
                store.current_revision(owner)
            self.assertNotIn(owner, store._owner_pages)
            self.assertFalse(any(page.owner_id == owner for page in store._pages.values()))
        lease.release()
        cache.evict("prefix:x", topology_id="topology:2")
        for store, source in zip((store0, store1), sources, strict=True):
            store.retire_owner(
                source.owner_id,
                expected_revision=source.owner_revision,
                expected_page_ids=source.page_ids,
            )
            self._source(store, f"reused:{store.rank_id}")

    def test_single_rank_mid_activation_failure_is_idempotently_cleaned(self) -> None:
        _, store = self._store(0)
        source = self._source(store, "source:0")
        manifest = PrefixBundleManifest("prefix:one", "topology:1", (0,), 2)
        cache = PrefixBundleCache((store,))
        cache.publish(manifest, (source,))
        lease = cache.acquire("prefix:one", topology_id="topology:1")
        assert lease is not None

        def fail_after_state(prepared) -> None:
            store._pages.update(prepared.pages)
            store._states.update(prepared.states)
            raise RuntimeError("mid activation")

        with (
            mock.patch.object(store, "_publish_prepared_delta", side_effect=fail_after_state),
            self.assertRaisesRegex(RuntimeError, "mid activation"),
        ):
            lease.restore(((0, "failed"),))
        with self.assertRaises(OwnershipError):
            store.current_revision("failed")
        self.assertFalse(any(page.owner_id == "failed" for page in store._pages.values()))
        self.assertFalse(any(owner == "failed" for owner, _ in store._states))
        lease.release()
        cache.evict("prefix:one", topology_id="topology:1")
        store.retire_owner(
            source.owner_id,
            expected_revision=source.owner_revision,
            expected_page_ids=source.page_ids,
        )
        self._source(store, "reused")

    def test_same_key_different_physical_content_is_persistently_quarantined(self) -> None:
        _, store0 = self._store(0)
        _, store1 = self._store(1)
        cache = PrefixBundleCache((store0, store1))
        first = (self._source(store0, "first:0"), self._source(store1, "first:1"))
        cache.publish(self._manifest(), first)
        duplicate = (
            self._source(store0, "duplicate:0"),
            self._source(store1, "duplicate:1"),
        )
        self.assertFalse(cache.publish(self._manifest(), duplicate))
        for store, source in zip((store0, store1), duplicate, strict=True):
            store.retire_owner(
                source.owner_id,
                expected_revision=source.owner_revision,
                expected_page_ids=source.page_ids,
            )
        old_lease = cache.acquire("prefix:x", topology_id="topology:2")
        assert old_lease is not None
        second = (
            self._source(store0, "different:0", value=7),
            self._source(store1, "different:1", value=8),
        )
        with self.assertRaises(PhysicalPrefixConflict):
            cache.publish(self._manifest(), second)
        with self.assertRaises(PhysicalPrefixConflict):
            cache.acquire("prefix:x", topology_id="topology:2")
        with self.assertRaises(PhysicalPrefixConflict):
            old_lease.restore(((0, "forbidden:0"), (1, "forbidden:1")))
        old_lease.release()
        with self.assertRaises(PhysicalPrefixConflict):
            cache.publish(self._manifest(), first)

    def test_restore_and_retire_mutate_only_touched_owner_entries(self) -> None:
        _, store0 = self._store(0)
        _, store1 = self._store(1)
        sources = (self._source(store0, "source:0"), self._source(store1, "source:1"))
        cache = PrefixBundleCache((store0, store1))
        cache.publish(self._manifest(), sources)
        mapping_ids = tuple(
            (id(store._pages), id(store._states), id(store._owner_revisions))
            for store in (store0, store1)
        )
        source_objects = tuple(
            id(store._pages[source.page_ids[0]])
            for store, source in zip((store0, store1), sources, strict=True)
        )
        lease = cache.acquire("prefix:x", topology_id="topology:2")
        assert lease is not None
        restores = lease.restore(((0, "restored:0"), (1, "restored:1")))
        lease.release()
        for index, (store, source, restore) in enumerate(
            zip((store0, store1), sources, restores, strict=True)
        ):
            self.assertEqual(
                mapping_ids[index],
                (id(store._pages), id(store._states), id(store._owner_revisions)),
            )
            self.assertEqual(source_objects[index], id(store._pages[source.page_ids[0]]))
            restored_pages = tuple(target for _, target in restore.source_to_restored_pages)
            store.retire_owner(
                restore.owner_id,
                expected_revision=restore.owner_revision,
                expected_page_ids=restored_pages,
            )
            self.assertEqual(source_objects[index], id(store._pages[source.page_ids[0]]))

    def test_partial_boundary_fails_before_physical_holds(self) -> None:
        _, store0 = self._store(0)
        _, store1 = self._store(1)
        sources = (self._source(store0, "source:0"), self._source(store1, "source:1"))
        cache = PrefixBundleCache((store0, store1))
        with self.assertRaisesRegex(ValueError, "block aligned"):
            cache.publish(
                PrefixBundleManifest("prefix:partial", "topology:2", (0, 1), 1),
                sources,
            )

    def test_extension_identity_is_required_for_whole_bundle_hit(self) -> None:
        _, store0 = self._store(0)
        _, store1 = self._store(1)
        sources = (self._source(store0, "source:0"), self._source(store1, "source:1"))
        cache = PrefixBundleCache((store0, store1))
        manifest = self._mtp_manifest()
        target_manifest = replace(self._manifest(), prefix_key="prefix:mtp")
        self.assertTrue(cache.publish(target_manifest, sources))
        self.assertTrue(
            cache.publish(
                manifest, sources, extension_payloads=self._mtp_payloads(manifest)
            )
        )
        target_lease = cache.acquire("prefix:mtp", topology_id="topology:2")
        assert target_lease is not None
        self.assertIsNone(target_lease.extension)
        target_lease.release()
        second_manifest = self._mtp_manifest(semantic="b")
        self.assertTrue(
            cache.publish(
                second_manifest,
                sources,
                extension_payloads=self._mtp_payloads(second_manifest),
            )
        )
        second_lease = cache.acquire(
            "prefix:mtp",
            topology_id="topology:2",
            extension_schema=self._extension_schema(semantic="b"),
        )
        assert second_lease is not None
        self.assertEqual(
            second_lease.extension,
            (second_manifest.extension_schema, second_manifest.extensions),
        )
        second_lease.release()
        lease = cache.acquire(
            "prefix:mtp",
            topology_id="topology:2",
            extension_schema=self._extension_schema(),
        )
        assert lease is not None
        self.assertEqual(lease.extension, (manifest.extension_schema, manifest.extensions))
        first_generation = lease.generation
        lease.release()
        self.assertTrue(
            cache.evict(
                "prefix:mtp",
                topology_id="topology:2",
                extension_schema=self._extension_schema(),
            )
        )
        self.assertTrue(
            cache.publish(
                manifest, sources, extension_payloads=self._mtp_payloads(manifest)
            )
        )
        replacement = cache.acquire(
            "prefix:mtp",
            topology_id="topology:2",
            extension_schema=self._extension_schema(),
        )
        assert replacement is not None
        self.assertEqual(replacement.generation, first_generation + 1)
        replacement.release()
        target_after = cache.acquire("prefix:mtp", topology_id="topology:2")
        other_after = cache.acquire(
            "prefix:mtp",
            topology_id="topology:2",
            extension_schema=self._extension_schema(semantic="b"),
        )
        assert target_after is not None and other_after is not None
        target_after.release()
        other_after.release()

    def test_qwen_style_extension_lag_and_partial_tail_at_full_target_pages(self) -> None:
        schema = self._extension_schema()
        for boundary in (16, 32):
            with self.subTest(boundary=boundary):
                extension = self._rank_extension(0, boundary=boundary)
                extension.validate(schema)
                sequence = extension.components[0]
                self.assertEqual(sequence.token_count, boundary - 1)
                self.assertEqual(sequence.position_end, boundary - 1)
                self.assertEqual(sequence.valid_tail_tokens, 15)
                self.assertEqual(extension.components[1].position_start, boundary)

    def test_invalid_extension_schema_fails_before_physical_holds(self) -> None:
        _, store0 = self._store(0)
        _, store1 = self._store(1)
        sources = (self._source(store0, "source:0"), self._source(store1, "source:1"))
        cache = PrefixBundleCache((store0, store1))
        manifest = self._mtp_manifest("prefix:invalid-schema")
        assert manifest.extension_schema is not None
        manifest = replace(
            manifest,
            extension_schema=replace(manifest.extension_schema, semantic_digest="invalid"),
        )
        with (
            mock.patch.object(store0, "pin_prefix_part", wraps=store0.pin_prefix_part) as pin0,
            mock.patch.object(store1, "pin_prefix_part", wraps=store1.pin_prefix_part) as pin1,
            self.assertRaises(ValueError),
        ):
            cache.publish(manifest, sources)
        pin0.assert_not_called()
        pin1.assert_not_called()

    def test_extension_variant_rejects_unhashable_bool_and_wrong_types_before_lookup(self) -> None:
        _, store0 = self._store(0)
        _, store1 = self._store(1)
        sources = (self._source(store0, "source:0"), self._source(store1, "source:1"))
        cache = PrefixBundleCache((store0, store1))
        schema = self._extension_schema()
        invalid_schemas = (
            replace(schema, components=list(schema.components)),  # type: ignore[arg-type]
            replace(
                schema,
                components=(replace(schema.components[0], boundary_lag_tokens=True),),
            ),
            replace(schema, components=(object(),)),  # type: ignore[arg-type]
            replace(schema, kind=object()),  # type: ignore[arg-type]
        )
        for invalid in invalid_schemas:
            with self.subTest(schema=invalid), self.assertRaises(ValueError):
                cache.acquire(
                    "prefix:canonical",
                    topology_id="topology:2",
                    extension_schema=invalid,
                )

        manifest = self._mtp_manifest("prefix:canonical")
        first = manifest.extensions[0]
        invalid_rank_extensions = (
            replace(first, rank_id=True),
            replace(first, components=list(first.components)),  # type: ignore[arg-type]
            replace(first, components=(object(),)),  # type: ignore[arg-type]
            replace(
                first,
                components=(
                    replace(first.components[0], token_count=True),
                    first.components[1],
                ),
            ),
            replace(
                first,
                components=(
                    replace(first.components[0], storage_kind=object()),  # type: ignore[arg-type]
                    first.components[1],
                ),
            ),
        )
        for invalid in invalid_rank_extensions:
            candidate = replace(
                manifest,
                prefix_key=f"prefix:canonical:{id(invalid)}",
                extensions=(invalid, manifest.extensions[1]),
            )
            with (
                self.subTest(extension=invalid),
                mock.patch.object(store0, "pin_prefix_part", wraps=store0.pin_prefix_part) as pin0,
                mock.patch.object(store1, "pin_prefix_part", wraps=store1.pin_prefix_part) as pin1,
                self.assertRaises(ValueError),
            ):
                cache.publish(candidate, sources)
            pin0.assert_not_called()
            pin1.assert_not_called()

    def test_invalid_extension_relation_quarantines_without_taking_holds(self) -> None:
        _, store0 = self._store(0)
        _, store1 = self._store(1)
        sources = (self._source(store0, "source:0"), self._source(store1, "source:1"))
        cache = PrefixBundleCache((store0, store1))
        base = self._mtp_manifest()
        target_manifest = replace(self._manifest(), prefix_key="prefix:variant-isolation")
        other = self._mtp_manifest("prefix:variant-isolation", semantic="b")
        self.assertTrue(cache.publish(target_manifest, sources))
        self.assertTrue(
            cache.publish(other, sources, extension_payloads=self._mtp_payloads(other))
        )
        first = base.extensions[0]
        invalid_extensions = (
            (
                "wrong-lag",
                replace(
                    first,
                    components=(
                        replace(first.components[0], boundary_lag_tokens=0),
                        first.components[1],
                    ),
                ),
            ),
            ("missing-hidden", replace(first, components=(first.components[0],))),
            (
                "wrong-tail",
                replace(
                    first,
                    components=(
                        replace(first.components[0], valid_tail_tokens=2),
                        first.components[1],
                    ),
                ),
            ),
        )
        for name, invalid in invalid_extensions:
            key = "prefix:variant-isolation"
            manifest = replace(base, prefix_key=key, extensions=(invalid, base.extensions[1]))
            with (
                self.subTest(name=name),
                mock.patch.object(store0, "pin_prefix_part", wraps=store0.pin_prefix_part) as pin0,
                mock.patch.object(store1, "pin_prefix_part", wraps=store1.pin_prefix_part) as pin1,
                self.assertRaises(ValueError),
            ):
                cache.publish(manifest, sources)
            pin0.assert_not_called()
            pin1.assert_not_called()
            with self.assertRaises(PhysicalPrefixConflict):
                cache.acquire(
                    key,
                    topology_id="topology:2",
                    extension_schema=self._extension_schema(),
                )
            target_lease = cache.acquire(key, topology_id="topology:2")
            assert target_lease is not None
            target_lease.release()
            other_lease = cache.acquire(
                key,
                topology_id="topology:2",
                extension_schema=self._extension_schema(semantic="b"),
            )
            assert other_lease is not None
            other_lease.release()


if __name__ == "__main__":
    unittest.main()
