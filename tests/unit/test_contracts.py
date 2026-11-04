from __future__ import annotations

import unittest
from dataclasses import replace
from typing import cast

from cubic_serve.contracts.execution import (
    ActiveLaunchRow,
    BatchPhase,
    BatchPlan,
    BatchShape,
    CandidateBackend,
    CandidateBinding,
    ExecutionMode,
    ExecutionPlan,
    GraphBucket,
    GraphSafetyContract,
    LaunchPlan,
    OpBinding,
    OpSignature,
    PaddingIsolation,
    PaddingLaunchRow,
    ParallelTopology,
    PlanSegment,
    PrefixPublishPlan,
    RankReplayResources,
    ReplayLease,
    StateAccess,
    StateTxn,
    SpeculativeIteration,
    TensorSpec,
)
from cubic_serve.contracts.model import (
    CacheSpec,
    ModalitySpec,
    ModelSpec,
    ParameterDTypeOverride,
    PrefixReplayPolicy,
    ScalarDType,
    StateSpec,
    TensorRole,
    WeightEncoding,
    WeightManifest,
    WeightVariantMeta,
)
from cubic_serve.contracts.request import (
    ContentPart,
    DecodePositionPlan,
    EncoderJob,
    InputPlan,
    MediaDescriptor,
    TokenPlacement,
    TokenPosition,
)
from tests.guardrail.operator_evidence_fixture import sealed_test_performance_plan


def _replay_lease(
    *,
    launch_plan_id: str,
    execution_plan_id: str,
    bucket_id: str,
    template_id: str,
    topology_id: str = "single",
    partition_id: str = "toy",
) -> ReplayLease:
    return ReplayLease(
        lease_id=f"lease:{launch_plan_id}",
        epoch=7,
        launch_plan_id=launch_plan_id,
        execution_plan_id=execution_plan_id,
        bucket_id=bucket_id,
        template_id=template_id,
        topology_id=topology_id,
        partition_id=partition_id,
        rank_resources=(
            RankReplayResources(
                rank=0,
                graph_instance_id="graph-instance:0",
                stream_lane_id="stream:0",
                buffer_set_id="buffers:0",
                workspace_arena_id="workspace:0",
                scratch_arena_id="scratch:0",
            ),
        ),
    )


class ModelContractTests(unittest.TestCase):
    def _weights(
        self,
        *overrides: ParameterDTypeOverride,
        complete: bool = True,
    ) -> WeightManifest:
        return WeightManifest(
            manifest_digest=f"b2s256:{'a' * 64}",
            encoding=WeightEncoding.DENSE,
            default_dtype=ScalarDType.BF16,
            tensor_count=10,
            total_numel=1000,
            complete=complete,
            dtype_overrides=overrides,
        )

    def test_full_bf16_includes_weight_and_kv_dtype(self) -> None:
        spec = ModelSpec(
            model_id="fixture",
            architecture="toy_dense",
            revision="b2s256:model",
            weights=self._weights(),
            cache=CacheSpec(
                kv_dtype=ScalarDType.BF16,
                page_size_tokens=16,
                layout="layer_page_head_token_dim",
                prefix_enabled=True,
            ),
            modalities=(
                ModalitySpec(name="text"),
                ModalitySpec(
                    name="image",
                    mime_prefixes=("image/",),
                    encoder_id="vision",
                    feature_hidden_size=128,
                ),
                ModalitySpec(
                    name="video",
                    mime_prefixes=("video/",),
                    encoder_id="vision",
                    feature_hidden_size=128,
                ),
            ),
            layer_count=2,
        )

        self.assertTrue(spec.is_full_bf16)
        spec.require_full_bf16()

    def test_encoded_modality_requires_a_declared_feature_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "declared together"):
            ModalitySpec(name="image", encoder_id="vision")
        with self.assertRaisesRegex(ValueError, "declared together"):
            ModalitySpec(name="image", feature_hidden_size=128)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            ModalitySpec(name="image", encoder_id="vision", feature_hidden_size=0)

    def test_full_bf16_rejects_non_bf16_kv(self) -> None:
        spec = ModelSpec(
            model_id="fixture",
            architecture="toy_dense",
            revision="b2s256:model",
            weights=self._weights(),
            cache=CacheSpec(
                kv_dtype=ScalarDType.FP8_E4M3FN,
                page_size_tokens=16,
                layout="paged",
                prefix_enabled=True,
            ),
            modalities=(ModalitySpec(name="text"),),
        )

        with self.assertRaisesRegex(ValueError, "BF16 attention KV"):
            spec.require_full_bf16()

    def test_full_bf16_manifest_allows_only_exact_declared_checkpoint_exceptions(self) -> None:
        stability = ParameterDTypeOverride(
            group_id="gdn-stability",
            role=TensorRole.STABILITY_PARAMETER,
            dtype=ScalarDType.FP32,
            tensor_count=2,
            total_numel=64,
            tensor_names_digest=f"b2s256:{'b' * 64}",
            reason="model-declared FP32 recurrence parameters",
        )
        metadata = ParameterDTypeOverride(
            group_id="ple-indices",
            role=TensorRole.METADATA,
            dtype=ScalarDType.INT64,
            tensor_count=1,
            total_numel=3,
            tensor_names_digest=f"b2s256:{'c' * 64}",
            reason="non-numerical index metadata",
        )
        self.assertTrue(self._weights(stability, metadata).is_full_bf16_weights)

        disguised_weight = ParameterDTypeOverride(
            group_id="fp32-matrix",
            role=TensorRole.PARAMETER,
            dtype=ScalarDType.FP32,
            tensor_count=1,
            total_numel=64,
            tensor_names_digest=f"b2s256:{'d' * 64}",
            reason="not an allowed BF16 baseline exception",
        )
        self.assertFalse(self._weights(disguised_weight).is_full_bf16_weights)


class ExecutionContractTests(unittest.TestCase):
    def test_speculative_iteration_keeps_next_root_out_of_target_state(self) -> None:
        for proposal_count in (1, 2, 3):
            drafts = tuple(range(101, 101 + proposal_count))
            for accepted_count in range(proposal_count + 1):
                with self.subTest(proposals=proposal_count, accepted=accepted_count):
                    iteration = SpeculativeIteration(
                        target_state_token_count=16,
                        root_token_id=100,
                        draft_token_ids=drafts,
                        accepted_draft_count=accepted_count,
                        replacement_token_id=200,
                    )

                    self.assertEqual(iteration.verify_token_ids, (100, *drafts))
                    self.assertEqual(
                        iteration.committed_target_token_ids,
                        (100, *drafts[:accepted_count]),
                    )
                    self.assertEqual(iteration.target_commit_token_count, 17 + accepted_count)
                    self.assertEqual(iteration.committed_target_row_count, 1 + accepted_count)
                    self.assertEqual(iteration.prefix_token_count, 17 + accepted_count)
                    self.assertEqual(
                        iteration.emitted_token_ids,
                        (*drafts[:accepted_count], 200),
                    )
                    self.assertEqual(iteration.next_root_token_id, 200)
                    self.assertNotIn(200, iteration.committed_target_token_ids)

    def test_speculative_iteration_rejects_ambiguous_or_invalid_boundaries(self) -> None:
        arguments = dict(
            target_state_token_count=16,
            root_token_id=100,
            draft_token_ids=(101,),
            accepted_draft_count=0,
            replacement_token_id=200,
        )
        invalid = (
            ("negative state", {"target_state_token_count": -1}),
            ("mutable drafts", {"draft_token_ids": [101]}),
            ("empty drafts", {"draft_token_ids": ()}),
            ("negative root", {"root_token_id": -1}),
            ("negative draft", {"draft_token_ids": (-1,)}),
            ("negative replacement", {"replacement_token_id": -1}),
            ("negative acceptance", {"accepted_draft_count": -1}),
            ("excess acceptance", {"accepted_draft_count": 2}),
        )
        for name, override in invalid:
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                SpeculativeIteration(**(arguments | override))

    def test_speculative_iteration_selects_only_exact_complete_page_checkpoint(self) -> None:
        for proposal_count in (1, 2, 3):
            for accepted_count in range(proposal_count + 1):
                committed_rows = accepted_count + 1
                for committed_boundary in (1, 15, 16, 17, 31, 32, 33):
                    if committed_boundary < committed_rows:
                        continue
                    with self.subTest(
                        proposals=proposal_count,
                        accepted=accepted_count,
                        boundary=committed_boundary,
                    ):
                        iteration = SpeculativeIteration(
                            target_state_token_count=committed_boundary - committed_rows,
                            root_token_id=100,
                            draft_token_ids=tuple(range(101, 101 + proposal_count)),
                            accepted_draft_count=accepted_count,
                            replacement_token_id=200,
                        )
                        physical = committed_boundary // 16 * 16
                        available = () if physical == 0 else (physical,)
                        expected = None if physical == 0 else physical
                        self.assertEqual(
                            iteration.prefix_checkpoint_boundary(
                                block_size_tokens=16,
                                available_target_boundaries=available,
                            ),
                            expected,
                        )
                        if physical:
                            self.assertIsNone(
                                iteration.prefix_checkpoint_boundary(
                                    block_size_tokens=16,
                                    available_target_boundaries=(physical - 1,),
                                )
                            )

    def test_speculative_prefix_checkpoint_inputs_fail_closed(self) -> None:
        iteration = SpeculativeIteration(15, 100, (101,), 0, 200)
        for block_size, checkpoints in (
            (0, (16,)),
            (16, [16]),
            (16, (16, 16)),
            (16, (17, 16)),
            (16, (-1,)),
        ):
            with self.subTest(block_size=block_size, checkpoints=checkpoints):
                with self.assertRaises(ValueError):
                    iteration.prefix_checkpoint_boundary(
                        block_size_tokens=block_size,
                        available_target_boundaries=checkpoints,  # type: ignore[arg-type]
                    )

    def test_single_node_tp_pp_ep_groups_are_explicit(self) -> None:
        topology = ParallelTopology.single_node(
            topology_id="8gpu:tp4:pp2:ep2",
            device_indices=tuple(range(8)),
            tp_size=4,
            pp_size=2,
            ep_size=4,
        )

        self.assertEqual(topology.world_size, 8)
        self.assertEqual(
            {group.group_id for group in topology.groups},
            {
                "tp:0",
                "tp:1",
                "pp:0",
                "pp:1",
                "pp:2",
                "pp:3",
                "ep:0",
                "ep:1",
            },
        )

    def test_batch_plan_owns_logical_m(self) -> None:
        plan = BatchPlan(
            plan_id="batch:1",
            phase=BatchPhase.MIXED,
            input_plan_ids=("request:a", "request:b"),
            segments=(
                PlanSegment(
                    request_id="request:a",
                    input_plan_digest=f"b2s256:{'a' * 64}",
                    execution_input_digest=f"b2s256:{'a' * 64}",
                    phase=BatchPhase.PREFILL,
                    produces_output=True,
                    token_start=0,
                    token_count=3,
                    token_ids=(10, 11, 12),
                    positions=tuple(TokenPosition((index,)) for index in range(3)),
                    cache_slots=(0, 1, 2),
                    cache_pages=(0,),
                ),
                PlanSegment(
                    request_id="request:b",
                    input_plan_digest=f"b2s256:{'b' * 64}",
                    execution_input_digest=f"b2s256:{'b' * 64}",
                    phase=BatchPhase.DECODE,
                    produces_output=True,
                    token_start=8,
                    token_count=1,
                    token_ids=(20,),
                    positions=(TokenPosition((8,)),),
                    cache_slots=(16,),
                    cache_pages=(1,),
                ),
            ),
            shape=BatchShape(logical_m=4),
            topology_id="single",
            partition_id="toy",
            state_transactions=(
                StateTxn(
                    transaction_id="txn:a",
                    request_id="request:a",
                    base_revision=0,
                    target_revision=1,
                    reads=(),
                    writes=(StateAccess("kv:request:a", 0, slots=(0, 1, 2), pages=(0,)),),
                    expected_owner_ranks=(0,),
                ),
                StateTxn(
                    transaction_id="txn:b",
                    request_id="request:b",
                    base_revision=0,
                    target_revision=1,
                    reads=(),
                    writes=(StateAccess("kv:request:b", 0, slots=(16,), pages=(1,)),),
                    expected_owner_ranks=(0,),
                ),
            ),
        )

        self.assertEqual(plan.shape.logical_m, 4)

    def test_capacity_profiles_are_canonical_and_reference_may_omit_them(self) -> None:
        self.assertEqual(BatchShape(logical_m=1).capacity_items, ())
        self.assertEqual(
            BatchShape(
                logical_m=1,
                capacity_items=(("cache_pages", 8), ("request_slots", 4)),
            ).capacity_items,
            (("cache_pages", 8), ("request_slots", 4)),
        )
        for items in (
            (("request_slots", 4), ("cache_pages", 8)),
            (("cache_pages", 8), ("cache_pages", 9)),
            (("cache-pages", 8),),
            (("cache_pages", 0),),
            (("cache_pages", True),),
        ):
            with self.subTest(items=items), self.assertRaises(ValueError):
                BatchShape(logical_m=1, capacity_items=items)

    def test_padded_launch_requires_explicit_private_rows(self) -> None:
        bucket = GraphBucket(
            bucket_id="decode:2",
            template_id="graph:decode:2",
            scope_id="model.forward",
            replay_abi_id="cubic.replay.decode.v1",
            topology_id="single",
            partition_id="toy",
            phase=BatchPhase.DECODE,
            capacity_items=(("request_slots", 2),),
            logical_m_values=(1, 2),
            captured_m=2,
            row_map_schema_id="cubic.rows.v1",
            scratch_schema_id="cubic.scratch.v1",
            graph_safety_contract_ids=("graph-safe:model-forward:v1",),
            evidence_id="evidence:bucket:decode:2",
        )
        with self.assertRaisesRegex(ValueError, "non-empty capacity profile"):
            replace(bucket, capacity_items=())
        with self.assertRaisesRegex(TypeError, "typed replay lease"):
            LaunchPlan(
                launch_plan_id="launch:1",
                execution_plan_id="performance:1",
                source_plan_id="batch:1",
                logical_m=1,
                bucket=bucket,
                rows=(ActiveLaunchRow(0), PaddingLaunchRow(0)),
                replay_lease=cast(ReplayLease, None),
            )

        launch = LaunchPlan(
            launch_plan_id="launch:2",
            execution_plan_id="performance:1",
            source_plan_id="batch:1",
            logical_m=1,
            bucket=bucket,
            rows=(PaddingLaunchRow(0), ActiveLaunchRow(0)),
            replay_lease=_replay_lease(
                launch_plan_id="launch:2",
                execution_plan_id="performance:1",
                bucket_id=bucket.bucket_id,
                template_id=bucket.template_id,
            ),
        )
        self.assertEqual(launch.active_physical_indices, (1,))

        full_launch = LaunchPlan(
            launch_plan_id="launch:full",
            execution_plan_id="performance:1",
            source_plan_id="batch:2",
            logical_m=2,
            bucket=bucket,
            rows=(ActiveLaunchRow(0), ActiveLaunchRow(1)),
            replay_lease=_replay_lease(
                launch_plan_id="launch:full",
                execution_plan_id="performance:1",
                bucket_id=bucket.bucket_id,
                template_id=bucket.template_id,
            ),
        )
        self.assertEqual(full_launch.active_physical_indices, (0, 1))

    def test_performance_bucket_cannot_disguise_an_exact_m_graph(self) -> None:
        with self.assertRaisesRegex(ValueError, "singleton exact-M graphs are forbidden"):
            GraphBucket(
                bucket_id="decode:exact-m:8",
                template_id="graph:decode:exact-m:8",
                scope_id="model.forward",
                replay_abi_id="cubic.replay.decode.v1",
                topology_id="single",
                partition_id="toy",
                phase=BatchPhase.DECODE,
                capacity_items=(("request_slots", 8),),
                logical_m_values=(8,),
                captured_m=8,
                row_map_schema_id="cubic.rows.v1",
                scratch_schema_id="cubic.scratch.v1",
                graph_safety_contract_ids=("graph-safe:model-forward:v1",),
                evidence_id="evidence:forbidden:exact-m:8",
            )

    def test_replay_lease_must_match_the_full_launch_identity(self) -> None:
        bucket = GraphBucket(
            bucket_id="decode:8",
            template_id="graph:decode:8",
            scope_id="model.forward",
            replay_abi_id="cubic.replay.decode.v1",
            topology_id="single",
            partition_id="toy",
            phase=BatchPhase.DECODE,
            capacity_items=(("request_slots", 8),),
            logical_m_values=tuple(range(1, 9)),
            captured_m=8,
            row_map_schema_id="cubic.rows.v1",
            scratch_schema_id="cubic.scratch.v1",
            graph_safety_contract_ids=("graph-safe:model-forward:v1",),
            evidence_id="evidence:decode:8",
        )
        with self.assertRaisesRegex(ValueError, "lease identity"):
            LaunchPlan(
                launch_plan_id="launch:decode:7",
                execution_plan_id="performance:decode",
                source_plan_id="batch:decode:7",
                logical_m=7,
                bucket=bucket,
                rows=(*tuple(ActiveLaunchRow(index) for index in range(7)), PaddingLaunchRow(0)),
                replay_lease=_replay_lease(
                    launch_plan_id="launch:other",
                    execution_plan_id="performance:decode",
                    bucket_id=bucket.bucket_id,
                    template_id=bucket.template_id,
                ),
            )

    def test_padded_launch_rejects_missing_or_aliased_rows(self) -> None:
        bucket = GraphBucket(
            bucket_id="mixed:4",
            template_id="graph:mixed:4",
            scope_id="model.forward",
            replay_abi_id="cubic.replay.mixed.v1",
            topology_id="single",
            partition_id="toy",
            phase=BatchPhase.MIXED,
            capacity_items=(("request_slots", 4),),
            logical_m_values=(1, 2, 3, 4),
            captured_m=4,
            row_map_schema_id="cubic.rows.v1",
            scratch_schema_id="cubic.scratch.v1",
            graph_safety_contract_ids=("graph-safe:model-forward:v1",),
            evidence_id="evidence:bucket:mixed:4",
        )
        for rows, error in (
            (
                (ActiveLaunchRow(0), ActiveLaunchRow(0), PaddingLaunchRow(0), PaddingLaunchRow(1)),
                "every logical row exactly once",
            ),
            (
                (ActiveLaunchRow(0), ActiveLaunchRow(1), PaddingLaunchRow(0), PaddingLaunchRow(0)),
                "unique dense scratch slot",
            ),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                LaunchPlan(
                    launch_plan_id=f"launch:{error}",
                    execution_plan_id="performance:1",
                    source_plan_id="batch:1",
                    logical_m=2,
                    bucket=bucket,
                    rows=rows,
                    replay_lease=_replay_lease(
                        launch_plan_id=f"launch:{error}",
                        execution_plan_id="performance:1",
                        bucket_id=bucket.bucket_id,
                        template_id=bucket.template_id,
                    ),
                )

    def test_ep_cannot_imply_an_undeclared_replica_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "replica axis"):
            ParallelTopology.single_node(
                topology_id="invalid:tp4:ep3",
                device_indices=tuple(range(4)),
                tp_size=4,
                pp_size=1,
                ep_size=3,
            )

    def test_hybrid_state_requires_aligned_prefix_replay_metadata(self) -> None:
        state = StateSpec(
            state_id="linear_attention",
            dtype=ScalarDType.FP32,
            layout="request_layer_head_state",
            shape_symbols=("request", "layer", "head", "state"),
            commit_alignment_tokens=16,
            prefix_replay=PrefixReplayPolicy.FROM_LAST_ALIGNED_COMMIT,
        )
        self.assertEqual(state.commit_alignment_tokens, 16)

    def test_reference_execution_rejects_cuda_candidate(self) -> None:
        signature = OpSignature(
            semantic_op="attention.full",
            phase=BatchPhase.PREFILL,
            inputs=(
                TensorSpec("query", ScalarDType.BF16, (1, 1, "m", "head_dim")),
                TensorSpec("key", ScalarDType.BF16, (1, 1, "context", "head_dim")),
                TensorSpec("value", ScalarDType.BF16, (1, 1, "context", "head_dim")),
                TensorSpec("query_ordinals", ScalarDType.INT64, (1, "m")),
                TensorSpec("key_ordinals", ScalarDType.INT64, (1, "context")),
            ),
            outputs=(TensorSpec("attended", ScalarDType.BF16, (1, 1, "m", "head_dim")),),
            accumulator_dtype=ScalarDType.FP32,
            layout="bshd",
            shape_items=(("head_dim", 128), ("m", 4)),
            cache_form="paged_bf16",
            state_form=None,
            topology_id="single",
        )
        candidate = CandidateBinding(
            candidate_id="flash",
            implementation_id="b2s256:binary",
            backend=CandidateBackend.CUDA,
            launch_signature="grid=(1,1,1)",
            workspace_bytes=0,
            evidence_id="evidence:1",
        )

        with self.assertRaisesRegex(ValueError, "performance candidates"):
            ExecutionPlan(
                execution_plan_id="reference:1",
                mode=ExecutionMode.REFERENCE,
                bindings=(OpBinding(signature=signature, candidate=candidate),),
            )

    def test_state_transaction_status_is_runtime_owned(self) -> None:
        plan = StateTxn(
            transaction_id="txn:prefix",
            request_id="request:a",
            base_revision=7,
            target_revision=8,
            reads=(StateAccess(resource_id="kv", owner_rank=0, pages=(0,)),),
            writes=(
                StateAccess(resource_id="kv", owner_rank=0, pages=(1,)),
                StateAccess(resource_id="recurrent", owner_rank=1, slots=(0,)),
            ),
            expected_owner_ranks=(0, 1),
            prefix_publish=PrefixPublishPlan(
                input_plan_id="request:a",
                prefix_token_count=23,
                state_commit_position=16,
                commit_alignment_tokens=16,
            ),
        )

        self.assertEqual(plan.prefix_publish.replay_token_count, 7)

    def test_transaction_requires_every_owner_write(self) -> None:
        with self.assertRaisesRegex(ValueError, "every expected owner"):
            StateTxn(
                transaction_id="txn:partial",
                request_id="request:a",
                base_revision=0,
                target_revision=1,
                reads=(),
                writes=(StateAccess(resource_id="kv", owner_rank=0, pages=(0,)),),
                expected_owner_ranks=(0, 1),
            )

    def test_transaction_rejects_multiple_regions_for_one_resource_owner(self) -> None:
        for collection in ("reads", "writes"):
            duplicate_regions = (
                StateAccess(resource_id="kv:request:a", owner_rank=0, pages=(0,)),
                StateAccess(resource_id="kv:request:a", owner_rank=0, pages=(1,)),
            )
            arguments = {
                "reads": duplicate_regions if collection == "reads" else (),
                "writes": (
                    duplicate_regions
                    if collection == "writes"
                    else (StateAccess("kv:request:a", 0, pages=(0,)),)
                ),
            }
            with (
                self.subTest(collection=collection),
                self.assertRaisesRegex(
                    ValueError,
                    f"one {collection[:-1]} region per resource/owner",
                ),
            ):
                StateTxn(
                    transaction_id=f"txn:duplicate:{collection}",
                    request_id="request:a",
                    base_revision=0,
                    target_revision=1,
                    reads=arguments["reads"],
                    writes=arguments["writes"],
                    expected_owner_ranks=(0,),
                )

    def test_reference_binding_is_a_distinct_contract(self) -> None:
        signature = OpSignature(
            semantic_op="linear",
            phase=BatchPhase.DECODE,
            inputs=(
                TensorSpec("inputs", ScalarDType.BF16, ("m", "k")),
                TensorSpec("weight", ScalarDType.BF16, ("n", "k")),
            ),
            outputs=(TensorSpec("result", ScalarDType.BF16, ("m", "n")),),
            accumulator_dtype=ScalarDType.FP32,
            layout="mk_kn",
            shape_items=(("k", 8), ("m", 1), ("n", 8)),
            cache_form=None,
            state_form=None,
            topology_id="single",
        )
        plan = ExecutionPlan(
            execution_plan_id="reference:linear",
            mode=ExecutionMode.REFERENCE,
            bindings=(
                OpBinding(
                    signature=signature,
                    reference_implementation_id="cubic.reference.linear.v1",
                ),
            ),
        )
        self.assertEqual(plan.mode, ExecutionMode.REFERENCE)

    def test_performance_rejects_reference_fallback(self) -> None:
        signature = OpSignature(
            semantic_op="linear",
            phase=BatchPhase.DECODE,
            inputs=(
                TensorSpec("inputs", ScalarDType.BF16, ("m", "k")),
                TensorSpec("weight", ScalarDType.BF16, ("n", "k")),
            ),
            outputs=(TensorSpec("result", ScalarDType.BF16, ("m", "n")),),
            accumulator_dtype=ScalarDType.FP32,
            layout="mk_kn",
            shape_items=(("k", 8), ("m", 1), ("n", 8)),
            cache_form=None,
            state_form=None,
            topology_id="single",
        )
        bucket = GraphBucket(
            bucket_id="decode:8",
            template_id="graph:decode:8",
            scope_id="model.forward",
            replay_abi_id="cubic.replay.decode.v1",
            topology_id="single",
            partition_id="toy",
            phase=BatchPhase.DECODE,
            capacity_items=(("request_slots", 8),),
            logical_m_values=tuple(range(1, 9)),
            captured_m=8,
            row_map_schema_id="cubic.rows.v1",
            scratch_schema_id="cubic.scratch.v1",
            graph_safety_contract_ids=("graph-safe:linear:v1",),
            evidence_id="evidence:graph:decode:8",
        )
        with self.assertRaisesRegex(ValueError, "reference fallback"):
            ExecutionPlan(
                execution_plan_id="performance:graph",
                mode=ExecutionMode.PERFORMANCE,
                bindings=(
                    OpBinding(
                        signature=signature,
                        reference_implementation_id="cubic.reference.linear.v1",
                    ),
                ),
                graph_buckets=(bucket,),
            )

    def test_performance_has_only_certified_graph_buckets_and_no_fallback(self) -> None:
        signature = OpSignature(
            semantic_op="linear",
            phase=BatchPhase.DECODE,
            inputs=(
                TensorSpec("inputs", ScalarDType.BF16, ("m", "k")),
                TensorSpec("weight", ScalarDType.BF16, ("n", "k")),
            ),
            outputs=(TensorSpec("result", ScalarDType.BF16, ("m", "n")),),
            accumulator_dtype=ScalarDType.FP32,
            layout="mk_kn",
            shape_items=(("k", 8), ("m", 1), ("n", 8)),
            cache_form=None,
            state_form=None,
            topology_id="single",
        )
        graph_safety = GraphSafetyContract(
            contract_id="graph-safe:linear:v1",
            row_map_schema_id="cubic.rows.v1",
            isolation=PaddingIsolation.PRIVATE_SCRATCH,
            evidence_id="evidence:padding-poison:linear",
        )
        candidate = CandidateBinding(
            candidate_id="linear:cuda",
            implementation_id="b2s256:binary",
            backend=CandidateBackend.CUDA,
            launch_signature="block_m=16",
            workspace_bytes=0,
            evidence_id="evidence:arithmetic:linear",
            graph_safety=graph_safety,
        )
        bucket = GraphBucket(
            bucket_id="decode:8",
            template_id="graph:decode:8",
            scope_id="model.forward",
            replay_abi_id="cubic.replay.decode.v1",
            topology_id="single",
            partition_id="toy",
            phase=BatchPhase.DECODE,
            capacity_items=(("request_slots", 8),),
            logical_m_values=tuple(range(1, 9)),
            captured_m=8,
            row_map_schema_id="cubic.rows.v1",
            scratch_schema_id="cubic.scratch.v1",
            graph_safety_contract_ids=(graph_safety.contract_id,),
            evidence_id="evidence:graph:decode:8",
        )
        with self.assertRaisesRegex(ValueError, "graph-safety evidence"):
            ExecutionPlan(
                execution_plan_id="performance:unsafe",
                mode=ExecutionMode.PERFORMANCE,
                bindings=(
                    OpBinding(signature=signature, candidate=replace(candidate, graph_safety=None)),
                ),
                graph_buckets=(bucket,),
            )
        plan = sealed_test_performance_plan(
            "performance:graph",
            (OpBinding(signature=signature, candidate=candidate),),
            (bucket,),
        )
        bucket = plan.graph_buckets[0]

        self.assertTrue(plan.graph_enabled)
        self.assertIs(
            plan.bucket_for(
                scope_id="model.forward",
                replay_abi_id="cubic.replay.decode.v1",
                phase=BatchPhase.DECODE,
                logical_m=7,
                capacity_items=(("request_slots", 8),),
                topology_id="single",
                partition_id="toy",
            ),
            bucket,
        )
        with self.assertRaisesRegex(ValueError, "fallback is forbidden"):
            plan.bucket_for(
                scope_id="model.forward",
                replay_abi_id="cubic.replay.decode.v1",
                phase=BatchPhase.DECODE,
                logical_m=9,
                capacity_items=(("request_slots", 8),),
                topology_id="single",
                partition_id="toy",
            )


class RequestContractTests(unittest.TestCase):
    def test_reasoning_progress_cannot_claim_uncommitted_tokens(self) -> None:
        from cubic_serve.contracts.request import TokenEvent, TokenEventKind

        for output, reasoning in ((None, 0), (2, 3), (2, -1), (2, True), (2, 1.0)):
            with self.subTest(output=output, reasoning=reasoning), self.assertRaises(ValueError):
                TokenEvent(
                    "request", 0, TokenEventKind.TEXT_DELTA, text="",
                    output_token_count=output, reasoning_token_count=reasoning,
                )
        event = TokenEvent(
            "request", 0, TokenEventKind.TEXT_DELTA, text="",
            output_token_count=3, reasoning_token_count=2,
        )
        self.assertEqual(event.reasoning_token_count, 2)

    def test_token_placement_rejects_negative_consumer_layer(self) -> None:
        with self.assertRaisesRegex(ValueError, "consumer layers must be non-negative"):
            TokenPlacement(
                content_part_ordinal=1,
                token_start=0,
                token_count=1,
                feature_row_start=0,
                feature_row_count=1,
                consumer_layers=(-1,),
            )

    def test_three_input_modalities_form_one_input_plan(self) -> None:
        image = MediaDescriptor(
            content_digest=f"b2s256:{'1' * 64}",
            mime_type="image/png",
            byte_length=128,
            width=8,
            height=8,
        )
        video = MediaDescriptor(
            content_digest=f"b2s256:{'2' * 64}",
            mime_type="video/mp4",
            byte_length=1024,
            width=8,
            height=8,
            frame_count=2,
            duration_ms=100,
        )
        plan = InputPlan(
            request_id="request:multimodal",
            model_id="fixture",
            processor_recipe_digest=f"b2s256:{'a' * 64}",
            weight_variant=WeightVariantMeta(
                base_model_id="fixture",
                base_revision="b2s256:model",
            ),
            content_parts=(
                ContentPart(ordinal=0, modality="text", text="describe"),
                ContentPart(ordinal=1, modality="image", media=image),
                ContentPart(ordinal=2, modality="video", media=video),
            ),
            token_ids=(1, 2, 3, 4, 5, 6, 7, 8),
            positions=tuple(TokenPosition((index,)) for index in range(8)),
            decode_positions=DecodePositionPlan(
                first=TokenPosition((8,)),
                step_axes=(1,),
            ),
            placements=(
                TokenPlacement(
                    content_part_ordinal=1,
                    token_start=1,
                    token_count=2,
                    feature_row_start=0,
                    feature_row_count=2,
                    consumer_layers=(0, 1),
                ),
                TokenPlacement(
                    content_part_ordinal=2,
                    token_start=3,
                    token_count=4,
                    feature_row_start=2,
                    feature_row_count=4,
                    consumer_layers=(0, 1),
                ),
            ),
            encoder_jobs=(
                EncoderJob(
                    job_id="encoder:image",
                    content_part_ordinal=1,
                    encoder_id="vision",
                    processor_id="fixture",
                    cache_key="encoder:1",
                ),
                EncoderJob(
                    job_id="encoder:video",
                    content_part_ordinal=2,
                    encoder_id="vision",
                    processor_id="fixture",
                    cache_key="encoder:2",
                ),
            ),
        )

        self.assertEqual(
            tuple(part.modality for part in plan.content_parts), ("text", "image", "video")
        )
        with self.assertRaisesRegex(ValueError, "processor recipe digest"):
            replace(plan, processor_recipe_digest="processor:v1")


if __name__ == "__main__":
    unittest.main()
