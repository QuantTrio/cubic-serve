from __future__ import annotations

import json
import unittest
from dataclasses import replace
from hashlib import blake2s as content_digest
from pathlib import Path
from unittest import mock

import torch

from cubic_serve.contracts.request import (
    ContentPart,
    DecodePositionPlan,
    EncoderJob,
    InputPlan,
    MediaDescriptor,
    TokenPlacement,
    TokenPosition,
)
from cubic_serve.guardrail.engine_replay import ReferenceEngineReplayExecutor
from cubic_serve.guardrail.replay import (
    PrefixCandidateEvidence,
    PrefixLifecycleEvidence,
    ReplayTrace,
    record_trace,
    replay_trace,
)
from cubic_serve.inputs import EncodedFeature, EncodedFeatureBundle
from cubic_serve.models import FullBF16WeightPolicy, PreparedInput, load_model_adapter
from cubic_serve.runtime.cache import KVLayout
from cubic_serve.runtime.engine import ReferenceEngine

REPOSITORY = Path(__file__).resolve().parents[2]
TOY_WEIGHT_POLICY = FullBF16WeightPolicy(
    expected_names=(
        "embedding",
        "k_weight",
        "lm_weight",
        "out_weight",
        "q_weight",
        "v_weight",
    )
)


class EngineReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = load_model_adapter(REPOSITORY / "examples" / "toy_adapter.py")

    def engine(self) -> ReferenceEngine:
        return ReferenceEngine(
            self.loaded,
            kv_layout=KVLayout((0,), 4, 1, 8),
            weight_policy=TOY_WEIGHT_POLICY,
            token_budget=4,
            prefill_chunk_tokens=4,
        )

    def input_plan(self, request_id: str = "trace") -> InputPlan:
        descriptor = self.loaded.descriptor
        return InputPlan(
            request_id=request_id,
            model_id=descriptor.model_spec.model_id,
            processor_recipe_digest=f"b2s256:{'a' * 64}",
            weight_variant=descriptor.weight_variant,
            content_parts=(ContentPart(0, "text", text="replay"),),
            token_ids=(1, 2, 3, 4, 5, 6),
            positions=tuple(TokenPosition((index,)) for index in range(6)),
            decode_positions=DecodePositionPlan(TokenPosition((6,)), (1,)),
        )

    def executor(self) -> ReferenceEngineReplayExecutor:
        return ReferenceEngineReplayExecutor(
            engine_factory=self.engine,
            max_new_tokens={"trace": 2},
        )

    def multimodal_prepared_input(
        self,
        *,
        feature_delta: float = 0.0,
    ) -> PreparedInput:
        descriptor = self.loaded.descriptor
        image_digest = f"b2s256:{'1' * 64}"
        video_digest = f"b2s256:{'2' * 64}"
        plan = InputPlan(
            request_id="trace-mm",
            model_id=descriptor.model_spec.model_id,
            processor_recipe_digest=f"b2s256:{'a' * 64}",
            weight_variant=descriptor.weight_variant,
            content_parts=(
                ContentPart(0, "text", text="describe"),
                ContentPart(
                    1,
                    "image",
                    media=MediaDescriptor(
                        image_digest,
                        "image/png",
                        64,
                        width=8,
                        height=8,
                    ),
                ),
                ContentPart(
                    2,
                    "video",
                    media=MediaDescriptor(
                        video_digest,
                        "video/mp4",
                        128,
                        width=8,
                        height=8,
                        frame_count=2,
                        duration_ms=100,
                    ),
                ),
            ),
            token_ids=(1, 2, 3, 4, 5, 6),
            positions=tuple(TokenPosition((index, index, index)) for index in range(6)),
            decode_positions=DecodePositionPlan(TokenPosition((6, 6, 6)), (1, 1, 1)),
            placements=(
                TokenPlacement(1, 1, 2, 0, 2, (0,)),
                TokenPlacement(2, 4, 1, 2, 1, (0,)),
            ),
            encoder_jobs=(
                EncoderJob("image", 1, "toy.vision", "toy", "image:1"),
                EncoderJob("video", 2, "toy.vision", "toy", "video:2"),
            ),
        )
        base = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 8
        features = (
            EncodedFeature(
                "image",
                1,
                "toy.vision",
                "toy",
                image_digest,
                (base + feature_delta).to(torch.bfloat16),
            ),
            EncodedFeature(
                "video",
                2,
                "toy.vision",
                "toy",
                video_digest,
                (torch.flip(base, dims=(0, 1)) + 4).to(torch.bfloat16),
            ),
        )
        return PreparedInput(plan, EncodedFeatureBundle.from_input_plan(plan, features))

    def test_real_batch_plans_events_and_committed_state_replay_exactly(self) -> None:
        input_plan = self.input_plan()
        planner = self.engine()
        planner.admit(PreparedInput(input_plan), max_new_tokens=2, use_prefix_cache=False)
        planned_steps = planner.run_until_drained()
        plans = tuple(step.batch_plan for step in planned_steps)

        trace = record_trace(
            input_plans=(input_plan,),
            batch_plans=plans,
            executor_factory=self.executor,
        )
        sealed = ReplayTrace.from_json(trace.to_json())
        verdict = replay_trace(sealed, executor_factory=self.executor)

        self.assertTrue(verdict.passed)
        self.assertTrue(verdict.execution_evidence_match)
        self.assertTrue(verdict.event_sequence_match)
        self.assertTrue(verdict.terminal_digest_match)
        self.assertTrue(verdict.logits_evidence_complete)
        self.assertTrue(verdict.logits_evidence_match)
        self.assertEqual(
            len(trace.recorded_logits_evidence),
            sum(len(step.logits_rows) for step in planned_steps),
        )
        self.assertGreater(len(trace.recorded_logits_evidence), 0)
        self.assertIsNone(verdict.first_logits_mismatch_key)
        self.assertEqual(trace.execution.executor_id, "cubic.reference.engine.v1")
        self.assertEqual(
            trace.execution.model_digest, self.engine().execution_fingerprint.model_digest
        )
        self.assertEqual(len(trace.recorded_terminal_digests), 1)
        self.assertEqual(trace.recorded_terminal_digests[0].state_revision, len(plans))

    def test_multimodal_replay_binds_external_feature_artifact_bits(self) -> None:
        prepared = self.multimodal_prepared_input()
        planner = self.engine()
        planner.admit(prepared, max_new_tokens=1, use_prefix_cache=False)
        plans = tuple(step.batch_plan for step in planner.run_until_drained())

        def executor_for(value: PreparedInput) -> ReferenceEngineReplayExecutor:
            return ReferenceEngineReplayExecutor(
                engine_factory=self.engine,
                max_new_tokens={value.input_plan.request_id: 1},
                prepared_inputs={value.input_plan.request_id: value},
            )

        with self.assertRaisesRegex(ValueError, "exact external PreparedInput"):
            record_trace(
                input_plans=(prepared.input_plan,),
                batch_plans=plans,
                executor_factory=lambda: ReferenceEngineReplayExecutor(
                    engine_factory=self.engine,
                    max_new_tokens={prepared.input_plan.request_id: 1},
                ),
            )

        trace = record_trace(
            input_plans=(prepared.input_plan,),
            batch_plans=plans,
            executor_factory=lambda: executor_for(prepared),
        )
        exact = replay_trace(trace, executor_factory=lambda: executor_for(prepared))
        self.assertTrue(exact.passed)

        changed = self.multimodal_prepared_input(feature_delta=1.0)
        with self.assertRaisesRegex(ValueError, "exact PreparedInput"):
            replay_trace(trace, executor_factory=lambda: executor_for(changed))

    def test_trace_replays_cold_publish_then_warm_restore_without_page_identity(
        self,
    ) -> None:
        cold_input = self.input_plan("cold")
        warm_input = self.input_plan("warm")
        planner = self.engine()
        cold_admission = planner.admit(
            PreparedInput(cold_input),
            max_new_tokens=2,
            use_prefix_cache=True,
        )
        self.assertFalse(cold_admission.hit)
        cold_steps = planner.run_until_drained()
        warm_admission = planner.admit(
            PreparedInput(warm_input),
            max_new_tokens=2,
            use_prefix_cache=True,
        )
        self.assertTrue(warm_admission.hit)
        warm_steps = planner.run_until_drained()
        plans = tuple(step.batch_plan for step in (*cold_steps, *warm_steps))

        def executor() -> ReferenceEngineReplayExecutor:
            return ReferenceEngineReplayExecutor(
                engine_factory=self.engine,
                max_new_tokens={"cold": 2, "warm": 2},
                use_prefix_cache={"cold": True, "warm": True},
            )

        trace = record_trace(
            input_plans=(cold_input, warm_input),
            batch_plans=plans,
            executor_factory=executor,
        )
        verdict = replay_trace(trace, executor_factory=executor)

        self.assertTrue(verdict.passed)
        self.assertTrue(verdict.prefix_lifecycle_match)
        admissions = trace.recorded_prefix_lifecycle.admissions
        self.assertEqual(tuple(item.request_id for item in admissions), ("cold", "warm"))
        self.assertFalse(admissions[0].hit)
        self.assertTrue(admissions[1].hit)
        self.assertEqual(admissions[1].source_request_id, "cold")
        self.assertEqual(
            admissions[1].base_revision,
            admissions[1].source_revision,
        )
        self.assertLess(
            admissions[1].arithmetic_state_commit_boundary,
            admissions[1].physical_block_boundary,
        )
        cold_publication = next(
            item
            for item in trace.recorded_prefix_lifecycle.publications
            if item.request_id == "cold"
        )
        self.assertEqual(admissions[1].prefix_key, cold_publication.prefix_key)
        self.assertEqual(
            admissions[1].snapshot_semantic_digest,
            cold_publication.snapshot_semantic_digest,
        )
        lifecycle_json = trace.to_dict()["recorded_prefix_lifecycle"]
        self.assertNotIn("page_id", json.dumps(lifecycle_json, sort_keys=True))

        def canonical_logits(request_id: str) -> tuple[tuple[str, tuple[int, ...], str], ...]:
            return tuple(
                (item.dtype, item.shape, item.bit_digest)
                for item in trace.recorded_logits_evidence
                if item.request_id == request_id
            )

        self.assertEqual(canonical_logits("warm"), canonical_logits("cold"))
        terminals = {item.request_id: item for item in trace.recorded_terminal_digests}
        self.assertEqual(terminals["warm"].cache_digest, terminals["cold"].cache_digest)
        self.assertEqual(terminals["warm"].state_digest, terminals["cold"].state_digest)

        forged_key = f"b2s256:{'f' * 64}"
        forged_warm_candidates = tuple(
            replace(candidate, prefix_key=forged_key)
            if candidate.token_count == admissions[1].physical_block_boundary
            else candidate
            for candidate in admissions[1].candidates
        )
        forged_admissions = (
            admissions[0],
            replace(
                admissions[1],
                prefix_key=forged_key,
                candidates=forged_warm_candidates,
            ),
        )
        forged_publications = tuple(
            replace(item, prefix_key=forged_key) if item is cold_publication else item
            for item in trace.recorded_prefix_lifecycle.publications
        )
        forged_trace = replace(
            trace,
            recorded_prefix_lifecycle=PrefixLifecycleEvidence(
                forged_admissions,
                forged_publications,
            ),
        )
        with (
            mock.patch.object(
                ReferenceEngine,
                "step",
                side_effect=AssertionError("model execution must not start"),
            ) as step,
            self.assertRaisesRegex(ValueError, "publication key/boundary/revision"),
        ):
            replay_trace(forged_trace, executor_factory=executor)
        step.assert_not_called()

        payload = json.loads(trace.to_json())
        warm_payload = payload["recorded_prefix_lifecycle"]["admissions"][1]
        warm_payload["source_revision"] += 1
        warm_payload["base_revision"] += 1
        body = {key: value for key, value in payload.items() if key != "trace_id"}
        canonical = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        payload["trace_id"] = f"b2s256:{content_digest(canonical.encode()).hexdigest()}"
        with self.assertRaisesRegex(ValueError, "base revision"):
            ReplayTrace.from_json(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )

        with self.assertRaisesRegex(ValueError, "candidate exceeds"):
            replace(
                trace,
                recorded_prefix_lifecycle=replace(
                    trace.recorded_prefix_lifecycle,
                    admissions=(
                        admissions[0],
                        replace(
                            admissions[1],
                            physical_block_boundary=len(warm_input.token_ids) + 1,
                            candidates=(
                                PrefixCandidateEvidence(
                                    len(warm_input.token_ids) + 1,
                                    admissions[1].prefix_key or forged_key,
                                ),
                            ),
                        ),
                    ),
                ),
            )

        forged_cold_hit = replace(
            admissions[0],
            hit=True,
            prefix_key=admissions[1].prefix_key,
            candidates=admissions[1].candidates,
            physical_block_boundary=admissions[1].physical_block_boundary,
            arithmetic_state_commit_boundary=(admissions[1].arithmetic_state_commit_boundary),
            state_commit_alignment=admissions[1].state_commit_alignment,
            base_revision=admissions[1].base_revision,
            source_request_id=admissions[1].source_request_id,
            source_rank=admissions[1].source_rank,
            source_revision=admissions[1].source_revision,
            snapshot_semantic_digest=admissions[1].snapshot_semantic_digest,
        )
        with self.assertRaisesRegex(ValueError, "resume boundary"):
            replace(
                trace,
                recorded_prefix_lifecycle=replace(
                    trace.recorded_prefix_lifecycle,
                    admissions=(forged_cold_hit, admissions[1]),
                ),
            )

        with self.assertRaisesRegex(ValueError, "execution fingerprint"):
            replace(
                trace,
                recorded_prefix_lifecycle=replace(
                    trace.recorded_prefix_lifecycle,
                    admissions=(
                        replace(
                            admissions[0],
                            execution_fingerprint_digest=forged_key,
                        ),
                        admissions[1],
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
