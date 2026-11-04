from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from cubic_serve.contracts.request import (
    ContentPart,
    DecodePositionPlan,
    EncoderJob,
    InputPlan,
    MediaDescriptor,
    TokenEventKind,
    TokenPlacement,
    TokenPosition,
)
from cubic_serve.executor.runner import require_reference_execution
from cubic_serve.guardrail.compare import tensor_digest
from cubic_serve.inputs import (
    EncodedFeature,
    EncodedFeatureBundle,
    FeatureAdmissionError,
    FeatureReader,
)
from cubic_serve.models import (
    FullBF16WeightPolicy,
    ModelSegment,
    ModelStepResult,
    PreparedInput,
    WeightIntegrityError,
    load_model_adapter,
)
from cubic_serve.runtime.cache import KVLayout, OwnershipError, PrefixCache
from cubic_serve.runtime.engine import EngineStep, ReferenceEngine

REPOSITORY = Path(__file__).resolve().parents[2]
TOY_ADAPTER = REPOSITORY / "examples" / "toy_adapter.py"
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


class _FailAfterMutation:
    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def initialize_request(self) -> tuple[object, ...]:
        return self._wrapped.initialize_request()

    def execute_segment(
        self, invocation: object, ops: object, state: object, features: object
    ) -> object:
        self._wrapped.execute_segment(invocation, ops, state, features)
        raise RuntimeError("fixture failure after tentative KV mutation")


class _NonFiniteAfterMutation:
    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def initialize_request(self) -> tuple[object, ...]:
        return self._wrapped.initialize_request()

    def execute_segment(
        self, invocation: object, ops: object, state: object, features: object
    ) -> object:
        result = self._wrapped.execute_segment(invocation, ops, state, features)
        return ModelStepResult(logits=torch.full_like(result.logits, float("nan")))


class _NonFiniteKVState:
    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def read_kv(self, layer_id: int, token_ordinals: tuple[int, ...]) -> object:
        return self._wrapped.read_kv(layer_id, token_ordinals)

    def write_kv(
        self,
        layer_id: int,
        token_ordinals: tuple[int, ...],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        nonfinite_key = key.clone()
        nonfinite_key.flatten()[0] = float("nan")
        self._wrapped.write_kv(layer_id, token_ordinals, nonfinite_key, value)

    def read_recurrent_state(self, state_id: str) -> torch.Tensor:
        return self._wrapped.read_recurrent_state(state_id)

    def write_recurrent_state(
        self,
        state_id: str,
        value: torch.Tensor,
        *,
        target_position: int,
        valid: bool = True,
    ) -> None:
        self._wrapped.write_recurrent_state(
            state_id,
            value,
            target_position=target_position,
            valid=valid,
        )


class _NonFiniteKVProgram:
    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def initialize_request(self) -> tuple[object, ...]:
        return self._wrapped.initialize_request()

    def execute_segment(
        self, invocation: object, ops: object, state: object, features: object
    ) -> object:
        return self._wrapped.execute_segment(
            invocation,
            ops,
            _NonFiniteKVState(state),
            features,
        )


class _NonFiniteInitialization:
    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def initialize_request(self) -> tuple[object, ...]:
        initializations = self._wrapped.initialize_request()
        initializations[0].value.flatten()[0] = float("inf")
        return initializations

    def execute_segment(
        self, invocation: object, ops: object, state: object, features: object
    ) -> object:
        return self._wrapped.execute_segment(invocation, ops, state, features)


class _CorruptingState:
    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def read_kv(self, layer_id: int, token_ordinals: tuple[int, ...]) -> object:
        return self._wrapped.read_kv(layer_id, token_ordinals)

    def write_kv(
        self,
        layer_id: int,
        token_ordinals: tuple[int, ...],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        corrupted = key.clone()
        corrupted.flatten()[0] += 1
        self._wrapped.write_kv(layer_id, token_ordinals, corrupted, value)

    def read_recurrent_state(self, state_id: str) -> torch.Tensor:
        return self._wrapped.read_recurrent_state(state_id)

    def write_recurrent_state(
        self,
        state_id: str,
        value: torch.Tensor,
        *,
        target_position: int,
        valid: bool = True,
    ) -> None:
        self._wrapped.write_recurrent_state(
            state_id,
            value,
            target_position=target_position,
            valid=valid,
        )


class _CorruptReplayKV:
    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def initialize_request(self) -> tuple[object, ...]:
        return self._wrapped.initialize_request()

    def execute_segment(
        self, invocation: object, ops: object, state: object, features: object
    ) -> object:
        return self._wrapped.execute_segment(
            invocation,
            ops,
            _CorruptingState(state),
            features,
        )


class _ReferenceContextProbe:
    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped
        self.initialized = False
        self.executed = False

    def initialize_request(self) -> tuple[object, ...]:
        require_reference_execution()
        self.initialized = True
        return self._wrapped.initialize_request()

    def execute_segment(
        self, invocation: object, ops: object, state: object, features: object
    ) -> object:
        require_reference_execution()
        self.executed = True
        return self._wrapped.execute_segment(invocation, ops, state, features)


class _RetainStateAcrossSegments:
    """Try to reuse request A's state authority while request B is executing."""

    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped
        self._retained: object | None = None

    def initialize_request(self) -> tuple[object, ...]:
        return self._wrapped.initialize_request()

    def execute_segment(
        self, invocation: object, ops: object, state: object, features: object
    ) -> object:
        if self._retained is not None:
            self._retained.read_recurrent_state("toy.running_hidden")
        self._retained = state
        return self._wrapped.execute_segment(invocation, ops, state, features)


class _ReadFutureVideo:
    """Malicious fixture that tries to consume video before its token placement."""

    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def initialize_request(self) -> tuple[object, ...]:
        return self._wrapped.initialize_request()

    def execute_segment(
        self,
        segment: ModelSegment,
        ops: object,
        state: object,
        features: FeatureReader,
    ) -> object:
        object.__getattribute__(segment, "input_plan_digest")
        return self._wrapped.execute_segment(segment, ops, state, features)


class ReferenceEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = load_model_adapter(TOY_ADAPTER)

    def engine(
        self,
        *,
        prefix_cache: PrefixCache | None = None,
        budget: int = 4,
        chunk: int = 4,
    ) -> ReferenceEngine:
        return ReferenceEngine(
            self.loaded,
            kv_layout=KVLayout(
                layer_ids=(0,),
                block_size_tokens=4,
                num_kv_heads=1,
                head_dim=8,
            ),
            weight_policy=TOY_WEIGHT_POLICY,
            token_budget=budget,
            prefill_chunk_tokens=chunk,
            prefix_cache=prefix_cache,
        )

    def input_plan(
        self,
        request_id: str,
        *,
        image_digest: str = "1",
        video_digest: str = "2",
    ) -> InputPlan:
        descriptor = self.loaded.descriptor
        image = MediaDescriptor(
            content_digest=f"b2s256:{image_digest * 64}",
            mime_type="image/png",
            byte_length=64,
            width=8,
            height=8,
        )
        video = MediaDescriptor(
            content_digest=f"b2s256:{video_digest * 64}",
            mime_type="video/mp4",
            byte_length=128,
            width=8,
            height=8,
            frame_count=2,
            duration_ms=100,
        )
        return InputPlan(
            request_id=request_id,
            model_id=descriptor.model_spec.model_id,
            processor_recipe_digest=f"b2s256:{'a' * 64}",
            weight_variant=descriptor.weight_variant,
            content_parts=(
                ContentPart(ordinal=0, modality="text", text="describe"),
                ContentPart(ordinal=1, modality="image", media=image),
                ContentPart(ordinal=2, modality="video", media=video),
            ),
            token_ids=(1, 2, 3, 4, 5, 6),
            positions=tuple(TokenPosition((index, index, index)) for index in range(6)),
            decode_positions=DecodePositionPlan(
                first=TokenPosition((6, 6, 6)),
                step_axes=(1, 1, 1),
            ),
            placements=(
                TokenPlacement(
                    content_part_ordinal=1,
                    token_start=1,
                    token_count=2,
                    feature_row_start=0,
                    feature_row_count=2,
                    consumer_layers=(0,),
                ),
                TokenPlacement(
                    content_part_ordinal=2,
                    token_start=4,
                    token_count=1,
                    feature_row_start=2,
                    feature_row_count=1,
                    consumer_layers=(0,),
                ),
            ),
            encoder_jobs=(
                EncoderJob("image", 1, "toy.vision", "toy", f"image:{image_digest}"),
                EncoderJob("video", 2, "toy.vision", "toy", f"video:{video_digest}"),
            ),
        )

    def prepared_input(
        self,
        request_id: str,
        *,
        image_digest: str = "1",
        video_digest: str = "2",
        image_feature_delta: float = 0.0,
        video_feature_delta: float = 0.0,
        positions: tuple[TokenPosition, ...] | None = None,
    ) -> PreparedInput:
        plan = self.input_plan(
            request_id,
            image_digest=image_digest,
            video_digest=video_digest,
        )
        if positions is not None:
            plan = replace(plan, positions=positions)
        base = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 8
        image = EncodedFeature(
            job_id="image",
            content_part_ordinal=1,
            encoder_id="toy.vision",
            processor_id="toy",
            source_media_digest=f"b2s256:{image_digest * 64}",
            tensor=(base + image_feature_delta).to(torch.bfloat16),
        )
        video = EncodedFeature(
            job_id="video",
            content_part_ordinal=2,
            encoder_id="toy.vision",
            processor_id="toy",
            source_media_digest=f"b2s256:{video_digest * 64}",
            tensor=(torch.flip(base, dims=(0, 1)) + 4 + video_feature_delta).to(torch.bfloat16),
        )
        return PreparedInput(
            plan,
            EncodedFeatureBundle.from_input_plan(plan, (image, video)),
        )

    @staticmethod
    def token_ids(steps: tuple[EngineStep, ...], request_id: str) -> tuple[int, ...]:
        return tuple(
            event.token_id
            for step in steps
            for event in step.events
            if event.request_id == request_id and event.kind is TokenEventKind.TOKEN
        )

    @staticmethod
    def canonical_output_logits(
        steps: tuple[EngineStep, ...],
        request_id: str,
    ) -> tuple[tuple[str, tuple[int, ...], str], ...]:
        """Exact output-boundary rows, independent of batching lifecycle ids."""

        rows: list[tuple[str, tuple[int, ...], str]] = []
        for step in steps:
            for row in step.logits_rows:
                if row.request_id != request_id:
                    continue
                segment = step.batch_plan.segments[row.segment_index]
                if (
                    row.plan_id != step.batch_plan.plan_id
                    or segment.request_id != request_id
                    or row.row_index != segment.token_count - 1
                ):
                    raise AssertionError(
                        "logits row does not identify its exact output-boundary segment"
                    )
                rows.append(
                    (
                        str(row.tensor.dtype),
                        tuple(row.tensor.shape),
                        tensor_digest(row.tensor),
                    )
                )
        return tuple(rows)

    def test_three_input_reference_path_prefix_hit_and_bf16_kv_are_exact(self) -> None:
        cache = PrefixCache()
        cold = self.engine(prefix_cache=cache)
        self.assertFalse(cold.admit(self.prepared_input("cold"), max_new_tokens=3).hit)
        cold_steps = cold.run_until_drained()
        cold_tokens = self.token_ids(cold_steps, "cold")

        self.assertEqual(len(cold_tokens), 3)
        self.assertTrue(cold_steps[1].published_prefix_keys)
        prefix_key = cold_steps[1].published_prefix_keys[0]
        lease = cache.acquire(prefix_key, source_rank=0)
        self.assertIsNotNone(lease)
        assert lease is not None
        with lease:
            snapshot = lease.snapshot
        self.assertEqual(
            (
                snapshot.physical_block_boundary,
                snapshot.arithmetic_state_commit_boundary,
                snapshot.replay_tokens,
                snapshot.recurrent_states[0].position,
            ),
            (4, 3, (4,), 3),
        )
        self.assertEqual(cold_steps[3].batch_plan.segments[0].token_ids, (cold_tokens[0],))
        self.assertEqual(
            cold_steps[3].batch_plan.segments[0].positions,
            (TokenPosition((6, 6, 6)),),
        )
        table = cold.allocator.page_table("cold")
        key, value = cold.store.read_kv("cold", table.logical_to_physical[0], 0, (0, 1, 2, 3))
        self.assertIs(key.dtype, torch.bfloat16)
        self.assertIs(value.dtype, torch.bfloat16)

        warm = self.engine(prefix_cache=cache)
        admission = warm.admit(self.prepared_input("warm"), max_new_tokens=3)
        self.assertTrue(admission.hit)
        self.assertEqual(admission.token_count, 4)
        self.assertEqual(admission.resume_token_count, 3)
        warm_steps = warm.run_until_drained()
        self.assertEqual(self.token_ids(warm_steps, "warm"), cold_tokens)
        self.assertEqual(len(warm_steps), len(cold_steps) - 1)
        self.assertEqual(
            warm.terminal_digest("warm").cache_digest,
            cold.terminal_digest("cold").cache_digest,
        )
        self.assertEqual(
            warm.terminal_digest("warm").state_digest,
            cold.terminal_digest("cold").state_digest,
        )
        self.assertEqual(
            self.canonical_output_logits(warm_steps, "warm"),
            self.canonical_output_logits(cold_steps, "cold"),
        )
        self.assertEqual(len(self.canonical_output_logits(warm_steps, "warm")), 3)

    def test_descriptor_cannot_hide_different_actual_weight_bits_from_prefix_cache(
        self,
    ) -> None:
        cache = PrefixCache()
        source = self.engine(prefix_cache=cache)
        source.admit(self.prepared_input("weight-source"), max_new_tokens=1)
        source.run_until_drained()

        original_build = self.loaded.adapter.build

        def build_with_different_bits(context: object) -> object:
            program = original_build(context)
            program.embedding[0, 0] = program.embedding[0, 0] + 1
            return program

        with (
            mock.patch.object(
                self.loaded.adapter,
                "build",
                side_effect=build_with_different_bits,
            ),
            self.assertRaisesRegex(
                WeightIntegrityError,
                "descriptor WeightManifest differs",
            ),
        ):
            self.engine(prefix_cache=cache)

        matching = self.engine(prefix_cache=cache)
        self.assertTrue(
            matching.admit(self.prepared_input("weight-matching"), max_new_tokens=1).hit
        )

    def test_weight_mutation_is_rejected_before_prefix_or_step_use(self) -> None:
        prefix_engine = self.engine()
        prefix_input = self.prepared_input("weight-prefix-drift")
        prefix_engine._identity_program.embedding[0, 0] += 1
        with self.assertRaisesRegex(WeightIntegrityError, "prefix identity"):
            prefix_engine.prefix_key(prefix_input, 4)
        with self.assertRaisesRegex(WeightIntegrityError, "request admission"):
            prefix_engine.admit(prefix_input, max_new_tokens=1)

        step_engine = self.engine()
        step_engine.admit(
            self.prepared_input("weight-step-drift"),
            max_new_tokens=1,
            use_prefix_cache=False,
        )
        step_engine._identity_program.lm_weight[0, 0] += 1
        with self.assertRaisesRegex(WeightIntegrityError, "step planning"):
            step_engine.step()
        state = step_engine.scheduler.request_state("weight-step-drift")
        self.assertIsNone(state.inflight)
        self.assertEqual(state.state_revision, 0)

    def test_prefix_key_binds_environment_selected_program_recipe(self) -> None:
        prepared_input = self.prepared_input("recipe")
        baseline = self.engine()
        baseline_key = baseline.prefix_key(prepared_input, 4)
        baseline_weight_digest = baseline.weight_evidence.semantic_digest
        original_build = self.loaded.adapter.build

        def build_with_alternate_recipe(context: object) -> object:
            program = original_build(context)
            recipe = program.execution_recipe()
            program.execution_recipe = lambda: replace(
                recipe,
                settings=tuple(sorted((*recipe.settings, ("environment_branch", "alternate")))),
            )
            return program

        with mock.patch.object(
            self.loaded.adapter,
            "build",
            side_effect=build_with_alternate_recipe,
        ):
            alternate = self.engine()

        self.assertEqual(
            baseline_weight_digest,
            alternate.weight_evidence.semantic_digest,
        )
        self.assertNotEqual(baseline_key, alternate.prefix_key(prepared_input, 4))
        self.assertNotEqual(
            baseline.execution_fingerprint.model_digest,
            alternate.execution_fingerprint.model_digest,
        )

    def test_prefix_key_binds_resolved_implementation_callable(self) -> None:
        prepared_input = self.prepared_input("implementation")
        baseline = self.engine()

        def alternate_text_implementation(value: torch.Tensor) -> torch.Tensor:
            return value

        alternate_catalog = replace(
            self.loaded.reference_catalog,
            implementations=tuple(
                replace(implementation, function=alternate_text_implementation)
                if implementation.semantic_op == "fixture.input.text"
                else implementation
                for implementation in self.loaded.reference_catalog.implementations
            ),
        )
        alternate_loaded = replace(
            self.loaded,
            reference_catalog=alternate_catalog,
        )
        alternate = ReferenceEngine(
            alternate_loaded,
            kv_layout=KVLayout(
                layer_ids=(0,),
                block_size_tokens=4,
                num_kv_heads=1,
                head_dim=8,
            ),
            weight_policy=TOY_WEIGHT_POLICY,
        )

        self.assertNotEqual(
            baseline.execution_fingerprint.implementation_digest,
            alternate.execution_fingerprint.implementation_digest,
        )
        self.assertNotEqual(
            baseline.prefix_key(prepared_input, 4),
            alternate.prefix_key(prepared_input, 4),
        )

    def test_prefix_replay_rejects_any_recomputed_kv_drift_before_commit(self) -> None:
        cache = PrefixCache()
        cold = self.engine(prefix_cache=cache)
        cold.admit(self.prepared_input("source"), max_new_tokens=1)
        cold.run_until_drained()

        warm = self.engine(prefix_cache=cache)
        admission = warm.admit(self.prepared_input("drift"), max_new_tokens=1)
        self.assertEqual((admission.token_count, admission.resume_token_count), (4, 3))
        base_revision = warm.store.current_revision("drift")
        warm._program = _CorruptReplayKV(warm._program)
        with self.assertRaisesRegex(RuntimeError, "recomputed KV differs"):
            warm.step()

        state = warm.scheduler.request_state("drift")
        self.assertEqual((state.prefill_cursor, state.state_revision), (3, base_revision))
        self.assertEqual(warm.store.current_revision("drift"), base_revision)

    def test_full_prompt_hybrid_prefix_replays_one_row_to_regenerate_logits(self) -> None:
        descriptor = self.loaded.descriptor

        def text_input(request_id: str) -> PreparedInput:
            plan = InputPlan(
                request_id=request_id,
                model_id=descriptor.model_spec.model_id,
                processor_recipe_digest=f"b2s256:{'a' * 64}",
                weight_variant=descriptor.weight_variant,
                content_parts=(ContentPart(0, "text", text="full-prefix"),),
                token_ids=(1, 2, 3, 4),
                positions=tuple(TokenPosition((index, index, index)) for index in range(4)),
                decode_positions=DecodePositionPlan(
                    TokenPosition((4, 4, 4)),
                    (1, 1, 1),
                ),
            )
            return PreparedInput(plan)

        cache = PrefixCache()
        cold = self.engine(prefix_cache=cache)
        cold.admit(text_input("full-cold"), max_new_tokens=1)
        cold_steps = cold.run_until_drained()
        self.assertEqual(len(cold_steps), 2)
        self.assertTrue(cold_steps[-1].published_prefix_keys)

        warm = self.engine(prefix_cache=cache)
        admission = warm.admit(text_input("full-warm"), max_new_tokens=1)
        self.assertEqual(
            (admission.hit, admission.token_count, admission.resume_token_count),
            (True, 4, 3),
        )
        warm_steps = warm.run_until_drained()
        self.assertEqual(len(warm_steps), 1)
        self.assertEqual(
            self.token_ids(warm_steps, "full-warm"),
            self.token_ids(cold_steps, "full-cold"),
        )
        self.assertEqual(
            warm.terminal_digest("full-warm").cache_digest,
            cold.terminal_digest("full-cold").cache_digest,
        )
        self.assertEqual(
            warm.terminal_digest("full-warm").state_digest,
            cold.terminal_digest("full-cold").state_digest,
        )
        self.assertEqual(
            self.canonical_output_logits(warm_steps, "full-warm"),
            self.canonical_output_logits(cold_steps, "full-cold"),
        )

    def test_prefix_identity_ignores_later_media_but_binds_affected_media(self) -> None:
        cache = PrefixCache()
        source = self.engine(prefix_cache=cache)
        source.admit(self.prepared_input("source"), max_new_tokens=1)
        source.run_until_drained()

        later_media_changed = self.engine(prefix_cache=cache)
        self.assertTrue(
            later_media_changed.admit(
                self.prepared_input("later", video_digest="3"), max_new_tokens=1
            ).hit
        )
        later_media_changed.run_until_drained()
        affected_media_changed = self.engine(prefix_cache=cache)
        self.assertFalse(
            affected_media_changed.admit(
                self.prepared_input("affected", image_digest="4"), max_new_tokens=1
            ).hit
        )
        affected_feature_bits_changed = self.engine(prefix_cache=cache)
        self.assertFalse(
            affected_feature_bits_changed.admit(
                self.prepared_input("affected-bits", image_feature_delta=8),
                max_new_tokens=1,
            ).hit
        )

    def test_cold_prefix_state_cannot_observe_that_the_request_ends_here(self) -> None:
        long_input = self.prepared_input(
            "has-later-suffix",
            video_digest="3",
            video_feature_delta=16,
        )
        long_plan = long_input.input_plan
        assert long_input.feature_bundle is not None
        image_feature = next(
            binding.feature
            for binding in long_input.feature_bundle.bindings
            if binding.feature.job_id == "image"
        )
        short_plan = InputPlan(
            request_id="ends-at-prefix",
            model_id=long_plan.model_id,
            processor_recipe_digest=long_plan.processor_recipe_digest,
            weight_variant=long_plan.weight_variant,
            content_parts=long_plan.content_parts[:2],
            token_ids=long_plan.token_ids[:4],
            positions=long_plan.positions[:4],
            decode_positions=DecodePositionPlan(
                first=TokenPosition((4, 4, 4)),
                step_axes=(1, 1, 1),
            ),
            placements=long_plan.placements[:1],
            encoder_jobs=long_plan.encoder_jobs[:1],
        )
        short_input = PreparedInput(
            short_plan,
            EncodedFeatureBundle.from_input_plan(short_plan, (image_feature,)),
        )

        short = self.engine(budget=4, chunk=4)
        long = self.engine(budget=4, chunk=4)
        long.allocator.allocate(
            request_id="allocator-noise",
            token_start=0,
            token_count=1,
        )
        short.admit(short_input, max_new_tokens=1, use_prefix_cache=False)
        long.admit(long_input, max_new_tokens=1, use_prefix_cache=False)

        short.step()
        long.step()
        short_boundary = short.step()
        long_boundary = long.step()
        short_segment = short_boundary.batch_plan.segments[0]
        long_segment = long_boundary.batch_plan.segments[0]
        self.assertEqual(
            (short_segment.token_start, short_segment.token_ids),
            (long_segment.token_start, long_segment.token_ids),
        )
        self.assertTrue(short_segment.produces_output)
        self.assertFalse(long_segment.produces_output)
        self.assertEqual(len(short_boundary.logits_rows), 1)
        self.assertFalse(long_boundary.logits_rows)

        short_page = short.allocator.page_table("ends-at-prefix").logical_to_physical[0]
        long_page = long.allocator.page_table("has-later-suffix").logical_to_physical[0]
        self.assertNotEqual(short_page, long_page)
        offsets = (0, 1, 2, 3)
        short_key, short_value = short.store.read_kv(
            "ends-at-prefix", short_page, 0, offsets
        )
        long_key, long_value = long.store.read_kv(
            "has-later-suffix", long_page, 0, offsets
        )
        self.assertTrue(torch.equal(short_key, long_key))
        self.assertTrue(torch.equal(short_value, long_value))

        short_state = short.store.checkpoint_recurrent_state(
            "ends-at-prefix", "toy.running_hidden"
        )
        long_state = long.store.checkpoint_recurrent_state(
            "has-later-suffix", "toy.running_hidden"
        )
        self.assertEqual(
            (short_state.position, short_state.valid),
            (long_state.position, long_state.valid),
        )
        self.assertTrue(torch.equal(short_state.value, long_state.value))

    def test_encoded_feature_and_every_position_axis_change_real_kv_and_logits(self) -> None:
        def observe(prepared_input: PreparedInput) -> tuple[str, torch.Tensor]:
            engine = self.engine()
            engine.admit(prepared_input, max_new_tokens=1, use_prefix_cache=False)
            steps = engine.run_until_drained()
            logits = tuple(
                row.tensor
                for step in steps
                for row in step.logits_rows
                if row.request_id == prepared_input.input_plan.request_id
            )
            self.assertEqual(len(logits), 1)
            return (
                engine.terminal_digest(prepared_input.input_plan.request_id).cache_digest,
                logits[0],
            )

        baseline_positions = tuple(TokenPosition((index, index, index)) for index in range(6))
        baseline_cache, baseline_logits = observe(
            self.prepared_input("baseline", positions=baseline_positions)
        )
        feature_cache, feature_logits = observe(
            self.prepared_input(
                "feature-bits",
                image_feature_delta=8,
                positions=baseline_positions,
            )
        )
        self.assertNotEqual(feature_cache, baseline_cache)
        self.assertFalse(torch.equal(feature_logits, baseline_logits))

        for axis in range(3):
            changed_positions = list(baseline_positions)
            changed_axes = list(changed_positions[-1].axes)
            changed_axes[axis] += 16
            changed_positions[-1] = TokenPosition(tuple(changed_axes))
            position_cache, position_logits = observe(
                self.prepared_input(
                    f"position-axis-{axis}",
                    positions=tuple(changed_positions),
                )
            )
            with self.subTest(axis=axis):
                self.assertNotEqual(position_cache, baseline_cache)
                self.assertFalse(torch.equal(position_logits, baseline_logits))

    def test_admission_detaches_model_input_from_caller_feature_tensors(self) -> None:
        engine = self.engine()
        prepared = self.prepared_input("owned-features")
        assert prepared.feature_bundle is not None
        original_rows = tuple(
            binding.feature.tensor.clone() for binding in prepared.feature_bundle.bindings
        )

        engine.admit(prepared, max_new_tokens=1, use_prefix_cache=False)

        owned = engine._prepared_inputs[prepared.input_plan.request_id]
        assert owned.feature_bundle is not None
        self.assertIsNot(owned, prepared)
        for caller_binding, owned_binding, expected in zip(
            prepared.feature_bundle.bindings,
            owned.feature_bundle.bindings,
            original_rows,
            strict=True,
        ):
            self.assertNotEqual(
                caller_binding.feature.tensor.data_ptr(),
                owned_binding.feature.tensor.data_ptr(),
            )
            caller_binding.feature.tensor.fill_(float("nan"))
            self.assertTrue(torch.equal(owned_binding.feature.tensor, expected))

        steps = engine.run_until_drained()
        self.assertEqual(len(self.token_ids(steps, prepared.input_plan.request_id)), 1)

    def test_model_segment_cannot_expose_request_suffix_or_scheduler_identity(self) -> None:
        self.assertEqual(
            {"phase", "token_start", "token_ids", "positions"},
            set(ModelSegment.__dataclass_fields__),
        )

    def test_adapter_cannot_read_video_before_current_segment_reaches_placement(self) -> None:
        engine = self.engine(budget=4, chunk=4)
        engine._program = _ReadFutureVideo(engine._program)
        engine.admit(
            self.prepared_input("early-video"),
            max_new_tokens=1,
            use_prefix_cache=False,
        )

        with self.assertRaisesRegex(AttributeError, "input_plan_digest"):
            engine.step()
        self.assertEqual(engine.store.current_revision("early-video"), 0)

    def test_non_text_preparation_and_admission_fail_atomically(self) -> None:
        engine = self.engine()
        valid = self.prepared_input("atomic-features")
        with self.assertRaisesRegex(ValueError, "requires exactly one"):
            PreparedInput(valid.input_plan)

        assert valid.feature_bundle is not None
        incomplete = EncodedFeatureBundle(
            valid.input_plan.request_id,
            valid.feature_bundle.bindings[:1],
        )
        with self.assertRaisesRegex(FeatureAdmissionError, "exactly match"):
            PreparedInput(valid.input_plan, incomplete)

        bad_features = tuple(
            replace(
                binding.feature,
                tensor=torch.zeros((3, 7), dtype=torch.bfloat16),
            )
            if binding.feature.job_id == "image"
            else binding.feature
            for binding in valid.feature_bundle.bindings
        )
        wrong_hidden = PreparedInput(
            valid.input_plan,
            EncodedFeatureBundle.from_input_plan(valid.input_plan, bad_features),
        )
        with self.assertRaisesRegex(ValueError, "hidden size"):
            engine.admit(wrong_hidden, max_new_tokens=1)

        self.assertEqual(len(engine.features), 0)
        with self.assertRaises(KeyError):
            engine.scheduler.request_state(valid.input_plan.request_id)
        with self.assertRaises(OwnershipError):
            engine.store.current_revision(valid.input_plan.request_id)

        self.assertFalse(engine.admit(valid, max_new_tokens=1, use_prefix_cache=False).hit)
        self.assertEqual(len(engine.features), 2)

    def test_text_only_prepared_input_needs_no_feature_bundle(self) -> None:
        descriptor = self.loaded.descriptor
        plan = InputPlan(
            request_id="text-only",
            model_id=descriptor.model_spec.model_id,
            processor_recipe_digest=f"b2s256:{'a' * 64}",
            weight_variant=descriptor.weight_variant,
            content_parts=(ContentPart(0, "text", text="hello"),),
            token_ids=(1, 2, 3),
            positions=tuple(TokenPosition((index, index, index)) for index in range(3)),
            decode_positions=DecodePositionPlan(TokenPosition((3, 3, 3)), (1, 1, 1)),
        )
        engine = self.engine()

        engine.admit(PreparedInput(plan), max_new_tokens=1, use_prefix_cache=False)
        steps = engine.run_until_drained()

        self.assertEqual(len(self.token_ids(steps, plan.request_id)), 1)
        self.assertEqual(len(engine.features), 0)

    def test_input_plan_encoder_identity_is_validated_before_admission(self) -> None:
        engine = self.engine()
        valid = self.prepared_input("encoder")
        jobs = tuple(
            replace(job, encoder_id="wrong.encoder") if job.content_part_ordinal == 1 else job
            for job in valid.input_plan.encoder_jobs
        )
        bad_plan = replace(valid.input_plan, encoder_jobs=jobs)
        assert valid.feature_bundle is not None
        bad_features = tuple(
            replace(binding.feature, encoder_id="wrong.encoder")
            if binding.feature.content_part_ordinal == 1
            else binding.feature
            for binding in valid.feature_bundle.bindings
        )
        bad = PreparedInput(
            bad_plan,
            EncodedFeatureBundle.from_input_plan(bad_plan, bad_features),
        )
        with self.assertRaisesRegex(ValueError, "encoder identity"):
            engine.admit(bad, max_new_tokens=1)

        admission = engine.admit(valid, max_new_tokens=1, use_prefix_cache=False)
        self.assertFalse(admission.hit)
        self.assertEqual(engine.store.current_revision("encoder"), 0)

    def test_admission_validates_every_placement_for_one_media_part(self) -> None:
        engine = self.engine()
        valid = self.prepared_input("placement-layers")
        plan = valid.input_plan
        layer_count = self.loaded.descriptor.model_spec.layer_count
        assert layer_count is not None
        image, video = plan.placements
        invalid_early_image = replace(image, consumer_layers=(layer_count,))
        valid_late_image = TokenPlacement(
            content_part_ordinal=image.content_part_ordinal,
            token_start=3,
            token_count=1,
            feature_row_start=2,
            feature_row_count=1,
            consumer_layers=(0,),
        )
        bad_plan = replace(
            plan,
            placements=(invalid_early_image, valid_late_image, video),
        )
        assert valid.feature_bundle is not None
        features_by_job = {
            binding.feature.job_id: binding.feature for binding in valid.feature_bundle.bindings
        }
        bad = PreparedInput(
            bad_plan,
            EncodedFeatureBundle.from_input_plan(
                bad_plan,
                tuple(features_by_job[job_id] for job_id in sorted(features_by_job)),
            ),
        )

        with self.assertRaisesRegex(ValueError, "placement consumes a layer outside ModelSpec"):
            engine.admit(bad, max_new_tokens=1, use_prefix_cache=False)

        self.assertEqual(len(engine.features), 0)
        with self.assertRaises(KeyError):
            engine.scheduler.request_state(plan.request_id)
        with self.assertRaises(OwnershipError):
            engine.store.current_revision(plan.request_id)

    def test_invalid_limit_and_failed_prefix_bind_leave_admission_reusable(self) -> None:
        cache = PrefixCache()
        cold = self.engine(prefix_cache=cache)
        cold.admit(self.prepared_input("source"), max_new_tokens=1)
        cold.run_until_drained()

        warm = self.engine(prefix_cache=cache)
        prepared = self.prepared_input("reusable")
        with self.assertRaisesRegex(ValueError, "positive"):
            warm.admit(prepared, max_new_tokens=0)
        with (
            mock.patch.object(
                warm.allocator,
                "bind_restored_prefix",
                side_effect=RuntimeError("injected bind failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "injected bind"),
        ):
            warm.admit(prepared, max_new_tokens=1)

        with self.assertRaises(KeyError):
            warm.scheduler.request_state("reusable")
        with self.assertRaises(OwnershipError):
            warm.store.current_revision("reusable")
        self.assertEqual(len(warm.features), 0)
        self.assertTrue(warm.admit(prepared, max_new_tokens=1).hit)

    def test_interleaved_segment_plan_matches_isolated_requests(self) -> None:
        interleaved = self.engine(budget=4, chunk=2)
        interleaved.admit(self.prepared_input("a"), max_new_tokens=2, use_prefix_cache=False)
        interleaved.admit(self.prepared_input("b"), max_new_tokens=2, use_prefix_cache=False)
        interleaved_steps = interleaved.run_until_drained()
        self.assertEqual(interleaved_steps[0].batch_plan.input_plan_ids, ("a", "b"))
        self.assertEqual(interleaved_steps[0].batch_plan.shape.logical_m, 4)
        self.assertFalse(
            any(step.published_prefix_keys for step in interleaved_steps),
            "disabling prefix cache must disable both lookup and publication",
        )

        isolated = self.engine(budget=2, chunk=2)
        isolated.admit(self.prepared_input("iso"), max_new_tokens=2, use_prefix_cache=False)
        isolated_steps = isolated.run_until_drained()
        expected = self.token_ids(isolated_steps, "iso")
        self.assertEqual(self.token_ids(interleaved_steps, "a"), expected)
        self.assertEqual(self.token_ids(interleaved_steps, "b"), expected)
        self.assertEqual(
            interleaved.terminal_digest("a").cache_digest,
            isolated.terminal_digest("iso").cache_digest,
        )
        self.assertEqual(
            interleaved.terminal_digest("b").cache_digest,
            isolated.terminal_digest("iso").cache_digest,
        )
        self.assertEqual(
            interleaved.terminal_digest("a").state_digest,
            isolated.terminal_digest("iso").state_digest,
        )
        self.assertEqual(
            interleaved.terminal_digest("b").state_digest,
            isolated.terminal_digest("iso").state_digest,
        )
        isolated_logits = self.canonical_output_logits(isolated_steps, "iso")
        self.assertEqual(
            self.canonical_output_logits(interleaved_steps, "a"),
            isolated_logits,
        )
        self.assertEqual(
            self.canonical_output_logits(interleaved_steps, "b"),
            isolated_logits,
        )
        self.assertEqual(len(isolated_logits), 2)

    def test_post_commit_prefix_failure_is_reported_without_retrying_committed_rows(
        self,
    ) -> None:
        cache = PrefixCache()
        engine = self.engine(prefix_cache=cache)
        engine.admit(self.prepared_input("publish-failure"), max_new_tokens=2)
        with mock.patch.object(
            cache,
            "publish",
            side_effect=RuntimeError("injected prefix publication failure"),
        ):
            steps = engine.run_until_drained()

        failures = tuple(failure for step in steps for failure in step.prefix_publication_failures)
        self.assertTrue(failures)
        self.assertEqual(failures[0].request_id, "publish-failure")
        self.assertEqual(failures[0].error_type, "RuntimeError")
        self.assertIn("injected prefix", failures[0].message)
        self.assertEqual(len(self.token_ids(steps, "publish-failure")), 2)
        self.assertTrue(engine.scheduler.drained)
        self.assertEqual(
            engine.store.current_revision("publish-failure"),
            len(steps),
        )

        baseline = self.engine()
        baseline.admit(
            self.prepared_input("publish-baseline"),
            max_new_tokens=2,
            use_prefix_cache=False,
        )
        baseline.run_until_drained()
        self.assertEqual(
            engine.terminal_digest("publish-failure").cache_digest,
            baseline.terminal_digest("publish-baseline").cache_digest,
        )
        self.assertEqual(
            engine.terminal_digest("publish-failure").state_digest,
            baseline.terminal_digest("publish-baseline").state_digest,
        )

    def test_prefill_chunk_m_changes_only_lifecycle_not_semantic_state(self) -> None:
        observations = []
        for chunk in (1, 2, 4):
            engine = self.engine(budget=chunk, chunk=chunk)
            request_id = f"m{chunk}"
            engine.admit(
                self.prepared_input(request_id),
                max_new_tokens=2,
                use_prefix_cache=False,
            )
            steps = engine.run_until_drained()
            digest = engine.terminal_digest(request_id)
            observations.append(
                (
                    self.token_ids(steps, request_id),
                    digest.cache_digest,
                    digest.state_digest,
                    self.canonical_output_logits(steps, request_id),
                    digest.state_revision,
                )
            )

        self.assertEqual(len({item[0] for item in observations}), 1)
        self.assertEqual(len({item[1] for item in observations}), 1)
        self.assertEqual(len({item[2] for item in observations}), 1)
        self.assertEqual(len({item[3] for item in observations}), 1)
        self.assertEqual(len(observations[0][3]), 2)
        self.assertGreater(len({item[4] for item in observations}), 1)

    def test_transaction_rejects_reordered_segment_cache_slots(self) -> None:
        engine = self.engine(budget=2, chunk=2)
        engine.admit(
            self.prepared_input("slot-order"),
            max_new_tokens=1,
            use_prefix_cache=False,
        )
        plan = engine.scheduler.next_plan()
        assert plan is not None
        segment = plan.segments[0]
        self.assertEqual(len(segment.cache_slots), 2)
        reordered = replace(segment, cache_slots=tuple(reversed(segment.cache_slots)))

        with self.assertRaisesRegex(ValueError, "page-table ordinal mapping"):
            engine._begin_transaction(reordered, plan.state_transactions[0])

    def test_program_initialize_and_execute_callbacks_enter_reference_context(self) -> None:
        engine = self.engine(budget=2, chunk=2)
        probe = _ReferenceContextProbe(engine._program)
        engine._program = probe

        engine.admit(
            self.prepared_input("reference-context"),
            max_new_tokens=1,
            use_prefix_cache=False,
        )
        engine.step()

        self.assertTrue(probe.initialized)
        self.assertTrue(probe.executed)

    def test_state_capability_cannot_cross_segments_in_one_open_batch(self) -> None:
        engine = self.engine(budget=4, chunk=2)
        for request_id in ("capability-a", "capability-b"):
            engine.admit(
                self.prepared_input(request_id),
                max_new_tokens=1,
                use_prefix_cache=False,
            )
        engine._program = _RetainStateAcrossSegments(engine._program)

        with self.assertRaisesRegex(RuntimeError, "state capability was revoked"):
            engine.step()

        for request_id in ("capability-a", "capability-b"):
            state = engine.scheduler.request_state(request_id)
            self.assertEqual((state.prefill_cursor, state.state_revision), (0, 0))
            self.assertIsNone(state.inflight)

    def test_failure_aborts_tentative_kv_and_scheduler_retry_is_exact(self) -> None:
        engine = self.engine()
        engine.admit(self.prepared_input("retry"), max_new_tokens=1, use_prefix_cache=False)
        original = engine._program
        engine._program = _FailAfterMutation(original)
        with self.assertRaisesRegex(RuntimeError, "tentative KV"):
            engine.step()

        state = engine.scheduler.request_state("retry")
        self.assertEqual((state.prefill_cursor, state.state_revision), (0, 0))
        page = engine.allocator.page_table("retry").logical_to_physical[0]
        self.assertFalse(bool(torch.any(engine.store.page_written_mask("retry", page)).item()))
        self.assertTrue(engine.store.unwritten_storage_is_zero("retry", page))

        engine._program = original
        steps = engine.run_until_drained()
        self.assertEqual(len(self.token_ids(steps, "retry")), 1)

    def test_nonfinite_initial_state_rolls_back_admission(self) -> None:
        engine = self.engine()
        engine._program = _NonFiniteInitialization(engine._program)

        with self.assertRaisesRegex(ValueError, "finite"):
            engine.admit(
                self.prepared_input("bad-initial-state"),
                max_new_tokens=1,
                use_prefix_cache=False,
            )

        self.assertEqual(len(engine.features), 0)
        with self.assertRaises(KeyError):
            engine.scheduler.request_state("bad-initial-state")
        with self.assertRaises(OwnershipError):
            engine.store.current_revision("bad-initial-state")

    def test_nonfinite_kv_cannot_commit_from_intermediate_prefill_without_logits(self) -> None:
        engine = self.engine(budget=2, chunk=2)
        engine.admit(
            self.prepared_input("nan-prefill"),
            max_new_tokens=1,
            use_prefix_cache=False,
        )
        engine._program = _NonFiniteKVProgram(engine._program)

        with self.assertRaisesRegex(ValueError, "finite"):
            engine.step()

        state = engine.scheduler.request_state("nan-prefill")
        self.assertEqual((state.prefill_cursor, state.state_revision), (0, 0))
        self.assertIsNone(state.inflight)
        self.assertEqual(engine.store.current_revision("nan-prefill"), 0)
        page = engine.allocator.page_table("nan-prefill").logical_to_physical[0]
        self.assertFalse(
            bool(torch.any(engine.store.page_written_mask("nan-prefill", page)).item())
        )

    def test_nonfinite_logits_abort_before_cache_commit(self) -> None:
        engine = self.engine()
        engine.admit(self.prepared_input("nan"), max_new_tokens=1, use_prefix_cache=False)
        first = engine.step()
        self.assertIsNotNone(first)
        self.assertEqual(engine.store.current_revision("nan"), 1)
        second = engine.step()
        self.assertIsNotNone(second)
        self.assertEqual(engine.store.current_revision("nan"), 2)

        original = engine._program
        engine._program = _NonFiniteAfterMutation(original)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            engine.step()

        state = engine.scheduler.request_state("nan")
        self.assertEqual((state.prefill_cursor, state.state_revision), (4, 2))
        self.assertIsNone(state.inflight)
        self.assertEqual(engine.store.current_revision("nan"), 2)
        table = engine.allocator.page_table("nan")
        second_page = table.logical_to_physical[1]
        self.assertFalse(bool(torch.any(engine.store.page_written_mask("nan", second_page)).item()))

    def test_batch_commit_failure_aborts_every_request_without_partial_publish(self) -> None:
        engine = self.engine(budget=4, chunk=2)
        for request_id in ("a", "b"):
            engine.admit(
                self.prepared_input(request_id),
                max_new_tokens=1,
                use_prefix_cache=False,
            )

        with (
            mock.patch.object(
                engine.store,
                "commit_bundle",
                side_effect=RuntimeError("injected bundle failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "injected bundle"),
        ):
            engine.step()

        for request_id in ("a", "b"):
            state = engine.scheduler.request_state(request_id)
            self.assertEqual((state.prefill_cursor, state.state_revision), (0, 0))
            self.assertIsNone(state.inflight)
            page = engine.allocator.page_table(request_id).logical_to_physical[0]
            self.assertFalse(
                bool(torch.any(engine.store.page_written_mask(request_id, page)).item())
            )


if __name__ == "__main__":
    unittest.main()
