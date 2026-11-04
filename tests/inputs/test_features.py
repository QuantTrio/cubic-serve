from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from cubic_serve.contracts.execution import BatchPhase, PlanSegment
from cubic_serve.contracts.model import WeightVariantMeta
from cubic_serve.contracts.request import (
    ContentPart,
    DecodePositionPlan,
    EncoderJob,
    InputPlan,
    MediaDescriptor,
    TokenPlacement,
    TokenPosition,
)
from cubic_serve.inputs import (
    EncodedFeature,
    EncodedFeatureBundle,
    EncodedFeatureCache,
    FeatureAccessError,
    FeatureAdmissionError,
    FeatureIntegrityError,
    FeatureStore,
)
from cubic_serve.models import PreparedInput


def _media(digest_character: str = "1") -> MediaDescriptor:
    return MediaDescriptor(
        content_digest=f"b2s256:{digest_character * 64}",
        mime_type="image/png",
        byte_length=64,
        width=8,
        height=8,
    )


def _input_plan(request_id: str = "request:features") -> InputPlan:
    return InputPlan(
        request_id=request_id,
        model_id="fixture",
        processor_recipe_digest=f"b2s256:{'a' * 64}",
        weight_variant=WeightVariantMeta(
            base_model_id="fixture",
            base_revision="b2s256:model",
        ),
        content_parts=(
            ContentPart(ordinal=0, modality="text", text="describe"),
            ContentPart(ordinal=1, modality="image", media=_media()),
        ),
        token_ids=(1, 2, 3),
        positions=tuple(TokenPosition((index,)) for index in range(3)),
        decode_positions=DecodePositionPlan(
            first=TokenPosition((3,)),
            step_axes=(1,),
        ),
        placements=(
            TokenPlacement(
                content_part_ordinal=1,
                token_start=1,
                token_count=2,
                feature_row_start=1,
                feature_row_count=2,
                consumer_layers=(2, 4),
            ),
        ),
        encoder_jobs=(
            EncoderJob(
                job_id="encoder:image",
                content_part_ordinal=1,
                encoder_id="fixture.vision",
                processor_id="fixture.processor",
                cache_key="fixture:image:1",
            ),
        ),
    )


def _feature(
    *,
    digest_character: str = "1",
    tensor: torch.Tensor | None = None,
) -> EncodedFeature:
    if tensor is None:
        tensor = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    return EncodedFeature(
        job_id="encoder:image",
        content_part_ordinal=1,
        encoder_id="fixture.vision",
        processor_id="fixture.processor",
        source_media_digest=f"b2s256:{digest_character * 64}",
        tensor=tensor,
    )


def _bundle(
    plan: InputPlan,
    features: tuple[EncodedFeature, ...] | None = None,
) -> EncodedFeatureBundle:
    return EncodedFeatureBundle.from_input_plan(
        plan,
        (_feature(),) if features is None else features,
    )


def _video_input_plan(request_id: str = "request:video-spans") -> InputPlan:
    return InputPlan(
        request_id=request_id,
        model_id="fixture",
        processor_recipe_digest=f"b2s256:{'a' * 64}",
        weight_variant=WeightVariantMeta(
            base_model_id="fixture",
            base_revision="b2s256:model",
        ),
        content_parts=(
            ContentPart(ordinal=0, modality="text", text="compare the clips"),
            ContentPart(
                ordinal=1,
                modality="video",
                media=MediaDescriptor(
                    content_digest=f"b2s256:{'2' * 64}",
                    mime_type="video/mp4",
                    byte_length=512,
                    width=16,
                    height=8,
                    frame_count=5,
                    duration_ms=500,
                ),
            ),
        ),
        token_ids=tuple(range(10, 19)),
        positions=tuple(TokenPosition((index,)) for index in range(9)),
        decode_positions=DecodePositionPlan(
            first=TokenPosition((9,)),
            step_axes=(1,),
        ),
        placements=(
            TokenPlacement(
                content_part_ordinal=1,
                token_start=1,
                token_count=2,
                feature_row_start=0,
                feature_row_count=2,
                consumer_layers=(2, 4),
            ),
            TokenPlacement(
                content_part_ordinal=1,
                token_start=5,
                token_count=3,
                feature_row_start=2,
                feature_row_count=3,
                consumer_layers=(2, 4),
            ),
        ),
        encoder_jobs=(
            EncoderJob(
                job_id="encoder:video",
                content_part_ordinal=1,
                encoder_id="fixture.vision",
                processor_id="fixture.processor",
                cache_key="fixture:video:2",
            ),
        ),
    )


def _video_feature() -> EncodedFeature:
    return EncodedFeature(
        job_id="encoder:video",
        content_part_ordinal=1,
        encoder_id="fixture.vision",
        processor_id="fixture.processor",
        source_media_digest=f"b2s256:{'2' * 64}",
        tensor=torch.arange(15, dtype=torch.bfloat16).reshape(5, 3),
    )


def _segment(
    prepared_input: PreparedInput,
    *,
    token_start: int,
    token_count: int,
    phase: BatchPhase = BatchPhase.PREFILL,
) -> PlanSegment:
    plan = prepared_input.input_plan
    if phase is BatchPhase.PREFILL:
        token_ids = plan.token_ids[token_start : token_start + token_count]
        positions = plan.positions[token_start : token_start + token_count]
    else:
        token_ids = tuple(7 for _ in range(token_count))
        positions = tuple(
            plan.decode_positions.at(token_start - len(plan.token_ids) + offset)
            for offset in range(token_count)
        )
    return PlanSegment(
        request_id=plan.request_id,
        input_plan_digest=plan.semantic_digest,
        execution_input_digest=prepared_input.semantic_digest,
        phase=phase,
        produces_output=(
            phase is BatchPhase.DECODE
            or token_start + token_count == len(plan.token_ids)
        ),
        token_start=token_start,
        token_count=token_count,
        token_ids=token_ids,
        positions=positions,
        cache_slots=tuple(range(token_start, token_start + token_count)),
        cache_pages=(0,),
    )


class EncodedFeatureTests(unittest.TestCase):
    def test_semantic_identity_binds_media_digest_and_exact_tensor_bits(self) -> None:
        baseline = _feature()
        different_media = _feature(digest_character="2")
        changed_tensor = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
        changed_tensor[0, 0] = 99
        different_bits = _feature(tensor=changed_tensor)

        self.assertNotEqual(baseline.semantic_digest, different_media.semantic_digest)
        self.assertEqual(baseline.tensor_content_digest, different_media.tensor_content_digest)
        self.assertNotEqual(baseline.semantic_digest, different_bits.semantic_digest)
        self.assertNotEqual(baseline.tensor_content_digest, different_bits.tensor_content_digest)
        self.assertEqual(baseline.device, torch.device("cpu"))

    def test_bundle_semantic_identity_is_request_independent_but_request_bound(self) -> None:
        first_plan = _input_plan("request:first")
        second_plan = _input_plan("request:second")
        first = _bundle(first_plan)
        second = _bundle(second_plan)

        self.assertEqual(first.semantic_digest, second.semantic_digest)
        self.assertEqual(
            first.semantic_digest_for_parts((1,)),
            second.semantic_digest_for_parts((1,)),
        )
        with self.assertRaisesRegex(FeatureAdmissionError, "request identity"):
            first.verify_for_input_plan(second_plan)

    def test_tensor_contract_is_exact_bf16_rows_hidden_contiguous(self) -> None:
        with self.assertRaisesRegex(TypeError, "BF16"):
            _feature(tensor=torch.ones((4, 3), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, r"\[rows, hidden\]"):
            _feature(tensor=torch.ones(4, dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            _feature(tensor=torch.ones((3, 4), dtype=torch.bfloat16).transpose(0, 1))
        for nonfinite in (float("nan"), float("inf"), float("-inf")):
            tensor = torch.ones((4, 3), dtype=torch.bfloat16)
            tensor[0, 0] = nonfinite
            with (
                self.subTest(nonfinite=nonfinite),
                self.assertRaisesRegex(ValueError, "finite"),
            ):
                _feature(tensor=tensor)

    def test_feature_cache_is_byte_bounded_and_rejects_identity_drift(self) -> None:
        first = _feature()
        second = _feature(digest_character="2")
        cache = EncodedFeatureCache(first.tensor.numel() * first.tensor.element_size())

        self.assertIs(cache.put("first", first), first)
        self.assertIs(cache.get("first"), first)
        cache.put("second", second)
        self.assertIsNone(cache.get("first"))
        self.assertIs(cache.get("second"), second)
        with self.assertRaisesRegex(FeatureIntegrityError, "changed identity"):
            cache.put("second", first)


class FeatureStoreTests(unittest.TestCase):
    def test_segment_reader_owns_only_current_rows_and_returns_detached_copies(self) -> None:
        plan = _input_plan()
        feature = _feature()
        original = feature.tensor.clone()
        store = FeatureStore()

        source_bundle = _bundle(plan, (feature,))
        bundle = store.publish(plan, source_bundle)
        self.assertIsNot(bundle, source_bundle)
        self.assertIsNot(bundle.bindings[0].feature, feature)
        self.assertNotEqual(bundle.bindings[0].feature.tensor.data_ptr(), feature.tensor.data_ptr())
        feature.tensor.add_(100)
        prepared = PreparedInput(plan, bundle)
        reader = store.reader_for_segment(
            plan,
            _segment(prepared, token_start=1, token_count=2),
        )
        self.assertFalse(hasattr(reader, "_store"))
        spans = reader.spans("decoder.input", 2)
        self.assertEqual(len(spans), 1)
        observed = spans[0].rows

        self.assertTrue(torch.equal(observed, original[1:3]))
        self.assertEqual((spans[0].token_offset, spans[0].token_count), (0, 2))
        self.assertEqual(bundle.semantic_digest, store.bundle_digest(plan.request_id))
        self.assertEqual(len(store), 1)

        observed.zero_()
        reread = reader.spans("decoder.input", 4)[0].rows
        self.assertTrue(torch.equal(reread, original[1:3]))
        reader.revoke()
        with self.assertRaisesRegex(FeatureAccessError, "expired"):
            reader.spans("decoder.input", 2)

    def test_reader_exposes_only_current_span_without_job_or_digest_metadata(self) -> None:
        plan = _input_plan()
        feature = _feature()
        store = FeatureStore()
        bundle = store.publish(plan, _bundle(plan, (feature,)))
        prepared = PreparedInput(plan, bundle)
        reader = store.reader_for_segment(
            plan,
            _segment(prepared, token_start=1, token_count=1),
        )
        spans = reader.spans("decoder.input", 2)

        self.assertEqual(len(spans), 1)
        self.assertEqual((spans[0].token_offset, spans[0].token_count), (0, 1))
        self.assertFalse(hasattr(spans[0], "job_id"))
        self.assertFalse(hasattr(spans[0], "semantic_digest"))
        self.assertEqual(reader.spans("decoder.input", 3), ())
        self.assertEqual(reader.spans("decoder.layer.2.pre", 2), ())

    def test_decode_segment_has_no_prompt_media_capability(self) -> None:
        plan = _input_plan()
        feature = _feature()
        store = FeatureStore()
        bundle = store.publish(plan, _bundle(plan, (feature,)))
        prepared = PreparedInput(plan, bundle)
        reader = store.reader_for_segment(
            plan,
            _segment(
                prepared,
                token_start=len(plan.token_ids),
                token_count=1,
                phase=BatchPhase.DECODE,
            ),
        )

        self.assertEqual(reader.spans("decoder.input", 2), ())

    def test_publication_rejects_wrong_source_digest_and_duplicate_job(self) -> None:
        plan = _input_plan()
        feature = _feature()
        with self.assertRaisesRegex(FeatureAdmissionError, "source media digest"):
            _bundle(plan, (_feature(digest_character="2"),))
        with self.assertRaisesRegex(FeatureAdmissionError, "duplicate encoder job"):
            _bundle(plan, (feature, feature))

        store = FeatureStore()
        bundle = _bundle(plan, (feature,))
        store.publish(plan, bundle)
        with self.assertRaisesRegex(FeatureAdmissionError, "already has"):
            store.publish(plan, bundle)

    def test_release_removes_request_owned_features(self) -> None:
        plan = _input_plan()
        feature = _feature()
        store = FeatureStore()
        store.publish(plan, _bundle(plan, (feature,)))

        store.release(plan.request_id)

        self.assertEqual(len(store), 0)
        with self.assertRaisesRegex(FeatureAccessError, "no feature bundle"):
            store.bundle_digest(plan.request_id)


class MultiPlacementVideoTests(unittest.TestCase):
    def test_one_video_feature_backs_two_noncontiguous_segment_spans(self) -> None:
        plan = _video_input_plan()
        feature = _video_feature()
        source_bundle = _bundle(plan, (feature,))

        self.assertEqual(len(source_bundle.bindings), 2)
        self.assertEqual(
            {id(binding.feature) for binding in source_bundle.bindings},
            {id(feature)},
        )

        store = FeatureStore()
        bundle = store.publish(plan, source_bundle)
        prepared = PreparedInput(plan, bundle)
        self.assertEqual(len(store), 1)
        self.assertEqual(len({id(binding.feature) for binding in bundle.bindings}), 1)
        self.assertIsNot(bundle.bindings[0].feature, feature)

        reader = store.reader_for_segment(
            plan,
            _segment(prepared, token_start=0, token_count=len(plan.token_ids)),
        )
        spans = reader.spans("decoder.input", 2)

        self.assertEqual(tuple(span.token_offset for span in spans), (1, 5))
        self.assertTrue(torch.equal(spans[0].rows, feature.tensor[0:2]))
        self.assertTrue(torch.equal(spans[1].rows, feature.tensor[2:5]))

    def test_segment_reader_exposes_only_each_placement_overlap(self) -> None:
        plan = _video_input_plan()
        feature = _video_feature()
        store = FeatureStore()
        bundle = store.publish(plan, _bundle(plan, (feature,)))
        prepared = PreparedInput(plan, bundle)

        reader = store.reader_for_segment(
            plan,
            _segment(prepared, token_start=2, token_count=4),
        )
        spans = reader.spans("decoder.input", 4)

        self.assertEqual(tuple(span.token_offset for span in spans), (0, 3))
        self.assertTrue(torch.equal(spans[0].rows, feature.tensor[1:2]))
        self.assertTrue(torch.equal(spans[1].rows, feature.tensor[2:3]))
        self.assertEqual(reader.spans("decoder.input", 3), ())

    def test_tamper_overlap_and_wrong_feature_coverage_fail_closed(self) -> None:
        plan = _video_input_plan()
        first, second = plan.placements

        with self.assertRaisesRegex(ValueError, "token placement spans cannot overlap"):
            replace(plan, placements=(first, replace(second, token_start=2)))
        with self.assertRaisesRegex(ValueError, "feature placement spans cannot overlap"):
            replace(plan, placements=(first, replace(second, feature_row_start=1)))

        out_of_bounds = replace(
            plan,
            placements=(first, replace(second, feature_row_start=4)),
        )
        with self.assertRaisesRegex(FeatureAdmissionError, "does not fit its placement"):
            _bundle(out_of_bounds, (_video_feature(),))

        feature = _video_feature()
        bundle = _bundle(plan, (feature,))
        drifted_plan = replace(plan, placements=(first, replace(second, token_start=6)))
        with self.assertRaisesRegex(FeatureAdmissionError, "semantic identity"):
            PreparedInput(drifted_plan, bundle)

        feature.tensor[0, 0] = 99
        with self.assertRaisesRegex(FeatureIntegrityError, "sealed digest"):
            bundle.verify_integrity()


if __name__ == "__main__":
    unittest.main()
