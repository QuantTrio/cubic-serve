from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import Tensor

from cubic_serve.contracts.execution import BatchPhase, OpSignature, PlanSegment
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
from cubic_serve.executor.runner import ReferenceExecutionContext
from cubic_serve.inputs import EncodedFeature, EncodedFeatureBundle, FeatureSpan, FeatureStore
from cubic_serve.models.protocol import ModelSegment, PreparedInput
from cubic_serve.models.qwen3_5.model import (
    FullAttentionConfig,
    GatedDeltaNetConfig,
    ModalityTokenConfig,
    ModelIdentity,
    MRoPEConfig,
    MTPMetadata,
    MTPStatus,
    Qwen3_5Config,
    Qwen3_5FeatureFusionError,
    Qwen3_5GraphConfig,
    Qwen3_5GraphVariant,
    Qwen3_5Program,
    Qwen3_5ReferenceModel,
    Qwen3_5ReferenceWeights,
    Qwen3_5TextConfig,
    VisionConfig,
    _decoder_input_features,
    fuse_qwen3_5_decoder_input,
    interleaved_mrope_cos_sin,
)
from cubic_serve.ops.eager.linear import linear as reference_linear

_IMAGE_TOKEN = 14
_VIDEO_TOKEN = 15
_HIDDEN = 6


class _TinyWeights:
    def __init__(self) -> None:
        embeddings = torch.zeros(16, _HIDDEN, dtype=torch.bfloat16)
        embeddings[1, 0] = 1
        embeddings[2, 1] = 1
        embeddings[_IMAGE_TOKEN, 2] = 1
        fc = torch.zeros(_HIDDEN, _HIDDEN * 2, dtype=torch.bfloat16)
        fc[:, :_HIDDEN] = torch.eye(_HIDDEN, dtype=torch.bfloat16)
        self._tensors: dict[str, Tensor] = {
            "lm_head.weight": torch.eye(16, _HIDDEN, dtype=torch.bfloat16),
            "model.language_model.embed_tokens.weight": embeddings,
            "model.language_model.norm.weight": torch.zeros(_HIDDEN, dtype=torch.bfloat16),
            "mtp.fc.weight": fc,
        }
        shapes = {
            "mtp.norm.weight": (_HIDDEN,),
            "mtp.pre_fc_norm_embedding.weight": (_HIDDEN,),
            "mtp.pre_fc_norm_hidden.weight": (_HIDDEN,),
            "mtp.layers.0.input_layernorm.weight": (_HIDDEN,),
            "mtp.layers.0.post_attention_layernorm.weight": (_HIDDEN,),
            "mtp.layers.0.self_attn.q_proj.weight": (_HIDDEN * 2, _HIDDEN),
            "mtp.layers.0.self_attn.k_proj.weight": (_HIDDEN, _HIDDEN),
            "mtp.layers.0.self_attn.v_proj.weight": (_HIDDEN, _HIDDEN),
            "mtp.layers.0.self_attn.o_proj.weight": (_HIDDEN, _HIDDEN),
            "mtp.layers.0.self_attn.q_norm.weight": (_HIDDEN,),
            "mtp.layers.0.self_attn.k_norm.weight": (_HIDDEN,),
            "mtp.layers.0.mlp.gate_proj.weight": (12, _HIDDEN),
            "mtp.layers.0.mlp.up_proj.weight": (12, _HIDDEN),
            "mtp.layers.0.mlp.down_proj.weight": (_HIDDEN, 12),
        }
        self._tensors.update(
            {name: torch.zeros(shape, dtype=torch.bfloat16) for name, shape in shapes.items()}
        )
        self.device = torch.device("cpu")
        self.inventory_digest = f"b2s256:{'1' * 64}"
        self.semantic_plan_digest = f"b2s256:{'2' * 64}"

    def __getitem__(self, name: str) -> Tensor:
        return self._tensors[name]

    def named_tensors(self) -> tuple[tuple[str, Tensor], ...]:
        return tuple(sorted(self._tensors.items()))


def _tiny_config() -> Qwen3_5Config:
    text = Qwen3_5TextConfig(
        model_type="qwen3_5_text",
        dtype="bfloat16",
        vocabulary_size=16,
        hidden_size=_HIDDEN,
        layer_count=0,
        max_position_embeddings=64,
        activation="silu",
        rms_norm_epsilon=1e-6,
        layer_schedule=(),
        full_attention_interval=1,
        dense_intermediate_size=12,
        full_attention=FullAttentionConfig(1, 1, _HIDDEN, True),
        gated_delta_net=GatedDeltaNetConfig(2, 2, 2, 1, 1, "float32", "swish"),
        moe=None,
        mrope=MRoPEConfig(True, (1, 1, 1), 1.0, 10_000.0, "default"),
        mtp=MTPMetadata(MTPStatus.METADATA_ONLY, 1, False),
    )
    graph = Qwen3_5GraphConfig(
        variant=Qwen3_5GraphVariant.DENSE,
        root_model_type="qwen3_5",
        architecture="fixture",
        text=text,
        vision=VisionConfig(
            "qwen3_5",
            1,
            2,
            4,
            _HIDDEN,
            1,
            3,
            2,
            1,
            2,
            4,
            "gelu_pytorch_tanh",
            (),
        ),
        modality_tokens=ModalityTokenConfig(_IMAGE_TOKEN, _VIDEO_TOKEN, 12, 13),
    )
    return Qwen3_5Config(
        ModelIdentity("fixture", "fixture", "qwen3_5", "qwen3_5", "fixture"),
        graph,
    )


def _tiny_model() -> Qwen3_5ReferenceModel:
    weights = cast(Qwen3_5ReferenceWeights, cast(Any, _TinyWeights()))
    return Qwen3_5ReferenceModel(config=_tiny_config(), weights=weights)


class Qwen3_5MTPReferenceTests(unittest.TestCase):
    def test_eager_mtp_step_requires_explicit_target_positions(self) -> None:
        model = _tiny_model()
        root = torch.tensor([[2]], dtype=torch.int64)
        target_hidden = torch.ones(1, 1, _HIDDEN, dtype=torch.bfloat16)
        first_position = torch.zeros((3, 1, 1), dtype=torch.int64)
        second_position = torch.ones((3, 1, 1), dtype=torch.int64)

        with ReferenceExecutionContext(), patch(
            "cubic_serve.models.qwen3_5.model.interleaved_mrope_cos_sin",
            wraps=interleaved_mrope_cos_sin,
        ) as rope:
            first = model.mtp_forward(root, target_hidden, position_ids=first_position)
            self.assertTrue(torch.equal(rope.call_args.args[0], first_position))
            second = model.mtp_forward(
                root, target_hidden, position_ids=second_position, cache=first.cache
            )
            with self.assertRaisesRegex(TypeError, "position_ids"):
                model.mtp_forward(root, target_hidden)  # type: ignore[call-arg]
            with self.assertRaisesRegex(ValueError, "position domain"):
                model.mtp_forward(
                    root, target_hidden, position_ids=torch.full((3, 1, 1), -1)
                )

        self.assertEqual(tuple(first.logits.shape), (1, 1, 16))
        self.assertEqual(first.hidden_states.dtype, torch.bfloat16)
        self.assertEqual(first.cache.sequence_length, 1)
        self.assertEqual(second.cache.sequence_length, 2)
        assert second.cache.layer.key is not None
        assert second.cache.layer.value is not None
        self.assertEqual(tuple(second.cache.layer.key.shape), (1, 1, 2, _HIDDEN))
        self.assertEqual(second.cache.layer.key.dtype, torch.bfloat16)
        self.assertEqual(second.cache.layer.value.dtype, torch.bfloat16)
        self.assertEqual(int(first.logits[0, 0].argmax().item()), 1)


def _input_plan(*, consumer_layer: int = 0) -> InputPlan:
    media_digest = f"b2s256:{'3' * 64}"
    return InputPlan(
        request_id=f"fusion-layer-{consumer_layer}",
        model_id="fixture",
        processor_recipe_digest=f"b2s256:{'4' * 64}",
        weight_variant=WeightVariantMeta("fixture", f"b2s256:{'5' * 64}"),
        content_parts=(
            ContentPart(0, "text", text="look"),
            ContentPart(
                1,
                "image",
                media=MediaDescriptor(media_digest, "image/png", 16, 2, 2),
            ),
        ),
        token_ids=(1, _IMAGE_TOKEN),
        positions=(TokenPosition((0, 0, 0)), TokenPosition((1, 1, 1))),
        decode_positions=DecodePositionPlan(TokenPosition((2, 2, 2)), (1, 1, 1)),
        placements=(TokenPlacement(1, 1, 1, 0, 1, (consumer_layer,)),),
        encoder_jobs=(EncoderJob("vision:1", 1, "vision", "processor", "cache:1"),),
    )


def _prepared_and_store(*, consumer_layer: int = 0) -> tuple[PreparedInput, FeatureStore]:
    plan = _input_plan(consumer_layer=consumer_layer)
    feature = EncodedFeature(
        "vision:1",
        1,
        "vision",
        "processor",
        f"b2s256:{'3' * 64}",
        torch.tensor([[0, 1, 0, 0, 0, 0]], dtype=torch.bfloat16),
    )
    source = EncodedFeatureBundle.from_input_plan(plan, (feature,))
    store = FeatureStore()
    return PreparedInput(plan, store.publish(plan, source)), store


def _segment(
    prepared: PreparedInput,
    *,
    phase: BatchPhase = BatchPhase.PREFILL,
) -> PlanSegment:
    plan = prepared.input_plan
    if phase is BatchPhase.PREFILL:
        token_start = 0
        token_ids = plan.token_ids
        positions = plan.positions
    else:
        token_start = len(plan.token_ids)
        token_ids = (3,)
        positions = (plan.decode_positions.first,)
    return PlanSegment(
        request_id=plan.request_id,
        input_plan_digest=plan.semantic_digest,
        execution_input_digest=prepared.semantic_digest,
        phase=phase,
        produces_output=True,
        token_start=token_start,
        token_count=len(token_ids),
        token_ids=token_ids,
        positions=positions,
        cache_slots=tuple(range(token_start, token_start + len(token_ids))),
        cache_pages=(0,),
    )


class _ReferenceOps:
    def call(self, signature: OpSignature, /, *args: Any, **kwargs: Any) -> Any:
        if signature.semantic_op != "linear" or kwargs or len(args) != 2:
            raise AssertionError("zero-layer fixture only permits the final reference linear")
        return reference_linear(cast(Tensor, args[0]), cast(Tensor, args[1]))


class _UnusedState:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"zero-layer fixture cannot access state through {name}")


class _StaticReader:
    def __init__(self, spans: tuple[FeatureSpan, ...]) -> None:
        self._spans = spans

    def spans(self, consumer_site: str, consumer_layer: int) -> tuple[FeatureSpan, ...]:
        if consumer_site == "decoder.input" and consumer_layer == 0:
            return self._spans
        return ()


class Qwen3_5DecoderFusionTests(unittest.TestCase):
    def test_identity_rows_replace_every_and_only_media_placeholder(self) -> None:
        input_ids = torch.tensor([[1, _IMAGE_TOKEN, _IMAGE_TOKEN, 2, _VIDEO_TOKEN]])
        text = torch.arange(30, dtype=torch.bfloat16).reshape(1, 5, _HIDDEN)
        image = torch.tensor(
            [[90, 91, 92, 93, 94, 95], [80, 81, 82, 83, 84, 85]],
            dtype=torch.bfloat16,
        )
        video = torch.tensor([[70, 71, 72, 73, 74, 75]], dtype=torch.bfloat16)

        actual = fuse_qwen3_5_decoder_input(
            input_ids,
            text,
            (FeatureSpan(1, image), FeatureSpan(4, video)),
            image_token_id=_IMAGE_TOKEN,
            video_token_id=_VIDEO_TOKEN,
        )
        expected = text.clone()
        expected[0, 1:3] = image
        expected[0, 4:5] = video

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual[0, (0, 3)], text[0, (0, 3)]))
        plain = torch.tensor([[1, 2]])
        plain_embeddings = text[:, :2]
        self.assertIs(
            fuse_qwen3_5_decoder_input(
                plain,
                plain_embeddings,
                (),
                image_token_id=_IMAGE_TOKEN,
                video_token_id=_VIDEO_TOKEN,
            ),
            plain_embeddings,
        )

    def test_shape_dtype_device_and_coverage_fail_closed(self) -> None:
        input_ids = torch.tensor([[1, _IMAGE_TOKEN]])
        text = torch.zeros((1, 2, _HIDDEN), dtype=torch.bfloat16)
        valid = FeatureSpan(1, torch.ones((1, _HIDDEN), dtype=torch.bfloat16))
        cases: tuple[tuple[str, tuple[FeatureSpan, ...]], ...] = (
            ("cover", ()),
            ("cover", (FeatureSpan(0, valid.rows),)),
            ("exceeds", (FeatureSpan(2, valid.rows),)),
            ("row ABI", (FeatureSpan(1, torch.ones((1, 4), dtype=torch.bfloat16)),)),
            (
                "non-finite",
                (FeatureSpan(1, torch.full((1, _HIDDEN), float("nan"), dtype=torch.bfloat16)),),
            ),
            (
                "row ABI",
                (FeatureSpan(1, torch.ones((1, _HIDDEN), dtype=torch.bfloat16, device="meta")),),
            ),
        )
        for message, spans in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(Qwen3_5FeatureFusionError, message),
            ):
                fuse_qwen3_5_decoder_input(
                    input_ids,
                    text,
                    spans,
                    image_token_id=_IMAGE_TOKEN,
                    video_token_id=_VIDEO_TOKEN,
                )

        wrong_dtype = FeatureSpan(1, valid.rows.clone())
        object.__setattr__(wrong_dtype, "rows", torch.ones((1, _HIDDEN)))
        with self.assertRaisesRegex(Qwen3_5FeatureFusionError, "row ABI"):
            fuse_qwen3_5_decoder_input(
                input_ids,
                text,
                (wrong_dtype,),
                image_token_id=_IMAGE_TOKEN,
                video_token_id=_VIDEO_TOKEN,
            )

    def test_reference_forward_matches_equivalent_text_embedding_exactly(self) -> None:
        model = _tiny_model()
        feature = FeatureSpan(
            1,
            torch.tensor([[0, 1, 0, 0, 0, 0]], dtype=torch.bfloat16),
        )
        with ReferenceExecutionContext():
            expected = model.forward(torch.tensor([[1, 2]])).logits
            actual = model.forward(
                torch.tensor([[1, _IMAGE_TOKEN]]),
                decoder_input_features=(feature,),
            ).logits
            placeholder = model.forward(torch.tensor([[1, _IMAGE_TOKEN]])).logits

        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, placeholder))

    def test_program_consumes_only_segment_authorized_rows(self) -> None:
        prepared, store = _prepared_and_store()
        segment = _segment(prepared)
        reader = store.reader_for_segment(prepared.input_plan, segment)
        program = Qwen3_5Program(_tiny_model(), ())

        with ReferenceExecutionContext():
            result = program.execute_segment(
                ModelSegment.from_plan(segment),
                _ReferenceOps(),
                _UnusedState(),
                reader,
            )
            expected = _tiny_model().forward(torch.tensor([[1, 2]])).logits[0, -1]

        self.assertTrue(torch.equal(result.logits, expected))
        reader.revoke()

    def test_consumer_layer_and_decode_authority_fail_closed(self) -> None:
        wrong, store = _prepared_and_store(consumer_layer=1)
        wrong_segment = _segment(wrong)
        wrong_reader = store.reader_for_segment(wrong.input_plan, wrong_segment)
        config = _tiny_config()
        config = replace(
            config,
            graph=replace(
                config.graph,
                text=replace(config.graph.text, layer_count=2),
            ),
        )
        with self.assertRaisesRegex(Qwen3_5FeatureFusionError, "layer 0"):
            _decoder_input_features(ModelSegment.from_plan(wrong_segment), wrong_reader, config)

        valid, valid_store = _prepared_and_store()
        decode = _segment(valid, phase=BatchPhase.DECODE)
        decode_reader = valid_store.reader_for_segment(valid.input_plan, decode)
        self.assertIsNone(
            _decoder_input_features(ModelSegment.from_plan(decode), decode_reader, _tiny_config())
        )
        leaked = FeatureSpan(0, torch.ones((1, _HIDDEN), dtype=torch.bfloat16))
        with self.assertRaisesRegex(Qwen3_5FeatureFusionError, "decode"):
            _decoder_input_features(
                ModelSegment.from_plan(decode),
                _StaticReader((leaked,)),
                _tiny_config(),
            )


if __name__ == "__main__":
    unittest.main()


class MediaPixelLimitTests(unittest.TestCase):
    def test_checkpoint_modern_legacy_and_partial_limits_share_official_defaults(self):
        import json
        import tempfile
        from pathlib import Path
        from cubic_serve.models.qwen3_5.multimodal import qwen_media_pixel_limits

        defaults = {'shortest_edge': 3136, 'longest_edge': 1003520}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'preprocessor_config.json'
            cases = [
                ({'size': {'shortest_edge': 65536, 'longest_edge': 16777216}},
                 (65536, 16777216)),
                ({'min_pixels': 4096, 'max_pixels': 4194304}, (4096, 4194304)),
                ({'size': {'longest_edge': 16777216}},
                 (defaults['shortest_edge'], 16777216)),
                ({}, (defaults['shortest_edge'], defaults['longest_edge'])),
            ]
            for document, expected in cases:
                path.write_text(json.dumps(document))
                with self.subTest(document=document):
                    self.assertEqual(qwen_media_pixel_limits(folder), expected)
            path.unlink()
            self.assertEqual(qwen_media_pixel_limits(folder),
                             (defaults['shortest_edge'], defaults['longest_edge']))
            self.assertEqual(qwen_media_pixel_limits(folder, video=True), (131072, 786432))

    def test_invalid_limits_do_not_silently_fall_back(self):
        import json
        import tempfile
        from pathlib import Path
        from cubic_serve.models.qwen3_5.multimodal import (
            Qwen3_5MultimodalError, qwen_media_pixel_limits,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'preprocessor_config.json'
            for size in (None, {'shortest_edge': 0, 'longest_edge': 10},
                         {'shortest_edge': 100, 'longest_edge': 10},
                         {'shortest_edge': 1, 'longest_edge': True}):
                path.write_text(json.dumps({'size': size}))
                with self.subTest(size=size), self.assertRaises(Qwen3_5MultimodalError):
                    qwen_media_pixel_limits(folder)
