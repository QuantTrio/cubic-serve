from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import Any, ClassVar

import torch

from cubic_serve.contracts.model import WeightVariantMeta
from cubic_serve.contracts.request import ChatRole, ContentPart, MediaDescriptor
from cubic_serve.executor.runner import ReferenceExecutionContext

_CHECKPOINT = Path("/pubdata/llm_models/Qwen/Qwen3.8-27B")
_DEVICE = torch.device("cuda:6")
_RUN_REAL = os.environ.get("CUBIC_RUN_REAL_QWEN_MULTIMODAL") == "1"


@unittest.skipUnless(
    _RUN_REAL and _CHECKPOINT.is_dir(),
    "real Qwen multimodal test is opt-in",
)
class Qwen3_5MultimodalRealTests(unittest.TestCase):
    official: ClassVar[Any]
    official_inputs: ClassVar[Any]
    oracle_vision: ClassVar[Any]
    position_helper: ClassVar[Any]
    prepared: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 6:
            raise unittest.SkipTest("physical GPU6 is unavailable")

        import numpy as np
        import transformers
        from PIL import Image
        from transformers import AutoProcessor
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5Model,
            Qwen3_5VisionModel,
        )
        from transformers.video_utils import VideoMetadata

        from cubic_serve.models.qwen3_5.multimodal import (
            Qwen3_5MediaPatches,
            Qwen3_5MultimodalBridge,
            Qwen3_5TrustedMessage,
            Qwen3_5VideoMetadata,
            load_qwen3_5_vision_reference,
        )

        if transformers.__version__ != "5.14.1":
            raise AssertionError("multimodal oracle must be Transformers 5.14.1")
        cls.official = AutoProcessor.from_pretrained(_CHECKPOINT, local_files_only=True)
        image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        video = np.zeros((4, 32, 32, 3), dtype=np.uint8)
        structured_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before "},
                    {"type": "image"},
                    {"type": "text", "text": " middle "},
                    {"type": "video"},
                    {"type": "text", "text": " after"},
                ],
            }
        ]
        rendered = cls.official.apply_chat_template(
            structured_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        metadata = VideoMetadata(
            total_num_frames=4,
            fps=2.0,
            duration=2.0,
            video_backend="numpy",
            frames_indices=[0, 1, 2, 3],
        )
        cls.official_inputs = cls.official(
            text=[rendered],
            images=[image],
            videos=[video],
            video_metadata=[metadata],
            do_sample_frames=False,
            return_tensors="pt",
        )

        vision = load_qwen3_5_vision_reference(_CHECKPOINT, device=_DEVICE)
        bridge = Qwen3_5MultimodalBridge(_CHECKPOINT, vision=vision)
        parts = (
            ContentPart(0, "text", text="before "),
            ContentPart(
                1,
                "image",
                media=MediaDescriptor(
                    content_digest=f"b2s256:{'1' * 64}",
                    mime_type="image/png",
                    byte_length=32 * 32 * 3,
                    width=32,
                    height=32,
                ),
            ),
            ContentPart(2, "text", text=" middle "),
            ContentPart(
                3,
                "video",
                media=MediaDescriptor(
                    content_digest=f"b2s256:{'2' * 64}",
                    mime_type="video/raw",
                    byte_length=4 * 32 * 32 * 3,
                    width=32,
                    height=32,
                    frame_count=4,
                    duration_ms=2000,
                ),
            ),
            ContentPart(4, "text", text=" after"),
        )
        media = (
            Qwen3_5MediaPatches(
                1,
                "image",
                cls.official_inputs.pixel_values,
                tuple(cls.official_inputs.image_grid_thw[0].tolist()),
            ),
            Qwen3_5MediaPatches(
                3,
                "video",
                cls.official_inputs.pixel_values_videos,
                tuple(cls.official_inputs.video_grid_thw[0].tolist()),
                Qwen3_5VideoMetadata((0, 1, 2, 3), 2.0),
            ),
        )
        cls.prepared = bridge.prepare(
            request_id="mixed-image-video",
            model_id="qwen3_5:qwen3.8-27b:multimodal",
            weight_variant=WeightVariantMeta(
                base_model_id="qwen3_5:qwen3.8-27b:multimodal",
                base_revision=f"b2s256:{'a' * 64}",
            ),
            messages=(Qwen3_5TrustedMessage(ChatRole.USER, parts),),
            media_patches=media,
            enable_thinking=False,
        )

        config = Qwen3_5Config.from_pretrained(_CHECKPOINT, local_files_only=True)
        with torch.device("meta"):
            cls.position_helper = Qwen3_5Model(config)
        config.vision_config._attn_implementation = "eager"
        with torch.device("meta"):
            cls.oracle_vision = Qwen3_5VisionModel(config.vision_config)
        state = {
            name.removeprefix("model.visual."): tensor
            for name, tensor in vision.weights.named_tensors()
        }
        cls.oracle_vision.load_state_dict(state, strict=True, assign=True)
        rotary_dim = (config.vision_config.hidden_size // config.vision_config.num_heads) // 2
        cls.oracle_vision.rotary_pos_emb.inv_freq = (
            1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        ).to(_DEVICE)
        cls.oracle_vision.eval()

    def test_interleaved_tokens_timestamps_jobs_and_placements_are_exact(self) -> None:
        plan = self.prepared.input_plan
        expected_ids = tuple(self.official_inputs.input_ids[0].tolist())
        self.assertEqual(plan.token_ids, expected_ids)
        self.assertEqual(len(plan.token_ids), 103)
        self.assertEqual(
            tuple(part.modality for part in plan.content_parts),
            (
                "text",
                "image",
                "text",
                "video",
                "text",
            ),
        )
        decoded = self.official.tokenizer.decode(plan.token_ids, skip_special_tokens=False)
        self.assertIn(
            "<0.2 seconds><|vision_start|><|video_pad|><|vision_end|>"
            "<1.2 seconds><|vision_start|><|video_pad|><|vision_end|>",
            decoded,
        )
        self.assertEqual(
            tuple(
                (
                    item.content_part_ordinal,
                    item.token_start,
                    item.token_count,
                    item.feature_row_start,
                    item.feature_row_count,
                )
                for item in plan.placements
            ),
            ((1, 6, 64, 0, 64), (3, 81, 1, 0, 1), (3, 90, 1, 1, 1)),
        )
        self.assertTrue(all(placement.consumer_layers == (0,) for placement in plan.placements))
        self.assertEqual(tuple(job.content_part_ordinal for job in plan.encoder_jobs), (1, 3))
        self.assertTrue(all(job.cache_key.startswith("b2s256:") for job in plan.encoder_jobs))

    def test_three_axis_mrope_and_decode_continuation_match_official_helper(self) -> None:
        plan = self.prepared.input_plan
        expected, delta = self.position_helper.get_rope_index(
            self.official_inputs.input_ids,
            self.official_inputs.mm_token_type_ids,
            self.official_inputs.image_grid_thw,
            self.official_inputs.video_grid_thw,
        )
        actual = torch.tensor([position.axes for position in plan.positions]).T
        self.assertTrue(torch.equal(actual, expected[:, 0]))
        self.assertEqual(delta.item(), -56)
        self.assertEqual(plan.decode_positions.first.axes, (47, 47, 47))
        self.assertEqual(plan.decode_positions.step_axes, (1, 1, 1))

    def test_encoded_feature_bundle_matches_official_eager_vision_exactly(self) -> None:
        inputs = self.official_inputs
        with torch.inference_mode(), ReferenceExecutionContext():
            expected_image = self.oracle_vision(
                hidden_states=inputs.pixel_values.to(_DEVICE),
                grid_thw=inputs.image_grid_thw.to(_DEVICE),
            ).pooler_output
            expected_video = self.oracle_vision(
                hidden_states=inputs.pixel_values_videos.to(_DEVICE),
                grid_thw=inputs.video_grid_thw.to(_DEVICE),
            ).pooler_output

        bundle = self.prepared.feature_bundle
        assert bundle is not None
        bundle.verify_for_input_plan(self.prepared.input_plan)
        features = {
            binding.feature.content_part_ordinal: binding.feature for binding in bundle.bindings
        }
        self.assertEqual(tuple(features), (1, 3))
        self.assertEqual(tuple(features[1].tensor.shape), (64, 5120))
        self.assertEqual(tuple(features[3].tensor.shape), (2, 5120))
        self.assertIs(features[1].tensor.dtype, torch.bfloat16)
        self.assertIs(features[3].tensor.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(features[1].tensor, expected_image))
        self.assertTrue(torch.equal(features[3].tensor, expected_video))


if __name__ == "__main__":
    unittest.main()
