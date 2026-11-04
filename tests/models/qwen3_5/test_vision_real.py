from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import Any

import torch

from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
from cubic_serve.models.qwen3_5.multimodal import (
    Qwen3_5VisionReferenceModel,
    load_qwen3_5_vision_reference,
)

_CHECKPOINT = Path("/pubdata/llm_models/Qwen/Qwen3.8-27B")
_DEVICE = torch.device("cuda:6")
_RUN_REAL = os.environ.get("CUBIC_RUN_REAL_QWEN_VISION") == "1"


@unittest.skipUnless(_RUN_REAL and _CHECKPOINT.is_dir(), "real Qwen vision test is opt-in")
class Qwen3_5VisionRealTests(unittest.TestCase):
    subject: Qwen3_5VisionReferenceModel
    oracle: Any
    processor: Any

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 6:
            raise unittest.SkipTest("physical GPU6 is unavailable")

        import transformers
        from transformers import AutoProcessor
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        if transformers.__version__ != "5.14.1":
            raise AssertionError("vision oracle must be Transformers 5.14.1")
        configure_reference_math()
        cls.subject = load_qwen3_5_vision_reference(_CHECKPOINT, device=_DEVICE)
        if cls.subject.weights.tensor_count != 333 or not all(
            name.startswith("model.visual.") for name, _ in cls.subject.weights.named_tensors()
        ):
            raise AssertionError("vision reference did not load the exact 333-tensor slice")

        full_config = Qwen3_5Config.from_pretrained(_CHECKPOINT, local_files_only=True)
        vision_config = full_config.vision_config
        vision_config._attn_implementation = "eager"
        with torch.device("meta"):
            cls.oracle = Qwen3_5VisionModel(vision_config)
        state = {
            name.removeprefix("model.visual."): tensor
            for name, tensor in cls.subject.weights.named_tensors()
        }
        cls.oracle.load_state_dict(state, strict=True, assign=True)
        rotary_dim = (vision_config.hidden_size // vision_config.num_heads) // 2
        cls.oracle.rotary_pos_emb.inv_freq = (
            1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        ).to(_DEVICE)
        cls.oracle.eval()
        cls.processor = AutoProcessor.from_pretrained(_CHECKPOINT, local_files_only=True)

    def assert_exact_oracle(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        expected_shape: tuple[int, int],
    ) -> None:
        pixels = pixel_values.to(_DEVICE)
        grid = grid_thw.to(_DEVICE)
        with torch.inference_mode(), ReferenceExecutionContext():
            actual = self.subject.forward(pixels, grid)
            expected = self.oracle(hidden_states=pixels, grid_thw=grid).pooler_output

        self.assertEqual(tuple(actual.shape), expected_shape)
        self.assertEqual(tuple(expected.shape), expected_shape)
        self.assertIs(actual.dtype, torch.bfloat16)
        self.assertIs(expected.dtype, torch.bfloat16)
        if not torch.equal(actual, expected):
            difference = (actual.float() - expected.float()).abs()
            self.fail(
                "vision eager output differs from Transformers 5.14.1 eager oracle: "
                f"elements={torch.count_nonzero(actual != expected).item()}, "
                f"max_abs={difference.max().item()}"
            )

    def test_32_by_32_zero_image_matches_eager_oracle_exactly(self) -> None:
        import numpy as np
        from PIL import Image

        image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        inputs = self.processor(text=["vision probe"], images=[image], return_tensors="pt")
        self.assertEqual(tuple(inputs.pixel_values.shape), (256, 1536))
        self.assertEqual(inputs.image_grid_thw.tolist(), [[1, 16, 16]])
        self.assert_exact_oracle(
            inputs.pixel_values,
            inputs.image_grid_thw,
            expected_shape=(64, 5120),
        )

    def test_four_frame_zero_video_matches_eager_oracle_exactly(self) -> None:
        import numpy as np

        video = np.zeros((4, 32, 32, 3), dtype=np.uint8)
        inputs = self.processor(text=["vision probe"], videos=[video], return_tensors="pt")
        self.assertEqual(tuple(inputs.pixel_values_videos.shape), (8, 1536))
        self.assertEqual(inputs.video_grid_thw.tolist(), [[2, 2, 2]])
        self.assert_exact_oracle(
            inputs.pixel_values_videos,
            inputs.video_grid_thw,
            expected_shape=(2, 5120),
        )


if __name__ == "__main__":
    unittest.main()
