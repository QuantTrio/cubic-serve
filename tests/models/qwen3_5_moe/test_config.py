from __future__ import annotations

import unittest
from pathlib import Path
from typing import cast

from cubic_serve.models.qwen3_5 import AttentionKind, FeedForwardKind, LayerStateKind
from cubic_serve.models.qwen3_5_moe import (
    Qwen3_5ConfigError,
    parse_qwen3_5_moe_config,
    parse_qwen3_5_moe_config_json,
)

_LOCAL_CONFIG = Path("/pubdata/llm_models/Qwen/Qwen3.5-122B-A10B/config.json")


def _fixture() -> dict[str, object]:
    layers = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention" for index in range(48)
    ]
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "image_token_id": 248056,
        "model_type": "qwen3_5_moe",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "dtype": "bfloat16",
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 3072,
            "layer_types": layers,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 64,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "mlp_only_layers": [],
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 1024,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "num_attention_heads": 32,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 48,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10_000_000,
                "rope_type": "default",
            },
            "shared_expert_intermediate_size": 1024,
            "use_cache": True,
            "vocab_size": 248320,
        },
        "tie_word_embeddings": False,
        "video_token_id": 248057,
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 27,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "intermediate_size": 4304,
            "model_type": "qwen3_5_moe",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 3072,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053,
    }


class Qwen3_5MoeConfigTests(unittest.TestCase):
    def test_moe_loader_key_is_independent_but_reuses_hybrid_graph(self) -> None:
        config = parse_qwen3_5_moe_config(_fixture())
        plan = config.architecture_plan()
        self.assertEqual(config.identity.release_family, "qwen3_5")
        self.assertEqual(config.identity.model_type, "qwen3_5_moe")
        self.assertEqual(config.identity.graph_family, "qwen3_5")
        self.assertEqual(config.graph.text.gated_delta_net.value_heads, 64)
        assert config.graph.text.moe is not None
        self.assertEqual(config.graph.text.moe.expert_count, 256)
        self.assertEqual(config.graph.text.moe.experts_per_token, 8)
        self.assertEqual(
            sum(layer.attention is AttentionKind.GATED_DELTA_NET for layer in plan.decoder_layers),
            36,
        )
        self.assertEqual(
            {layer.feed_forward for layer in plan.decoder_layers},
            {FeedForwardKind.MIXTURE_OF_EXPERTS},
        )
        self.assertEqual(plan.mtp_layers[0].attention, AttentionKind.FULL_ATTENTION)
        self.assertEqual(plan.mtp_layers[0].state_requirements, (LayerStateKind.ATTENTION_KV,))

    def test_architecture_validation_and_independent_geometry(self) -> None:
        raw = _fixture()
        raw["architectures"] = ["Qwen3_5MoeModel"]
        with self.assertRaises(Qwen3_5ConfigError):
            parse_qwen3_5_moe_config(raw)

        for key, value in (
            ("moe_intermediate_size", 640),
            ("shared_expert_intermediate_size", 640),
            ("num_experts", 128),
            ("mtp_num_hidden_layers", 0),
            ("linear_key_head_dim", 64),
        ):
            raw = _fixture()
            cast(dict[str, object], raw["text_config"])[key] = value
            with self.subTest(key=key):
                parse_qwen3_5_moe_config(raw)

        raw = _fixture()
        cast(dict[str, object], raw["vision_config"])["spatial_merge_size"] = 4
        parse_qwen3_5_moe_config(raw)

    def test_bad_schedule_modality_and_nested_tie_fail_closed(self) -> None:
        raw = _fixture()
        text = cast(dict[str, object], raw["text_config"])
        cast(list[str], text["layer_types"])[0] = "full_attention"
        with self.assertRaisesRegex(Qwen3_5ConfigError, "layer_types"):
            parse_qwen3_5_moe_config(raw)

        raw = _fixture()
        raw["vision_end_token_id"] = raw["vision_start_token_id"]
        with self.assertRaisesRegex(Qwen3_5ConfigError, "sentinel"):
            parse_qwen3_5_moe_config(raw)

        raw = _fixture()
        cast(dict[str, object], raw["text_config"])["tie_word_embeddings"] = True
        with self.assertRaisesRegex(Qwen3_5ConfigError, "contradicts"):
            parse_qwen3_5_moe_config(raw)

    @unittest.skipUnless(_LOCAL_CONFIG.is_file(), "local qwen3_5_moe config is unavailable")
    def test_real_qwen3_5_122b_config(self) -> None:
        config = parse_qwen3_5_moe_config_json(_LOCAL_CONFIG.read_bytes())
        text = config.graph.text
        vision = config.graph.vision
        assert text.moe is not None
        self.assertEqual(config.identity.model_type, "qwen3_5_moe")
        self.assertEqual(config.identity.release_profile, "qwen3_5_moe")
        self.assertEqual(
            (
                text.vocabulary_size,
                text.hidden_size,
                text.layer_count,
                text.full_attention.query_heads,
                text.full_attention.key_value_heads,
                text.gated_delta_net.key_heads,
                text.gated_delta_net.value_heads,
                text.moe.expert_count,
                text.moe.experts_per_token,
                text.moe.expert_intermediate_size,
                text.moe.shared_expert_intermediate_size,
                text.mtp.hidden_layer_count,
            ),
            (248320, 3072, 48, 32, 2, 16, 64, 256, 8, 1024, 1024, 1),
        )
        self.assertEqual(
            (
                vision.layer_count,
                vision.hidden_size,
                vision.intermediate_size,
                vision.output_hidden_size,
            ),
            (27, 1152, 4304, 3072),
        )


if __name__ == "__main__":
    unittest.main()
