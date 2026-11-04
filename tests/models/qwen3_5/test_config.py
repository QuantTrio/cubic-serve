from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path
from typing import cast

from cubic_serve.models.qwen3_5 import (
    AttentionKind,
    LayerStateKind,
    Qwen3_5ConfigError,
    parse_qwen3_5_config,
    parse_qwen3_5_config_json,
)

_LOCAL_CONFIG = Path("/pubdata/llm_models/Qwen/Qwen3.8-27B/config.json")


def _schedule(count: int) -> list[str]:
    return [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention" for index in range(count)
    ]


def _fixture() -> dict[str, object]:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 248056,
        "language_model_only": False,
        "model_type": "qwen3_5",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "dtype": "bfloat16",
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "layer_types": _schedule(64),
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "num_attention_heads": 24,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "output_gate_type": "swish",
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10_000_000,
                "rope_type": "default",
            },
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
            "model_type": "qwen3_5",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 5120,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053,
    }


class Qwen3_5ConfigTests(unittest.TestCase):
    def test_dense_profile_keeps_release_and_model_type_distinct(self) -> None:
        config = parse_qwen3_5_config(_fixture())
        plan = config.architecture_plan()

        self.assertEqual(config.identity.release_family, "qwen3_5")
        self.assertEqual(config.identity.release_profile, "qwen3_5")
        self.assertEqual(config.identity.model_type, "qwen3_5")
        self.assertEqual(config.identity.graph_family, "qwen3_5")
        self.assertEqual(len(plan.decoder_layers), 64)
        self.assertEqual(
            Counter(layer.attention for layer in plan.decoder_layers),
            {AttentionKind.GATED_DELTA_NET: 48, AttentionKind.FULL_ATTENTION: 16},
        )
        state_counts = Counter(
            requirement for layer in plan.decoder_layers for requirement in layer.state_requirements
        )
        self.assertEqual(state_counts[LayerStateKind.GDN_CONVOLUTION], 48)
        self.assertEqual(state_counts[LayerStateKind.GDN_RECURRENT_MATRIX], 48)
        self.assertEqual(state_counts[LayerStateKind.ATTENTION_KV], 16)
        state_ids = tuple(
            state_id for layer in plan.decoder_layers for state_id in layer.state_node_ids
        )
        self.assertEqual(len(state_ids), len(set(state_ids)))
        self.assertEqual(len(plan.vision_layers), 27)
        self.assertEqual(plan.mtp_layers[0].attention, AttentionKind.FULL_ATTENTION)
        self.assertEqual(
            plan.mtp_layers[0].state_requirements,
            (LayerStateKind.ATTENTION_KV,),
        )

    def test_model_type_moe_cannot_enter_dense_package(self) -> None:
        raw = _fixture()
        raw["model_type"] = "qwen3_5_moe"
        with self.assertRaisesRegex(Qwen3_5ConfigError, "model_type"):
            parse_qwen3_5_config(raw)

    def test_unknown_semantic_switch_and_bad_gdn_group_fail(self) -> None:
        raw = _fixture()
        text = cast(dict[str, object], raw["text_config"])
        text["future_attention_switch"] = True
        with self.assertRaisesRegex(Qwen3_5ConfigError, "unknown"):
            parse_qwen3_5_config(raw)

        raw = _fixture()
        text = cast(dict[str, object], raw["text_config"])
        text["linear_num_value_heads"] = 50
        with self.assertRaisesRegex(Qwen3_5ConfigError, "divisible"):
            parse_qwen3_5_config(raw)

    def test_architecture_schedule_rope_and_modality_fail_closed(self) -> None:
        cases: list[dict[str, object]] = []

        raw = _fixture()
        raw["architectures"] = ["Qwen3_5Model"]
        cases.append(raw)

        raw = _fixture()
        text = cast(dict[str, object], raw["text_config"])
        layers = cast(list[str], text["layer_types"])
        layers[0] = "full_attention"
        cases.append(raw)

        raw = _fixture()
        text = cast(dict[str, object], raw["text_config"])
        rope = cast(dict[str, object], text["rope_parameters"])
        rope["mrope_section"] = [10, 10, 10]
        cases.append(raw)

        raw = _fixture()
        text = cast(dict[str, object], raw["text_config"])
        rope = cast(dict[str, object], text["rope_parameters"])
        rope["partial_rotary_factor"] = 0.5
        cases.append(raw)

        raw = _fixture()
        raw["video_token_id"] = raw["image_token_id"]
        cases.append(raw)

        raw = _fixture()
        raw["image_token_id"] = 248320
        cases.append(raw)

        for raw in cases:
            with self.subTest(case=len(cases)), self.assertRaises(Qwen3_5ConfigError):
                parse_qwen3_5_config(raw)

    def test_config_accepts_independent_geometry_and_optional_mtp(self) -> None:
        mutations: tuple[tuple[str, object], ...] = (
            ("linear_value_head_dim", 64),
            ("linear_key_head_dim", 64),
            ("output_gate_type", "sigmoid"),
            ("mtp_num_hidden_layers", 0),
        )
        for key, value in mutations:
            raw = _fixture()
            cast(dict[str, object], raw["text_config"])[key] = value
            with self.subTest(key=key):
                parse_qwen3_5_config(raw)

        raw = _fixture()
        cast(dict[str, object], raw["vision_config"])["patch_size"] = 14
        parse_qwen3_5_config(raw)

        raw = _fixture()
        text = cast(dict[str, object], raw["text_config"])
        text["full_attention_interval"] = 8
        text["layer_types"] = [
            "full_attention" if (index + 1) % 8 == 0 else "linear_attention" for index in range(64)
        ]
        parse_qwen3_5_config(raw)

    def test_nested_tie_and_strict_json_fail_closed(self) -> None:
        raw = _fixture()
        cast(dict[str, object], raw["text_config"])["tie_word_embeddings"] = True
        with self.assertRaisesRegex(Qwen3_5ConfigError, "contradicts"):
            parse_qwen3_5_config(raw)

        documents = (
            "[]",
            "{",
            '{"model_type":"qwen3_5","model_type":"qwen3_5"}',
            '{"model_type":"qwen3_5","value":NaN}',
        )
        for document in documents:
            with self.subTest(document=document), self.assertRaises(Qwen3_5ConfigError):
                parse_qwen3_5_config_json(document)

    @unittest.skipUnless(_LOCAL_CONFIG.is_file(), "local qwen3_5 config is unavailable")
    def test_real_qwen3_8_27b_config(self) -> None:
        config = parse_qwen3_5_config_json(_LOCAL_CONFIG.read_bytes())
        text = config.graph.text
        vision = config.graph.vision
        self.assertEqual(config.identity.model_type, "qwen3_5")
        self.assertEqual(config.identity.release_family, "qwen3_5")
        self.assertEqual(
            (
                text.vocabulary_size,
                text.hidden_size,
                text.layer_count,
                text.full_attention_interval,
                text.dense_intermediate_size,
                text.full_attention.query_heads,
                text.full_attention.key_value_heads,
                text.full_attention.head_dim,
                text.gated_delta_net.key_heads,
                text.gated_delta_net.value_heads,
                text.gated_delta_net.key_head_dim,
                text.gated_delta_net.value_head_dim,
                text.mtp.hidden_layer_count,
            ),
            (248320, 5120, 64, 4, 17408, 24, 4, 256, 16, 48, 128, 128, 1),
        )
        self.assertEqual(text.mrope.section, (11, 11, 10))
        self.assertEqual(
            (
                vision.layer_count,
                vision.hidden_size,
                vision.intermediate_size,
                vision.output_hidden_size,
                vision.patch_size,
                vision.temporal_patch_size,
            ),
            (27, 1152, 4304, 5120, 16, 2),
        )


if __name__ == "__main__":
    unittest.main()
