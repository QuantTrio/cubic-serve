from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path
from typing import cast

from cubic_serve.models.qwen4_exp import (
    AttentionKind,
    FullAttentionKind,
    LayerStateKind,
    MTPStatus,
    PrefixCacheBoundary,
    Qwen4ExpConfigError,
    StateReplication,
    parse_qwen4_exp_config,
    parse_qwen4_exp_config_json,
)

_LOCAL_CONFIG = Path("/pubdata/llm_models/Qwen/Qwen3.8-Flash-Next/config.json")


def _fixture() -> dict[str, object]:
    layers = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention" for index in range(48)
    ]
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "image_token_id": 248056,
        "language_model_only": False,
        "model_type": "qwen4_exp",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "dtype": "bfloat16",
            "full_attention_interval": 4,
            "hc_count": 4,
            "hc_lowrank": 320,
            "head_dim": 256,
            "heads_per_ngram": 8,
            "hidden_act": "silu",
            "hidden_size": 2560,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 128,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 4,
            "layer_types": layers,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "make_ngram_vocab_size_divisible_by": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "model_type": "qwen4_exp_text",
            "moe_intermediate_size": 640,
            "mtp": {
                "hybrid": True,
                "layer_types": ["full_attention"],
                "mtp_use_hidden_state_from_layer": None,
                "num_hidden_layers": 1,
                "rope_theta": 10_000_000,
            },
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "ngram_size": 3,
            "ngram_vocab_size_base": 20_000_000,
            "num_attention_heads": 24,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "num_hidden_layers": 48,
            "num_key_value_heads": 2,
            "output_gate_type": "sigmoid",
            "partial_rotary_factor": 0.25,
            "ple_conv_kernel_size": 4,
            "ple_embed_dim": 2560,
            "ple_layer_ids": [2],
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10_000_000,
                "rope_type": "default",
            },
            "shared_expert_intermediate_size": 640,
            "split_ngram_parts": 128,
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
            "model_type": "qwen4_exp",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 2560,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053,
    }


class Qwen4ExpConfigTests(unittest.TestCase):
    def test_flash_next_owns_qwen4_exp_graph(self) -> None:
        config = parse_qwen4_exp_config(_fixture())
        plan = config.architecture_plan()
        self.assertEqual(config.identity.release_family, "qwen4_exp")
        self.assertEqual(config.identity.release_profile, "qwen4_exp")
        self.assertEqual(config.identity.model_type, "qwen4_exp")
        self.assertEqual(config.identity.graph_family, "qwen4_exp")
        self.assertEqual(plan.final_mixer_node_id, "text.final_hyperconnection_mixer")
        self.assertEqual(len(plan.decoder_layers), 48)
        self.assertEqual(
            Counter(layer.attention for layer in plan.decoder_layers),
            {AttentionKind.GATED_DELTA_NET: 36, AttentionKind.FULL_ATTENTION: 12},
        )
        self.assertTrue(plan.decoder_layers[1].uses_ple)
        self.assertEqual(config.text.hyperconnection.stream_count, 4)
        self.assertEqual(config.text.moe.expert_count, 512)
        self.assertEqual(config.text.moe.experts_per_token, 10)
        self.assertEqual(config.text.mtp.status, MTPStatus.METADATA_ONLY)
        self.assertEqual(config.text.mtp.layer_schedule, (AttentionKind.FULL_ATTENTION,))
        state_counts = Counter(
            requirement for layer in plan.decoder_layers for requirement in layer.state_requirements
        )
        self.assertEqual(state_counts[LayerStateKind.GDN_CONVOLUTION], 36)
        self.assertEqual(state_counts[LayerStateKind.GDN_RECURRENT_MATRIX], 36)
        self.assertEqual(state_counts[LayerStateKind.ATTENTION_KV], 12)
        self.assertEqual(state_counts[LayerStateKind.QSA_RAW_KEY_RING], 12)
        self.assertEqual(state_counts[LayerStateKind.QSA_COMPRESSED_KEY_CACHE], 12)
        self.assertEqual(state_counts[LayerStateKind.PLE_CONVOLUTION], 1)
        self.assertEqual(state_counts[LayerStateKind.PLE_NGRAM_TOKEN_CONTEXT], 1)
        self.assertEqual(state_counts[LayerStateKind.QSA_POSITION_RING], 12)
        self.assertEqual(sum(state_counts.values()), 122)
        self.assertTrue(
            all(
                layer.full_attention_kind is FullAttentionKind.QSA
                for layer in plan.decoder_layers
                if layer.attention is AttentionKind.FULL_ATTENTION
            )
        )
        self.assertEqual(
            plan.mtp_layers[0].state_requirements,
            (
                LayerStateKind.ATTENTION_KV,
                LayerStateKind.QSA_RAW_KEY_RING,
                LayerStateKind.QSA_POSITION_RING,
                LayerStateKind.QSA_COMPRESSED_KEY_CACHE,
            ),
        )
        self.assertEqual(plan.mtp_layers[0].full_attention_kind, FullAttentionKind.QSA)
        self.assertEqual(plan.qsa_state.raw_key_width, 128)
        self.assertEqual(plan.qsa_state.raw_storage_width, 140)
        self.assertEqual(plan.qsa_state.raw_base_capacity, 4)
        self.assertEqual(plan.qsa_state.compressed_rows_divisor, 4)
        self.assertEqual(
            plan.qsa_state.side_cache_replication,
            StateReplication.TP_PARTITIONED,
        )
        self.assertEqual(
            plan.qsa_state.prefix_cache_boundary,
            PrefixCacheBoundary.COMPRESSION_GROUP_ALIGNED_ONLY,
        )
        self.assertEqual(plan.ple_state.convolution_channels, 10240)
        self.assertEqual(plan.ple_state.convolution_context_width, 9)
        self.assertEqual(plan.ple_state.token_context_length, 2)

    def test_other_model_type_cannot_enter_qwen4_exp_package(self) -> None:
        raw = _fixture()
        raw["model_type"] = "qwen3_5"
        with self.assertRaisesRegex(Qwen4ExpConfigError, "model_type"):
            parse_qwen4_exp_config(raw)

    def test_nested_mtp_and_qsa_contradictions_fail_closed(self) -> None:
        raw = _fixture()
        text = cast(dict[str, object], raw["text_config"])
        text["mtp_num_hidden_layers"] = 2
        with self.assertRaisesRegex(Qwen4ExpConfigError, "layer count"):
            parse_qwen4_exp_config(raw)

        raw = _fixture()
        text = cast(dict[str, object], raw["text_config"])
        mtp = cast(dict[str, object], text["mtp"])
        mtp["mtp_use_hidden_state_from_layer"] = 48
        with self.assertRaisesRegex(Qwen4ExpConfigError, "text layer"):
            parse_qwen4_exp_config(raw)

        raw = _fixture()
        text = cast(dict[str, object], raw["text_config"])
        text["indexer_kv_heads"] = 2
        with self.assertRaisesRegex(Qwen4ExpConfigError, "indexer_kv_heads"):
            parse_qwen4_exp_config(raw)

    def test_architecture_schedule_rope_modality_and_unknown_switch_fail(self) -> None:
        cases: list[dict[str, object]] = []
        raw = _fixture()
        raw["architectures"] = ["Qwen4ExpModel"]
        cases.append(raw)

        raw = _fixture()
        text = cast(dict[str, object], raw["text_config"])
        cast(list[str], text["layer_types"])[0] = "full_attention"
        cases.append(raw)

        raw = _fixture()
        text = cast(dict[str, object], raw["text_config"])
        cast(dict[str, object], text["rope_parameters"])["mrope_section"] = [10, 10, 10]
        cases.append(raw)

        raw = _fixture()
        text = cast(dict[str, object], raw["text_config"])
        cast(dict[str, object], text["rope_parameters"])["partial_rotary_factor"] = 0.5
        cases.append(raw)

        raw = _fixture()
        raw["image_token_id"] = raw["video_token_id"]
        cases.append(raw)

        raw = _fixture()
        raw["video_token_id"] = 248320
        cases.append(raw)

        raw = _fixture()
        cast(dict[str, object], raw["text_config"])["future_qsa_switch"] = True
        cases.append(raw)

        for raw in cases:
            with self.subTest(keys=tuple(raw)), self.assertRaises(Qwen4ExpConfigError):
                parse_qwen4_exp_config(raw)

    def test_independent_hc_qsa_ple_moe_mtp_and_vision_geometry(self) -> None:
        for key, value in (
            ("hc_lowrank", 256),
            ("indexer_n_heads", 8),
            ("indexer_budget", 8192),
            ("ple_conv_kernel_size", 3),
            ("moe_intermediate_size", 512),
            ("linear_value_head_dim", 64),
        ):
            raw = _fixture()
            cast(dict[str, object], raw["text_config"])[key] = value
            with self.subTest(key=key):
                parse_qwen4_exp_config(raw)

        raw = _fixture()
        text = cast(dict[str, object], raw["text_config"])
        cast(dict[str, object], text["mtp"])["hybrid"] = False
        parse_qwen4_exp_config(raw)

        raw = _fixture()
        text = cast(dict[str, object], raw["text_config"])
        cast(dict[str, object], text["mtp"])["layer_types"] = ["linear_attention"]
        parse_qwen4_exp_config(raw)

        raw = _fixture()
        cast(dict[str, object], raw["vision_config"])["patch_size"] = 14
        parse_qwen4_exp_config(raw)

    def test_nested_tie_and_strict_json_fail_closed(self) -> None:
        raw = _fixture()
        cast(dict[str, object], raw["text_config"])["tie_word_embeddings"] = True
        with self.assertRaisesRegex(Qwen4ExpConfigError, "contradicts"):
            parse_qwen4_exp_config(raw)

        for document in (
            "[]",
            "{",
            '{"model_type":"qwen4_exp","model_type":"qwen4_exp"}',
            '{"model_type":"qwen4_exp","value":Infinity}',
        ):
            with self.subTest(document=document), self.assertRaises(Qwen4ExpConfigError):
                parse_qwen4_exp_config_json(document)

    @unittest.skipUnless(_LOCAL_CONFIG.is_file(), "local qwen4_exp config is unavailable")
    def test_real_flash_next_config(self) -> None:
        config = parse_qwen4_exp_config_json(_LOCAL_CONFIG.read_bytes())
        text = config.text
        self.assertEqual(config.identity.model_type, "qwen4_exp")
        self.assertEqual(config.identity.release_profile, "qwen4_exp")
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
                text.hyperconnection.stream_count,
                text.hyperconnection.low_rank,
                text.qsa.query_heads,
                text.qsa.key_value_heads,
                text.qsa.head_dim,
                text.qsa.token_budget,
                text.qsa.compression_ratio,
            ),
            (248320, 2560, 48, 24, 2, 16, 48, 512, 10, 4, 320, 4, 1, 128, 2048, 4),
        )
        self.assertEqual(text.ple.layer_ids, (2,))
        self.assertEqual(text.mtp.layer_schedule, (AttentionKind.FULL_ATTENTION,))
        self.assertEqual(text.mtp.hidden_state_source_layer, None)
        self.assertEqual(
            (
                config.vision.layer_count,
                config.vision.hidden_size,
                config.vision.intermediate_size,
                config.vision.output_hidden_size,
            ),
            (27, 1152, 4304, 2560),
        )


if __name__ == "__main__":
    unittest.main()
