from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from cubic_serve.models.qwen3_5_moe import (
    parse_qwen3_5_moe_config,
    parse_qwen3_5_moe_config_json,
)
from cubic_serve.models.qwen3_5_moe.weights import (
    _expected_headers,
    plan_qwen3_5_moe_weight_headers,
)
from cubic_serve.models.weight_schema import TensorHeader, WeightSchemaCoverage, WeightSchemaError
from tests.models._checkpoint_headers import read_index_headers
from tests.models.qwen3_5_moe.test_config import _fixture

_LOCAL_ROOT = Path("/pubdata/llm_models/Qwen/Qwen3.5-122B-A10B")


def _headers() -> list[TensorHeader]:
    config = parse_qwen3_5_moe_config(_fixture())
    return [
        TensorHeader(name, contract.dtype, contract.shape)
        for name, contract in _expected_headers(config).items()
    ]


class Qwen3_5MoeWeightSchemaTests(unittest.TestCase):
    def test_closed_partitions_and_fp32_stability_names(self) -> None:
        plan = plan_qwen3_5_moe_weight_headers(parse_qwen3_5_moe_config(_fixture()), _headers())

        self.assertEqual(plan.coverage, WeightSchemaCoverage.EXACT_CHECKPOINT_HEADERS)
        self.assertEqual(
            (
                len(plan.checkpoint_names),
                len(plan.base_language_names),
                len(plan.vision_names),
                len(plan.lm_head_names),
                len(plan.mtp_extension_names),
            ),
            (1949, 830, 333, 1, 785),
        )
        self.assertEqual(len(plan.fp32_stability_names), 72)
        self.assertTrue(
            all(
                name.endswith((".A_log", ".linear_attn.norm.weight"))
                for name in plan.fp32_stability_names
            )
        )
        self.assertFalse(plan.int64_metadata_names)

    def test_mtp_expert_gap_and_stability_dtype_mismatch_fail_closed(self) -> None:
        config = parse_qwen3_5_moe_config(_fixture())
        headers = _headers()
        expert = next(
            header
            for header in headers
            if header.name == "mtp.layers.0.mlp.experts.255.down_proj.weight"
        )
        stability = next(header for header in headers if header.dtype == "F32")
        for candidate in (
            [header for header in headers if header is not expert],
            [
                replace(stability, dtype="BF16") if header is stability else header
                for header in headers
            ],
            [replace(expert, shape=(1,)) if header is expert else header for header in headers],
        ):
            with self.subTest(size=len(candidate)), self.assertRaises(WeightSchemaError):
                plan_qwen3_5_moe_weight_headers(config, candidate)

    @unittest.skipUnless(
        (_LOCAL_ROOT / "model.safetensors.index.json").is_file(),
        "local qwen3_5_moe checkpoint headers are unavailable",
    )
    def test_real_122b_headers_without_tensor_payload_reads(self) -> None:
        config = parse_qwen3_5_moe_config_json((_LOCAL_ROOT / "config.json").read_bytes())
        plan = plan_qwen3_5_moe_weight_headers(config, read_index_headers(_LOCAL_ROOT))
        self.assertEqual(len(plan.checkpoint_names), 1949)
        self.assertEqual(len(plan.fp32_stability_names), 72)


if __name__ == "__main__":
    unittest.main()


def test_weight_mapping_follows_config_geometry_without_known_repository():
    for layers, hidden, experts, mtp in ((8, 768, 16, 0), (12, 1536, 32, 1)):
        raw = _fixture()
        text = raw["text_config"]
        text["num_hidden_layers"] = layers
        text["linear_num_key_heads"] = 8
        text["linear_num_value_heads"] = 16
        text["linear_key_head_dim"] = 64
        text["linear_value_head_dim"] = 32
        text["linear_conv_kernel_dim"] = 3
        text["head_dim"] = 128
        text["rope_parameters"]["partial_rotary_factor"] = 0.5
        if "partial_rotary_factor" in text:
            text["partial_rotary_factor"] = 0.5
        text["hidden_size"] = hidden
        text["vocab_size"] += 128
        text["layer_types"] = [
            "full_attention"
            if (i + 1) % text["full_attention_interval"] == 0
            else "linear_attention"
            for i in range(layers)
        ]
        text["mtp_num_hidden_layers"] = mtp
        vision = raw["vision_config"]
        vision["depth"] = 2
        vision["out_hidden_size"] = hidden
        text["num_experts"] = experts
        config = parse_qwen3_5_moe_config(raw)
        expected = _expected_headers(config)
        headers = [TensorHeader(name, item.dtype, item.shape) for name, item in expected.items()]
        plan = plan_qwen3_5_moe_weight_headers(config, headers)
        embedding = next(h for h in headers if h.name == "model.language_model.embed_tokens.weight")
        assert embedding.shape == (text["vocab_size"], hidden)
        assert bool(plan.mtp_extension_names) == bool(mtp)
        assert all(
            int(n.split("layers.")[1].split(".")[0]) < layers
            for n in plan.base_language_names
            if ".layers." in n
        )
        broken = [
            TensorHeader(h.name, h.dtype, (h.shape[0] + 1, *h.shape[1:]))
            if h.name == embedding.name
            else h
            for h in headers
        ]
        with unittest.TestCase().assertRaises(WeightSchemaError):
            plan_qwen3_5_moe_weight_headers(config, broken)
