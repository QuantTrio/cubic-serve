from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from cubic_serve.models.qwen3_5 import parse_qwen3_5_config, parse_qwen3_5_config_json
from cubic_serve.models.qwen3_5.weights import (
    _expected_headers,
    plan_qwen3_5_checkpoint_weights,
    plan_qwen3_5_weight_headers,
    qwen3_5_mtp_weight_policy,
    qwen3_5_text_weight_policy,
)
from cubic_serve.models.weight_schema import (
    HeaderEvidenceKind,
    TensorHeader,
    WeightSchemaCoverage,
    WeightSchemaError,
)
from tests.models._checkpoint_headers import read_index_headers
from tests.models.qwen3_5.test_config import _fixture

_LOCAL_ROOT = Path("/pubdata/llm_models/Qwen/Qwen3.8-27B")


class _FakeInventory:
    tensors: tuple[object, ...] = ()
    expected_names: tuple[str, ...] = ()
    inventory_digest = "untrusted inventory"


def _headers() -> list[TensorHeader]:
    config = parse_qwen3_5_config(_fixture())
    return [
        TensorHeader(name, contract.dtype, contract.shape)
        for name, contract in _expected_headers(config).items()
    ]


class Qwen3_5WeightSchemaTests(unittest.TestCase):
    def test_public_checkpoint_api_rejects_duck_typed_attestation(self) -> None:
        with self.assertRaisesRegex(WeightSchemaError, "core-issued CheckpointInventory"):
            plan_qwen3_5_checkpoint_weights(  # type: ignore[arg-type]
                parse_qwen3_5_config(_fixture()),
                _FakeInventory(),
            )

    def test_dense_checkpoint_is_exact_and_mtp_is_a_separate_extension(self) -> None:
        plan = plan_qwen3_5_weight_headers(parse_qwen3_5_config(_fixture()), _headers())

        self.assertEqual(plan.coverage, WeightSchemaCoverage.EXACT_CHECKPOINT_HEADERS)
        self.assertEqual(plan.header_evidence, HeaderEvidenceKind.HEADER_RECORDS_ONLY)
        self.assertIsNone(plan.inventory_digest)
        self.assertEqual(len(plan.checkpoint_names), 1199)
        self.assertEqual(len(plan.base_language_names), 850)
        self.assertEqual(len(plan.vision_names), 333)
        self.assertEqual(len(plan.lm_head_names), 1)
        self.assertEqual(len(plan.base_executable_names), 1184)
        self.assertEqual(len(plan.mtp_extension_names), 15)
        self.assertEqual(plan.base_executable_policy.expected_names, plan.base_executable_names)
        text_policy = qwen3_5_text_weight_policy(plan)
        self.assertEqual(len(text_policy.expected_names), 851)
        self.assertEqual(
            text_policy.expected_names,
            tuple(sorted((*plan.base_language_names, *plan.lm_head_names))),
        )
        mtp_policy = qwen3_5_mtp_weight_policy(plan)
        self.assertEqual(len(mtp_policy.expected_names), 866)
        self.assertEqual(
            mtp_policy.expected_names,
            tuple(
                sorted((*plan.base_language_names, *plan.lm_head_names, *plan.mtp_extension_names))
            ),
        )
        self.assertEqual(plan.full_checkpoint_policy.expected_names, plan.checkpoint_names)
        bound = tuple(name for binding in plan.node_bindings for name in binding.tensor_names)
        self.assertEqual(set(bound), set(plan.checkpoint_names))
        self.assertEqual(len(bound), len(set(bound)))
        with self.assertRaises(TypeError):
            replace(plan, checkpoint_header_digest="b2s256:" + "0" * 64)

    def test_missing_unknown_dtype_shape_and_duplicate_fail_closed(self) -> None:
        config = parse_qwen3_5_config(_fixture())
        headers = _headers()
        cases = (
            headers[1:],
            [*headers, TensorHeader("model.language_model.layers.64.x", "BF16", (1,))],
            [replace(headers[0], dtype="F32"), *headers[1:]],
            [replace(headers[0], shape=(1,)), *headers[1:]],
            [*headers, headers[0]],
        )
        for candidate in cases:
            with self.subTest(size=len(candidate)), self.assertRaises(WeightSchemaError):
                plan_qwen3_5_weight_headers(config, candidate)

    @unittest.skipUnless(
        (_LOCAL_ROOT / "model.safetensors.index.json").is_file(),
        "local qwen3_5 checkpoint headers are unavailable",
    )
    def test_real_27b_headers_without_tensor_payload_reads(self) -> None:
        config = parse_qwen3_5_config_json((_LOCAL_ROOT / "config.json").read_bytes())
        plan = plan_qwen3_5_weight_headers(config, read_index_headers(_LOCAL_ROOT))
        self.assertEqual(
            (
                len(plan.checkpoint_names),
                len(plan.base_language_names),
                len(plan.vision_names),
                len(plan.lm_head_names),
                len(plan.mtp_extension_names),
            ),
            (1199, 850, 333, 1, 15),
        )


if __name__ == "__main__":
    unittest.main()


def test_weight_mapping_follows_config_geometry_without_known_repository():
    for layers, hidden, _experts, mtp in ((8, 768, 16, 0), (12, 1536, 32, 1)):
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
        config = parse_qwen3_5_config(raw)
        expected = _expected_headers(config)
        headers = [TensorHeader(name, item.dtype, item.shape) for name, item in expected.items()]
        plan = plan_qwen3_5_weight_headers(config, headers)
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
            plan_qwen3_5_weight_headers(config, broken)
