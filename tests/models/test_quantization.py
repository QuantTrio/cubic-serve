import unittest

from cubic_serve.models.quantization import (
    ActivationFormat,
    QuantizationConfig,
    QuantizationError,
    WeightFormat,
)


def fp8_config():
    return {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "fp8": {
                "format": "float-quantized",
                "targets": ["model.embed_tokens", "lm_head"],
                "weights": {
                    "num_bits": 8,
                    "strategy": "channel",
                    "type": "float",
                    "dynamic": False,
                    "symmetric": True,
                },
                "input_activations": {
                    "num_bits": 8,
                    "strategy": "tensor",
                    "type": "float",
                    "dynamic": True,
                    "symmetric": True,
                },
            }
        },
    }


class QuantizationConfigTests(unittest.TestCase):
    def test_regex_union_only_rejects_impossible_rules(self):
        from unittest.mock import patch

        from cubic_serve.models.quantization import EncodingRule, _target_union

        base = QuantizationConfig.from_model_config({"quantization_config": fp8_config()})
        encoding = base.rules[0].encoding
        targets = (
            ("re:^model[.]layers[.][0-3][.]", "re:.*proj$"),
            (r"re:^(model)[.]\1$",),
            ("re:(?i)^MODEL[.]X$",),
            ("Linear", "model.layers"),
            ("re:^x$", "re:^xy$"),
        )
        for group in targets:
            config = QuantizationConfig(
                base.method, (EncodingRule("test", group, encoding),),
                ("re:^ignored[.]", "re:^skip$"), base.source_scope, base.digest,
                target_subtrees=True,
            )
            for name in (
                "model.layers.2.proj", "model.layers.2.other", "foo.proj", "model.model",
                "MODEL.X", "ignored.foo", "skip", "x", "xy", "re:^x$", "re:^x$.child",
            ):
                for aliases in ((), ("model.layers.1.proj",)):
                    actual = config.resolve(name, aliases=aliases)
                    with patch("cubic_serve.models.quantization._may_match", return_value=True):
                        expected = config.resolve(name, aliases=aliases)
                    self.assertEqual(actual, expected)
        self.assertIsNone(_target_union(targets[1]))
        self.assertIsNone(_target_union(targets[2]))

    def test_nested_config_and_distinct_consumers(self):
        config = QuantizationConfig.from_model_config(
            {"text_config": {"quantization_config": fp8_config()}}
        )
        self.assertEqual(config.source_scope, "text_config")
        embedding = config.resolve(
            "model.language_model.embed_tokens",
            aliases=("model.embed_tokens",),
            operation="embedding",
        )
        head = config.resolve("lm_head")
        self.assertEqual(embedding.format, WeightFormat.FP8)
        self.assertEqual(embedding.activation, ActivationFormat.IDENTITY)
        self.assertEqual(head.activation, ActivationFormat.FP8_TOKEN)

    def test_conflicting_scopes_fail_closed(self):
        with self.assertRaises(QuantizationError):
            QuantizationConfig.from_model_config(
                {"quantization_config": fp8_config(), "text_config": {"quantization_config": {}}}
            )

    def test_awq_ignore_does_not_override_other_experts(self):
        config = QuantizationConfig.from_model_config(
            {
                "quantization_config": {
                    "quant_method": "awq",
                    "bits": 4,
                    "group_size": 128,
                    "version": "gemm",
                    "zero_point": True,
                    "modules_to_not_convert": ["model.layers.0.", "shared_expert"],
                }
            }
        )
        self.assertIsNone(config.resolve("model.layers.0.mlp.experts.0.gate_proj"))
        self.assertIsNone(config.resolve("model.layers.1.mlp.shared_expert.gate_proj"))
        self.assertEqual(
            config.resolve("model.layers.1.mlp.experts.0.gate_proj").format, WeightFormat.AWQ
        )

    def test_nvfp4_expert_subtree(self):
        group = {
            "format": "nvfp4-pack-quantized",
            "targets": ["model.layers.1.mlp.experts"],
            "weights": {
                "num_bits": 4,
                "group_size": 16,
                "strategy": "tensor_group",
                "scale_dtype": "torch.float8_e4m3fn",
                "dynamic": False,
                "symmetric": True,
            },
            "input_activations": {
                "num_bits": 4,
                "group_size": 16,
                "strategy": "tensor_group",
                "dynamic": "local",
                "symmetric": True,
            },
        }
        config = QuantizationConfig.from_model_config(
            {
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": {"nvfp4": group},
                }
            }
        )
        encoded = config.resolve("model.layers.1.mlp.experts.255.down_proj")
        self.assertEqual(encoded.activation, ActivationFormat.NVFP4_LOCAL)
        self.assertIsNone(config.resolve("model.layers.11.mlp.experts.0.down_proj"))

    def test_missing_quantization_remains_bf16(self):
        self.assertIsNone(QuantizationConfig.from_model_config({"model_type": "qwen3_5"}))

    def test_invalid_regex_rejected_at_admission(self):
        raw = fp8_config()
        raw["config_groups"]["fp8"]["targets"] = ["re:["]
        with self.assertRaises(QuantizationError):
            QuantizationConfig.from_model_config({"quantization_config": raw})


if __name__ == "__main__":
    unittest.main()
