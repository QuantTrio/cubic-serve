from __future__ import annotations

import unittest
from dataclasses import replace

from cubic_serve.contracts.execution import ParallelTopology
from cubic_serve.models import MODEL_DEFINITION_API_VERSION
from cubic_serve.models.builtin import (
    BUILTIN_MODEL_TYPES,
    BuiltinModelConfigError,
    builtin_model_definition,
    builtin_text_config,
    parse_builtin_model_config,
    parse_builtin_model_config_json,
)
from cubic_serve.models.qwen3_5 import MODEL_DEFINITION as DENSE_DEFINITION
from cubic_serve.models.qwen3_5 import Qwen3_5Config, parse_qwen3_5_config
from cubic_serve.models.qwen3_5.weights import logical_quantized_weights as dense_weights
from cubic_serve.models.qwen3_5.weights import quantized_owner_slices as dense_slices
from cubic_serve.models.qwen3_5_moe import MODEL_DEFINITION as MOE_DEFINITION
from cubic_serve.models.qwen3_5_moe import Qwen3_5MoeConfig, parse_qwen3_5_moe_config
from cubic_serve.models.qwen3_5_moe.weights import logical_quantized_weights as moe_weights
from cubic_serve.models.qwen3_5_moe.weights import quantized_owner_slices as moe_slices
from cubic_serve.models.qwen4_exp import MODEL_DEFINITION as FLASH_DEFINITION
from cubic_serve.models.qwen4_exp import Qwen4ExpConfig
from cubic_serve.models.qwen4_exp.model import parse_qwen4_exp_config
from cubic_serve.models.qwen4_exp.weights import logical_quantized_weights, quantized_owner_slices
from tests.models.qwen3_5.test_config import _fixture as qwen3_5_fixture
from tests.models.qwen3_5_moe.test_config import _fixture as qwen3_5_moe_fixture
from tests.models.qwen4_exp.test_config import _fixture as qwen4_exp_fixture


class BuiltinModelConfigDispatchTests(unittest.TestCase):
    def test_explicit_definition_owns_checkpoint_and_draft_without_config_type_dispatch(self):
        from pathlib import Path
        from unittest.mock import Mock, patch

        from cubic_serve.models.builtin import (
            admit_model_checkpoint,
            model_draft_provider,
            resolve_model_definition,
        )
        from cubic_serve.models.quantized_checkpoint import CheckpointWeights

        config, planned, draft, admitted = object(), (), object(), object()
        weights = Mock(return_value=planned)
        provider = Mock(return_value=draft)
        definition = replace(DENSE_DEFINITION, logical_weights=weights, draft_provider=provider)
        with self.assertRaises(BuiltinModelConfigError):
            resolve_model_definition(config)
        self.assertIs(resolve_model_definition(config, definition=definition), definition)
        with patch.object(CheckpointWeights, "admit", return_value=admitted) as admit:
            self.assertIs(
                admit_model_checkpoint("/unused", model_config=config, definition=definition),
                admitted,
            )
            weights.assert_called_once_with(config)
            admit.assert_called_once_with(
                Path("/unused"), planned, verify_payloads=False, require_quantization=False,
            )
        self.assertIs(
            model_draft_provider(config, method="mtp", max_depth=3, definition=definition), draft,
        )
        provider.assert_called_once_with(config, method="mtp", max_depth=3)

    def test_builtin_model_list_is_exact_and_immutable(self) -> None:
        self.assertEqual(
            BUILTIN_MODEL_TYPES,
            ("qwen3_5", "qwen3_5_moe", "qwen4_exp"),
        )

    def test_exact_root_model_type_selects_one_parser(self) -> None:
        cases = (
            (qwen3_5_fixture(), Qwen3_5Config),
            (qwen3_5_moe_fixture(), Qwen3_5MoeConfig),
            (qwen4_exp_fixture(), Qwen4ExpConfig),
        )
        for raw, expected_type in cases:
            with self.subTest(model_type=raw["model_type"]):
                config = parse_builtin_model_config(raw)
                self.assertIsInstance(config, expected_type)
                self.assertEqual(builtin_model_definition(config).model_type, raw["model_type"])
                original = config.text if isinstance(config, Qwen4ExpConfig) else config.graph.text
                self.assertIs(config.text_config, original)
                self.assertIs(builtin_text_config(config), original)

    def test_marketing_names_and_fuzzy_aliases_are_not_loader_keys(self) -> None:
        for value in ("Qwen3.8-27B", "qwen3_8", "qwen", "qwen4-exp"):
            with self.subTest(value=value), self.assertRaises(BuiltinModelConfigError):
                parse_builtin_model_config({"model_type": value})

    def test_definition_preserves_original_semantic_functions(self) -> None:
        from cubic_serve.models.qwen3_5.multimodal import (
            Qwen3_5TextProcessor,
            QwenMultimodalProcessor,
        )
        from cubic_serve.models.qwen3_5.output import Qwen3_5OutputCodec
        for definition, parser, weights, slices in (
            (DENSE_DEFINITION, parse_qwen3_5_config, dense_weights, dense_slices),
            (MOE_DEFINITION, parse_qwen3_5_moe_config, moe_weights, moe_slices),
            (FLASH_DEFINITION, parse_qwen4_exp_config, logical_quantized_weights,
             quantized_owner_slices),
        ):
            with self.subTest(model_type=definition.model_type):
                self.assertEqual(definition.api_version, MODEL_DEFINITION_API_VERSION)
                self.assertIs(definition.parse_config, parser)
                self.assertIs(definition.logical_weights, weights)
                self.assertIs(definition.owner_slices, slices)
                self.assertEqual(
                    definition.frontend_factories(multimodal=False),
                    (Qwen3_5TextProcessor, Qwen3_5OutputCodec),
                )
                self.assertEqual(
                    definition.frontend_factories(multimodal=True),
                    (QwenMultimodalProcessor, Qwen3_5OutputCodec),
                )

    def test_derivative_overrides_only_its_changed_configuration(self) -> None:
        def parse_derived(raw):
            config = FLASH_DEFINITION.parse_config(raw)
            return replace(config, text=replace(config.text, max_position_embeddings=65536))

        derived = replace(FLASH_DEFINITION, parse_config=parse_derived)
        config = derived.parse_config(qwen4_exp_fixture())
        base_config = FLASH_DEFINITION.parse_config(qwen4_exp_fixture())
        self.assertEqual(config.text.max_position_embeddings, 65536)
        self.assertNotEqual(base_config.text.max_position_embeddings, 65536)
        self.assertIs(derived.logical_weights, FLASH_DEFINITION.logical_weights)
        self.assertIs(derived.owner_slices, FLASH_DEFINITION.owner_slices)
        self.assertIs(derived.text_input_processor, FLASH_DEFINITION.text_input_processor)
        self.assertIs(
            derived.multimodal_input_processor, FLASH_DEFINITION.multimodal_input_processor
        )
        self.assertIs(derived.output_codec, FLASH_DEFINITION.output_codec)
        logical = derived.logical_weights(config)
        topology = ParallelTopology.single_node(
            topology_id="definition.inherited.tp2.ep2", device_indices=(0, 1),
            tp_size=2, pp_size=1, ep_size=2,
        )
        for rank in range(2):
            self.assertEqual(
                derived.owner_slices(config, topology, rank, logical_weights=logical),
                FLASH_DEFINITION.owner_slices(
                    base_config, topology, rank,
                    logical_weights=FLASH_DEFINITION.logical_weights(base_config),
                ),
            )

    def test_invalid_definition_fails_before_model_preparation(self) -> None:
        for version in (2, True):
            with self.subTest(version=version), self.assertRaises(ValueError):
                replace(FLASH_DEFINITION, api_version=version)
        with self.assertRaises(TypeError):
            replace(FLASH_DEFINITION, owner_slices=None)
        for field in (
            "text_input_processor",
            "multimodal_input_processor",
            "output_codec",
            "compute_config",
        ):
            with self.subTest(field=field), self.assertRaises(TypeError):
                replace(FLASH_DEFINITION, **{field: object()})
        with self.assertRaises(ValueError):
            replace(FLASH_DEFINITION, output_codec=None).frontend_factories(multimodal=True)

    def test_draft_declaration_keeps_original_native_constructors(self) -> None:
        from cubic_serve.executor.draft import HiddenConditionedReference
        from cubic_serve.executor.mtp_runner import PackedMTPStage, build_flash_mtp_stage
        from cubic_serve.models.builtin import builtin_draft_provider
        from cubic_serve.models.qwen4_exp.model import Qwen4ExpMTPReference

        for raw, stage, reference, columns in (
            (qwen3_5_fixture(), PackedMTPStage, HiddenConditionedReference, ("mtp.fc.weight",)),
            (
                qwen4_exp_fixture(),
                build_flash_mtp_stage,
                Qwen4ExpMTPReference,
                ("mtp.fc_embedding.weight", "mtp.fc_hidden.weight"),
            ),
        ):
            config = parse_builtin_model_config(raw)
            for depth in (1, 2, 3):
                with self.subTest(model_type=raw["model_type"], depth=depth):
                    provider = builtin_draft_provider(config, method="mtp", max_depth=depth)
                    self.assertIs(provider.build_stage.func, stage)
                    self.assertIs(provider.build_stage.args[0], config)
                    self.assertEqual(provider.build_stage.keywords["max_depth"], depth)
                    self.assertIs(provider.build_reference.func, reference)
                    self.assertIs(provider.build_reference.keywords["spec"], provider.spec)
                    self.assertEqual(provider.reference_columns, columns)

    def test_declared_draft_override_is_not_a_builtin_model_branch(self) -> None:
        from unittest.mock import patch

        from cubic_serve.models.builtin import builtin_draft_provider

        config, wanted = object(), object()
        calls = []

        def factory(actual, *, method, max_depth):
            calls.append((actual, method, max_depth))
            return wanted

        derived = replace(DENSE_DEFINITION, model_type="repo_derivative", draft_provider=factory)
        with patch("cubic_serve.models.builtin.builtin_model_definition", return_value=derived):
            self.assertIs(builtin_draft_provider(config, method="repo_draft", max_depth=3), wanted)
        self.assertEqual(calls, [(config, "repo_draft", 3)])
        self.assertIs(
            DENSE_DEFINITION.draft_provider,
            replace(DENSE_DEFINITION, model_type="inherited").draft_provider,
        )

    def test_unqualified_or_invalid_draft_declaration_remains_closed(self) -> None:
        from unittest.mock import patch

        from cubic_serve.models.builtin import builtin_draft_provider

        self.assertIsNone(MOE_DEFINITION.draft_provider)
        with self.assertRaises(BuiltinModelConfigError):
            builtin_draft_provider(
                parse_builtin_model_config(qwen3_5_moe_fixture()), method="mtp", max_depth=1
            )
        with self.assertRaises(TypeError):
            replace(DENSE_DEFINITION, draft_provider=object())
        absent = replace(DENSE_DEFINITION, draft_provider=None)
        with (
            patch("cubic_serve.models.builtin.builtin_model_definition", return_value=absent),
            self.assertRaises(BuiltinModelConfigError),
        ):
            builtin_draft_provider(object(), method="mtp", max_depth=1)

    def test_compute_document_transform_is_inherited(self) -> None:
        self.assertIs(DENSE_DEFINITION.compute_config, MOE_DEFINITION.compute_config)
        self.assertIs(DENSE_DEFINITION.compute_config, FLASH_DEFINITION.compute_config)
        raw = qwen3_5_fixture()
        original_dtype = raw["text_config"].get("dtype")
        prepared = DENSE_DEFINITION.compute_config(raw, "float16")
        self.assertEqual(prepared["text_config"]["dtype"], "float16")
        self.assertEqual(raw["text_config"].get("dtype"), original_dtype)

    def test_explicit_definition_owns_its_compute_document_layout(self) -> None:
        from cubic_serve.models.builtin import parse_model_config, parse_model_config_json
        from tests.models.test_quantization import fp8_config

        raw = {
            "model_type": "repo_model",
            "runtime_dtype": "unset",
            "quantization_config": fp8_config(),
        }
        calls = []

        def prepare(document, dtype):
            calls.append(dtype)
            return {**document, "runtime_dtype": dtype}

        declaration = replace(
            DENSE_DEFINITION,
            model_type="repo_model",
            compute_config=prepare,
            parse_config=lambda document: document["runtime_dtype"],
        )
        self.assertEqual(
            parse_model_config(raw, definition=declaration, compute_dtype="fp16"), "float16"
        )
        self.assertEqual(calls, ["float16"])
        with self.assertRaises(BuiltinModelConfigError):
            parse_model_config(raw)
        self.assertEqual(calls, ["float16"])
        self.assertEqual(raw["runtime_dtype"], "unset")
        with self.assertRaises(BuiltinModelConfigError):
            parse_model_config(
                raw, definition=replace(declaration, compute_config=None), compute_dtype="bf16"
            )
        with self.assertRaises(BuiltinModelConfigError):
            parse_model_config_json('{"model_type":"a","model_type":"b"}', definition=declaration)

    def test_dispatch_json_is_strict_before_model_parser_selection(self) -> None:
        for document in (
            "[]",
            "{",
            '{"model_type":"qwen3_5","model_type":"qwen4_exp"}',
            '{"model_type":"qwen3_5","value":NaN}',
        ):
            with self.subTest(document=document), self.assertRaises(BuiltinModelConfigError):
                parse_builtin_model_config_json(document)


if __name__ == "__main__":
    unittest.main()
