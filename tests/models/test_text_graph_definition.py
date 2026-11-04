from __future__ import annotations

import unittest
from dataclasses import replace
from math import lcm

import torch

from cubic_serve.contracts.execution import ParallelTopology
from cubic_serve.executor.quantized_model_runner import QuantizedDecodeStage
from cubic_serve.executor.text_serving import FlashNextDecodeStage
from cubic_serve.models.qwen3_5 import MODEL_DEFINITION as DENSE
from cubic_serve.models.qwen3_5_moe import MODEL_DEFINITION as MOE
from cubic_serve.models.qwen4_exp import MODEL_DEFINITION as FLASH
from tests.models.qwen3_5.test_config import _fixture as dense_fixture
from tests.models.qwen3_5_moe.test_config import _fixture as moe_fixture
from tests.models.qwen4_exp.test_config import _fixture as flash_fixture


class TextGraphDefinitionTests(unittest.TestCase):
    def topology(self, tp=1, ep=1):
        return ParallelTopology.single_node(
            topology_id="startup.declaration", device_indices=tuple(range(tp)),
            tp_size=tp, pp_size=1, ep_size=ep,
        )

    def test_hybrid_constructor_and_draft_options_keep_native_contract(self):
        for definition, fixture, operations in (
            (DENSE, dense_fixture, ("linear",)),
            (MOE, moe_fixture, ("linear", "moe")),
        ):
            config = definition.parse_config(fixture())
            for tp in (1, 2, 4):
                with self.subTest(model=definition.model_type, tp=tp):
                    graph = definition.text_graph(
                        config, self.topology(tp), block_size=16,
                        recurrent_state_dtype="bfloat16",
                    )
                    self.assertIs(graph.stage_type, QuantizedDecodeStage)
                    self.assertFalse(graph.named_state_fields)
                    self.assertEqual(graph.stage_options, {"route_completion_costs": {}})
                    self.assertEqual(graph.required_operations, operations)
                    self.assertEqual(graph.block_size_tokens, 16)
            for dtype in ("auto", "float32"):
                with self.assertRaisesRegex(ValueError, "require.*bfloat16"):
                    definition.text_graph(
                        config, self.topology(), block_size=16, recurrent_state_dtype=dtype,
                    )

    def test_named_state_precision_and_alignment_follow_model_and_topology(self):
        config = FLASH.parse_config(flash_fixture())
        for tp in (1, 2, 4):
            for dtype in ("auto", "bfloat16", "float32"):
                with self.subTest(tp=tp, dtype=dtype):
                    graph = FLASH.text_graph(
                        config, self.topology(tp, tp), block_size=16,
                        recurrent_state_dtype=dtype,
                    )
                    expected = (config.text.gated_delta_net.recurrent_state_dtype
                                if dtype == "auto" else dtype)
                    self.assertIs(graph.stage_type, FlashNextDecodeStage)
                    self.assertTrue(graph.named_state_fields)
                    self.assertEqual(graph.state_audit_steps, 2 * config.text.qsa.compression_ratio)
                    self.assertEqual(graph.recurrent_state_dtype, expected)
                    self.assertEqual(
                        graph.stage_options, {"recurrent_state_dtype": getattr(torch, expected)},
                    )
                    self.assertEqual(graph.required_operations, ("linear", "moe"))
                    self.assertEqual(
                        graph.block_size_tokens, lcm(16, config.text.qsa.compression_ratio * tp),
                    )

    def test_derivatives_inherit_binding_and_mutable_costs_are_instance_owned(self):
        self.assertIs(MOE.text_graph, DENSE.text_graph)
        derivative = replace(DENSE, model_type="user_derivative")
        self.assertIs(derivative.text_graph, DENSE.text_graph)
        config = DENSE.parse_config(dense_fixture())
        kwargs = dict(block_size=16, recurrent_state_dtype="bfloat16")
        first = derivative.text_graph(config, self.topology(), **kwargs)
        second = derivative.text_graph(config, self.topology(), **kwargs)
        first.stage_options["route_completion_costs"][1] = 3
        self.assertEqual(second.stage_options["route_completion_costs"], {})
        with self.assertRaises(TypeError):
            replace(DENSE, text_graph=object())
