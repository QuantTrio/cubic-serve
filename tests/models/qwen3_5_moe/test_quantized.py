"""Quantized router consumer and preservation of the existing segment interface."""

import unittest
from unittest.mock import Mock

import torch

from cubic_serve.contracts.execution import BatchPhase
from cubic_serve.executor.runner import ReferenceExecutionContext
from cubic_serve.models.quantization import QuantizationScheme, WeightFormat
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.models.qwen3_5.model import _SegmentOperators
from cubic_serve.models.qwen3_5_moe.model import route_qwen3_5_moe
from cubic_serve.ops.eager.quantization import QuantizedReferenceOperators, decode_matrix


class QuantizedRouterTests(unittest.TestCase):
    def test_existing_segment_binding_does_not_require_model_lookup(self):
        # Model embedding is not an extra semantic op imposed on BF16 segments.
        binding = _SegmentOperators(Mock(), BatchPhase.DECODE)
        self.assertTrue(callable(binding.linear))
        self.assertTrue(callable(binding.full_attention))

    def test_router_uses_the_same_declared_linear_consumer(self):
        matrix = QuantizedMatrix(
            {
                "qweight": torch.arange(256, dtype=torch.int32).reshape(128, 2),
                "qzeros": torch.full((1, 2), 0x77777777, dtype=torch.int32),
                "scales": torch.full((1, 16), 0.03125, dtype=torch.float16),
            },
            QuantizationScheme(WeightFormat.AWQ, 4, 128),
            (16, 128),
        )
        with ReferenceExecutionContext():
            for dtype in (torch.float16, torch.bfloat16):
                inputs = torch.arange(256, dtype=torch.float32).reshape(2, 128).to(dtype) / 128
                expected = route_qwen3_5_moe(inputs, decode_matrix(matrix).to(dtype), top_k=4)
                actual = route_qwen3_5_moe(
                    inputs, matrix, top_k=4, operators=QuantizedReferenceOperators()
                )
                default = route_qwen3_5_moe(inputs, matrix, top_k=4)
                self.assertTrue(torch.equal(actual.expert_ids, expected.expert_ids))
                self.assertTrue(torch.equal(actual.weights, expected.weights))
                self.assertTrue(torch.equal(default.weights, actual.weights))
                self.assertEqual(actual.weights.dtype, dtype)
