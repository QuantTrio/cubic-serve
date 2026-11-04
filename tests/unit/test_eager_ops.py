from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from cubic_serve.contracts.execution import BatchPhase, OpSignature, PlanSegment, TensorSpec
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.contracts.request import TokenPosition
from cubic_serve.executor.runner import (
    ReferenceExecutionContext,
    ReferenceOpExecutor,
    ReferenceSegmentIdentity,
    configure_reference_math,
)
from cubic_serve.ops.catalog import ReferenceCatalog, ReferenceImplementation, seal_reference_plan
from cubic_serve.ops.eager.attention import full_attention, qkv_attention_row
from cubic_serve.ops.eager.linear import linear
from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope


def _linear_signature() -> OpSignature:
    return OpSignature(
        semantic_op="linear",
        phase=BatchPhase.DECODE,
        inputs=(
            TensorSpec("inputs", ScalarDType.BF16, ("m", "hidden")),
            TensorSpec("weight", ScalarDType.BF16, ("hidden", "hidden")),
        ),
        outputs=(TensorSpec("result", ScalarDType.BF16, ("m", "hidden")),),
        accumulator_dtype=ScalarDType.FP32,
        layout="mk_kn",
        shape_items=(("hidden", 2), ("m", 1)),
        cache_form=None,
        state_form=None,
        topology_id="single",
    )


def _segment_identity(
    *,
    phase: BatchPhase = BatchPhase.DECODE,
    topology_id: str = "single",
) -> ReferenceSegmentIdentity:
    digest = f"b2s256:{'0' * 64}"
    return ReferenceSegmentIdentity(
        plan_id="plan:fixture",
        topology_id=topology_id,
        partition_id="partition:fixture",
        segment=PlanSegment(
            request_id="request:fixture",
            input_plan_digest=digest,
            execution_input_digest=digest,
            phase=phase,
            produces_output=True,
            token_start=0,
            token_count=1,
            token_ids=(1,),
            positions=(TokenPosition((0,)),),
            cache_slots=(0,),
            cache_pages=(0,),
        ),
    )


class EagerMathPolicyTests(unittest.TestCase):
    def test_policy_disables_implicit_fast_math(self) -> None:
        policy = configure_reference_math()

        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertTrue(policy.deterministic_algorithms)
        self.assertEqual(policy.fp32_precision, "ieee")
        self.assertFalse(policy.tf32)
        self.assertFalse(policy.reduced_precision_reduction)
        self.assertFalse(policy.fp16_accumulation)
        self.assertFalse(policy.cudnn_benchmark)
        self.assertFalse(policy.autocast)
        self.assertFalse(policy.compiled)
        self.assertEqual(torch.backends.fp32_precision, "ieee")
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "ieee")
        self.assertEqual(torch.backends.cudnn.fp32_precision, "ieee")
        self.assertFalse(torch.backends.cuda.matmul.allow_fp16_accumulation)

    def test_reference_context_rejects_ambient_autocast(self) -> None:
        with (
            torch.autocast(device_type="cpu", dtype=torch.bfloat16),
            self.assertRaisesRegex(RuntimeError, "ambient autocast"),
            ReferenceExecutionContext(),
        ):
            self.fail("context should not be entered")

    def test_reference_context_rejects_graph_compilation(self) -> None:
        with (
            patch("torch.compiler.is_compiling", return_value=True),
            self.assertRaisesRegex(RuntimeError, "graph compilation"),
            ReferenceExecutionContext(),
        ):
            self.fail("context should not be entered")

    def test_reference_context_detects_and_restores_math_policy_drift(self) -> None:
        with (
            self.assertRaisesRegex(RuntimeError, "policy drift"),
            ReferenceExecutionContext(),
        ):
            torch.use_deterministic_algorithms(False)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_reference_context_rejects_cuda_graph_capture(self) -> None:
        with (
            patch(
                "cubic_serve.executor.runner._cuda_graph_capture_active",
                return_value=True,
            ),
            self.assertRaisesRegex(RuntimeError, "CUDA graph capture"),
            ReferenceExecutionContext(),
        ):
            self.fail("context should not be entered")
        with self.assertRaisesRegex(RuntimeError, "ReferenceExecutionContext"):
            linear(torch.ones((1, 1)), torch.ones((1, 1)))


class EagerLinearTests(unittest.TestCase):
    def test_bf16_storage_uses_explicit_fp32_accumulation(self) -> None:
        inputs = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        weight = torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.bfloat16)
        bias = torch.tensor([0.5, -0.5], dtype=torch.bfloat16)

        with ReferenceExecutionContext():
            result = linear(inputs, weight, bias, output_dtype=torch.bfloat16)

        self.assertEqual(result.dtype, torch.bfloat16)
        torch.testing.assert_close(
            result.float(),
            torch.tensor([[11.5, 16.5]], dtype=torch.float32),
            rtol=0,
            atol=0,
        )

    def test_packed_rows_are_bitwise_equal_to_isolated_rows(self) -> None:
        inputs = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]], dtype=torch.bfloat16)
        weight = torch.tensor([[3.0, 4.0, 1.0], [5.0, 6.0, -2.0]], dtype=torch.bfloat16)

        with ReferenceExecutionContext():
            packed = linear(inputs, weight)
            isolated = torch.cat(tuple(linear(row[None, :], weight) for row in inputs), dim=0)

        self.assertTrue(torch.equal(packed, isolated))

    def test_bare_reference_op_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ReferenceExecutionContext"):
            linear(torch.ones((1, 1)), torch.ones((1, 1)))


class EagerDispatcherABITests(unittest.TestCase):
    def setUp(self) -> None:
        signature = _linear_signature()
        catalog = ReferenceCatalog(
            (ReferenceImplementation("linear", "fixture.linear.v1", linear),)
        )
        plan = seal_reference_plan("reference:linear:abi", (signature,), catalog)
        self.signature = signature
        self.executor = ReferenceOpExecutor(plan, catalog)
        self.inputs = torch.ones((1, 2), dtype=torch.bfloat16)
        self.weight = torch.eye(2, dtype=torch.bfloat16)

    def test_exact_tensor_abi_is_executable(self) -> None:
        with ReferenceExecutionContext():
            result = self.executor.call(self.signature, self.inputs, self.weight)

        self.assertEqual((result.shape, result.dtype), ((1, 2), torch.bfloat16))

    def test_segment_capability_is_phase_topology_bound_and_revocable(self) -> None:
        with ReferenceExecutionContext():
            unused = self.executor.bind_segment(_segment_identity())
            with self.assertRaisesRegex(RuntimeError, "bypassed"):
                unused.verify_used()
            unused.revoke()

            capability = self.executor.bind_segment(_segment_identity())
            result = capability.call(self.signature, self.inputs, self.weight)
            capability.verify_used()
            capability.revoke()
            with self.assertRaisesRegex(RuntimeError, "capability was revoked"):
                capability.call(self.signature, self.inputs, self.weight)

            wrong_phase = self.executor.bind_segment(_segment_identity(phase=BatchPhase.PREFILL))
            with self.assertRaisesRegex(ValueError, "phase differs"):
                wrong_phase.call(self.signature, self.inputs, self.weight)

            wrong_topology = self.executor.bind_segment(_segment_identity(topology_id="other"))
            with self.assertRaisesRegex(ValueError, "topology differs"):
                wrong_topology.call(self.signature, self.inputs, self.weight)

        self.assertEqual(result.dtype, torch.bfloat16)

    def test_input_dtype_rank_and_shape_fail_before_execution(self) -> None:
        cases = (
            (self.inputs.float(), self.weight, TypeError, "dtype"),
            (self.inputs[0], self.weight, ValueError, "rank"),
            (
                torch.ones((1, 3), dtype=torch.bfloat16),
                self.weight,
                ValueError,
                "axis",
            ),
        )
        with ReferenceExecutionContext():
            for inputs, weight, error_type, reason in cases:
                with self.subTest(reason=reason), self.assertRaisesRegex(error_type, reason):
                    self.executor.call(self.signature, inputs, weight)

    def test_tensor_operand_count_is_exact(self) -> None:
        with (
            ReferenceExecutionContext(),
            self.assertRaisesRegex(TypeError, "sealed callable ABI"),
        ):
            self.executor.call(self.signature, self.inputs)

    def test_undeclared_tensor_and_dtype_controls_fail_closed(self) -> None:
        with ReferenceExecutionContext():
            with self.assertRaisesRegex(TypeError, "undeclared tensor operand 'bias'"):
                self.executor.call(
                    self.signature,
                    self.inputs,
                    self.weight,
                    torch.zeros(2, dtype=torch.bfloat16),
                )
            with self.assertRaisesRegex(TypeError, "output_dtype differs"):
                self.executor.call(
                    self.signature,
                    self.inputs,
                    self.weight,
                    output_dtype=torch.float32,
                )

    def test_multi_output_structure_and_each_tensor_are_checked(self) -> None:
        def invalid_pair(value: torch.Tensor) -> list[torch.Tensor]:
            return [value, value]

        signature = OpSignature(
            semantic_op="fixture.pair",
            phase=BatchPhase.DECODE,
            inputs=(TensorSpec("value", ScalarDType.BF16, ("m",)),),
            outputs=(
                TensorSpec("left", ScalarDType.BF16, ("m",)),
                TensorSpec("right", ScalarDType.BF16, ("m",)),
            ),
            accumulator_dtype=ScalarDType.FP32,
            layout="vector_pair",
            shape_items=(("m", 2),),
            cache_form=None,
            state_form=None,
            topology_id="single",
        )
        catalog = ReferenceCatalog(
            (
                ReferenceImplementation(
                    "fixture.pair",
                    "fixture.pair.invalid.v1",
                    invalid_pair,
                ),
            )
        )
        executor = ReferenceOpExecutor(
            seal_reference_plan("reference:pair", (signature,), catalog), catalog
        )

        with (
            ReferenceExecutionContext(),
            self.assertRaisesRegex(TypeError, "exact 2-tensor tuple"),
        ):
            executor.call(signature, torch.ones(2, dtype=torch.bfloat16))

    def test_returned_tensor_dtype_is_checked_without_a_callable_control(self) -> None:
        def wrong_dtype(value: torch.Tensor) -> torch.Tensor:
            return value.float()

        signature = OpSignature(
            semantic_op="fixture.wrong_dtype",
            phase=BatchPhase.DECODE,
            inputs=(TensorSpec("value", ScalarDType.BF16, ("m",)),),
            outputs=(TensorSpec("result", ScalarDType.BF16, ("m",)),),
            accumulator_dtype=ScalarDType.FP32,
            layout="vector",
            shape_items=(("m", 2),),
            cache_form=None,
            state_form=None,
            topology_id="single",
        )
        catalog = ReferenceCatalog(
            (
                ReferenceImplementation(
                    "fixture.wrong_dtype",
                    "fixture.wrong_dtype.v1",
                    wrong_dtype,
                ),
            )
        )
        executor = ReferenceOpExecutor(
            seal_reference_plan("reference:wrong-dtype", (signature,), catalog), catalog
        )

        with (
            ReferenceExecutionContext(),
            self.assertRaisesRegex(TypeError, "output 'result' dtype"),
        ):
            executor.call(signature, torch.ones(2, dtype=torch.bfloat16))


class EagerAttentionTests(unittest.TestCase):
    def test_exhausted_kv_split_preserves_uniform_attention_weights(self) -> None:
        # Uniform scores make the answer the exact arithmetic mean, independent
        # of how many iterations the shorter final split needs. Nonzero scores
        # expose accidental rescaling after that split has consumed its keys.
        for dtype in (torch.bfloat16, torch.float16):
            for length in (33, 511, 513, 1025):
                for splits in (1, 32):
                    span = (length + splits - 1) // splits
                    tail = length - ((length - 1) // span) * span
                    value = torch.zeros((length, 2), dtype=dtype)
                    value[-tail:, 0] = 1
                    value[-tail:, 1] = -1
                    expected = torch.tensor([tail / length, -tail / length], dtype=dtype)
                    for score in (-1024.0, -2.0, 0.0, 2.0, 1024.0):
                        with (
                            self.subTest(dtype=dtype, length=length, splits=splits, score=score),
                            operator_strategy_scope(
                                OperatorConfig(attention_dtype="auto", softmax_dtype="fp32"),
                                device="", model="",
                            ),
                            ReferenceExecutionContext(),
                        ):
                            key = torch.zeros((length, 2), dtype=dtype)
                            key[:, 0] = score
                            actual = qkv_attention_row(
                                torch.tensor([1.0, 0.0], dtype=dtype),
                                key, value, scale=1.0, kv_splits=splits,
                            )
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_positions_define_causality_for_decode_and_prefix(self) -> None:
        query = torch.zeros((1, 2, 2, 1), dtype=torch.bfloat16)
        key = torch.zeros((1, 1, 3, 1), dtype=torch.bfloat16)
        value = torch.tensor([[[[2.0], [4.0], [8.0]]]], dtype=torch.bfloat16)
        query_positions = torch.tensor([[1, 2]], dtype=torch.int64)
        key_positions = torch.tensor([[0, 1, 2]], dtype=torch.int64)

        with ReferenceExecutionContext():
            result = full_attention(
                query,
                key,
                value,
                query_ordinals=query_positions,
                key_ordinals=key_positions,
            )

        self.assertEqual(result.shape, (1, 2, 2, 1))
        self.assertEqual(result.dtype, torch.bfloat16)
        expected = torch.tensor(
            [[[[3.0], [14.0 / 3.0]], [[3.0], [14.0 / 3.0]]]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(result.float(), expected, rtol=0, atol=0.02)

    def test_fully_masked_query_is_rejected(self) -> None:
        query = torch.zeros((1, 1, 1, 2), dtype=torch.bfloat16)
        key = torch.zeros((1, 1, 1, 2), dtype=torch.bfloat16)
        value = torch.zeros((1, 1, 1, 2), dtype=torch.bfloat16)

        with (
            ReferenceExecutionContext(),
            self.assertRaisesRegex(ValueError, "at least one key"),
        ):
            full_attention(
                query,
                key,
                value,
                query_ordinals=torch.tensor([[0]]),
                key_ordinals=torch.tensor([[0]]),
                attention_mask=torch.tensor([[[False]]]),
            )

    def test_causal_order_is_not_a_multimodal_rope_axis(self) -> None:
        query = torch.zeros((1, 1, 2, 1), dtype=torch.bfloat16)
        key = torch.zeros((1, 1, 3, 1), dtype=torch.bfloat16)
        value = torch.tensor([[[[2.0], [4.0], [8.0]]]], dtype=torch.bfloat16)
        # A corresponding M-RoPE axis could be (0, 0, 0); it is intentionally absent here.
        with ReferenceExecutionContext():
            result = full_attention(
                query,
                key,
                value,
                query_ordinals=torch.tensor([[1, 2]], dtype=torch.int64),
                key_ordinals=torch.tensor([[0, 1, 2]], dtype=torch.int64),
            )

        expected = torch.tensor([[[[3.0], [14.0 / 3.0]]]], dtype=torch.float32)
        torch.testing.assert_close(result.float(), expected, rtol=0, atol=0.02)

    def test_prefill_row_is_bitwise_equal_to_isolated_decode_row(self) -> None:
        query = torch.tensor([[[[1.0, 0.5], [0.25, 2.0], [3.0, -1.0]]]], dtype=torch.bfloat16)
        key = torch.tensor([[[[0.5, 1.0], [2.0, -0.5], [1.0, 1.0]]]], dtype=torch.bfloat16)
        value = torch.tensor([[[[2.0, 3.0], [4.0, -1.0], [8.0, 2.0]]]], dtype=torch.bfloat16)
        ordinals = torch.tensor([[0, 1, 2]], dtype=torch.int64)

        with ReferenceExecutionContext():
            prefill = full_attention(
                query,
                key,
                value,
                query_ordinals=ordinals,
                key_ordinals=ordinals,
            )
            decode = full_attention(
                query[:, :, 2:3, :],
                key,
                value,
                query_ordinals=ordinals[:, 2:3],
                key_ordinals=ordinals,
            )

        self.assertTrue(torch.equal(prefill[:, :, 2:3, :], decode))


if __name__ == "__main__":
    unittest.main()
