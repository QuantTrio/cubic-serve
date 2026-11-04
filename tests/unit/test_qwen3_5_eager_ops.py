from __future__ import annotations

import math
import unittest
from collections.abc import Callable

import torch
from torch import Tensor

from cubic_serve.executor.runner import ReferenceExecutionContext
from cubic_serve.ops.eager.hybrid import (
    apply_partial_rotary,
    depthwise_causal_conv1d,
    gated_delta_net_recurrent,
    gated_rms_norm,
    interleaved_mrope_cos_sin,
    sigmoid_output_gate,
    zero_centered_rms_norm,
)
from cubic_serve.ops.strategies import activation_precision

BF16 = torch.bfloat16
FP32 = torch.float32


def _bf16(values: object) -> Tensor:
    return torch.tensor(values, dtype=BF16)


def _round_bf16(value: float) -> float:
    """Round one independently computed scalar at an explicit graph boundary."""

    return float(torch.tensor(value, dtype=BF16).item())


def _sigmoid_scalar(value: float) -> float:
    if activation_precision() == "fp32":
        return 1.0 / (1.0 + math.exp(-value))
    argument = _round_bf16(-abs(_round_bf16(value)) * _round_bf16(math.log2(math.e)))
    exponential = _round_bf16(2.0**argument)
    denominator = _round_bf16(1.0 + exponential)
    return _round_bf16((1.0 if value >= 0.0 else exponential) / denominator)


def _silu_scalar(value: float) -> float:
    if activation_precision() == "fp32":
        return value / (1.0 + math.exp(-value))
    return _round_bf16(_round_bf16(value) * _sigmoid_scalar(value))


def _conv_fixture() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    inputs = (torch.arange(36, dtype=FP32).reshape(2, 6, 3) - 17.0).to(BF16) / 16
    weight = _bf16(
        (
            (0.25, -0.5, 0.75, 1.0),
            (-1.0, 0.5, 0.25, -0.125),
            (0.0625, 0.125, -0.25, 0.5),
        )
    )
    state = (torch.arange(18, dtype=FP32).reshape(2, 3, 3) - 8.0).to(BF16) / 32
    bias = _bf16((0.125, -0.25, 0.5))
    return inputs, weight, state, bias


def _run_conv_chunks(
    inputs: Tensor,
    weight: Tensor,
    initial_state: Tensor,
    bias: Tensor,
    chunk_sizes: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    cursor = 0
    state = initial_state
    outputs: list[Tensor] = []
    for chunk_size in chunk_sizes:
        output, state = depthwise_causal_conv1d(
            inputs[:, cursor : cursor + chunk_size],
            weight,
            state,
            bias,
            activation="silu",
            output_dtype=BF16,
        )
        outputs.append(output)
        cursor += chunk_size
    if cursor != inputs.shape[1]:
        raise AssertionError("test chunk sizes do not cover the sequence")
    return torch.cat(outputs, dim=1), state


def _gdn_fixture() -> tuple[Tensor, ...]:
    query = (torch.arange(18, dtype=FP32).reshape(1, 6, 1, 3) - 8.0).to(BF16) / 16
    key = (torch.arange(18, dtype=FP32).reshape(1, 6, 1, 3) + 1.0).to(BF16) / 20
    value = (torch.arange(24, dtype=FP32).reshape(1, 6, 2, 2) - 10.0).to(BF16) / 12
    decay_input = (torch.arange(12, dtype=FP32).reshape(1, 6, 2) - 5.0).to(BF16) / 8
    beta_input = (torch.arange(12, dtype=FP32).reshape(1, 6, 2) - 7.0).to(BF16) / 10
    a_log = _bf16((math.log(0.5), math.log(1.25)))
    dt_bias = _bf16((-0.25, 0.375))
    state = (torch.arange(12, dtype=FP32).reshape(1, 2, 2, 3) - 4.0) / 64
    return query, key, value, decay_input, beta_input, a_log, dt_bias, state


def _run_gdn_chunks(
    arguments: tuple[Tensor, ...], chunk_sizes: tuple[int, ...]
) -> tuple[Tensor, Tensor]:
    query, key, value, decay_input, beta_input, a_log, dt_bias, initial_state = arguments
    cursor = 0
    state = initial_state
    outputs: list[Tensor] = []
    for chunk_size in chunk_sizes:
        token_slice = slice(cursor, cursor + chunk_size)
        output, state = gated_delta_net_recurrent(
            query[:, token_slice],
            key[:, token_slice],
            value[:, token_slice],
            decay_input[:, token_slice],
            beta_input[:, token_slice],
            a_log,
            dt_bias,
            state,
            l2_epsilon=1e-6,
            output_dtype=BF16,
        )
        outputs.append(output)
        cursor += chunk_size
    if cursor != query.shape[1]:
        raise AssertionError("test chunk sizes do not cover the sequence")
    return torch.cat(outputs, dim=1), state


class Qwen3_5RMSNormEagerTests(unittest.TestCase):
    def test_zero_centered_rms_norm_matches_independent_fp64_scalar_oracle(self) -> None:
        inputs = _bf16(((1.5, -2.0, 0.5, 4.0), (-0.25, 0.75, -1.25, 2.5)))
        weight = _bf16((0.25, -0.5, 0.0, 1.0))
        epsilon = 1e-6

        expected_rows: list[list[float]] = []
        for row in inputs.tolist():
            variance = sum(float(value) ** 2 for value in row) / len(row)
            inverse_rms = 1.0 / math.sqrt(variance + epsilon)
            expected_rows.append(
                [
                    float(value) * inverse_rms * (1.0 + float(scale))
                    for value, scale in zip(row, weight.tolist(), strict=True)
                ]
            )

        with ReferenceExecutionContext():
            actual = zero_centered_rms_norm(
                inputs,
                weight,
                epsilon=epsilon,
                output_dtype=BF16,
            )
            isolated = torch.cat(
                tuple(
                    zero_centered_rms_norm(
                        row.unsqueeze(0),
                        weight,
                        epsilon=epsilon,
                        output_dtype=BF16,
                    )
                    for row in inputs
                ),
                dim=0,
            )

        self.assertTrue(torch.equal(actual, torch.tensor(expected_rows, dtype=BF16)))
        self.assertTrue(torch.equal(actual, isolated))

    def test_rms_norm_contract_fails_closed(self) -> None:
        good_inputs = _bf16(((1.0, 2.0),))
        good_weight = _bf16((0.0, 0.0))
        with ReferenceExecutionContext():
            cases: tuple[tuple[Callable[[], Tensor], type[Exception], str], ...] = (
                (
                    lambda: zero_centered_rms_norm(
                        good_inputs.float(),
                        good_weight,
                        epsilon=1e-6,
                        output_dtype=BF16,
                    ),
                    TypeError,
                    "bfloat16",
                ),
                (
                    lambda: zero_centered_rms_norm(
                        good_inputs,
                        good_weight,
                        epsilon=0.0,
                        output_dtype=BF16,
                    ),
                    ValueError,
                    "positive",
                ),
                (
                    lambda: zero_centered_rms_norm(
                        good_inputs,
                        _bf16((0.0,)),
                        epsilon=1e-6,
                        output_dtype=BF16,
                    ),
                    ValueError,
                    "hidden width",
                ),
                (
                    lambda: zero_centered_rms_norm(
                        _bf16(((float("nan"), 1.0),)),
                        good_weight,
                        epsilon=1e-6,
                        output_dtype=BF16,
                    ),
                    ValueError,
                    "finite",
                ),
                (
                    lambda: zero_centered_rms_norm(
                        good_inputs,
                        good_weight,
                        epsilon=1e-6,
                        output_dtype=FP32,
                    ),
                    TypeError,
                    "output_dtype",
                ),
            )
            for invoke, error_type, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(error_type, message):
                    invoke()

    def test_bare_call_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ReferenceExecutionContext"):
            zero_centered_rms_norm(
                _bf16(((1.0,),)),
                _bf16((0.0,)),
                epsilon=1e-6,
                output_dtype=BF16,
            )


class Qwen3_5MRoPEEagerTests(unittest.TestCase):
    def test_three_axis_interleaving_matches_fp64_oracle_and_partial_apply(self) -> None:
        positions = torch.tensor(
            (
                ((1, 2),),
                ((10, 20),),
                ((100, 200),),
            ),
            dtype=torch.int64,
        )
        rotary_dim = 8
        theta = 10_000.0
        axes = (0, 1, 2, 0)
        expected_frequencies: list[list[float]] = []
        for token in range(2):
            row: list[float] = []
            for frequency_index, axis in enumerate(axes):
                inverse_frequency = 1.0 / theta ** ((2 * frequency_index) / rotary_dim)
                row.append(float(positions[axis, 0, token].item()) * inverse_frequency)
            expected_frequencies.append(row + row)
        expected_cosine = torch.tensor(
            [[tuple(math.cos(value) for value in row) for row in expected_frequencies]],
            dtype=BF16,
        )
        expected_sine = torch.tensor(
            [[tuple(math.sin(value) for value in row) for row in expected_frequencies]],
            dtype=BF16,
        )
        query = (torch.arange(40, dtype=FP32).reshape(1, 2, 2, 10) - 17.0).to(BF16) / 8
        key = (torch.arange(20, dtype=FP32).reshape(1, 1, 2, 10) + 3.0).to(BF16) / 16

        with ReferenceExecutionContext():
            cosine, sine = interleaved_mrope_cos_sin(
                positions,
                rotary_dim=rotary_dim,
                mrope_section=(2, 1, 1),
                rope_theta=theta,
                output_dtype=BF16,
            )
            rotated_query, rotated_key = apply_partial_rotary(
                query,
                key,
                cosine,
                sine,
                output_dtype=BF16,
            )

        self.assertTrue(torch.equal(cosine, expected_cosine))
        self.assertTrue(torch.equal(sine, expected_sine))
        for original, rotated in ((query, rotated_query), (key, rotated_key)):
            expected = original.clone()
            for batch in range(original.shape[0]):
                for head in range(original.shape[1]):
                    for token in range(original.shape[2]):
                        source = [float(value) for value in original[batch, head, token, :8]]
                        rotated_half = [-value for value in source[4:]] + source[:4]
                        for width in range(8):
                            left_product = torch.tensor(
                                source[width] * float(cosine[batch, token, width]),
                                dtype=FP32,
                            )
                            right_product = torch.tensor(
                                rotated_half[width] * float(sine[batch, token, width]),
                                dtype=FP32,
                            )
                            expected[batch, head, token, width] = (left_product + right_product).to(
                                BF16
                            )
            self.assertTrue(torch.equal(rotated, expected))
            self.assertTrue(torch.equal(rotated[..., 8:], original[..., 8:]))

    def test_mrope_contract_rejects_ambiguous_layouts_and_invalid_tensors(self) -> None:
        positions = torch.zeros((3, 1, 1), dtype=torch.int64)
        with ReferenceExecutionContext():
            with self.assertRaisesRegex(ValueError, "sum"):
                interleaved_mrope_cos_sin(
                    positions,
                    rotary_dim=8,
                    mrope_section=(1, 1, 1),
                    rope_theta=10_000.0,
                    output_dtype=BF16,
                )
            with self.assertRaisesRegex(ValueError, "interleaving"):
                interleaved_mrope_cos_sin(
                    positions,
                    rotary_dim=8,
                    mrope_section=(1, 2, 1),
                    rope_theta=10_000.0,
                    output_dtype=BF16,
                )
            with self.assertRaisesRegex(TypeError, "int64"):
                interleaved_mrope_cos_sin(
                    positions.float(),
                    rotary_dim=8,
                    mrope_section=(2, 1, 1),
                    rope_theta=10_000.0,
                    output_dtype=BF16,
                )
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                interleaved_mrope_cos_sin(
                    positions - 1,
                    rotary_dim=8,
                    mrope_section=(2, 1, 1),
                    rope_theta=10_000.0,
                    output_dtype=BF16,
                )

    def test_partial_apply_rejects_shape_dtype_and_nonfinite_values(self) -> None:
        query = torch.ones((1, 2, 1, 8), dtype=BF16)
        key = torch.ones((1, 1, 1, 8), dtype=BF16)
        cosine = torch.ones((1, 1, 4), dtype=BF16)
        sine = torch.zeros_like(cosine)
        with ReferenceExecutionContext():
            with self.assertRaisesRegex(TypeError, "bfloat16"):
                apply_partial_rotary(query.float(), key, cosine, sine, output_dtype=BF16)
            with self.assertRaisesRegex(ValueError, "batch/token"):
                apply_partial_rotary(
                    query,
                    key,
                    cosine.expand(1, 2, 4),
                    sine.expand(1, 2, 4),
                    output_dtype=BF16,
                )
            bad_sine = sine.clone()
            bad_sine[0, 0, 0] = float("inf")
            with self.assertRaisesRegex(ValueError, "finite"):
                apply_partial_rotary(query, key, cosine, bad_sine, output_dtype=BF16)


class Qwen3_5CausalConvEagerTests(unittest.TestCase):
    def test_conv_matches_independent_scalar_oracle_with_nonzero_prefix_state(self) -> None:
        inputs = _bf16((((0.5, -1.0), (1.5, 0.25), (-0.75, 2.0), (0.125, -0.5)),))
        weight = _bf16(((0.25, -0.5, 1.0), (-1.0, 0.5, 0.125)))
        initial_state = _bf16((((0.75, -0.25), (-0.5, 1.25)),))
        bias = _bf16((0.125, -0.375))

        scalar_state = [[float(item) for item in initial_state[0, channel]] for channel in range(2)]
        expected_rows: list[list[float]] = []
        for token in range(inputs.shape[1]):
            expected_row: list[float] = []
            for channel in range(inputs.shape[2]):
                window = scalar_state[channel] + [float(inputs[0, token, channel])]
                accumulated = sum(
                    item * float(scale) for item, scale in zip(window, weight[channel], strict=True)
                ) + float(bias[channel])
                expected_row.append(_silu_scalar(accumulated))
                scalar_state[channel] = window[1:]
            expected_rows.append(expected_row)

        with ReferenceExecutionContext():
            output, final_state = depthwise_causal_conv1d(
                inputs,
                weight,
                initial_state,
                bias,
                activation="silu",
                output_dtype=BF16,
            )

        self.assertTrue(torch.equal(output, torch.tensor([expected_rows], dtype=BF16)))
        self.assertTrue(torch.equal(final_state, torch.tensor([scalar_state], dtype=BF16)))
        self.assertTrue(torch.equal(initial_state, _bf16((((0.75, -0.25), (-0.5, 1.25)),))))

    def test_full_chunks_decode_and_prefix_resume_are_bitwise_identical(self) -> None:
        inputs, weight, initial_state, bias = _conv_fixture()
        caller_snapshot = initial_state.clone()
        with ReferenceExecutionContext():
            full_output, full_state = _run_conv_chunks(
                inputs, weight, initial_state, bias, (inputs.shape[1],)
            )
            chunk_output, chunk_state = _run_conv_chunks(
                inputs, weight, initial_state, bias, (2, 1, 3)
            )
            decode_output, decode_state = _run_conv_chunks(
                inputs, weight, initial_state, bias, (1, 1, 1, 1, 1, 1)
            )
            prefix_output, prefix_state = _run_conv_chunks(
                inputs[:, :3], weight, initial_state, bias, (2, 1)
            )
            prefix_snapshot = prefix_state.clone()
            suffix_output, resumed_state = _run_conv_chunks(
                inputs[:, 3:], weight, prefix_snapshot, bias, (1, 2)
            )

        self.assertTrue(torch.equal(initial_state, caller_snapshot))
        self.assertTrue(torch.equal(full_output, chunk_output))
        self.assertTrue(torch.equal(full_output, decode_output))
        self.assertTrue(torch.equal(full_state, chunk_state))
        self.assertTrue(torch.equal(full_state, decode_state))
        self.assertTrue(torch.equal(prefix_output, full_output[:, :3]))
        self.assertTrue(torch.equal(suffix_output, full_output[:, 3:]))
        self.assertTrue(torch.equal(resumed_state, full_state))
        self.assertTrue(torch.equal(prefix_snapshot, prefix_state))

    def test_conv_contract_rejects_bad_dtype_shape_device_and_finiteness(self) -> None:
        inputs, weight, state, bias = _conv_fixture()
        with ReferenceExecutionContext():
            with self.assertRaisesRegex(TypeError, "state must use torch.bfloat16"):
                depthwise_causal_conv1d(
                    inputs,
                    weight,
                    state.float(),
                    bias,
                    activation="silu",
                    output_dtype=BF16,
                )
            with self.assertRaisesRegex(ValueError, "state must have shape"):
                depthwise_causal_conv1d(
                    inputs,
                    weight,
                    state[..., :-1],
                    bias,
                    activation="silu",
                    output_dtype=BF16,
                )
            bad_inputs = inputs.clone()
            bad_inputs[0, 0, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                depthwise_causal_conv1d(
                    bad_inputs,
                    weight,
                    state,
                    bias,
                    activation="silu",
                    output_dtype=BF16,
                )
            meta_state = torch.empty(state.shape, dtype=BF16, device="meta")
            with self.assertRaisesRegex(ValueError, "same device"):
                depthwise_causal_conv1d(
                    inputs,
                    weight,
                    meta_state,
                    bias,
                    activation="silu",
                    output_dtype=BF16,
                )
            with self.assertRaisesRegex(ValueError, "activation"):
                depthwise_causal_conv1d(
                    inputs,
                    weight,
                    state,
                    bias,
                    activation="relu",
                    output_dtype=BF16,
                )


class Qwen3_5GatedDeltaEagerTests(unittest.TestCase):
    def setUp(self) -> None:
        from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

        self.enterContext(operator_strategy_scope(
            OperatorConfig(attention_dtype="fp32"), device="", model=""
        ))

    def test_full_chunks_decode_and_prefix_resume_are_bitwise_identical(self) -> None:
        arguments = _gdn_fixture()
        initial_state = arguments[-1]
        caller_snapshot = initial_state.clone()
        with ReferenceExecutionContext():
            full_output, full_state = _run_gdn_chunks(arguments, (6,))
            chunk_output, chunk_state = _run_gdn_chunks(arguments, (2, 1, 3))
            decode_output, decode_state = _run_gdn_chunks(arguments, (1, 1, 1, 1, 1, 1))

            prefix_arguments = tuple(
                tensor[:, :3] if index < 5 else tensor for index, tensor in enumerate(arguments)
            )
            prefix_output, prefix_state = _run_gdn_chunks(prefix_arguments, (1, 2))
            prefix_snapshot = prefix_state.clone()
            suffix_arguments = (
                *(tensor[:, 3:] for tensor in arguments[:5]),
                arguments[5],
                arguments[6],
                prefix_snapshot,
            )
            suffix_output, resumed_state = _run_gdn_chunks(suffix_arguments, (2, 1))

        self.assertTrue(torch.equal(initial_state, caller_snapshot))
        self.assertTrue(torch.equal(full_output, chunk_output))
        self.assertTrue(torch.equal(full_output, decode_output))
        self.assertTrue(torch.equal(full_state, chunk_state))
        self.assertTrue(torch.equal(full_state, decode_state))
        self.assertTrue(torch.equal(prefix_output, full_output[:, :3]))
        self.assertTrue(torch.equal(suffix_output, full_output[:, 3:]))
        self.assertTrue(torch.equal(resumed_state, full_state))
        self.assertTrue(torch.equal(prefix_snapshot, prefix_state))

    def test_one_token_rule_matches_independent_scalar_oracle(self) -> None:
        query = _bf16(((((0.5, -1.0),),),))
        key = _bf16(((((1.5, 0.25),),),))
        value = _bf16(((((2.0, -0.5, 1.25),),),))
        decay_input = _bf16((((0.2,),),))
        beta_input = _bf16((((-0.3,),),))
        a_log = _bf16((math.log(0.75),))
        dt_bias = _bf16((0.1,))
        state = torch.tensor(
            [[[[0.125, -0.25], [0.375, 0.5], [-0.625, 0.75]]]],
            dtype=FP32,
        )
        epsilon = 1e-6

        with ReferenceExecutionContext():
            output, final_state = gated_delta_net_recurrent(
                query,
                key,
                value,
                decay_input,
                beta_input,
                a_log,
                dt_bias,
                state,
                l2_epsilon=epsilon,
                output_dtype=BF16,
            )

        query_row = [float(item) for item in query[0, 0, 0]]
        key_row = [float(item) for item in key[0, 0, 0]]
        value_row = [float(item) for item in value[0, 0, 0]]
        query_inverse_norm = 1.0 / math.sqrt(sum(item * item for item in query_row) + epsilon)
        key_inverse_norm = 1.0 / math.sqrt(sum(item * item for item in key_row) + epsilon)
        normalized_query = [
            _round_bf16(item * query_inverse_norm) / math.sqrt(len(query_row)) for item in query_row
        ]
        normalized_key = [_round_bf16(item * key_inverse_norm) for item in key_row]
        softplus_input = float(decay_input.item()) + float(dt_bias.item())
        softplus = math.log1p(math.exp(softplus_input))
        decay = math.exp(-math.exp(float(a_log.item())) * softplus)
        beta = float(
            torch.tensor(
                _sigmoid_scalar(float(beta_input.item())),
                dtype=BF16,
            )
        )
        expected_state: list[list[float]] = []
        expected_output: list[float] = []
        for value_index, value_item in enumerate(value_row):
            decayed_row = [float(item) * decay for item in state[0, 0, value_index]]
            memory = sum(
                state_item * key_item
                for state_item, key_item in zip(decayed_row, normalized_key, strict=True)
            )
            delta = (value_item - memory) * beta
            updated_row = [
                state_item + delta * key_item
                for state_item, key_item in zip(decayed_row, normalized_key, strict=True)
            ]
            expected_state.append(updated_row)
            expected_output.append(
                sum(
                    state_item * query_item
                    for state_item, query_item in zip(updated_row, normalized_query, strict=True)
                )
            )

        self.assertTrue(torch.equal(output, torch.tensor([[[expected_output]]], dtype=BF16)))
        torch.testing.assert_close(
            final_state,
            torch.tensor([[expected_state]], dtype=FP32),
            rtol=2e-6,
            atol=2e-6,
        )
        self.assertEqual(final_state.shape, (1, 1, 3, 2))
        self.assertEqual(final_state.dtype, FP32)

    def test_l2_normalized_qk_cross_the_canonical_bf16_boundary(self) -> None:
        query = _bf16([[[[1.5, 2.0]]]])
        key = _bf16([[[[0.75, 1.0]]]])
        value = _bf16([[[[1.0]]]])
        decay_input = _bf16([[[0.0]]])
        beta_input = _bf16([[[0.0]]])
        a_log = _bf16([0.0])
        dt_bias = _bf16([0.0])
        state = torch.zeros((1, 1, 1, 2), dtype=FP32)

        with ReferenceExecutionContext():
            output, final_state = gated_delta_net_recurrent(
                query,
                key,
                value,
                decay_input,
                beta_input,
                a_log,
                dt_bias,
                state,
                l2_epsilon=1e-6,
                output_dtype=BF16,
            )

        query_row = [1.5, 2.0]
        key_row = [0.75, 1.0]
        query_inverse_norm = 1.0 / math.sqrt(sum(item * item for item in query_row) + 1e-6)
        key_inverse_norm = 1.0 / math.sqrt(sum(item * item for item in key_row) + 1e-6)
        normalized_query = [_round_bf16(item * query_inverse_norm) for item in query_row]
        normalized_key = [_round_bf16(item * key_inverse_norm) for item in key_row]
        beta = _round_bf16(0.5)
        expected_state_row = [beta * item for item in normalized_key]
        expected_output = _round_bf16(
            sum(
                state_item * query_item / math.sqrt(len(query_row))
                for state_item, query_item in zip(expected_state_row, normalized_query, strict=True)
            )
        )

        # This is the former all-FP32-normalization result.  The fixture is deliberately
        # rounding-sensitive so removing the canonical BF16 boundary cannot pass silently.
        unrounded_query = [item * query_inverse_norm for item in query_row]
        unrounded_key = [item * key_inverse_norm for item in key_row]
        legacy_output = _round_bf16(
            beta
            * sum(
                key_item * query_item / math.sqrt(len(query_row))
                for key_item, query_item in zip(unrounded_key, unrounded_query, strict=True)
            )
        )
        self.assertNotEqual(legacy_output, expected_output)
        self.assertTrue(torch.equal(output, _bf16([[[[expected_output]]]])))
        self.assertTrue(
            torch.equal(final_state, torch.tensor([[[expected_state_row]]], dtype=FP32))
        )

    def test_gdn_contract_rejects_bad_state_shape_dtype_heads_and_nonfinite_math(self) -> None:
        arguments = _gdn_fixture()
        query, key, value, decay_input, beta_input, a_log, dt_bias, state = arguments
        with ReferenceExecutionContext():
            bf16_state = gated_delta_net_recurrent(
                query,
                key,
                value,
                decay_input,
                beta_input,
                a_log,
                dt_bias,
                state.to(BF16),
                l2_epsilon=1e-6,
                output_dtype=BF16,
            )
            self.assertEqual(bf16_state[1].dtype, BF16)
            with self.assertRaisesRegex(
                TypeError, "state must use torch.bfloat16 or torch.float32"
            ):
                gated_delta_net_recurrent(
                    query,
                    key,
                    value,
                    decay_input,
                    beta_input,
                    a_log,
                    dt_bias,
                    state.to(torch.float16),
                    l2_epsilon=1e-6,
                    output_dtype=BF16,
                )
            float_a_log = gated_delta_net_recurrent(
                query,
                key,
                value,
                decay_input,
                beta_input,
                a_log.float(),
                dt_bias,
                state,
                l2_epsilon=1e-6,
                output_dtype=BF16,
            )
            bf16_a_log = gated_delta_net_recurrent(
                query,
                key,
                value,
                decay_input,
                beta_input,
                a_log,
                dt_bias,
                state,
                l2_epsilon=1e-6,
                output_dtype=BF16,
            )
            self.assertTrue(torch.equal(float_a_log[0], bf16_a_log[0]))
            self.assertTrue(torch.equal(float_a_log[1], bf16_a_log[1]))
            with self.assertRaisesRegex(
                TypeError, "a_log must use torch.bfloat16 or torch.float32"
            ):
                gated_delta_net_recurrent(
                    query,
                    key,
                    value,
                    decay_input,
                    beta_input,
                    a_log.half(),
                    dt_bias,
                    state,
                    l2_epsilon=1e-6,
                    output_dtype=BF16,
                )
            with self.assertRaisesRegex(ValueError, "state must have shape"):
                gated_delta_net_recurrent(
                    query,
                    key,
                    value,
                    decay_input,
                    beta_input,
                    a_log,
                    dt_bias,
                    state[..., :-1],
                    l2_epsilon=1e-6,
                    output_dtype=BF16,
                )
            three_head_value = value[:, :, :1].repeat(1, 1, 3, 1)
            with self.assertRaisesRegex(ValueError, "divisible"):
                gated_delta_net_recurrent(
                    query.repeat(1, 1, 2, 1),
                    key.repeat(1, 1, 2, 1),
                    three_head_value,
                    decay_input[:, :, :1].repeat(1, 1, 3),
                    beta_input[:, :, :1].repeat(1, 1, 3),
                    a_log[:1].repeat(3),
                    dt_bias[:1].repeat(3),
                    torch.zeros((1, 3, 2, 3), dtype=FP32),
                    l2_epsilon=1e-6,
                    output_dtype=BF16,
                )
            overflowing_a_log = a_log.clone()
            overflowing_a_log[0] = 100.0
            with self.assertRaisesRegex(ValueError, "decay log"):
                gated_delta_net_recurrent(
                    query,
                    key,
                    value,
                    decay_input,
                    beta_input,
                    overflowing_a_log,
                    dt_bias,
                    state,
                    l2_epsilon=1e-6,
                    output_dtype=BF16,
                )


class Qwen3_5OutputGateEagerTests(unittest.TestCase):
    def test_gated_norm_and_sigmoid_gate_match_fp64_scalar_oracles(self) -> None:
        inputs = _bf16(((1.0, -2.0, 0.5),))
        gate = _bf16(((0.25, -0.75, 1.5),))
        weight = _bf16((0.5, 1.25, -0.25))
        epsilon = 1e-6
        variance = sum(float(value) ** 2 for value in inputs[0]) / inputs.shape[-1]
        expected_norm = []
        expected_sigmoid = []
        for value, gate_value, scale in zip(
            inputs[0].tolist(), gate[0].tolist(), weight.tolist(), strict=True
        ):
            silu = _silu_scalar(float(gate_value))
            normalized_bf16 = _round_bf16(float(value) / math.sqrt(variance + epsilon))
            weighted_bf16 = _round_bf16(normalized_bf16 * float(scale))
            expected_norm.append(weighted_bf16 * silu)
            sigmoid = torch.tensor(_sigmoid_scalar(float(gate_value)), dtype=BF16)
            expected_sigmoid.append(float(torch.tensor(float(value) * float(sigmoid), dtype=BF16)))

        with ReferenceExecutionContext():
            actual_norm = gated_rms_norm(
                inputs,
                gate,
                weight,
                epsilon=epsilon,
                output_dtype=BF16,
            )
            actual_sigmoid = sigmoid_output_gate(inputs, gate, output_dtype=BF16)

        self.assertTrue(torch.equal(actual_norm, torch.tensor((expected_norm,), dtype=BF16)))
        self.assertTrue(torch.equal(actual_sigmoid, torch.tensor((expected_sigmoid,), dtype=BF16)))

    def test_gated_norm_preserves_official_intermediate_bf16_rounding(self) -> None:
        inputs = _bf16(((-4.0, -3.0),))
        gate = _bf16((((0.5 if activation_precision() == "fp32" else 0.625), -1.0),))
        weight = _bf16((0.125, 1.0))
        epsilon = 1e-6
        variance = sum(float(item) ** 2 for item in inputs[0]) / inputs.shape[-1]

        expected: list[float] = []
        legacy_all_fp32: list[float] = []
        for value, gate_value, scale in zip(
            inputs[0].tolist(), gate[0].tolist(), weight.tolist(), strict=True
        ):
            normalized = float(value) / math.sqrt(variance + epsilon)
            normalized_bf16 = _round_bf16(normalized)
            weighted_bf16 = _round_bf16(normalized_bf16 * float(scale))
            silu = _silu_scalar(float(gate_value))
            expected.append(_round_bf16(weighted_bf16 * silu))
            legacy_all_fp32.append(_round_bf16(normalized * float(scale) * silu))

        with ReferenceExecutionContext():
            actual = gated_rms_norm(
                inputs,
                gate,
                weight,
                epsilon=epsilon,
                output_dtype=BF16,
            )

        self.assertNotEqual(legacy_all_fp32[0], expected[0])
        self.assertTrue(torch.equal(actual, _bf16((expected,))))

    def test_gated_norm_accepts_moe_declared_fp32_weight_without_extra_rounding(self) -> None:
        inputs = _bf16(((-4.0, -3.0),))
        gate = _bf16((((0.5 if activation_precision() == "fp32" else 0.625), -1.0),))
        weight = torch.tensor((0.1234567, 0.9876543), dtype=FP32)
        epsilon = 1e-6
        variance = sum(float(item) ** 2 for item in inputs[0]) / inputs.shape[-1]
        expected = []
        for value, gate_value, scale in zip(
            inputs[0].tolist(), gate[0].tolist(), weight.tolist(), strict=True
        ):
            normalized_bf16 = _round_bf16(float(value) / math.sqrt(variance + epsilon))
            silu = _silu_scalar(float(gate_value))
            expected.append(_round_bf16(normalized_bf16 * float(scale) * silu))
        with ReferenceExecutionContext():
            actual = gated_rms_norm(
                inputs,
                gate,
                weight,
                epsilon=epsilon,
                output_dtype=BF16,
            )
        self.assertTrue(torch.equal(actual, _bf16((expected,))))

    def test_gate_contract_fails_closed(self) -> None:
        inputs = torch.ones((1, 2), dtype=BF16)
        gate = torch.ones_like(inputs)
        weight = torch.ones(2, dtype=BF16)
        with ReferenceExecutionContext():
            bad_gate = gate.clone()
            bad_gate[0, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                gated_rms_norm(
                    inputs,
                    bad_gate,
                    weight,
                    epsilon=1e-6,
                    output_dtype=BF16,
                )
            with self.assertRaisesRegex(ValueError, "same non-scalar shape"):
                sigmoid_output_gate(inputs, gate[:, :1], output_dtype=BF16)
            with self.assertRaisesRegex(TypeError, "output_dtype"):
                sigmoid_output_gate(inputs, gate, output_dtype=FP32)


if __name__ == "__main__":
    unittest.main()
