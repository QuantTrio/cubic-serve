import math
import unittest
from types import SimpleNamespace

import torch

from cubic_serve.executor.runner import ReferenceExecutionContext
from cubic_serve.guardrail.quantization import QuantizedCausalReplay, QuantizedOperatorObserver
from cubic_serve.models.quantization import ActivationFormat, QuantizationScheme, WeightFormat
from cubic_serve.ops.eager.quantization import (
    QuantizedMatrix,
    decode_awq,
    decode_cubic,
    decode_matrix,
    decode_nvfp4,
    embedding_matrix,
    linear_matrix,
    quantize_activation,
    unpack_cubic,
)


class QuantizedReferenceTests(unittest.TestCase):
    def test_causal_replay_rejects_signed_zero_producer_change(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.tensor([[0.0, 1.0]], dtype=dtype)
                weight = torch.eye(2, dtype=dtype)
                observer = QuantizedOperatorObserver(SimpleNamespace(linear=lambda x, w: x.clone()))
                observer.linear(x, weight)
                changed = x.clone()
                changed[0, 0] = -0.0
                replay = QuantizedCausalReplay(observer.events, None)
                with self.assertRaisesRegex(AssertionError, "unwitnessed producing boundary"):
                    replay.linear(changed, weight)

    def test_causal_proof_reuse_preserves_signed_zero_identity(self):
        x = torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16)
        weight = torch.eye(2, dtype=x.dtype)
        changed = x.clone()
        changed[0, 0] = -0.0
        observer = QuantizedOperatorObserver(SimpleNamespace(linear=lambda x, w: x.clone()))
        for operand in (x, x, changed):
            observer.linear(operand, weight)
        replay = QuantizedCausalReplay(observer.events, None)
        for operand in (x, x, changed):
            replay.linear(operand, weight)
        replay.finish()
        self.assertEqual(replay.reports[1]["reused_from_event"], 1)
        self.assertNotIn("reused_from_event", replay.reports[2])

    def test_causal_embedding_rejects_signed_zero_output_change(self):
        ids = torch.tensor([0])
        weight = torch.tensor([[0.0]], dtype=torch.bfloat16)
        observer = QuantizedOperatorObserver(
            SimpleNamespace(embedding=lambda ids, w, **kw: -w[ids])
        )
        observer.embedding(ids, weight, dtype=weight.dtype)
        replay = QuantizedCausalReplay(
            observer.events, SimpleNamespace(embedding=lambda ids, w, **kw: w[ids])
        )
        with self.assertRaisesRegex(AssertionError, "embedding differs"):
            replay.embedding(ids, weight, dtype=weight.dtype)

    def test_fp16_and_bf16_linear_round_at_requested_boundary(self):
        encoding = QuantizationScheme(WeightFormat.FP8, 8, 1)
        payload = {
            "weight": torch.tensor([[1.0, 2.0], [3.0, -4.0]]).to(torch.float8_e4m3fn),
            "weight_scale": torch.tensor([[0.12345], [0.54321]]),
        }
        matrix = QuantizedMatrix(payload, encoding, (2, 2))
        results = []
        with ReferenceExecutionContext():
            for dtype in (torch.float16, torch.bfloat16):
                x = torch.tensor([[1.25, -0.75]], dtype=dtype)
                actual = linear_matrix(x, matrix)
                rounded = torch.tensor([0.12345, 0.24690, 1.62963, -2.17284]).to(dtype).float()
                expected = torch.stack(
                    (1.25 * rounded[0] - 0.75 * rounded[1], 1.25 * rounded[2] - 0.75 * rounded[3])
                )[None].to(dtype)
                self.assertEqual(actual.dtype, dtype)
                self.assertTrue(torch.equal(actual, expected))
                results.append(actual.float())
        self.assertFalse(torch.equal(*results))

    def test_cubic_row_slice_retains_original_partition_globals(self):
        encoding = QuantizationScheme(
            WeightFormat.CUBIC, 4, 2, metadata="int8-scale-int4-ab-global-fp32"
        )
        payload = {
            "weight_packed": torch.full((6, 1), 0x77, dtype=torch.uint8),
            "weight_scale": torch.ones((6, 1), dtype=torch.int8),
            "weight_ab": torch.zeros((6, 1), dtype=torch.uint8),
            "weight_scale_global": torch.tensor([1.0, 2.0, 4.0]),
            "weight_a_global": torch.ones(3),
            "weight_b_global": torch.ones(3),
        }
        matrix = QuantizedMatrix(payload, encoding, (6, 2), (2, 2, 2))
        self.assertTrue(torch.equal(decode_matrix(matrix.rows(1, 5)), decode_matrix(matrix)[1:5]))
        ids = torch.tensor([[5, 0, 3]])
        self.assertTrue(
            torch.equal(
                embedding_matrix(ids, matrix, dtype=torch.float16),
                decode_matrix(matrix)[ids].half(),
            )
        )

    def test_nvfp4_local_activation_is_invariant_to_concurrent_rows(self):
        encoding = QuantizationScheme(
            WeightFormat.NVFP4, 4, 16, activation=ActivationFormat.NVFP4_LOCAL
        )
        one = torch.arange(32).reshape(1, 32).half() / 17
        original = quantize_activation(one, encoding, global_inverse=torch.ones(1))
        for count in (8, 33):
            values = one.repeat(count, 1)
            values[1:] *= torch.arange(1, count)[:, None]
            batched = quantize_activation(values, encoding, global_inverse=torch.ones(1))
            self.assertTrue(torch.equal(original.codes[0], batched.codes[0]))
            self.assertTrue(torch.equal(original.scales[0].float(), batched.scales[0].float()))
            self.assertTrue(torch.equal(original.dequantized[0], batched.dequantized[0]))

    def test_awq_word_order_and_zero_points(self):
        order = [0, 4, 1, 5, 2, 6, 3, 7]
        word = sum((column + 1) << (4 * order[column]) for column in range(8))
        if word >= 2**31:
            word -= 2**32
        zeros = sum(2 << (4 * lane) for lane in range(8))
        payload = {
            "qweight": torch.full((128, 1), word, dtype=torch.int32),
            "qzeros": torch.full((1, 1), zeros, dtype=torch.int32),
            "scales": torch.full((1, 8), 0.5, dtype=torch.float16),
        }
        expected = torch.tensor([(n - 1) * 0.5 for n in range(8)])[:, None].expand(8, 128)
        self.assertTrue(torch.equal(decode_awq(payload, (8, 128), 128), expected))

    def test_cubic_all_codes_cross_byte_and_reserved_zero(self):
        for bits in range(1, 9):
            raw = list(range(1 << bits))
            stream = sum(code << (i * bits) for i, code in enumerate(raw))
            packed = torch.tensor(
                list(stream.to_bytes(math.ceil(len(raw) * bits / 8), "little")), dtype=torch.uint8
            )[None]
            decoded = unpack_cubic(packed, bits=bits, input_size=len(raw)).tolist()[0]
            expected = []
            for code in raw:
                if bits == 1:
                    expected.append(2 * code - 1)
                elif code == 1 << (bits - 1):
                    expected.append(0)
                else:
                    expected.append(code - (1 << bits) if code >= 1 << (bits - 1) else code)
            self.assertEqual(decoded, expected)

    def test_cubic_compact_partition_coefficients(self):
        scheme = QuantizationScheme(
            WeightFormat.CUBIC, 4, 2, metadata="int8-scale-int4-ab-global-fp32"
        )
        payload = {
            "weight_packed": torch.tensor([[0x77], [0x77]], dtype=torch.uint8),
            "weight_scale": torch.tensor([[2], [3]], dtype=torch.int8),
            "weight_ab": torch.zeros((2, 1), dtype=torch.uint8),
            "weight_scale_global": torch.tensor([0.5, 2.0]),
            "weight_a_global": torch.tensor([0.125, 0.25]),
            "weight_b_global": torch.tensor([0.25, 0.5]),
        }
        self.assertTrue(
            torch.equal(
                decode_cubic(payload, scheme, (2, 2), (1, 1)),
                torch.tensor([[1.0, 1.0], [6.0, 6.0]]),
            )
        )

    def test_cubic_curve2_metadata_uses_high_two_bits(self):
        scheme = QuantizationScheme(
            WeightFormat.CUBIC, 4, 2, metadata="e5m9-scale-curve2-tensor-fp16"
        )
        # FP16 1.0 = 0x3c00; scale stores that bit pattern shifted right one.
        payload = {
            "weight_packed": torch.tensor([[0x77]] * 4, dtype=torch.uint8),
            "weight_metadata": torch.tensor(
                [[0x1E00 | (i << 14)] for i in range(4)], dtype=torch.uint16
            ),
            "weight_curve_a": torch.tensor([[1.0, 0.5, 0.25, 0.0]], dtype=torch.float16),
            "weight_curve_b": torch.zeros((1, 4), dtype=torch.float16),
        }
        self.assertTrue(torch.equal(decode_cubic(payload, scheme, (4, 2), (4,)), torch.ones(4, 2)))

    def test_nvfp4_nibbles_and_inverse_global_scale(self):
        payload = {
            "weight_packed": torch.arange(0, 256, 17, dtype=torch.uint8)[None],
            "weight_scale": torch.ones((1, 2)).to(torch.float8_e4m3fn),
            "weight_global_scale": torch.tensor([2.0]),
        }
        levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        expected = [v * sign / 2 for sign in (1, -1) for v in levels for _ in range(2)]
        self.assertTrue(torch.equal(decode_nvfp4(payload, (1, 32)), torch.tensor([expected])))

    def test_nvfp4_midpoints_round_to_even_codes(self):
        scheme = QuantizationScheme(
            WeightFormat.NVFP4, 4, 16, activation=ActivationFormat.NVFP4_LOCAL
        )
        values = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
        x = torch.tensor([values + [-v for v in values]])
        result = quantize_activation(x, scheme, global_inverse=torch.tensor([1.0]))
        codes = [0, 2, 2, 4, 4, 6, 6, 7]
        self.assertEqual(result.codes.tolist(), [codes + [c + 8 for c in codes]])
        self.assertEqual(result.scales.float().item(), 1.0)

    def test_fp8_token_scale_row_permutation_and_zero(self):
        scheme = QuantizationScheme(WeightFormat.FP8, 8, 1, activation=ActivationFormat.FP8_TOKEN)
        x = torch.tensor([[0.0, 0.0], [1.0, -2.0], [448.0, -448.0]])
        actual = quantize_activation(x, scheme)
        permuted = quantize_activation(x[[2, 0, 1]], scheme)
        self.assertTrue(torch.equal(actual.dequantized[[2, 0, 1]], permuted.dequantized))
        self.assertTrue(torch.equal(actual.scales[[2, 0, 1]], permuted.scales))
        self.assertTrue(torch.equal(actual.dequantized[0], torch.zeros(2)))


if __name__ == "__main__":
    unittest.main()
