from __future__ import annotations

import os
import unittest
from contextlib import ExitStack
from unittest import mock

import torch
from torch import Tensor

from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
from cubic_serve.ops.cuda.normalization import (
    BF16ResidualRMSNormPrimitive,
    ZeroCenteredRMSNormPrimitive,
)
from cubic_serve.ops.eager.hybrid import zero_centered_rms_norm

RUN_REAL = os.environ.get("CUBIC_RUN_CUDA_NORMALIZATION_TESTS") == "1"


def _physical_rows(logical_m: int) -> tuple[int | None, ...]:
    return {
        1: (None, 0, None, None),
        2: (1, None, None, 0),
        4: (2, 0, 3, 1),
    }[logical_m]


class ZeroCenteredRMSNormValidationTests(unittest.TestCase):
    def test_rejects_non_cuda_storage(self) -> None:
        x = torch.empty((4, 16), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "CUDA FP16/BF16"):
            ZeroCenteredRMSNormPrimitive(
                x,
                torch.empty((16,), dtype=torch.bfloat16),
                torch.empty_like(x),
                epsilon=1e-6,
            )


@unittest.skipUnless(
    RUN_REAL,
    "set CUBIC_RUN_CUDA_NORMALIZATION_TESTS=1 for real CUDA qualification",
)
class ZeroCenteredRMSNormGraphTests(unittest.TestCase):
    def test_one_captured_shape_is_row_invariant_for_m1_m2_m4(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        if torch.cuda.get_device_capability(device) != (12, 0):
            self.skipTest("this real qualification is pinned to SM120")

        torch.manual_seed(991)
        x = torch.empty((4, 5120), device=device, dtype=torch.bfloat16)
        weight = torch.randn((5120,), device=device, dtype=torch.bfloat16) * 0.02
        output = torch.empty_like(x)
        eager_output = torch.empty_like(x)
        primitive = ZeroCenteredRMSNormPrimitive(x, weight, output, epsilon=1e-6)
        eager = ZeroCenteredRMSNormPrimitive(x, weight, eager_output, epsilon=1e-6)
        addresses = (x.data_ptr(), weight.data_ptr(), output.data_ptr())
        self.assertFalse(any(isinstance(value, Tensor) for value in vars(primitive).values()))

        stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.device(device), torch.cuda.stream(stream):
            primitive.launch(x, weight, output)
            primitive.launch(x, weight, output)
        stream.synchronize()

        def forbidden(*args, **kwargs):
            del args, kwargs
            raise AssertionError("RMSNorm launch performed a forbidden host operation")

        with ExitStack() as stack:
            for owner, names in (
                (torch, ("empty", "empty_like", "zeros", "ones", "full", "tensor", "compile")),
                (Tensor, ("clone", "item", "new_empty", "new_zeros", "new_full")),
                (torch.cuda, ("synchronize",)),
                (torch.cuda.Stream, ("synchronize",)),
            ):
                for name in names:
                    stack.enter_context(mock.patch.object(owner, name, forbidden))
            with torch.cuda.device(device), torch.cuda.stream(stream):
                primitive.launch(x, weight, output)
        stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            primitive.launch(x, weight, output)
        stream.synchronize()

        for logical_m in (1, 2, 4):
            logical_x = torch.randn((logical_m, 5120), device=device, dtype=torch.bfloat16)
            poison_results: list[Tensor] = []
            for poison in (37.0, -73.0):
                for physical, logical in enumerate(_physical_rows(logical_m)):
                    if logical is None:
                        x[physical].fill_(poison)
                    else:
                        x[physical].copy_(logical_x[logical])
                graph.replay()
                torch.cuda.synchronize(device)
                eager.launch(x, weight, eager_output)
                torch.cuda.synchronize(device)
                self.assertTrue(torch.equal(output, eager_output))
                logical_output = torch.stack(
                    tuple(
                        output[physical]
                        for logical in range(logical_m)
                        for physical, mapped in enumerate(_physical_rows(logical_m))
                        if mapped == logical
                    )
                )
                poison_results.append(logical_output)

            self.assertTrue(torch.equal(poison_results[0], poison_results[1]))
            reference_x = logical_x.float()
            reference = reference_x * torch.rsqrt(
                (reference_x * reference_x).sum(dim=1, keepdim=True) / 5120 + 1e-6
            )
            reference = (reference * (weight.float() + 1.0)).to(torch.bfloat16)
            # Triton's fixed per-row tree and PyTorch's eager reduction may round
            # the variance differently.  Padding isolation remains exact above;
            # the only admitted arithmetic drift here is one final BF16 ULP from
            # that FP32 reduction reassociation.
            actual = poison_results[0]
            upper = torch.nextafter(reference, torch.full_like(reference, float("inf")))
            lower = torch.nextafter(reference, torch.full_like(reference, -float("inf")))
            allowed = torch.maximum(
                (upper.float() - reference.float()).abs(),
                (reference.float() - lower.float()).abs(),
            )
            self.assertTrue(torch.all((actual.float() - reference.float()).abs() <= allowed))

        self.assertEqual(addresses, (x.data_ptr(), weight.data_ptr(), output.data_ptr()))
        with self.assertRaisesRegex(ValueError, "fixed launch meta"):
            primitive.launch(x[:3], weight, output[:3])

    def test_activation_dtype_rounding_graph_rows_and_binding(self) -> None:
        configure_reference_math()
        with ReferenceExecutionContext():
            for dtype in (torch.float16, torch.bfloat16):
                # A midpoint residual must round before normalization. All
                # squares and sums here are exact, so no reduction allowance
                # can mask a changed storage boundary.
                width = 5120
                weight = (torch.arange(width, device="cuda:0") % 8).to(dtype) / 8
                pattern = (
                    torch.arange(33, device="cuda:0")[:, None]
                    + torch.arange(width, device="cuda:0")[None, :]
                )
                corpus = torch.where(pattern % 3 == 0, -1.0, 1.0).to(dtype)
                update_value = 2 ** (-11 if dtype is torch.float16 else -8)
                update_corpus = corpus * update_value
                expected_sum = corpus + update_corpus
                expected = zero_centered_rms_norm(
                    expected_sum,
                    weight,
                    epsilon=3.0,
                    output_dtype=dtype,
                )
                for capacity, logical in ((1, 1), (8, 8), (64, 33)):
                    residual = torch.zeros(capacity, width, dtype=dtype, device="cuda:0")
                    update = torch.zeros_like(residual)
                    summed, output = torch.empty_like(residual), torch.empty_like(residual)
                    primitive = BF16ResidualRMSNormPrimitive(
                        residual,
                        update,
                        summed,
                        weight,
                        output,
                        epsilon=3.0,
                    )
                    primitive.launch(residual, update, summed, weight, output)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        primitive.launch(residual, update, summed, weight, output)
                    for reverse in (False, True):
                        residual.fill_(float("nan"))
                        update.fill_(float("nan"))
                        residual[:logical].copy_(
                            corpus[:logical].flip(0) if reverse else corpus[:logical]
                        )
                        update[:logical].copy_(
                            update_corpus[:logical].flip(0) if reverse else update_corpus[:logical]
                        )
                        graph.replay()
                        actual_sum = summed[:logical].flip(0) if reverse else summed[:logical]
                        actual = output[:logical].flip(0) if reverse else output[:logical]
                        self.assertTrue(torch.equal(actual_sum, expected_sum[:logical]))
                        self.assertTrue(torch.equal(actual, expected[:logical]))
                    other = torch.bfloat16 if dtype is torch.float16 else torch.float16
                    with self.assertRaisesRegex(ValueError, "fixed launch meta"):
                        primitive.launch(
                            *(t.to(other) for t in (residual, update, summed, weight, output))
                        )
                    standalone = ZeroCenteredRMSNormPrimitive(residual, weight, output, epsilon=3.0)
                    with self.assertRaisesRegex(ValueError, "fixed launch meta"):
                        standalone.launch(*(t.to(other) for t in (residual, weight, output)))

    def test_fused_residual_matches_two_launch_boundary_exactly(self) -> None:
        device = torch.device("cuda:0")
        residual = torch.randn((4, 5120), dtype=torch.bfloat16, device=device)
        update = torch.randn_like(residual)
        weight = torch.randn(5120, dtype=torch.bfloat16, device=device)
        expected_sum = torch.empty_like(residual)
        expected_norm = torch.empty_like(residual)
        actual_sum = torch.empty_like(residual)
        actual_norm = torch.empty_like(residual)
        expected_sum.copy_(residual + update)
        ZeroCenteredRMSNormPrimitive(expected_sum, weight, expected_norm, epsilon=1e-6).launch(
            expected_sum, weight, expected_norm
        )
        fused = BF16ResidualRMSNormPrimitive(
            residual, update, actual_sum, weight, actual_norm, epsilon=1e-6
        )
        fused.launch(residual, update, actual_sum, weight, actual_norm)
        torch.cuda.synchronize(device)
        self.assertTrue(torch.equal(actual_sum, expected_sum))
        self.assertTrue(torch.equal(actual_norm, expected_norm))


if __name__ == "__main__":
    unittest.main()
