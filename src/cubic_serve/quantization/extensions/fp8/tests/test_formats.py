import os
import unittest

import torch
import triton
import triton.language as tl

from cubic_serve.executor.runner import configure_reference_math
from cubic_serve.ops.cuda.quantized_formats import fp8_e4m3fn_decode, fp8_e4m3fn_encode


@triton.jit
def _encode(X, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(Y + i, fp8_e4m3fn_encode(tl.load(X + i, i < N, 0)), i < N)


@triton.jit
def _decode(X, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(Y + i, fp8_e4m3fn_decode(tl.load(X + i, i < N, 0)), i < N)


@unittest.skipUnless(os.environ.get("CUBIC_TEST_QUANT_CUDA") == "1", "explicit CUDA audit required")
class PortableFP8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_reference_math()

    def test_all_encodings_and_signed_zero(self):
        codes = torch.arange(256, dtype=torch.uint8)
        expected = codes.view(torch.float8_e4m3fn).half()
        actual = torch.empty(256, dtype=torch.float16, device="cuda")
        _decode[(1,)](codes.cuda(), actual, 256, 256)
        actual = actual.cpu()
        finite = expected.isfinite()
        self.assertTrue(torch.equal(actual.isnan(), expected.isnan()))
        self.assertTrue(
            torch.equal(actual[finite].view(torch.uint8), expected[finite].view(torch.uint8))
        )

    def test_all_low_precision_inputs_and_fp32_rounding_boundaries(self):
        bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
        positive = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        mid = (positive[:-1] + positive[1:]) / 2
        edges = torch.cat(
            (
                mid,
                torch.nextafter(mid, torch.full_like(mid, float("inf"))),
                torch.nextafter(mid, torch.full_like(mid, -float("inf"))),
            )
        )
        generator = torch.Generator().manual_seed(921)
        random = (
            torch.randint(-(2**31), 2**31, (200000,), generator=generator).int().view(torch.float32)
        )
        values = torch.cat(
            (
                bits.view(torch.float16).float(),
                bits.view(torch.bfloat16).float(),
                edges,
                -edges,
                random,
            )
        )
        expected = values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        device_values = values.cuda()
        actual = torch.empty_like(device_values, dtype=torch.uint8)
        _encode[(triton.cdiv(values.numel(), 256),)](device_values, actual, values.numel(), 256)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _encode[(triton.cdiv(values.numel(), 256),)](device_values, actual, values.numel(), 256)
        graph.replay()
        actual = actual.cpu()
        finite_input = ~values.isnan()
        self.assertTrue(torch.equal(actual[finite_input], expected[finite_input]))
        self.assertTrue(torch.all((actual[~finite_input] & 127) == 127).item())
