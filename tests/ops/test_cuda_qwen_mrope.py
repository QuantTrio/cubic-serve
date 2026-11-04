from __future__ import annotations

import os
import unittest
from contextlib import ExitStack
from unittest import mock

import torch
from torch import Tensor

from cubic_serve.ops.cuda.position import QwenMRoPEApplyPrimitive
from cubic_serve.ops.cuda.position import qwen_mrope as implementation

RUN_REAL = os.environ.get("CUBIC_RUN_CUDA_QWEN_MROPE_TESTS") == "1"
_C, _QH, _KVH, _D, _R = 4, 24, 4, 256, 64
_ROWS = {
    1: (-1, 0, -3, -2),
    2: (1, -2, -1, 0),
    4: (2, 0, 3, 1),
}


def _rotate_oracle(source: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Independent scalar PyTorch oracle with FP32 products and a final activation cast."""

    output = source.clone()
    half = cosine.shape[1] // 2
    for row in range(source.shape[0]):
        for head in range(source.shape[1]):
            for column in range(cosine.shape[1]):
                pair = column + half if column < half else column - half
                paired = float(source[row, head, pair])
                rotated = -paired if column < half else paired
                left = torch.tensor(
                    float(source[row, head, column]) * float(cosine[row, column]),
                    dtype=torch.float32,
                )
                right = torch.tensor(rotated * float(sine[row, column]), dtype=torch.float32)
                output[row, head, column] = (left + right).to(source.dtype)
    return output


def _oracle(query: Tensor, key: Tensor, cosine: Tensor, sine: Tensor) -> tuple[Tensor, Tensor]:
    return _rotate_oracle(query, cosine, sine), _rotate_oracle(key, cosine, sine)


def _addresses(tensors: tuple[Tensor, ...]) -> tuple[int, ...]:
    return tuple(tensor.data_ptr() for tensor in tensors)


@unittest.skipUnless(RUN_REAL, "set CUBIC_RUN_CUDA_QWEN_MROPE_TESTS=1 for real CUDA qualification")
class QwenMRoPEApplyTests(unittest.TestCase):
    def test_warmed_m4_graph_is_bit_exact_for_m1_m2_m4_and_padding(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            self._qualify_dtype(dtype)

    def _qualify_dtype(self, dtype: torch.dtype) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        if torch.cuda.get_device_capability(device) != (12, 0):
            self.skipTest("this qualification is pinned to SM120")

        torch.manual_seed(3508)
        logical_query = (torch.randn((_C, _QH, _D)) * 1.7).to(dtype)
        logical_key = (torch.randn((_C, _KVH, _D)) * 1.9).to(dtype)
        angles = torch.randn((_C, _R)) * 2.0
        logical_cosine, logical_sine = angles.cos().to(dtype), angles.sin().to(dtype)
        oracle_query, _ = _oracle(logical_query, logical_key, logical_cosine, logical_sine)
        rotated = torch.cat(
            (-logical_query[..., _R // 2 : _R], logical_query[..., : _R // 2]), dim=-1
        )
        fused = (
            logical_query[..., :_R].float() * logical_cosine[:, None].float()
            + rotated.float() * logical_sine[:, None].float()
        ).to(dtype)
        self.assertTrue(torch.equal(fused, oracle_query[..., :_R]))
        step_rounded = (
            logical_query[..., :_R] * logical_cosine[:, None] + rotated * logical_sine[:, None]
        )
        self.assertFalse(torch.equal(step_rounded, oracle_query[..., :_R]))

        query = torch.empty((_C, _QH, _D), device=device, dtype=dtype)
        key = torch.empty((_C, _KVH, _D), device=device, dtype=dtype)
        cosine = torch.empty((_C, _R), device=device, dtype=dtype)
        sine = torch.empty_like(cosine)
        out_query, out_key = torch.empty_like(query), torch.empty_like(key)
        primitive = QwenMRoPEApplyPrimitive(query, key, cosine, sine, out_query, out_key)
        self.assertIn(
            "qh=24,kvh=4,d=256,r=64,fp32-rotation,"
            + ("bf16" if dtype is torch.bfloat16 else "fp16")
            + "-output",
            primitive.launch_signature,
        )
        self.assertFalse(any(isinstance(value, Tensor) for value in vars(primitive).values()))
        self.assertNotEqual(_QH, _KVH)

        tensors = query, key, cosine, sine, out_query, out_key
        addresses = _addresses(tensors)

        def fill(logical_m: int, poison: float) -> None:
            for physical, logical in enumerate(_ROWS[logical_m]):
                if logical >= 0:
                    query[physical].copy_(logical_query[logical])
                    key[physical].copy_(logical_key[logical])
                    cosine[physical].copy_(logical_cosine[logical])
                    sine[physical].copy_(logical_sine[logical])
                else:
                    query[physical].fill_(poison + physical / 16.0)
                    key[physical].fill_(-poison - physical / 16.0)
                    cosine[physical].fill_(poison / 128.0)
                    sine[physical].fill_(-poison / 192.0)
            out_query.fill_(float("nan"))
            out_key.fill_(float("nan"))

        def launch() -> None:
            primitive.launch(query, key, cosine, sine, out_query, out_key)

        fill(4, 37.0)
        stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream(device))
        # GraphProgram qualification owns this fixed-specialization warmup before seal.
        with torch.cuda.device(device), torch.cuda.stream(stream):
            launch()
            launch()
        stream.synchronize()

        def forbidden(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("sealed primitive launch performed a forbidden operation")

        with ExitStack() as stack:
            for owner, names in (
                (torch, ("empty", "empty_like", "zeros", "ones", "full", "tensor", "compile")),
                (Tensor, ("clone", "item", "new_empty", "new_zeros", "new_full")),
                (torch.cuda, ("synchronize",)),
                (torch.cuda.Stream, ("synchronize",)),
                (type(implementation._apply), ("_do_compile",)),
            ):
                for name in names:
                    stack.enter_context(mock.patch.object(owner, name, forbidden))
            with torch.cuda.device(device), torch.cuda.stream(stream):
                launch()
        stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launch()
        stream.synchronize()

        by_m: dict[int, tuple[Tensor, Tensor]] = {}
        for logical_m in (1, 2, 4):
            poison_results: list[tuple[Tensor, Tensor]] = []
            for poison in (37.0, -73.0):
                fill(logical_m, poison)
                expected_query, expected_key = _oracle(
                    query.cpu(), key.cpu(), cosine.cpu(), sine.cpu()
                )
                graph.replay()
                torch.cuda.synchronize(device)
                self.assertTrue(torch.equal(out_query, expected_query.to(device)))
                self.assertTrue(torch.equal(out_key, expected_key.to(device)))
                self.assertTrue(torch.equal(out_query[..., _R:], query[..., _R:]))
                self.assertTrue(torch.equal(out_key[..., _R:], key[..., _R:]))
                rows = [_ROWS[logical_m].index(logical) for logical in range(logical_m)]
                poison_results.append((out_query[rows].clone(), out_key[rows].clone()))
            self.assertTrue(torch.equal(poison_results[0][0], poison_results[1][0]))
            self.assertTrue(torch.equal(poison_results[0][1], poison_results[1][1]))
            by_m[logical_m] = poison_results[0]

        for logical_m in (1, 2):
            self.assertTrue(torch.equal(by_m[logical_m][0], by_m[4][0][:logical_m]))
            self.assertTrue(torch.equal(by_m[logical_m][1], by_m[4][1][:logical_m]))
        self.assertEqual(addresses, _addresses(tensors))


if __name__ == "__main__":
    unittest.main()
