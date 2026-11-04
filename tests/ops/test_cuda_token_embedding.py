from __future__ import annotations

import os
import unittest
from contextlib import ExitStack
from unittest import mock

import torch
from torch import Tensor

from cubic_serve.ops.cuda.embedding import BF16TokenEmbeddingPrimitive
from cubic_serve.ops.cuda.embedding import token as implementation

RUN_REAL = os.environ.get("CUBIC_RUN_CUDA_EMBEDDING_TESTS") == "1"
_C, _V, _H = 4, 257, 5120
_ROWS = {
    1: (-1, 0, -3, -2),
    2: (1, -2, -1, 0),
    4: (2, 0, 3, 1),
}


@unittest.skipUnless(RUN_REAL, "set CUBIC_RUN_CUDA_EMBEDDING_TESTS=1 for real CUDA")
class BF16TokenEmbeddingTests(unittest.TestCase):
    def test_warmed_m4_graph_is_exact_for_m1_m2_m4_and_padding(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        if torch.cuda.get_device_capability(device) != (12, 0):
            self.skipTest("this qualification is pinned to SM120")

        torch.manual_seed(3511)
        logical_ids = torch.tensor([3, 29, 71, 191], device=device)
        token_ids = torch.empty((_C,), device=device, dtype=torch.int64)
        weight = torch.randn((_V, _H), device=device, dtype=torch.bfloat16)
        output = torch.empty((_C, _H), device=device, dtype=torch.bfloat16)
        primitive = BF16TokenEmbeddingPrimitive(token_ids, weight, output)
        self.assertIn("c=4,v=257,h=5120", primitive.launch_signature)
        self.assertFalse(any(isinstance(value, Tensor) for value in vars(primitive).values()))
        addresses = tuple(tensor.data_ptr() for tensor in (token_ids, weight, output))

        def fill(logical_m: int, poison_id: int) -> None:
            for physical, logical in enumerate(_ROWS[logical_m]):
                token_ids[physical] = logical_ids[logical] if logical >= 0 else poison_id + physical
            output.fill_(float("nan"))

        def launch() -> None:
            primitive.launch(token_ids, weight, output)

        fill(4, 241)
        stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.device(device), torch.cuda.stream(stream):
            launch()
            launch()
        stream.synchronize()

        def forbidden(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("sealed embedding launch performed a forbidden operation")

        with ExitStack() as stack:
            for owner, names in (
                (torch, ("empty", "empty_like", "zeros", "ones", "full", "tensor", "compile")),
                (Tensor, ("clone", "item", "new_empty", "new_zeros", "new_full")),
                (torch.cuda, ("synchronize",)),
                (torch.cuda.Stream, ("synchronize",)),
                (type(implementation._gather), ("_do_compile",)),
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

        by_m: dict[int, Tensor] = {}
        for logical_m in (1, 2, 4):
            poison_results: list[Tensor] = []
            for poison_id in (211, 223):
                fill(logical_m, poison_id)
                expected = weight[token_ids].clone()
                graph.replay()
                torch.cuda.synchronize(device)
                self.assertTrue(torch.equal(output, expected))
                physical_rows = [_ROWS[logical_m].index(logical) for logical in range(logical_m)]
                poison_results.append(output[physical_rows].clone())
            self.assertTrue(torch.equal(poison_results[0], poison_results[1]))
            by_m[logical_m] = poison_results[0]

        for logical_m in (1, 2):
            self.assertTrue(torch.equal(by_m[logical_m], by_m[4][:logical_m]))
        self.assertEqual(
            addresses,
            tuple(tensor.data_ptr() for tensor in (token_ids, weight, output)),
        )
        with self.assertRaises(ValueError):
            primitive.launch(token_ids[:3], weight, output[:3])


if __name__ == "__main__":
    unittest.main()
