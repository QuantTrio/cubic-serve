from __future__ import annotations

import os
import unittest
from contextlib import ExitStack
from unittest import mock

import torch
from torch import Tensor

from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
from cubic_serve.ops.cuda.position import (
    MRoPECosSinGatherPrimitive,
    build_bf16_mrope_tables,
)
from cubic_serve.ops.cuda.position import qwen_mrope_table as implementation
from cubic_serve.ops.eager.hybrid import interleaved_mrope_cos_sin

RUN_REAL = os.environ.get("CUBIC_RUN_CUDA_QWEN_MROPE_TABLE_TESTS") == "1"
if RUN_REAL:
    configure_reference_math()
CAPACITY, ROTARY, MAX_POSITION = 4, 64, 256
SECTION, THETA = (11, 11, 10), 10_000_000.0
ROWS = {1: (-1, 0, -3, -2), 2: (1, -2, -1, 0), 4: (2, 0, 3, 1)}
LOGICAL = torch.tensor(((2, 17, 44, 99), (7, 31, 65, 120), (13, 39, 80, 150)), dtype=torch.int64)


def _plan(logical_m: int, poison: int) -> Tensor:
    poison_axes = (
        ((201, 202, 203, 204), (211, 212, 213, 214), (221, 222, 223, 224)),
        ((231, 232, 233, 234), (241, 242, 243, 244), (251, 252, 253, 254)),
    )
    result = torch.empty((3, CAPACITY), dtype=torch.int64)
    for physical, logical in enumerate(ROWS[logical_m]):
        if logical >= 0:
            result[:, physical].copy_(LOGICAL[:, logical])
        else:
            result[:, physical] = torch.tensor(
                tuple(axis[physical] for axis in poison_axes[poison]), dtype=torch.int64
            )
    return result


def _validate_host_plan(plan: Tensor) -> None:
    if plan.shape != (3, CAPACITY) or plan.dtype is not torch.int64:
        raise ValueError("position plan must be host int64 [3,C]")
    if int(plan.min()) < 0 or int(plan.max()) >= MAX_POSITION:
        raise ValueError("position plan is outside the sealed table")


def _addresses(tensors: tuple[Tensor, ...]) -> tuple[int, ...]:
    return tuple(tensor.data_ptr() for tensor in tensors)


@unittest.skipUnless(
    RUN_REAL, "set CUBIC_RUN_CUDA_QWEN_MROPE_TABLE_TESTS=1 for real CUDA qualification"
)
class QwenMRoPETableTests(unittest.TestCase):
    def test_warmed_m4_graph_gathers_canonical_tables_exactly(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        plans = {(m, poison): _plan(m, poison) for m in (1, 2, 4) for poison in (0, 1)}
        for plan in plans.values():
            _validate_host_plan(plan)
        outside = plans[4, 0].clone()
        outside[0, 0] = MAX_POSITION
        with self.assertRaisesRegex(ValueError, "outside the sealed table"):
            _validate_host_plan(outside)
        with self.assertRaisesRegex(ValueError, r"int64 \[3,C\]"):
            _validate_host_plan(plans[4, 0][:2])
        self.assertTrue(torch.all(LOGICAL[0] != LOGICAL[1]))
        self.assertTrue(torch.all(LOGICAL[0] != LOGICAL[2]))
        self.assertTrue(torch.all(LOGICAL[1] != LOGICAL[2]))

        expected: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
        with ReferenceExecutionContext():
            device = torch.device("cuda:0")
            if torch.cuda.get_device_capability(device) != (12, 0):
                self.skipTest("this qualification is pinned to SM120")
            with self.assertRaisesRegex(ValueError, "interleaving"):
                build_bf16_mrope_tables(
                    max_position=MAX_POSITION, device=device, section=(10, 11, 11)
                )
            cosine_table, sine_table, evidence = build_bf16_mrope_tables(
                max_position=MAX_POSITION,
                device=device,
                rotary_dim=ROTARY,
                section=SECTION,
                rope_theta=THETA,
            )
            device_plans = {key: plan.to(device) for key, plan in plans.items()}
            for key, plan in device_plans.items():
                cosine, sine = interleaved_mrope_cos_sin(
                    plan.view(3, CAPACITY, 1),
                    rotary_dim=ROTARY,
                    mrope_section=SECTION,
                    rope_theta=THETA,
                    output_dtype=torch.bfloat16,
                )
                expected[key] = cosine.squeeze(1).cpu(), sine.squeeze(1).cpu()

            position_ids = torch.empty((3, CAPACITY), device=device, dtype=torch.int64)
            out_cosine = torch.empty((CAPACITY, ROTARY), device=device, dtype=torch.bfloat16)
            out_sine = torch.empty_like(out_cosine)

        primitive = MRoPECosSinGatherPrimitive(
            position_ids,
            cosine_table,
            sine_table,
            out_cosine,
            out_sine,
            evidence=evidence,
        )
        self.assertIn("section=(11, 11, 10)", primitive.launch_signature)
        self.assertIn(evidence.table_digest, primitive.launch_signature)
        self.assertFalse(any(isinstance(value, Tensor) for value in vars(primitive).values()))
        corrupted = cosine_table.clone()
        corrupted[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "evidence digest"):
            MRoPECosSinGatherPrimitive(
                position_ids,
                corrupted,
                sine_table,
                out_cosine,
                out_sine,
                evidence=evidence,
            )

        tensors = position_ids, cosine_table, sine_table, out_cosine, out_sine
        addresses = _addresses(tensors)

        def fill(logical_m: int, poison: int) -> None:
            plan = plans[logical_m, poison]
            _validate_host_plan(plan)
            position_ids.copy_(device_plans[logical_m, poison])
            out_cosine.fill_(float("nan"))
            out_sine.fill_(float("nan"))

        def launch() -> None:
            primitive.launch(position_ids, cosine_table, sine_table, out_cosine, out_sine)

        fill(4, 0)
        stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream(device))
        # GraphProgram qualification owns fixed-specialization warmup before seal/capture.
        with torch.cuda.device(device), torch.cuda.stream(stream):
            launch()
            launch()
        stream.synchronize()

        def forbidden(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("sealed gather launch performed a forbidden operation")

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

        by_m: dict[int, tuple[Tensor, Tensor]] = {}
        for logical_m in (1, 2, 4):
            active_results: list[tuple[Tensor, Tensor]] = []
            for poison in (0, 1):
                fill(logical_m, poison)
                graph.replay()
                torch.cuda.synchronize(device)
                want_cosine, want_sine = expected[logical_m, poison]
                self.assertTrue(torch.equal(out_cosine.cpu(), want_cosine))
                self.assertTrue(torch.equal(out_sine.cpu(), want_sine))
                rows = [ROWS[logical_m].index(logical) for logical in range(logical_m)]
                active_results.append((out_cosine[rows].cpu(), out_sine[rows].cpu()))
            self.assertTrue(torch.equal(active_results[0][0], active_results[1][0]))
            self.assertTrue(torch.equal(active_results[0][1], active_results[1][1]))
            by_m[logical_m] = active_results[0]

        for logical_m in (1, 2):
            self.assertTrue(torch.equal(by_m[logical_m][0], by_m[4][0][:logical_m]))
            self.assertTrue(torch.equal(by_m[logical_m][1], by_m[4][1][:logical_m]))
        self.assertEqual(addresses, _addresses(tensors))


if __name__ == "__main__":
    unittest.main()
