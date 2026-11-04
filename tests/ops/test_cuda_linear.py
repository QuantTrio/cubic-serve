from __future__ import annotations

import math
import os
import unittest
from collections.abc import Mapping, MutableMapping
from contextlib import ExitStack
from unittest import mock

import torch
from torch import Tensor

import cubic_serve.ops.cuda.linear as linear_ops
from cubic_serve.contracts.execution import (
    ActiveLaunchRow,
    BatchPhase,
    GraphBucket,
    LaunchPlan,
    LaunchRow,
    PaddingLaunchRow,
    RankReplayResources,
)
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.executor.replay_lease import ReplayLeasePool
from cubic_serve.guardrail.compare import (
    ComparisonPolicy,
    DifferenceClass,
    ReductionOrderProof,
    stable_digest,
)
from cubic_serve.guardrail.evidence import EvidenceProfile, QualificationFingerprint
from cubic_serve.guardrail.graph_padding import (
    GraphPaddingBuffers,
    GraphPaddingCase,
    GraphPaddingHarness,
    GraphPaddingPoison,
    GraphQualificationRun,
    GraphQualificationStateFactory,
)
from cubic_serve.guardrail.harness import QualificationCase, QualificationHarness, TensorRule
from cubic_serve.guardrail.oracle import (
    OracleCallable,
    ReductionWitnessMaterial,
    clone_tensor_map,
    reduction_material_identity,
)
from cubic_serve.ops.cuda.linear.candidate import (
    IMPLEMENTATION_ID,
    REPLAY_ABI_ID,
    BF16LinearGraphCandidate,
    LinearGraphRejected,
)
from cubic_serve.ops.cuda.linear.kernel import BF16_LINEAR_OPERATOR_ID, BF16LinearPrimitive
from cubic_serve.ops.cuda.linear.mmvf import (
    BF16_MMVF_OPERATOR_ID,
    BF16_MMVF_REDUCTION_ABI,
    BF16MMVFPrimitive,
)

RUN_REAL = os.environ.get("CUBIC_RUN_CUDA_LINEAR_TESTS") == "1"


class BF16LinearProductionIdentityTests(unittest.TestCase):
    def test_candidate_identity_is_the_vendor_launch_identity(self) -> None:
        self.assertEqual(IMPLEMENTATION_ID, BF16_LINEAR_OPERATOR_ID)
        self.assertIn("torch-cublas", IMPLEMENTATION_ID)


class BF16MMVFStructureTests(unittest.TestCase):
    def test_mmvf_is_an_explicit_portable_small_m_primitive(self) -> None:
        self.assertIsNot(BF16MMVFPrimitive, BF16LinearPrimitive)
        self.assertEqual(BF16_MMVF_REDUCTION_ABI["m"], "positive")
        self.assertEqual(BF16_MMVF_REDUCTION_ABI["active_rows"], "device-int32-active-prefix")
        self.assertEqual(BF16_MMVF_REDUCTION_ABI["program"], "one-output-channel")
        self.assertEqual(BF16_MMVF_REDUCTION_ABI["block_k"], "next-power-of-two")
        self.assertEqual(BF16_MMVF_REDUCTION_ABI["num_warps"], 8)
        self.assertEqual(BF16_MMVF_REDUCTION_ABI["accumulator"], "fp32-reassociation")
        self.assertEqual(BF16_MMVF_REDUCTION_ABI["output"], "bf16-rne")
        self.assertNotIn("sm120", BF16_MMVF_OPERATOR_ID.lower())
        self.assertNotIn("__dict__", BF16MMVFPrimitive.__slots__)

    def test_small_m_dispatch_is_owned_by_the_linear_operator(self) -> None:
        operands = (mock.sentinel.weight, mock.sentinel.output, mock.sentinel.active)
        with (
            mock.patch.object(
                linear_ops, "BF16MMVFPrimitive", return_value=mock.sentinel.mmvf
            ) as mmvf,
            mock.patch.object(
                linear_ops, "BF16LinearPrimitive", return_value=mock.sentinel.vendor
            ) as vendor,
        ):
            for captured_m in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                x = mock.Mock(shape=(captured_m, 8))
                for reduction in ("canonical", "fixed_tile"):
                    self.assertIs(
                        linear_ops.bind_bf16_linear_kernel(
                            x, *operands, vendor=mock.sentinel.prebound_vendor,
                            reduction=reduction,
                        ),
                        mock.sentinel.mmvf,
                    )
        self.assertEqual(mmvf.call_count, 18)
        vendor.assert_not_called()



@unittest.skipUnless(RUN_REAL, "set CUBIC_RUN_CUDA_LINEAR_TESTS=1 for real CUDA qualification")
class BF16MMVFPrimitiveTests(unittest.TestCase):
    def test_both_reduction_plans_preserve_rows_through_m256(self) -> None:
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(122)
        x_owned = torch.randint(
            -8, 9, (3, 512), device=device, generator=generator
        ).to(torch.bfloat16) / 8
        weight = torch.randint(
            -8, 9, (257, 512), device=device, generator=generator
        ).to(torch.bfloat16) / 8
        oracle = (x_owned.double() @ weight.double().t()).to(torch.bfloat16)
        for reduction in ("canonical", "fixed_tile"):
            for capacity in (4, 8, 16, 32, 64, 128, 256):
                for permuted in (False, True):
                    with self.subTest(reduction=reduction, capacity=capacity, permuted=permuted):
                        active_count = capacity if permuted else 3
                        physical = (capacity - 1, capacity // 2, 0) if permuted else (0, 1, 2)
                        indices = torch.tensor(physical, device=device)
                        x = torch.zeros((capacity, 512), dtype=torch.bfloat16, device=device)
                        x.index_copy_(0, indices, x_owned)
                        x[active_count:].fill_(torch.nan)
                        output = torch.empty((capacity, 257), dtype=torch.bfloat16, device=device)
                        active = torch.tensor([active_count], dtype=torch.int32, device=device)
                        primitive = BF16MMVFPrimitive(
                            x, weight, output, active, reduction=reduction
                        )
                        primitive.launch(x, weight, output, active)
                        self.assertTrue(torch.equal(output.index_select(0, indices), oracle))
                        self.assertFalse(bool(output[active_count:].view(torch.int16).count_nonzero()))

    def test_wide_exponent_rows_keep_the_m1_reduction_across_capacities(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(18)
        for k in (128, 256, 512, 1536, 2048, 3072, 6144):
            weight = torch.randn(
                (4096, k), device=device, dtype=torch.bfloat16, generator=generator
            )
            inputs = torch.randn(
                (2, k), device=device, dtype=torch.bfloat16, generator=generator
            )
            for wide in (False, True):
                if wide:
                    for value in (weight, inputs):
                        exponent = torch.randint(
                            -12, 12, value.shape, device=device, generator=generator
                        )
                        value.mul_(torch.exp2(exponent.float()).to(torch.bfloat16))
                canonical = []
                for row in inputs:
                    x = row[None].clone()
                    output = torch.empty((1, 4096), device=device, dtype=torch.bfloat16)
                    active = torch.ones((1,), device=device, dtype=torch.int32)
                    BF16MMVFPrimitive(x, weight, output, active).launch(x, weight, output, active)
                    canonical.append(output.clone())
                expected = torch.cat(canonical)
                for capacity in (2, 4, 8):
                    x = torch.full(
                        (capacity, k), float("nan"), device=device, dtype=torch.bfloat16
                    )
                    x[:2].copy_(inputs.flip(0))
                    output = torch.empty((capacity, 4096), device=device, dtype=torch.bfloat16)
                    active = torch.tensor([2], device=device, dtype=torch.int32)
                    BF16MMVFPrimitive(x, weight, output, active).launch(x, weight, output, active)
                    self.assertTrue(torch.equal(output[:2], expected.flip(0)), (k, wide, capacity))
                    self.assertTrue(torch.equal(output[2:], torch.zeros_like(output[2:])))

    def test_small_m_rows_are_exact_across_capacity_permutation_and_padding(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(1952)
        # Integer products and sums fit FP32 exactly; BF16 is the sole rounding
        # boundary. This independent oracle requires no error tolerance.
        inputs = torch.randint(-4, 5, (8, 1536), device=device, generator=generator).to(
            torch.bfloat16
        )
        weight = torch.randint(-4, 5, (257, 1536), device=device, generator=generator).to(
            torch.bfloat16
        )
        oracle = (inputs.double() @ weight.double().t()).to(torch.bfloat16)
        for capacity in (1, 2, 3, 4, 5, 8):
            x = inputs[:capacity].clone()
            output = torch.empty((capacity, 257), device=device, dtype=torch.bfloat16)
            active = torch.tensor([capacity], device=device, dtype=torch.int32)
            primitive = BF16MMVFPrimitive(x, weight, output, active)
            for logical_m in range(1, capacity + 1):
                order = torch.arange(logical_m - 1, -1, -1, device=device)
                for poison in (float("nan"), -23.0):
                    x.fill_(poison)
                    x[:logical_m].copy_(inputs.index_select(0, order))
                    active.fill_(logical_m)
                    primitive.launch(x, weight, output, active)
                    self.assertTrue(torch.equal(output[:logical_m], oracle.index_select(0, order)))
                    self.assertTrue(
                        torch.equal(output[logical_m:], torch.zeros_like(output[logical_m:]))
                    )
                    self.assertFalse(bool(torch.signbit(output[logical_m:]).any()))

    def test_physical_m1_graph_matches_vendor(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        torch.manual_seed(1950)
        x = torch.randn((1, 1536), device=device, dtype=torch.bfloat16).mul_(0.02)
        weight = torch.randn((257, 1536), device=device, dtype=torch.bfloat16).mul_(0.02)
        output = torch.empty((1, 257), device=device, dtype=torch.bfloat16)
        active_rows = torch.ones((1,), device=device, dtype=torch.int32)
        primitive = linear_ops.bind_bf16_linear_kernel(x, weight, output, active_rows)
        self.assertIsInstance(primitive, BF16MMVFPrimitive)
        stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            linear_ops.launch_bf16_linear_kernel(
                primitive, x, weight, output, active_rows
            )
        stream.synchronize()
        graph.replay()
        torch.cuda.synchronize(device)
        self.assertEqual(tuple(output.shape), (1, 257))
        self.assertTrue(torch.equal(output, torch.mm(x, weight.t())))

    def test_one_captured_m2_kernel_handles_active_one_and_two(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        torch.manual_seed(1951)
        x = torch.randn((2, 1536), device=device, dtype=torch.bfloat16).mul_(0.02)
        weight = torch.randn((257, 1536), device=device, dtype=torch.bfloat16).mul_(0.02)
        output = torch.empty((2, 257), device=device, dtype=torch.bfloat16)
        active_rows = torch.tensor([2], device=device, dtype=torch.int32)
        primitive = linear_ops.bind_bf16_linear_kernel(x, weight, output, active_rows)
        self.assertIsInstance(primitive, BF16MMVFPrimitive)

        artifact = primitive.project_artifact()
        self.assertEqual(artifact.backend, "triton")
        self.assertEqual(artifact.operator_id, BF16_MMVF_OPERATOR_ID)
        self.assertEqual(artifact.grid, (257, 1, 1))
        self.assertIn("num_warps=8", artifact.launch_signature)
        self.assertIn("block_k=2048", artifact.launch_signature)

        receipt = primitive.launch(x, weight, output, active_rows)
        primitive.validate_receipt(receipt)

        compatible_x = x.clone()
        compatible_weight = weight.clone()
        compatible_output = torch.empty_like(output)
        compatible_receipt = primitive.launch(
            compatible_x, compatible_weight, compatible_output, active_rows
        )
        primitive.validate_receipt(compatible_receipt)
        self.assertTrue(
            torch.equal(compatible_output, torch.mm(compatible_x, compatible_weight.t()))
        )
        with self.assertRaises(ValueError):
            primitive.validate_bound_operands(
                compatible_x, compatible_weight, compatible_output, active_rows
            )

        def forbidden(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("BF16 MMVF launch compiled, allocated, or vendor-dispatched")

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(torch, "mm", forbidden))
            for name in (
                "empty",
                "empty_like",
                "empty_strided",
                "zeros",
                "zeros_like",
                "ones",
                "ones_like",
                "full",
                "full_like",
                "tensor",
                "compile",
            ):
                stack.enter_context(mock.patch.object(torch, name, forbidden))
            primitive.launch(x, weight, output, active_rows)

        stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            primitive.launch(x, weight, output, active_rows)
        stream.synchronize()

        active_rows.fill_(2)
        graph.replay()
        torch.cuda.synchronize(device)
        active_two = output.clone()
        vendor = torch.mm(x, weight.t())
        oracle = torch.mm(x.float(), weight.float().t())
        candidate_error = (active_two.float() - oracle).abs()
        vendor_error = (vendor.float() - oracle).abs()
        rounding_envelope = (
            float(vendor_error.max())
            + torch.finfo(torch.bfloat16).eps * float(oracle.abs().max())
            + 1.0e-6
        )
        self.assertTrue(torch.isfinite(active_two).all())
        self.assertLessEqual(float(candidate_error.max()), rounding_envelope)
        self.assertLessEqual(
            float((active_two.float() - vendor.float()).abs().max()), rounding_envelope
        )

        active_one_rows = []
        for poison in (17.0, -23.0):
            output.fill_(poison)
            active_rows.fill_(1)
            graph.replay()
            torch.cuda.synchronize(device)
            active_one_rows.append(output[0].clone())
            self.assertTrue(torch.equal(output[1], torch.zeros_like(output[1])))
            self.assertFalse(bool(torch.signbit(output[1]).any()))
        self.assertTrue(torch.equal(active_one_rows[0], active_one_rows[1]))
        self.assertTrue(torch.equal(active_one_rows[0], active_two[0]))

        output.fill_(31.0)
        active_rows.fill_(2)
        graph.replay()
        torch.cuda.synchronize(device)
        self.assertTrue(torch.equal(output, active_two))

        output.fill_(-41.0)
        active_rows.fill_(1)
        graph.replay()
        torch.cuda.synchronize(device)
        self.assertTrue(torch.equal(output[0], active_two[0]))
        self.assertTrue(torch.equal(output[1], torch.zeros_like(output[1])))
        self.assertFalse(bool(torch.signbit(output[1]).any()))

        with self.assertRaises(ValueError):
            primitive.validate_bound_operands(x, weight, output, active_rows.clone())

        invalid_active = torch.tensor([0], device=device, dtype=torch.int32)
        with self.assertRaises(ValueError):
            BF16MMVFPrimitive(x, weight, output, invalid_active)


class _LinearQualificationRun:
    def __init__(self, seed: Mapping[str, Tensor]) -> None:
        self.state = clone_tensor_map(seed, "linear qualification state")

    @property
    def production_state(self) -> object:
        return self.state

    def committed_snapshot(self) -> Mapping[str, Tensor]:
        return clone_tensor_map(self.state, "linear qualification snapshot")

    def close(self) -> None:
        return None

    def abort(self) -> None:
        return None


class _LinearQualificationStateFactory:
    state_path_id = "test.linear.no-state.v1"
    state_path_digest = "b2s256:" + "6" * 64

    def create(self, canonical_seed: Mapping[str, Tensor]) -> GraphQualificationRun:
        return _LinearQualificationRun(canonical_seed)


class _LinearAttempt:
    def __init__(
        self,
        candidate: BF16LinearGraphCandidate,
        plan: LaunchPlan,
        pool: ReplayLeasePool,
        factory: GraphQualificationStateFactory,
    ) -> None:
        self.candidate, self._plan, self.pool, self.factory = (
            candidate,
            plan,
            pool,
            factory,
        )

    @property
    def launch_plan(self) -> LaunchPlan:
        return self._plan

    @property
    def state_factory(self) -> GraphQualificationStateFactory:
        return self.factory

    def replay(
        self,
        inputs: Mapping[str, Tensor],
        state: object,
        padding: GraphPaddingBuffers,
    ) -> Mapping[str, Tensor]:
        return self.candidate.replay(self._plan, inputs, _linear_state(state), padding)

    def close(self) -> None:
        self.pool.release(self._plan.replay_lease)


def _linear_state(state: object) -> MutableMapping[str, Tensor]:
    if not isinstance(state, MutableMapping):
        raise TypeError("linear test requires its explicit no-state fixture")
    return state


def _resource() -> RankReplayResources:
    return RankReplayResources(
        rank=0,
        graph_instance_id="bf16-linear:graph:0",
        stream_lane_id="bf16-linear:stream:0",
        buffer_set_id="bf16-linear:buffers:0",
        workspace_arena_id="bf16-linear:workspace:0",
        scratch_arena_id="bf16-linear:scratch:0",
    )


def _rows(logical_m: int) -> tuple[LaunchRow, ...]:
    rows: dict[int, tuple[LaunchRow, ...]] = {
        1: (PaddingLaunchRow(0), ActiveLaunchRow(0), PaddingLaunchRow(2), PaddingLaunchRow(1)),
        2: (ActiveLaunchRow(1), PaddingLaunchRow(1), PaddingLaunchRow(0), ActiveLaunchRow(0)),
        4: (ActiveLaunchRow(2), ActiveLaunchRow(0), ActiveLaunchRow(3), ActiveLaunchRow(1)),
    }
    return rows[logical_m]


def _oracle(weight: Tensor) -> OracleCallable:
    weight_f32 = weight.detach().to(device="cpu", dtype=torch.float32)

    def run(
        inputs: Mapping[str, Tensor], state: MutableMapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        del state
        x = inputs["x"]
        products = x.detach().to(device="cpu", dtype=torch.float32)[:, None, :] * weight_f32
        sums = [math.fsum(row) for row in products.reshape(-1, products.shape[-1]).tolist()]
        output = torch.tensor(sums, dtype=torch.float32).reshape(x.shape[0], weight.shape[0])
        return {"y": output.to(device=x.device, dtype=torch.bfloat16)}

    return run


def _reduction_rule(x: Tensor, weight: Tensor) -> TensorRule:
    products = x.float()[:, None, :] * weight.float()[None, :, :]
    indices = torch.arange(weight.shape[1], device=x.device).expand(products.shape)
    # Tensor-core reduction order is not observable through the launch ABI.  Bind
    # the same logical products and full index coverage on both sides; the FP32
    # error envelope permits reassociation without inventing a hardware tree.
    material = ReductionWitnessMaterial(
        reference_operands=products,
        reference_indices=indices,
        candidate_operands=products,
        candidate_indices=indices,
    )
    identity = reduction_material_identity(material)
    return TensorRule(
        policy=ComparisonPolicy(
            DifferenceClass.REDUCTION_ORDER,
            reduction=ReductionOrderProof(
                operands_digest=identity.operands_digest,
                indices_digest=identity.indices_digest,
                reduction_length=identity.reduction_length,
                accumulator_dtype=ScalarDType.FP32,
            ),
        ),
        material=material,
    )


@unittest.skipUnless(RUN_REAL, "set CUBIC_RUN_CUDA_LINEAR_TESTS=1 for real CUDA qualification")
class BF16LinearGraphCandidateTests(unittest.TestCase):
    def test_caller_owned_primitive_composes_in_outer_multi_m_graph(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        if torch.cuda.get_device_capability(device) != (12, 0):
            self.skipTest("this real qualification is pinned to SM120")

        torch.manual_seed(947)
        x = torch.empty((4, 64), device=device, dtype=torch.bfloat16)
        weight = torch.randn((48, 64), device=device, dtype=torch.bfloat16)
        output = torch.empty((4, 48), device=device, dtype=torch.bfloat16)
        eager_output = torch.empty_like(output)
        primitive = BF16LinearPrimitive(x, weight, output)
        addresses = (x.data_ptr(), weight.data_ptr(), output.data_ptr())
        self.assertFalse(hasattr(primitive, "__dict__"))

        stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.device(device), torch.cuda.stream(stream):
            primitive.launch(x, weight, output)
            primitive.launch(x, weight, output)
        stream.synchronize()

        def forbidden(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("the BF16 linear launch performed a forbidden host operation")

        with ExitStack() as stack:
            for owner, names in (
                (
                    torch,
                    (
                        "empty",
                        "empty_like",
                        "empty_strided",
                        "zeros",
                        "zeros_like",
                        "ones",
                        "ones_like",
                        "full",
                        "full_like",
                        "tensor",
                        "clone",
                        "compile",
                    ),
                ),
                (
                    Tensor,
                    (
                        "clone",
                        "item",
                        "new_empty",
                        "new_empty_strided",
                        "new_zeros",
                        "new_ones",
                        "new_full",
                    ),
                ),
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
            logical_x = torch.randn((logical_m, 64), device=device, dtype=torch.bfloat16)
            active_results: list[Tensor] = []
            for poison in (37.0, -73.0):
                for physical, row in enumerate(_rows(logical_m)):
                    source = (
                        logical_x[row.logical_index]
                        if isinstance(row, ActiveLaunchRow)
                        else torch.full((64,), poison, device=device, dtype=torch.bfloat16)
                    )
                    x[physical].copy_(source)
                graph.replay()
                torch.cuda.synchronize(device)
                primitive.launch(x, weight, eager_output)
                torch.cuda.synchronize(device)
                self.assertTrue(torch.equal(output, eager_output))
                physical_rows = [
                    physical
                    for physical, row in enumerate(_rows(logical_m))
                    if isinstance(row, ActiveLaunchRow)
                ]
                active_results.append(output[physical_rows].clone())
            self.assertTrue(torch.equal(active_results[0], active_results[1]))

        self.assertEqual(addresses, (x.data_ptr(), weight.data_ptr(), output.data_ptr()))
        with self.assertRaises(ValueError):
            primitive.launch(x[:3], weight, output[:3])

    def test_one_graph_qualifies_multiple_logical_m_with_reduction_witness(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        torch.manual_seed(941)
        device = torch.device("cuda:0")
        weight = torch.randn((48, 64), device=device, dtype=torch.bfloat16)
        resource = _resource()
        candidate = BF16LinearGraphCandidate(weight=weight, captured_m=4, resource=resource)
        bucket = GraphBucket(
            bucket_id="bf16-linear:m4",
            template_id="bf16-linear:torch-cublas:m4:n48:k64",
            scope_id="operator.linear",
            replay_abi_id=REPLAY_ABI_ID,
            topology_id="single:rank0",
            partition_id="linear:rank0",
            phase=BatchPhase.MIXED,
            capacity_items=candidate.capacity_items,
            logical_m_values=(1, 2, 4),
            captured_m=4,
            row_map_schema_id="cubic.rows.v1",
            scratch_schema_id="cubic.scratch.bf16-linear.v1",
            graph_safety_contract_ids=("graph-safe:bf16-linear:v1",),
            evidence_id="evidence:qualification:bf16-linear",
        )
        pool = ReplayLeasePool(((resource,),))
        reference_oracle = _oracle(weight)
        launch_serial = 0

        def acquire_launch(logical_m: int, label: str) -> LaunchPlan:
            nonlocal launch_serial
            launch_serial += 1
            launch_id = f"launch:bf16-linear:m{logical_m}:{label}:{launch_serial}"
            lease = pool.acquire(
                launch_plan_id=launch_id,
                execution_plan_id="qualification:bf16-linear",
                bucket_id=bucket.bucket_id,
                template_id=bucket.template_id,
                topology_id=bucket.topology_id,
                partition_id=bucket.partition_id,
            )
            return LaunchPlan(
                launch_plan_id=launch_id,
                execution_plan_id=lease.execution_plan_id,
                source_plan_id=f"qualification-case:m{logical_m}",
                logical_m=logical_m,
                bucket=bucket,
                rows=_rows(logical_m),
                replay_lease=lease,
            )

        candidate.capture()
        for logical_m in bucket.logical_m_values:
            with self.subTest(logical_m=logical_m):
                x = torch.randn((logical_m, 64), device=device, dtype=torch.bfloat16)
                padding_m = bucket.captured_m - logical_m
                poisons = tuple(
                    GraphPaddingPoison(
                        poison_id=poison_id,
                        padding_inputs=(
                            {
                                "x": torch.full(
                                    (padding_m, 64), value, device=device, dtype=torch.bfloat16
                                )
                            }
                            if padding_m
                            else {}
                        ),
                        scratch_state=(
                            {
                                "y": torch.full(
                                    (padding_m, 48), value, device=device, dtype=torch.bfloat16
                                )
                            }
                            if padding_m
                            else {}
                        ),
                        guards={"sentinel": torch.tensor([sentinel], device=device)},
                    )
                    for poison_id, value, sentinel in (
                        ("positive", 37.0, 101),
                        ("negative", -73.0, 211),
                    )
                )
                case = GraphPaddingCase(
                    case_id=f"bf16-linear/m{logical_m}/bucket4",
                    logical_inputs={"x": x},
                    initial_committed_state={
                        "qualification_revision": torch.tensor([0], device=device)
                    },
                    poisons=poisons,
                )

                def fixed_m_eager(
                    inputs: Mapping[str, Tensor],
                    state: MutableMapping[str, Tensor],
                    _logical_m: int = logical_m,
                    _poisons: tuple[GraphPaddingPoison, ...] = poisons,
                ) -> Mapping[str, Tensor]:
                    plan = acquire_launch(_logical_m, "oracle")
                    buffers = GraphPaddingBuffers.from_poison(_poisons[0])
                    try:
                        return candidate.qualification_eager(plan, inputs, state, buffers)
                    finally:
                        pool.release(plan.replay_lease)

                factory = _LinearQualificationStateFactory()

                def attempt_factory(
                    _logical_m: int = logical_m,
                    _factory: _LinearQualificationStateFactory = factory,
                ) -> _LinearAttempt:
                    return _LinearAttempt(
                        candidate,
                        acquire_launch(_logical_m, "poison"),
                        pool,
                        _factory,
                    )

                padding = GraphPaddingHarness(fixed_m_eager).qualify(
                    case=case,
                    bucket=bucket,
                    rows=_rows(logical_m),
                    attempt_factory=attempt_factory,
                )
                self.assertTrue(
                    padding.passed,
                    {check.path: check.result.reason for check in padding.checks},
                )

                def replay_candidate(
                    inputs: Mapping[str, Tensor],
                    state: MutableMapping[str, Tensor],
                    _logical_m: int = logical_m,
                    _poisons: tuple[GraphPaddingPoison, ...] = poisons,
                ) -> Mapping[str, Tensor]:
                    plan = acquire_launch(_logical_m, "arithmetic")
                    buffers = GraphPaddingBuffers.from_poison(_poisons[0])
                    try:
                        return candidate.replay(plan, inputs, state, buffers)
                    finally:
                        pool.release(plan.replay_lease)

                arithmetic = QualificationHarness(reference_oracle).qualify(
                    profile=EvidenceProfile.FULL,
                    fingerprint=QualificationFingerprint(
                        contract_digest=stable_digest(repr(bucket)),
                        source_digest=stable_digest(IMPLEMENTATION_ID),
                        binary_digest=stable_digest(candidate.launch_signature),
                        op_signature_digest=stable_digest(
                            ("linear", logical_m, 48, 64, "bf16", "fp32")
                        ),
                        implementation_id=IMPLEMENTATION_ID,
                        device_capability=f"sm{torch.cuda.get_device_capability(device)[0]}"
                        f"{torch.cuda.get_device_capability(device)[1]}",
                        driver_version="qualified-host",
                        runtime_version=f"torch:{torch.__version__}/cuda:{torch.version.cuda}",
                        topology_digest=stable_digest(bucket.topology_id),
                    ),
                    case=QualificationCase(
                        case_id=f"bf16-linear/m{logical_m}",
                        semantic_op="linear",
                        inputs={"x": x},
                        initial_state=case.initial_committed_state,
                        rules={"output:y": _reduction_rule(x, weight)},
                    ),
                    candidate=replay_candidate,
                    candidate_id=IMPLEMENTATION_ID,
                    launch_signature=candidate.launch_signature,
                )
                self.assertTrue(
                    arithmetic.passed,
                    {record.case_id: record.result.reason for record in arithmetic.records},
                )
                output_record = next(
                    record for record in arithmetic.records if record.case_id.endswith("output:y")
                )
                self.assertIs(
                    output_record.result.difference_class,
                    DifferenceClass.REDUCTION_ORDER,
                )
                self.assertIsNotNone(output_record.result.witness_digest)

        self.assertEqual(pool.in_flight, 0)
        with self.assertRaises(LinearGraphRejected):
            candidate.capture()


if __name__ == "__main__":
    unittest.main()
