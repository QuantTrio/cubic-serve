from __future__ import annotations

import os
import unittest
from functools import partial
from hashlib import blake2s as content_digest
from pathlib import Path

import torch
from torch import Tensor

from cubic_serve.contracts.model import ScalarDType
from cubic_serve.executor.runner import configure_reference_math
from cubic_serve.guardrail.compare import (
    ComparisonContext,
    ComparisonPolicy,
    DifferenceClass,
    ReductionOrderProof,
    stable_digest,
    tensor_digest,
)
from cubic_serve.guardrail.oracle import (
    ReductionWitnessMaterial,
    reduction_material_identity,
    verify_observed_bf16_linear_sample,
    verify_reduction_witness,
)
from cubic_serve.ops.cuda.moe import BF16RoutedExpertsPrimitive
from cubic_serve.ops.cuda.moe import fused as fused_moe
from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope


def _witness_linear_differences(
    *,
    x: Tensor,
    weight: Tensor,
    candidate: Tensor,
    oracle: Tensor,
    primitive: BF16RoutedExpertsPrimitive,
    stage: str,
) -> int:
    """Bind every unequal output column to complete-K candidate and eager witnesses."""

    columns = tuple(
        int(index)
        for index in torch.nonzero((candidate != oracle).any(dim=0), as_tuple=False)
        .flatten()
        .tolist()
    )
    if not columns:
        return 0
    source_digest = "b2s256:" + content_digest(Path(fused_moe.__file__).read_bytes()).hexdigest()
    maximum_products = 1 << 20
    chunk_size = max(1, maximum_products // (x.shape[0] * x.shape[1]))
    for start in range(0, len(columns), chunk_size):
        selected = columns[start : start + chunk_size]
        common = {
            "x": x,
            "weight": weight,
            "input_rows": tuple(range(x.shape[0])),
            "output_columns": selected,
        }
        try:
            verify_observed_bf16_linear_sample(
                **common,
                output=candidate,
                operand_binding_digest=stable_digest(
                    {
                        "stage": stage,
                        "input": tensor_digest(x),
                        "weight": tensor_digest(weight),
                    }
                ),
                artifact_digest=stable_digest(
                    {
                        "source": source_digest,
                        "launch": primitive.launch_signature,
                        "stage": stage,
                    }
                ),
                launch_specialization_digest=stable_digest(
                    {
                        "launch": primitive.launch_signature,
                        "stage": stage,
                        "shape": tuple(candidate.shape),
                    }
                ),
            )
        except ValueError as error:
            difference = (candidate.float() - oracle.float()).abs()
            raise AssertionError(
                f"{stage} candidate linear witness failed: "
                f"unequal={torch.count_nonzero(candidate != oracle).item()} "
                f"max_abs={float(difference.max())}"
            ) from error
        verify_observed_bf16_linear_sample(
            **common,
            output=oracle,
            operand_binding_digest=stable_digest(
                {
                    "stage": stage,
                    "input": tensor_digest(x),
                    "weight": tensor_digest(weight),
                    "implementation": "independent-eager-mm",
                }
            ),
            artifact_digest=stable_digest("independent-eager-mm"),
            launch_specialization_digest=stable_digest(
                {"shape": tuple(oracle.shape), "reduction": "complete-k"}
            ),
        )
    return len(columns)


def _reference(
    hidden: Tensor,
    router_logits: Tensor,
    gate_up: Tensor,
    down: Tensor,
    *,
    first_expert: int,
    top_k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    selected = torch.argsort(router_logits, dim=-1, descending=True, stable=True)[:, :top_k]
    selected_logits = router_logits.float().gather(1, selected)
    route_weights = torch.softmax(selected_logits, dim=-1).to(torch.bfloat16)
    route_ids, order = torch.sort(selected, dim=-1)
    route_weights = route_weights.gather(1, order)
    routes = torch.zeros(
        (*hidden.shape[:1], top_k, hidden.shape[1]),
        dtype=torch.bfloat16,
        device=hidden.device,
    )
    for local_expert in range(gate_up.shape[0]):
        global_expert = first_expert + local_expert
        matches = route_ids == global_expert
        rows, slots = torch.where(matches)
        if not rows.numel():
            continue
        packed = torch.matmul(hidden[rows], gate_up[local_expert].t())
        gate, up = packed.chunk(2, dim=-1)
        silu = (gate.float() / (1.0 + torch.exp(-gate.float()))).to(torch.bfloat16)
        active = (silu.float() * up.float()).to(torch.bfloat16)
        routes[rows, slots] = torch.matmul(active, down[local_expert].t())
    output = torch.zeros_like(hidden)
    for slot in range(top_k):
        weighted = (routes[:, slot].float() * route_weights[:, slot, None].float()).to(
            torch.bfloat16
        )
        output = (output + weighted).to(torch.bfloat16)
    return output, route_ids, route_weights


@unittest.skipUnless(
    os.environ.get("CUBIC_RUN_CUDA_MOE_AUDIT") == "1",
    "set CUBIC_RUN_CUDA_MOE_AUDIT=1 to run the CUDA MoE audit",
)
class BF16RoutedExpertsAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        # Historical midpoint/reduction corpus qualifies the explicit FP32
        # activation mode. Native low-precision arithmetic has its own audit.
        self.enterContext(operator_strategy_scope(
            OperatorConfig(activation_dtype="fp32"), device="", model=""
        ))

    def test_expert_grids_preserve_grouping_and_padding(self) -> None:
        configure_reference_math()
        device = torch.device("cuda:0")
        for dtype in (torch.bfloat16, torch.float16):
            up = torch.full((17, 256, 128), 1 / 128, device=device, dtype=dtype)
            down = torch.full((17, 128, 128), 1 / 128, device=device, dtype=dtype)
            down[1].mul_(2)
            for capacity in (1, 3, 8, 17, 33):
                logical = max(1, capacity - 2)
                hidden = torch.full((capacity, 128), 1 / 8, device=device, dtype=dtype)
                logits = torch.full((capacity, 65), -100, device=device, dtype=dtype)
                logits[:, 16], logits[:, 17], logits[:, 64] = 3, 2, 1
                activity = torch.zeros(capacity, device=device, dtype=torch.int32)
                activity[:logical] = 1
                active = torch.tensor([logical], device=device, dtype=torch.int32)
                output = torch.empty_like(hidden)
                reference = None
                for percent, compact_threshold in ((0, 64), (100000, 64), (0, 1)):
                    config = OperatorConfig.from_dict({
                        "version": 1, "activation_dtype": "fp32",
                        "overrides": {"launch": {"max_candidates": 1}, "moe": {
                            "compact_min_routes_per_expert": compact_threshold,
                            "sparse_route_grid_percent": percent,
                        }},
                    })
                    with operator_strategy_scope(config, device="", model=""):
                        op = BF16RoutedExpertsPrimitive(
                            hidden, logits, activity, active, up, down, output,
                            first_expert=16, top_k=3,
                        )
                    launch = partial(op.launch, hidden, logits, activity, active, up, down, output)
                    launch()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        launch()
                    op.packed.fill_(91)
                    op.routes.fill_(73)
                    hidden[logical:] = 37
                    logits[logical:] = -37
                    graph.replay()
                    torch.cuda.synchronize()
                    snapshot = tuple(t.clone() for t in (
                        output, op.packed, op.routes, op.route_ids, op.route_weights,
                    ))
                    if reference is None:
                        reference = snapshot
                    else:
                        self.assertTrue(all(
                            torch.equal(a, b) for a, b in zip(reference, snapshot, strict=True)
                        ))
                    # Independent dyadic matmuls retain each storage boundary;
                    # the nonlocal third route must contribute nothing.
                    packed = (hidden[:logical].float() @ up[0].float().T).to(dtype)
                    gate, value = packed.chunk(2, dim=-1)
                    activated = (torch.nn.functional.silu(gate.float()).to(dtype) * value).to(dtype)
                    weights = torch.softmax(torch.tensor([3., 2., 1.], device=device), 0).to(dtype)
                    expected = torch.zeros_like(output[:logical])
                    for expert in (0, 1):
                        route = (activated.float() @ down[expert].float().T).to(dtype)
                        product = (route.float() * weights[expert].float()).to(dtype)
                        expected = (expected.float() + product.float()).to(dtype)
                    self.assertTrue(torch.equal(output[:logical], expected))
                    self.assertTrue(torch.equal(
                        output[logical:], torch.zeros_like(output[logical:]),
                    ))
                    graph.reset()

    def test_fp16_expert_geometry_preserves_independent_reference_and_padding(self) -> None:
        from cubic_serve.ops.cuda.moe import RoutedExpertsPrimitive

        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        device = torch.device("cuda:0")
        weights = torch.full((4, 256, 128), 1 / 32, dtype=torch.float16, device=device)
        down = torch.full((4, 128, 128), 1 / 32, dtype=torch.float16, device=device)
        reference_rows = None
        for capacity in (33, 64, 128):
            logical = 33
            hidden = torch.full((capacity, 128), 1 / 16, dtype=torch.float16, device=device)
            router = torch.zeros((capacity, 4), dtype=hidden.dtype, device=device)
            activity = torch.zeros(capacity, dtype=torch.int32, device=device)
            activity[:logical] = 1
            active = torch.tensor([logical], dtype=torch.int32, device=device)
            output = torch.empty_like(hidden)
            primitive = RoutedExpertsPrimitive(hidden, router, activity, active, weights, down,
                                               output, first_expert=0, top_k=2)
            primitive.launch(hidden, router, activity, active, weights, down, output)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                primitive.launch(hidden, router, activity, active, weights, down, output)
            hidden[logical:] = 73
            router[logical:] = -37
            primitive.routes.fill_(91)
            graph.replay()
            torch.cuda.synchronize()
            # Dyadic inputs isolate placement and dtype from dot-order allowances.
            up = (hidden[:logical].float() @ weights[0].float().T).half()
            gate, value = up.chunk(2, dim=-1)
            activated = (torch.nn.functional.silu(gate.float()).half() * value).half()
            expert = (activated.float() @ down[0].float().T).half()
            weighted = (expert * 0.5).half()
            expected = (weighted + weighted).half()
            self.assertTrue(torch.equal(output[:logical], expected))
            self.assertTrue(torch.equal(output[logical:], torch.zeros_like(output[logical:])))
            self.assertEqual(primitive.packed.dtype, torch.float16)
            if reference_rows is not None:
                self.assertTrue(torch.equal(output[:logical], reference_rows))
            reference_rows = output[:logical].clone()
            graph.reset()

    def test_route_softmax_preserves_bf16_midpoint_under_padding(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        configure_reference_math()
        device = torch.device("cuda:0")
        # Actual C33 layer-18 top logits: fast exp changes the second weight.
        scores = torch.tensor(
            [-3.484375, -3.640625, -3.703125, -3.984375, -4.0, -4.34375, -4.375, -4.4375],
            device=device,
        )
        experts = torch.tensor([15, 2, 9, 0, 12, 1, 7, 4], device=device)
        order = torch.argsort(experts)
        expected_ids = experts[order]
        expected_weights = torch.softmax(scores, dim=0)[order].bfloat16()
        for capacity in (64, 128):
            hidden = torch.zeros((capacity, 128), device=device, dtype=torch.bfloat16)
            router = torch.full((capacity, 16), -100.0, device=device, dtype=torch.bfloat16)
            router[:, experts] = scores.bfloat16()
            gate_up = torch.zeros((4, 256, 128), device=device, dtype=torch.bfloat16)
            down = torch.zeros((4, 128, 128), device=device, dtype=torch.bfloat16)
            activity = torch.ones(capacity, device=device, dtype=torch.int32)
            active_rows = torch.tensor([capacity], device=device, dtype=torch.int32)
            output = torch.empty_like(hidden)
            primitive = BF16RoutedExpertsPrimitive(
                hidden,
                router,
                activity,
                active_rows,
                gate_up,
                down,
                output,
                first_expert=5,
                top_k=8,
            )
            primitive.launch(hidden, router, activity, active_rows, gate_up, down, output)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                primitive.launch(hidden, router, activity, active_rows, gate_up, down, output)
            for logical in (8, 32, 33):
                activity.zero_()
                activity[:logical].fill_(1)
                active_rows.fill_(logical)
                router.fill_(-100.0)
                router[:logical, experts] = scores.bfloat16()
                router[logical:].fill_(73.0)
                graph.replay()
                torch.cuda.synchronize(device)
                self.assertTrue(
                    torch.equal(primitive.route_ids[:logical], expected_ids.expand(logical, -1))
                )
                self.assertTrue(
                    torch.equal(
                        primitive.route_weights[:logical], expected_weights.expand(logical, -1)
                    )
                )
                self.assertTrue(bool((primitive.route_ids[logical:] == -1).all()))
                self.assertTrue(bool((primitive.route_weights[logical:] == 0).all()))

    def test_combine_preserves_bf16_boundaries_with_unowned_and_padded_rows(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        configure_reference_math()
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(9221)
        capacity, width, intermediate = 64, 128, 128

        def random(*shape: int) -> Tensor:
            return torch.randn(shape, generator=generator, device=device).bfloat16()

        hidden = random(capacity, width)
        router = random(capacity, 16)
        gate_up = random(4, 2 * intermediate, width) * 0.05
        down = random(4, width, intermediate) * 0.05
        activity = torch.zeros(capacity, dtype=torch.int32, device=device)
        active_rows = torch.zeros(1, dtype=torch.int32, device=device)
        routed = torch.empty_like(hidden)
        primitive = BF16RoutedExpertsPrimitive(
            hidden,
            router,
            activity,
            active_rows,
            gate_up,
            down,
            routed,
            first_expert=5,
            top_k=8,
        )
        shared, gate = random(capacity, width), random(capacity, 1)
        combined = shared.clone()
        combine = fused_moe.BF16MoeSharedCombinePrimitive(combined, gate, routed, combined)
        primitive.launch(hidden, router, activity, active_rows, gate_up, down, routed)
        combine.launch(combined, gate, routed, combined)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            primitive.launch(hidden, router, activity, active_rows, gate_up, down, routed)
            combine.launch(combined, gate, routed, combined)
        for logical in (8, 32, 33):
            activity.zero_()
            activity[:logical].fill_(1)
            active_rows.fill_(logical)
            for poison in (37.0, -73.0):
                # Unowned route scratch is deliberately stale, not zero-filled.
                primitive.routes.fill_(poison)
                combined.copy_(shared)
                graph.replay()
                torch.cuda.synchronize(device)
                expected = torch.zeros_like(routed)
                for slot in range(8):
                    ids = primitive.route_ids[:, slot]
                    owned = (ids >= 5) & (ids < 9)
                    value = torch.where(owned[:, None], primitive.routes[:, slot], 0)
                    weighted = (
                        value.float() * primitive.route_weights[:, slot, None].float()
                    ).bfloat16()
                    expected = (expected.float() + weighted.float()).bfloat16()
                self.assertTrue(torch.equal(routed, expected), f"routed M={logical}")
                self.assertTrue(torch.equal(routed[logical:], torch.zeros_like(routed[logical:])))
                gated = (shared.float() * torch.sigmoid(gate).bfloat16().float()).bfloat16()
                expected = (routed.float() + gated.float()).bfloat16()
                self.assertTrue(torch.equal(combined, expected), f"shared M={logical}")

    def test_real_k_dimensions_routes_and_graph_match_reference(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        configure_reference_math()
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        generator = torch.Generator(device=device).manual_seed(731)
        global_experts, local_experts, first_expert, top_k = 16, 4, 5, 8
        hidden_size, intermediate = 3072, 1024
        for rows in (1, 8, 128):
            hidden = (
                torch.randn(
                    (rows, hidden_size), generator=generator, dtype=torch.float32, device=device
                )
                * 0.05
            ).to(torch.bfloat16)
            router_logits = (
                torch.randn(
                    (rows, global_experts),
                    generator=generator,
                    dtype=torch.float32,
                    device=device,
                )
                * 0.5
            ).to(torch.bfloat16)
            if rows == 128:
                router_logits.fill_(-10)
                fillers = (0, 1, 2, 3, 4, 9, 10, 11)
                for row in range(rows):
                    selected = [first_expert]
                    if row < 16:
                        selected.append(first_expert + 1)
                    if row < 8:
                        selected.append(first_expert + 2)
                    selected.extend(expert for expert in fillers if len(selected) < top_k)
                    router_logits[row, selected] = torch.arange(
                        top_k, 0, -1, dtype=torch.bfloat16, device=device
                    )
            else:
                router_logits[0].zero_()
            gate_up = (
                torch.randn(
                    (local_experts, 2 * intermediate, hidden_size),
                    generator=generator,
                    dtype=torch.float32,
                    device=device,
                )
                * 0.01
            ).to(torch.bfloat16)
            down = (
                torch.randn(
                    (local_experts, hidden_size, intermediate),
                    generator=generator,
                    dtype=torch.float32,
                    device=device,
                )
                * 0.01
            ).to(torch.bfloat16)
            actual = torch.empty_like(hidden)
            row_activity = torch.ones((rows,), dtype=torch.int32, device=device)
            active_rows = torch.tensor([rows], dtype=torch.int32, device=device)
            primitive = BF16RoutedExpertsPrimitive(
                hidden,
                router_logits,
                row_activity,
                active_rows,
                gate_up,
                down,
                actual,
                first_expert=first_expert,
                top_k=top_k,
            )
            expected, expected_ids, expected_weights = _reference(
                hidden,
                router_logits,
                gate_up,
                down,
                first_expert=first_expert,
                top_k=top_k,
            )
            observations: dict[str, Tensor] = {}
            primitive.launch_qualification(
                hidden,
                router_logits,
                row_activity,
                active_rows,
                gate_up,
                down,
                actual,
                lambda phase, value, observations=observations: observations.__setitem__(
                    phase, value.detach().clone()
                ),
            )
            torch.cuda.synchronize(device)
            eager = actual.clone()
            if rows == 128:
                self.assertEqual(primitive.route_counts.cpu().tolist(), [128, 16, 8, 0])
            primitive.launch(
                hidden,
                router_logits,
                row_activity,
                active_rows,
                gate_up,
                down,
                actual,
            )
            torch.cuda.synchronize(device)
            self.assertTrue(torch.equal(actual, eager), f"M={rows}: eager replay is not exact")
            self.assertTrue(
                torch.equal(primitive.route_ids, expected_ids),
                f"M={rows}: stable route ids differ",
            )
            self.assertTrue(
                torch.equal(primitive.route_weights, expected_weights),
                f"M={rows}: BF16 route weights differ",
            )
            self.assertTrue(torch.equal(observations["route_ids"], expected_ids))
            self.assertTrue(torch.equal(observations["route_weights"], expected_weights))
            for local_expert in range(local_experts):
                global_expert = first_expert + local_expert
                selected_rows, selected_slots = torch.where(expected_ids == global_expert)
                if not selected_rows.numel():
                    continue
                selected_hidden = hidden.index_select(0, selected_rows)
                candidate_gate_up = observations["gate_up"][selected_rows, selected_slots]
                eager_gate_up = torch.matmul(selected_hidden, gate_up[local_expert].t())
                _witness_linear_differences(
                    x=selected_hidden,
                    weight=gate_up[local_expert],
                    candidate=candidate_gate_up,
                    oracle=eager_gate_up,
                    primitive=primitive,
                    stage=f"gate_up:expert={global_expert}:M={rows}",
                )
                gate, up = candidate_gate_up.chunk(2, dim=-1)
                candidate_silu = (gate.float() / (1.0 + torch.exp(-gate.float()))).to(
                    torch.bfloat16
                )
                expected_active = (candidate_silu.float() * up.float()).to(torch.bfloat16)
                candidate_active = observations["active"][selected_rows, selected_slots]
                self.assertTrue(
                    torch.equal(candidate_active, expected_active),
                    f"M={rows}: expert {global_expert} SwiGLU boundary differs",
                )
                candidate_routes = observations["routes"][selected_rows, selected_slots]
                eager_routes = torch.matmul(candidate_active, down[local_expert].t())
                _witness_linear_differences(
                    x=candidate_active,
                    weight=down[local_expert],
                    candidate=candidate_routes,
                    oracle=eager_routes,
                    primitive=primitive,
                    stage=f"down:expert={global_expert}:M={rows}",
                )
            coherent_output = torch.zeros_like(hidden)
            combine_operands: list[Tensor] = []
            for slot in range(top_k):
                weighted = (
                    observations["routes"][:, slot].float()
                    * observations["route_weights"][:, slot, None].float()
                ).to(torch.bfloat16)
                coherent_output = (coherent_output + weighted).to(torch.bfloat16)
                combine_operands.append(weighted)
            if not torch.equal(observations["output"], coherent_output):
                operands = torch.stack(combine_operands, dim=-1)
                indices = torch.arange(top_k, dtype=torch.int64, device=device).expand_as(operands)
                material = ReductionWitnessMaterial(operands, indices, operands, indices)
                identity = reduction_material_identity(material)
                verify_reduction_witness(
                    coherent_output,
                    observations["output"],
                    ComparisonPolicy(
                        DifferenceClass.REDUCTION_ORDER,
                        reduction=ReductionOrderProof(
                            operands_digest=identity.operands_digest,
                            indices_digest=identity.indices_digest,
                            reduction_length=top_k,
                            accumulator_dtype=ScalarDType.BF16,
                            candidate_accumulator_dtype=ScalarDType.BF16,
                        ),
                    ),
                    ComparisonContext(
                        stable_digest({"stage": "route_combine", "M": rows}),
                        stable_digest(
                            {
                                "source": "b2s256:"
                                + content_digest(Path(fused_moe.__file__).read_bytes()).hexdigest(),
                                "launch": primitive.launch_signature,
                            }
                        ),
                    ),
                    material,
                )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                primitive.launch(
                    hidden,
                    router_logits,
                    row_activity,
                    active_rows,
                    gate_up,
                    down,
                    actual,
                )
            graph.replay()
            torch.cuda.synchronize(device)
            self.assertTrue(torch.equal(actual, eager), f"M={rows}: graph differs from eager")
            non_owned = ~((expected_ids >= first_expert) & (expected_ids < first_expert + 4))
            self.assertTrue(
                torch.equal(
                    primitive.routes[non_owned],
                    torch.zeros_like(primitive.routes[non_owned]),
                ),
                f"M={rows}: non-owned routes are not positive zero",
            )
            if rows == 128:
                row_activity[1::2].zero_()
                graph.replay()
                torch.cuda.synchronize(device)
                self.assertTrue(
                    torch.equal(actual[::2], eager[::2]),
                    float((actual[::2].float() - eager[::2].float()).abs().max()),
                )
                self.assertTrue(torch.equal(actual[1::2], torch.zeros_like(actual[1::2])))
                self.assertTrue(bool((primitive.route_ids[1::2] == -1).all().item()))

                row_activity.fill_(1)
                active_rows.fill_(64)
                graph.replay()
                torch.cuda.synchronize(device)
                self.assertTrue(torch.equal(actual[:64], eager[:64]))
                self.assertTrue(torch.equal(actual[64:], torch.zeros_like(actual[64:])))
                self.assertTrue(bool((primitive.route_ids[64:] == -1).all().item()))


if __name__ == "__main__":
    unittest.main()
