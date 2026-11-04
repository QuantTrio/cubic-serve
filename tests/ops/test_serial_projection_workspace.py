"""Serial routed projection scratch: independent math and poisoned graph replay."""

import os
from unittest.mock import patch

import pytest
import torch

from cubic_serve.executor.runner import configure_reference_math
from cubic_serve.executor.text_graph import TextGraphBindings
from cubic_serve.guardrail.quantization import check_linear_enclosure
from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
from cubic_serve.ops.cuda.moe.quantized import PackedRoutedExpertsPrimitive
from tests.ops.test_int_quantization import fixture


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1" or not torch.cuda.is_available(),
    reason="explicit CUDA audit required",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bits", [4, 5, 8])
@pytest.mark.parametrize("first_expert", [0, 1])
def test_serial_projection_workspace(dtype, bits, first_expert):
    from tests.ops.test_cuda_quantized_moe import PackedMoETests

    configure_reference_math()
    torch.set_num_threads(1)
    torch.manual_seed(109)
    device = torch.device("cuda", 0)
    h, i, e = 512, 256, 2
    k = e + first_expert
    weights = [
        tuple(fixture(bits, n=n, k=width, group=128)[0] for _ in range(e))
        for n, width in ((i, h), (i, h), (h, i))
    ]
    for group in weights:
        group[1].payload["weight_scale"].mul_(1.25)
    corpus = torch.randn(3, h, device=device, dtype=dtype) * 0.05
    route_corpus = torch.randn(3, k, device=device, dtype=dtype)
    logical_outputs = {}
    for capacity in (4, 16):
        x = torch.full((capacity, h), float("nan"), device=device, dtype=dtype)
        logits = torch.full((capacity, k), float("nan"), device=device, dtype=dtype)
        rows = torch.tensor([capacity - 1, 0, capacity // 2], device=device)
        x[rows] = corpus
        other = x.clone()
        other[rows] *= -0.25
        logits[rows] = route_corpus
        activity = torch.zeros(capacity, device=device, dtype=torch.int32)
        activity[rows] = 1
        active = torch.tensor([capacity], device=device, dtype=torch.int32)
        operators = PackedLinearOperators()
        helper = TextGraphBindings(operators, device, dtype)
        observed = []

        def workspace(t, helper=helper, observed=observed):
            shared = helper.buffer("moe.projection.partial", *t.shape, dtype=t.dtype)
            observed.append(shared)
            return shared

        def activations(role, shape, dtype, helper=helper):
            return helper.buffer("moe.activation." + role, *shape, dtype=dtype)

        with patch(
            "cubic_serve.quantization.extensions.int.marlin.plan_dense_partitions", return_value=2
        ):
            blocks = [
                PackedRoutedExpertsPrimitive(
                    x if j < 2 else other,
                    logits,
                    activity,
                    active,
                    *weights,
                    torch.empty_like(x),
                    first_expert=first_expert,
                    top_k=k,
                    operators=operators,
                    projection_scratch=workspace if j % 2 else None,
                    workspace=activations if j % 2 else None,
                )
                for j in range(4)
            ]
        assert len(observed) == 4
        assert observed[0].data_ptr() == observed[2].data_ptr()
        assert observed[1].data_ptr() == observed[3].data_ptr()
        for role in ("packed", "activated", "routes"):
            assert getattr(blocks[1], role).data_ptr() == getattr(blocks[3], role).data_ptr()
        snapshots = [tuple(torch.empty_like(t) for t in (b.packed, b.activated, b.routes))
                     for b in blocks]

        def launch_and_snapshot(block, snapshot):
            block.launch()
            for destination, source in zip(
                snapshot, (block.packed, block.activated, block.routes), strict=True,
            ):
                destination.copy_(source)

        for block, snapshot in zip(blocks, snapshots, strict=True):
            launch_and_snapshot(block, snapshot)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for block, snapshot in zip(blocks, snapshots, strict=True):
                launch_and_snapshot(block, snapshot)
        for iteration in range(2):
            for t in helper._buffers.values():
                t.fill_(float("nan"))
            x[rows] *= -0.5
            other[rows] *= 0.75
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(blocks[0]._operands[-1], blocks[1]._operands[-1])
            assert torch.equal(blocks[2]._operands[-1], blocks[3]._operands[-1])
            for block, snapshot in zip(blocks, snapshots, strict=True):
                packed = snapshot[0].view(capacity, k, 2 * i)
                activated = snapshot[1].view(capacity, k, i)
                routes = snapshot[2]
                for expert in range(e):
                    route = first_expert + expert
                    operand = block._operands[0][rows]
                    check_linear_enclosure(operand, weights[0][expert], packed[rows, route, :i])
                    check_linear_enclosure(operand, weights[1][expert], packed[rows, route, i:])
                    PackedMoETests().assert_swiglu_enclosure(
                        packed[rows, route, :i], packed[rows, route, i:], activated[rows, route]
                    )
                    check_linear_enclosure(
                        activated[rows, route], weights[2][expert], routes[rows, route]
                    )
                probs = torch.softmax(logits[rows].float(), dim=-1).to(dtype)
                expected = torch.zeros_like(x[rows])
                for route in range(first_expert, k):
                    expected = expected + (routes[rows, route] * probs[:, route, None]).to(
                        dtype
                    )
                assert torch.equal(block._operands[-1][rows], expected)
                assert torch.all(block.route_ids[activity == 0] == -1)
                assert torch.all(block._operands[-1][activity == 0] == 0)
            canonical = torch.stack([b._operands[-1][rows] for b in blocks])
            if iteration in logical_outputs:
                assert torch.equal(canonical, logical_outputs[iteration])
            else:
                logical_outputs[iteration] = canonical.clone()
        graph.reset()


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1" or not torch.cuda.is_available(),
    reason="explicit CUDA audit required",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bits", [4, 5, 8])
@pytest.mark.parametrize("mixed", [False, True])
def test_shared_projection_lane_workspace(dtype, bits, mixed):
    """Reuse serial partials while a disjoint main lane runs concurrently."""
    from cubic_serve.ops.cuda.moe.fused import MoeBranchStreams

    configure_reference_math()
    torch.set_num_threads(1)
    torch.manual_seed(113)
    device = torch.device("cuda", 0)
    rows, width, columns = 16, 512, 256
    # Both an eligible grouped binding and two distinct format bindings retain
    # their own reduction plans; neither is split to manufacture scratch reuse.
    weights = tuple(fixture(b, n=columns, k=width)[0]
                    for b in (bits, (5 if bits != 5 else 4) if mixed else bits))
    weights[1].payload["weight_scale"].mul_(1.25)
    inputs = [torch.randn(rows, width, device=device, dtype=dtype) * scale
              for scale in (0.05, -0.025, 0.075)]
    outputs = [torch.empty(rows, columns, device=device, dtype=dtype) for _ in inputs]
    private_outputs = [torch.empty_like(t) for t in outputs]
    operators = PackedLinearOperators()
    main = TextGraphBindings(operators, device, dtype)
    lane = TextGraphBindings(operators, device, dtype)
    main._shared_expert_bindings = lane
    with patch(
        "cubic_serve.quantization.extensions.int.marlin.plan_dense_partitions", return_value=2,
    ):
        main_call = main.linear(inputs[0], weights[0], outputs[0])
        # Use both single and grouped paths, as shared gate/up and down do.
        lane_calls = (
            lane.linear(inputs[1], weights[0], outputs[1]),
            lane.linear(inputs[2], weights[1], outputs[2]),
        )
        grouped_outputs = tuple(torch.empty_like(outputs[0]) for _ in weights)
        grouped_call = lane.linear_group(inputs[2], weights, grouped_outputs)
        private_grouped_outputs = tuple(torch.empty_like(t) for t in grouped_outputs)
        private_grouped = operators.bind_linear_groups(inputs[2], weights, private_grouped_outputs)
        private = [operators.bind_linear(x, weights[0 if j < 2 else 1], y)
                   for j, (x, y) in enumerate(zip(inputs, private_outputs, strict=True))]
    assert len(main._buffers) == 1
    assert len(lane._buffers) == 2
    main_roots = {t.untyped_storage().data_ptr() for t in main._buffers.values()}
    lane_roots = {t.untyped_storage().data_ptr() for t in lane._buffers.values()}
    assert main_roots.isdisjoint(lane_roots)
    assert lane_roots <= {t.untyped_storage().data_ptr() for t in main.scratch}
    branches = MoeBranchStreams(device, 1)

    def launch():
        branches.begin()
        with branches.fork_shared():
            for call in lane_calls:
                call()
            grouped_call()
        main_call()
        branches.join_shared()

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(2):
        for t in main.scratch:
            if t.is_floating_point():
                t.fill_(float("nan"))
        for x in inputs:
            x.mul_(-0.5)
        graph.replay()
        for binding in private:
            binding.launch()
        for binding in private_grouped:
            binding.launch()
        torch.cuda.synchronize()
        for j, (x, actual, expected) in enumerate(
            zip(inputs, outputs, private_outputs, strict=True),
        ):
            assert torch.equal(actual, expected)
            check_linear_enclosure(x, weights[0 if j < 2 else 1], actual)
        for weight, actual, expected in zip(
            weights, grouped_outputs, private_grouped_outputs, strict=True,
        ):
            assert torch.equal(actual, expected)
            check_linear_enclosure(inputs[2], weight, actual)
    graph.reset()


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1" or not torch.cuda.is_available(),
    reason="explicit CUDA audit required",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [16, 32])
def test_parallel_projection_lanes_reuse_only_after_join(dtype, rows):
    configure_reference_math()
    torch.set_num_threads(1)
    torch.manual_seed(127)
    device = torch.device("cuda", 0)
    # Different encodings require two existing bindings, making overlap eligible.
    weights = tuple(fixture(bits, n=256, k=512)[0] for bits in (4, 5))
    inputs = [torch.randn(rows, 512, device=device, dtype=dtype) * scale
              for scale in (0.025, -0.05)]
    outputs = [tuple(torch.empty(rows, 256, device=device, dtype=dtype) for _ in weights)
               for _ in inputs]
    controls = [tuple(torch.empty_like(t) for t in group) for group in outputs]
    operators = PackedLinearOperators()
    helper = TextGraphBindings(operators, device, dtype)
    with (
        patch("cubic_serve.quantization.extensions.int.marlin.plan_dense_partitions",
              return_value=2),
        patch("cubic_serve.executor.text_graph.device_work_cost") as rate,
    ):
        rate.return_value.branch_costs.return_value = (2.0, 1.0)
        calls = [helper.linear_group(x, weights, y, overlap=True)
                 for x, y in zip(inputs, outputs, strict=True)]
        private = [operators.bind_linear_groups(x, weights, y)
                   for x, y in zip(inputs, controls, strict=True)]
    assert helper.projection_branch_costs[tuple(id(w) for w in weights)][2]
    assert len(helper._buffers) == 2
    pointers = {t.untyped_storage().data_ptr() for t in helper._buffers.values()}
    assert len(pointers) == 2
    assert {key[0] for key in helper._buffers} == {"linear.parallel.0", "linear.parallel.1"}
    for call in calls:
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    for _ in range(2):
        for t in helper._buffers.values():
            t.fill_(float("nan"))
        for x in inputs:
            x.mul_(-0.5)
        graph.replay()
        for group in private:
            for binding in group:
                binding.launch()
        torch.cuda.synchronize()
        for x, actuals, expected in zip(inputs, outputs, controls, strict=True):
            for weight, actual, control in zip(weights, actuals, expected, strict=True):
                assert torch.equal(actual, control)
                check_linear_enclosure(x, weight, actual)
    graph.reset()
