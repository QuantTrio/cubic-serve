import os

import pytest
import torch

from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
from cubic_serve.guardrail.quantized_graph.bounds import check, scaled_silu_bounds
from cubic_serve.ops.cuda.pointwise.dilated_conv import bind_dilated_depthwise_conv
from cubic_serve.ops.eager.activation import compute_dtype
from cubic_serve.ops.eager.ngram import dilated_depthwise_conv
from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

pytestmark = pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)


@pytest.fixture(params=["auto", "fp32"])
def activation_policy(request):
    with operator_strategy_scope(
        OperatorConfig(activation_dtype=request.param), device="", model="",
    ):
        yield request.param


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("channels,kernel,dilation", [(37, 3, 2), (10240, 4, 3)])
@pytest.mark.parametrize("strided", [False, True])
def test_segmented_dilated_conv_reference_cross_m_and_padding(
    dtype, channels, kernel, dilation, activation_policy, strided,
):
    configure_reference_math()
    torch.manual_seed(735)
    history_size = (kernel - 1) * dilation
    weight = (torch.randn(channels, 1, kernel) * 0.2).to(dtype).cuda()
    initial = torch.randn(1, history_size, channels).to(dtype).cuda()
    sequences = [torch.randn(1, length, channels).to(dtype).cuda() for length in (5, 2)]
    expected, histories, intervals = {}, {}, {}
    with ReferenceExecutionContext():
        for owner, sequence in enumerate(sequences):
            history = initial.clone() if owner == 0 else torch.zeros_like(initial)
            for step in range(sequence.shape[1]):
                combined = torch.cat((history.cpu(), sequence[:, step : step + 1].cpu()), 1)
                acc = torch.zeros(1, channels, dtype=torch.float32)
                for tap in range(kernel):
                    tap_value = combined[:, tap * dilation].float()
                    acc = acc + tap_value * weight[:, 0, tap].cpu().float()
                intervals[owner, step] = scaled_silu_bounds(
                    acc[0].to(dtype), divisor=1.0,
                    activation_dtype=compute_dtype(dtype, initial.device),
                )
                result, history = dilated_depthwise_conv(
                    sequence[:, step : step + 1], history, weight, dilation=dilation
                )
                expected[owner, step] = result[0, 0].cpu()
                histories[owner, step] = history[0].cpu()
    common = {}
    for capacity in (8, 16):
        x = torch.zeros(capacity, channels, dtype=dtype, device="cuda")
        backing = torch.full((10, (2 if strided else 1) * history_size, channels),
                             123, dtype=dtype, device="cuda")
        states = backing[:, :history_size]
        source, destination, segment, active = [
            torch.zeros(capacity, dtype=torch.int32, device="cuda") for _ in range(4)
        ]
        output = torch.empty_like(x)
        op = bind_dilated_depthwise_conv(
            x, states, source, destination, segment, active, weight, output, dilation=dilation
        )
        op.launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op.launch()
        for order in ((0, 1), (1, 0)):
            states.fill_(123)
            states[0].copy_(initial[0])
            x.fill_(float("nan"))
            output.fill_(77)
            source.fill_(999)
            destination.fill_(999)
            segment.fill_(999)
            active.zero_()
            row = 0
            mapping = []
            for owner in order:
                length = sequences[owner].shape[1]
                x[row : row + length].copy_(sequences[owner][0])
                for step in range(length):
                    source[row] = 0 if owner == 0 else -1
                    destination[row] = 1 + owner * 5 + step
                    segment[row] = length if step == 0 else 0
                    active[row] = 1
                    mapping.append((row, owner, step))
                    row += 1
            graph.replay()
            actual, after = output.cpu(), states.cpu()
            for physical, owner, step in mapping:
                lower, upper = intervals[owner, step]
                assert check(actual[physical], lower, upper)["outside_enclosure"] == 0
                assert check(expected[owner, step], lower, upper)["outside_enclosure"] == 0
                corrupt = torch.nextafter(upper.to(dtype), torch.full_like(upper, float("inf"),
                                                                          dtype=dtype))
                assert check(corrupt, lower, upper)["outside_enclosure"] > 0
                assert torch.equal(after[1 + owner * 5 + step], histories[owner, step])
                key = owner, step
                if key in common:
                    assert torch.equal(actual[physical], common[key])
                common[key] = actual[physical].clone()
            assert torch.equal(after[0], initial[0].cpu())
            assert (after[8:] == 123).all()
            assert (actual[row:] == 77).all()
            assert (backing[:, history_size:] == 123).all()
    # Permanent M1-style path, repeated in-place continuation, versus chunked rows.
    x = torch.empty(1, channels, dtype=dtype, device="cuda")
    states = initial.clone()
    source = torch.zeros(1, dtype=torch.int32, device="cuda")
    destination = torch.zeros_like(source)
    segment = torch.ones_like(source)
    active = torch.ones_like(source)
    output = torch.empty_like(x)
    op = bind_dilated_depthwise_conv(
        x, states, source, destination, segment, active, weight, output, dilation=dilation
    )
    x.zero_()
    op.launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.launch()
    states.copy_(initial)
    for step in range(5):
        x.copy_(sequences[0][:, step])
        graph.replay()
        assert torch.equal(output[0].cpu(), common[0, step])
        assert torch.equal(states[0].cpu(), histories[0, step])
