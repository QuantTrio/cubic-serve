"""Independent arithmetic and row-ownership checks for the retained AWQ backend."""

import os

import pytest
import torch

from cubic_serve.executor.runner import configure_reference_math
from cubic_serve.guardrail.quantization import check_linear_enclosure
from cubic_serve.models.quantization import QuantizationScheme, WeightFormat
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
from cubic_serve.ops.cuda.moe.quantized import PackedRoutedExpertsPrimitive
from cubic_serve.ops.strategies import OperatorConfig


@pytest.mark.parametrize(
    "overrides",
    [
        {"marlin_enabled": 1},
        {"marlin_stages": 7},
        {"marlin_warps": 3},
        {"marlin_unknown": 64},
    ],
)
def test_marlin_rejects_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        OperatorConfig.from_dict({"version": 1, "overrides": {"awq_moe": overrides}})


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
@pytest.mark.parametrize(
    "dtype,group,bm",
    [
        (torch.bfloat16, 32, 16),
        (torch.float16, 64, 16),
        (torch.bfloat16, 128, 32),
    ],
)
@pytest.mark.parametrize("checkpoint_device", ["cpu", "cuda"])
def test_marlin_group_dtype_and_cross_m_ownership(dtype, group, bm, checkpoint_device):
    configure_reference_math()
    torch.set_num_threads(1)
    torch.manual_seed(173)
    device = "cuda:0"
    experts, hidden, intermediate, top_k = 4, 256, 128, 2
    scheme = QuantizationScheme(WeightFormat.AWQ, 4, group)

    def matrices(n, k):
        return tuple(
            QuantizedMatrix(
                {
                    "qweight": torch.randint(
                        -(2**31), 2**31 - 1, (k, n // 8), dtype=torch.int32, device=device
                    ),
                    "qzeros": torch.randint(
                        -(2**31), 2**31 - 1, (k // group, n // 8), dtype=torch.int32, device=device
                    ),
                    "scales": (torch.rand(k // group, n, device=device) * 0.04 + 0.001).half(),
                },
                scheme,
                (n, k),
            )
            for _ in range(experts)
        )

    gate, up, down = (
        matrices(intermediate, hidden),
        matrices(intermediate, hidden),
        matrices(hidden, intermediate),
    )
    gate, up, down = (
        tuple(w.to(checkpoint_device) for w in weights) for weights in (gate, up, down)
    )
    corpus = torch.randn(33, hidden, device=device, dtype=dtype) * 0.01
    router = torch.randn(33, experts, device=device, dtype=dtype)
    config = OperatorConfig.from_dict(
        {
            "version": 1,
            "overrides": {
                "awq_moe": {
                    "marlin_block_m": bm,
                    "fixed_block_m": 16,
                    "fixed_block_n": 32,
                    "fixed_warps": 4,
                    "fixed_stages": 1,
                }
            },
        }
    )
    operators = PackedLinearOperators(config)
    previous = None
    for cap, logical in ((1, 1), (8, 8), (64, 33)):
        x = torch.full((cap, hidden), float("nan"), device=device, dtype=dtype)
        logits = torch.full((cap, experts), float("nan"), device=device, dtype=dtype)
        x[:logical], logits[:logical] = corpus[:logical], router[:logical]
        activity = torch.zeros(cap, dtype=torch.int32, device=device)
        activity[:logical] = 1
        op = PackedRoutedExpertsPrimitive(
            x,
            logits,
            activity,
            torch.tensor([logical], dtype=torch.int32, device=device),
            gate,
            up,
            down,
            torch.empty_like(x),
            first_expert=0,
            top_k=top_k,
            operators=operators,
        )
        binding = op.projections[0][2]
        assert binding.project_artifacts()[-1].backend == "cuda-driver"
        assert op.projections[2][2].project_artifacts()[-1].backend == "cuda-driver"
        op.launch()
        torch.cuda.synchronize()
        result = op.packed[: logical * top_k].clone()
        for expert in range(experts):
            selected = torch.where(op.route_ids[:logical].flatten() == expert)[0]
            if selected.numel():
                operands = x[selected // top_k]
                check_linear_enclosure(operands, gate[expert], result[selected, :intermediate])
                check_linear_enclosure(operands, up[expert], result[selected, intermediate:])
                check_linear_enclosure(
                    op.activated[selected], down[expert], op.routes.reshape(-1, hidden)[selected]
                )
        down_result = op.routes.reshape(-1, hidden)[: logical * top_k].clone()
        if previous is not None:
            assert torch.equal(
                result[: len(previous[0])].view(torch.uint8), previous[0].view(torch.uint8)
            )
            assert torch.equal(
                down_result[: len(previous[1])].view(torch.uint8), previous[1].view(torch.uint8)
            )
        previous = result, down_result
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op.launch()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(result.view(torch.uint8), op.packed[: logical * top_k].view(torch.uint8))
        assert torch.equal(
            down_result.view(torch.uint8), op.routes.reshape(-1, hidden)[: logical * top_k].view(torch.uint8)
        )
        x[:logical], logits[:logical] = corpus[:logical].flip(0), router[:logical].flip(0)
        graph.replay()
        torch.cuda.synchronize()
        expected = result.reshape(logical, top_k, -1).flip(0).contiguous()
        assert torch.equal(
            expected.view(torch.uint8),
            op.packed[: logical * top_k].reshape_as(expected).view(torch.uint8),
        )
        expected_down = down_result.reshape(logical, top_k, -1).flip(0).contiguous()
        assert torch.equal(
            expected_down.view(torch.uint8),
            op.routes.reshape(-1, hidden)[: logical * top_k].reshape_as(expected_down).view(torch.uint8),
        )
        graph.reset()
