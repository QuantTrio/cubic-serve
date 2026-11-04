"""Independent intervals follow the actual probability storage boundary."""

import os

import pytest
import torch

from cubic_serve.guardrail.quantized_graph.attention_bounds import enclose


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("activation_policy", ["auto", "fp32"])
def test_native_gdn_intervals_follow_declared_math_and_reject_corruption(
    dtype, state_dtype, activation_policy,
):
    from cubic_serve.executor.runner import ReferenceExecutionContext
    from cubic_serve.guardrail.quantized_graph.bounds import check
    from cubic_serve.guardrail.quantized_graph.state_bounds import (
        native_gdn_output_range,
        native_gdn_state_range,
    )
    from cubic_serve.ops.eager.hybrid import _l2_normalize_stored, gated_delta_net_recurrent
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    generator = torch.Generator().manual_seed(945)
    def random(*shape, kind=dtype):
        return (torch.randn(*shape, generator=generator) * .1).to(kind)

    q, k = random(2, 1, 1, 128), random(2, 1, 1, 128)
    v, a, b = random(2, 1, 3, 17), random(2, 1, 3), random(2, 1, 3)
    alog, bias = random(3, kind=torch.float32), random(3)
    initial = random(2, 3, 17, 128, kind=state_dtype)
    with operator_strategy_scope(
        OperatorConfig(activation_dtype=activation_policy, attention_dtype="auto"),
        device="", model="",
    ), ReferenceExecutionContext():
        output, final = gated_delta_net_recurrent(
            q, k, v, a, b, alog, bias, initial, l2_epsilon=1e-6, output_dtype=dtype,
        )
        nq = _l2_normalize_stored(q[:, 0], epsilon=1e-6, output_dtype=dtype)
        nk = _l2_normalize_stored(k[:, 0], epsilon=1e-6, output_dtype=dtype)
    lo, hi = native_gdn_state_range(
        initial, nk.repeat_interleave(3, dim=1), v[:, 0], a[:, 0], b[:, 0], alog, bias,
        activation_dtype=dtype if activation_policy == "auto" else torch.float32,
    )
    assert check(final, lo, hi)["outside_enclosure"] == 0
    corrupted = torch.nextafter(hi.to(state_dtype), torch.full_like(final, float("inf")))
    assert check(corrupted, lo, hi)["outside_enclosure"] > 0
    olo, ohi = native_gdn_output_range(final, nq, dtype=dtype)
    assert check(output[:, 0], olo, ohi)["outside_enclosure"] == 0
    corrupted = torch.nextafter(ohi.to(dtype), torch.full_like(output[:, 0], float("inf")))
    assert check(corrupted, olo, ohi)["outside_enclosure"] > 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("activation_policy", ["auto", "fp32"])
def test_conv_interval_resolves_activation_after_wide_sum(dtype, activation_policy):
    from cubic_serve.executor.runner import ReferenceExecutionContext
    from cubic_serve.guardrail.quantized_graph.bounds import check
    from cubic_serve.guardrail.quantized_graph.state_bounds import conv_silu_range
    from cubic_serve.ops.eager.hybrid import depthwise_causal_conv1d
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    generator = torch.Generator().manual_seed(936)
    x = torch.randn(2, 1, 17, generator=generator).to(dtype)
    weight = torch.randn(17, 4, generator=generator).to(dtype)
    state = torch.randn(2, 17, 3, generator=generator).to(dtype)
    with operator_strategy_scope(OperatorConfig(activation_dtype=activation_policy),
                                 device="", model=""), ReferenceExecutionContext():
        output, _ = depthwise_causal_conv1d(x, weight, state, activation="silu", output_dtype=dtype)
    lo, hi = conv_silu_range(
        x[:, 0], weight, state, dtype=dtype,
        activation_dtype=dtype if activation_policy == "auto" else torch.float32,
    )
    assert check(output[:, 0], lo, hi)["outside_enclosure"] == 0
    corrupted = torch.nextafter(hi.to(dtype), torch.full_like(output[:, 0], float("inf")))
    assert check(corrupted, lo, hi)["outside_enclosure"] > 0



@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("activation_dtype", [torch.bfloat16, torch.float16])
def test_gdn_state_interval_witnesses_each_rounding_boundary(state_dtype, activation_dtype):
    from cubic_serve.guardrail.quantized_graph.bounds import check
    from cubic_serve.guardrail.quantized_graph.state_bounds import fp32_gdn_state_range

    generator = torch.Generator().manual_seed(991)

    def random(*shape, dtype=activation_dtype):
        return torch.randn(*shape, generator=generator).to(dtype)

    state = random(2, 3, 8, 128, dtype=state_dtype)
    key = random(2, 3, 128) / 16
    value, a, b, alog, bias = random(2, 3, 8), random(2, 3), random(2, 3), random(3), random(3)
    lo, hi = fp32_gdn_state_range(state, key, value, a, b, alog, bias)
    decay = torch.exp(
        -torch.exp(alog.float()) * torch.nn.functional.softplus(a.float() + bias.float())
    )
    beta = torch.sigmoid(b.float()).to(activation_dtype).float()
    decayed = state.float() * decay[..., None, None]
    terms = decayed * key.float()[..., None, :]
    for ordered in (terms, terms.flip(-1)):
        memory = ordered.sum(-1)
        delta = (value.float() - memory) * beta[..., None]
        after = (decayed + delta[..., None] * key.float()[..., None, :]).to(state_dtype)
        assert check(after, lo, hi)["outside_enclosure"] == 0
    corrupt = torch.nextafter(
        hi.to(state_dtype), torch.full_like(hi, float("inf"), dtype=state_dtype)
    )
    assert check(corrupt, lo, hi)["outside_enclosure"] > 0


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gdn_output_sum_interval_preserves_storage_and_rejects_corruption(
    state_dtype, output_dtype
):
    from cubic_serve.guardrail.quantized_graph.bounds import check
    from cubic_serve.guardrail.quantized_graph.state_bounds import fp32_gdn_output_range

    generator = torch.Generator().manual_seed(20260918)
    state = torch.randn(2, 6, 8, 128, generator=generator).to(state_dtype)
    query = torch.randn(2, 2, 128, generator=generator).to(output_dtype)
    lower, upper = fp32_gdn_output_range(state, query, dtype=output_dtype)
    scaled = (query.float() * 128**-0.5).repeat_interleave(3, dim=1)
    terms = state.float() * scaled.unsqueeze(-2)
    for ordered in (terms, terms.flip(-1)):
        output = ordered.sum(-1).to(output_dtype)
        assert check(output, lower, upper)["outside_enclosure"] == 0
    corrupted = torch.nextafter(
        upper.to(output_dtype), torch.full_like(upper, float("inf"), dtype=output_dtype)
    )
    assert check(corrupted, lower, upper)["outside_enclosure"] > 0


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
@pytest.mark.parametrize("precision", ["auto", "fp32"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_probability_boundary_intervals_detect_corruption(precision, dtype):
    from cubic_serve.ops.cuda.attention.paged_decode import PagedGQADecodePrimitive
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    torch.manual_seed(141)
    query = torch.randn(1, 6, 256, device="cuda", dtype=dtype)
    key = torch.randn(2, 1, 16, 2, 256, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    table = torch.tensor([[1, 0]], device="cuda", dtype=torch.int32)
    length = torch.ones(1, device="cuda", dtype=torch.int32)
    output = torch.empty_like(query)
    with operator_strategy_scope(
        OperatorConfig(attention_dtype=precision, softmax_dtype="fp32"), device="", model=""
    ):
        op = PagedGQADecodePrimitive(query, key, value, table, length, output, layer=0)
    ordered_k = key[table[0].long(), 0].reshape(1, 32, 2, 256)
    ordered_v = value[table[0].long(), 0].reshape_as(ordered_k)
    for count in (1, 2, 4, 17, 31):
        length.fill_(count)
        op.launch(query, key, value, table, length, output)
        options = dict(scale=256**-0.5, splits=1,
                       probability_dtype=torch.float32 if precision == "fp32" else dtype)
        proof = enclose(query, ordered_k[:, :count], ordered_v[:, :count], output, **options)
        assert proof["outside_enclosure"] == 0, proof
        corrupted = output.clone()
        corrupted[0, 0, 0] += 1
        rejected = enclose(query, ordered_k[:, :count], ordered_v[:, :count], corrupted, **options)
        assert rejected["outside_enclosure"] > 0

@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation_policy", ["auto", "fp32"])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_scaled_activation_actual_kernel_enclosure_and_corruption(
    dtype, activation_policy, activation,
):
    from cubic_serve.guardrail.quantized_graph.bounds import check, scaled_activation_bounds
    from cubic_serve.ops.cuda.pointwise.residual_streams import bind_scaled_activation
    from cubic_serve.ops.eager.activation import compute_dtype
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    torch.manual_seed(1027)
    x = (torch.randn(3, 257, device="cuda") * 8).to(dtype)
    y = torch.empty_like(x)
    with operator_strategy_scope(
        OperatorConfig(activation_dtype=activation_policy), device="", model="",
    ):
        options = {"divisor": 4.0, "activation": activation,
                   "multiplier": 2.0 if activation == "sigmoid" else 1.0}
        binding = bind_scaled_activation(x, y, **options)
        math_dtype = compute_dtype(dtype, x.device)
    binding.launch()
    lo, hi = scaled_activation_bounds(x.cpu(), **options, activation_dtype=math_dtype)
    assert check(y.cpu(), lo, hi)["outside_enclosure"] == 0
    corrupted = torch.nextafter(hi.to(dtype), torch.full_like(hi, float("inf"), dtype=dtype))
    assert check(corrupted, lo, hi)["outside_enclosure"] > 0


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation_policy", ["auto", "fp32"])
@pytest.mark.parametrize("fused", [False, True])
def test_stream_combine_actual_rounded_residual_and_corruption(dtype, activation_policy, fused):
    from unittest.mock import patch

    from cubic_serve.guardrail.quantized_graph.bounds import check, stream_combine_bounds
    from cubic_serve.ops.cuda.pointwise.residual_streams import bind_stream_combine_norm
    from cubic_serve.ops.eager.activation import compute_dtype
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    torch.manual_seed(1028)
    residual = torch.randn(3, 3 * 80, device="cuda", dtype=dtype)
    initial = residual.clone()
    update = torch.randn(3, 80, device="cuda", dtype=dtype)
    logits = torch.randn(3, 3, device="cuda", dtype=dtype)
    weight, output = torch.zeros(3 * 80, device="cuda", dtype=dtype), torch.empty_like(residual)
    gates = torch.empty_like(logits)
    with operator_strategy_scope(
        OperatorConfig(activation_dtype=activation_policy), device="", model="",
    ), patch("cubic_serve.ops.cuda.launch_cost.select_launch", return_value=fused):
        op = bind_stream_combine_norm(
            residual, update, logits, weight, output, gates, epsilon=1e-6, choices={},
        )
        math_dtype = compute_dtype(dtype, residual.device)
    op.launch()
    lo, hi = stream_combine_bounds(initial.cpu(), update.cpu(), logits.cpu(),
                                   activation_dtype=math_dtype)
    assert check(residual.cpu(), lo, hi)["outside_enclosure"] == 0
    corrupted = torch.nextafter(hi.to(dtype), torch.full_like(hi, float("inf"), dtype=dtype))
    assert check(corrupted, lo, hi)["outside_enclosure"] > 0


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation_policy", ["auto", "fp32"])
def test_packed_swiglu_actual_kernel_stored_product_and_corruption(dtype, activation_policy):
    from cubic_serve.guardrail.quantized_graph.bounds import check, swiglu_bounds
    from cubic_serve.ops.cuda.pointwise import BF16PackedSwiGLUPrimitive
    from cubic_serve.ops.eager.activation import compute_dtype
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    torch.manual_seed(1029)
    packed = (torch.randn(3, 2 * 257, device="cuda") * 3).to(dtype)
    output = torch.empty(3, 257, device="cuda", dtype=dtype)
    with operator_strategy_scope(
        OperatorConfig(activation_dtype=activation_policy), device="", model="",
    ):
        op = BF16PackedSwiGLUPrimitive(packed, output)
        math_dtype = compute_dtype(dtype, packed.device)
    op.launch(packed, output)
    lo, hi = swiglu_bounds(packed[:, :257].cpu(), packed[:, 257:].cpu(),
                           activation_dtype=math_dtype)
    assert check(output.cpu(), lo, hi)["outside_enclosure"] == 0
    corrupted = torch.nextafter(hi.to(dtype), torch.full_like(hi, float("inf"), dtype=dtype))
    assert check(corrupted, lo, hi)["outside_enclosure"] > 0


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation_policy", ["auto", "fp32"])
def test_shared_moe_combine_actual_input_enclosure_and_corruption(dtype, activation_policy):
    from cubic_serve.guardrail.quantized_graph.bounds import check, moe_shared_combine_bounds
    from cubic_serve.ops.cuda.moe import MoeSharedCombinePrimitive
    from cubic_serve.ops.eager.activation import compute_dtype
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    torch.manual_seed(1030)
    shared = torch.randn(3, 257, device="cuda", dtype=dtype)
    logits = torch.randn(3, 1, device="cuda", dtype=dtype)
    routed, output = torch.randn_like(shared), torch.empty_like(shared)
    with operator_strategy_scope(
        OperatorConfig(activation_dtype=activation_policy), device="", model="",
    ):
        op = MoeSharedCombinePrimitive(shared, logits, routed, output)
        math_dtype = compute_dtype(dtype, shared.device)
    op.launch(shared, logits, routed, output)
    lo, hi = moe_shared_combine_bounds(shared.cpu(), logits.cpu(), routed.cpu(),
                                     activation_dtype=math_dtype)
    assert check(output.cpu(), lo, hi)["outside_enclosure"] == 0
    corrupted = torch.nextafter(hi.to(dtype), torch.full_like(hi, float("inf"), dtype=dtype))
    assert check(corrupted, lo, hi)["outside_enclosure"] > 0


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation_policy", ["auto", "fp32"])
def test_signed_root_actual_gate_and_independent_readout_enclosure(dtype, activation_policy):
    from cubic_serve.executor.runner import ReferenceExecutionContext
    from cubic_serve.guardrail.quantized_graph.bounds import (
        check,
        product,
        signed_root_sigmoid_bounds,
    )
    from cubic_serve.ops.cuda.pointwise.residual_streams import bind_signed_root_gate
    from cubic_serve.ops.eager.activation import compute_dtype
    from cubic_serve.ops.eager.hyperconnection import signed_root_gate
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    torch.manual_seed(1031)
    key = torch.randn(3, 3 * 80, device="cuda", dtype=dtype)
    query = torch.randn_like(key)
    key[0].zero_()
    value = torch.randn(3, 80, device="cuda", dtype=dtype)
    gates, output = torch.empty(3, 3, device="cuda", dtype=dtype), torch.empty_like(key)
    with operator_strategy_scope(
        OperatorConfig(activation_dtype=activation_policy), device="", model="",
    ), ReferenceExecutionContext():
        op = bind_signed_root_gate(key, query, value, gates, output)
        math_dtype = compute_dtype(dtype, key.device)
        independent = signed_root_gate(key.reshape(3, 3, 80), query.reshape(3, 3, 80), value)
    op.launch()
    lo, hi = signed_root_sigmoid_bounds(key.cpu().reshape(3, 3, 80),
        query.cpu().reshape(3, 3, 80), activation_dtype=math_dtype)
    assert check(gates.cpu(), lo, hi)["outside_enclosure"] == 0
    operand = value.cpu().double().unsqueeze(1)
    lower, upper = product(operand, operand, lo.unsqueeze(-1), hi.unsqueeze(-1))
    lower = lower.float().to(dtype).double().flatten(1)
    upper = upper.float().to(dtype).double().flatten(1)
    assert check(output.cpu(), lower, upper)["outside_enclosure"] == 0
    assert check(independent.cpu(), lower, upper)["outside_enclosure"] == 0
    corrupted = torch.nextafter(hi.to(dtype), torch.full_like(hi, float("inf"), dtype=dtype))
    assert check(corrupted, lo, hi)["outside_enclosure"] > 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required",
))])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_interval_product_preserves_corners_and_immutable_broadcast_views(dtype, device):
    from cubic_serve.guardrail.quantized_graph.bounds import product

    values = torch.tensor([-float("inf"), -3., -0., 0., 2., float("inf"), float("nan")],
                          dtype=dtype, device=device)
    operands = torch.meshgrid(values, values, values, values, indexing="ij")
    before = [t.contiguous().view(torch.uint8).clone() for t in operands]
    corners = torch.stack([operands[i] * operands[j]
                           for i, j in ((0, 2), (0, 3), (1, 2), (1, 3))])
    expected = corners.amin(0), corners.amax(0)
    actual = product(*operands)
    for result, reference in zip(actual, expected, strict=True):
        assert torch.equal(result.view(torch.uint8), reference.view(torch.uint8))
    actual[0].zero_()
    assert torch.equal(actual[1].view(torch.uint8), expected[1].view(torch.uint8))
    for operand, original in zip(operands, before, strict=True):
        assert torch.equal(operand.contiguous().view(torch.uint8), original)


def test_interval_product_retains_mixed_dtype_and_gradient_contracts():
    from cubic_serve.guardrail.quantized_graph.bounds import product

    a = torch.tensor([-2., 1.], dtype=torch.float32, requires_grad=True)
    b = torch.tensor([3., 4.], dtype=torch.float64, requires_grad=True)
    c, d = torch.tensor([-4., -1.]), torch.tensor([2., 3.])
    corners = torch.stack([a * c, a * d, b * c, b * d])
    expected = corners.amin(0), corners.amax(0)
    actual = product(a, b, c, d)
    assert all(torch.equal(x, y) and x.dtype == y.dtype
               for x, y in zip(actual, expected, strict=True))
    gradients = torch.autograd.grad(sum(t.sum() for t in actual), (a, b))
    reference = torch.autograd.grad(sum(t.sum() for t in expected), (a, b))
    assert all(torch.equal(x, y) for x, y in zip(gradients, reference, strict=True))
