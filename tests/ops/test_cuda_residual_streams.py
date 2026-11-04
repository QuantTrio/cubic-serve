"""Residual-stream graph boundaries, row permutations and padding isolation."""

import os

import pytest
import torch

from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
from cubic_serve.ops.cuda.pointwise.residual_streams import (
    bind_scaled_activation,
    bind_stream_combine,
    bind_stream_mix,
)
from cubic_serve.ops.eager.activation import sigmoid_stored, silu_stored
from cubic_serve.ops.eager.hyperconnection import combine
from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

pytestmark = pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)


def test_stream_boundaries_graph_padding_and_cross_m():
    configure_reference_math()
    torch.manual_seed(631)
    with operator_strategy_scope(OperatorConfig(activation_dtype="fp32"), device="", model=""):
        for dtype in (torch.bfloat16, torch.float16):
            for streams, width in ((2, 64), (3, 80), (4, 2560)):
                corpus = torch.randn(17, streams * width, device="cuda", dtype=dtype)
                # Exactly representable preactivations avoid using approximation
                # error as a waiver of the intervening storage boundaries.
                logits = (
                    ((torch.arange(corpus.numel(), device="cuda") % 5) - 2)
                    .reshape_as(corpus)
                    .to(dtype)
                )
                updates = torch.randn(17, width, device="cuda", dtype=dtype)
                injection = (
                    ((torch.arange(17 * streams, device="cuda") % 5) - 2)
                    .reshape(17, streams)
                    .to(dtype)
                )
                with ReferenceExecutionContext():
                    expected_mix = (
                        (corpus * sigmoid_stored(logits))
                        .reshape(17, streams, width)
                        .float()
                        .mean(1)
                        .to(dtype)
                    )
                    expected_gate = sigmoid_stored(injection / streams) * 2
                    expected_low = silu_stored(logits / streams)
                    expected_output = combine(corpus, updates, expected_gate)
                for capacity, logical in ((1, 1), (8, 8), (32, 17)):
                    x = torch.empty(capacity, streams * width, device="cuda", dtype=dtype)
                    g, low = torch.empty_like(x), torch.empty_like(x)
                    update = torch.empty(capacity, width, device="cuda", dtype=dtype)
                    mixed = torch.empty_like(update)
                    projected = torch.empty(capacity, streams, device="cuda", dtype=dtype)
                    gates, output = torch.empty_like(projected), torch.empty_like(x)
                    ops = (
                        bind_scaled_activation(g, low, divisor=streams),
                        bind_stream_mix(x, g, mixed),
                        bind_scaled_activation(
                            projected, gates, divisor=streams, multiplier=2, activation="sigmoid"
                        ),
                        bind_stream_combine(x, update, gates, output),
                    )
                    for op in ops:
                        op.launch()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for op in ops:
                            op.launch()
                    for reverse in (False, True):
                        for actual, original in (
                            (x, corpus),
                            (g, logits),
                            (update, updates),
                            (projected, injection),
                        ):
                            actual.fill_(float("nan"))
                            source = original[:logical]
                            actual[:logical].copy_(source.flip(0) if reverse else source)
                        graph.replay()
                        for actual, expected in (
                            (mixed, expected_mix),
                            (gates, expected_gate),
                            (low, expected_low),
                            (output, expected_output),
                        ):
                            visible = actual[:logical].flip(0) if reverse else actual[:logical]
                            assert torch.equal(visible, expected[:logical])


def test_signed_root_gate_stored_boundaries_and_cross_m():
    import math

    from cubic_serve.ops.cuda.pointwise.residual_streams import bind_signed_root_gate

    configure_reference_math()
    torch.manual_seed(111)
    with operator_strategy_scope(OperatorConfig(activation_dtype="fp32"), device="", model=""):
        for dtype in (torch.bfloat16, torch.float16):
            for streams, width in ((3, 80), (4, 2560)):
                # Binary rational operands make the reduction exact; signs and
                # zero cover both signed-root branches and the floor boundary.
                key = torch.randint(-4, 5, (9, streams * width), device="cuda").to(dtype) / 8
                query = torch.randint(-4, 5, key.shape, device="cuda").to(dtype) / 8
                query[0].zero_()
                value = torch.randn(9, width, device="cuda", dtype=dtype)
                with ReferenceExecutionContext():
                    score = (key * query).reshape(9, streams, width).float().sum(-1).to(
                        dtype
                    ) / math.sqrt(width)
                    gate = sigmoid_stored(score.abs().clamp_min(1e-6).sqrt() * score.sign())
                    expected = (value[:, None] * gate[:, :, None]).flatten(1)
                common = {}
                for capacity in (1, 16, 32):
                    k = torch.empty(capacity, streams * width, device="cuda", dtype=dtype)
                    q = torch.empty_like(k)
                    v = torch.empty(capacity, width, device="cuda", dtype=dtype)
                    g = torch.empty(capacity, streams, device="cuda", dtype=dtype)
                    output = torch.empty_like(k)
                    op = bind_signed_root_gate(k, q, v, g, output)
                    k.zero_()
                    q.zero_()
                    v.zero_()
                    op.launch()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        op.launch()
                    mappings = (
                        [(i,) for i in range(9)]
                        if capacity == 1
                        else [tuple(range(9)), tuple(reversed(range(9)))]
                    )
                    for mapping in mappings:
                        k.fill_(float("nan"))
                        q.fill_(float("nan"))
                        v.fill_(float("nan"))
                        for physical, logical in enumerate(mapping):
                            k[physical].copy_(key[logical])
                            q[physical].copy_(query[logical])
                            v[physical].copy_(value[logical])
                        graph.replay()
                        actual, gates = output.cpu(), g.cpu()
                        for physical, logical in enumerate(mapping):
                            assert torch.equal(gates[physical], gate[logical].cpu())
                            assert torch.equal(actual[physical], expected[logical].cpu())
                            if logical in common:
                                assert torch.equal(actual[physical], common[logical])
                            common[logical] = actual[physical].clone()
