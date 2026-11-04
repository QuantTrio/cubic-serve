"""Composite graph semantic checks; these are not serving performance evidence."""

import math
import os

import pytest
import torch

from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
from cubic_serve.executor.text_graph import TextGraphBindings
from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
from cubic_serve.ops.eager.activation import sigmoid_stored
from cubic_serve.ops.eager.hyperconnection import grouped_rms_norm
from cubic_serve.ops.eager.ngram import dilated_depthwise_conv, hash_indices
from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

pytestmark = pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)


def test_retained_body_composition_preserves_dependencies_lifetime_and_conditional_writes():
    from cuda.bindings import runtime as rt
    from cubic_serve.ops.cuda.native import _check
    from cubic_serve.executor.resident_window import (
        append_captured_graph, conditional_handle, add_conditional_body,
    )
    from cuda.bindings import driver as cuda_driver

    x = torch.ones(8, device="cuda")
    y = torch.zeros_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.add(x, 2, out=y)
    stream.synchronize()
    child = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(child, stream=stream):
        torch.add(x, 2, out=y)
    child.instantiate()
    parent = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(parent, stream=stream):
        x.mul_(2)
        append_captured_graph(child)
        y.mul_(3)
    parent.instantiate()
    stream.synchronize()
    for enabled in (0, 1):
        root = _check(rt.cudaGraphCreate(0))[0]
        executable = None
        try:
            handle = conditional_handle(root, enabled)
            node, body = add_conditional_body(root, [], handle, while_loop=False)
            parameters = _check(cuda_driver.cuGraphNodeGetParams(node))[0]
            context = _check(cuda_driver.cuCtxGetCurrent())[0]
            assert int(parameters.conditional.ctx) == int(context) != 0
            _check(rt.cudaGraphAddChildGraphNode(body, [], 0,
                                               parent.raw_cuda_graph()))
            executable = _check(cuda_driver.cuGraphInstantiate(root, 0))[0]
            for value in (5., 9., -7.):
                x.fill_(value)
                y.fill_(-99)
                stream.wait_stream(torch.cuda.current_stream())
                _check(rt.cudaGraphLaunch(executable, stream.cuda_stream))
                stream.synchronize()
                assert x.cpu().tolist() == [value * 2 if enabled else value] * 8
                assert y.cpu().tolist() == [(value * 2 + 2) * 3 if enabled else -99] * 8
        finally:
            if executable is not None:
                _check(rt.cudaGraphExecDestroy(executable))
            _check(rt.cudaGraphDestroy(root))
    # Child ownership was not transferred to either parent or conditional graph.
    x.fill_(4)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        child.replay()
    stream.synchronize()
    assert y.cpu().tolist() == [6.] * 8
    with pytest.raises(ValueError, match="active startup capture"):
        append_captured_graph(child)
    parent.reset()
    child.reset()


def test_native_only_nested_loop_binds_context_and_resets_on_every_replay():
    from cuda.bindings import driver as cuda_driver, runtime as rt
    from cubic_serve.executor.resident_window import (
        check, conditional_handle, add_conditional_body,
    )
    from cubic_serve.ops.cuda.native import CudaScalar
    from cubic_serve.ops.cuda.native_compiler import compile_native

    counter = torch.zeros(1, dtype=torch.int32, device='cuda')
    source = b'''#include <cuda_runtime.h>
extern "C" __global__ void step(int* counter, unsigned long long handle) {
    if (threadIdx.x == 0) {
        counter[0] += 1;
        cudaGraphSetConditional(handle, counter[0] < 3);
    }
}
'''
    module, entry = compile_native((('entry.cu', source),), counter.device, 'context_loop_audit')
    root = check(rt.cudaGraphCreate(0))[0]
    executable = None
    captured = torch.cuda.CUDAGraph(keep_graph=True)
    stream = torch.cuda.Stream()
    try:
        loop = conditional_handle(root, 1)
        branch = conditional_handle(root, 1)
        call = module.bind(entry, (counter, CudaScalar('i64', int(loop))),
                           operator_id='audit.context_loop', grid=(1, 1, 1), block=(32, 1, 1))
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(captured, stream=stream):
            call.launch()
        captured.instantiate()
        loop_node, loop_body = add_conditional_body(root, [], loop, while_loop=True)
        branch_node, branch_body = add_conditional_body(loop_body, [], branch, while_loop=False)
        context = check(cuda_driver.cuCtxGetCurrent())[0]
        for node in (loop_node, branch_node):
            params = check(cuda_driver.cuGraphNodeGetParams(node))[0]
            assert int(params.conditional.ctx) == int(context) != 0
        check(rt.cudaGraphAddChildGraphNode(branch_body, [], 0, captured.raw_cuda_graph()))
        executable = check(cuda_driver.cuGraphInstantiate(root, 0))[0]
        for _ in range(3):
            counter.zero_()
            stream.wait_stream(torch.cuda.current_stream())
            check(rt.cudaGraphLaunch(executable, stream.cuda_stream))
            stream.synchronize()
            assert counter.item() == 3
    finally:
        if executable is not None:
            check(rt.cudaGraphExecDestroy(executable))
        check(rt.cudaGraphDestroy(root))
        captured.reset()


def test_rope_gather_reuses_attestation_only_for_unchanged_owned_tables():
    from unittest.mock import patch

    from cubic_serve.ops.cuda.position import qwen_mrope_table as implementation

    configure_reference_math()
    device = torch.device("cuda", 0)
    helper = TextGraphBindings(PackedLinearOperators(config=OperatorConfig()),
                               device, torch.bfloat16)
    cosine, sine, evidence = implementation.build_bf16_mrope_tables(
        max_position=256, device=device,
    )
    positions = torch.zeros((3, 2), dtype=torch.int64, device=device)
    out_cosine = torch.empty((2, 64), dtype=torch.bfloat16, device=device)
    out_sine = torch.empty_like(out_cosine)
    with patch.object(implementation, "_digest", wraps=implementation._digest) as digest:
        first = helper._rope_gather(positions, cosine, sine, out_cosine, out_sine,
                                    evidence=evidence)
        second = helper._rope_gather(positions.clone(), cosine, sine,
                                     out_cosine.clone(), out_sine.clone(), evidence=evidence)
        assert first is second
        assert digest.call_count == 1
        cosine[0, 0] = 0
        with pytest.raises(ValueError, match="evidence digest"):
            helper._rope_gather(positions, cosine, sine, out_cosine, out_sine,
                                evidence=evidence)
        assert digest.call_count == 2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shared_scratch", [False, True])
def test_bound_lexical_graph_reference_and_cross_m(dtype, shared_scratch):
    configure_reference_math()
    torch.manual_seed(193)
    device = torch.device("cuda", 0)
    policy = OperatorConfig(activation_dtype="fp32", attention_dtype="fp32")
    width, streams, eos, steps = 32, 2, 7, 11
    table = torch.randn(46, 8, device=device).to(dtype)
    scale = torch.ones((), device=device, dtype=torch.float32)
    salts = torch.tensor([19, 37, 53], device=device, dtype=torch.int64)
    sizes = torch.tensor([23, 23], device=device, dtype=torch.int64)
    offsets = torch.tensor([0, 23], device=device, dtype=torch.int64)
    # One nonzero per projection row isolates composite wiring from GEMM bounds.
    key_weight = torch.zeros(streams * width, 16, device=device, dtype=dtype)
    value_weight = torch.zeros(width, 16, device=device, dtype=dtype)
    for n, weight in enumerate((key_weight, value_weight)):
        weight[torch.arange(weight.shape[0]), (torch.arange(weight.shape[0]) + n) % 16] = 0.5
    norm = torch.zeros(streams * width, device=device, dtype=dtype)
    conv_weight = torch.zeros(streams * width, 1, 4, device=device, dtype=dtype)
    conv_weight[:, :, 0] = 0.25
    conv_weight[:, :, -1] = 0.5
    corpus = torch.randn(steps, streams * width, device=device).to(dtype)
    token_corpus = [3, 2, eos, 4, 1, 8, 9, 5, 6, 2, 1]
    common = {}
    scratch = {} if shared_scratch else None
    with operator_strategy_scope(policy, device="", model=""):
        for capacity in ((1, 8, 4, 2) if shared_scratch else (1, 4)):
            helper = TextGraphBindings(PackedLinearOperators(config=policy), device, dtype,
                                       shared_scratch=scratch)
            residual = torch.zeros(capacity, streams * width, device=device, dtype=dtype)
            tokens = torch.zeros(capacity, device=device, dtype=torch.int64)
            source = torch.zeros(capacity, device=device, dtype=torch.int32)
            destination = torch.arange(capacity, device=device, dtype=torch.int32)
            active = torch.ones(capacity, device=device, dtype=torch.int32)
            hash_state = torch.full((capacity, 2), eos, device=device, dtype=torch.int64)
            conv_state = torch.zeros(capacity, 9, streams * width, device=device, dtype=dtype)
            block = helper.ple(
                residual,
                tokens,
                shards=((0, table[:17]), (17, table[17:])),
                table_scale=scale,
                multipliers=salts,
                vocab_sizes=sizes,
                offsets=offsets,
                eos_token_id=eos,
                hash_state=hash_state,
                hash_sources=source,
                hash_destinations=destination,
                conv_state=conv_state,
                conv_sources=source,
                conv_destinations=destination,
                segment_lengths=active,
                row_activity=active,
                key_projection=key_weight,
                value_projection=value_weight,
                key_norm=norm,
                query_norm=norm,
                conv_norm=norm,
                conv_weight=conv_weight,
                streams=streams,
                dilation=3,
                epsilon=1e-6,
            )
            block.launch()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                block.launch()
            hash_state.fill_(eos)
            conv_state.zero_()
            history = hash_state[:1].clone()
            previous = conv_state[:1].clone()
            for step, token in enumerate(token_corpus):
                residual.copy_(corpus[step].expand_as(residual))
                tokens.fill_(token)
                source.copy_(destination)
                if step == 0:
                    source.fill_(-1)
                graph.replay()
                with ReferenceExecutionContext():
                    ids, history = hash_indices(
                        tokens[:1, None], history, salts, sizes, offsets, eos_token_id=eos
                    )
                    embedding = table[ids].flatten(-2)
                    key = grouped_rms_norm(
                        (embedding.float() @ key_weight.float().T).to(dtype),
                        norm,
                        group_size=width,
                        eps=1e-6,
                    )
                    query = grouped_rms_norm(
                        corpus[step][None, None], norm, group_size=width, eps=1e-6
                    )
                    score = (key * query).reshape(1, 1, streams, width).float().sum(-1).to(dtype)
                    score = score / math.sqrt(width)
                    gate = sigmoid_stored(score.abs().clamp_min(1e-6).sqrt() * score.sign())
                    value = (embedding.float() @ value_weight.float().T).to(dtype)
                    gated = (gate[..., None] * value.unsqueeze(-2)).flatten(-2)
                    conv_input = grouped_rms_norm(gated, norm, group_size=width, eps=1e-6)
                    convolved, previous = dilated_depthwise_conv(
                        conv_input, previous, conv_weight, dilation=3
                    )
                    expected = corpus[step] + (gated + convolved).reshape(-1)
                # Explicitly require exact results for this structured corpus.
                assert torch.equal(residual, expected.expand_as(residual))
                assert torch.equal(hash_state, history.expand_as(hash_state))
                assert torch.equal(conv_state, previous.expand_as(conv_state))
                if step in common:
                    assert torch.equal(residual[0], common[step])
                common[step] = residual[0].clone()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("quantized_cache", [False, True])
@pytest.mark.parametrize("shared_scratch", [False, True])
def test_bound_qsa_graph_tail_selection_pages_and_cross_m(dtype, quantized_cache, shared_scratch):
    from cubic_serve.models.qwen4_exp.model import Qwen4ExpLayerState
    from cubic_serve.ops.cuda.position.qwen_mrope_table import build_bf16_mrope_tables
    from tests.models.qwen4_exp.test_reference import tiny_model

    configure_reference_math()
    reference = tiny_model()
    t, p = reference.config.text, "model.language_model.layers.1.self_attn"
    device = torch.device("cuda", 0)
    for name in reference.weights:
        reference.weights[name] = reference.weights[name].to(device=device, dtype=dtype)
    for name, weight in reference.weights.items():
        if name.startswith(p + "."):
            weight.zero_()
    # Zero Q/K gives exact uniform attention and deterministic earliest-block ties.
    # Nonzero value/output selectors test original-token addressing across pages.
    for name in ("v_proj", "o_proj"):
        weight = reference.weights[p + "." + name + ".weight"]
        weight[torch.arange(weight.shape[0]), torch.arange(weight.shape[0]) % weight.shape[1]] = 0.5
    policy = OperatorConfig(activation_dtype="fp32", attention_dtype="fp32")
    cosine, sine, evidence = build_bf16_mrope_tables(
        max_position=16,
        device=device,
        rotary_dim=8,
        section=t.mrope.section,
        rope_theta=t.mrope.rope_theta,
        dtype=dtype,
    )
    common = {}
    scratch = {} if shared_scratch else None
    with operator_strategy_scope(policy, device="", model=""):
        for capacity in ((1, 8, 4, 2) if shared_scratch else (1, 4)):

            def zeros(shape, dt=dtype):
                return torch.zeros(shape, device=device, dtype=dt)

            x = zeros((capacity, 16))
            key, value = zeros((capacity * 3, 1, 4, 1, 8)), zeros((capacity * 3, 1, 4, 1, 8))
            written = zeros((capacity * 3, 1, 4), torch.bool)
            compressed = zeros((capacity * 3, 2, 8))
            raw = zeros((capacity, 2, 8))
            history = zeros((capacity, 2, 3), torch.int64)
            source = zeros((capacity,), torch.int32)
            destination = torch.arange(capacity, device=device, dtype=torch.int32)
            cache_options = {}
            arena = None
            if quantized_cache:
                from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout
                from cubic_serve.runtime.cache.pages import PageFieldSpec
                from cubic_serve.runtime.cache.quantization import (
                    CacheDType, CacheGroupAxis, CacheStorageSpec, KVCacheStorageSpec,
                )
                layout = KVLayout(
                    (1,), 4, 1, 8, dtype=dtype,
                    page_fields=(PageFieldSpec("compressed", dtype, (2, 8)),),
                    storage=KVCacheStorageSpec(
                        CacheStorageSpec(CacheDType.FP8, CacheGroupAxis.TOKENS, 4),
                        CacheStorageSpec(CacheDType.FP8, CacheGroupAxis.CHANNELS, 4),
                    ),
                )
                arena = CacheArena(
                    CacheArenaSpec("qsa", capacity * 3, capacity, layout.quantization_state_specs),
                    layout, device, rank_id=0,
                )
                key, value, written = arena.key, arena.value, arena.written
                compressed = arena.page_fields["compressed"]
                cache_options = dict(cache_arena=arena, cache_source_slots=source,
                                     cache_destination_slots=destination)
            helper = TextGraphBindings(PackedLinearOperators(config=policy), device, dtype,
                                       shared_scratch=scratch, **cache_options)
            active = torch.ones_like(destination)
            positions = zeros((3, capacity), torch.int64)
            logical = zeros((capacity,), torch.int64)
            pages, offsets, lengths = (zeros((capacity,), torch.int32) for _ in range(3))
            table = torch.arange(capacity * 3, device=device, dtype=torch.int32).reshape(
                capacity, 3
            )
            block = helper.qsa(
                x,
                {
                    n: reference.weights[p + "." + role + ".weight"]
                    for n, role in {
                        "q": "q_proj",
                        "k": "k_proj",
                        "v": "v_proj",
                        "out": "o_proj",
                        "index_qk": "indexer.index_qk_proj",
                    }.items()
                },
                {
                    n: reference.weights[p + "." + role + ".weight"]
                    for n, role in {
                        "q": "q_norm",
                        "k": "k_norm",
                        "index_q": "indexer.q_layernorm",
                        "index_k": "indexer.k_layernorm",
                    }.items()
                },
                query_heads=2,
                head_dim=8,
                index_heads=2,
                compression=2,
                token_budget=2,
                epsilon=t.rms_norm_epsilon,
                key_cache=key,
                value_cache=value,
                written=written,
                cache_layer=0,
                compressed_cache=compressed,
                raw_history=raw,
                position_history=history,
                sources=source,
                destinations=destination,
                segment_lengths=active,
                row_activity=active,
                logical_positions=logical,
                positions=positions,
                page_table=table,
                page_destinations=pages,
                token_offsets=offsets,
                visible_lengths=lengths,
                cosine_table=cosine,
                sine_table=sine,
                rope_evidence=evidence,
            )
            pages.copy_(table[:, 0])
            lengths.fill_(1)
            block.launch()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                block.launch()
            for operand in (key, value, written, compressed, raw, history):
                operand.zero_()
            if arena is not None:
                for operand in (*arena.states.values(), *arena.page_fields.values()):
                    operand.zero_()
            state = Qwen4ExpLayerState()
            for step in range(9):
                x.fill_((step + 1) / 8)
                source.copy_(destination)
                if step == 0:
                    source.fill_(-1)
                positions.fill_(step)
                logical.fill_(step)
                pages.copy_(table[:, step // 4])
                offsets.fill_(step % 4)
                lengths.fill_(step + 1)
                graph.replay()
                with ReferenceExecutionContext():
                    expected, state = reference._qsa(x[:1].unsqueeze(0), state, p, step)
                assert torch.equal(block.output, expected.reshape(1, -1).expand_as(block.output))
                ids = helper.buffer("qsa.selected", capacity, 3, dtype=torch.int32)
                selected = [0] if step == 0 else [0, 1] + ([step] if step % 2 == 0 else [])
                wanted = torch.tensor(selected + [-1] * (3 - len(selected)), device=device)
                assert torch.equal(ids, wanted.expand_as(ids))
                expected_positions = torch.tensor([max(0, step - 1), step], device=device)
                assert torch.equal(history[0, :, 0], expected_positions)
                assert torch.equal(history, history[:1].expand_as(history))
                if step in common:
                    assert torch.equal(block.output[0], common[step])
                common[step] = block.output[0].clone()

            if capacity > 1:
                snapshot = history[1:].clone()
                active[1:] = 0
                source[1:] = 999999
                destination[1:] = 999999
                helper.buffer("qsa.group_rows", capacity, 3, dtype=torch.int64).fill_(-(2**60))
                helper.buffer("qsa.selected_counts", capacity, dtype=torch.int32).fill_(999999)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(history[1:], snapshot)
                assert torch.equal(block.output[0], common[8])
                assert (
                    helper.buffer("qsa.selected_counts", capacity, dtype=torch.int32)[1:] == 0
                ).all()


@pytest.mark.parametrize("grouped", [False, True])
def test_linear_group_keeps_binding_scratch_and_fallback_order(grouped):
    from types import SimpleNamespace

    calls = []
    scratch = [object() for _ in range(3)]
    weights, outputs = (object(), object()), (object(), object())

    def bind(index):
        return SimpleNamespace(scratch=(scratch[index],), reusable_scratch=False,
                               launch=lambda: calls.append(index))

    class Operators:
        def bind_linear_groups(self, x, members, targets):
            assert members == weights and targets == outputs
            return (bind(2),) if grouped else (bind(0), bind(1))

    helper = TextGraphBindings(Operators(), torch.device("cuda"), torch.bfloat16)
    launch = helper.linear_group(object(), weights, outputs)
    assert calls == []
    assert helper.scratch == ((scratch[2],) if grouped else tuple(scratch[:2]))
    launch()
    launch()
    assert calls == ([2, 2] if grouped else [0, 1, 0, 1])


def test_linear_group_overlap_rejects_shared_storage_before_pricing():
    from types import SimpleNamespace
    from unittest.mock import patch

    x = torch.ones(1, 16)
    weights = (torch.ones(4, 16), torch.ones(4, 16))
    outputs = (torch.empty(1, 4), torch.empty(1, 4))
    backing, calls = torch.empty(16), []
    bindings = tuple(SimpleNamespace(
        scratch=(backing[i:i + 8],), output=outputs[i], reusable_scratch=False,
        launch=lambda i=i: calls.append(i),
    ) for i in (0, 1))
    operators = SimpleNamespace(bind_linear_groups=lambda *args: bindings)
    helper = TextGraphBindings(operators, torch.device("cuda", 0), torch.bfloat16)
    with patch("cubic_serve.executor.text_graph.device_work_cost",
               side_effect=AssertionError("alias must reject before pricing")):
        launch = helper.linear_group(x, weights, outputs, overlap=True)
    launch()
    assert calls == [0, 1]
    assert helper._projection_branches is None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_linear_group_overlap_role_reuse_joins_before_consumers(dtype):
    from types import SimpleNamespace
    from unittest.mock import patch

    from cubic_serve.guardrail.quantization import check_linear_enclosure
    from tests.ops.test_int_quantization import fixture

    configure_reference_math()
    torch.manual_seed(149)
    device = torch.device("cuda", 0)
    integer, _ = fixture(8, n=320, k=2048, group=64, sym=True)
    plain = torch.randn(4, 2048, device=device, dtype=dtype) * 0.01
    weights = (integer, plain)
    policy = OperatorConfig(activation_dtype="fp32", attention_dtype="fp32")
    with operator_strategy_scope(policy, device="", model=""):
        for rows in (1, 4, 32):
            helper = TextGraphBindings(PackedLinearOperators(policy), device, dtype)
            inputs = [torch.randn(rows, 2048, device=device, dtype=dtype) for _ in range(2)]
            outputs = [tuple(torch.empty(rows, w.shape[0], device=device, dtype=dtype)
                             for w in weights) for _ in inputs]
            snapshots = [tuple(torch.empty_like(y) for y in group) for group in outputs]
            # Force the eligible branch to test lifetimes independently of
            # device cost fitting and public-serving performance evidence.
            with patch("cubic_serve.executor.text_graph.device_work_cost",
                       return_value=SimpleNamespace(branch_costs=lambda *a, **k: (2.0, 1.0))):
                calls = [helper.linear_group(x, weights, group, overlap=True)
                         for x, group in zip(inputs, outputs, strict=True)]
            assert helper._projection_branches is not None
            for call in calls:
                call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for call, group, saved in zip(calls, outputs, snapshots, strict=True):
                    call()
                    for output, snapshot in zip(group, saved, strict=True):
                        snapshot.copy_(output)
            for _ in range(3):
                for x in inputs:
                    x.mul_(-0.5)
                for group in outputs:
                    for output in group:
                        output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                for x, group in zip(inputs, snapshots, strict=True):
                    for weight, output in zip(weights, group, strict=True):
                        check_linear_enclosure(x, weight, output)
            graph.reset()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_linear_group_graph_independent_and_cross_m(dtype):
    from cubic_serve.guardrail.quantization import check_linear_enclosure

    configure_reference_math()
    torch.manual_seed(631)
    device = torch.device("cuda", 0)
    policy = OperatorConfig(activation_dtype="fp32", attention_dtype="fp32")
    k = 2048
    weights = tuple(torch.randn(n, k, device=device).to(dtype) * 0.01 for n in (128, 8, 8, 64))
    corpus = torch.randn(1, k, device=device).to(dtype)
    common = None
    with operator_strategy_scope(policy, device="", model=""):
        for capacity in (1, 4, 32):
            x = corpus.expand(capacity, -1).contiguous()
            outputs = tuple(torch.empty(capacity, w.shape[0], device=device, dtype=dtype)
                            for w in weights)
            helper = TextGraphBindings(PackedLinearOperators(config=policy), device, dtype)
            launch = helper.linear_group(x, weights, outputs)
            launch()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch()
            graph.replay()
            for weight, output in zip(weights, outputs, strict=True):
                check_linear_enclosure(x, weight, output)
            if common is None:
                common = tuple(output[:1].clone() for output in outputs)
            for expected, output in zip(common, outputs, strict=True):
                assert torch.equal(expected.expand_as(output), output)
            graph.reset()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("combine_norm", [False, True])
def test_hc_early_injection_and_serial_group_scratch(dtype, combine_norm):
    from cubic_serve.ops.eager import hyperconnection

    configure_reference_math()
    policy = OperatorConfig(activation_dtype="fp32", attention_dtype="fp32")
    with operator_strategy_scope(policy, device="", model=""):
        torch.manual_seed(553)
        device = torch.device("cuda", 0)
        streams, width, rows = 4, 512, 4
        full = streams * width
        # Constant rows isolate projection ordering and scratch lifetime from
        # the independent norm/exp reduction allowances audited separately.
        residual = torch.tensor([-1, 1, -2, 2], device=device, dtype=dtype)[:, None].expand(
            rows, full
        ).contiguous()
        original = residual.clone()
        helper = TextGraphBindings(PackedLinearOperators(config=policy), device, dtype)
        calls, operands = [], []
        pending = None
        for layer in range(2):
            norm = torch.zeros(full, device=device, dtype=dtype)
            down = torch.zeros(128, full, device=device, dtype=dtype)
            up = torch.zeros(full, 128, device=device, dtype=dtype)
            injection = torch.zeros(streams, full, device=device, dtype=dtype)
            down[torch.arange(128), (torch.arange(128) + layer) % full] = 0.5
            up[torch.arange(full), (torch.arange(full) + layer) % 128] = 0.5
            injection[torch.arange(streams), torch.arange(streams) + layer] = 0.5
            logits = helper.buffer("hc.injection_logits", rows, streams)
            mixed, normalized = helper.mix(residual, norm, down, up, streams=streams,
                                           epsilon=1e-6, injection=(injection, logits),
                                           residual_update=pending)
            combined = helper.inject(residual, mixed.output, normalized, injection,
                                     streams=streams, logits=logits)
            calls.append(mixed.launch)
            if combine_norm and layer == 0:
                pending = mixed.output, logits
            else:
                calls.append(combined.launch)
            operands.append((norm, down, up, injection))
        private = [key for key in helper._buffers if key[0] == "linear.grouped.private"]
        assert len(private) == 1
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for call in calls:
                call()
        residual.copy_(original)
        graph.replay()
        with ReferenceExecutionContext():
            expected = original
            for norm, down, up, injection in operands:
                mixed, normalized = hyperconnection.mix(expected, norm, down, up,
                                                        streams=streams, eps=1e-6)
                gates = hyperconnection.injection(normalized, injection, streams=streams)
                expected = hyperconnection.combine(expected, mixed, gates)
        assert torch.equal(residual, expected)
        graph.reset()


@pytest.mark.parametrize("successor", ["mix", "intervening", "end"])
def test_model_residual_fusion_requires_immediate_mix(successor):
    from types import SimpleNamespace

    from cubic_serve.models.qwen4_exp.model import Qwen4ExpOperationBinder

    events, mixtures = [], []

    class Helper:
        def buffer(self, role, *shape):
            return torch.zeros(shape)

        def mix(self, *args, **kwargs):
            pending = kwargs.get("residual_update")
            mixtures.append(pending)
            label = "combined_mix" if pending is not None else "mix"
            return (SimpleNamespace(output=torch.zeros(1, 4),
                                    launch=lambda: events.append(label)), torch.zeros(1, 8))

        def inject(self, *args, **kwargs):
            return SimpleNamespace(launch=lambda: events.append("inject"))

    config = SimpleNamespace(text=SimpleNamespace(
        hidden_size=4, hyperconnection=SimpleNamespace(stream_count=2),
        moe=SimpleNamespace(expert_count=2), rms_norm_epsilon=1e-6,
    ))
    resources = SimpleNamespace(hidden=torch.zeros(1, 8), first_expert=0, local_expert_count=2,
                                embedding=lambda: events.append("intervening"))
    weights = {f"{prefix}.{name}": torch.zeros(8) for prefix in ("a", "b") for name in (
        "hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight",
        "block_inject_weight.weight",
    )}
    binder = Qwen4ExpOperationBinder(config, weights, Helper(), resources)
    first = binder.bind("hc_mix", "a", layer_id=0)
    update = binder.update = torch.ones(1, 4)
    old_logits = binder.injection_logits
    inject = binder.bind("hc_inject", "a", layer_id=0)
    middle = binder.bind("embedding", "", layer_id=None) if successor == "intervening" else None
    next_mix = binder.bind("hc_mix", "b", layer_id=1) if successor != "end" else None
    first()
    inject()
    if middle is not None:
        middle()
    if next_mix is not None:
        next_mix()
    if successor == "mix":
        assert events == ["mix", "combined_mix"]
        assert mixtures[-1][0] is update and mixtures[-1][1] is old_logits
    elif successor == "end":
        assert events == ["mix", "inject"]
    else:
        assert events == ["mix", "inject", "intervening", "mix"]
        assert mixtures[-1] is None


@pytest.mark.parametrize('element,pattern', [(1, 0xa5), (2, 0x1234), (4, 0x12345678)])
def test_conditional_transport_preserves_bytes_tail_and_source_ownership(element, pattern):
    from cuda.bindings import driver as d
    from cubic_serve.executor.conditional_graph import add_conditional_child, _nodes
    from cubic_serve.executor.resident_window import check, conditional_handle, add_conditional_body

    # Prefix/suffix sentinels witness the exact write set as well as typed bytes.
    storage = torch.full((513 * element + 32,), 99, dtype=torch.uint8, device='cuda')
    output = storage[16:-16]
    source = check(d.cuGraphCreate(0))[0]
    root = check(d.cuGraphCreate(0))[0]
    executable = None
    try:
        params = d.CUDA_MEMSET_NODE_PARAMS()
        params.dst, params.elementSize = output.data_ptr(), element
        params.width, params.height, params.value = 513, 1, pattern
        context = check(d.cuCtxGetCurrent())[0]
        check(d.cuGraphAddMemsetNode(source, [], 0, params, context))
        handle = conditional_handle(root, 1)
        _, body = add_conditional_body(root, [], handle, while_loop=False)
        added = add_conditional_child(body, [], source)
        embedded = check(d.cuGraphChildGraphNodeGetGraph(added))[0]
        assert [check(d.cuGraphNodeGetType(n))[0] for n in _nodes(embedded)] == [
            d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL]
        assert [check(d.cuGraphNodeGetType(n))[0] for n in _nodes(source)] == [
            d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET]
        # Source definitions and temporary captures need not survive instantiation.
        check(d.cuGraphDestroy(source))
        source = None
        executable = check(d.cuGraphInstantiate(root, 0))[0]
        stream = torch.cuda.Stream()
        for _ in range(3):
            storage.fill_(99)
            stream.wait_stream(torch.cuda.current_stream())
            check(d.cuGraphLaunch(executable, stream.cuda_stream))
            stream.synchronize()
            expected = bytes([99] * 16) + pattern.to_bytes(element, 'little') * 513 + bytes([99] * 16)
            assert bytes(storage.cpu().tolist()) == expected
    finally:
        if executable is not None:
            check(d.cuGraphExecDestroy(executable))
        check(d.cuGraphDestroy(root))
        if source is not None:
            check(d.cuGraphDestroy(source))


def test_conditional_copy_preserves_nested_fanout_and_original_child():
    from cuda.bindings import driver as d
    from cubic_serve.executor.conditional_graph import add_conditional_child, _nodes
    from cubic_serve.executor.resident_window import check, conditional_handle, add_conditional_body

    x = torch.arange(3073, device='cuda', dtype=torch.int32).to(torch.uint8)
    a, b, c = (torch.empty_like(x) for _ in range(3))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captured = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(captured, stream=stream):
        a.copy_(x)
        b.copy_(a)
        c.copy_(a)
        b.add_(1)
        c.add_(2)
    captured.instantiate()
    source = check(d.cuGraphCreate(0))[0]
    check(d.cuGraphAddChildGraphNode(source, [], 0, captured.raw_cuda_graph()))
    root = check(d.cuGraphCreate(0))[0]
    executable = None
    try:
        handle = conditional_handle(root, 1)
        _, body = add_conditional_body(root, [], handle, while_loop=False)
        add_conditional_child(body, [], source)
        original_types = [check(d.cuGraphNodeGetType(n))[0]
                          for n in _nodes(captured.raw_cuda_graph())]
        assert original_types.count(d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY) == 3
        check(d.cuGraphDestroy(source))
        source = None
        captured.reset()
        executable = check(d.cuGraphInstantiate(root, 0))[0]
        for _ in range(3):
            x.random_(0, 256)
            stream.wait_stream(torch.cuda.current_stream())
            check(d.cuGraphLaunch(executable, stream.cuda_stream))
            stream.synchronize()
            expected = x.cpu()
            assert torch.equal(a.cpu(), expected)
            assert torch.equal(b.cpu(), expected + 1)
            assert torch.equal(c.cpu(), expected + 2)
    finally:
        if executable is not None:
            check(d.cuGraphExecDestroy(executable))
        check(d.cuGraphDestroy(root))
        if source is not None:
            check(d.cuGraphDestroy(source))
        captured.reset()
