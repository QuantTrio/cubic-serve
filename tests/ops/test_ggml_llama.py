"""Independent Q8_1 operand proof and original GGML integer-dot boundaries."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from gguf import GGML_QUANT_SIZES
from gguf import GGMLQuantizationType as Kind

from cubic_serve.quantization.api import RoutedRows
from cubic_serve.quantization.extensions.ggml.llama import bind_matmul
from cubic_serve.quantization.extensions.ggml.reference import decode_rows

KINDS = (Kind.Q8_0, Kind.Q6_K, Kind.IQ3_S, Kind.IQ4_NL, Kind.IQ4_XS)
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_vector_register_reuse_respects_thread_waves_and_device_resources():
    from cubic_serve.quantization.extensions.ggml.policy import routed_output_rows_per_warp

    assert routed_output_rows_per_warp(n=128, routes=4, sms=8, max_threads_per_sm=1024) == 2
    assert routed_output_rows_per_warp(n=512, routes=4, sms=8, max_threads_per_sm=1024) == 4
    assert routed_output_rows_per_warp(n=512, routes=4, sms=16, max_threads_per_sm=1024) == 2
    assert routed_output_rows_per_warp(n=512, routes=8, sms=16, max_threads_per_sm=1024) == 4
    with pytest.raises(ValueError, match="positive"):
        routed_output_rows_per_warp(n=0, routes=4, sms=8, max_threads_per_sm=1024)


@pytest.fixture(scope="module", autouse=True)
def _independent_reference_math():
    from cubic_serve.executor.runner import configure_reference_math

    configure_reference_math()



def test_upstream_files_match_pinned_source_archive():
    root = (Path(__file__).resolve().parents[2]
            / "src/cubic_serve/quantization/extensions/ggml/upstream")
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["commit"] == "0c1e57098bba43ac29e6e3b677cdceebdd22334f"
    for item in manifest["files"]:
        digest = hashlib.sha256((root / item["local_path"]).read_bytes()).hexdigest()
        assert digest == item["sha256"]
        assert item["modifications"] == []


def _quantize(x):
    """Coordinate reference, independent of CUDA shuffles and pointwise dots."""
    values = x.float().cpu().numpy().reshape(x.shape[0], -1, 32)
    scale = np.max(np.abs(values), axis=-1, keepdims=True) / np.float32(127)
    ratio = np.divide(values, scale, out=np.zeros_like(values), where=scale != 0)
    magnitude = np.abs(ratio)
    integral = np.floor(magnitude)
    codes = np.copysign(integral + (magnitude - integral >= 0.5), ratio).astype(np.int8)
    rounded = scale.astype(np.float16).astype(np.float32)
    return codes, rounded


def _weight(kind, n, k):
    block, size = GGML_QUANT_SIZES[kind]
    rng = np.random.default_rng(37 + int(kind))
    raw = rng.integers(0, 256, (n, k // block, size), dtype=np.uint8)
    offset = 208 if kind == Kind.Q6_K else 0
    raw[:, :, offset:offset + 2] = rng.uniform(-0.02, 0.02, (n, k // block)).astype(
        "<f2").view(np.uint8).reshape(n, -1, 2)
    raw = raw.reshape(n, -1)
    return raw, torch.from_numpy(decode_rows(raw, int(kind), k))


def _check(actual, x, weights):
    codes, scale = _quantize(x)
    operands = torch.from_numpy((codes.astype(np.float32) * scale).reshape(x.shape)).double()
    w = weights.double()
    exact = operands @ w.T
    # Integer dots of up to 32 signed bytes are exact. The outer FP32 scale
    # products and sums have at most 2*K+64 elementary rounding operations.
    operations = 2 * x.shape[1] + 64
    gamma = operations * 2**-24 / (1 - operations * 2**-24)
    radius = gamma * (operands.abs() @ w.abs().T)
    low, high = (exact - radius).to(x.dtype).double(), (exact + radius).to(x.dtype).double()
    value = actual.double().cpu()
    assert bool(((value >= low) & (value <= high)).all())


def test_q8_group_uses_geometry_and_sm_supply_without_changing_m1():
    from cubic_serve.quantization.extensions.ggml.q8_group import selected

    assert selected(m=4, widths=(10240,), sms=156)
    assert not selected(m=4, widths=(2560,), sms=156)
    assert selected(m=8, widths=(2560,), sms=156)
    assert not selected(m=1, widths=(5120, 3072), sms=156)
    assert selected(m=4, widths=(5120, 3072), sms=156)
    assert not selected(m=4, widths=(320, 320), sms=156)
    assert selected(m=4, widths=(320, 320), sms=20)
    assert not selected(m=4, widths=(320, 320), sms=21)
    with pytest.raises(ValueError, match="positive"):
        selected(m=0, widths=(320, 320), sms=20)


@CUDA
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("k", (320, 512))
@pytest.mark.parametrize("m", (1, 2, 4, 17))
@pytest.mark.parametrize("members", (1, 2))
@pytest.mark.parametrize("arithmetic", ("ordered", "full32"))
def test_q8_group_independent_math_member_tails_and_original_reduction(
    dtype, k, m, members, arithmetic,
):
    from cubic_serve.quantization.common.binding import PackedOperatorBinding
    from cubic_serve.quantization.extensions.ggml.q8_group import bind_pair, bind_single

    widths = (37, 69)[:members]
    originals = tuple(_weight(Kind.Q8_0, n, k) for n in widths)
    packed = tuple(torch.from_numpy(raw).cuda() for raw, _ in originals)
    torch.manual_seed(769)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    activity = torch.ones(m, device="cuda", dtype=torch.int32)
    if m > 1:
        activity[1] = 0
        x[1].fill_(float("nan"))
    storage = tuple(torch.full((m, n + 7), 19, device="cuda", dtype=dtype) for n in widths)
    outputs = tuple(t[:, :n] for t, n in zip(storage, widths, strict=True))
    baseline = tuple(torch.empty_like(y) for y in outputs)
    old = tuple(bind_matmul(x, w, 8, y, activity=activity, algorithm="vector")
                for w, y in zip(packed, baseline, strict=True))
    launch = (bind_single(x, packed[0], outputs[0], old[0].scratch[0], activity=activity)
              if members == 1 else
              bind_pair(x, packed, outputs, old[0].scratch[0], activity=activity))
    if arithmetic == "full32":
        from cubic_serve.quantization.extensions.ggml.q8_mma import bind as bind_mma

        launch = bind_mma(x, packed, outputs, old[0].scratch[0], activity=activity)
        assert launch.shared_bytes == (27392 if k > 512 else 0)
    binding = PackedOperatorBinding(outputs[0], (old[0]._launches[0], launch),
                                    scratch=old[0].scratch)
    for member in old:
        member.launch()
    binding.launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.launch()
    for value in storage:
        value.fill_(19)
    graph.replay()
    torch.cuda.synchronize()
    kept = activity.cpu() > 0
    for y, reference, (_, weight), full, n in zip(
        outputs, baseline, originals, storage, widths, strict=True,
    ):
        _check(y[kept], x[kept], weight)
        if arithmetic == "ordered":
            assert torch.equal(y[kept], reference[kept])
        assert bool((full[:, n:] == 19).all())
        if m > 1:
            assert bool((full[1] == 19).all())
    graph.reset()


@CUDA
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("k", (640, 4096))
@pytest.mark.parametrize("m", (1, 17))
@pytest.mark.parametrize("members", (1, 2))
def test_q8_pipeline_independent_math_tails_and_protected_storage(dtype, k, m, members):
    test_q8_group_independent_math_member_tails_and_original_reduction(
        dtype, k, m, members, "full32",
    )


def test_q8_pipeline_price_protects_register_and_near_crossover_paths():
    from cubic_serve.quantization.extensions.ggml.policy import select_pipelined_q8_matrix

    def choose(m, n, k, *, shared=True, resident=2):
        return select_pipelined_q8_matrix(
            m=m, n=n, k=k, sms=156, matrix_shared=shared,
            matrix_resident=resident, pipeline_resident=3,
        )

    assert choose(512, 320, 10240)
    assert choose(1024, 320, 10240)
    assert choose(256, 1536, 2560)
    assert choose(512, 640, 5120)  # Held out from fitting.
    assert not choose(512, 1536, 2560, shared=False, resident=5)
    assert not choose(512, 2560, 2560, shared=False, resident=5)
    assert not choose(512, 1024, 2560)  # Less than the protected margin.
    assert not choose(256, 320, 512)
    with pytest.raises(ValueError, match="positive"):
        choose(0, 320, 10240)


@CUDA
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("m", (1, 3, 17))
def test_dense_original_packed_dot_and_graph_padding(kind, dtype, m):
    k, n = 512, 37
    raw, weight = _weight(kind, n, k)
    torch.manual_seed(73)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    x[0].zero_()
    output = torch.full((m, n + 11), 19, device="cuda", dtype=dtype)
    active = torch.ones(m, device="cuda", dtype=torch.int32)
    binding = bind_matmul(x, torch.from_numpy(raw).cuda(), int(kind), output[:, :n],
                          activity=active)
    binding.launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.launch()
    graph.replay()
    torch.cuda.synchronize()
    _check(output[:, :n], x, weight)
    codes, _ = _quantize(x)
    qbytes = binding.scratch[0].cpu().numpy()
    np.testing.assert_array_equal(qbytes[:, :, 4:].view(np.int8), codes)
    if m > 1:
        active[1] = 0
        output[1].fill_(19)
        graph.replay()
        torch.cuda.synchronize()
        assert bool((output[1] == 19).all())
    assert bool((output[:, n:] == 19).all())


@CUDA
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("matrix_rows", (16, 32))
def test_matrix_original_blocks_and_partial_rows(kind, dtype, matrix_rows):
    k, n, m = 512, 67, 17
    raw, weight = _weight(kind, n, k)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    output = torch.full((m, n + 7), 19, device="cuda", dtype=dtype)
    activity = torch.ones(m, device="cuda", dtype=torch.int32)
    activity[3] = 0
    binding = bind_matmul(x, torch.from_numpy(raw).cuda(), int(kind), output[:, :n],
                          activity=activity, algorithm="matrix", matrix_rows=matrix_rows)
    assert binding.row_tile_size == matrix_rows
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.launch()
    graph.replay()
    torch.cuda.synchronize()
    kept = torch.arange(m) != 3
    _check(output[kept, :n], x[kept], weight)
    assert bool((output[3] == 19).all())
    assert bool((output[:, n:] == 19).all())


@CUDA
@pytest.mark.parametrize("kind", (Kind.Q8_0, Kind.IQ4_NL))
@pytest.mark.parametrize("k", (32, 96, 160, 288, 640))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("matrix_rows", (16, 32))
def test_matrix_short_final_weight_tile(kind, k, dtype, matrix_rows):
    raw, weight = _weight(kind, 67, k)
    x = torch.randn(17, k, device="cuda", dtype=dtype)
    activity = torch.ones(17, device="cuda", dtype=torch.int32)
    activity[3] = 0
    x[3].fill_(float("nan"))
    storage = torch.full((17, 74), 19, device="cuda", dtype=dtype)
    binding = bind_matmul(x, torch.from_numpy(raw).cuda(), int(kind), storage[:, :67],
                          activity=activity, algorithm="matrix", matrix_rows=matrix_rows)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.launch()
    graph.replay()
    torch.cuda.synchronize()
    kept = activity.cpu() > 0
    _check(storage[kept, :67], x[kept], weight)
    assert bool((storage[3] == 19).all())
    assert bool((storage[:, 67:] == 19).all())
    if kind == Kind.IQ4_NL:
        assert binding._launches[-1].shared_bytes == 0
    graph.reset()


def test_matrix_work_rule_retains_small_projections_and_single_row():
    from cubic_serve.quantization.extensions.ggml.policy import select_matrix

    resources = {"sms": 156, "l2_bytes": 128 << 20}
    assert not select_matrix(m=1, n=124160, k=2560, row_bytes=2100, **resources)
    assert select_matrix(m=4, n=124160, k=2560, row_bytes=2100, **resources)
    assert select_matrix(m=128, n=5120, k=2560, row_bytes=2720, **resources)
    assert not select_matrix(m=128, n=320, k=10240, row_bytes=10880, **resources)
    assert not select_matrix(m=128, n=320, k=2560, row_bytes=2720, **resources)


@CUDA
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("top_k", (1, 3))
@pytest.mark.parametrize("short_tail", (False, True))
@pytest.mark.parametrize("matrix_rows", (8, 16))
def test_matrix_routed_prefix_and_permuted_original_owners(kind, top_k, short_tail, matrix_rows):
    import math

    if short_tail and kind not in (Kind.Q8_0, Kind.IQ4_NL):
        pytest.skip("this encoding owns complete 256-value blocks")
    m, experts, n, k = 17, 7, 67, 512
    if short_tail:
        k = 160
    routes = m * top_k
    raw, weight = _weight(kind, experts * n, k)
    counts = [routes // experts + (j < routes % experts) for j in range(experts)]
    counts[1] += counts[0]
    counts[0] = 0
    starts = np.cumsum([0, *counts[:-1]]).tolist()
    blocks = np.cumsum([0, *(math.ceil(c / matrix_rows) for c in counts[:-1])]).tolist()
    mapping = list(reversed(range(routes)))
    rows = RoutedRows(*(torch.tensor(v, device="cuda", dtype=torch.int32) for v in (
        counts, starts, blocks, [sum(math.ceil(c / matrix_rows) for c in counts)], mapping,
    )), routing_top_k=top_k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    output = torch.full((routes, n + 5), 19, device="cuda", dtype=x.dtype)
    activity = torch.ones(m, device="cuda", dtype=torch.int32)
    activity[2] = 0
    binding = bind_matmul(x, torch.from_numpy(raw.reshape(experts, n, -1)).cuda(),
                          int(kind), output[:, :n], activity=activity, rows=rows,
                          input_top_k=top_k, algorithm="matrix", matrix_rows=matrix_rows)
    assert binding.row_tile_size == matrix_rows
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.launch()
    graph.replay()
    torch.cuda.synchronize()
    for expert, (start, count) in enumerate(zip(starts, counts, strict=True)):
        for route in mapping[start:start + count]:
            row = route // top_k
            if row == 2:
                assert bool((output[route] == 19).all())
            else:
                _check(output[route:route + 1, :n], x[row:row + 1],
                       weight[expert * n:(expert + 1) * n])
    assert bool((output[:, n:] == 19).all())


def test_matrix_transport_prices_k_preparation_and_compiled_residency():
    from cubic_serve.quantization.extensions.ggml.policy import select_register_matrix_transport

    resources = dict(row_tile=16, sms=156, shared_resident=2, register_resident=5)
    # Short original blocks and sufficient supplied waves favor registers.
    assert select_register_matrix_transport(m=64, n=10240, k=320, **resources)
    assert select_register_matrix_transport(m=192, n=2048, k=640, **resources)
    assert select_register_matrix_transport(m=256, n=2560, k=2560, **resources)
    # Preserve the shared loader for underfilled long-K blocks.
    assert not select_register_matrix_transport(m=48, n=512, k=512, **resources)
    assert not select_register_matrix_transport(m=48, n=320, k=1280, **resources)
    assert not select_register_matrix_transport(m=128, n=2560, k=2560, **resources)
    with pytest.raises(ValueError, match="geometry/resources"):
        select_register_matrix_transport(m=0, n=64, k=32, **resources)
    with pytest.raises(ValueError, match="finite"):
        select_register_matrix_transport(m=64, n=64, k=32, margin=float("nan"), **resources)


def test_routed_matrix_rule_rejects_sparse_owner_padding_cost():
    from cubic_serve.quantization.extensions.ggml.policy import select_routed_matrix

    geometry = dict(experts=512, n=320, k=2560, row_bytes=1100,
                    sms=156, l2_bytes=112 << 20)
    assert not select_routed_matrix(mean_rows=None, **geometry)
    assert not select_routed_matrix(mean_rows=0.625, **geometry)
    assert select_routed_matrix(mean_rows=1.25, **geometry)
    assert select_routed_matrix(mean_rows=2.5, **geometry)
    assert select_routed_matrix(mean_rows=5.0, **geometry)


def test_routed_decode_work_preserves_sparse_and_uncompressed_pricing():
    from cubic_serve.quantization.extensions.ggml.policy import select_routed_matrix

    geometry = dict(experts=512, n=160, k=2560, row_bytes=1100,
                    sms=156, l2_bytes=112 << 20)
    for mean in (0.01953125, 0.078125, 0.3125, 0.625, 0.9375, 1.09375):
        assert select_routed_matrix(mean_rows=mean, **geometry) == select_routed_matrix(
            mean_rows=mean, decode_reuse_work=0.0, **geometry,
        )
    for mean in (2.5, 5.0):
        assert select_routed_matrix(mean_rows=mean, **geometry)
        assert not select_routed_matrix(mean_rows=mean, decode_reuse_work=0.0, **geometry)
    geometry.update(n=320, row_bytes=2720)
    for mean in (0.01953125, 0.625, 1.25, 2.5, 5.0):
        assert select_routed_matrix(mean_rows=mean, **geometry) == select_routed_matrix(
            mean_rows=mean, decode_reuse_work=0.0, **geometry,
        )
    for coefficient in (-1.0, float("inf")):
        with pytest.raises(ValueError, match="decode work"):
            select_routed_matrix(mean_rows=1.0, decode_reuse_work=coefficient, **geometry)
    # An otherwise identical direct Q8 dot needs more expert reuse. Applying
    # the IQ crossover to it regressed the measured complete draft block.
    geometry["row_bytes"] = 2720
    assert not select_routed_matrix(mean_rows=2.5, **geometry)
    assert select_routed_matrix(mean_rows=5.0, **geometry)


@CUDA
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("top_k", (1, 2))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_routed_original_expert_ownership_and_permuted_rows(kind, top_k, dtype):
    m, e, n, k = 3, 3, 19, 512
    raw, weight = _weight(kind, e * n, k)
    raw = raw.reshape(e, n, -1)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    routes = m * top_k
    mapping = list(reversed(range(routes)))
    counts_cpu = [routes // e + (j < routes % e) for j in range(e)]
    starts_cpu = np.cumsum([0, *counts_cpu[:-1]]).tolist()
    counts = torch.tensor(counts_cpu, device="cuda", dtype=torch.int32)
    starts = torch.tensor(starts_cpu, device="cuda", dtype=torch.int32)
    row_map = torch.tensor(mapping, device="cuda", dtype=torch.int32)
    rows = RoutedRows(counts, starts, None, None, row_map, routing_top_k=top_k)
    active = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32)
    output = torch.full((routes, n), 31, device="cuda", dtype=x.dtype)
    binding = bind_matmul(x, torch.from_numpy(raw).cuda(), int(kind), output,
                          rows=rows, input_top_k=top_k, activity=active)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.launch()
    graph.replay()
    torch.cuda.synchronize()
    for expert, (start, count) in enumerate(zip(starts_cpu, counts_cpu, strict=True)):
        for route in mapping[start:start + count]:
            row = route // top_k
            if row == 1:
                assert bool((output[route] == 31).all())
            else:
                _check(output[route:route + 1], x[row:row + 1], weight[expert*n:(expert+1)*n])


@CUDA
@pytest.mark.parametrize("kind", KINDS)
def test_registered_format_binding_and_independent_guardrail(kind):
    from cubic_serve.executor.runner import ReferenceExecutionContext
    from cubic_serve.guardrail.quantization import check_linear_enclosure
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.ops.eager.quantization import linear_matrix
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.extensions.ggml.checkpoint import scheme

    raw, _ = _weight(kind, 19, 512)
    matrix = QuantizedMatrix({"weight_packed": torch.from_numpy(raw)},
                              scheme(int(kind)), (19, 512))
    x = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(3, 19, device="cuda", dtype=x.dtype)
    operators = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
    binding = operators.bind_linear(x, matrix, output)
    binding.launch()
    torch.cuda.synchronize()
    check_linear_enclosure(x, matrix, output, compute_mode="wna8")
    with ReferenceExecutionContext():
        reference = linear_matrix(x, matrix, compute_mode="wna8")
    check_linear_enclosure(x, matrix, reference, compute_mode="wna8")


@CUDA
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_grouped_a8_uses_shared_binding_with_independent_projection_outputs(kind, dtype):
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.extensions.ggml.checkpoint import scheme

    m, k = 17, 512
    matrices, weights, outputs = [], [], []
    for n in (19, 67):
        raw, weight = _weight(kind, n, k)
        matrices.append(QuantizedMatrix({"weight_packed": torch.from_numpy(raw)},
                                        scheme(int(kind)), (n, k)))
        weights.append(weight)
        outputs.append(torch.empty(m, n, device="cuda", dtype=dtype))
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    operators = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
    binding = operators.bind_linear_group(x, tuple(matrices), tuple(outputs))
    assert binding is not None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.launch()
    graph.replay()
    torch.cuda.synchronize()
    for output, weight in zip(outputs, weights, strict=True):
        _check(output, x, weight)


@CUDA
def test_q8_1_group_direction_rounding_and_scale_storage():
    from cubic_serve.quantization.extensions.ggml.reference import wna8_operand

    x = torch.zeros(1, 64, device="cuda", dtype=torch.float16)
    x[0, :5] = torch.tensor([127, 0.5, -0.5, 1.5, -1.5], device=x.device)
    x[0, 32:] = torch.linspace(-0.001, 0.001, 32, device=x.device)
    raw, _ = _weight(Kind.Q8_0, 1, 64)
    output = torch.empty(1, 1, device=x.device, dtype=x.dtype)
    binding = bind_matmul(x, torch.from_numpy(raw).cuda(), int(Kind.Q8_0), output)
    binding.launch()
    torch.cuda.synchronize()
    codes, scale = _quantize(x)
    stored = binding.scratch[0].cpu().numpy()
    np.testing.assert_array_equal(stored[:, :, 4:].view(np.int8), codes)
    np.testing.assert_array_equal(stored[:, :, :2].copy().view("<f2").reshape(1, 2, 1),
                                  scale.astype(np.float16))
    assert codes[0, 0, :5].tolist() == [127, 1, -1, 2, -2]
    expected = torch.from_numpy((codes.astype(np.float32) * scale).reshape(1, 64)).to(x.device)
    assert torch.equal(wna8_operand(x), expected)


def test_reference_roundf_does_not_turn_near_ties_into_ties():
    from cubic_serve.quantization.extensions.ggml.reference import wna8_operand

    x = torch.zeros(1, 32)
    x[0, :5] = torch.tensor([127.0, 0.5 - 2**-25, -0.5 + 2**-25, 0.5, -0.5])
    assert wna8_operand(x)[0, :5].tolist() == [127.0, 0.0, -0.0, 1.0, -1.0]


@CUDA
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_reference_and_native_q8_1_div_rn_at_half_integer_boundaries(dtype):
    from cubic_serve.quantization.extensions.ggml.reference import wna8_operand

    maxima = torch.arange(128, 256, device="cuda", dtype=torch.float32)
    x = (maxima[:, None] * torch.arange(32, device="cuda")[None, :] / 32).to(dtype)
    x[:, -1] = maxima.to(dtype)
    packed = torch.zeros(1, 34, device=x.device, dtype=torch.uint8)
    output = torch.empty(128, 1, device=x.device, dtype=dtype)
    binding = bind_matmul(x, packed, int(Kind.Q8_0), output)
    binding.launch()
    torch.cuda.synchronize()
    raw = binding.scratch[0]
    codes = raw[:, :, 4:].view(torch.int8).reshape_as(x).float()
    scale = raw[:, :, :2].contiguous().view(torch.float16).float().reshape(128, 1)
    assert torch.equal(wna8_operand(x), codes * scale)


@pytest.mark.parametrize("routed,reduction", ((False, "matrix"), (False, "mma"), (True, "matrix")))
@pytest.mark.parametrize("wide", (False, True))
def test_ggml_arithmetic_plan_replays_selected_reduction_at_relative_rows(routed, reduction, wide):
    from types import SimpleNamespace

    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.common.small_rows_cost import (
        row_arithmetic_plan,
        row_arithmetic_scope,
    )
    from cubic_serve.quantization.extensions.ggml.checkpoint import scheme
    from cubic_serve.quantization.extensions.ggml.policy import (
        GGMLLinearParameters,
        GGMLMoEParameters,
        bind_arithmetic,
    )

    weight = QuantizedMatrix({"weight_packed": torch.zeros(64, 34, dtype=torch.uint8)},
                             scheme(8), (64, 32))
    params = GGMLMoEParameters() if routed else GGMLLinearParameters()
    owner = SimpleNamespace(config=OperatorConfig(compute_mode="wna8"))
    extent = 3 if routed else 1
    options = dict(routed=routed, experts=7 if routed else 1, routing_top_k=extent)
    seen = []

    selected_rows = (8 if routed else 16) * (2 if wide else 1)

    def native(algorithm, matrix_rows):
        seen.append((algorithm, matrix_rows))
        mode = reduction if algorithm == "auto" else algorithm
        identities = {"matrix": "cubic.cuda.ggml.llama.mmq.v1",
                      "mma": "cubic.cuda.ggml.q8_mma.v1"}
        return SimpleNamespace(_launches=(SimpleNamespace(operator_id=identities[mode]),),
                               row_tile_size=selected_rows if matrix_rows is None else matrix_rows)

    bind_arithmetic(owner, torch.empty(4 * extent, 32), weight, params, native, **options)
    assert seen == [("auto", None)]
    plan = row_arithmetic_plan(owner, 4)
    anchor = SimpleNamespace(config=owner.config)
    with row_arithmetic_scope(plan):
        bind_arithmetic(anchor, torch.empty(extent, 32), weight, params, native, **options)
    assert seen[-1] == (reduction, selected_rows if reduction == "matrix" else None)
    assert row_arithmetic_plan(anchor, 1) == plan
    with row_arithmetic_scope({}), pytest.raises(ValueError, match="absent"):
        bind_arithmetic(anchor, torch.empty(extent, 32), weight, params, native, **options)
    bad = {key: ("ggml-mmq-v1", 1, 0) for key in plan}
    with row_arithmetic_scope(bad), pytest.raises(ValueError, match="row tile"):
        bind_arithmetic(anchor, torch.empty(extent, 32), weight, params, native, **options)
    if reduction == "matrix":
        def ignores_forced_rows(algorithm, matrix_rows):
            result = native(algorithm, matrix_rows)
            result.row_tile_size = (8 if routed else 16) * (1 if wide else 2)
            return result

        with row_arithmetic_scope(plan), pytest.raises(ValueError, match="requested reduction"):
            bind_arithmetic(anchor, torch.empty(extent, 32), weight, params,
                            ignores_forced_rows, **options)


@CUDA
def test_ggml_matrix_anchor_retains_independently_checked_row_arithmetic():
    from types import SimpleNamespace

    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.common.small_rows_cost import (
        row_arithmetic_plan,
        row_arithmetic_scope,
    )
    from cubic_serve.quantization.extensions.ggml.checkpoint import scheme
    from cubic_serve.quantization.extensions.ggml.policy import (
        GGMLLinearParameters,
        bind_arithmetic,
    )

    raw, decoded = _weight(Kind.Q8_0, 67, 512)
    packed = torch.from_numpy(raw).cuda()
    weight = QuantizedMatrix({"weight_packed": packed}, scheme(8), (67, 512))
    torch.manual_seed(761)
    x = torch.randn(17, 512, device="cuda", dtype=torch.bfloat16)
    outputs = []
    owner = SimpleNamespace(config=OperatorConfig(compute_mode="wna8"))
    params = GGMLLinearParameters()

    def build(ops, inputs, default):
        y = torch.empty(inputs.shape[0], 67, device="cuda", dtype=inputs.dtype)
        binding = bind_arithmetic(
            ops, inputs, weight, params,
            lambda algorithm, matrix_rows: bind_matmul(
                inputs, packed, 8, y, algorithm=algorithm, matrix_rows=matrix_rows,
            ),
            default=default,
        )
        binding.launch()
        outputs.append(y)

    build(owner, x, "matrix")
    with row_arithmetic_scope(row_arithmetic_plan(owner, 17)):
        build(SimpleNamespace(config=owner.config), x[:1], "auto")
    torch.cuda.synchronize()
    _check(outputs[0], x, decoded)
    _check(outputs[1], x[:1], decoded)
    assert torch.equal(outputs[0][:1], outputs[1])


@CUDA
def test_ggml_forced_routed_matrix_returns_its_actual_prefix_tile():
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.common.small_rows_cost import (
        row_arithmetic_plan,
        row_arithmetic_scope,
    )
    from cubic_serve.quantization.extensions.ggml.checkpoint import scheme
    from cubic_serve.quantization.extensions.ggml.moe import bind_grouped
    from cubic_serve.quantization.extensions.ggml.policy import GGMLMoEParameters

    raw, decoded = _weight(Kind.Q8_0, 67, 512)
    weight = QuantizedMatrix({"weight_packed": torch.from_numpy(raw).cuda()}, scheme(8), (67, 512))
    weights = (weight,) * 7
    x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    counts = torch.tensor([4, 4, 4, 0, 0, 0, 0], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 4, 8, 12, 12, 12, 12], device="cuda", dtype=torch.int32)
    blocks = torch.tensor([0, 1, 2, 3, 3, 3, 3], device="cuda", dtype=torch.int32)
    total = torch.tensor([3], device="cuda", dtype=torch.int32)
    mapping = torch.tensor([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11],
                           device="cuda", dtype=torch.int32)
    rows = RoutedRows(counts, starts, blocks, total, mapping, 12 / 7, 3)
    ops = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
    output = torch.empty(12, 134, device="cuda", dtype=x.dtype)
    tile, _ = bind_grouped(ops, x, weights, weights, output, rows, 3, GGMLMoEParameters())
    assert tile == 1
    plan = {key: ("ggml-mmq-v1", 8, 0) for key in row_arithmetic_plan(ops, 4)}
    single = RoutedRows(
        counts.clamp(max=1), blocks, blocks, total,
        torch.arange(3, device="cuda", dtype=torch.int32), 3 / 7, 3,
    )
    output = torch.empty(3, 134, device="cuda", dtype=x.dtype)
    with row_arithmetic_scope(plan):
        tile, binding = bind_grouped(
            PackedLinearOperators(ops.config), x[:1], weights, weights, output, single, 3,
            GGMLMoEParameters(),
        )
    assert tile == 8
    binding.launch()
    torch.cuda.synchronize()
    operands = x[:1].repeat_interleave(3, dim=0)
    _check(output[:, :67], operands, decoded)
    _check(output[:, 67:], operands, decoded)


@CUDA
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_routed_group_preserves_original_experts_tail_rows_and_single_storage(kind, dtype):
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.common.small_rows_cost import (
        row_arithmetic_plan, row_arithmetic_scope,
    )
    from cubic_serve.quantization.extensions.ggml.checkpoint import scheme
    from cubic_serve.quantization.extensions.ggml.moe import bind_grouped
    from cubic_serve.quantization.extensions.ggml.policy import GGMLMoEParameters

    n, k, experts, m = 37, 512, 3, 4
    raw, _ = _weight(kind, n, k)
    gate, up, decoded = [], [], []
    for expert in range(experts):
        g = np.roll(raw, expert, axis=0).copy()
        u = np.roll(raw[::-1], expert + 1, axis=0).copy()
        gate.append(QuantizedMatrix({"weight_packed": torch.from_numpy(g)},
                                    scheme(int(kind)), (n, k)))
        up.append(QuantizedMatrix({"weight_packed": torch.from_numpy(u)},
                                  scheme(int(kind)), (n, k)))
        decoded.append((torch.from_numpy(decode_rows(g, int(kind), k)),
                        torch.from_numpy(decode_rows(u, int(kind), k))))
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    x[-1].fill_(float("nan"))
    rows = RoutedRows(
        torch.tensor([3, 3, 3], device="cuda", dtype=torch.int32),
        torch.tensor([0, 3, 6], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32),
        torch.tensor([3], device="cuda", dtype=torch.int32),
        torch.tensor([0, 3, 6, 1, 4, 7, 2, 5, 8, -1, -1, -1],
                     device="cuda", dtype=torch.int32),
        3.0, 3,
    )
    ops = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
    storage = torch.full((m * 3, 2 * n + 6), 97, device="cuda", dtype=dtype)
    output = storage[:, 3:3 + 2 * n]
    bind_grouped(ops, x, tuple(gate), tuple(up), output, rows, 3, GGMLMoEParameters())
    plan = row_arithmetic_plan(ops, 1)
    assert plan
    for recipe in (("ggml-mmvq-v1", 0, 0), ("ggml-mmq-v1", 8, 0)):
        output[:9].fill_(float("nan"))
        candidate_ops = PackedLinearOperators(ops.config, prepared_weights=ops.prepared_weights)
        with row_arithmetic_scope({key: recipe for key in plan}):
            _, binding = bind_grouped(candidate_ops, x, tuple(gate), tuple(up), output, rows, 3,
                                      GGMLMoEParameters())
        binding.launch()
        torch.cuda.synchronize()
        for expert, (g, u) in enumerate(decoded):
            _check(output[expert:9:3, :n], x[:3], g)
            _check(output[expert:9:3, n:], x[:3], u)
        assert bool((storage[:, :3] == 97).all())
        assert bool((storage[:, -3:] == 97).all())
        assert bool((output[9:] == 97).all())
    device_weight_bytes = sum(value.numel() * value.element_size()
                              for _, value in ops.prepared_weights.values())
    if kind == Kind.Q8_0:
        # One aligned carrier per original expert, shared by both recipes.
        # Charge its real padded N bytes; never retain a raw device replica.
        assert device_weight_bytes == experts * ((2 * n + 15) // 16) * 16 * (k // 32) * 34
        assert not ops.materialized_weights
    else:
        assert device_weight_bytes == 2 * experts * raw.nbytes


def test_ggml_group_context_keeps_distinct_reductions_for_shared_weights():
    from types import SimpleNamespace

    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.common.small_rows_cost import row_arithmetic_plan
    from cubic_serve.quantization.extensions.ggml.checkpoint import scheme
    from cubic_serve.quantization.extensions.ggml.policy import (
        GGMLLinearParameters,
        bind_arithmetic,
    )

    weight = QuantizedMatrix({"weight_packed": torch.zeros(64, 34, dtype=torch.uint8)},
                             scheme(8), (64, 32))
    owner = SimpleNamespace(config=OperatorConfig(compute_mode="wna8"))
    for group, mode in (((), "vector"), ((weight, weight), "mma")):
        identity = ("cubic.cuda.ggml.llama.mmvq.v1" if mode == "vector"
                    else "cubic.cuda.ggml.q8_mma.v1")
        binding = SimpleNamespace(_launches=(SimpleNamespace(operator_id=identity),))
        bind_arithmetic(owner, torch.empty(8, 32), weight, GGMLLinearParameters(),
                        lambda algorithm, matrix_rows, binding=binding: binding, group=group)
    assert len(row_arithmetic_plan(owner, 8)) == 2


def test_wide_matrix_reuse_preserves_resource_and_cache_boundaries():
    from cubic_serve.quantization.extensions.ggml.policy import prefer_wider_matrix_rows

    geometry = dict(m=512, n=320, k=2560, row_bytes=1100, experts=512,
                    mean_rows=10, sms=156, l2_bytes=117440512,
                    baseline_rows=8, wider_rows=16, baseline_resident=4,
                    wider_resident=3, baseline_local=32, wider_local=0)
    assert prefer_wider_matrix_rows(**geometry)
    assert prefer_wider_matrix_rows(**dict(geometry, row_bytes=1360,
                                          baseline_resident=3, wider_resident=2))
    # Held-out shorter K retains its cache-resident baseline. A near-flat
    # low-occupancy IQ payload does not become a selected wider candidate.
    assert not prefer_wider_matrix_rows(**dict(geometry, k=1280, row_bytes=680))
    assert not prefer_wider_matrix_rows(**dict(geometry, m=64, mean_rows=1.25,
                                              row_bytes=1360, baseline_resident=3,
                                              wider_resident=2))
    for changes in (dict(l2_bytes=1 << 30), dict(wider_local=33),
                    dict(wider_resident=1), dict(mean_rows=None),
                    dict(mean_rows=float("nan")), dict(mean_rows=0)):
        assert not prefer_wider_matrix_rows(**dict(geometry, **changes))
    dense = dict(geometry, n=62080, row_bytes=2100, experts=1, mean_rows=None,
                 baseline_rows=16, wider_rows=32, baseline_resident=2,
                 wider_resident=2, baseline_local=64)
    assert prefer_wider_matrix_rows(**dense)
    assert prefer_wider_matrix_rows(**dict(dense, m=17))
    assert not prefer_wider_matrix_rows(**dict(dense, m=1))
    for changes in (dict(sms=0), dict(wider_local=-1),
                    dict(dot_byte_work=float("nan")), dict(margin=.9)):
        with pytest.raises(ValueError):
            prefer_wider_matrix_rows(**dict(geometry, **changes))


def test_bound_row_tile_declaration_requires_an_exact_positive_integer():
    from cubic_serve.quantization.common.binding import PackedOperatorBinding

    output = torch.empty(1)
    assert PackedOperatorBinding(output, (), row_tile_size=16).row_tile_size == 16
    for invalid in (True, 0, -1, 16.0):
        with pytest.raises(ValueError, match="row tile size"):
            PackedOperatorBinding(output, (), row_tile_size=invalid)


@CUDA
def test_q8_mma_format_audit_anchor_forces_the_declared_arithmetic_at_m1():
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.common.binding import PackedOperatorBinding
    from cubic_serve.quantization.common.small_rows_cost import (
        row_arithmetic_plan,
        row_arithmetic_scope,
    )
    from cubic_serve.quantization.extensions.ggml.checkpoint import scheme
    from cubic_serve.quantization.extensions.ggml.q8_mma import bind as bind_mma

    raw, decoded = _weight(Kind.Q8_0, 67, 512)
    packed = torch.from_numpy(raw).cuda()
    weight = QuantizedMatrix({"weight_packed": packed}, scheme(8), (67, 512))
    config = OperatorConfig(compute_mode="wna8")
    owner = PackedLinearOperators(config)
    x = torch.randn(17, 512, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(17, 67, device="cuda", dtype=x.dtype)
    original = owner.bind_linear(x, weight, output)
    plan = {key: (("ggml-full32-v1", 8, 8)
                  if key[0][0] == "ggml.row-arithmetic.v1" else recipe)
            for key, recipe in row_arithmetic_plan(owner, 17).items()}
    source = PackedOperatorBinding(
        output, (original._launches[0], bind_mma(x, (packed,), (output,), original.scratch[0])),
        scratch=original.scratch,
    )
    anchor_output = torch.empty(1, 67, device="cuda", dtype=x.dtype)
    with row_arithmetic_scope(plan):
        anchor = PackedLinearOperators(config).bind_linear(x[:1], weight, anchor_output)
    assert anchor._launches[-1].operator_id == "cubic.cuda.ggml.q8_mma.v1"
    source.launch()
    anchor.launch()
    torch.cuda.synchronize()
    _check(output, x, decoded)
    _check(anchor_output, x[:1], decoded)
    assert torch.equal(output[:1], anchor_output)
