"""FP8 token-scale contract, native capture and physical row ownership."""

import os

import pytest
import torch

from cubic_serve.executor.runner import configure_reference_math
from cubic_serve.guardrail.quantization import check_linear_enclosure
from cubic_serve.models.quantization import ActivationFormat, QuantizationScheme, WeightFormat
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
from cubic_serve.quantization.extensions.fp8 import linear
from cubic_serve.quantization.extensions.fp8.policy import Fp8LinearParameters


def byte_exact(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
def test_native_output_shared_row_map_reuse():
    """Exercise persistent CTAs across row tiles, including under racecheck.

    One output tile per CTA cannot expose an output reader racing the next
    tile's shared row-map writer. Size the corpus from SM count, not a SKU.
    """
    configure_reference_math()
    torch.manual_seed(812)
    params = Fp8LinearParameters(marlin_blocks_per_sm=1)
    sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
    m = (sm_count + 1) * params.marlin_block_m
    n, k = params.marlin_block_n, params.marlin_block_k * 2
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = QuantizedMatrix(
        {
            "weight": (torch.randn(n, k, device="cuda") * 0.1).to(torch.float8_e4m3fn),
            "weight_scale": torch.full((n, 1), 0.1, device="cuda"),
        },
        QuantizationScheme(WeightFormat.FP8, 8, 1, activation=ActivationFormat.FP8_TOKEN),
        (n, k),
    )
    y = torch.empty(m, n, device="cuda", dtype=x.dtype)
    binding = linear.bind_linear(PackedLinearOperators(), x, weight, y, params)
    assert any(a.backend == "cuda-driver" for a in binding.project_artifacts())
    binding.launch()
    torch.cuda.synchronize()
    check_linear_enclosure(x, weight, y)
    expected = y.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.launch()
    y.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert byte_exact(expected, y)
    graph.reset()


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
@pytest.mark.parametrize("bits", [4, 8])
def test_k_packed_repack_preserves_every_code(bits):
    from cubic_serve.quantization.common.marlin.repack import repack_words

    configure_reference_math()
    # Two leading axes witness projection/expert ordering. The old N-packed
    # path is an independent integer layout witness, not a floating oracle.
    torch.manual_seed(493)
    codes = torch.randint(1 << bits, (2, 3, 128, 256), device="cuda")
    codes[0, 0, 0] = torch.arange(256, device="cuda") % (1 << bits)
    pack = 32 // bits
    shifts = torch.arange(pack, device="cuda") * bits
    k_words = (codes.reshape(2, 3, 128 // pack, pack, 256) << shifts[:, None]).sum(-2)
    order = [0, 2, 4, 6, 1, 3, 5, 7] if bits == 4 else [0, 2, 1, 3]
    n_words = (codes.reshape(2, 3, 128, 256 // pack, pack)[..., order] << shifts).sum(-1)
    actual = repack_words(k_words.int().contiguous(), bits=bits, axis="k")
    expected = repack_words(n_words.int().contiguous(), bits=bits)
    assert byte_exact(actual, expected)
    # A caller can prepare directly into final projection/expert slots.
    direct = torch.empty_like(actual)
    source = k_words.int().contiguous()
    for projection in range(2):
        for expert in range(3):
            result = repack_words(
                source[projection, expert], bits=bits, axis="k",
                output=direct[projection, expert],
            )
            assert result.data_ptr() == direct[projection, expert].data_ptr()
    assert byte_exact(direct, expected)
    with pytest.raises(ValueError, match="overlap"):
        repack_words(source, bits=bits, axis="k", output=source.view_as(actual))


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("checkpoint_device", ["cpu", "cuda"])
@pytest.mark.parametrize("n,k,external", [(256, 3072, False), (3072, 256, True)])
def test_token_scales_native_cross_m(dtype, n, k, external, checkpoint_device):
    configure_reference_math()
    torch.set_num_threads(1)
    torch.manual_seed(111)
    corpus = torch.randn(128, k, device="cuda", dtype=dtype) * 0.1
    corpus[0].zero_()
    w = (torch.randn(n, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    codes = torch.cat(
        (
            torch.arange(127, device="cuda", dtype=torch.uint8),
            torch.arange(128, 255, device="cuda", dtype=torch.uint8),
        )
    )
    w.view(torch.uint8)[0, :254] = codes
    weight = QuantizedMatrix(
        {
            "weight": w,
            "weight_scale": torch.rand(n, 1, device="cuda") * 0.1 + 0.001,
        },
        QuantizationScheme(WeightFormat.FP8, 8, 1, activation=ActivationFormat.FP8_TOKEN),
        (n, k),
    )
    weight = weight.to(checkpoint_device)
    operators = PackedLinearOperators()
    common = None
    for logical, cap in ((1, 1), (8, 8), (33, 64), (128, 128)):
        x = torch.full((cap, k), float("nan"), device="cuda", dtype=dtype)
        x[:logical].copy_(corpus[:logical])
        y = torch.empty(cap, n, device="cuda", dtype=dtype)
        scales = torch.empty(cap, device="cuda") if external else None
        scale_binding = linear.bind_token_scales(x, scales) if external else None
        binding = linear.bind_linear(operators, x, weight, y, Fp8LinearParameters(), scales)
        assert any(a.backend == "cuda-driver" for a in binding.project_artifacts())
        if checkpoint_device == "cpu":
            assert all(t.device.type == "cpu" for t in weight.payload.values())
            assert all(key[0] == "marlin.fp8.v1" for key in operators.prepared_weights)

        def launch(scale_binding=scale_binding, binding=binding):
            if scale_binding is not None:
                scale_binding.launch()
                # Simulate another TP shard with max |x| = 0.75. The complete-K
                # scale intentionally differs from this shard's local maximum.
                scale_binding.output.clamp_min_(0.75 / 448.0)
            binding.launch()

        launch()
        torch.cuda.synchronize()
        actual = y[:logical].clone()
        check_linear_enclosure(
            x[:logical], weight, actual, fp8_token_scales=scales[:logical] if external else None
        )
        if common is not None:
            assert byte_exact(common, actual[: len(common)])
        common = actual.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        y.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert byte_exact(actual, y[:logical])
        order = torch.randperm(logical, device="cuda")
        x[:logical].copy_(corpus[order])
        graph.replay()
        torch.cuda.synchronize()
        assert byte_exact(actual[order], y[:logical])
        graph.reset()


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("checkpoint_device", ["cpu", "cuda"])
@pytest.mark.parametrize("k", [256, 3072])
def test_grouped_original_scales_cross_m(dtype, checkpoint_device, k):
    from cubic_serve.ops.cuda.linear.quantized import PackedProjectionDispatch
    from cubic_serve.ops.strategies import OperatorConfig

    configure_reference_math()
    torch.set_num_threads(1)
    torch.manual_seed(912)
    scheme = QuantizationScheme(WeightFormat.FP8, 8, 1, activation=ActivationFormat.FP8_TOKEN)
    # Different widths and per-channel scales must not be collapsed/rescaled.
    weights = tuple(QuantizedMatrix({
        "weight": (torch.randn(n, k) * 0.2).to(torch.float8_e4m3fn),
        "weight_scale": torch.rand(n, 1) * scale + 0.001,
    }, scheme, (n, k)).to(checkpoint_device) for n, scale in ((128, 0.05), (256, 0.3)))
    # Mixed declared groups retain each format's arithmetic and member order.
    plain = torch.randn(16, k, device="cuda", dtype=dtype) * 0.01
    members = (weights[0], plain, weights[1])
    corpus = torch.randn(33, k, device="cuda", dtype=dtype) * 0.1
    common = None
    for logical, cap in ((1, 1), (8, 8), (33, 64)):
        results = []
        for grouped in (False, True):
            x = torch.full((cap, k), float("nan"), device="cuda", dtype=dtype)
            x[:logical].copy_(corpus[:logical])
            config = OperatorConfig.from_dict({"version": 1, "overrides": {
                "fp8_linear": {"grouped_projections": grouped},
            }})
            operators = PackedLinearOperators(config)
            outputs = tuple(torch.empty(cap, w.shape[0], device="cuda", dtype=dtype)
                            for w in members)
            dispatch = PackedProjectionDispatch(operators)
            dispatch.prepare_group(x, members, outputs)
            dispatch.seal()
            # A merged group does not also prepare redundant individual weights.
            assert len(operators.prepared_weights) == (3 if grouped else 6)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for w, y in zip(members, outputs, strict=True):
                    dispatch.launch(x, w, y)
            graph.replay()
            torch.cuda.synchronize()
            actual = tuple(y[:logical].clone() for y in outputs)
            for w, y in zip(members, actual, strict=True):
                check_linear_enclosure(x[:logical], w, y)
            order = torch.randperm(logical, device="cuda")
            x[:logical].copy_(corpus[order])
            graph.replay()
            torch.cuda.synchronize()
            assert all(byte_exact(a[order], y[:logical])
                       for a, y in zip(actual, outputs, strict=True))
            results.append(actual)
            graph.reset()
        assert all(byte_exact(a, b) for a, b in zip(*results, strict=True))
        if common is not None:
            assert all(byte_exact(a, b[:len(a)])
                       for a, b in zip(common, results[1], strict=True))
        common = results[1]


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
def test_grouped_output_ownership_and_scale_cache_identity():
    from cubic_serve.quantization.extensions.fp8.marlin import bind_grouped

    configure_reference_math()
    torch.manual_seed(711)
    x = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16) * 0.1
    raw = (torch.randn(128, 256) * 0.1).to(torch.float8_e4m3fn)
    scheme = QuantizationScheme(WeightFormat.FP8, 8, 1, activation=ActivationFormat.FP8_TOKEN)
    weights = tuple(QuantizedMatrix({"weight": raw, "weight_scale": torch.full((128, 1), s)},
                                   scheme, (128, 256)) for s in (0.1, 0.2))
    outputs = tuple(torch.empty(1, 128, device="cuda", dtype=x.dtype) for _ in weights)
    operators = PackedLinearOperators()
    binding = bind_grouped(operators, x, weights, outputs, Fp8LinearParameters())
    assert binding is not None
    binding.launch()
    torch.cuda.synchronize()
    assert not byte_exact(*outputs)
    for w, y in zip(weights, outputs, strict=True):
        check_linear_enclosure(x, w, y)
        single = operators.bind_linear(x, w, torch.empty_like(y))
        single.launch()
        torch.cuda.synchronize()
        assert byte_exact(y, single.output)
    with pytest.raises(ValueError, match="alias"):
        operators.bind_linear_group(x, weights, (outputs[0], outputs[0]))
    outputs[1].resize_(2, 64)
    with pytest.raises(ValueError, match="ABI"):
        binding.launch()
