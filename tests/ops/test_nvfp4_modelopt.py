import os

import pytest
import torch

from cubic_serve.models.quantization import QuantizationConfig, QuantizationError
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.eager.quantization import decode_matrix
from cubic_serve.quantization.extensions.nvfp4.reference import activation_from_payload


def config():
    declaration = {"dynamic": False, "group_size": 16, "num_bits": 4, "type": "float"}
    return QuantizationConfig.from_model_config(
        {
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "ignore": ["*.self_attn.*", "*hyper_connection*", "lm_head"],
                "config_groups": {
                    "main": {
                        "targets": ["Linear"],
                        "weights": declaration,
                        "input_activations": declaration,
                    }
                },
            }
        }
    )


def matrix():
    encoding = config().resolve("model.layers.0.mlp.experts.0.gate_proj")
    return QuantizedMatrix(
        {
            "weight": torch.full((8, 16), 0x22, dtype=torch.uint8),
            "weight_scale": torch.ones(8, 2).to(torch.float8_e4m3fn),
            "weight_scale_2": torch.tensor(0.125),
            "input_scale": torch.tensor(0.25),
        },
        encoding,
        (8, 32),
    )


def test_modelopt_globs_are_not_literal_or_substring_rules():
    c = config()
    assert c.resolve("model.layers.0.self_attn.q_proj") is None
    assert c.resolve("model.hyper_connection_mixer.input_mix_weight_down") is None
    assert c.resolve("lm_head") is None
    assert c.resolve("lm_head_extra") is not None


def test_modelopt_direct_scale_and_original_byte_slices():
    m = matrix()
    assert torch.equal(decode_matrix(m), torch.full((8, 32), 0.125))
    for sliced in (m.rows(2, 6), m.columns(16, 32)):
        assert torch.equal(decode_matrix(sliced), torch.full(sliced.shape, 0.125))
        assert sliced.payload["input_scale"].shape == ()
        assert sliced.payload["weight_scale_2"].data_ptr() == m.payload["weight_scale_2"].data_ptr()
    assert torch.equal(m.columns(16, 32).payload["weight"], m.payload["weight"][:, 8:])


def test_modelopt_activation_uses_direct_dequantization_scale():
    m = matrix()
    x = torch.full((2, 32), 1.5, dtype=torch.bfloat16)
    a = activation_from_payload(x, m.payload, m.encoding)
    assert torch.equal(a.dequantized, x.float())
    assert torch.equal(a.codes, torch.full_like(a.codes, 7))
    with pytest.raises(QuantizationError):
        activation_from_payload(x.fill_(float("nan")), m.payload, m.encoding)


def test_modelopt_scalar_slices_keep_fused_output_partition_geometry():
    source = matrix()
    fused = QuantizedMatrix(source.payload, source.encoding, source.shape, (4, 4))
    sliced = fused.rows(2, 6)
    assert sliced.partitions == (2, 2)
    assert sliced.payload["weight_scale_2"].shape == ()
    assert torch.equal(decode_matrix(sliced), torch.full((4, 32), 0.125))


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
@pytest.mark.parametrize("n,k", [(64, 128), (160, 2560), (2560, 160)])
def test_modelopt_cuda_linear_independent_enclosure_cross_m_and_graph(n, k):
    from cubic_serve.executor.runner import configure_reference_math
    from cubic_serve.guardrail.quantization import check_linear_enclosure
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators

    configure_reference_math()
    torch.manual_seed(903)
    encoding = matrix().encoding
    payload = {
        "weight": torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda"),
        "weight_scale": (torch.rand(n, k // 16, device="cuda") * 10).to(torch.float8_e4m3fn),
        "weight_scale_2": torch.tensor(0.001873, device="cuda"),
        "input_scale": torch.tensor(0.01317, device="cuda"),
    }
    weight = QuantizedMatrix(payload, encoding, (n, k))
    for dtype in (torch.bfloat16, torch.float16):
        x = torch.randn(33, k, dtype=dtype, device="cuda")
        common = None
        for m in (1, 8, 33):
            out = torch.empty(m, n, dtype=dtype, device="cuda")
            binding = PackedLinearOperators().bind_linear(x[:m], weight, out)
            assert any(
                call.operator_id == "cubic.cuda.nvfp4.marlin.linear.v1"
                for call in binding._launches
            )
            binding.launch()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                binding.launch()
            graph.replay()
            torch.cuda.synchronize()
            check_linear_enclosure(x[:m], weight, out)
            if common is not None:
                assert torch.equal(out[:1], common)
            common = out[:1].clone()


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
@pytest.mark.parametrize("hidden,intermediate", [(3072, 256), (2560, 160), (768, 144)])
def test_modelopt_routed_marlin_original_globals_padding_and_cross_m(
    hidden, intermediate, monkeypatch
):
    from dataclasses import replace

    from cubic_serve.quantization.extensions.nvfp4 import marlin
    from tests.ops.test_cuda_quantized_moe import PackedMoETests

    native = marlin.bind_projection_group
    native_shapes = []

    def require_native(*args, **kwargs):
        result = native(*args, **kwargs)
        assert result is not None, "tail audit must exercise native kernels, not fallback"
        native_shapes.append(args[2][0][0].shape)
        return result

    monkeypatch.setattr(marlin, "bind_projection_group", require_native)
    audit = PackedMoETests()
    audit.setUpClass()
    original = audit.matrix

    def modelopt(n, k, fmt, dtype, index):
        source = original(n, k, fmt, dtype, index)
        p = source.payload
        return QuantizedMatrix(
            {
                "weight": p["weight_packed"],
                "weight_scale": p["weight_scale"],
                "input_scale": p["input_global_scale"].reshape(()).reciprocal(),
                "weight_scale_2": p["weight_global_scale"].reshape(()).reciprocal(),
            },
            replace(source.encoding, metadata="modelopt"),
            source.shape,
        )

    audit.matrix = modelopt
    audit.audit_routed_formats(("nvfp4",), hidden, intermediate)
    assert (intermediate, hidden) in native_shapes
    assert (hidden, intermediate) in native_shapes
