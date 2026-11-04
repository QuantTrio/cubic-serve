"""Storage arithmetic checks, separate from CUDA timing/model qualification."""

import numpy as np
import pytest
from gguf import GGML_QUANT_SIZES, dequantize
from gguf import GGMLQuantizationType as Kind

from cubic_serve.models.quantization import QuantizationError
from cubic_serve.quantization.extensions.ggml.reference import decode_rows


@pytest.mark.parametrize("kind,scale_offset", [
    (Kind.Q8_0, 0), (Kind.Q6_K, 208), (Kind.IQ3_S, 0),
    (Kind.IQ4_NL, 0), (Kind.IQ4_XS, 0),
])
def test_random_original_blocks_match_upstream_decoder(kind, scale_offset):
    rng = np.random.default_rng(401)
    block, size = GGML_QUANT_SIZES[kind]
    # Cover sign/high bits and adjacent blocks/rows, with finite signed scales.
    raw = rng.integers(0, 256, (17 * 3, size), dtype=np.uint8)
    raw[:, scale_offset:scale_offset + 2] = rng.uniform(-2, 2, raw.shape[0]).astype(
        "<f2"
    ).view(np.uint8).reshape(-1, 2)
    raw = raw.reshape(17, 3 * size)
    original = raw.copy()
    np.testing.assert_array_equal(decode_rows(raw, int(kind), 3 * block), dequantize(raw, kind))
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("strided", [False, True])
def test_q6_quarters_preserve_scalar_fp32_boundaries_and_storage(strided):
    rng = np.random.default_rng(941)
    raw = rng.integers(0, 256, (3, 4, 210), dtype=np.uint8)
    scale_bits = np.array([0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00,
                           0x7bff, 0xfbff, 0x0400, 0x8400, 0x3555, 0xb555], dtype="<u2")
    raw[:, :, 208:210] = scale_bits.view(np.uint8).reshape(3, 4, 2)
    raw[:, :, 192:208] = np.resize(np.array([-128, -1, 0, 1, 127], dtype=np.int8),
                                  (3, 4, 16)).view(np.uint8)
    payload = raw.reshape(3, -1)
    if strided:
        backing = np.full((3, payload.shape[1] * 2), 93, dtype=np.uint8)
        backing[:, ::2] = payload
        payload = backing[:, ::2]
    original = payload.copy()
    expected = np.empty((3, 1024), dtype=np.float32)
    for row in range(3):
        for block_id in range(4):
            block = raw[row, block_id]
            d = np.float32(block[208:210].copy().view("<f2")[0])
            for position in range(256):
                half, quarter, lane = position // 128, position % 128 // 32, position % 32
                ql = int(block[half * 64 + (quarter % 2) * 32 + lane])
                qh = int(block[128 + half * 32 + lane])
                code = ((ql >> (4 * (quarter // 2))) & 15) | (((qh >> (2 * quarter)) & 3) << 4)
                scale = block[192 + position // 16:193 + position // 16].view(np.int8)[0]
                expected[row, block_id * 256 + position] = (
                    np.float32(d * np.float32(scale)) * np.float32(code - 32)
                )
    actual = decode_rows(payload, int(Kind.Q6_K), 1024)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(payload, original)
    if strided:
        np.testing.assert_array_equal(backing[:, 1::2], 93)


def test_known_iq3_sign_and_group_scale_boundaries():
    raw = np.zeros((1, 110), dtype=np.uint8)
    raw[:, :2] = np.frombuffer(np.float16(1).tobytes(), dtype=np.uint8)
    raw[0, 74] = 0b10000001
    raw[0, 106] = 0x21
    expected = np.ones((1, 256), dtype=np.float32)
    expected[0, :32] = 3
    expected[0, 32:64] = 5
    expected[0, [0, 7]] *= -1
    np.testing.assert_array_equal(decode_rows(raw, int(Kind.IQ3_S), 256), expected)


def test_iq3_all_half_scale_encodings_preserve_coordinate_arithmetic():
    rng = np.random.default_rng(502)
    raw = rng.integers(0, 256, (65536, 110), dtype=np.uint8)
    raw[:, :2] = np.arange(65536, dtype="<u2").view(np.uint8).reshape(-1, 2)
    with np.errstate(invalid="ignore", over="ignore"):
        actual = decode_rows(raw, int(Kind.IQ3_S), 256)
        expected = dequantize(raw, Kind.IQ3_S)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    owned = ~np.isnan(expected)
    # Includes signed zero, subnormal scales and signed infinity. NaN payload
    # propagation is not a finite weight's numerical contract.
    np.testing.assert_array_equal(actual.view(np.uint32)[owned], expected.view(np.uint32)[owned])


@pytest.mark.parametrize("kind", [Kind.F32, Kind.F16, Kind.BF16])
def test_plain_storage_preserves_representable_values(kind):
    values = np.array([[0, 1, -2, 0.5]], dtype=np.float32)
    if kind == Kind.BF16:
        payload = (values.view(np.uint32) >> 16).astype("<u2").view(np.uint8)
    else:
        payload = values.astype("<f4" if kind == Kind.F32 else "<f2").view(np.uint8)
    np.testing.assert_array_equal(decode_rows(payload, int(kind), 4), values)


def test_invalid_geometry_is_rejected():
    with pytest.raises(QuantizationError):
        decode_rows(np.zeros((1, 18), dtype=np.uint8), int(Kind.IQ4_NL), 31)
    with pytest.raises(QuantizationError):
        decode_rows(np.zeros((1, 19), dtype=np.uint8), int(Kind.IQ4_NL), 32)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("kind,scale_offset", [
    (Kind.Q8_0, 0), (Kind.Q6_K, 208), (Kind.IQ3_S, 0),
    (Kind.IQ4_NL, 0), (Kind.IQ4_XS, 0),
])
@pytest.mark.parametrize("tile,decode_limit", [(32, None), (128, 192)])
def test_prepared_reference_keeps_dot_shapes_bytes_and_inputs(
    device, dtype_name, kind, scale_offset, tile, decode_limit, monkeypatch,
):
    import importlib

    import torch

    from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.eager.quantization import linear_matrix
    from cubic_serve.quantization.extensions.ggml import reference
    from cubic_serve.quantization.extensions.ggml.checkpoint import scheme

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    configure_reference_math()
    torch.set_num_threads(1)
    rng = np.random.default_rng(281)
    block, size = GGML_QUANT_SIZES[kind]
    n, k = 273, 256
    raw = rng.integers(0, 256, (n * (k // block), size), dtype=np.uint8)
    raw[:, scale_offset:scale_offset + 2] = np.frombuffer(
        np.float16(0.001).tobytes(), dtype=np.uint8
    )
    packed = torch.from_numpy(raw.reshape(n, -1).copy()).to(device)
    matrix = QuantizedMatrix({"weight_packed": packed}, scheme(int(kind)), (n, k))
    x = torch.linspace(-0.2, 0.2, 3 * k).reshape(3, k).to(
        device=device, dtype=getattr(torch, dtype_name)
    )
    x_before, packed_before = x.clone(), packed.clone()
    linear_module = importlib.import_module("cubic_serve.ops.eager.linear")
    original_linear = linear_module.linear
    shapes = []

    def observed(a, b, **kwargs):
        shapes.append((tuple(a.shape), tuple(b.shape)))
        return original_linear(a, b, **kwargs)

    monkeypatch.setattr(linear_module, "linear", observed)
    prepare = reference.prepare_wna8_linear
    with ReferenceExecutionContext():
        monkeypatch.setattr(reference, "prepare_wna8_linear", lambda *a, **kw: None)
        expected = linear_matrix(x, matrix, output_rows=tile, compute_mode="wna8")
        old_shapes = list(shapes)
        shapes.clear()
        monkeypatch.setattr(reference, "prepare_wna8_linear", prepare)
        operand = reference.wna8_operand
        calls = []

        def observed_operand(a):
            calls.append(tuple(a.shape))
            return operand(a)

        monkeypatch.setattr(reference, "wna8_operand", observed_operand)
        actual = linear_matrix(x, matrix, output_rows=tile, compute_mode="wna8",
                               decode_rows=decode_limit)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert shapes == old_shapes
    assert calls == [tuple(x.shape)]
    assert torch.equal(x.view(torch.int16), x_before.view(torch.int16))
    assert torch.equal(packed, packed_before)


@pytest.mark.parametrize("kind", [Kind.Q8_0, Kind.Q6_K, Kind.IQ3_S, Kind.IQ4_NL, Kind.IQ4_XS])
@pytest.mark.parametrize("sliced", [False, True])
def test_decoded_storage_is_owned_and_independent(kind, sliced):
    import torch

    from cubic_serve.quantization.extensions.ggml.checkpoint import scheme
    from cubic_serve.quantization.extensions.ggml.reference import decode

    block, size = GGML_QUANT_SIZES[kind]
    k = max(block, 32)
    packed = torch.zeros((2, k // block * size), dtype=torch.uint8)
    original = packed.clone()
    offset = 1 if sliced else 0
    encoding = scheme(int(kind), offset=offset)
    a = decode({"weight_packed": packed}, encoding, (2, k - offset))
    b = decode({"weight_packed": packed}, encoding, (2, k - offset))
    a.fill_(7)
    assert torch.equal(packed, original)
    assert torch.count_nonzero(b) == 0
    assert a.is_contiguous() and b.is_contiguous()


def test_prepared_reference_declines_insufficient_workspace():
    import torch

    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.quantization.extensions.ggml.checkpoint import scheme
    from cubic_serve.quantization.extensions.ggml.reference import prepare_wna8_linear

    matrix = QuantizedMatrix({"weight_packed": torch.zeros((32, 34), dtype=torch.uint8)},
                             scheme(int(Kind.Q8_0)), (32, 32))
    assert prepare_wna8_linear(torch.zeros((1, 32)), matrix, output_rows=32,
                               byte_budget=1) is None


@pytest.mark.parametrize("operation", ["linear", "grouped", "routed"])
def test_partial_block_wna8_fails_before_materialization(operation):
    from types import SimpleNamespace

    import torch

    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.quantization.extensions.ggml import linear, moe
    from cubic_serve.quantization.extensions.ggml.checkpoint import scheme
    from cubic_serve.quantization.extensions.ggml.policy import (
        GGMLLinearParameters,
        GGMLMoEParameters,
    )

    operators = SimpleNamespace(config=SimpleNamespace(compute_mode="wna8"))
    weight = QuantizedMatrix({"weight_packed": torch.zeros((2, 18), dtype=torch.uint8)},
                             scheme(int(Kind.IQ4_NL), offset=1), (2, 31))
    x, out = torch.empty((1, 32)), torch.empty((1, 2))
    with pytest.raises(ValueError, match="does not support wna8"):
        if operation == "linear":
            linear.bind_native(operators, x, weight, out, GGMLLinearParameters())
        elif operation == "grouped":
            moe.bind_grouped(operators, x, (weight,), (weight,), out, None, 1,
                             GGMLMoEParameters())
        else:
            moe.bind_routed(operators, x, (weight,), out, None, GGMLMoEParameters())


@pytest.mark.parametrize("tokens", [1, 4, 8, 17, 128])
@pytest.mark.parametrize("top_k", [1, 3, 10])
def test_default_vector_selection_preserves_gate_down_source_row_identity(tokens, top_k):
    from cubic_serve.quantization.extensions.ggml.policy import select_vector_rows

    dense = select_vector_rows(-1, input_rows=tokens)
    gate = select_vector_rows(-1, input_rows=tokens, input_top_k=top_k, routing_top_k=top_k)
    down = select_vector_rows(-1, input_rows=tokens * top_k, routing_top_k=top_k)
    assert gate == down
    assert gate == (1 if tokens <= 4 else 0)
    assert dense == (1 if tokens <= 16 else 0)
    assert select_vector_rows(4, input_rows=tokens) == 4


def test_q8_affine_transport_preserves_every_signed_code_and_exact_scale_arithmetic():
    from types import SimpleNamespace

    import torch

    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.quantization.extensions.ggml.affine import prepare
    from cubic_serve.quantization.extensions.ggml.checkpoint import scheme
    from cubic_serve.quantization.extensions.int.reference import decode

    scales = np.array([2**-24, 2**-14, 0.125, 1, 17, 128, 1024, 65504], dtype="<f2")
    raw = np.empty((64, 8, 34), dtype=np.uint8)
    raw[:, :, :2] = scales.view(np.uint8).reshape(8, 2)
    raw[:, :, 2:] = np.arange(256, dtype=np.uint8).reshape(8, 32)
    matrix = QuantizedMatrix({"weight_packed": torch.from_numpy(raw.reshape(64, -1))},
                              scheme(int(Kind.Q8_0)), (64, 256))
    operators = SimpleNamespace(materialized_weights={})
    proxy = prepare(operators, matrix)
    assert proxy is not None
    assert prepare(operators, matrix) is proxy
    expected = torch.from_numpy(decode_rows(raw.reshape(64, -1), int(Kind.Q8_0), 256))
    actual = decode(proxy.payload, proxy.encoding, proxy.shape, None)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert proxy.payload["weight_packed"].dtype == torch.int32
    assert proxy.payload["weight_scale"].dtype == torch.float32


@pytest.mark.parametrize("scale", [0.0, -0.0, -1.0])
def test_q8_transport_does_not_weaken_shared_positive_scale_contract(scale):
    from types import SimpleNamespace

    import torch

    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.quantization.extensions.ggml.affine import prepare
    from cubic_serve.quantization.extensions.ggml.checkpoint import scheme

    raw = np.zeros((64, 34), dtype=np.uint8)
    raw[:, :2] = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    matrix = QuantizedMatrix({"weight_packed": torch.from_numpy(raw)},
                              scheme(int(Kind.Q8_0)), (64, 32))
    assert prepare(SimpleNamespace(materialized_weights={}), matrix) is None
