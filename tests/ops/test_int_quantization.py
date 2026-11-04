"""Independent serialized-code, owner-slice and native arithmetic witnesses."""

import math
import os
import unittest
from contextlib import nullcontext
from itertools import product

import pytest
import torch

from cubic_serve.models.quantization import QuantizationScheme
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
from cubic_serve.quantization.extensions.int.layout import pack
from cubic_serve.quantization.extensions.int.reference import decode


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("packing", ["dense", "legacy"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_original_int_words_match_scalar_bitstream_with_signed_words_and_offsets(
    bits, packing, axis, device, cached,
):
    from cubic_serve.executor.runner import configure_reference_math
    from cubic_serve.quantization.common.reference import (
        IntegerGeometryMemo,
        integer_geometry_scope,
    )
    from cubic_serve.quantization.extensions.int.reference import unpack

    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA device required")
        configure_reference_math()
    geometry = IntegerGeometryMemo()
    patterns = (0, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0x12345678, 0xDEADBEEF)
    for offset in (0, 1, 15, 31):
        for count in (0, 1, 17, 65):
            words = max(1, math.ceil((offset + count) * bits / 32) if packing == "dense"
                        else math.ceil((offset + count) / (32 // bits)))
            rows = [[patterns[(j + row) % len(patterns)] for j in range(words)]
                    for row in range(3)]
            expected = []
            for row in rows:
                if packing == "dense":
                    stream = sum(word << (32 * j) for j, word in enumerate(row))
                    codes = [(stream >> (bits * j)) % (1 << bits)
                             for j in range(offset, offset + count)]
                else:
                    # Parse complete word-local slots independently, including
                    # unused top bits; then select the requested logical codes.
                    codes = [word // (1 << (slot * bits)) % (1 << bits)
                             for word in row for slot in range(32 // bits)]
                    codes = codes[offset:offset + count]
                expected.append([code - (1 << (bits - 1)) for code in codes])
            expected = torch.tensor(expected, dtype=torch.int8).reshape(3, count)
            signed = [[word if word < 1 << 31 else word - (1 << 32) for word in row]
                      for row in rows]
            packed = torch.tensor(signed, dtype=torch.int32)
            if axis == 0:
                packed, expected = packed.T.contiguous(), expected.T.contiguous()
            with integer_geometry_scope(geometry) if cached else nullcontext():
                actual = unpack(packed.to(device), bits, count, axis, packing, offset)
            assert torch.equal(actual.cpu(), expected)
            assert geometry.bytes <= geometry.max_bytes
            assert len(geometry.entries) <= geometry.max_entries


def test_original_int_unpack_keeps_wide_transport_low_word_semantics():
    from cubic_serve.quantization.extensions.int.reference import unpack

    packed = torch.tensor([[0x123480000000, 0x5678FFFFFFFF]], dtype=torch.int64)
    expected = torch.tensor([[-8] * 7 + [0] + [7] * 8], dtype=torch.int8)
    assert torch.equal(unpack(packed, 4, 16), expected)


def fixture(bits, n=64, k=256, group=128, sym=False, dtype=torch.float16, packing="dense"):
    generator = torch.Generator().manual_seed(bits * 100 + n + k)
    q = torch.randint(
        -(1 << (bits - 1)), 1 << (bits - 1), (n, k), generator=generator, dtype=torch.int8
    )
    s = (torch.rand(n, (k + group - 1) // group, generator=generator) * 0.03 + 0.001).to(dtype)
    z = torch.randint(
        -(1 << (bits - 1)), 1 << (bits - 1), s.shape, generator=generator, dtype=torch.int8
    )
    if sym:
        z.zero_()
    dt = {torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32"}[dtype]
    enc = QuantizationScheme(
        "int-pack-v1",
        bits,
        group,
        metadata=f"{packing}:{'sym' if sym else 'asym'}:{dt}",
    )
    payload = {
        "weight_packed": pack(q, bits, packing),
        "weight_scale": s,
        "weight_shape": torch.tensor([n, k]),
    }
    if not sym:
        payload["weight_zero_point"] = pack(z, bits, packing, 0)
    expected = (
        (
            (q.float() - z.float().repeat_interleave(group, 1)[:, :k])
            * s.float().repeat_interleave(group, 1)[:, :k]
        )
        .to(dtype)
        .float()
    )
    return QuantizedMatrix(payload, enc, (n, k)), expected


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1" or not torch.cuda.is_available(),
    reason="explicit native CUDA audit required",
)
@pytest.mark.parametrize("bits", range(2, 9))
@pytest.mark.parametrize("packing", ("dense", "legacy"))
@pytest.mark.parametrize("sym", (False, True))
@pytest.mark.parametrize("k,group,padded_group", ((256, 128, 128), (320, 96, 96)))
def test_int_a8_routed_group_transport_preserves_every_prepared_byte(
    bits, packing, sym, k, group, padded_group,
):
    from types import SimpleNamespace

    from cubic_serve.executor.runner import configure_reference_math
    from cubic_serve.quantization.extensions.int.marlin import (
        _prepare_a8_routed,
        combine_a8_matrices,
    )

    configure_reference_math()
    # Legacy odd-width zero words have word-local slots; 480 rows closes every
    # supported word. Dense 160 also covers the serving gate/up geometry.
    n = 160 if packing == "dense" else 480
    groups = tuple(tuple(
        fixture(bits, n=n, k=k, group=group, sym=sym, packing=packing)[0]
        for _ in range(3)
    ) for _ in range(2))
    sources = tuple(w for members in groups for w in members)
    for index, weight in enumerate(sources):
        weight.payload["weight_scale"].mul_(index + 1)
    snapshots = tuple(tuple(t.clone() for t in w.payload.values()) for w in sources)
    old = SimpleNamespace(prepared_weights={})
    combined = tuple(combine_a8_matrices(old, tuple(g[e] for g in groups))
                     for e in range(3))
    expected = _prepare_a8_routed(old, combined, torch.device("cuda"), padded_group)
    new = SimpleNamespace(prepared_weights={})
    actual = _prepare_a8_routed(
        new, groups[0], torch.device("cuda"), padded_group, projection_groups=groups,
    )
    torch.cuda.synchronize()
    for left, right in zip(actual, expected, strict=True):
        if left is None or right is None:
            assert left is right
        else:
            assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
    repeated = _prepare_a8_routed(
        new, groups[0], torch.device("cuda"), padded_group, projection_groups=groups,
    )
    assert all(a is b for a, b in zip(actual, repeated, strict=True))
    assert len(new.prepared_weights) == 4
    for key, (owner, value) in new.prepared_weights.items():
        assert key[0][0] == "int.marlin.a8.routed.grouped.v1"
        assert all(a is b for a, b in zip(owner.weights, sources, strict=True))
        assert owner.matches()
        assert value is None or value.device.type == "cuda"
    for weight, frozen in zip(sources, snapshots, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(weight.payload.values(), frozen, strict=True))
    # Cache validity remains tied to every original operand's complete ABI.
    source = sources[-1].payload["weight_packed"]
    source.transpose_(0, 1)
    assert not owner.matches()
    source.transpose_(0, 1)
    assert owner.matches()


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1" or not torch.cuda.is_available(),
    reason="explicit native CUDA audit required",
)
@pytest.mark.parametrize("bits", (3, 4, 5, 8))
@pytest.mark.parametrize("packing", ("dense", "legacy"))
@pytest.mark.parametrize("sym", (False, True))
def test_int_a8_preparation_chunks_preserve_expert_boundaries(monkeypatch, bits, packing, sym):
    from types import SimpleNamespace
    from cubic_serve.quantization.extensions.int.marlin import _prepare_a8_routed

    n, k, group = 160, 320, 96
    weights = tuple(fixture(bits, n=n, k=k, group=group, sym=sym, packing=packing)[0]
                    for _ in range(5))
    for expert, weight in enumerate(weights):
        weight.payload["weight_scale"].mul_(expert + 1)
    expected = tuple(_prepare_a8_routed(SimpleNamespace(prepared_weights={}), (weight,),
                                      torch.device("cuda"), group) for weight in weights)
    row_bytes = math.ceil(k / (32 // bits)) * 4 if packing == "legacy" else math.ceil(k * bits / 8)
    per_expert = n * row_bytes + 2 * (math.ceil(k / group) * group * n * bits // 8) + 32 * math.ceil(k / group) * n
    # Two experts per bounded transfer: five owners exercise both full and tail chunks.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (2 * per_expert * 32, 1 << 30))
    actual = _prepare_a8_routed(SimpleNamespace(prepared_weights={}), weights,
                                torch.device("cuda"), group)
    for field, value in enumerate(actual):
        reference = expected[0][field]
        if reference is None:
            assert value is None
        else:
            assert torch.equal(value.view(torch.uint8),
                               torch.cat(tuple(item[field] for item in expected)).view(torch.uint8))


def test_int_wna8_reference_preserves_group_rounding_at_bf16_boundary():
    """The original integer groups determine a near-midpoint BF16 result."""
    from cubic_serve.executor.runner import ReferenceExecutionContext
    from cubic_serve.ops.eager.quantization import linear_matrix

    n, k, group = 128, 320, 128
    generator = torch.Generator().manual_seed(123)
    codes = torch.randint(-128, 128, (n, k), generator=generator, dtype=torch.int8)
    scales = (torch.rand(n, 3, generator=generator) * 0.03 + 0.001).to(torch.float16)
    matrix = QuantizedMatrix(
        {
            "weight_packed": pack(codes, 8, "dense"),
            "weight_scale": scales,
            "weight_shape": torch.tensor([n, k]),
        },
        QuantizationScheme("int-pack-v1", 8, group, metadata="dense:sym:F16"),
        (n, k),
    )
    torch.manual_seed(48)
    inputs = (torch.randn(1, k, dtype=torch.bfloat16) * 0.08).contiguous()
    with ReferenceExecutionContext():
        tiled = linear_matrix(inputs, matrix, output_rows=64, decode_rows=64, compute_mode="wna8")
        whole = linear_matrix(inputs, matrix, output_rows=256, decode_rows=256, compute_mode="wna8")
    assert torch.equal(tiled, whole)
    assert tiled[0, 96].item() == 0.2451171875


@pytest.mark.parametrize('bits,group,sym,dtype', [
    (2, 80, True, torch.bfloat16),
    (5, 128, False, torch.float16),
    (6, 80, False, torch.bfloat16),
    (8, 128, True, torch.float16),
    (8, 2048, False, torch.bfloat16),
])
def test_int_reference_row_batch_preserves_all_rounding_boundaries(
    monkeypatch, bits, group, sym, dtype,
):
    from cubic_serve.executor.runner import ReferenceExecutionContext
    from cubic_serve.ops.eager.quantization import linear_matrix
    from cubic_serve.quantization.extensions.int import reference

    matrix, _ = fixture(bits, n=520, k=max(320, group), group=group, sym=sym)
    torch.manual_seed(391)
    inputs = torch.randn(3, matrix.shape[1], dtype=dtype)
    inputs[0].zero_()
    inputs[1, 0] = 64
    with ReferenceExecutionContext():
        with monkeypatch.context() as old:
            old.setattr(reference, 'wna8_output_row_batch_size', lambda *a, **kw: None)
            canonical = linear_matrix(inputs, matrix, compute_mode='wna8')
        batched = linear_matrix(inputs, matrix, compute_mode='wna8')
        bounded = linear_matrix(inputs, matrix, decode_rows=192, compute_mode='wna8')
    assert torch.equal(canonical.view(torch.uint8), batched.view(torch.uint8))
    assert torch.equal(canonical.view(torch.uint8), bounded.view(torch.uint8))


def test_int_reference_row_batch_accounts_for_group_dot_workspace():
    from types import SimpleNamespace
    from cubic_serve.quantization.extensions.int.reference import wna8_output_row_batch_size

    k, rows, group, budget = 65536, 1024, 64, 256 << 20
    matrix = SimpleNamespace(shape=(8, k), encoding=QuantizationScheme(
        'int-pack-v1', 8, group, metadata='dense:sym:F16'))
    count = wna8_output_row_batch_size(
        SimpleNamespace(numel=lambda: rows * k), matrix, byte_budget=budget)
    assert count * (64 * k + 16 * rows * (k // group) + 32 * rows) <= budget
    assert count < 256
    huge = 1 << 50
    matrix = SimpleNamespace(shape=(8, huge), encoding=QuantizationScheme(
        'int-pack-v1', 8, huge, metadata='dense:sym:F16'))
    assert wna8_output_row_batch_size(
        SimpleNamespace(numel=lambda: huge), matrix, byte_budget=budget) is None


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1" or not torch.cuda.is_available(),
    reason="explicit CUDA audit required",
)
def test_int_wna8_original_packed_codes_and_group_scales():
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.common.reference import wna8_activation
    from cubic_serve.quantization.extensions.int.reference import wna8_weight

    for bits, packing, symmetric, group, n, k in (
        (2, "dense", False, 40, 67, 259),
        (3, "legacy", False, 40, 67, 259),
        (4, "dense", False, 128, 64, 256),
        (5, "legacy", False, 80, 64, 256),
        (6, "dense", False, 128, 64, 256),
        (7, "legacy", True, 40, 67, 259),
        (8, "dense", True, 40, 64, 256),
        (8, "legacy", True, 80, 64, 256),
        (8, "dense", True, 128, 64, 256),
    ):
        weight, _ = fixture(bits, n=n, k=k, group=group,
                            sym=symmetric, packing=packing)
        x = (torch.randn(3, k, device="cuda", dtype=torch.bfloat16) * 0.08).contiguous()
        y = torch.empty(3, n, device="cuda", dtype=x.dtype)
        operators = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
        binding = operators.bind_linear(x, weight, y)
        assert any("int.marlin.linear.a8" in call.operator_id for call in binding._launches)
        binding.launch()
        expected = wna8_activation(x) @ wna8_weight(weight.to(x.device)).t()
        assert torch.allclose(y.float(), expected.float(), rtol=0.02, atol=0.002)


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1" or not torch.cuda.is_available(),
    reason="explicit compact affine integer-MMA proof",
)
@pytest.mark.parametrize("bits", range(2, 8))
@pytest.mark.parametrize("packing", ("dense", "legacy"))
def test_compact_affine_dense_matches_exact_integer_reference(bits, packing):
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.common.int8 import Int8Activation

    weight, _ = fixture(bits, n=64, k=256, group=128, sym=False, packing=packing)
    # Powers of two make every integer product, group scale and FP32 sum exact.
    # This witnesses zero subtraction and metadata ownership without a tolerance.
    weight.payload["weight_scale"].fill_(0.125)
    generator = torch.Generator().manual_seed(193 + bits)
    codes = torch.randint(-4, 5, (3, 256), dtype=torch.int8, generator=generator)
    scales = torch.full((3,), 0.03125)
    decoded = decode(weight.payload, weight.encoding, weight.shape)
    expected = ((codes.float() @ decoded.float().T) * scales[:, None]).bfloat16()
    activation = Int8Activation(codes.cuda(), scales.cuda(), torch.bfloat16)
    operators = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
    operators.register_int8_input(activation)
    output = torch.empty(3, 64, dtype=torch.bfloat16, device="cuda")
    binding = operators.bind_linear(activation.codes, weight, output)
    binding.launch()
    assert torch.equal(output.cpu().view(torch.uint8), expected.view(torch.uint8))
    prepared = [value for key, (_, value) in operators.prepared_weights.items()
                if isinstance(key, tuple) and len(key) == 2 and key[1] == "codes"]
    assert len(prepared) == 1 and prepared[0].numel() * 4 == 64 * 256 * bits // 8


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1" or not torch.cuda.is_available(),
    reason="explicit CUDA audit required",
)
@pytest.mark.parametrize("bits,symmetric,m", ((4, False, 1), (8, True, 4)))
def test_int_wna8_split_k_hc_down_fragment_layout(bits, symmetric, m):
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.common.reference import wna8_activation
    from cubic_serve.quantization.extensions.int.reference import wna8_weight

    n, k = 320, 10240
    weight, _ = fixture(bits, n=n, k=k, group=128, sym=symmetric)
    generator = torch.Generator(device="cuda").manual_seed(bits + m)
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=generator)
         * 0.08).contiguous()
    y = torch.empty(m, n, device="cuda", dtype=x.dtype)
    operators = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
    binding = operators.bind_linear(x, weight, y)
    assert any("int.marlin.linear.a8" in call.operator_id for call in binding._launches)
    assert any(call.operator_id == "cubic.cuda.marlin.a8_partial_reduce.v1"
               for call in binding._launches)
    binding.launch()
    expected = wna8_activation(x) @ wna8_weight(weight.to(x.device)).t()
    assert torch.allclose(y.float(), expected.float(), rtol=0.02, atol=0.02)


def test_int_wna8_format_admission_and_removed_mode():
    from cubic_serve.cli import build_parser
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.extensions.int.policy import supports_compute_mode

    wide = QuantizationScheme("int-pack-v1", 8, 128, metadata="dense:sym:F16")
    narrow_group = QuantizationScheme("int-pack-v1", 8, 64, metadata="dense:sym:F16")
    expert = QuantizationScheme("int-pack-v1", 4, 128, metadata="dense:asym:F16")
    assert supports_compute_mode(wide, "linear", "wna8")
    assert supports_compute_mode(narrow_group, "linear", "wna8")
    assert supports_compute_mode(expert, "moe", "wna8")
    assert not supports_compute_mode(wide, "linear", "wna8-flow")
    assert not supports_compute_mode(expert, "moe", "wna8-flow")
    with pytest.raises(ValueError, match="compute_mode must be"):
        OperatorConfig(compute_mode="wna8-flow")
    with pytest.raises(SystemExit) as rejected:
        build_parser().parse_args(["serve", "unused", "--compute-mode", "wna8-flow"])
    assert rejected.value.code == 2


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1" or not torch.cuda.is_available(),
    reason="explicit CUDA audit required",
)
@pytest.mark.parametrize("up_bits,intermediate", ((4, 64), (4, 80), (5, 64)))
@pytest.mark.parametrize("top_k", (1, 2))
def test_int_wna8_routed_original_expert_order(up_bits, intermediate, top_k):
    from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
    from cubic_serve.ops.cuda.moe.quantized import PackedRoutedExpertsPrimitive
    from cubic_serve.ops.eager.activation import silu_stored
    from cubic_serve.ops.eager.quantization import linear_matrix
    from cubic_serve.ops.strategies import OperatorConfig

    configure_reference_math()
    hidden = 128
    gate = tuple(fixture(4, n=intermediate, k=hidden, group=64)[0] for _ in range(2))
    up = tuple(fixture(up_bits, n=intermediate, k=hidden, group=64)[0] for _ in range(2))
    down = tuple(fixture(4, n=hidden, k=intermediate, group=64)[0] for _ in range(2))
    x = (torch.randn(2, hidden, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    logits = torch.tensor([[1., 0.], [0., 1.]], device="cuda", dtype=torch.bfloat16)
    activity = torch.ones(2, device="cuda", dtype=torch.int32)
    active_rows = torch.tensor([2], device="cuda", dtype=torch.int32)
    output = torch.empty_like(x)
    op = PackedRoutedExpertsPrimitive(
        x, logits, activity, active_rows, gate, up, down, output,
        first_expert=0, top_k=top_k,
        operators=PackedLinearOperators(OperatorConfig(compute_mode="wna8")),
    )
    native = [call.operator_id for _, _, binding in op.projections if binding is not None
              for call in binding._launches]
    assert sum("int.marlin.routed.a8" in key for key in native) == (2 if up_bits == 4 else 3)
    assert not any("_routed_mm" in key for key in native)
    op.launch()
    assert torch.isfinite(output).all()
    assert op.route_ids.tolist() == ([[0], [1]] if top_k == 1 else [[0, 1], [0, 1]])
    with ReferenceExecutionContext():
        for row in range(x.shape[0]):
            for slot, expert in enumerate(op.route_ids[row].tolist()):
                gate_value = linear_matrix(x[row:row + 1], gate[expert], compute_mode="wna8")
                up_value = linear_matrix(x[row:row + 1], up[expert], compute_mode="wna8")
                activated = (silu_stored(gate_value) * up_value).to(x.dtype)
                expected = linear_matrix(activated, down[expert], compute_mode="wna8")
                actual = op.routes[row:row + 1, slot]
                assert torch.allclose(actual.float(), expected.float(),
                                      rtol=0.03, atol=0.003)
                if top_k == 1:
                    assert torch.allclose(output[row:row + 1].float(), expected.float(),
                                          rtol=0.03, atol=0.003)


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1" or not torch.cuda.is_available(),
    reason="explicit CUDA audit required",
)
@pytest.mark.parametrize("bits,member_n", ((4, 64), (4, 80), (4, 160), (3, 64)))
def test_int_wna8_grouped_checkpoint_reuses_one_gpu_owner_across_graph_buckets(bits, member_n):
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.api import RoutedRows
    from cubic_serve.quantization.extensions.int.moe import bind_grouped
    from cubic_serve.quantization.extensions.int.policy import IntegerMoEParameters

    gate = tuple(fixture(bits, n=member_n, k=128, group=128, packing="legacy")[0]
                 for _ in range(2))
    up = tuple(fixture(bits, n=member_n, k=128, group=128, packing="legacy")[0]
               for _ in range(2))
    prepared, materialized = {}, {}
    pointers = []
    for capacity in (2, 4):
        x = torch.zeros((capacity, 128), device="cuda", dtype=torch.bfloat16)
        y = torch.empty((capacity, 2 * member_n), device="cuda", dtype=x.dtype)
        counts = torch.tensor((1, 1), device=x.device, dtype=torch.int32)
        starts = torch.tensor((0, 1), device=x.device, dtype=torch.int32)
        blocks = torch.tensor((0, 1), device=x.device, dtype=torch.int32)
        total = torch.tensor((2,), device=x.device, dtype=torch.int32)
        row_map = torch.tensor((0, 1), device=x.device, dtype=torch.int32)
        operators = PackedLinearOperators(
            OperatorConfig(compute_mode="wna8"),
            prepared_weights=prepared, materialized_weights=materialized,
        )
        _, binding = bind_grouped(
            operators, x, gate, up, y,
            RoutedRows(counts, starts, blocks, total, row_map),
            1, IntegerMoEParameters(),
        )
        native = next(call for call in binding._launches
                      if "int.marlin.routed.a8" in call.operator_id)
        pointers.append(native.arguments[1].data_ptr())
    assert pointers[0] == pointers[1]
    if bits == 4 and member_n == 64:
        assert not any(key[0] == "int.a8.grouped.checkpoint.v2" for key in prepared)


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1" or not torch.cuda.is_available(),
    reason="explicit CUDA audit required",
)
def test_int_wna8_dense_grouped_keeps_independent_output_rounding():
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.common.reference import wna8_activation
    from cubic_serve.quantization.extensions.int.reference import wna8_weight

    weights = tuple(fixture(4, n=n, k=256, group=128, packing="legacy")[0]
                    for n in (67, 80, 64))
    prepared, materialized = {}, {}
    pointers = []
    for rows in (1, 4):
        x = (torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16) * 0.08).contiguous()
        outputs = tuple(torch.empty(rows, w.shape[0], device=x.device, dtype=x.dtype)
                        for w in weights)
        operators = PackedLinearOperators(
            OperatorConfig(compute_mode="wna8"),
            prepared_weights=prepared, materialized_weights=materialized,
        )
        binding = operators.bind_linear_group(x, weights, outputs)
        assert binding is not None
        native = [call for call in binding._launches
                  if "int.marlin.linear.a8" in call.operator_id]
        assert len(native) == 1
        pointers.append(native[0].arguments[1].data_ptr())
        binding.launch()
        for weight, output in zip(weights, outputs, strict=True):
            expected = wna8_activation(x) @ wna8_weight(weight.to(x.device)).t()
            assert torch.allclose(output.float(), expected.float(), rtol=0.02, atol=0.002)
    assert pointers[0] == pointers[1]


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1" or not torch.cuda.is_available(),
    reason="explicit CUDA audit required",
)
@pytest.mark.parametrize("widths", ((160, 160), (160, 128, 96, 80)))
def test_int_wna8_partitioned_group_merge_writes_outputs_directly(widths):
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.extensions.int.marlin import (
        bind_a8_dense_symmetric,
        combine_a8_matrices,
    )

    weights = tuple(
        fixture(8, n=n, k=2560, group=128, packing="legacy", sym=True)[0]
        for n in widths
    )
    operators = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
    combined = combine_a8_matrices(operators, weights)
    for rows in (1, 128):
        x = (torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16) * 0.08).contiguous()
        outputs = tuple(torch.full((rows, n), float("nan"), device=x.device, dtype=x.dtype)
                        for n in widths)
        direct = operators.bind_linear_group(x, weights, outputs)
        assert direct is not None
        names = [launch.operator_id for launch in direct._launches]
        assert any("_merge_dense_partials" in name for name in names)
        assert not any("_split_a8_group" in name for name in names)
        assert not any("_unpad_output" in name for name in names)
        temporary = torch.empty((rows, sum(widths)), device=x.device, dtype=x.dtype)
        original = bind_a8_dense_symmetric(
            operators, x, combined, temporary, operators._parameters(weights[0])
        )
        original.launch()
        direct.launch()
        offset = 0
        for width, output in zip(widths, outputs, strict=True):
            assert torch.equal(output, temporary[:, offset:offset + width])
            offset += width


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1" or not torch.cuda.is_available(),
    reason="explicit CUDA audit required",
)
def test_int_wna8_routed_many_groups_and_topk_zero_correction():
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.api import RoutedRows
    from cubic_serve.quantization.common.reference import wna8_activation
    from cubic_serve.quantization.extensions.int.marlin import bind_a8_routed
    from cubic_serve.quantization.extensions.int.policy import IntegerMoEParameters
    from cubic_serve.quantization.extensions.int.reference import wna8_weight

    weights = tuple(fixture(4, n=320, k=2560, group=128,
                            packing="legacy")[0] for _ in range(2))
    x = (torch.randn((1, 2560), device="cuda", dtype=torch.bfloat16) * 0.08).contiguous()
    y = torch.empty((2, 320), device=x.device, dtype=x.dtype)
    counts = torch.tensor((1, 1), device=x.device, dtype=torch.int32)
    starts = torch.tensor((0, 1), device=x.device, dtype=torch.int32)
    blocks = torch.tensor((0, 1), device=x.device, dtype=torch.int32)
    total = torch.tensor((2,), device=x.device, dtype=torch.int32)
    row_map = torch.tensor((0, 1), device=x.device, dtype=torch.int32)
    rows = RoutedRows(counts, starts, blocks, total, row_map, routing_top_k=2)
    operators = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
    _, binding = bind_a8_routed(operators, x, weights, y, rows, IntegerMoEParameters(), 2)
    binding.launch()
    expected = torch.cat(tuple(
        wna8_activation(x) @ wna8_weight(weight.to(x.device)).t()
        for weight in weights
    ))
    assert torch.allclose(y.float(), expected.float(), rtol=0.02, atol=0.002)


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1" or not torch.cuda.is_available(),
    reason="explicit CUDA audit required",
)
@pytest.mark.parametrize("group", (48, 80, 112))
def test_int_wna8_routed_w5_asymmetric_uneven_groups(group):
    from cubic_serve.guardrail.quantization import check_linear_enclosure
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.api import RoutedRows
    from cubic_serve.quantization.extensions.int.marlin import bind_a8_routed
    from cubic_serve.quantization.extensions.int.policy import IntegerMoEParameters

    first, _ = fixture(5, n=2560, k=2 * group, group=group, packing="legacy")
    second = QuantizedMatrix(
        {key: value.roll(1, 0) if key in ("weight_packed", "weight_scale",
                                         "weight_zero_point") else value
         for key, value in first.payload.items()},
        first.encoding, first.shape,
    )
    weights = (first, second)
    generator = torch.Generator(device="cuda").manual_seed(280)
    # Routed down receives the already expanded expert rows: 3 tokens x top-k 10.
    x = (torch.randn(30, 2 * group, device="cuda", dtype=torch.bfloat16,
                     generator=generator) * 0.08).contiguous()
    y = torch.full((30, 2560), float("nan"), device=x.device, dtype=x.dtype)
    rows = RoutedRows(
        torch.tensor((15, 15), device=x.device, dtype=torch.int32),
        torch.tensor((0, 15), device=x.device, dtype=torch.int32),
        torch.tensor((0, 1), device=x.device, dtype=torch.int32),
        torch.tensor((2,), device=x.device, dtype=torch.int32),
        torch.tensor((*range(0, 30, 2), *range(1, 30, 2)),
                     device=x.device, dtype=torch.int32),
        routing_top_k=10,
    )
    operators = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
    _, binding = bind_a8_routed(
        operators, x, weights, y, rows, IntegerMoEParameters(), 1
    )
    ids = [call.operator_id for call in binding._launches]
    assert sum("int.marlin.routed.a8" in name for name in ids) == 1
    assert not any("_a8_correct_zero_routed" in name for name in ids)
    binding.launch()
    assert torch.isfinite(y).all()
    for expert, weight in enumerate(weights):
        check_linear_enclosure(x[expert::2], weight.to(x.device), y[expert::2],
                               compute_mode="wna8")


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1" or not torch.cuda.is_available(),
    reason="explicit CUDA audit required",
)
def test_int_wna8_routed_w5_zero_survives_graph_stage_reuse():
    from cubic_serve.guardrail.quantization import check_linear_enclosure
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.api import RoutedRows
    from cubic_serve.quantization.extensions.int.marlin import bind_a8_routed
    from cubic_serve.quantization.extensions.int.policy import IntegerMoEParameters

    count = 8
    first, _ = fixture(5, n=2560, k=160, group=80, packing="legacy")
    weights = tuple(
        QuantizedMatrix(
            {key: value.roll(expert, 0)
             if key in ("weight_packed", "weight_scale", "weight_zero_point") else value
             for key, value in first.payload.items()},
            first.encoding, first.shape,
        ) for expert in range(count)
    )
    generator = torch.Generator(device="cuda").manual_seed(281)
    x = (torch.randn(count, 160, device="cuda", dtype=torch.bfloat16,
                     generator=generator) * 0.08).contiguous()
    y = torch.empty(count, 2560, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(count, device="cuda", dtype=torch.int32)
    rows = RoutedRows(
        torch.ones(count, device="cuda", dtype=torch.int32), indices, indices,
        torch.tensor((count,), device="cuda", dtype=torch.int32), indices,
        routing_top_k=1,
    )
    _, binding = bind_a8_routed(
        PackedLinearOperators(OperatorConfig(compute_mode="wna8")),
        x, weights, y, rows, IntegerMoEParameters(), 1,
    )
    binding.launch()
    eager = y.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.launch()
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, eager)
    for expert, weight in enumerate(weights):
        check_linear_enclosure(x[expert:expert + 1], weight.to(x.device),
                               y[expert:expert + 1], compute_mode="wna8")




class IntegerReferenceTests(unittest.TestCase):
    def test_legacy_word_rebase_all_offsets_and_tail_bits(self):
        from unittest.mock import patch

        from cubic_serve.quantization.extensions.int import layout

        for bits in range(2, 9):
            per_word = 32 // bits
            matrix, expected = fixture(bits, n=7, k=103, group=13,
                                       sym=True, packing="legacy")
            # Unused high bits are not codes and must never leak into a join.
            unused = (0xFFFFFFFF ^ ((1 << (per_word * bits)) - 1))
            if unused:
                matrix.payload["weight_packed"].bitwise_or_(unused - 2**32)
            for start in range(2 * per_word):
                for length in (1, per_word - 1, per_word, per_word + 1, 103 - start):
                    with (
                        patch.object(layout, "pack", side_effect=AssertionError("repack")),
                        patch.object(layout, "unpack", side_effect=AssertionError("unpack")),
                    ):
                        owned = matrix.columns(start, start + length)
                    self.assertTrue(torch.equal(
                        decode(owned.payload, owned.encoding, owned.shape),
                        expected[:, start:start + length],
                    ))

    def test_legacy_word_aligned_owners_preserve_serialized_words(self):
        from contextlib import nullcontext
        from unittest.mock import patch

        from cubic_serve.quantization.extensions.int import layout

        for bits in range(2, 9):
            per_word = 32 // bits
            width = math.lcm(32, per_word)
            for sym in (False, True):
                matrix, expected = fixture(bits, n=67, k=2 * width, group=40,
                                           sym=sym, packing="legacy")
                for start in (0, width):
                    # Symmetric word-aligned ownership must not decode/repack
                    # even once. Asymmetric zero points still need their own
                    # N-packed boundary treatment.
                    with (
                        patch.object(layout, "pack", side_effect=AssertionError("repack"))
                        if sym else nullcontext(),
                        patch.object(layout, "unpack", side_effect=AssertionError("unpack"))
                        if sym else nullcontext(),
                    ):
                        owned = matrix.columns(start, start + width)
                    self.assertTrue(owned.encoding.metadata.startswith("legacy:"))
                    self.assertTrue(torch.equal(
                        owned.payload["weight_packed"],
                        matrix.payload["weight_packed"][
                            :, start // per_word:(start + width) // per_word
                        ],
                    ))
                    self.assertTrue(torch.equal(
                        decode(owned.payload, owned.encoding, owned.shape),
                        expected[:, start:start + width],
                    ))

    def test_all_codes_and_owner_boundaries(self):
        for bits in range(2, 9):
            for packing in ("dense", "legacy"):
                for sym in (False, True):
                    for dt in (torch.float16, torch.bfloat16, torch.float32):
                        w, expected = fixture(
                            bits, n=67, k=259, group=40, sym=sym, dtype=dt, packing=packing
                        )
                        self.assertTrue(
                            torch.equal(decode(w.payload, w.encoding, w.shape), expected)
                        )
                        whole = w.columns(0, w.shape[1])
                        self.assertEqual(whole.encoding, w.encoding)
                        self.assertTrue(
                            torch.equal(whole.payload["weight_packed"], w.payload["weight_packed"])
                        )
                        self.assertTrue(
                            torch.equal(whole.payload["weight_scale"], w.payload["weight_scale"])
                        )
                        for part, target in (
                            (whole, expected),
                            (w.rows(3, 64), expected[3:64]),
                            (w.columns(17, 244), expected[:, 17:244]),
                            (w.columns(32, 160), expected[:, 32:160]),
                        ):
                            self.assertTrue(
                                torch.equal(decode(part.payload, part.encoding, part.shape), target)
                            )

    def test_independent_word_codes(self):
        for bits in range(2, 9):
            q = torch.arange(-(1 << (bits - 1)), 1 << (bits - 1), dtype=torch.int16).to(torch.int8)[
                None
            ]
            for packing in ("dense", "legacy"):
                words = pack(q, bits, packing)
                for column, v in enumerate(q[0].tolist()):
                    bit = (
                        column * bits
                        if packing == "dense"
                        else column // (32 // bits) * 32 + column % (32 // bits) * bits
                    )
                    raw = int(words[0, bit // 32]) & 0xFFFFFFFF
                    if bit % 32 + bits > 32:
                        raw |= (int(words[0, bit // 32 + 1]) & 0xFFFFFFFF) << 32
                    self.assertEqual(
                        ((raw >> (bit % 32)) & ((1 << bits) - 1)) - (1 << (bits - 1)), v
                    )


@unittest.skipUnless(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") == "1" and torch.cuda.is_available(),
    "explicit CUDA audit required",
)
class IntegerLinearTests(unittest.TestCase):
    def test_k16_pipeline_and_logical_column_tail(self):
        from unittest.mock import patch

        from cubic_serve.guardrail.quantization import check_linear_enclosure

        # K80 exercises alternating K8 scale halves. N81 makes the physical
        # N256 tile write a logical tail directly, without an unpadding launch.
        for bits, packing, partitions in product(range(2, 9), ("dense", "legacy"), (1, 5)):
            weight, _ = fixture(bits, n=81, k=80, group=40, packing=packing)
            for dtype in (torch.float16, torch.bfloat16):
                operators = PackedLinearOperators()
                corpus = torch.randn(3, 80, device="cuda", dtype=dtype) * 0.05
                anchor = None
                for capacity in (1, 9):
                    count = min(3, capacity)
                    rows = torch.arange(capacity - 1, capacity - count - 1, -1,
                                        device="cuda")
                    x = torch.full((capacity, 80), float("nan"), device="cuda", dtype=dtype)
                    x[rows] = corpus[:count]
                    y = torch.full((capacity, 81), float("nan"), device="cuda", dtype=dtype)
                    with (
                        patch("cubic_serve.quantization.extensions.int.marlin.plan_dense_partitions",
                              return_value=partitions),
                        patch("cubic_serve.quantization.extensions.int.marlin.select",
                              side_effect=lambda p, **kw: p),
                    ):
                        binding = operators.bind_linear(x, weight, y)
                    names = [call.operator_id for call in binding._launches]
                    self.assertEqual(len(names), 2)
                    self.assertIn("int.marlin.linear", names[0])
                    self.assertIn("_merge_dense_partials", names[1])
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        binding.launch()
                    graph.replay()
                    logical = y[rows].clone()
                    check_linear_enclosure(corpus[:count], weight, logical)
                    if anchor is not None:
                        self.assertTrue(torch.equal(anchor, logical[:1]))
                    anchor = logical[:1].clone()
                    x[rows] = corpus[:count].flip(0)
                    graph.replay()
                    self.assertTrue(torch.equal(y[rows], logical.flip(0)))
                    graph.reset()

    def test_prepared_cache_reuses_operands_and_rejects_changed_source_abi(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from cubic_serve.quantization.extensions.int import marlin

        device = torch.device("cuda", 0)
        weight, _ = fixture(4)
        operators = SimpleNamespace(prepared_weights={})
        first = marlin._prepare(operators, ((weight,),), device)
        # A hit retains full source ABI validation without weak-cache lookups.
        with patch.object(marlin, "bind_tensor", side_effect=AssertionError("rebind")):
            hit = marlin._prepare(operators, ((weight,),), device)
        self.assertTrue(all(a is b for a, b in zip(first, hit, strict=True)))
        scale = weight.payload["weight_scale"]
        # .data reassignment need not advance Tensor._version.
        scale.data = scale.clone().mul_(2)
        changed = marlin._prepare(operators, ((weight,),), device)
        fresh = marlin._prepare(SimpleNamespace(prepared_weights={}), ((weight,),), device)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(changed, fresh, strict=True)))
        self.assertIsNot(changed[1], first[1])
        self.assertFalse(torch.equal(changed[1], first[1]))
        other, _ = fixture(4)
        separate = marlin._prepare(operators, ((other,),), device)
        self.assertIsNot(separate[0], changed[0])
        self.assertTrue(torch.equal(separate[1], first[1]))
        # The same ordered sources can mean two experts or two projections.
        experts = marlin._prepare(operators, ((weight, other),), device)
        projections = marlin._prepare(operators, ((weight,), (other,)), device)
        self.assertEqual(experts[0].shape[0], 2)
        self.assertEqual(projections[0].shape[0], 1)
        fresh = marlin._prepare(
            SimpleNamespace(prepared_weights={}), ((weight,), (other,)), device
        )
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(projections, fresh, strict=True)))
        # The cache registry still exposes only device tensors as its second field.
        self.assertTrue(all(
            isinstance(t, torch.Tensor) for _, t in operators.prepared_weights.values()
        ))

    def test_batched_expert_preparation_preserves_individual_marlin_operands(self):
        from types import SimpleNamespace

        from cubic_serve.quantization.extensions.int.marlin import METADATA_HEADER_WORDS, _prepare

        device = torch.device("cuda", 0)
        for bits in range(2, 9):
            for packing in ("dense", "legacy"):
                for sym in (False, True):
                    groups = []
                    for rows in (64, 128):
                        group = []
                        for expert in range(3):
                            w, _ = fixture(bits, n=rows, k=256, group=40,
                                           sym=sym, packing=packing)
                            # Distinct experts expose transposed/reordered ownership.
                            w.payload["weight_packed"].copy_(
                                w.payload["weight_packed"].roll(expert + 1, 0)
                            )
                            w.payload["weight_scale"].mul_(expert + 1)
                            group.append(w)
                        groups.append(tuple(group))
                    codes, metadata = _prepare(
                        SimpleNamespace(prepared_weights={}), tuple(groups), device,
                    )
                    expected_codes, expected_metadata = [], []
                    for expert in range(3):
                        c, m = _prepare(SimpleNamespace(prepared_weights={}),
                                        tuple((g[expert],) for g in groups), device)
                        expected_codes.append(c)
                        self.assertTrue(torch.equal(
                            m[:METADATA_HEADER_WORDS].view(torch.int32),
                            metadata[:METADATA_HEADER_WORDS].view(torch.int32),
                        ))
                        expected_metadata.append(m[METADATA_HEADER_WORDS:])
                    self.assertTrue(torch.equal(codes, torch.cat(expected_codes)))
                    self.assertTrue(torch.equal(metadata[METADATA_HEADER_WORDS:],
                                                torch.cat(expected_metadata)))

    def test_legacy_owner_and_dense_repack_prepare_identical_marlin_operands(self):
        from dataclasses import replace
        from types import SimpleNamespace

        from cubic_serve.quantization.extensions.int.marlin import _prepare
        from cubic_serve.quantization.extensions.int.reference import unpack

        device = torch.device("cuda", 0)
        for bits in range(2, 9):
            width = math.lcm(32, 32 // bits)
            for sym in (False, True):
                matrix, _ = fixture(bits, n=64, k=2 * width + 32, group=40,
                                    sym=sym, packing="legacy")
                for start in (0, 1, width):
                    owned = matrix.columns(start, start + width)
                    payload = dict(owned.payload)
                    payload["weight_packed"] = pack(
                        unpack(payload["weight_packed"], bits, width, packing="legacy"), bits,
                    )
                    if not sym:
                        payload["weight_zero_point"] = pack(
                            unpack(payload["weight_zero_point"], bits, 64, 0, "legacy"),
                            bits, axis=0,
                        )
                    dense = QuantizedMatrix(
                        payload, replace(owned.encoding, metadata=owned.encoding.metadata.replace(
                            "legacy:", "dense:", 1,
                        )), owned.shape,
                    )
                    old = _prepare(SimpleNamespace(prepared_weights={}), ((dense,),), device)
                    new = _prepare(SimpleNamespace(prepared_weights={}), ((owned,),), device)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(old, new, strict=True)))

    def test_grouped_dense_respects_member_overrides(self):
        from unittest.mock import patch

        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
        from cubic_serve.quantization.extensions.int import marlin
        from cubic_serve.quantization.extensions.int.linear import bind_grouped
        from cubic_serve.quantization.extensions.int.policy import IntegerLinearParameters

        weights = tuple(fixture(8, n=n)[0] for n in (160, 96))
        x = torch.empty(1, 256, device="cuda", dtype=torch.bfloat16)
        outputs = tuple(torch.empty(1, w.shape[0], device=x.device, dtype=x.dtype)
                        for w in weights)
        with (
            patch("cubic_serve.ops.strategies.parameter_overrides",
                  return_value={"marlin_block_n": 64}),
            patch.object(marlin, "_bind",
                  side_effect=AssertionError("member override must prevent merged planning")),
        ):
            self.assertIsNone(bind_grouped(
                PackedLinearOperators(), x, weights, outputs, IntegerLinearParameters()
            ))

    def test_mixed_dense_group_preserves_formats_outputs_and_cross_m(self):
        from unittest.mock import patch

        from cubic_serve.executor.text_graph import TextGraphBindings
        from cubic_serve.guardrail.quantization import check_linear_enclosure
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators

        cases = [
            (bits, packing, dtype, 256, (160, 96))
            for bits in range(2, 9)
            for packing in ("dense", "legacy")
            for dtype in (torch.float16, torch.bfloat16)
        ] + [(8, "dense", torch.bfloat16, 2560, (160, 160))]
        for bits, packing, dtype, k, ns in cases:
            torch.manual_seed(39)
            gate = torch.randn(512, k, device="cuda", dtype=dtype) * 0.05
            shared_gate = torch.randn(1, k, device="cuda", dtype=dtype) * 0.05
            left, _ = fixture(bits, ns[0], k, 128, packing=packing)
            right, _ = fixture(bits, ns[1], k, 128, packing=packing)
            right.payload["weight_scale"].mul_(1.5)
            # Interleave encodings: subgroups must retain the output identity.
            weights = (left, gate, right, shared_gate)
            corpus = torch.randn(3, k, device="cuda", dtype=dtype) * 0.05
            ops = PackedLinearOperators()
            common = None
            for capacity in (1, 4, 17):
                count = min(3, capacity)
                physical = torch.arange(capacity - 1, capacity - 1 - count, -1, device="cuda")
                x = torch.full((capacity, k), float("nan"), device="cuda", dtype=dtype)
                x[physical] = corpus[:count]
                outputs = tuple(torch.empty(capacity, w.shape[0], device="cuda", dtype=dtype)
                                for w in weights)
                bindings = []
                bind_groups = ops.bind_linear_groups

                def bind_declared(*args, _bind_groups=bind_groups, _bindings=bindings):
                    result = _bind_groups(*args)
                    _bindings.extend(result)
                    return result

                helper = TextGraphBindings(ops, x.device, dtype)
                with (
                    patch("cubic_serve.quantization.extensions.int.marlin.plan_dense_partitions",
                          return_value=2),
                    patch("cubic_serve.quantization.extensions.int.marlin.select",
                          side_effect=lambda p, **kw: p),
                    patch.object(ops, "bind_linear_groups", side_effect=bind_declared),
                ):
                    launch = helper.linear_group(x, weights, outputs)
                self.assertEqual(len(bindings), 2)
                packed = bindings[0]
                names = [call.operator_id for call in packed._launches]
                self.assertEqual(sum("int.marlin.linear" in name for name in names), 1)
                self.assertEqual(sum("_merge_dense_partials" in name for name in names), 1)
                self.assertEqual(len(names), 2)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch()
                graph.replay()
                logical = tuple(y[physical] for y in outputs)
                for w, y in zip(weights, logical, strict=True):
                    check_linear_enclosure(corpus[:count], w, y)
                if common is not None:
                    for before, after in zip(common, logical, strict=True):
                        self.assertTrue(torch.equal(before, after[:1]))
                common = tuple(y[:1].clone() for y in logical)
                graph.reset()
            # Whole-declaration validation must still reject cross-format aliases.
            with self.assertRaises(ValueError):
                ops.bind_linear_groups(x, (left, gate), (outputs[0], outputs[0]))

    def test_m8_partial_layout_and_row_identity(self):
        from unittest.mock import patch

        from cubic_serve.guardrail.quantization import check_linear_enclosure
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
        from cubic_serve.quantization.extensions.int.marlin import _bind
        from cubic_serve.quantization.extensions.int.policy import IntegerLinearParameters

        # Fix the reduction plan to exercise accumulator layout independently
        # of cost calibration, including K8 metadata boundaries and >1 M tile.
        for bits in range(2, 9):
            for dtype in (torch.float16, torch.bfloat16):
                for n, k, group, bk, bn, partitions in (
                    (64, 256, 128, 128, 64, 2),
                    (256, 320, 40, 32, 256, 5),
                ):
                    weight, _ = fixture(bits, n=n, k=k, group=group)
                    operators = PackedLinearOperators()
                    corpus = torch.randn(3, k, device="cuda", dtype=dtype) * 0.05
                    common = None
                    for capacity in (1, 8, 9, 17):
                        count = min(3, capacity)
                        physical = torch.arange(
                            capacity - 1, capacity - 1 - count, -1, device="cuda"
                        )
                        x = torch.full((capacity, k), float("nan"), device="cuda", dtype=dtype)
                        x[physical] = corpus[:count]
                        y = torch.empty(capacity, n, device="cuda", dtype=dtype)
                        params = IntegerLinearParameters(
                            marlin_block_m=8, marlin_block_k=bk, marlin_block_n=bn,
                            marlin_auto_geometry=False,
                        )
                        with patch(
                            "cubic_serve.quantization.extensions.int.marlin.plan_dense_partitions",
                            return_value=partitions,
                        ):
                            binding = _bind(operators, x, ((weight,),), y, params)
                        self.assertIsNotNone(binding)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            binding.launch()
                        graph.replay()
                        logical = y[physical].clone()
                        check_linear_enclosure(corpus[:count], weight, logical)
                        if common is not None:
                            self.assertTrue(torch.equal(common, logical[:1]))
                        common = logical[:1].clone()
                        x[physical] = corpus[:count].flip(0)
                        graph.replay()
                        self.assertTrue(torch.equal(y[physical], logical.flip(0)))
                        graph.reset()

    def test_k8_scale_boundaries_and_alignment_tiles(self):
        from cubic_serve.guardrail.quantization import check_linear_enclosure
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators

        for bits, group, k, n in (
            (2, 8, 256, 65),
            (3, 24, 384, 64),
            (4, 40, 320, 160),
            (5, 72, 1152, 128),
            (6, 8, 256, 64),
            (7, 40, 320, 64),
            (8, 40, 320, 10240),
            (2, 80, 160, 65),
            (3, 24, 96, 256),
            (4, 40, 160, 160),
            (5, 56, 224, 256),
            (6, 80, 160, 65),
            (7, 40, 160, 128),
            (8, 80, 160, 2560),
        ):
            weight, expected = fixture(bits, n=n, k=k, group=group, sym=False)
            for dtype in (torch.float16, torch.bfloat16):
                operators = PackedLinearOperators()
                corpus = torch.randn(3, k, device="cuda", dtype=dtype) * 0.05
                first = None
                for m in (1, 4, 17):
                    count = min(m, 3)
                    physical = torch.arange(m - 1, m - 1 - count, -1, device="cuda")
                    x = torch.full((m, k), float("nan"), device="cuda", dtype=dtype)
                    x[physical] = corpus[:count]
                    y = torch.empty(m, n, device="cuda", dtype=dtype)
                    binding = operators.bind_linear(x, weight, y)
                    self.assertTrue(any("int.marlin" in c.operator_id for c in binding._launches))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        binding.launch()
                    graph.replay()
                    logical = y[physical]
                    check_linear_enclosure(corpus[:count], weight, logical)
                    if first is not None:
                        self.assertTrue(torch.equal(first, logical[:1]))
                    first = logical[:1].clone()
                    graph.reset()
                indices = torch.tensor(
                    [7, 8, group - 1, group, group + 7, 31, 32, 63, 64, k - 1], device="cuda"
                )
                x = torch.zeros(indices.numel(), k, device="cuda", dtype=dtype)
                x[torch.arange(indices.numel(), device="cuda"), indices] = 1
                y = torch.empty(indices.numel(), n, device="cuda", dtype=dtype)
                operators.bind_linear(x, weight, y).launch()
                self.assertTrue(torch.equal(y, expected[:, indices.cpu()].T.to("cuda", dtype)))

    def test_partitioned_long_k_cross_m(self):
        from cubic_serve.guardrail.quantization import check_linear_enclosure
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
        from cubic_serve.quantization.extensions.int.marlin import bind_linear as bind_marlin
        from cubic_serve.quantization.extensions.int.policy import IntegerLinearParameters

        # Exercise both automatic dispatch and the explicitly named partition
        # kernel. A valid small-row choice is not a numerical test failure.
        for (bits, k, group), native in product(
            ((8, 10240, 128), (5, 2560, 80)), (False, True)
        ):
            weight, _ = fixture(bits, n=320, k=k, group=group, sym=False)
            operators = PackedLinearOperators()
            corpus = torch.randn(3, k, device="cuda", dtype=torch.bfloat16) * 0.05
            first = None
            for m in (1, 4, 17):
                count = min(m, 3)
                physical = torch.arange(m - 1, m - 1 - count, -1, device="cuda")
                x = torch.full((m, k), float("nan"), device="cuda", dtype=corpus.dtype)
                x[physical] = corpus[:count]
                y = torch.empty(m, 320, device="cuda", dtype=corpus.dtype)
                binding = (bind_marlin(operators, x, weight, y, IntegerLinearParameters())
                           if native else operators.bind_linear(x, weight, y))
                self.assertIsNotNone(binding)
                if native:
                    self.assertTrue(any(".partial." in c.operator_id for c in binding._launches))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    binding.launch()
                graph.replay()
                logical = y[physical]
                check_linear_enclosure(corpus[:count], weight, logical)
                if first is not None:
                    self.assertTrue(torch.equal(first, logical[:1]))
                first = logical[:1].clone()
                graph.reset()

    def test_native_all_bits_and_rounding(self):
        from cubic_serve.guardrail.quantization import check_linear_enclosure
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators

        for bits in range(2, 9):
            for dtype in (torch.float16, torch.bfloat16):
                for sym in (False, True):
                    w, _ = fixture(bits, sym=sym)
                    x = torch.randn(3, 256, device="cuda", dtype=dtype)
                    y = torch.empty(3, 64, device="cuda", dtype=dtype)
                    binding = PackedLinearOperators().bind_linear(x, w, y)
                    self.assertTrue(
                        any("int.marlin.linear" in call.operator_id for call in binding._launches)
                    )
                    binding.launch()
                    check_linear_enclosure(x, w, y)

    def test_serialized_scale_precision_at_k8_boundaries(self):
        from unittest.mock import patch

        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
        from cubic_serve.quantization.extensions.int.marlin import bind_linear
        from cubic_serve.quantization.extensions.int.policy import IntegerLinearParameters

        # One-hot inputs expose the exact stored weight boundary without a
        # reduction allowance hiding premature scale or operand rounding.
        indices = torch.tensor([39, 40, 79, 80, 319], device="cuda")
        for bits in (2, 5, 8):
            for scale_dtype in (torch.float16, torch.bfloat16, torch.float32):
                w, expected = fixture(bits, n=256, k=320, group=40, dtype=scale_dtype)
                for dtype in (torch.float16, torch.bfloat16):
                    x = torch.zeros(5, 320, device="cuda", dtype=dtype)
                    x[torch.arange(5, device="cuda"), indices] = 1
                    y = torch.empty(5, 256, device="cuda", dtype=dtype)
                    params = IntegerLinearParameters(
                        marlin_block_m=8, marlin_block_k=32, marlin_block_n=256,
                        marlin_auto_geometry=False,
                    )
                    with patch(
                        "cubic_serve.quantization.extensions.int.marlin.plan_dense_partitions",
                        return_value=5,
                    ):
                        binding = bind_linear(PackedLinearOperators(), x, w, y, params)
                    self.assertIsNotNone(binding)
                    binding.launch()
                    reference = expected[:, indices.cpu()].T.to(device="cuda", dtype=dtype)
                    self.assertTrue(torch.equal(y, reference))

    def test_linear_graph_groups_and_one_hot(self):
        from cubic_serve.guardrail.quantization import check_linear_enclosure
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators

        for bits, group, k in (
            (2, 80, 640),
            (3, 128, 256),
            (4, 40, 320),
            (5, 80, 640),
            (6, 128, 256),
            (7, 72, 1152),
            (8, 32, 256),
        ):
            for sd in (torch.float16, torch.bfloat16, torch.float32):
                w, expected = fixture(bits, n=64, k=k, group=group, dtype=sd, packing="legacy")
                for dt in (torch.bfloat16, torch.float16):
                    operators = PackedLinearOperators()
                    corpus = torch.randn(3, k, device="cuda", dtype=dt) * 0.1
                    common = None
                    for m in (1, 4, 17, 64):
                        count = min(3, m)
                        physical = torch.tensor([m - 1, m // 2, 0][:count], device="cuda")
                        x = torch.full((m, k), float("nan"), device="cuda", dtype=dt)
                        x[physical] = corpus[:count]
                        out = torch.empty(m, 64, device="cuda", dtype=dt)
                        binding = operators.bind_linear(x, w, out)
                        self.assertTrue(
                            any("int.marlin" in c.operator_id for c in binding._launches)
                        )
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            binding.launch()
                        graph.replay()
                        y = out[physical]
                        check_linear_enclosure(corpus[:count], w, y)
                        if common is not None:
                            self.assertTrue(torch.equal(y[:1], common))
                        common = y[:1].clone()
                        graph.reset()
                    # An identity input independently exposes every rounded weight.
                    x = torch.eye(k, device="cuda", dtype=dt)
                    out = torch.empty(k, 64, device="cuda", dtype=dt)
                    operators.bind_linear(x, w, out).launch()
                    self.assertTrue(torch.equal(out, expected.to(device="cuda", dtype=dt).T))

    def test_embedding_all_bits(self):
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators

        for bits in range(2, 9):
            w, expected = fixture(bits, n=67, k=160, group=32, packing="legacy")
            ids = torch.tensor([66, 1, 12, 1, 0], device="cuda")
            output = torch.empty(5, 160, device="cuda", dtype=torch.bfloat16)
            operators = PackedLinearOperators()
            binding = operators.bind_embedding(ids, w, output)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                binding.launch()
            graph.replay()
            self.assertTrue(torch.equal(output, expected[ids.cpu()].to("cuda").bfloat16()))
            graph.reset()

    def test_native_output_padding_codes_and_cross_m(self):
        from cubic_serve.guardrail.quantization import check_linear_enclosure
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
        from cubic_serve.quantization.extensions.int.marlin import bind_linear
        from cubic_serve.quantization.extensions.int.policy import IntegerLinearParameters

        for bits in range(2, 9):
            for n in (1, 63, 65, 160, 191):
                w, expected = fixture(
                    bits, n=n, k=256, group=128, packing="legacy", sym=bits % 2 == 0
                )
                for dt in (torch.float16, torch.bfloat16):
                    corpus = torch.randn(3, 256, device="cuda", dtype=dt) * 0.05
                    common = None
                    op = PackedLinearOperators()
                    for m in (1, 4, 17):
                        x = torch.full((m, 256), float("nan"), device="cuda", dtype=dt)
                        count = min(m, 3)
                        x[:count] = corpus[:count]
                        out = torch.empty((m, n), device="cuda", dtype=dt)
                        binding = bind_linear(
                            op, x, w, out, IntegerLinearParameters(marlin_auto_geometry=False)
                        )
                        self.assertIsNotNone(binding)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            binding.launch()
                        graph.replay()
                        check_linear_enclosure(corpus[:count], w, out[:count])
                        if common is not None:
                            self.assertTrue(torch.equal(common, out[:1]))
                        common = out[:1].clone()
                        expected_rows = out[:count].clone()
                        x[:count] = corpus[:count].flip(0)
                        graph.replay()
                        self.assertTrue(torch.equal(out[:count], expected_rows.flip(0)))
                        graph.reset()


@unittest.skipUnless(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") == "1" and torch.cuda.is_available(),
    "explicit CUDA audit required",
)
class IntegerMoETests(unittest.TestCase):
    def test_routed_preflight_checks_actual_kernel_weights_before_producers(self):
        from unittest.mock import patch

        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
        from cubic_serve.ops.cuda.moe.quantized import PackedRoutedExpertsPrimitive

        for native in (False, True):
            operators = PackedLinearOperators()
            x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16) * 0.01
            logits = torch.randn(2, 2, device="cuda", dtype=x.dtype)
            activity = torch.ones(2, device="cuda", dtype=torch.int32)
            active = torch.tensor([2], device="cuda", dtype=torch.int32)
            groups = []
            for n, k in ((128, 256), (128, 256), (256, 128)):
                groups.append(tuple(
                    fixture(4, n=n, k=k, group=64, sym=True)[0] if native
                    else torch.randn(n, k, device="cuda", dtype=x.dtype) * 0.01
                    for _ in range(2)
                ))
            block = PackedRoutedExpertsPrimitive(
                x, logits, activity, active, *groups, torch.empty_like(x),
                first_expert=0, top_k=1, operators=operators,
            )
            block.launch()
            if native:
                self.assertEqual(block._weight_tensors, ())
                weight = next(t for (_, t) in operators.prepared_weights.values()
                              if t.dtype == torch.int32 and t.ndim == 3)
            else:
                self.assertTrue(block._weight_tensors)
                weight = block._weight_tensors[0]
            weight.set_(torch.empty_like(weight))
            with patch.object(block.routing, "launch") as route:
                with self.assertRaisesRegex(ValueError, "operands changed"):
                    block.launch()
                route.assert_not_called()

    def test_grouped_tp_padding_preserves_one_marlin_and_math(self):
        from cubic_serve.guardrail.quantization import check_linear_enclosure
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
        from cubic_serve.quantization.api import RoutedRows
        from cubic_serve.quantization.extensions.int.marlin import bind_grouped
        from cubic_serve.quantization.extensions.int.policy import IntegerMoEParameters

        cases = [
            (bits, 160, 256, 128, packing, dtype)
            for bits in range(2, 9)
            for packing in ("dense", "legacy")
            for dtype in (torch.float16, torch.bfloat16)
        ] + [
            (bits, n, k, group, "dense", torch.bfloat16)
            for bits, n, k, group in (
                (5, 160, 2560, 128), (6, 160, 2560, 128),
                (5, 96, 160, 40), (8, 288, 288, 72), (4, 64, 256, 128),
                (3, 65, 256, 128), (7, 161, 256, 128),
            )
        ]
        for bits, n, k, group, packing, dtype in cases:
            weights = []
            for projection in range(2):
                experts = []
                for expert in range(2):
                    w, _ = fixture(bits, n, k, group, packing=packing)
                    w.payload["weight_scale"].mul_(1 + projection / 2 + expert / 4)
                    experts.append(w)
                weights.append(tuple(experts))
            torch.manual_seed(123)
            corpus = torch.randn(3, k, device="cuda", dtype=dtype) * 0.05
            ops = PackedLinearOperators()
            common = None
            for capacity in (1, 4, 17):
                count = min(3, capacity)
                physical = torch.arange(capacity - 1, capacity - 1 - count, -1,
                                        device="cuda", dtype=torch.int32)
                x = torch.full((capacity, k), float("nan"), device="cuda", dtype=dtype)
                x[physical.long()] = corpus[:count]
                counts = torch.full((2,), count, device="cuda", dtype=torch.int32)
                starts = torch.arange(2, device="cuda", dtype=torch.int32) * count
                blocks = torch.arange(2, device="cuda", dtype=torch.int32)
                total = torch.tensor([2], device="cuda", dtype=torch.int32)
                row_map = torch.cat([physical * 2 + i for i in range(2)])
                y = torch.full((capacity * 2, n * 2), float("nan"), device="cuda", dtype=dtype)
                # Keep small-K adaptation enabled, but geometry explicit across M.
                bk = math.gcd(128, k)
                bn = math.lcm(64, 8192 // math.gcd(bk, 8192))
                p = IntegerMoEParameters(marlin_block_k=bk, marlin_block_n=bn,
                                         marlin_auto_geometry=False)
                candidate = bind_grouped(
                    ops, x, weights[0], weights[1], y,
                    RoutedRows(counts, starts, blocks, total, row_map), 2, p,
                )
                self.assertIsNotNone(candidate)
                binding = candidate[1]
                kernels = [v.operator_id for v in binding._launches]
                self.assertEqual(sum("int.marlin.routed" in key for key in kernels), 1,
                                 (bits, n, k, packing, kernels))
                self.assertEqual(sum("_pad_routes" in key for key in kernels), 1)
                physical_n = math.ceil(2 * n / bn) * bn
                native = next(v for v in binding._launches if "int.marlin.routed" in v.operator_id)
                self.assertEqual(native.arguments[18].value, physical_n)
                self.assertEqual(sum("_unpad_output" in key for key in kernels),
                                 int(physical_n != 2 * n))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    binding.launch()
                graph.replay()
                actual = y.view(capacity, 2, 2, n)[physical.long()]
                for expert in range(2):
                    for projection in range(2):
                        check_linear_enclosure(corpus[:count], weights[projection][expert],
                                               actual[:, expert, projection])
                if common is not None:
                    self.assertTrue(torch.equal(common, actual[:1]))
                common = actual[:1].clone()
                graph.reset()

    def test_routed_native_bits_groups_poison_and_row_identity(self):
        from cubic_serve.guardrail.quantization import check_linear_enclosure
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
        from cubic_serve.quantization.api import RoutedRows
        from cubic_serve.quantization.extensions.int.marlin import _bind
        from cubic_serve.quantization.extensions.int.policy import (
            IntegerMoEParameters,
        )

        for bits in range(2, 9):
            for group, k in ((128, 256), (80, 640), (40, 640), (72, 1152)):
                for dt in (torch.float16, torch.bfloat16):
                    weights = tuple(
                        fixture(bits, n=64, k=k, group=group, sym=(e == 1))[0] for e in range(2)
                    )
                    # One routed launch must have a homogeneous encoding. Distinct expert values
                    # remain independent; mixed schemes are dispatched by the common executor.
                    weights = (weights[0], fixture(bits, n=64, k=k, group=group)[0])
                    corpus = torch.randn(3, k, device="cuda", dtype=dt) * 0.05
                    common = None
                    for capacity in (1, 4, 32):
                        m = min(capacity, 3)
                        e = 2
                        x = torch.full((capacity, k), float("nan"), device="cuda", dtype=dt)
                        physical = torch.tensor(
                            [capacity - 1, capacity // 2, 0][:m], device="cuda", dtype=torch.int32
                        )
                        x[physical.long()] = corpus[:m]
                        counts = torch.full((e,), m, device="cuda", dtype=torch.int32)
                        starts = torch.arange(e, device="cuda", dtype=torch.int32) * m
                        blocks = torch.arange(e, device="cuda", dtype=torch.int32)
                        total = torch.tensor([e], device="cuda", dtype=torch.int32)
                        row_map = torch.cat([physical * e + i for i in range(e)])
                        out = torch.full((capacity * e, 64), float("nan"), device="cuda", dtype=dt)
                        candidate = _bind(
                            PackedLinearOperators(),
                            x,
                            (weights,),
                            out,
                            IntegerMoEParameters(marlin_auto_geometry=False),
                            RoutedRows(counts, starts, blocks, total, row_map),
                            e,
                        )
                        self.assertIsNotNone(candidate)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            candidate[1].launch()
                        graph.replay()
                        y = out.view(capacity, e, 64)[physical.long()]
                        for i, w in enumerate(weights):
                            check_linear_enclosure(corpus[:m], w, y[:, i])
                        if common is not None:
                            self.assertTrue(torch.equal(common, y[:1]))
                        common = y[:1].clone()
                        graph.reset()

    def test_tp_routed_projection_padding_and_small_k(self):
        from cubic_serve.guardrail.quantization import check_linear_enclosure
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
        from cubic_serve.quantization.api import RoutedRows
        from cubic_serve.quantization.extensions.int.marlin import _bind_routed_projection
        from cubic_serve.quantization.extensions.int.policy import IntegerMoEParameters

        for bits in range(2, 9):
            for n, k, group in ((160, 256, 128), (256, 160, 40), (256, 288, 72)):
                for dt in (torch.float16, torch.bfloat16):
                    w, _ = fixture(bits, n=n, k=k, group=group)
                    corpus = torch.randn(2, k, device="cuda", dtype=dt) * 0.05
                    common = None
                    operators = PackedLinearOperators()
                    for capacity in (2, 4, 32):
                        x = torch.full((capacity, k), float("nan"), device="cuda", dtype=dt)
                        physical = torch.tensor([capacity - 1, 0], device="cuda", dtype=torch.int32)
                        x[physical.long()] = corpus
                        rows = RoutedRows(
                            torch.tensor([2], device="cuda", dtype=torch.int32),
                            torch.tensor([0], device="cuda", dtype=torch.int32),
                            torch.tensor([0], device="cuda", dtype=torch.int32),
                            torch.tensor([1], device="cuda", dtype=torch.int32), physical,
                        )
                        out = torch.full((capacity, n), float("nan"), device="cuda", dtype=dt)
                        candidate = _bind_routed_projection(
                            operators, x, (w,), out, rows, IntegerMoEParameters()
                        )
                        self.assertIsNotNone(candidate)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            candidate[1].launch()
                        graph.replay()
                        y = out[physical.long()]
                        check_linear_enclosure(corpus, w, y)
                        if common is not None:
                            self.assertTrue(torch.equal(common, y))
                        common = y.clone()
                        graph.reset()

    def test_tp_mixed_grouped_projection_padding(self):
        from cubic_serve.guardrail.quantization import check_linear_enclosure
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
        from cubic_serve.quantization.api import RoutedRows
        from cubic_serve.quantization.extensions.int.marlin import bind_grouped
        from cubic_serve.quantization.extensions.int.policy import IntegerMoEParameters

        gate, _ = fixture(8, n=160, k=256, group=128, sym=True)
        up, _ = fixture(4, n=160, k=256, group=128)
        x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16) * 0.05
        rows = RoutedRows(
            torch.tensor([2], device="cuda", dtype=torch.int32),
            torch.tensor([0], device="cuda", dtype=torch.int32),
            torch.tensor([0], device="cuda", dtype=torch.int32),
            torch.tensor([1], device="cuda", dtype=torch.int32),
            torch.tensor([1, 0], device="cuda", dtype=torch.int32),
        )
        out = torch.empty((2, 320), device="cuda", dtype=x.dtype)
        candidate = bind_grouped(
            PackedLinearOperators(), x, (gate,), (up,), out, rows, 1, IntegerMoEParameters()
        )
        self.assertIsNotNone(candidate)
        candidate[1].launch()
        check_linear_enclosure(x, gate, out[:, :160])
        check_linear_enclosure(x, up, out[:, 160:])

    def test_full_routing_swiglu_down_and_fixed_owner(self):
        from unittest.mock import patch

        from tests.ops.test_cuda_quantized_moe import PackedMoETests

        audit = PackedMoETests()
        torch.set_num_threads(2)
        for bits in range(2, 9):

            def matrix(n, k, fmt, dtype, index, bits=bits):
                w, _ = fixture(bits, n=n, k=k, group=32)
                # Different expert/role scales expose ownership and gate/up swaps.
                p = dict(w.payload)
                p["weight_scale"] = (p["weight_scale"] * (1 + index * 0.125)).contiguous()
                return QuantizedMatrix(p, w.encoding, w.shape)

            with patch.object(audit, "matrix", matrix):
                audit.audit_routed_formats(("int",), 128, 128, "cpu")

    def test_mixed_projection_bits_groups_require_marlin(self):
        from unittest.mock import patch

        from tests.ops.test_cuda_quantized_moe import PackedMoETests

        audit = PackedMoETests()

        # The shared MoE ABI (also used by Cubic) requires one encoding per
        # projection across local experts; gate/up/down may differ independently.
        schemes = iter(([(4, 80, False)] * 3 + [(5, 40, True)] * 3 + [(6, 128, False)] * 3) * 2)

        def matrix(n, k, fmt, dtype, index):
            bits, group, sym = next(schemes)
            w, _ = fixture(bits, n=n, k=k, group=group, sym=sym, packing="legacy")
            payload = dict(w.payload)
            payload["weight_scale"] = (payload["weight_scale"] * (1 + index * 0.125)).contiguous()
            return QuantizedMatrix(payload, w.encoding, w.shape)

        with patch.object(audit, "matrix", matrix), patch(
            "cubic_serve.quantization.extensions.int.moe.operands",
            side_effect=AssertionError("INT projection escaped native Marlin"),
        ):
            audit.audit_routed_formats(("int",), 640, 640, "cpu")

    def test_large_expert_inventory_calibration_uses_integral_anchors(self):
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
        from cubic_serve.ops.cuda.moe.quantized import PackedRoutedExpertsPrimitive

        e, h, intermediate, topk = 128, 2560, 640, 10
        gate = fixture(4, n=intermediate, k=h, group=128)[0]
        down = fixture(4, n=h, k=intermediate, group=80)[0]
        x = torch.full((1, h), 0.01, device="cuda", dtype=torch.bfloat16)
        logits = torch.arange(e, device="cuda", dtype=torch.bfloat16)[None].contiguous()
        active = torch.ones(1, device="cuda", dtype=torch.int32)
        out = torch.empty_like(x)
        op = PackedRoutedExpertsPrimitive(
            x,
            logits,
            active,
            active,
            (gate,) * e,
            (gate,) * e,
            (down,) * e,
            out,
            first_expert=0,
            top_k=topk,
            operators=PackedLinearOperators(),
        )
        op.launch()
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(int(op.counts.sum()), topk)
        self.assertEqual(set(op.route_ids.flatten().tolist()), set(range(e - topk, e)))


class IntegerCheckpointTests(unittest.TestCase):
    def test_checkpoint_header_admission_and_owner_materialization(self):
        import json
        import tempfile
        from pathlib import Path

        from safetensors.torch import save_file

        from cubic_serve.models.parallel_weights import AxisSpan
        from cubic_serve.models.quantized_checkpoint import LogicalWeight, QuantizedCheckpoint

        for bits in range(2, 9):
            for packing in ("dense", "legacy"):
                w, expected = fixture(bits, n=67, k=259, group=40, packing=packing)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    config = {
                        "quantization_config": {
                            "quant_method": "compressed-tensors",
                            "format": "pack-quantized",
                            "config_groups": {
                                "integer": {
                                    "targets": ["projection"],
                                    "weights": {
                                        "num_bits": bits,
                                        "type": "int",
                                        "strategy": "group",
                                        "group_size": 40,
                                        "symmetric": False,
                                    },
                                }
                            },
                        }
                    }
                    (root / "config.json").write_text(json.dumps(config))
                    save_file(
                        {"projection." + key: value for key, value in w.payload.items()},
                        root / "model.safetensors",
                    )
                    logical = LogicalWeight(
                        "projection.weight", w.shape, "BF16", "language", "linear"
                    )
                    plan = QuantizedCheckpoint.admit(root, (logical,), verify_payloads=False)
                    owner = plan.weights[0]
                    for axis, spans, target in (
                        (
                            0,
                            (AxisSpan(3, 9), AxisSpan(60, 67)),
                            torch.cat((expected[3:9], expected[60:67])),
                        ),
                        (1, (AxisSpan(17, 244),), expected[:, 17:244]),
                    ):
                        part = plan.materialize(owner, axis=axis, spans=spans)
                        self.assertTrue(
                            torch.equal(decode(part.payload, part.encoding, part.shape), target)
                        )
                    # Logical geometry comes from the model, not an untrusted payload field.
                    payload = {"projection." + key: value for key, value in w.payload.items()}
                    payload["projection.weight_shape"] = torch.tensor([67, 260])
                    save_file(payload, root / "model.safetensors")
                    corrupt = QuantizedCheckpoint.admit(root, (logical,), verify_payloads=False)
                    with self.assertRaisesRegex(ValueError, "weight_shape"):
                        corrupt.materialize(corrupt.weights[0])

    def test_legacy_word_rebase_matches_independent_slots_without_mutating_source(self):
        from cubic_serve.quantization.extensions.int.layout import _rebase_legacy_words

        # Include signed words and poisoned unused high bits. Decode code slots
        # and rebuild destination words with scalar integers, independently of
        # the production pack/unpack routines.
        unsigned = [0xFFFFFFFF, 0x80000000, 0x12345678, 0xDEADBEEF, 0x7FFFFFFF]
        raw = torch.tensor(
            [[x if x < (1 << 31) else x - (1 << 32) for x in unsigned]],
            dtype=torch.int32,
        ).repeat(3, 1)
        for bits in range(2, 9):
            slots = 32 // bits
            codes = [(x >> (slot * bits)) & ((1 << bits) - 1)
                     for x in unsigned for slot in range(slots)]
            for start in range(2 * slots):
                for count in (1, slots - 1, slots, slots + 1, 2 * slots + 1):
                    selected = codes[start:start + count]
                    expected = []
                    for base in range(0, count, slots):
                        word = sum(x << (j * bits) for j, x in enumerate(selected[base:base + slots]))
                        expected.append(word if word < (1 << 31) else word - (1 << 32))
                    expected = torch.tensor([expected], dtype=torch.int32).repeat(3, 1)
                    cover = raw[:, start // slots:(start + count + slots - 1) // slots]
                    for axis in (0, 1):
                        source = cover.T if axis == 0 else cover
                        before = source.clone()
                        actual = _rebase_legacy_words(source, bits, start, count, axis)
                        self.assertTrue(torch.equal(actual, expected.T if axis == 0 else expected))
                        self.assertTrue(torch.equal(source, before))

    def test_ambiguous_packing(self):
        from types import SimpleNamespace

        from cubic_serve.models.quantized_checkpoint import LogicalWeight
        from cubic_serve.quantization.extensions.int.layout import admit_encoding

        w, _ = fixture(5, n=2, k=2, group=2, sym=True)
        records = {
            "projection." + key: SimpleNamespace(
                shape=tuple(value.shape),
                dtype={torch.int32: "I32", torch.int64: "I64", torch.float16: "F16"}[value.dtype],
            )
            for key, value in w.payload.items()
        }
        enc = QuantizationScheme("int-pack-v1", 5, 2, metadata="auto:sym:auto")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            admit_encoding(
                LogicalWeight("projection.weight", (2, 2), "BF16", "language", "linear"),
                enc,
                records,
            )


@unittest.skipUnless(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") == "1" and torch.cuda.is_available(),
    "explicit CUDA audit required",
)
class IntegerShardedEmbeddingTests(unittest.TestCase):
    def test_host_cross_m_original_rows_and_permuted_padding(self):
        from cubic_serve.ops.cuda.embedding.token import bind_scaled_sharded_embedding

        for bits in (4, 5, 8):
            matrix, decoded = fixture(bits, n=67, k=160, group=32, sym=True)
            host = QuantizedMatrix({k: t.pin_memory() for k, t in matrix.payload.items()},
                                   matrix.encoding, matrix.shape)
            corpus = torch.tensor([[0, 66], [17, 9], [31, 45]])
            expected = (decoded[corpus].float() * 0.75).bfloat16().flatten(1)
            common = {}
            for m in (1, 4, 16, 128):
                ids = torch.zeros((m, 2), dtype=torch.int64, device="cuda")
                active = torch.ones(m, dtype=torch.int32, device="cuda")
                out = torch.empty((m, 320), dtype=torch.bfloat16, device="cuda")
                binding = bind_scaled_sharded_embedding(
                    ids, ((0, host),), torch.tensor([0.75], device="cuda"), active, out)
                binding.launch()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    binding.launch()
                maps = ((0,), (1,), (2,)) if m == 1 else ((0, 1, 2), (2, 0, 1))
                for mapping in maps:
                    ids.fill_(-(2**60))
                    active.zero_()
                    out.fill_(77)
                    for physical, logical in enumerate(mapping):
                        ids[physical].copy_(corpus[logical])
                        active[physical] = 1
                    graph.replay()
                    actual = out.cpu()
                    for physical, logical in enumerate(mapping):
                        assert torch.equal(actual[physical], expected[logical])
                        if logical in common:
                            assert torch.equal(actual[physical], common[logical])
                        common[logical] = actual[physical].clone()
                    assert bool((actual[len(mapping):] == 77).all())

    def test_owned_rows_gaps_padding_and_graph(self):
        self._owned_rows(host=False)

    def test_host_owned_rows_gaps_padding_and_graph(self):
        self._owned_rows(host=True)

    def _owned_rows(self, *, host):
        from cubic_serve.ops.cuda.embedding.token import bind_scaled_sharded_embedding

        for bits in range(2, 9):
            for packing in ("dense", "legacy"):
                for sym in (False, True):
                    a, da = fixture(bits, n=67, k=259, group=40, sym=sym, packing=packing)
                    b, db = fixture(bits, n=33, k=259, group=40, sym=sym, packing=packing)

                    def device(w):
                        return QuantizedMatrix(
                            {k: v.pin_memory() if host else v.cuda()
                             for k, v in w.payload.items()}, w.encoding, w.shape
                        )

                    ids = torch.tensor(
                        [[7, 73, 85], [100, 132, 6], [-999, -999, -999]],
                        device="cuda",
                        dtype=torch.int64,
                    )
                    active = torch.tensor([1, 1, 0], device="cuda", dtype=torch.int32)
                    out = torch.full((3, 3 * 259), -97.0, device="cuda", dtype=torch.bfloat16)
                    scale = torch.tensor([0.75], device="cuda")
                    binding = bind_scaled_sharded_embedding(
                        ids, ((7, device(a)), (100, device(b))), scale, active, out
                    )
                    binding.launch()
                    expected = (
                        torch.stack(
                            (da[0], da[66], torch.zeros(259), db[0], db[32], torch.zeros(259))
                        )
                        .reshape(2, -1)
                        .mul(0.75)
                        .bfloat16()
                        .cuda()
                    )
                    self.assertTrue(torch.equal(out[:2], expected))
                    self.assertTrue(torch.all(out[2] == -97))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        binding.launch()
                    ids[:2].copy_(ids[:2].flip(0))
                    out.fill_(-97)
                    graph.replay()
                    self.assertTrue(torch.equal(out[:2], expected.flip(0)))
                    self.assertTrue(torch.all(out[2] == -97))


@unittest.skipUnless(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") == "1" and torch.cuda.is_available(),
    "explicit CUDA audit required",
)
class IntegerRoutedPartitionTests(unittest.TestCase):
    def test_fixed_k_partials_preserve_experts_rows_and_padding(self):
        from unittest.mock import patch

        from cubic_serve.guardrail.quantization import check_linear_enclosure
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
        from cubic_serve.quantization.api import RoutedRows
        from cubic_serve.quantization.extensions.int.marlin import _bind_routed_projection
        from cubic_serve.quantization.extensions.int.policy import IntegerMoEParameters

        cases = [(bits, 128, 512, 4) for bits in range(2, 9)]
        cases += [(5, 80, 1280, 2), (4, 40, 320, 5)]
        for bits, group, k, partitions in cases:
            n = 256
            weights = tuple(fixture(bits, n, k, group, packing="legacy")[0] for _ in range(2))
            weights[1].payload["weight_scale"].mul_(1.25)
            bk = math.gcd(128, k)
            bn = math.lcm(64, 8192 // math.gcd(bk, 8192))
            params = IntegerMoEParameters(
                marlin_block_m=8, marlin_block_n=bn, marlin_block_k=bk,
                marlin_auto_geometry=False,
            )
            for dtype in (torch.float16, torch.bfloat16):
                torch.manual_seed(123)
                corpus = torch.randn(9, k, device="cuda", dtype=dtype) * 0.05
                operators = PackedLinearOperators()
                common = None
                for capacity in (1, 4, 17):
                    count = min(capacity, 9)
                    physical = torch.arange(capacity - 1, capacity - count - 1, -1,
                                            device="cuda", dtype=torch.int32)
                    x = torch.full((capacity, k), float("nan"), device="cuda", dtype=dtype)
                    x[physical.long()] = corpus[:count]
                    counts = torch.full((2,), count, device="cuda", dtype=torch.int32)
                    starts = torch.arange(2, device="cuda", dtype=torch.int32) * count
                    tiles = math.ceil(count / 8)
                    blocks = torch.arange(2, device="cuda", dtype=torch.int32) * tiles
                    total = torch.tensor([2 * tiles], device="cuda", dtype=torch.int32)
                    row_map = torch.cat([physical * 2 + expert for expert in range(2)])
                    rows = RoutedRows(counts, starts, blocks, total, row_map, routing_top_k=2)
                    output = torch.full((capacity * 2, n), float("nan"), device="cuda", dtype=dtype)
                    with patch(
                        "cubic_serve.quantization.extensions.int.marlin.plan_dense_partitions",
                        return_value=partitions,
                    ):
                        _, binding = _bind_routed_projection(
                            operators, x, weights, output, rows, params, top_k=2
                        )
                    for scratch in binding.scratch:
                        if scratch.dtype == torch.float32:
                            scratch.fill_(float("nan"))
                    binding.launch()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        binding.launch()
                    graph.replay()
                    actual = output.view(capacity, 2, n)[physical.long()]
                    for expert, weight in enumerate(weights):
                        check_linear_enclosure(corpus[:count], weight, actual[:, expert])
                    if common is not None:
                        self.assertTrue(torch.equal(common, actual[:1]))
                    common = actual[:1].clone()
                    inactive = torch.ones(capacity, device="cuda", dtype=torch.bool)
                    inactive[physical.long()] = False
                    self.assertTrue(torch.isnan(output.view(capacity, 2, n)[inactive]).all())
                    x[physical.long()] = corpus[:count].flip(0)
                    graph.replay()
                    self.assertTrue(torch.equal(
                        actual.flip(0), output.view(capacity, 2, n)[physical.long()]
                    ))
                    graph.reset()


@pytest.mark.parametrize(
    "warps,k,dense_bn,routed_bn",
    [(4, 160, 256, 128), (8, 160, 512, 256), (4, 320, 128, 64), (8, 320, 256, 128)],
)
def test_alignment_preserves_dense_pipeline(warps, k, dense_bn, routed_bn):
    from cubic_serve.quantization.extensions.int.marlin import _align_k_tile
    from cubic_serve.quantization.extensions.int.policy import (
        IntegerLinearParameters,
        IntegerMoEParameters,
    )

    operators = PackedLinearOperators()
    dense = _align_k_tile(
        operators, IntegerLinearParameters(marlin_warps=warps), 10240, k, family="int_linear"
    )
    routed = _align_k_tile(
        operators, IntegerMoEParameters(marlin_warps=warps), 2560, k, family="int_moe"
    )
    assert (dense.marlin_block_n, routed.marlin_block_n) == (dense_bn, routed_bn)
    assert k % dense.marlin_block_k == k % routed.marlin_block_k == 0
    assert (
        dense.marlin_block_k * dense.marlin_block_n
        == 2 * routed.marlin_block_k * routed.marlin_block_n
    )


@pytest.mark.parametrize("warps,k", [(4, 16), (4, 48), (4, 80), (8, 80)])
def test_k16_alignment_closes_single_subtile_ownership(warps, k):
    from cubic_serve.quantization.common.marlin.parameters import pipeline_compatible
    from cubic_serve.quantization.extensions.int.marlin import _align_k_tile
    from cubic_serve.quantization.extensions.int.policy import IntegerLinearParameters

    p = _align_k_tile(
        PackedLinearOperators(), IntegerLinearParameters(marlin_warps=warps),
        1024, k, family="int_linear",
    )
    assert p.marlin_block_k == 16
    assert p.marlin_block_n == warps * 64
    assert pipeline_compatible(p, group=16, single_subtile=True)


@pytest.mark.parametrize("family", ["int_linear", "int_moe"])
def test_explicit_geometry_remains_authoritative(family):
    from cubic_serve.quantization.extensions.int.marlin import _align_k_tile
    from cubic_serve.quantization.extensions.int.policy import (
        IntegerLinearParameters,
        IntegerMoEParameters,
    )

    schema = IntegerLinearParameters if family == "int_linear" else IntegerMoEParameters
    p = schema(marlin_auto_geometry=False)
    assert _align_k_tile(PackedLinearOperators(), p, 2560, 160, family=family) is p


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("bits", [3, 6, 8])
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("packing", ["dense", "legacy"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reference_decode_coalescing_retains_canonical_gemm_bits(device, bits, sym, packing, dtype):
    if device == "cuda" and os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1":
        pytest.skip("explicit CUDA audit required")
    from cubic_serve.executor.runner import ReferenceExecutionContext
    from cubic_serve.ops.eager.quantization import linear_matrix

    # Include an incomplete last output tile and non-word-aligned packed codes.
    matrix, _ = fixture(bits, n=520, k=320, group=64, sym=sym, dtype=dtype, packing=packing)
    inputs = (torch.arange(3 * 320).reshape(3, 320).remainder(37).float() - 18) / 64
    inputs = inputs.to(device=device, dtype=dtype)
    with ReferenceExecutionContext():
        original = linear_matrix(inputs, matrix, decode_rows=256)
        grouped = linear_matrix(inputs, matrix)
    assert torch.equal(original, grouped)


@pytest.mark.parametrize("bits", range(2, 9))
@pytest.mark.parametrize("packing", ["dense", "legacy"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_int_aligned_zero_words_preserve_original_payload_and_group_coordinates(
    bits, packing, device,
):
    from unittest.mock import patch

    from cubic_serve.quantization.extensions.int import layout

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    row_unit = 32 // math.gcd(32, bits) if packing == "dense" else 32 // bits
    matrix, expected = fixture(
        bits, n=4 * row_unit, k=259, group=40, packing=packing,
        dtype=torch.bfloat16,
    )
    # Legacy's unused top bits are not codes and the old repack clears them.
    used = (32 // bits) * bits
    if packing == "legacy" and used < 32:
        matrix.payload["weight_zero_point"].bitwise_or_(-(1 << used))
    matrix = matrix.to(device)
    for start, end in ((0, 2 * row_unit), (row_unit, 3 * row_unit)):
        for column_start, column_end in ((0, 259), (17, 244)):
            # The independent old recipe decodes/re-packs only the selected
            # zero codes, then repeats the original scale-group coordinates.
            old_zero = layout.unpack(
                matrix.payload["weight_zero_point"], bits, matrix.shape[0],
                axis=0, packing=packing,
            )[start:end]
            local_group = (40 if column_start == 0 and column_end == 259 else
                           math.gcd(40, math.gcd(column_start, column_end - column_start)))
            groups = torch.arange(column_start, column_end, local_group, device=device) // 40
            wanted_zero = layout.pack(old_zero[:, groups], bits, packing, axis=0)
            wanted_scale = matrix.payload["weight_scale"][start:end][:, groups]
            # Columns span complete original weight words only in the full-K
            # case. Its zero-word path must not call either old codec.
            if column_start == 0:
                with (
                    patch.object(layout, "unpack", side_effect=AssertionError("zero decode")),
                    patch.object(layout, "pack", side_effect=AssertionError("zero repack")),
                ):
                    owned = matrix.rows(start, end)
            else:
                owned = matrix.rows(start, end).columns(column_start, column_end)
            assert torch.equal(owned.payload["weight_zero_point"], wanted_zero)
            assert torch.equal(owned.payload["weight_scale"], wanted_scale)
            assert torch.equal(
                decode(owned.payload, owned.encoding, owned.shape).cpu(),
                expected[start:end, column_start:column_end],
            )
