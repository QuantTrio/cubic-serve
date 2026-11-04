"""Lossless carrier and consumer tests; these are not serving qualification."""
from __future__ import annotations

import pytest
import torch

from cubic_serve.quantization.extensions.ggml.q8_storage import pack


@pytest.fixture(scope="module", autouse=True)
def _reference_policy_before_allocations():
    from cubic_serve.executor.runner import configure_reference_math

    configure_reference_math()


def _restore(storage, n, k, experts):
    """Invert whole planes on CPU, independently of CUDA address expressions."""
    kb, nt = k // 32, (n + 15) // 16
    plane = storage.cpu().reshape(experts, -1)
    codes = plane[:, :kb * nt * 512].contiguous().reshape(experts, nt * 16, kb, 32)
    scales = plane[:, kb * nt * 512:].contiguous().reshape(experts, nt * 16, kb, 2)
    raw = torch.cat((scales, codes), dim=-1)
    return raw[:, :n].reshape(experts, n, kb * 34), raw[:, n:]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n,k,experts", [(1, 32, 1), (17, 320, 1), (64, 512, 3),
                                       (320, 10240, 1)])
@pytest.mark.parametrize("source_device", ["cpu", "cuda"])
def test_q8_storage_lossless_planes_and_original_expert_ownership(n, k, experts, source_device):
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(714)
    raw = torch.randint(0, 256, (experts, n, k // 32 * 34), dtype=torch.uint8)
    expected = raw.clone()
    source = raw.to(device) if source_device == "cuda" else raw
    original = source if experts > 1 else source[0]
    prepared = pack(original, k=k, device=device)
    actual, padding = _restore(prepared.storage, n, k, experts)
    assert torch.equal(actual, expected)
    assert not bool(padding.any())
    assert torch.equal(source.cpu(), expected)
    assert prepared.storage.numel() == experts * ((n + 15) // 16) * 16 * (k // 32) * 34
    assert prepared.codes.untyped_storage().data_ptr() == prepared.storage.data_ptr()
    assert prepared.scales.untyped_storage().data_ptr() == prepared.storage.data_ptr()
    assert prepared.codes.numel() + prepared.scales.numel() == prepared.storage.numel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("k", [320, 640])
@pytest.mark.parametrize("m", [1, 4, 17])
@pytest.mark.parametrize("members", [1, 2])
@pytest.mark.parametrize("arithmetic", ["ordered", "full32"])
def test_q8_prepared_projection_uses_existing_independent_oracle(
    monkeypatch, dtype, k, m, members, arithmetic,
):
    from tests.ops.test_ggml_llama import (
        test_q8_group_independent_math_member_tails_and_original_reduction as check,
    )
    from cubic_serve.quantization.extensions.ggml import q8_group, q8_mma
    from cubic_serve.executor.runner import configure_reference_math

    configure_reference_math()
    original_group, original_mma = q8_group._bind, q8_mma.bind

    def prepare_members(x, weights):
        return tuple(pack(w, k=x.shape[1], device=x.device) for w in weights)

    def group(x, weights, *args, **kwargs):
        return original_group(x, prepare_members(x, weights), *args, **kwargs)

    def mma(x, weights, *args, **kwargs):
        return original_mma(x, prepare_members(x, weights), *args, **kwargs)

    monkeypatch.setattr(q8_group, "_bind", group)
    monkeypatch.setattr(q8_mma, "bind", mma)
    check(dtype, k, m, members, arithmetic)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("k", [32, 160, 640])
def test_q8_prepared_matrix_tail_uses_existing_independent_oracle(monkeypatch, dtype, k):
    from tests.ops import test_ggml_llama as oracle

    original = oracle.bind_matmul

    def bind(x, weight, type_id, *args, **kwargs):
        return original(x, pack(weight, k=x.shape[1], device=x.device), type_id, *args, **kwargs)

    monkeypatch.setattr(oracle, "bind_matmul", bind)
    oracle.test_matrix_short_final_weight_tile(oracle.Kind.Q8_0, k, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("short_tail", [False, True])
def test_q8_prepared_matrix_original_experts_and_permuted_rows(monkeypatch, top_k, short_tail):
    from tests.ops import test_ggml_llama as oracle

    original = oracle.bind_matmul

    def bind(x, weight, type_id, *args, **kwargs):
        return original(x, pack(weight, k=x.shape[1], device=x.device), type_id, *args, **kwargs)

    monkeypatch.setattr(oracle, "bind_matmul", bind)
    oracle.test_matrix_routed_prefix_and_permuted_original_owners(oracle.Kind.Q8_0, top_k, short_tail)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_q8_projection_and_device_lookup_share_one_prepared_allocation():
    from tests.ops.test_ggml_llama import _weight, Kind
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.extensions.ggml.checkpoint import scheme
    from cubic_serve.quantization.extensions.ggml.linear import bind_native
    from cubic_serve.quantization.extensions.ggml.policy import GGMLLinearParameters

    device = torch.device("cuda", torch.cuda.current_device())
    n, k = 67, 320
    raw, decoded = _weight(Kind.Q8_0, n, k)
    weight = QuantizedMatrix({"weight_packed": torch.from_numpy(raw)}, scheme(8), (n, k))
    ops = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
    x = torch.randn(1, k, dtype=torch.bfloat16, device=device)
    projection = bind_native(ops, x, weight, torch.empty(1, n, dtype=x.dtype, device=device),
                             GGMLLinearParameters())
    ids = torch.tensor([[11], [12], [77], [8]], dtype=torch.int64, device=device)
    activity = torch.tensor([1, 0, 1, 1], dtype=torch.int32, device=device)
    output = torch.full((4, k), 19, dtype=x.dtype, device=device)
    scale = torch.tensor([0.5], device=device)
    lookup = ops.bind_scaled_sharded_embedding(ids, ((11, weight),), scale, activity, output)
    assert len(ops.prepared_weights) == 1
    assert not ops.materialized_weights
    carrier = next(iter(ops.prepared_weights.values()))[1]
    assert lookup.payloads[0].data_ptr() == carrier.data_ptr()
    assert any(any(isinstance(a, torch.Tensor) and a.data_ptr() == carrier.data_ptr()
                   for a in launch.arguments) for launch in projection._launches)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        lookup.launch()
    graph.replay()
    torch.cuda.synchronize()
    expected = torch.full((4, k), 19, dtype=x.dtype)
    expected[0] = (decoded[0] * 0.5).to(x.dtype)
    expected[2] = (decoded[-1] * 0.5).to(x.dtype)
    expected[3].zero_()
    assert torch.equal(output.cpu(), expected)
    graph.reset()
