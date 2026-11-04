"""Small complete graph: GDN, PLE, sparse selection, MoE and immutable state."""

from dataclasses import fields, replace

import pytest
import torch

from cubic_serve.executor.runner import ReferenceExecutionContext
from cubic_serve.models.qwen4_exp import parse_qwen4_exp_config
from cubic_serve.models.qwen4_exp.model import AttentionKind, Qwen4ExpReferenceModel
from cubic_serve.models.qwen4_exp.weights import _expected_headers
from cubic_serve.ops.eager.linear import linear
from tests.models.qwen4_exp.test_config import _fixture


def tiny_model():
    original = parse_qwen4_exp_config(_fixture())
    t = original.text
    text = replace(
        t,
        hidden_size=16,
        vocabulary_size=32,
        layer_count=2,
        layer_schedule=(AttentionKind.GATED_DELTA_NET, AttentionKind.FULL_ATTENTION),
        hyperconnection=replace(t.hyperconnection, stream_count=2, low_rank=4),
        gated_delta_net=replace(
            t.gated_delta_net, key_heads=2, value_heads=2, key_head_dim=4, value_head_dim=4
        ),
        full_attention=replace(t.full_attention, query_heads=2, key_value_heads=1, head_dim=8),
        mrope=replace(t.mrope, partial_rotary_factor=1.0, section=(2, 1, 1)),
        qsa=replace(t.qsa, query_heads=2, head_dim=8, compression_ratio=2, token_budget=2),
        ple=replace(t.ple, embedding_dim=16, heads_per_ngram=2, split_parts=2, layer_ids=(1,)),
        moe=replace(
            t.moe,
            expert_count=4,
            experts_per_token=2,
            expert_intermediate_size=8,
            shared_expert_intermediate_size=8,
        ),
        eos_token_id=31,
    )
    c = replace(original, text=text)
    torch.manual_seed(87)
    weights = {}
    for name, h in _expected_headers(c).items():
        if h.partition not in ("language", "lm_head") or ".ple_embedding." in name:
            continue
        value = (torch.randn(h.shape) * 0.02).bfloat16()
        if name.endswith(".experts.gate_up_proj"):
            for e in range(4):
                weights[name.rsplit(".", 1)[0] + f".{e}.gate_proj.weight"] = value[e, :8]
                weights[name.rsplit(".", 1)[0] + f".{e}.up_proj.weight"] = value[e, 8:]
        elif name.endswith(".experts.down_proj"):
            for e in range(4):
                weights[name.rsplit(".", 1)[0] + f".{e}.down_proj.weight"] = value[e]
        else:
            weights[name] = value
    base = "model.language_model.layers.0.ple.ple_embedding"
    weights[base + ".layer_multipliers"] = torch.tensor([10007, 10009, 10037])
    weights[base + ".ngram_heads_vocab_sizes"] = torch.tensor([11, 13, 17, 19])
    weights[base + ".ngram_heads_offsets"] = torch.tensor([0, 11, 24, 41])
    table = (torch.randn(60, 4) * 0.05).bfloat16()

    class Operators:
        linear = staticmethod(linear)

        @staticmethod
        def embedding(ids, weight, *, dtype):
            return weight[ids].to(dtype)

        @staticmethod
        def embedding_shards(ids, base, weights, *, parts, dtype):
            return table[ids].to(dtype)

    return Qwen4ExpReferenceModel(c, weights, Operators())


def test_visual_embeddings_and_three_axis_positions_survive_step_boundaries():
    model = tiny_model()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    embeddings = model.operators.embedding(
        tokens, model.weights["model.language_model.embed_tokens.weight"], dtype=torch.bfloat16
    ).clone()
    embeddings[:, 1:3] += .25
    positions = torch.tensor([[[0, 1, 1, 2]], [[0, 1, 2, 3]], [[0, 2, 1, 3]]])
    with ReferenceExecutionContext():
        whole, state = model.forward(tokens, input_embeddings=embeddings, position_ids=positions)
        pieces, current = [], None
        for i in range(tokens.shape[1]):
            output, current = model.forward(
                tokens[:, i:i+1], current, input_embeddings=embeddings[:, i:i+1],
                position_ids=positions[:, :, i:i+1],
            )
            pieces.append(output)
        text, _ = model.forward(tokens)
        with pytest.raises(ValueError, match="explicit positions"):
            model.forward(tokens[:, :1], current)
    assert torch.equal(torch.cat(pieces, dim=1), whole)
    assert not torch.equal(text, whole)
    assert torch.equal(state.layers[1].index_positions, positions)
    assert torch.equal(current.layers[1].index_positions, positions)
    assert torch.equal(state.layers[1].compressed_index_keys,
                       current.layers[1].compressed_index_keys)


def test_reference_visual_input_width_and_position_admission():
    model = tiny_model()
    tokens = torch.tensor([[1]], dtype=torch.int64)
    with ReferenceExecutionContext():
        with pytest.raises(ValueError, match="feature boundary"):
            model.forward(tokens, input_embeddings=torch.zeros(1, 1, 17).bfloat16())
        with pytest.raises(ValueError, match="position axes"):
            model.forward(tokens, position_ids=torch.tensor([[[-1]], [[0]], [[0]]]))


@pytest.mark.parametrize("dtype", ["fp8", "int8", "int6", "int4"])
def test_complete_reference_model_inherits_generic_cache_attention_without_format_adapter(dtype):
    from cubic_serve.ops.eager.kv_cache import ReferenceCacheOperators
    from cubic_serve.runtime.cache.quantization import (
        CacheDType, CacheGroupAxis, CacheStorageSpec, KVCacheStorageSpec,
    )

    model = tiny_model()
    storage = KVCacheStorageSpec(
        CacheStorageSpec(CacheDType(dtype), CacheGroupAxis.TOKENS, 4),
        CacheStorageSpec(CacheDType(dtype), CacheGroupAxis.CHANNELS, 4),
    )
    model.operators = ReferenceCacheOperators(model.operators, storage)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    with ReferenceExecutionContext():
        first, prefix = model.forward(tokens[:, :3])
        # Both continuations keep independent raw producer state. A rejected
        # branch must not alter the shared prefix or its future quantization.
        before = prefix.layers[1].key.clone()
        _, rejected = model.forward(torch.tensor([[27]]), prefix)
        assert torch.equal(before, prefix.layers[1].key)
        tail, state = model.forward(tokens[:, 3:], prefix)
        whole, whole_state = model.forward(tokens)
    assert torch.equal(torch.cat((first, tail), dim=1), whole)
    assert torch.equal(state.layers[1].key, whole_state.layers[1].key)
    assert not torch.equal(rejected.layers[1].key, state.layers[1].key[:, :, :4])


@pytest.mark.parametrize("owners", [2, 4])
def test_expert_owners_fold_before_shared_expert(owners):
    """Missing nonowned weights are valid; owner sums precede the shared branch."""
    from cubic_serve.ops.eager.activation import sigmoid_stored, silu_stored
    from cubic_serve.ops.eager.moe import topk_routes

    model = tiny_model()
    prefix = "model.language_model.layers.0.mlp"
    hidden = torch.randn(3, 16).bfloat16()
    expert_count = model.config.text.moe.expert_count
    with ReferenceExecutionContext():
        ids, probabilities = topk_routes(
            model._linear(hidden, prefix + ".gate"), top_k=2,
            normalize=model.config.text.moe.normalize_topk_probability,
        )

        def expert(x, name):
            return model._linear(
                silu_stored(model._linear(x, name + ".gate_proj"))
                * model._linear(x, name + ".up_proj"), name + ".down_proj",
            )

        partials = [torch.zeros_like(hidden) for _ in range(owners)]
        for row in range(hidden.shape[0]):
            for slot, eid in enumerate(ids[row].tolist()):
                owner = eid // (expert_count // owners)
                partials[owner][row] += (
                    expert(hidden[row:row + 1], prefix + f".experts.{eid}")[0]
                    * probabilities[row, slot]
                )
        completed = torch.zeros_like(hidden, dtype=torch.float32)
        for part in partials:
            completed += part.float()
        completed = completed.to(hidden.dtype)
        expected = completed + expert(hidden, prefix + ".shared_expert") * sigmoid_stored(
            model._linear(hidden, prefix + ".shared_expert_gate")
        )
        for owner in range(owners):
            first, end = owner * expert_count // owners, (owner + 1) * expert_count // owners
            weights = {
                name: weight for name, weight in model.weights.items()
                if ".experts." not in name
                or first <= int(name.split(".experts.")[1].split(".")[0]) < end
            }
            calls = []

            def reduce(value, owner=owner, calls=calls):
                torch.testing.assert_close(value, partials[owner], rtol=0, atol=0)
                calls.append(True)
                return completed.clone()

            local = Qwen4ExpReferenceModel(
                model.config, weights, model.operators,
                expert_range=(first, end), reduce_experts=reduce,
            )
            torch.testing.assert_close(local._mlp(hidden, prefix), expected, rtol=0, atol=0)
            assert calls == [True]


def test_partitioned_reference_experts_require_completion():
    model = tiny_model()
    with pytest.raises(ValueError, match="owner reduction"):
        Qwen4ExpReferenceModel(model.config, model.weights, model.operators, expert_range=(0, 2))


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_complete_reference_prefill_decode_and_prefix_state_identity(state_dtype):
    model = tiny_model()
    model = Qwen4ExpReferenceModel(
        model.config, model.weights, model.operators, recurrent_state_dtype=state_dtype
    )
    tokens = torch.tensor([[1, 5, 31, 9, 2, 8]])
    with ReferenceExecutionContext():
        whole, final = model.forward(tokens)
        first, prefix = model.forward(tokens[:, :3])
        snapshot = [
            getattr(s, f.name).clone()
            for s in prefix.layers
            for f in fields(s)
            if getattr(s, f.name) is not None
        ]
        second, resumed = model.forward(tokens[:, 3:], prefix)
        state = None
        rows = []
        for t in range(tokens.shape[1]):
            out, state = model.forward(tokens[:, t : t + 1], state)
            rows.append(out)
    assert torch.equal(whole, torch.cat((first, second), 1))
    assert torch.equal(whole, torch.cat(rows, 1))
    assert resumed.length == state.length == final.length == 6
    assert all(s.recurrent.dtype == state_dtype for s in final.layers if s.recurrent is not None)
    for a, b, c in zip(final.layers, resumed.layers, state.layers, strict=False):
        for f in fields(a):
            x, y, z = getattr(a, f.name), getattr(b, f.name), getattr(c, f.name)
            assert (x is None and y is None and z is None) or (
                torch.equal(x, y) and torch.equal(x, z)
            ), f.name
    after = [
        getattr(s, f.name)
        for s in prefix.layers
        for f in fields(s)
        if getattr(s, f.name) is not None
    ]
    assert all(torch.equal(a, b) for a, b in zip(snapshot, after, strict=False))


def test_reference_rejects_missing_history_and_negative_tokens():
    import pytest

    model = tiny_model()
    with ReferenceExecutionContext():
        with pytest.raises(ValueError, match="vocabulary"):
            model.forward(torch.tensor([[-1]]))
        _, state = model.forward(torch.tensor([[1, 2]]))
        broken = replace(
            state, layers=(replace(state.layers[0], ple_history=None), state.layers[1])
        )
        with pytest.raises(ValueError, match="ple_history"):
            model.forward(torch.tensor([[3]]), broken)
        broken = replace(
            state, layers=(state.layers[0], replace(state.layers[1], compressed_index_keys=None))
        )
        with pytest.raises(ValueError, match="compressed_index_keys"):
            model.forward(torch.tensor([[3]]), broken)


def test_reference_preserves_supplied_compressed_state_across_group_boundaries():
    model = tiny_model()
    with ReferenceExecutionContext():
        _, prefix = model.forward(torch.tensor([[1, 2]]))
        compressed = prefix.layers[1].compressed_index_keys
        assert compressed is not None
        original = compressed.clone()
        # A supplied state is an input, not a request to reconstruct it from
        # older raw keys. Its stored bits must survive both a partial and a
        # completed subsequent group.
        supplied = torch.full_like(compressed, 0.125)
        snapshot = supplied.clone()
        state = replace(
            prefix,
            layers=(prefix.layers[0], replace(prefix.layers[1], compressed_index_keys=supplied)),
        )
        for token in (3, 4):
            _, state = model.forward(torch.tensor([[token]]), state)
            cached = state.layers[1].compressed_index_keys
            assert cached is not None and torch.equal(cached[:, :1], snapshot)
    assert torch.equal(supplied, snapshot)
    assert torch.equal(prefix.layers[1].compressed_index_keys, original)
    cached = state.layers[1].compressed_index_keys
    assert cached is not None and cached.shape[1] == 2
