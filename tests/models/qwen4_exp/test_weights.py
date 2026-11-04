from __future__ import annotations

import unittest
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest

from cubic_serve.models.qwen4_exp import parse_qwen4_exp_config, parse_qwen4_exp_config_json
from cubic_serve.models.qwen4_exp.weights import (
    _expected_headers,
    _logical_quantized_weights,
    logical_quantized_weights,
    plan_qwen4_exp_weight_headers,
)
from cubic_serve.models.weight_schema import TensorHeader, WeightSchemaCoverage, WeightSchemaError
from tests.models._checkpoint_headers import read_index_headers
from tests.models.qwen4_exp.test_config import _fixture

_LOCAL_ROOT = Path("/pubdata/llm_models/Qwen/Qwen3.8-Flash-Next")


def test_ple_storage_is_model_owned_and_does_not_stale_in_config_cache(monkeypatch):
    config = parse_qwen4_exp_config(_fixture())
    monkeypatch.setenv("CUBIC_QWEN4_PLE_STORAGE", "gpu")
    gpu = logical_quantized_weights(config)
    monkeypatch.delenv("CUBIC_QWEN4_PLE_STORAGE", raising=False)
    cpu = logical_quantized_weights(config)
    monkeypatch.setenv("CUBIC_QWEN4_PLE_STORAGE", "cpu")
    assert logical_quantized_weights(config) is cpu
    tables = [w for w in cpu if w.storage == "pinned_host"]
    assert tables and all(".ngram_embedding.shard_" in w.name for w in tables)
    assert tuple(replace(w, storage="device") for w in cpu) == gpu
    monkeypatch.setenv("CUBIC_QWEN4_PLE_STORAGE", "gpu")
    assert logical_quantized_weights(config) == gpu
    monkeypatch.setenv("CUBIC_QWEN4_PLE_STORAGE", "disk")
    with pytest.raises(ValueError, match="CUBIC_QWEN4_PLE_STORAGE"):
        logical_quantized_weights(config)


def test_mtp_reuses_declared_tp_columns_and_checkpoint_expert_ownership():
    from cubic_serve.contracts.execution import ParallelTopology
    from cubic_serve.models.qwen4_exp.weights import quantized_owner_slices

    config = parse_qwen4_exp_config(_fixture())
    logical = logical_quantized_weights(config)
    for tp, ep in ((1, 1), (2, 1), (2, 2), (4, 1), (4, 2), (4, 4)):
        topology = ParallelTopology.single_node(
            topology_id=f"test.mtp.tp{tp}.ep{ep}", device_indices=tuple(range(tp)),
            tp_size=tp, pp_size=1, ep_size=ep,
        )
        ranks = [quantized_owner_slices(config, topology, rank,
                                       logical_weights=logical, include_mtp=True)
                 for rank in range(tp)]
        for name in ("mtp.fc_embedding.weight", "mtp.fc_hidden.weight"):
            owners = [next(b for b in bindings if b.tensor_name == name) for bindings in ranks]
            spans = [s for owner in owners for s in owner.spans]
            assert all(owner.shard_axis == 0 for owner in owners)
            assert spans[0].start == 0 and spans[-1].stop == config.text.hidden_size
            assert all(a.stop == b.start for a, b in pairwise(spans))
        for rank, bindings in enumerate(ranks):
            owner, _ = topology.expert_coordinates(rank)
            first = owner * config.text.moe.expert_count // ep
            stop = (owner + 1) * config.text.moe.expert_count // ep
            experts = {int(b.tensor_name.split(".experts.")[1].split(".")[0])
                       for b in bindings if b.tensor_name.startswith("mtp.layers.")
                       and ".experts." in b.tensor_name}
            assert experts == set(range(first, stop))
            norms = [b for b in bindings if b.tensor_name in (
                "mtp.pre_fc_norm_hidden.weight", "mtp.pre_fc_norm_embedding.weight")]
            assert len(norms) == 2
            assert all(b.local_shape == b.global_shape for b in norms)


def _headers() -> list[TensorHeader]:
    config = parse_qwen4_exp_config(_fixture())
    return [
        TensorHeader(name, contract.dtype, contract.shape)
        for name, contract in _expected_headers(config).items()
    ]


class Qwen4ExpWeightSchemaTests(unittest.TestCase):
    def test_logical_inventory_cache_is_bound_to_complete_config(self) -> None:
        config = parse_qwen4_exp_config(_fixture())
        first = logical_quantized_weights(config)
        same = logical_quantized_weights(parse_qwen4_exp_config(_fixture()))
        self.assertIs(first, same)
        self.assertEqual(first, _logical_quantized_weights.__wrapped__(config))
        changed_config = replace(
            config, text=replace(config.text, moe=replace(config.text.moe, expert_count=256))
        )
        changed = logical_quantized_weights(changed_config)
        self.assertNotEqual(first, changed)
        self.assertEqual(changed, _logical_quantized_weights.__wrapped__(changed_config))

    def test_closed_partitions_and_three_i64_ple_metadata_tensors(self) -> None:
        plan = plan_qwen4_exp_weight_headers(parse_qwen4_exp_config(_fixture()), _headers())

        self.assertEqual(plan.coverage, WeightSchemaCoverage.EXACT_CHECKPOINT_HEADERS)
        self.assertEqual(
            (
                len(plan.checkpoint_names),
                len(plan.base_language_names),
                len(plan.vision_names),
                len(plan.lm_head_names),
                len(plan.mtp_extension_names),
            ),
            (1658, 1293, 333, 1, 31),
        )
        self.assertFalse(plan.fp32_stability_names)
        self.assertEqual(len(plan.int64_metadata_names), 3)
        self.assertTrue(all(".ple.ple_embedding." in name for name in plan.int64_metadata_names))

    def test_ple_shard_gap_i64_dtype_and_wrong_layer_fail_closed(self) -> None:
        config = parse_qwen4_exp_config(_fixture())
        headers = _headers()
        shard = next(
            header
            for header in headers
            if header.name.endswith("ple_embedding.ngram_embedding.shard_127.weight")
        )
        metadata = next(header for header in headers if header.dtype == "I64")
        for candidate in (
            [header for header in headers if header is not shard],
            [
                replace(metadata, dtype="BF16") if header is metadata else header
                for header in headers
            ],
            [
                *headers,
                TensorHeader(
                    "model.language_model.layers.48.self_attn.q_proj.weight", "BF16", (1,)
                ),
            ],
        ):
            with self.subTest(size=len(candidate)), self.assertRaises(WeightSchemaError):
                plan_qwen4_exp_weight_headers(config, candidate)

    @unittest.skipUnless(
        (_LOCAL_ROOT / "model.safetensors.index.json").is_file(),
        "local qwen4_exp checkpoint headers are unavailable",
    )
    def test_real_flash_headers_without_tensor_payload_reads(self) -> None:
        config = parse_qwen4_exp_config_json((_LOCAL_ROOT / "config.json").read_bytes())
        plan = plan_qwen4_exp_weight_headers(config, read_index_headers(_LOCAL_ROOT))
        self.assertEqual(len(plan.checkpoint_names), 1658)
        self.assertEqual(len(plan.int64_metadata_names), 3)


if __name__ == "__main__":
    unittest.main()


def test_general_tp_ownership_closes_ple_and_replicated_kv_heads():
    from cubic_serve.contracts.execution import ParallelTopology
    from cubic_serve.models.qwen4_exp.weights import (
        logical_quantized_weights,
        quantized_owner_slices,
    )

    config = parse_qwen4_exp_config(_fixture())
    logical = logical_quantized_weights(config)
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    tables = sorted(
        (w for w in logical if w.name.startswith(prefix + ".shard_")),
        key=lambda w: int(w.name.split(".shard_")[1].split(".")[0]),
    )
    for tp in (1, 2, 4, 8):
        topology = ParallelTopology.single_node(
            topology_id=f"test.tp{tp}",
            device_indices=tuple(range(tp)),
            tp_size=tp,
            pp_size=1,
            ep_size=1,
        )
        ranks = [
            quantized_owner_slices(config, topology, r, logical_weights=logical) for r in range(tp)
        ]
        for table in tables:
            spans = sorted(
                (s.start, s.stop)
                for bindings in ranks
                for binding in bindings
                if binding.tensor_name == table.name
                for s in binding.spans
            )
            assert spans[0][0] == 0 and spans[-1][1] == table.shape[0]
            assert all(left[1] == right[0] for left, right in pairwise(spans))
        from cubic_serve.models.parallel_weights import IndexerTP
        index = config.text.qsa
        name = "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight"
        owners = [next(b for b in bindings if b.tensor_name == name) for bindings in ranks]
        spans = sorted((s.start, s.stop) for owner in owners for s in owner.spans)
        assert spans[0][0] == 0 and spans[-1][1] == (index.query_heads + 1) * index.head_dim
        assert all(a[1] == b[0] for a, b in pairwise(spans))
        for rank, owner in enumerate(owners):
            plan = IndexerTP(index.query_heads, index.head_dim, index.compression_ratio, tp, rank)
            assert tuple((s.start, s.stop) for s in owner.spans) == plan.projection_ranges
        for bindings in ranks:
            assert all(b.local_shape == b.global_shape for b in bindings
                       if b.tensor_name.endswith((
                           ".input_mix_weight_down.weight", ".input_mix_weight_up.weight")))
        for rank, bindings in enumerate(ranks):
            kv = next(
                b
                for b in bindings
                if b.tensor_name == "model.language_model.layers.3.self_attn.k_proj.weight"
            )
            assert kv.local_shape[0] == 512 // min(tp, 2)
            if tp > 2:
                assert kv.spans[0].start == rank // (tp // 2) * 256
            assert all(not b.tensor_name.startswith("mtp.") for b in bindings)


def test_expert_ownership_and_matrix_shards_are_independent_axes():
    from cubic_serve.contracts.execution import ParallelTopology
    from cubic_serve.models.qwen4_exp.weights import (
        logical_quantized_weights,
        quantized_owner_slices,
    )

    config = parse_qwen4_exp_config(_fixture())
    logical = logical_quantized_weights(config)
    experts = [w for w in logical if w.partition == "language" and ".mlp.experts." in w.name]
    for tp in (1, 2, 4, 8):
        for ep in (size for size in (1, 2, 4, 8) if tp % size == 0):
            topology = ParallelTopology.single_node(
                topology_id=f"test.tp{tp}.ep{ep}",
                device_indices=tuple(range(tp)),
                tp_size=tp,
                pp_size=1,
                ep_size=ep,
            )
            ranks = [
                {
                    b.tensor_name: b
                    for b in quantized_owner_slices(
                        config,
                        topology,
                        rank,
                        logical_weights=logical,
                    )
                }
                for rank in range(tp)
            ]
            for weight in experts:
                eid = int(weight.name.split(".experts.")[1].split(".")[0])
                owner = eid // (config.text.moe.expert_count // ep)
                members = [rank for rank, bindings in enumerate(ranks) if weight.name in bindings]
                assert len(members) == tp // ep
                assert all(topology.expert_coordinates(rank)[0] == owner for rank in members)
                axis = 1 if weight.name.endswith("down_proj.weight") else 0
                spans = [span for rank in members for span in ranks[rank][weight.name].spans]
                assert spans[0].start == 0 and spans[-1].stop == weight.shape[axis]
                assert all(left.stop == right.start for left, right in pairwise(spans))
                assert all(ranks[rank][weight.name].shard_axis == axis for rank in members)
            for bindings in ranks:
                for name, binding in bindings.items():
                    if ".mlp.shared_expert." in name:
                        axis = binding.shard_axis
                        assert binding.local_shape[axis] * tp == binding.global_shape[axis]


def test_weight_mapping_follows_config_geometry_without_known_repository():
    for layers, hidden, experts, mtp in ((8, 768, 16, 0), (12, 1536, 32, 1)):
        raw = _fixture()
        text = raw["text_config"]
        text["num_hidden_layers"] = layers
        text["linear_num_key_heads"] = 8
        text["linear_num_value_heads"] = 16
        text["linear_key_head_dim"] = 64
        text["linear_value_head_dim"] = 32
        text["linear_conv_kernel_dim"] = 3
        text["head_dim"] = 128
        text["rope_parameters"]["partial_rotary_factor"] = 0.5
        if "partial_rotary_factor" in text:
            text["partial_rotary_factor"] = 0.5
        text["hidden_size"] = hidden
        text["vocab_size"] += 128
        text["layer_types"] = [
            "full_attention"
            if (i + 1) % text["full_attention_interval"] == 0
            else "linear_attention"
            for i in range(layers)
        ]
        text["mtp_num_hidden_layers"] = mtp
        vision = raw["vision_config"]
        vision["depth"] = 2
        vision["out_hidden_size"] = hidden
        text["num_experts"] = experts
        text["ple_layer_ids"] = [1, 5]
        text["mtp"]["num_hidden_layers"] = mtp
        text["mtp"]["layer_types"] = ["full_attention"] * mtp
        config = parse_qwen4_exp_config(raw)
        expected = _expected_headers(config)
        headers = [TensorHeader(name, item.dtype, item.shape) for name, item in expected.items()]
        plan = plan_qwen4_exp_weight_headers(config, headers)
        embedding = next(h for h in headers if h.name == "model.language_model.embed_tokens.weight")
        assert embedding.shape == (text["vocab_size"], hidden)
        assert bool(plan.mtp_extension_names) == bool(mtp)
        assert all(
            int(n.split("layers.")[1].split(".")[0]) < layers
            for n in plan.base_language_names
            if ".layers." in n
        )
        broken = [
            TensorHeader(h.name, h.dtype, (h.shape[0] + 1, *h.shape[1:]))
            if h.name == embedding.name
            else h
            for h in headers
        ]
        with unittest.TestCase().assertRaises(WeightSchemaError):
            plan_qwen4_exp_weight_headers(config, broken)
