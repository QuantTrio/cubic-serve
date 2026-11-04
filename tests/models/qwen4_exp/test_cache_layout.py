"""Checkpoint-derived cache ownership, independent of packed weight format."""

from dataclasses import replace
from math import lcm

import pytest
import torch

from cubic_serve.contracts.execution import ParallelTopology
from cubic_serve.models.qwen4_exp import parse_qwen4_exp_config
from cubic_serve.models.qwen4_exp.model import LayerStateKind, qwen4_exp_cache_layout
from tests.models.qwen4_exp.test_config import _fixture


@pytest.mark.parametrize("tp,pp", [(1, 1), (2, 1), (4, 1), (4, 2), (8, 1)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("state_dtype", [None, torch.bfloat16, torch.float32])
def test_declared_rank_cache_matches_architecture_state_owners(tp, pp, dtype, state_dtype):
    config = parse_qwen4_exp_config(_fixture())
    config = replace(config, text=replace(config.text, eos_token_id=248044))
    topology = ParallelTopology.single_node(
        topology_id="cache.schema",
        device_indices=tuple(range(tp * pp)),
        tp_size=tp,
        pp_size=pp,
        ep_size=1,
    )
    plan = config.architecture_plan()
    per_stage = config.text.layer_count // pp
    page_size = lcm(16, config.text.qsa.compression_ratio * tp)
    for rank in range(tp * pp):
        layout, states = qwen4_exp_cache_layout(
            config, topology, rank, page_size=page_size, dtype=dtype,
            recurrent_state_dtype=state_dtype,
        )
        stage = topology.placements[rank].pipeline_stage
        owned = plan.decoder_layers[stage * per_stage : (stage + 1) * per_stage]
        expected = {
            f"model.language_model.layers.{layer.layer_index}.state.{kind.value}"
            for layer in owned
            for kind in layer.state_requirements
            if kind is not LayerStateKind.ATTENTION_KV
        }
        assert {s.state_id for s in states} | {s.field_id for s in layout.page_fields} == expected
        assert layout.num_kv_heads == max(1, config.text.full_attention.key_value_heads // tp)
        assert layout.dtype == dtype
        assert len(layout.page_fields) == len(layout.layer_ids)
        for state in states:
            if state.state_id.endswith(LayerStateKind.GDN_RECURRENT_MATRIX.value):
                assert state.dtype == (state_dtype or torch.float32)
                assert state.shape == (48 // tp, 128, 128)
            elif state.state_id.endswith(LayerStateKind.QSA_RAW_KEY_RING.value):
                assert state.shape == (4, 128 // tp)
            elif state.state_id.endswith(LayerStateKind.QSA_POSITION_RING.value):
                assert state.dtype == torch.int64
                assert state.shape == (4, 3)
            elif state.state_id.endswith(LayerStateKind.PLE_NGRAM_TOKEN_CONTEXT.value):
                assert state.dtype == torch.int64
                assert state.shape == (2,)
                assert state.initial_value == config.text.eos_token_id
            else:
                assert state.dtype == dtype
        for field in layout.page_fields:
            assert field.shape == (page_size // (4 * tp), 128)
            assert field.dtype == dtype
        peer = stage * tp
        peer_layout, peer_states = qwen4_exp_cache_layout(
            config, topology, peer, page_size=page_size, dtype=dtype,
            recurrent_state_dtype=state_dtype,
        )
        assert states == peer_states
        assert layout == peer_layout



def test_cache_rejects_incompatible_compression_page_boundary():
    config = parse_qwen4_exp_config(_fixture())
    config = replace(config, text=replace(config.text, eos_token_id=248044))
    topology = ParallelTopology.single_node(
        topology_id="cache.schema", device_indices=(0,), tp_size=1, pp_size=1, ep_size=1
    )
    with pytest.raises(ValueError, match="page"):
        qwen4_exp_cache_layout(config, topology, 0, page_size=6)
