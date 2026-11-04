"""Declarations that undo this model's GGUF exporter, only at weight loading.

No module/forward or alternative model is defined here. The declarations follow
llama.cpp 987498f4592a76897863cf53711dce38380c082b, conversion/qwen4exp.py
and conversion/qwen.py. Original quantization codes and expert order are kept.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

import numpy as np
import torch
from torch import Tensor

from cubic_serve.models.gguf import GGUFInventory
from cubic_serve.models.gguf_weights import GGUFMetadataTensor, GGUFPart, GGUFWeightView
from cubic_serve.models.parallel_weights import AxisSpan
from cubic_serve.models.quantization import QuantizationError
from cubic_serve.models.quantized_checkpoint import LogicalWeight
from cubic_serve.models.qwen4_exp.model import Qwen4ExpConfig

_LAYER_NAMES = {
    "linear_attn.A_log": "ssm_a",
    "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.dt_bias": "ssm_dt.bias",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.norm.weight": "ssm_norm.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "self_attn.indexer.q_layernorm.weight": "indexer.q_norm.weight",
    "self_attn.indexer.k_layernorm.weight": "indexer.k_norm.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
    "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
    "ple.conv1d.weight": "ple_conv1d.weight",
    "ple.key_proj.weight": "ple_key.weight",
    "ple.value_proj.weight": "ple_value.weight",
    "ple.norm_conv.weight": "ple_norm_conv.weight",
    "ple.norm_key.weight": "ple_norm_key.weight",
    "ple.norm_query.weight": "ple_norm_query.weight",
}
_HC_NAMES = {
    "input_mix_weight_down.weight": "down.weight",
    "input_mix_weight_up.weight": "up.weight",
    "hc_norm.weight": "norm.weight",
    "block_inject_weight.weight": "inject.weight",
}
_VISION_NAMES = {
    "attn.qkv": "attn_qkv", "attn.proj": "attn_out",
    "mlp.linear_fc1": "ffn_up", "mlp.linear_fc2": "ffn_down",
    "norm1": "ln1", "norm2": "ln2",
}


def _zero_centered(value: Tensor) -> Tensor:
    return value - 1


def _log_decay(value: Tensor) -> Tensor:
    if not bool(torch.isfinite(value).all()) or not bool((value < 0).all()):
        raise QuantizationError("GGUF GDN decay must be finite and strictly negative")
    return torch.log(-value)


def _head_order(key_heads: int, value_heads: int, width: int) -> tuple[int, ...]:
    if value_heads % key_heads:
        raise QuantizationError("GGUF GDN head order requires whole value groups")
    # Export maps [key, value-per-key, dim] -> [value-per-key, key, dim].
    # Gather the inverse into the unchanged model's key-major coordinates.
    per_key = value_heads // key_heads
    return tuple(((v * key_heads + k) * width + d)
                 for k in range(key_heads) for v in range(per_key) for d in range(width))


def gguf_import(
    config: Qwen4ExpConfig, logical_weights: tuple[LogicalWeight, ...], *,
    bundles: Mapping[str, GGUFInventory],
) -> tuple[GGUFWeightView, ...]:
    if not {"main", "vision"} <= bundles.keys() or bundles.keys() - {"main", "vision", "draft"}:
        raise QuantizationError(
            "this model's GGUF requires main and vision bundles; draft is optional")
    return declare_gguf_weights(config, logical_weights, main=bundles["main"],
                                vision=bundles["vision"], draft=bundles.get("draft"))


def declare_gguf_weights(
    config: Qwen4ExpConfig,
    logical_weights: tuple[LogicalWeight, ...],
    *,
    main: GGUFInventory,
    vision: GGUFInventory,
    draft: GGUFInventory | None = None,
) -> tuple[GGUFWeightView, ...]:
    """Bind the existing logical plan; configuration/NN execution are unchanged."""
    if main.files[0].field("general.architecture") != "qwen4exp":
        raise QuantizationError("GGUF architecture differs from the declared model package")
    if draft is not None and draft.files[0].field("general.architecture") != "qwen4exp":
        raise QuantizationError("GGUF draft architecture differs from its target")
    _validate_metadata(config, main, vision)
    gdn = config.text.gated_delta_net
    head_order = _head_order(gdn.key_heads, gdn.value_heads, 1)
    value_order = _head_order(gdn.key_heads, gdn.value_heads, gdn.value_head_dim)
    qk_width = 2 * gdn.key_heads * gdn.key_head_dim
    qkv_order = (*range(qk_width), *(qk_width + i for i in value_order))
    views = []
    for logical in logical_weights:
        name = logical.name
        source = main
        row_order: tuple[int, ...] = ()
        column_order: tuple[int, ...] = ()
        transform: Callable[[Tensor], Tensor] | None = None
        parts: tuple[GGUFPart, ...]
        if name.startswith("model.visual."):
            parts = _vision_parts(name.removeprefix("model.visual."), config, vision)
            join = 2 if name == "model.visual.patch_embed.proj.weight" else 0
            views.append(GGUFWeightView(logical, parts, join_axis=join))
            continue
        match = re.fullmatch(r"(model\.language_model|mtp)\.layers\.(\d+)\.(.+)", name)
        if match:
            model_prefix, index, member = match.groups()
            block = int(index)
            if model_prefix == "mtp":
                if draft is None:
                    raise QuantizationError("GGUF draft weights are required by the logical plan")
                source = draft
                block += config.text.layer_count
            prefix = f"blk.{block}."
            expert = re.fullmatch(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight", member)
            if expert:
                expert_id, projection = int(expert[1]), expert[2]
                record = source.tensor(prefix + f"ffn_{projection}_exps.weight")
                rows = record.shape[1]
                parts = (GGUFPart(source, record.name,
                                  AxisSpan(expert_id * rows, (expert_id + 1) * rows)),)
            elif member == "self_attn.indexer.index_qk_proj.weight":
                parts = tuple(GGUFPart(source, prefix + f"indexer.{kind}_proj.weight")
                              for kind in ("q", "k"))
            elif member.startswith("ple.ple_embedding."):
                views.append(_ple_view(logical, main, config))
                continue
            else:
                physical = _LAYER_NAMES.get(member)
                hc = re.fullmatch(r"(attn|mlp)_hyper_connection\.(.+)", member)
                if hc:
                    physical = "hc_" + ("attn" if hc[1] == "attn" else "ffn") + "_"
                    physical += _HC_NAMES[hc[2]]
                if physical is None:
                    raise QuantizationError(f"model package has no GGUF weight declaration: {name}")
                record = source.tensor(prefix + physical)
                parts = (GGUFPart(source, record.name,
                                  reshape=logical.shape if record.type_id in (0, 1, 30) else ()),)
                if member.startswith("linear_attn."):
                    if member in ("linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight"):
                        row_order = qkv_order
                    elif member == "linear_attn.in_proj_z.weight":
                        row_order = value_order
                    elif member == "linear_attn.out_proj.weight":
                        column_order = value_order
                    elif member in ("linear_attn.A_log", "linear_attn.dt_bias",
                                    "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"):
                        row_order = head_order
                    if member == "linear_attn.A_log":
                        transform = _log_decay
                if (member.endswith("norm.weight") and member != "linear_attn.norm.weight"):
                    transform = _zero_centered
        else:
            parts, transform = _root_parts(logical, main, draft)
        view = GGUFWeightView(logical, parts, row_order=row_order,
                              column_order=column_order, transform=transform)
        views.append(view)
    return tuple(views)


def _validate_metadata(config: Qwen4ExpConfig, main: GGUFInventory, vision: GGUFInventory) -> None:
    """Shapes alone cannot prove head order, RoPE, QSA or PLE semantics."""
    t = config.text
    g, a, p, q = t.gated_delta_net, t.full_attention, t.ple, t.qsa
    fields = {
        "block_count": t.layer_count, "context_length": t.max_position_embeddings,
        "embedding_length": t.hidden_size,
        "attention.head_count": a.query_heads, "attention.head_count_kv": a.key_value_heads,
        "attention.key_length": a.head_dim, "attention.value_length": a.head_dim,
        "attention.layer_norm_rms_epsilon": float(np.float32(t.rms_norm_epsilon)),
        "rope.dimension_sections": (*t.mrope.section, 0),
        "rope.freq_base": float(np.float32(t.mrope.rope_theta)),
        "rope.dimension_count": int(a.head_dim * t.mrope.partial_rotary_factor),
        "expert_count": t.moe.expert_count, "expert_used_count": t.moe.experts_per_token,
        "expert_feed_forward_length": t.moe.expert_intermediate_size,
        "expert_shared_feed_forward_length": t.moe.shared_expert_intermediate_size,
        "ssm.conv_kernel": g.convolution_kernel_size, "ssm.state_size": g.key_head_dim,
        "ssm.group_count": g.key_heads, "ssm.time_step_rank": g.value_heads,
        "ssm.inner_size": g.value_heads * g.value_head_dim,
        "full_attention_interval": t.full_attention_interval,
        "hyper_connection.count": t.hyperconnection.stream_count,
        "hyper_connection.low_rank": t.hyperconnection.low_rank,
        "attention.indexer.head_count": q.query_heads,
        "attention.indexer.key_length": q.head_dim, "attention.indexer.top_k": q.token_budget,
        "attention.compress_ratios": tuple(q.compression_ratio if str(kind) == "full-attention"
                                            else 0 for kind in t.layer_schedule),
    }
    if p.layer_ids:
        fields.update({
            "ple.layers": tuple(i - 1 for i in p.layer_ids), "ple.ngram_size": p.ngram_size,
            "ple.heads_per_ngram": p.heads_per_ngram, "ple.conv_kernel": p.convolution_kernel_size,
            "ple.eos_token_id": t.eos_token_id,
            "ple.image_token_id": config.modality_tokens.image,
            "embedding_length_per_layer_input": p.embedding_dim // (
                (p.ngram_size - 1) * p.heads_per_ngram),
        })
    for field, value in fields.items():
        if main.files[0].field("qwen4exp." + field) != value:
            raise QuantizationError(f"GGUF metadata differs from the model declaration: {field}")
    v = config.vision
    for field, value in {
        "embedding_length": v.hidden_size, "feed_forward_length": v.intermediate_size,
        "block_count": v.layer_count, "attention.head_count": v.head_count,
        "patch_size": v.patch_size, "spatial_merge_size": v.spatial_merge_size,
    }.items():
        if vision.files[0].field("clip.vision." + field) != value:
            raise QuantizationError(
                f"GGUF vision metadata differs from its model declaration: {field}")


def _ple_view(logical: LogicalWeight, main: GGUFInventory, config: Qwen4ExpConfig
              ) -> GGUFWeightView:
    member = logical.name.split(".ple_embedding.", 1)[1]
    metadata = {"layer_multipliers": "layer_multipliers",
                "ngram_heads_offsets": "head_offsets",
                "ngram_heads_vocab_sizes": "head_vocab_sizes"}
    if member in metadata:
        return GGUFWeightView(logical, metadata=GGUFMetadataTensor(
            main, "qwen4exp.ple." + metadata[member]))
    shard = re.fullmatch(r"ngram_embedding\.shard_(\d+)\.weight", member)
    if shard is None:
        raise QuantizationError("GGUF PLE declaration has an unknown logical member")
    table = main.tensor("per_layer_token_embd.weight")
    count = config.text.ple.split_parts
    if table.shape[0] % count or table.shape[0] // count != logical.shape[0]:
        raise QuantizationError("GGUF PLE rows differ from the existing model's shard contract")
    rows = table.shape[0] // count
    return GGUFWeightView(logical, (GGUFPart(main, table.name,
        AxisSpan(int(shard[1]) * rows, (int(shard[1]) + 1) * rows)),))


def _root_parts(
    logical: LogicalWeight, main: GGUFInventory, draft: GGUFInventory | None,
) -> tuple[tuple[GGUFPart, ...], Callable[[Tensor], Tensor] | None]:
    name = logical.name
    physical = {"model.language_model.embed_tokens.weight": "token_embd.weight",
                "lm_head.weight": "output.weight"}.get(name)
    source = main
    if physical is not None:
        return (GGUFPart(source, physical),), None
    if name.startswith("mtp."):
        if draft is None:
            raise QuantizationError("GGUF draft bundle is absent")
        source = draft
        blocks = [int(t.name.split(".")[1]) for t in draft.tensors if t.name.startswith("blk.")]
        prefix = f"blk.{min(blocks)}.nextn."
        if name in ("mtp.fc_embedding.weight", "mtp.fc_hidden.weight"):
            width = logical.shape[1]
            start = 0 if name == "mtp.fc_embedding.weight" else width
            return (GGUFPart(source, prefix + "eh_proj.weight",
                             columns=AxisSpan(start, start + width)),), None
        norm = {"mtp.pre_fc_norm_embedding.weight": "enorm.weight",
                "mtp.pre_fc_norm_hidden.weight": "hnorm.weight"}.get(name)
        if norm:
            return (GGUFPart(source, prefix + norm, reshape=logical.shape),), _zero_centered
        member = name.removeprefix("mtp.hyper_connection_mixer.")
        physical = prefix + "hc_head_" + _HC_NAMES[member]
    else:
        member = name.removeprefix("model.language_model.hyper_connection_mixer.")
        physical = "output_hc_" + _HC_NAMES[member]
    record = source.tensor(physical)
    transform = _zero_centered if name.endswith("hc_norm.weight") else None
    return (GGUFPart(source, physical,
                     reshape=logical.shape if record.type_id in (0, 1, 30) else ()),), transform


def _vision_parts(member: str, config: Qwen4ExpConfig, source: GGUFInventory
                  ) -> tuple[GGUFPart, ...]:
    if member == "patch_embed.proj.weight":
        v = config.vision
        shape = (v.hidden_size, v.input_channels, 1, v.patch_size, v.patch_size)
        return tuple(GGUFPart(source, "v.patch_embd.weight" + (f".{i}" if i else ""),
                              reshape=shape) for i in range(v.temporal_patch_size))
    exact = {"patch_embed.proj.bias": "v.patch_embd.bias",
             "pos_embed.weight": "v.position_embd.weight"}
    physical = exact.get(member)
    block = re.fullmatch(r"blocks\.(\d+)\.(.+)\.(weight|bias)", member)
    if block:
        physical = f"v.blk.{block[1]}.{_VISION_NAMES[block[2]]}.{block[3]}"
    elif member.startswith("merger."):
        match = re.fullmatch(r"merger\.(norm|linear_fc1|linear_fc2)\.(weight|bias)", member)
        if match:
            physical = {"norm": "v.post_ln", "linear_fc1": "mm.0", "linear_fc2": "mm.2"}[match[1]]
            physical += "." + match[2]
    if physical is None:
        raise QuantizationError("model package has no GGUF vision weight declaration")
    record = source.tensor(physical)
    return (GGUFPart(source, physical, reshape=record.shape),)
