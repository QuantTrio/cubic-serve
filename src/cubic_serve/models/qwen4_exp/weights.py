"""Closed checkpoint-header partition for model_type ``qwen4_exp``."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from cubic_serve.contracts.execution import ParallelTopology
    from cubic_serve.models.parallel_weights import LogicalWeightSlice
    from cubic_serve.models.quantized_checkpoint import LogicalWeight

from cubic_serve.models.checkpoint import CheckpointInventory
from cubic_serve.models.qwen4_exp.model import AttentionKind, Qwen4ExpConfig
from cubic_serve.models.weight_schema import (
    CheckpointWeightPlan,
    TensorHeader,
    WeightArchitectureIdentity,
    WeightNodeBinding,
    WeightSchemaCoverage,
    WeightSchemaError,
    _issue_weight_plan,
    materialize_headers,
    tensor_headers_from_inventory,
)

Qwen4ExpWeightPlan = CheckpointWeightPlan

__all__ = (
    "Qwen4ExpWeightPlan",
    "TensorHeader",
    "WeightSchemaError",
    "plan_qwen4_exp_checkpoint_weights",
    "plan_qwen4_exp_weight_headers",
)


@dataclass(frozen=True, slots=True)
class _ExpectedHeader:
    dtype: str
    shape: tuple[int, ...]
    node_id: str
    partition: str


class _AddHeader(Protocol):
    def __call__(
        self,
        name: str,
        shape: tuple[int, ...],
        node_id: str,
        partition: str,
        dtype: str = "BF16",
    ) -> None: ...


def plan_qwen4_exp_checkpoint_weights(
    config: Qwen4ExpConfig, inventory: CheckpointInventory
) -> Qwen4ExpWeightPlan:
    """Plan an already-attested inventory without reading tensor payloads."""

    return _plan(
        config,
        tensor_headers_from_inventory(inventory),
        inventory=inventory,
    )


def plan_qwen4_exp_weight_headers(
    config: Qwen4ExpConfig, headers: Iterable[TensorHeader]
) -> Qwen4ExpWeightPlan:
    """Pure header-record API for CI and pre-attestation checks."""

    return _plan(config, materialize_headers(headers), inventory=None)


def _plan(
    config: Qwen4ExpConfig,
    headers: tuple[TensorHeader, ...],
    *,
    inventory: CheckpointInventory | None,
) -> Qwen4ExpWeightPlan:
    if config.identity.model_type != "qwen4_exp":
        raise WeightSchemaError("qwen4_exp weight schema received another model_type")
    expected = _expected_headers(config)
    actual = {header.name: header for header in headers}
    missing = sorted(set(expected) - set(actual))
    unknown = sorted(set(actual) - set(expected))
    if missing or unknown:
        raise WeightSchemaError(
            f"qwen4_exp checkpoint tensor set is not closed: missing={missing}, unknown={unknown}"
        )
    for name, contract in expected.items():
        header = actual[name]
        if header.dtype != contract.dtype:
            raise WeightSchemaError(
                f"qwen4_exp tensor {name!r} dtype mismatch: {header.dtype} != {contract.dtype}"
            )
        if header.shape != contract.shape:
            raise WeightSchemaError(
                f"qwen4_exp tensor {name!r} shape mismatch: {header.shape} != {contract.shape}"
            )

    partitions = {
        partition: tuple(
            sorted(name for name, item in expected.items() if item.partition == partition)
        )
        for partition in ("language", "vision", "lm_head", "mtp")
    }
    int64 = tuple(sorted(name for name, item in expected.items() if item.dtype == "I64"))
    by_node: dict[str, list[str]] = defaultdict(list)
    for name, contract in expected.items():
        by_node[contract.node_id].append(name)
    bindings = tuple(
        WeightNodeBinding(node_id=node, tensor_names=tuple(sorted(names)))
        for node, names in sorted(by_node.items())
    )
    plan = _issue_weight_plan(
        model_type="qwen4_exp",
        architecture_identity=WeightArchitectureIdentity(
            release_family=config.identity.release_family,
            release_profile=config.identity.release_profile,
            model_type=config.identity.model_type,
            graph_family=config.identity.graph_family,
            architecture=config.identity.architecture,
            plan_status=config.architecture_plan().status.value,
        ),
        coverage=WeightSchemaCoverage.EXACT_CHECKPOINT_HEADERS,
        headers=headers,
        inventory=inventory,
        base_language_names=partitions["language"],
        vision_names=partitions["vision"],
        lm_head_names=partitions["lm_head"],
        mtp_extension_names=partitions["mtp"],
        fp32_stability_names=(),
        int64_metadata_names=int64,
        allowed_node_ids=_architecture_node_ids(config),
        node_bindings=bindings,
    )
    return plan


def _architecture_node_ids(config: Qwen4ExpConfig) -> tuple[str, ...]:
    plan = config.architecture_plan()
    return tuple(
        sorted(
            (
                plan.vision_patch_embedding_node_id,
                *(layer.node_id for layer in plan.vision_layers),
                plan.vision_merger_node_id,
                plan.token_embedding_node_id,
                *(layer.node_id for layer in plan.decoder_layers),
                plan.final_mixer_node_id,
                plan.lm_head_node_id,
                *((plan.mtp_root_node_id,) if plan.mtp_layers else ()),
                *(layer.node_id for layer in plan.mtp_layers),
            )
        )
    )


def _expected_headers(config: Qwen4ExpConfig) -> dict[str, _ExpectedHeader]:
    text = config.text
    plan = config.architecture_plan()
    expected: dict[str, _ExpectedHeader] = {}

    def add(
        name: str,
        shape: tuple[int, ...],
        node_id: str,
        partition: str,
        dtype: str = "BF16",
    ) -> None:
        if name in expected:
            raise WeightSchemaError(f"internal qwen4_exp schema duplicates {name!r}")
        expected[name] = _ExpectedHeader(dtype, shape, node_id, partition)

    add_header: _AddHeader = add
    hidden = text.hidden_size
    vocabulary = text.vocabulary_size
    add(
        "model.language_model.embed_tokens.weight",
        (vocabulary, hidden),
        plan.token_embedding_node_id,
        "language",
    )
    _add_hyperconnection(
        add_header,
        prefix="model.language_model.hyper_connection_mixer",
        hidden=hidden,
        streams=text.hyperconnection.stream_count,
        low_rank=text.hyperconnection.low_rank,
        node=plan.final_mixer_node_id,
        partition="language",
        inject=False,
    )
    add("lm_head.weight", (vocabulary, hidden), plan.lm_head_node_id, "lm_head")

    for layer in plan.decoder_layers:
        prefix = f"model.language_model.layers.{layer.layer_index}"
        _add_hyperconnection_pair(
            add_header,
            prefix=prefix,
            hidden=hidden,
            streams=text.hyperconnection.stream_count,
            low_rank=text.hyperconnection.low_rank,
            node=layer.node_id,
            partition="language",
        )
        _add_packed_moe(
            add_header,
            prefix=f"{prefix}.mlp",
            config=config,
            node=layer.node_id,
            partition="language",
        )
        if layer.attention is AttentionKind.GATED_DELTA_NET:
            _add_gdn(
                add_header,
                prefix=f"{prefix}.linear_attn",
                config=config,
                node=layer.node_id,
                partition="language",
            )
        else:
            _add_qsa_attention(
                add_header,
                prefix=f"{prefix}.self_attn",
                config=config,
                node=layer.node_id,
                partition="language",
            )
        if layer.uses_ple:
            _add_ple(
                add_header,
                prefix=f"{prefix}.ple",
                config=config,
                node=layer.node_id,
            )

    _add_vision(add_header, config)
    if plan.mtp_layers:
        add("mtp.fc_embedding.weight", (hidden, hidden), plan.mtp_root_node_id, "mtp")
        add("mtp.fc_hidden.weight", (hidden, hidden), plan.mtp_root_node_id, "mtp")
        _add_hyperconnection(
            add_header,
            prefix="mtp.hyper_connection_mixer",
            hidden=hidden,
            streams=text.hyperconnection.stream_count,
            low_rank=text.hyperconnection.low_rank,
            node=plan.mtp_root_node_id,
            partition="mtp",
            inject=False,
        )
        add("mtp.pre_fc_norm_embedding.weight", (hidden,), plan.mtp_root_node_id, "mtp")
        add(
            "mtp.pre_fc_norm_hidden.weight",
            (hidden * text.hyperconnection.stream_count,),
            plan.mtp_root_node_id,
            "mtp",
        )
    for mtp_layer in plan.mtp_layers:
        prefix = f"mtp.layers.{mtp_layer.layer_index}"
        _add_hyperconnection_pair(
            add_header,
            prefix=prefix,
            hidden=hidden,
            streams=text.hyperconnection.stream_count,
            low_rank=text.hyperconnection.low_rank,
            node=mtp_layer.node_id,
            partition="mtp",
        )
        _add_packed_moe(
            add_header,
            prefix=f"{prefix}.mlp",
            config=config,
            node=mtp_layer.node_id,
            partition="mtp",
        )
        if mtp_layer.attention is not AttentionKind.FULL_ATTENTION:
            raise WeightSchemaError("initial qwen4_exp MTP schema requires QSA full attention")
        _add_qsa_attention(
            add_header,
            prefix=f"{prefix}.self_attn",
            config=config,
            node=mtp_layer.node_id,
            partition="mtp",
        )
    return expected


def _add_hyperconnection_pair(
    add: _AddHeader,
    *,
    prefix: str,
    hidden: int,
    streams: int,
    low_rank: int,
    node: str,
    partition: str,
) -> None:
    for name in ("attn_hyper_connection", "mlp_hyper_connection"):
        _add_hyperconnection(
            add,
            prefix=f"{prefix}.{name}",
            hidden=hidden,
            streams=streams,
            low_rank=low_rank,
            node=node,
            partition=partition,
            inject=True,
        )


def _add_hyperconnection(
    add: _AddHeader,
    *,
    prefix: str,
    hidden: int,
    streams: int,
    low_rank: int,
    node: str,
    partition: str,
    inject: bool,
) -> None:
    stream_hidden = hidden * streams
    if inject:
        add(
            f"{prefix}.block_inject_weight.weight",
            (streams, stream_hidden),
            node,
            partition,
        )
    add(f"{prefix}.hc_norm.weight", (stream_hidden,), node, partition)
    add(
        f"{prefix}.input_mix_weight_down.weight",
        (low_rank, stream_hidden),
        node,
        partition,
    )
    add(
        f"{prefix}.input_mix_weight_up.weight",
        (stream_hidden, low_rank),
        node,
        partition,
    )


def _add_packed_moe(
    add: _AddHeader,
    *,
    prefix: str,
    config: Qwen4ExpConfig,
    node: str,
    partition: str,
) -> None:
    hidden = config.text.hidden_size
    moe = config.text.moe
    add(
        f"{prefix}.experts.down_proj",
        (moe.expert_count, hidden, moe.expert_intermediate_size),
        node,
        partition,
    )
    add(
        f"{prefix}.experts.gate_up_proj",
        (moe.expert_count, moe.expert_intermediate_size * 2, hidden),
        node,
        partition,
    )
    add(f"{prefix}.gate.weight", (moe.expert_count, hidden), node, partition)
    add(
        f"{prefix}.shared_expert.down_proj.weight",
        (hidden, moe.shared_expert_intermediate_size),
        node,
        partition,
    )
    for projection in ("gate_proj", "up_proj"):
        add(
            f"{prefix}.shared_expert.{projection}.weight",
            (moe.shared_expert_intermediate_size, hidden),
            node,
            partition,
        )
    add(f"{prefix}.shared_expert_gate.weight", (1, hidden), node, partition)


def _add_gdn(
    add: _AddHeader,
    *,
    prefix: str,
    config: Qwen4ExpConfig,
    node: str,
    partition: str,
) -> None:
    hidden = config.text.hidden_size
    gdn = config.text.gated_delta_net
    key_dim = gdn.key_heads * gdn.key_head_dim
    value_dim = gdn.value_heads * gdn.value_head_dim
    convolution_dim = key_dim * 2 + value_dim
    add(f"{prefix}.A_log", (gdn.value_heads,), node, partition)
    add(
        f"{prefix}.conv1d.weight",
        (convolution_dim, 1, gdn.convolution_kernel_size),
        node,
        partition,
    )
    add(f"{prefix}.dt_bias", (gdn.value_heads,), node, partition)
    add(f"{prefix}.in_proj_a.weight", (gdn.value_heads, hidden), node, partition)
    add(f"{prefix}.in_proj_b.weight", (gdn.value_heads, hidden), node, partition)
    add(f"{prefix}.in_proj_qkv.weight", (convolution_dim, hidden), node, partition)
    add(f"{prefix}.in_proj_z.weight", (value_dim, hidden), node, partition)
    add(f"{prefix}.norm.weight", (gdn.value_head_dim,), node, partition)
    add(f"{prefix}.out_proj.weight", (hidden, value_dim), node, partition)


def _add_qsa_attention(
    add: _AddHeader,
    *,
    prefix: str,
    config: Qwen4ExpConfig,
    node: str,
    partition: str,
) -> None:
    hidden = config.text.hidden_size
    attention = config.text.full_attention
    qsa = config.text.qsa
    q_dim = attention.query_heads * attention.head_dim
    q_projection_dim = q_dim * (2 if attention.output_gate else 1)
    kv_dim = attention.key_value_heads * attention.head_dim
    add(f"{prefix}.q_proj.weight", (q_projection_dim, hidden), node, partition)
    add(f"{prefix}.k_proj.weight", (kv_dim, hidden), node, partition)
    add(f"{prefix}.v_proj.weight", (kv_dim, hidden), node, partition)
    add(f"{prefix}.o_proj.weight", (hidden, q_dim), node, partition)
    add(f"{prefix}.q_norm.weight", (attention.head_dim,), node, partition)
    add(f"{prefix}.k_norm.weight", (attention.head_dim,), node, partition)
    index_projection = (qsa.query_heads + qsa.key_value_heads) * qsa.head_dim
    add(
        f"{prefix}.indexer.index_qk_proj.weight",
        (index_projection, hidden),
        node,
        partition,
    )
    add(f"{prefix}.indexer.q_layernorm.weight", (qsa.head_dim,), node, partition)
    add(f"{prefix}.indexer.k_layernorm.weight", (qsa.head_dim,), node, partition)


def _add_ple(
    add: _AddHeader,
    *,
    prefix: str,
    config: Qwen4ExpConfig,
    node: str,
) -> None:
    text = config.text
    ple = text.ple
    stream_hidden = text.hidden_size * text.hyperconnection.stream_count
    add(
        f"{prefix}.conv1d.weight",
        (stream_hidden, 1, ple.convolution_kernel_size),
        node,
        "language",
    )
    add(f"{prefix}.key_proj.weight", (stream_hidden, text.hidden_size), node, "language")
    for name in ("norm_conv", "norm_key", "norm_query"):
        add(f"{prefix}.{name}.weight", (stream_hidden,), node, "language")
    embedding = f"{prefix}.ple_embedding"
    add(f"{embedding}.layer_multipliers", (ple.ngram_size,), node, "language", "I64")
    ngram_head_count = (ple.ngram_size - 1) * ple.heads_per_ngram
    shard_width = ple.embedding_dim // ngram_head_count
    for shard in range(ple.split_parts):
        add(
            f"{embedding}.ngram_embedding.shard_{shard}.weight",
            (2_500_012, shard_width),
            node,
            "language",
        )
    add(
        f"{embedding}.ngram_heads_offsets",
        (ngram_head_count,),
        node,
        "language",
        "I64",
    )
    add(
        f"{embedding}.ngram_heads_vocab_sizes",
        (ngram_head_count,),
        node,
        "language",
        "I64",
    )
    add(f"{prefix}.value_proj.weight", (text.hidden_size, text.hidden_size), node, "language")


def logical_quantized_weights(config: Qwen4ExpConfig) -> tuple[LogicalWeight, ...]:
    """Resolve the optional PLE placement without changing checkpoint mathematics."""
    storage = os.environ.get("CUBIC_QWEN4_PLE_STORAGE", "cpu")
    if storage not in ("gpu", "cpu"):
        raise ValueError("CUBIC_QWEN4_PLE_STORAGE must be gpu or cpu")
    weights = _logical_quantized_weights(config)
    if storage == "cpu":
        return weights
    return tuple(
        replace(weight, storage="device") if weight.storage == "pinned_host" else weight
        for weight in weights
    )


@lru_cache(maxsize=4)
def _logical_quantized_weights(config: Qwen4ExpConfig) -> tuple[LogicalWeight, ...]:
    """Model geometry independent of the expert matrix's serialized encoding.

    Split logical projections retain their original stacked BF16 source view.
    Format extensions resolve the physical NVFP4/Cubic payloads independently.
    PLE's declared FP8 storage and shared scale are explicit logical tensors;
    they are not inferred from the routed-expert quantization format.
    """
    from cubic_serve.models.parallel_weights import AxisSpan
    from cubic_serve.models.quantized_checkpoint import LogicalWeight, StackedMatrixSlice

    result = []
    scale_names = set()
    gdn = config.text.gated_delta_net
    for name, header in _expected_headers(config).items():
        if name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
            count, rows, columns = header.shape
            prefix, member = name.rsplit(".", 1)
            projections = ("gate_proj", "up_proj") if member == "gate_up_proj" else ("down_proj",)
            width = rows // len(projections)
            for expert in range(count):
                for index, projection in enumerate(projections):
                    module = f"{prefix}.{expert}.{projection}"
                    result.append(
                        LogicalWeight(
                            module + ".weight",
                            (width, columns),
                            header.dtype,
                            header.partition,
                            "linear",
                            (module.replace("model.language_model.", "model."), name, prefix),
                            stacked_source=StackedMatrixSlice(
                                name,
                                (count, rows, columns),
                                expert,
                                AxisSpan(index * width, (index + 1) * width),
                                (width,) * len(projections),
                            ),
                        )
                    )
            continue
        operation = "tensor"
        partitions: tuple[int, ...] = ()
        dtype = header.dtype
        module = name.removesuffix(".weight")
        is_ple_table = ".ngram_embedding.shard_" in name
        if len(header.shape) == 2 and name.endswith(".weight"):
            operation = (
                "embedding" if is_ple_table or name.endswith("embed_tokens.weight") else "linear"
            )
            if name.endswith("linear_attn.in_proj_qkv.weight"):
                partitions = (gdn.key_heads * gdn.key_head_dim,) * 2 + (
                    gdn.value_heads * gdn.value_head_dim,
                )
            elif name.endswith(".attn.qkv.weight"):
                partitions = (header.shape[0] // 3,) * 3
        if is_ple_table and config.text.ple_embedding_dtype == "float8_e4m3fn":
            dtype = "F8_E4M3"
            scale_names.add(name.rsplit(".shard_", 1)[0] + ".weight_scale")
        result.append(
            LogicalWeight(
                name,
                header.shape,
                dtype,
                header.partition,
                operation,
                (module.replace("model.language_model.", "model."),),
                partitions,
                storage="pinned_host" if is_ple_table else "device",
            )
        )
    result.extend(LogicalWeight(name, (1,), "BF16", "language") for name in sorted(scale_names))
    return tuple(sorted(result, key=lambda item: item.name))


def _add_vision(add: _AddHeader, config: Qwen4ExpConfig) -> None:
    vision = config.vision
    plan = config.architecture_plan()
    hidden = vision.hidden_size
    intermediate = vision.intermediate_size
    patch_node = plan.vision_patch_embedding_node_id
    add(
        "model.visual.patch_embed.proj.weight",
        (
            hidden,
            vision.input_channels,
            vision.temporal_patch_size,
            vision.patch_size,
            vision.patch_size,
        ),
        patch_node,
        "vision",
    )
    add("model.visual.patch_embed.proj.bias", (hidden,), patch_node, "vision")
    add("model.visual.pos_embed.weight", (vision.position_count, hidden), patch_node, "vision")
    for layer in plan.vision_layers:
        prefix = f"model.visual.blocks.{layer.layer_index}"
        for norm in ("norm1", "norm2"):
            add(f"{prefix}.{norm}.weight", (hidden,), layer.node_id, "vision")
            add(f"{prefix}.{norm}.bias", (hidden,), layer.node_id, "vision")
        add(f"{prefix}.attn.qkv.weight", (hidden * 3, hidden), layer.node_id, "vision")
        add(f"{prefix}.attn.qkv.bias", (hidden * 3,), layer.node_id, "vision")
        add(f"{prefix}.attn.proj.weight", (hidden, hidden), layer.node_id, "vision")
        add(f"{prefix}.attn.proj.bias", (hidden,), layer.node_id, "vision")
        add(
            f"{prefix}.mlp.linear_fc1.weight",
            (intermediate, hidden),
            layer.node_id,
            "vision",
        )
        add(f"{prefix}.mlp.linear_fc1.bias", (intermediate,), layer.node_id, "vision")
        add(
            f"{prefix}.mlp.linear_fc2.weight",
            (hidden, intermediate),
            layer.node_id,
            "vision",
        )
        add(f"{prefix}.mlp.linear_fc2.bias", (hidden,), layer.node_id, "vision")
    merge_hidden = hidden * vision.spatial_merge_size**2
    merger = plan.vision_merger_node_id
    add("model.visual.merger.norm.weight", (hidden,), merger, "vision")
    add("model.visual.merger.norm.bias", (hidden,), merger, "vision")
    add(
        "model.visual.merger.linear_fc1.weight",
        (merge_hidden, merge_hidden),
        merger,
        "vision",
    )
    add("model.visual.merger.linear_fc1.bias", (merge_hidden,), merger, "vision")
    add(
        "model.visual.merger.linear_fc2.weight",
        (vision.output_hidden_size, merge_hidden),
        merger,
        "vision",
    )
    add(
        "model.visual.merger.linear_fc2.bias",
        (vision.output_hidden_size,),
        merger,
        "vision",
    )


def quantized_owner_slices(
    config: Qwen4ExpConfig,
    topology: ParallelTopology,
    rank: int,
    *,
    include_mtp: bool = False,
    logical_weights: tuple[LogicalWeight, ...] | None = None,
) -> tuple[LogicalWeightSlice, ...]:
    """Fixed checkpoint-order TP/PP/EP placement, independent of matrix format.

    PLE tables are a single logical vocabulary split over serialized shards.
    Each TP range intersects those shards; no rank receives a full table replica.
    HC projections remain replicated; indexer projections have unique TP owners.
    Main attention KV heads
    replicate only when TP exceeds their declared count.
    """
    from cubic_serve.models.parallel_weights import (
        AxisSpan,
        IndexerTP,
        LogicalWeightSlice,
        checkpoint_expert_slice,
        equal_axis_span,
    )

    if include_mtp:
        from cubic_serve.models.qwen4_exp.model import validate_mtp_metadata

        validate_mtp_metadata(config)
        if topology.pp_size != 1:
            raise WeightSchemaError("MTP pipeline composition is not qualified")
    placement = next((p for p in topology.placements if p.rank == rank), None)
    text = config.text
    if placement is None or text.layer_count % topology.pp_size:
        raise WeightSchemaError("invalid rank or unequal pipeline layer ownership")
    tp, tr = topology.tp_size, placement.tensor_rank
    g, a, moe = text.gated_delta_net, text.full_attention, text.moe
    indexer_plan = IndexerTP(text.qsa.query_heads, text.qsa.head_dim,
                             text.qsa.compression_ratio, tp, tr)
    if any(n % tp for n in (text.vocabulary_size, a.query_heads, g.key_heads, g.value_heads)):
        raise WeightSchemaError("TP must divide vocabulary and query/GDN head dimensions")
    if a.key_value_heads % tp and tp % a.key_value_heads:
        raise WeightSchemaError("KV heads require integer partition or replication")
    logical = logical_weights if logical_weights is not None else logical_quantized_weights(config)
    table_groups: dict[str, list[tuple[int, LogicalWeight]]] = {}
    for w in logical:
        if ".ngram_embedding.shard_" in w.name:
            base, tail = w.name.rsplit(".shard_", 1)
            table_groups.setdefault(base, []).append((int(tail.split(".")[0]), w))
    table_owners = {}
    for entries in table_groups.values():
        ordered = sorted(entries)
        total = sum(w.shape[0] for _, w in ordered)
        # Floor boundaries partition arbitrary vocabulary sizes without a
        # divisibility requirement on individual serialized shard lengths.
        begin, end = total * tr // tp, total * (tr + 1) // tp
        offset = 0
        for _, w in ordered:
            lo, hi = max(begin, offset), min(end, offset + w.shape[0])
            if lo < hi:
                table_owners[w.name] = AxisSpan(lo - offset, hi - offset)
            offset += w.shape[0]
    result = []
    for w in logical:
        name = w.name
        if w.partition not in ("language", "lm_head") and not (
            include_mtp and w.partition == "mtp"
        ):
            continue
        if name.startswith("model.language_model.layers."):
            layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
            stage = layer // (text.layer_count // topology.pp_size)
        elif (
            name.startswith("model.language_model.hyper_connection_mixer.")
            or name == "lm_head.weight"
        ):
            stage = topology.pp_size - 1
        elif name == "model.language_model.embed_tokens.weight" or name.startswith("mtp."):
            stage = 0
        else:
            raise WeightSchemaError(f"unowned logical weight {name}")
        if stage != placement.pipeline_stage:
            continue
        axis = 0
        spans: tuple[AxisSpan, ...] = (AxisSpan(0, w.shape[0]),)
        if ".ngram_embedding.shard_" in name:
            if name not in table_owners:
                continue
            spans = (table_owners[name],)
        elif ".mlp.experts." in name:
            expert = int(name.split(".experts.", 1)[1].split(".", 1)[0])
            axis = 1 if name.endswith("down_proj.weight") else 0
            binding = checkpoint_expert_slice(
                name,
                w.shape,
                expert_id=expert,
                expert_count=moe.expert_count,
                matrix_axis=axis,
                topology=topology,
                rank=rank,
            )
            if binding is None:
                continue
            axis, spans = binding.shard_axis, binding.spans
        elif (name.endswith(("embed_tokens.weight", "lm_head.weight"))
              or name in ("mtp.fc_embedding.weight", "mtp.fc_hidden.weight")):
            spans = (equal_axis_span(w.shape[0], tr, tp),)
        elif ".linear_attn." in name:
            leaf = name.split(".linear_attn.", 1)[1]
            if leaf in ("in_proj_qkv.weight", "conv1d.weight"):
                kw, vw = g.key_heads * g.key_head_dim, g.value_heads * g.value_head_dim
                spans = tuple(
                    AxisSpan(offset + width * tr // tp, offset + width * (tr + 1) // tp)
                    for offset, width in ((0, kw), (kw, kw), (kw * 2, vw))
                )
            elif leaf == "out_proj.weight":
                axis, spans = 1, (equal_axis_span(w.shape[1], tr, tp),)
            elif leaf != "norm.weight":
                spans = (equal_axis_span(w.shape[0], tr, tp),)
        elif name.endswith(".indexer.index_qk_proj.weight"):
            spans = tuple(AxisSpan(begin, end) for begin, end in indexer_plan.projection_ranges)
        elif ".self_attn." in name and ".indexer." not in name:
            leaf = name.split(".self_attn.", 1)[1]
            if leaf == "o_proj.weight":
                axis, spans = 1, (equal_axis_span(w.shape[1], tr, tp),)
            elif leaf in ("k_proj.weight", "v_proj.weight"):
                parts = min(tp, a.key_value_heads)
                owner = tr if tp <= a.key_value_heads else tr // (tp // a.key_value_heads)
                spans = (equal_axis_span(w.shape[0], owner, parts),)
            elif leaf == "q_proj.weight":
                spans = (equal_axis_span(w.shape[0], tr, tp),)
        elif ".mlp.shared_expert." in name:
            axis = 1 if name.endswith("down_proj.weight") else 0
            spans = (equal_axis_span(w.shape[axis], tr, tp),)
        result.append(LogicalWeightSlice(name, axis, spans, w.shape))
    return tuple(result)
