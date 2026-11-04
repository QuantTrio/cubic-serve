"""Closed checkpoint-header partition for model_type ``qwen3_5_moe``."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace

from cubic_serve.contracts.execution import ParallelTopology
from cubic_serve.models.checkpoint import CheckpointInventory
from cubic_serve.models.parallel_weights import (
    AxisSpan,
    LogicalWeightSlice,
    checkpoint_expert_slice,
)
from cubic_serve.models.quantized_checkpoint import LogicalWeight
from cubic_serve.models.qwen3_5.model import AttentionKind
from cubic_serve.models.qwen3_5.weights import (
    _add_full_attention,
    _add_gdn,
    _add_vision,
    _AddHeader,
    _architecture_node_ids,
    _ExpectedHeader,
    _quantized_stage_weights,
    _quantized_tensor_slice,
)
from cubic_serve.models.qwen3_5_moe.config import Qwen3_5MoeConfig
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
from cubic_serve.models.weights import FullBF16WeightPolicy

Qwen3_5MoeWeightPlan = CheckpointWeightPlan

__all__ = (
    "Qwen3_5MoeWeightPlan",
    "TensorHeader",
    "WeightSchemaError",
    "plan_qwen3_5_moe_checkpoint_weights",
    "plan_qwen3_5_moe_weight_headers",
    "qwen3_5_moe_mtp_weight_policy",
    "qwen3_5_moe_text_weight_policy",
)


def qwen3_5_moe_text_weight_policy(
    plan: Qwen3_5MoeWeightPlan,
) -> FullBF16WeightPolicy:
    """Close the BF16/F32 text slice without loading vision or MTP tensors."""

    if plan.model_type != "qwen3_5_moe":
        raise TypeError("Qwen3.5 MoE text policy requires its validated weight plan")
    names = tuple(sorted((*plan.base_language_names, *plan.lm_head_names)))
    selected = set(names)
    return FullBF16WeightPolicy(
        expected_names=names,
        fp32_stability_names=tuple(name for name in plan.fp32_stability_names if name in selected),
        int64_metadata_names=(),
    )


def qwen3_5_moe_mtp_weight_policy(
    plan: Qwen3_5MoeWeightPlan,
) -> FullBF16WeightPolicy:
    """Close the BF16/F32 text graph and its MTP extension."""

    if plan.model_type != "qwen3_5_moe":
        raise TypeError("Qwen3.5 MoE MTP policy requires its validated weight plan")
    names = tuple(
        sorted((*plan.base_language_names, *plan.lm_head_names, *plan.mtp_extension_names))
    )
    selected = set(names)
    return FullBF16WeightPolicy(
        expected_names=names,
        fp32_stability_names=tuple(name for name in plan.fp32_stability_names if name in selected),
        int64_metadata_names=(),
    )


def plan_qwen3_5_moe_checkpoint_weights(
    config: Qwen3_5MoeConfig, inventory: CheckpointInventory
) -> Qwen3_5MoeWeightPlan:
    """Plan an already-attested inventory without reading tensor payloads."""

    return _plan(
        config,
        tensor_headers_from_inventory(inventory),
        inventory=inventory,
    )


def plan_qwen3_5_moe_weight_headers(
    config: Qwen3_5MoeConfig, headers: Iterable[TensorHeader]
) -> Qwen3_5MoeWeightPlan:
    """Pure header-record API for CI and pre-attestation checks."""

    return _plan(config, materialize_headers(headers), inventory=None)


def _plan(
    config: Qwen3_5MoeConfig,
    headers: tuple[TensorHeader, ...],
    *,
    inventory: CheckpointInventory | None,
) -> Qwen3_5MoeWeightPlan:
    if config.identity.model_type != "qwen3_5_moe":
        raise WeightSchemaError("qwen3_5_moe weight schema received another model_type")
    expected = _expected_headers(config)
    actual = {header.name: header for header in headers}
    missing = sorted(set(expected) - set(actual))
    unknown = sorted(set(actual) - set(expected))
    if missing or unknown:
        raise WeightSchemaError(
            f"qwen3_5_moe checkpoint tensor set is not closed: missing={missing}, unknown={unknown}"
        )
    for name, contract in expected.items():
        header = actual[name]
        if header.dtype != contract.dtype:
            raise WeightSchemaError(
                f"qwen3_5_moe tensor {name!r} dtype mismatch: {header.dtype} != {contract.dtype}"
            )
        if header.shape != contract.shape:
            raise WeightSchemaError(
                f"qwen3_5_moe tensor {name!r} shape mismatch: {header.shape} != {contract.shape}"
            )

    partitions = {
        partition: tuple(
            sorted(name for name, item in expected.items() if item.partition == partition)
        )
        for partition in ("language", "vision", "lm_head", "mtp")
    }
    fp32 = tuple(sorted(name for name, item in expected.items() if item.dtype == "F32"))
    by_node: dict[str, list[str]] = defaultdict(list)
    for name, contract in expected.items():
        by_node[contract.node_id].append(name)
    bindings = tuple(
        WeightNodeBinding(node_id=node, tensor_names=tuple(sorted(names)))
        for node, names in sorted(by_node.items())
    )
    plan = _issue_weight_plan(
        model_type="qwen3_5_moe",
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
        fp32_stability_names=fp32,
        int64_metadata_names=(),
        allowed_node_ids=_architecture_node_ids(config.architecture_plan()),
        node_bindings=bindings,
    )
    return plan


def _expected_headers(config: Qwen3_5MoeConfig) -> dict[str, _ExpectedHeader]:
    graph = config.graph
    text = graph.text
    moe = text.moe
    if moe is None:
        raise WeightSchemaError("qwen3_5_moe checkpoint requires MoE configuration")
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
            raise WeightSchemaError(f"internal qwen3_5_moe schema duplicates {name!r}")
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
    add("model.language_model.norm.weight", (hidden,), plan.final_norm_node_id, "language")
    add("lm_head.weight", (vocabulary, hidden), plan.lm_head_node_id, "lm_head")
    for layer in plan.decoder_layers:
        prefix = f"model.language_model.layers.{layer.layer_index}"
        add(f"{prefix}.input_layernorm.weight", (hidden,), layer.node_id, "language")
        add(
            f"{prefix}.post_attention_layernorm.weight",
            (hidden,),
            layer.node_id,
            "language",
        )
        _add_packed_moe(
            add_header,
            prefix=f"{prefix}.mlp",
            hidden=hidden,
            expert_count=moe.expert_count,
            expert_intermediate=moe.expert_intermediate_size,
            shared_intermediate=moe.shared_expert_intermediate_size,
            node=layer.node_id,
            partition="language",
        )
        if layer.attention is AttentionKind.GATED_DELTA_NET:
            _add_gdn(
                add_header,
                graph,
                prefix=f"{prefix}.linear_attn",
                node=layer.node_id,
                partition="language",
                stability_dtype="F32",
            )
        else:
            _add_full_attention(
                add_header,
                graph,
                prefix=f"{prefix}.self_attn",
                node=layer.node_id,
                partition="language",
            )

    _add_vision(add_header, graph, plan)
    if plan.mtp_layers:
        add("mtp.fc.weight", (hidden, hidden * 2), plan.mtp_root_node_id, "mtp")
        for suffix in ("norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"):
            add(f"mtp.{suffix}", (hidden,), plan.mtp_root_node_id, "mtp")
    for mtp_layer in plan.mtp_layers:
        prefix = f"mtp.layers.{mtp_layer.layer_index}"
        add(f"{prefix}.input_layernorm.weight", (hidden,), mtp_layer.node_id, "mtp")
        add(
            f"{prefix}.post_attention_layernorm.weight",
            (hidden,),
            mtp_layer.node_id,
            "mtp",
        )
        _add_unpacked_moe(
            add_header,
            prefix=f"{prefix}.mlp",
            hidden=hidden,
            expert_count=moe.expert_count,
            expert_intermediate=moe.expert_intermediate_size,
            shared_intermediate=moe.shared_expert_intermediate_size,
            node=mtp_layer.node_id,
            partition="mtp",
        )
        _add_full_attention(
            add_header,
            graph,
            prefix=f"{prefix}.self_attn",
            node=mtp_layer.node_id,
            partition="mtp",
        )
    return expected


def _add_packed_moe(
    add: _AddHeader,
    *,
    prefix: str,
    hidden: int,
    expert_count: int,
    expert_intermediate: int,
    shared_intermediate: int,
    node: str,
    partition: str,
) -> None:
    add(
        f"{prefix}.experts.down_proj",
        (expert_count, hidden, expert_intermediate),
        node,
        partition,
    )
    add(
        f"{prefix}.experts.gate_up_proj",
        (expert_count, expert_intermediate * 2, hidden),
        node,
        partition,
    )
    _add_moe_router_and_shared(
        add,
        prefix=prefix,
        hidden=hidden,
        expert_count=expert_count,
        shared_intermediate=shared_intermediate,
        node=node,
        partition=partition,
    )


def _add_unpacked_moe(
    add: _AddHeader,
    *,
    prefix: str,
    hidden: int,
    expert_count: int,
    expert_intermediate: int,
    shared_intermediate: int,
    node: str,
    partition: str,
) -> None:
    for expert in range(expert_count):
        expert_prefix = f"{prefix}.experts.{expert}"
        add(
            f"{expert_prefix}.down_proj.weight",
            (hidden, expert_intermediate),
            node,
            partition,
        )
        add(
            f"{expert_prefix}.gate_proj.weight",
            (expert_intermediate, hidden),
            node,
            partition,
        )
        add(
            f"{expert_prefix}.up_proj.weight",
            (expert_intermediate, hidden),
            node,
            partition,
        )
    _add_moe_router_and_shared(
        add,
        prefix=prefix,
        hidden=hidden,
        expert_count=expert_count,
        shared_intermediate=shared_intermediate,
        node=node,
        partition=partition,
    )


def _add_moe_router_and_shared(
    add: _AddHeader,
    *,
    prefix: str,
    hidden: int,
    expert_count: int,
    shared_intermediate: int,
    node: str,
    partition: str,
) -> None:
    add(f"{prefix}.gate.weight", (expert_count, hidden), node, partition)
    add(
        f"{prefix}.shared_expert.down_proj.weight",
        (hidden, shared_intermediate),
        node,
        partition,
    )
    add(
        f"{prefix}.shared_expert.gate_proj.weight",
        (shared_intermediate, hidden),
        node,
        partition,
    )
    add(
        f"{prefix}.shared_expert.up_proj.weight",
        (shared_intermediate, hidden),
        node,
        partition,
    )
    add(f"{prefix}.shared_expert_gate.weight", (1, hidden), node, partition)


def logical_quantized_weights(config: Qwen3_5MoeConfig) -> tuple[LogicalWeight, ...]:
    from cubic_serve.models.quantized_checkpoint import StackedMatrixSlice
    from cubic_serve.models.qwen3_5.weights import _logical_quantized_headers

    headers = {}
    sources = {}
    for name, header in _expected_headers(config).items():
        if not name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
            headers[name] = header
            continue
        count, n, k = header.shape
        prefix, member = name.rsplit(".", 1)
        projections = ("gate_proj", "up_proj") if member == "gate_up_proj" else ("down_proj",)
        shape = (n // 2 if member == "gate_up_proj" else n, k)
        for expert in range(count):
            for projection in projections:
                logical_name = f"{prefix}.{expert}.{projection}.weight"
                headers[logical_name] = replace(header, shape=shape)
                start = shape[0] if projection == "up_proj" else 0
                sources[logical_name] = StackedMatrixSlice(
                    name,
                    (count, n, k),
                    expert,
                    AxisSpan(start, start + shape[0]),
                    (shape[0],) * len(projections),
                )
    return tuple(
        replace(
            w,
            stacked_source=sources.get(w.name),
            aliases=(*w.aliases, sources[w.name].name, sources[w.name].name.rsplit(".", 1)[0])
            if w.name in sources
            else w.aliases,
        )
        for w in _logical_quantized_headers(config.graph, headers)
    )


def quantized_owner_slices(
    config: Qwen3_5MoeConfig,
    topology: ParallelTopology,
    rank: int,
    *,
    include_mtp: bool = False,
    logical_weights: tuple[LogicalWeight, ...] | None = None,
) -> tuple[LogicalWeightSlice, ...]:
    """Preserve logical expert IDs with TP shards or fixed contiguous EP owners.

    EP spans follow checkpoint expert order. No routing statistic, encoding,
    device generation or concurrency may alter this placement.
    """
    stage_weights, tensor_rank = _quantized_stage_weights(
        config,
        logical_weights if logical_weights is not None else logical_quantized_weights(config),
        topology,
        rank,
        include_mtp=include_mtp,
    )
    moe = config.graph.text.moe
    if moe is None:
        raise WeightSchemaError("MoE owner plan requires declared expert geometry")
    result = []
    for weight in stage_weights:
        name = weight.name
        if ".mlp.experts." in name:
            expert = int(name.split(".experts.", 1)[1].split(".", 1)[0])
            axis = 1 if name.endswith("down_proj.weight") else 0
            binding = checkpoint_expert_slice(
                name,
                weight.shape,
                expert_id=expert,
                expert_count=moe.expert_count,
                matrix_axis=axis,
                topology=topology,
                rank=rank,
            )
            if binding is not None:
                result.append(binding)
        elif name.endswith((".mlp.gate.weight", ".mlp.shared_expert_gate.weight")):
            result.append(
                LogicalWeightSlice(name, 0, (AxisSpan(0, weight.shape[0]),), weight.shape)
            )
        else:
            canonical = replace(weight, name=name.replace(".mlp.shared_expert.", ".mlp."))
            binding = _quantized_tensor_slice(config, canonical, tensor_rank, topology.tp_size)
            result.append(replace(binding, tensor_name=name))
    return tuple(result)
