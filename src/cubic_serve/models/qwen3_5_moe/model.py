"""Canonical eager MoE math used by the distributed Qwen3.5 reference.

Routing is global, while this module's expert evaluator accepts only one
contiguous expert-owner slice.  Dispatch and combine remain runtime concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import Tensor

from cubic_serve.contracts.execution import DecodeExtensionMeta, DecodeProvider
from cubic_serve.contracts.model import (
    CacheSpec,
    ModalitySpec,
    ModelSpec,
    ScalarDType,
    StateSharding,
    StateSpec,
    WeightVariantMeta,
)
from cubic_serve.executor.runner import require_reference_execution
from cubic_serve.models.adapter import ModelAdapterDescriptor
from cubic_serve.models.checkpoint import CheckpointInventory
from cubic_serve.models.protocol import ModelBuildContext, ModelProgram
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.models.qwen3_5.model import (
    QWEN3_5_MTP_MAX_PROPOSAL_TOKENS,
    Qwen3_5Config,
    Qwen3_5ModelOperators,
    Qwen3_5OperatorCalls,
    Qwen3_5ReferenceModel,
    Qwen3_5WeightAccess,
    _required_signatures,
    _state_id,
)
from cubic_serve.models.qwen3_5_moe.config import Qwen3_5MoeConfig
from cubic_serve.models.qwen3_5_moe.weights import (
    Qwen3_5MoeWeightPlan,
    qwen3_5_moe_text_weight_policy,
)
from cubic_serve.models.weights import checkpoint_weight_manifest
from cubic_serve.ops.eager.activation import sigmoid_stored as activation_sigmoid
from cubic_serve.ops.eager.activation import silu_stored as activation_silu
from cubic_serve.ops.eager.activation import softmax
from cubic_serve.ops.eager.linear import linear as reference_linear
from cubic_serve.ops.eager.quantization import linear_matrix


@dataclass(frozen=True, slots=True)
class Qwen3_5MoeRoutes:
    """Top-k expert ids and normalized BF16 weights in accumulation order."""

    expert_ids: Tensor
    weights: Tensor


def route_qwen3_5_moe(
    hidden: Tensor,
    gate_weight: Tensor | QuantizedMatrix,
    *,
    top_k: int,
    operators: Qwen3_5OperatorCalls | None = None,
) -> Qwen3_5MoeRoutes:
    """Apply the checkpoint router and order selected routes by expert id."""

    require_reference_execution()
    if (
        hidden.dtype not in (torch.float16, torch.bfloat16)
        or hidden.ndim < 2
        or hidden.numel() == 0
    ):
        raise ValueError("MoE routing requires non-empty rank>=2 BF16 hidden states")
    if (
        len(gate_weight.shape) != 2
        or gate_weight.shape[1] != hidden.shape[-1]
        or (isinstance(gate_weight, Tensor)
            and (gate_weight.dtype is not hidden.dtype or gate_weight.device != hidden.device))
    ):
        raise ValueError("MoE router weight must be BF16 [experts, hidden] on the hidden device")
    if type(top_k) is not int or not 0 < top_k <= gate_weight.shape[0]:
        raise ValueError("MoE top_k lies outside the expert set")
    flat = hidden.reshape(-1, hidden.shape[-1])
    if operators is not None:
        logits = operators.linear(flat, gate_weight)
    elif isinstance(gate_weight, QuantizedMatrix):
        logits = linear_matrix(flat, gate_weight)
    else:
        logits = reference_linear(flat, gate_weight)
    # Rank actual logits, so low-precision probability ties never change ownership.
    expert_ids = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :top_k]
    selected = logits.gather(1, expert_ids)
    weights = softmax(selected, dim=-1, storage_dtype=hidden.dtype).to(hidden.dtype)
    expert_ids, order = torch.sort(expert_ids, dim=-1)
    weights = weights.gather(1, order)
    return Qwen3_5MoeRoutes(expert_ids.contiguous(), weights.contiguous())


def execute_local_packed_experts(
    hidden: Tensor,
    expert_ids: Tensor,
    gate_up: Tensor,
    down: Tensor,
    *,
    first_expert_id: int,
) -> Tensor:
    """Evaluate received rows using only one owner's packed expert weights."""

    require_reference_execution()
    if hidden.dtype not in (torch.float16, torch.bfloat16) or hidden.ndim != 2:
        raise ValueError("local expert input must be BF16 [routes, hidden]")
    local_experts = gate_up.shape[0]
    if (
        expert_ids.dtype is not torch.int64
        or expert_ids.shape != (hidden.shape[0],)
        or expert_ids.device != hidden.device
        or gate_up.dtype is not hidden.dtype
        or down.dtype is not hidden.dtype
        or gate_up.device != hidden.device
        or down.device != hidden.device
        or gate_up.ndim != 3
        or down.ndim != 3
        or down.shape[0] != local_experts
        or gate_up.shape[1] != down.shape[2] * 2
        or gate_up.shape[2] != hidden.shape[1]
        or down.shape[1] != hidden.shape[1]
    ):
        raise ValueError("packed local expert payload violates its [E,2I,H]/[E,H,I] ABI")
    if type(first_expert_id) is not int or first_expert_id < 0:
        raise ValueError("first_expert_id must be non-negative")
    if hidden.shape[0] and bool(
        ((expert_ids < first_expert_id) | (expert_ids >= first_expert_id + local_experts))
        .any()
        .item()
    ):
        raise ValueError("a dispatched route is not owned by this expert slice")

    output = torch.empty_like(hidden)
    for local_id in range(local_experts):
        rows = torch.nonzero(expert_ids == first_expert_id + local_id, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        current = hidden.index_select(0, rows)
        gate, up = reference_linear(current, gate_up[local_id]).chunk(2, dim=-1)
        current = reference_linear(activation_silu(gate) * up, down[local_id])
        output.index_copy_(0, rows, current)
    return output


class Qwen3_5MoeFeedForward(Protocol):
    def __call__(
        self, hidden: Tensor, *, prefix: str, operators: Qwen3_5OperatorCalls
    ) -> Tensor: ...


class Qwen3_5MoeReferenceFeedForward:
    def __init__(self, config: Qwen3_5MoeConfig, weights: Qwen3_5WeightAccess) -> None:
        self.config, self.weights = config, weights

    def __call__(self, hidden: Tensor, *, prefix: str, operators: Qwen3_5OperatorCalls) -> Tensor:
        flat = hidden.reshape(-1, hidden.shape[-1])
        moe = self.config.graph.text.moe
        if moe is None:
            raise ValueError("reference MoE requires declared expert semantics")
        routes = route_qwen3_5_moe(
            flat,
            self.weights[f"{prefix}.gate.weight"],
            top_k=moe.experts_per_token,
            operators=operators,
        )
        result = torch.zeros_like(flat)
        # Canonical expert-id route order and rounded combine boundaries are
        # identical to the independent distributed reference's local rule.
        for row in range(flat.shape[0]):
            for slot, expert in enumerate(routes.expert_ids[row].tolist()):
                expert_prefix = f"{prefix}.experts.{expert}"
                x = flat[row : row + 1]
                gate = activation_silu(
                    operators.linear(x, self.weights[f"{expert_prefix}.gate_proj.weight"])
                )
                up = operators.linear(x, self.weights[f"{expert_prefix}.up_proj.weight"])
                out = operators.linear(gate * up, self.weights[f"{expert_prefix}.down_proj.weight"])
                result[row] = result[row] + (out[0] * routes.weights[row, slot]).to(hidden.dtype)
        gate = activation_silu(
            operators.linear(flat, self.weights[f"{prefix}.shared_expert.gate_proj.weight"])
        )
        up = operators.linear(flat, self.weights[f"{prefix}.shared_expert.up_proj.weight"])
        shared = operators.linear(
            gate * up, self.weights[f"{prefix}.shared_expert.down_proj.weight"]
        )
        shared_gate = activation_sigmoid(
            operators.linear(flat, self.weights[f"{prefix}.shared_expert_gate.weight"])
        )
        return (result + (shared * shared_gate).to(hidden.dtype)).reshape_as(hidden)


class Qwen3_5MoeStageModel(Qwen3_5ReferenceModel):
    """Reuse canonical Qwen attention/state math with an explicit distributed MoE call."""

    __slots__ = ("_moe",)
    _model_type = "qwen3_5_moe"

    def __init__(
        self,
        *,
        config: Qwen3_5MoeConfig,
        weights: Qwen3_5WeightAccess,
        moe: Qwen3_5MoeFeedForward,
        recurrent_state_dtype: torch.dtype | None = None,
        operators: Qwen3_5ModelOperators | None = None,
    ) -> None:
        super().__init__(
            config=cast(Qwen3_5Config, config),
            weights=weights,
            recurrent_state_dtype=recurrent_state_dtype,
            operators=operators,
        )
        self._moe = moe

    def _mlp(
        self,
        hidden: Tensor,
        *,
        prefix: str,
        operators: Qwen3_5OperatorCalls,
    ) -> Tensor:
        return self._moe(hidden, prefix=prefix, operators=operators)


class Qwen3_5MoeAdapter:
    """Minimal declaration for the built-in Qwen3.5 MoE checkpoint family."""

    def __init__(
        self,
        *,
        config: Qwen3_5MoeConfig,
        inventory: CheckpointInventory,
        plan: Qwen3_5MoeWeightPlan,
    ) -> None:
        self.config, self.inventory, self.plan = config, inventory, plan
        text = config.graph.text
        policy = qwen3_5_moe_text_weight_policy(plan)
        weights = checkpoint_weight_manifest(
            inventory,
            policy=policy,
            semantic_plan_digest=plan.semantic_plan_digest,
        )
        states = tuple(
            state
            for layer, kind in enumerate(text.layer_schedule)
            if kind.value == "gated-delta-net"
            for state in (
                StateSpec(
                    state_id=_state_id(layer, "convolution"),
                    dtype=ScalarDType.BF16,
                    layout="batch_channel_history",
                    shape_symbols=("batch", "channel", "history"),
                    sharding=StateSharding.TENSOR,
                ),
                StateSpec(
                    state_id=_state_id(layer, "recurrent"),
                    dtype=ScalarDType.FP32,
                    layout="batch_value_head_value_dim_key_dim",
                    shape_symbols=("batch", "value_head", "value_dim", "key_dim"),
                    sharding=StateSharding.TENSOR,
                ),
            )
        )
        model_id = "qwen3_5_moe:text"
        self.weight_policy = policy
        self._descriptor = ModelAdapterDescriptor(
            adapter_id="quanttrio.qwen3_5_moe",
            adapter_revision="1",
            model_spec=ModelSpec(
                model_id=model_id,
                architecture=config.identity.architecture,
                revision=inventory.inventory_digest,
                weights=weights,
                cache=CacheSpec(
                    kv_dtype=ScalarDType.BF16,
                    page_size_tokens=16,
                    layout="layer_token_kv_head_dim",
                    prefix_enabled=True,
                ),
                modalities=(
                    ModalitySpec(name="text"),
                    ModalitySpec(
                        name="image",
                        mime_prefixes=("image/",),
                        encoder_id=f"qwen3_5.vision.reference@{plan.semantic_plan_digest}",
                        feature_hidden_size=text.hidden_size,
                    ),
                    ModalitySpec(
                        name="video",
                        mime_prefixes=("video/",),
                        encoder_id=f"qwen3_5.vision.reference@{plan.semantic_plan_digest}",
                        feature_hidden_size=text.hidden_size,
                    ),
                ),
                states=states,
                layer_count=text.layer_count,
            ),
            weight_variant=WeightVariantMeta(
                base_model_id=model_id,
                base_revision=inventory.inventory_digest,
            ),
            decode_extension=DecodeExtensionMeta(
                provider=DecodeProvider.MTP,
                provider_version="qwen3_5_moe.mtp.v1",
                max_proposal_tokens=QWEN3_5_MTP_MAX_PROPOSAL_TOKENS,
                state_schema_id="qwen3_5_moe.mtp.full_attention.bf16_kv.v1",
            ),
            required_ops=_required_signatures(),
        )

    def describe(self) -> ModelAdapterDescriptor:
        return self._descriptor

    def build(self, context: ModelBuildContext) -> ModelProgram:
        raise RuntimeError("Qwen3.5 MoE requires a distributed TP/EP execution plan")


__all__ = (
    "Qwen3_5MoeAdapter",
    "Qwen3_5MoeRoutes",
    "Qwen3_5MoeStageModel",
    "execute_local_packed_experts",
    "route_qwen3_5_moe",
)


def build_qwen3_5_moe_text_model(
    *, config, weights, recurrent_state_dtype=None, operators=None,
):
    """Compose the original native body; state and execution remain engine-owned."""
    return Qwen3_5MoeStageModel(
        config=config,
        weights=weights,
        moe=Qwen3_5MoeReferenceFeedForward(config, weights),
        recurrent_state_dtype=recurrent_state_dtype,
        operators=operators,
    )
