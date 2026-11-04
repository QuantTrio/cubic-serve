"""Built-in surface for checkpoint model_type ``qwen3_5_moe``."""

from dataclasses import replace

from cubic_serve.models.qwen3_5 import MODEL_DEFINITION as _QWEN_FRONTEND_BASE
from cubic_serve.models.qwen3_5.model import Qwen3_5ConfigError
from cubic_serve.models.qwen3_5_moe.config import (
    Qwen3_5MoeConfig,
    parse_qwen3_5_moe_config,
    parse_qwen3_5_moe_config_json,
)
from cubic_serve.models.qwen3_5_moe.distributed import (
    Qwen3_5MoeDistributedReferenceError,
    Qwen3_5MoeDistributedReferenceModel,
    seal_qwen3_5_moe_distributed_reference,
)
from cubic_serve.models.qwen3_5_moe.model import (
    Qwen3_5MoeAdapter,
    Qwen3_5MoeRoutes,
    build_qwen3_5_moe_text_model,
    execute_local_packed_experts,
    route_qwen3_5_moe,
)
from cubic_serve.models.qwen3_5_moe.parallel import (
    MoeRankWeightBinding,
    MoeWeightLayout,
    Qwen3_5MoeParallelError,
    Qwen3_5MoeParallelPlan,
    Qwen3_5MoeRankWeights,
    load_qwen3_5_moe_rank_weights,
    plan_qwen3_5_moe_tp4_pp2_ep4,
    plan_qwen3_5_moe_weights,
)
from cubic_serve.models.qwen3_5_moe.weights import (
    Qwen3_5MoeWeightPlan,
    TensorHeader,
    WeightSchemaError,
    logical_quantized_weights,
    plan_qwen3_5_moe_checkpoint_weights,
    plan_qwen3_5_moe_weight_headers,
    quantized_owner_slices,
    qwen3_5_moe_mtp_weight_policy,
    qwen3_5_moe_text_weight_policy,
)

MODEL_DEFINITION = replace(
    _QWEN_FRONTEND_BASE,
    model_type="qwen3_5_moe",
    parse_config=parse_qwen3_5_moe_config,
    logical_weights=logical_quantized_weights,
    owner_slices=quantized_owner_slices,
    draft_provider=None,
    text_model=build_qwen3_5_moe_text_model,
)

__all__ = (
    "MODEL_DEFINITION",
    "MoeRankWeightBinding",
    "MoeWeightLayout",
    "Qwen3_5ConfigError",
    "Qwen3_5MoeAdapter",
    "Qwen3_5MoeConfig",
    "Qwen3_5MoeDistributedReferenceError",
    "Qwen3_5MoeDistributedReferenceModel",
    "Qwen3_5MoeParallelError",
    "Qwen3_5MoeParallelPlan",
    "Qwen3_5MoeRankWeights",
    "Qwen3_5MoeRoutes",
    "Qwen3_5MoeWeightPlan",
    "TensorHeader",
    "WeightSchemaError",
    "execute_local_packed_experts",
    "load_qwen3_5_moe_rank_weights",
    "parse_qwen3_5_moe_config",
    "parse_qwen3_5_moe_config_json",
    "plan_qwen3_5_moe_checkpoint_weights",
    "plan_qwen3_5_moe_tp4_pp2_ep4",
    "plan_qwen3_5_moe_weight_headers",
    "plan_qwen3_5_moe_weights",
    "qwen3_5_moe_mtp_weight_policy",
    "qwen3_5_moe_text_weight_policy",
    "route_qwen3_5_moe",
    "seal_qwen3_5_moe_distributed_reference",
)
