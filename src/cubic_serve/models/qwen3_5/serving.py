"""Model-owned startup bindings to the existing hybrid text stage."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from cubic_serve.models.adapter import TextGraphDefinition
from cubic_serve.models.qwen3_5.model import Qwen3_5GraphVariant

if TYPE_CHECKING:
    from cubic_serve.models.builtin import Qwen3_5Admission
    from cubic_serve.models.weights import CheckpointWeightSource
    from cubic_serve.runtime.performance import _Admission


def admit_reference(
    model_root: str | Path,
    *,
    verify_payloads: bool = True,
) -> Qwen3_5Admission:
    """Close one real checkpoint before runtime allocates resources."""

    import hashlib

    from cubic_serve.models.builtin import Qwen3_5Admission, parse_builtin_model_config_json
    from cubic_serve.models.checkpoint import CheckpointInventory, inspect_checkpoint
    from cubic_serve.models.loader import seal_builtin_model_adapter
    from cubic_serve.models.qwen3_5 import (
        AttentionKind,
        Qwen3_5Adapter,
        Qwen3_5Config,
        plan_qwen3_5_checkpoint_weights,
        qwen3_5_text_weight_policy,
    )
    from cubic_serve.models.qwen3_5_moe import (
        Qwen3_5MoeAdapter,
        Qwen3_5MoeConfig,
        plan_qwen3_5_moe_checkpoint_weights,
        qwen3_5_moe_text_weight_policy,
    )
    from cubic_serve.models.weights import CheckpointWeightSource
    from cubic_serve.runtime.cache import KVLayout

    block_size = 16
    root = Path(model_root).resolve(strict=True)
    config_document = (root / "config.json").read_bytes()
    config = parse_builtin_model_config_json(config_document)
    if not isinstance(config, Qwen3_5Config | Qwen3_5MoeConfig):
        raise ValueError("checkpoint is not a Qwen3.5 model")
    if config.graph.text.dtype != "bfloat16":
        raise ValueError("the unquantized sealed runtime currently requires BF16 compute")
    inventory: CheckpointInventory = inspect_checkpoint(
        root,
        verify_payloads=verify_payloads,
    )
    observed = f"b2s256:{hashlib.blake2s(config_document).hexdigest()}"
    if observed != inventory.config_file.content_digest:
        raise ValueError("qwen3_5 config changed during admission")
    if isinstance(config, Qwen3_5MoeConfig):
        plan = plan_qwen3_5_moe_checkpoint_weights(config, inventory)
        text_weight_policy = qwen3_5_moe_text_weight_policy(plan)
        adapter = seal_builtin_model_adapter(
            Qwen3_5MoeAdapter(config=config, inventory=inventory, plan=plan),
            source_path=Path(__file__).parent.parent / "qwen3_5_moe" / "model.py",
        )
    else:
        plan = plan_qwen3_5_checkpoint_weights(config, inventory)
        text_weight_policy = qwen3_5_text_weight_policy(plan)
        adapter = seal_builtin_model_adapter(
            Qwen3_5Adapter(config=config, inventory=inventory, plan=plan),
            source_path=Path(__file__).parent / "model.py",
        )
    full_attention_layers = tuple(
        index
        for index, kind in enumerate(config.graph.text.layer_schedule)
        if kind is AttentionKind.FULL_ATTENTION
    )
    attention = config.graph.text.full_attention
    return Qwen3_5Admission(
        model_root=root,
        vocabulary_size=config.graph.text.vocabulary_size,
        adapter=adapter,
        weight_policy=text_weight_policy,
        checkpoint_weights=CheckpointWeightSource(inventory, plan.semantic_plan_digest),
        kv_layout=KVLayout(
            layer_ids=full_attention_layers,
            block_size_tokens=block_size,
            num_kv_heads=attention.key_value_heads,
            head_dim=attention.head_dim,
        ),
        owned_layer_ids=tuple(range(config.graph.text.layer_count)),
        max_sequence_tokens=config.graph.text.max_position_embeddings,
    )


def load_input_encoder(model_root, *, parallel, **options):
    from cubic_serve.executor.vision import load_qwen_vision_graph, load_qwen_vision_tp

    if parallel:
        return load_qwen_vision_tp(model_root, **options)
    options.pop("model_definition_source", None)  # In-process encoder shares the caller lifetime.
    return load_qwen_vision_graph(model_root, **options)


def text_graph_definition(config, topology, *, block_size, recurrent_state_dtype, definition=None):
    from functools import partial

    from cubic_serve.executor.quantized_model_runner import QuantizedDecodeStage

    if recurrent_state_dtype != "bfloat16":
        raise ValueError("quantized graphs require --mamba-ssm-cache-dtype bfloat16")
    operations = (("linear", "moe") if config.graph.variant is Qwen3_5GraphVariant.MOE
                  else ("linear",))
    stage_type = QuantizedDecodeStage
    if definition is not None:
        if definition.text_model is None:
            raise ValueError("text stage requires a declared native text constructor")
        stage_type = partial(QuantizedDecodeStage, model_factory=definition.text_model)
    return TextGraphDefinition(
        stage_type=stage_type,
        named_state_fields=False,
        stage_options={"route_completion_costs": {}},
        required_operations=operations,
        block_size_tokens=block_size,
        recurrent_state_dtype=recurrent_state_dtype,
    )


def start_performance_executor(checkpoint, **options):
    """Keep the legacy native executor construction inside its model package."""
    from cubic_serve.runtime.performance import Qwen3_5TP2PP2PerformanceExecutor

    return Qwen3_5TP2PP2PerformanceExecutor.start(checkpoint, **options)


def admit_legacy_performance(
    checkpoint: Path,
    devices: tuple[int, ...],
    max_sequence_tokens: int | None,
    max_running_requests: int,
    tp_size: int = 2,
    pp_size: int = 2,
    ep_size: int = 1,
    num_speculative_tokens: int = 0,
    max_speculative_tokens: int = 0,
    dynamic_max_speculative_tokens: int = 0,
    dynamic_speculation: bool = False,
    use_prefix_cache: bool = True,
    mamba_ssm_cache_dtype: str = "bfloat16",
    prefill_chunk_tokens: int | None = None,
    checkpoint_weights: CheckpointWeightSource | None = None,
) -> _Admission:
    import hashlib
    import json

    from cubic_serve.executor.distributed_reference import (
        seal_qwen3_5_distributed_reference,
        seal_qwen3_5_target_mtp_distributed_reference,
    )
    from cubic_serve.models.checkpoint import inspect_checkpoint
    from cubic_serve.models.qwen3_5 import (
        parse_qwen3_5_config_json,
        plan_qwen3_5_checkpoint_weights,
        plan_qwen3_5_dense_weights,
        plan_qwen3_5_tp2_pp2_weights,
    )
    from cubic_serve.models.qwen3_5_moe import (
        parse_qwen3_5_moe_config_json,
        plan_qwen3_5_moe_checkpoint_weights,
        plan_qwen3_5_moe_weights,
        seal_qwen3_5_moe_distributed_reference,
    )
    from cubic_serve.models.weights import CheckpointWeightSource
    from cubic_serve.runtime.performance import (
        _LOGGER,
        _Admission,
        _resolve_max_sequence_tokens,
        _topology,
    )

    root = checkpoint.expanduser().resolve(strict=True)
    if (
        type(max_running_requests) is not int
        or max_running_requests <= 0
    ):
        raise ValueError("max_running_requests must be positive")
    config_document = (root / "config.json").read_bytes()
    raw_config = json.loads(config_document)
    is_moe = raw_config.get("model_type") == "qwen3_5_moe"
    config = (
        parse_qwen3_5_moe_config_json(config_document)
        if is_moe
        else parse_qwen3_5_config_json(config_document)
    )
    checkpoint_state_dtype = config.graph.text.gated_delta_net.recurrent_state_dtype
    if mamba_ssm_cache_dtype not in ("auto", "float32", "bfloat16"):
        raise ValueError("mamba_ssm_cache_dtype must be auto, float32, or bfloat16")
    resolved_state_dtype = (
        checkpoint_state_dtype
        if mamba_ssm_cache_dtype == "auto"
        else mamba_ssm_cache_dtype
    )
    if resolved_state_dtype not in ("float32", "bfloat16"):
        raise ValueError("checkpoint declares an unsupported Mamba SSM cache dtype")
    if resolved_state_dtype != checkpoint_state_dtype:
        _LOGGER.warning(
            "Mamba SSM cache uses bfloat16. Set "
            "--mamba-ssm-cache-dtype float32 to use the checkpoint dtype."
        )
    resolved_max_sequence_tokens = _resolve_max_sequence_tokens(config, max_sequence_tokens)
    if checkpoint_weights is None:
        inventory = inspect_checkpoint(root)
    else:
        if not isinstance(checkpoint_weights, CheckpointWeightSource):
            raise TypeError("checkpoint_weights must be a CheckpointWeightSource")
        inventory = checkpoint_weights.inventory
        if inventory.model_root != root:
            raise ValueError("checkpoint admission belongs to a different model root")
        config_digest = f"b2s256:{hashlib.blake2s(config_document).hexdigest()}"
        if inventory.config_file.content_digest != config_digest:
            raise ValueError("qwen3_5 config changed after checkpoint admission")
    weight_plan = (
        plan_qwen3_5_moe_checkpoint_weights(config, inventory)
        if is_moe
        else plan_qwen3_5_checkpoint_weights(config, inventory)
    )
    if (
        checkpoint_weights is not None
        and checkpoint_weights.semantic_plan_digest != weight_plan.semantic_plan_digest
    ):
        raise ValueError("checkpoint admission semantic plan changed")
    topology = _topology(devices, tp_size=tp_size, pp_size=pp_size, ep_size=ep_size)
    if (
        type(num_speculative_tokens) is not int
        or type(max_speculative_tokens) is not int
        or min(num_speculative_tokens, max_speculative_tokens) < 0
        or num_speculative_tokens > max_speculative_tokens
    ):
        raise ValueError("requested speculative depth exceeds the provider maximum")
    if type(dynamic_speculation) is not bool:
        raise TypeError("dynamic_speculation must be bool")
    if dynamic_speculation:
        if not num_speculative_tokens or not (
            num_speculative_tokens
            <= dynamic_max_speculative_tokens
            <= max_speculative_tokens
        ):
            raise ValueError("dynamic speculative bounds exceed the provider maximum")
    elif dynamic_max_speculative_tokens not in (0, num_speculative_tokens):
        raise ValueError("fixed speculative depth cannot carry a different dynamic maximum")
    if num_speculative_tokens and pp_size != 1:
        raise ValueError("MTP performance serving currently requires PP1")
    if type(use_prefix_cache) is not bool:
        raise TypeError("use_prefix_cache must be bool")
    if prefill_chunk_tokens is not None and (
        type(prefill_chunk_tokens) is not int or prefill_chunk_tokens <= 0
    ):
        raise ValueError("prefill_chunk_tokens must be a positive integer")
    if is_moe:
        if tp_size != 4 or pp_size not in (1, 2) or tp_size % ep_size:
            raise ValueError("Qwen3.5 MoE requires TP4 x PP1/2 with EP dividing TP")
        parallel_plan = plan_qwen3_5_moe_weights(
            config,
            weight_plan,
            topology,
            include_mtp=num_speculative_tokens > 0,
        )
        sealed = seal_qwen3_5_moe_distributed_reference(config, parallel_plan)
    else:
        parallel_plan = (
            plan_qwen3_5_tp2_pp2_weights(config, weight_plan, topology)
            if (tp_size, pp_size) == (2, 2)
            else plan_qwen3_5_dense_weights(
                config,
                weight_plan,
                topology,
                include_mtp=num_speculative_tokens > 0,
            )
        )
        sealed = (
            seal_qwen3_5_target_mtp_distributed_reference(config, parallel_plan)
            if num_speculative_tokens
            else seal_qwen3_5_distributed_reference(config, parallel_plan)
        )
    return _Admission(
        config,
        inventory,
        weight_plan,
        parallel_plan,
        sealed,
        resolved_max_sequence_tokens,
        max_sequence_tokens is None,
        max_running_requests,
        num_speculative_tokens,
        max_speculative_tokens,
        dynamic_max_speculative_tokens,
        dynamic_speculation,
        use_prefix_cache,
        resolved_state_dtype,
        prefill_chunk_tokens,
    )
