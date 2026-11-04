"""Model-owned startup bindings to the existing named-state text stage."""

from math import lcm

from cubic_serve.models.adapter import TextGraphDefinition


def text_graph_definition(config, topology, *, block_size, recurrent_state_dtype, definition=None):
    import torch

    from cubic_serve.executor.text_serving import FlashNextDecodeStage

    if recurrent_state_dtype == "auto":
        recurrent_state_dtype = config.text.gated_delta_net.recurrent_state_dtype
    return TextGraphDefinition(
        stage_type=FlashNextDecodeStage,
        named_state_fields=True,
        stage_options={"recurrent_state_dtype": getattr(torch, recurrent_state_dtype)},
        required_operations=("linear", "moe"),
        block_size_tokens=lcm(block_size, config.text.qsa.compression_ratio * topology.tp_size),
        recurrent_state_dtype=recurrent_state_dtype,
        state_audit_steps=2 * config.text.qsa.compression_ratio,
    )
