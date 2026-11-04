"""Model-owned dense hybrid state geometry for the original sealed runtime.

The engine keeps arena capacities, ownership, transactions and allocation.
These declarations retain the original BF16 convolution/FP32 recurrent state.
"""

import torch

from cubic_serve.models.qwen3_5.model import (
    CONV_STATE_ID,
    RECURRENT_STATE_ID,
    AttentionKind,
    Qwen3_5Config,
    Qwen3_5GraphVariant,
)
from cubic_serve.runtime import cache as rc


def dense_decode_state_layout(
    config: Qwen3_5Config, *, max_pages: int, block_size: int = 16,
) -> tuple[rc.KVLayout, tuple[rc.StateSlotSpec, ...], tuple[int, ...]]:
    if config.graph.variant is not Qwen3_5GraphVariant.DENSE or max_pages <= 0:
        raise ValueError("decode state requires a dense qwen3_5 graph and positive page capacity")
    text, gdn = config.graph.text, config.graph.text.gated_delta_net
    layers = {
        kind: tuple(i for i, actual in enumerate(text.layer_schedule) if actual is kind)
        for kind in AttentionKind
    }
    linear, full = layers[AttentionKind.GATED_DELTA_NET], layers[AttentionKind.FULL_ATTENTION]
    if not linear or not full:
        raise ValueError("dense qwen3_5 performance state requires a hybrid schedule")
    packed = gdn.key_heads * gdn.key_head_dim * 2 + gdn.value_heads * gdn.value_head_dim
    conv_shape = (len(linear), packed, gdn.convolution_kernel_size - 1)
    recurrent_shape = (len(linear), gdn.value_heads, gdn.value_head_dim, gdn.key_head_dim)
    states = (
        rc.StateSlotSpec(CONV_STATE_ID, torch.bfloat16, conv_shape),
        rc.StateSlotSpec(RECURRENT_STATE_ID, torch.float32, recurrent_shape),
    )
    layout = rc.KVLayout(
        full, block_size, text.full_attention.key_value_heads, text.full_attention.head_dim
    )
    return layout, states, linear
