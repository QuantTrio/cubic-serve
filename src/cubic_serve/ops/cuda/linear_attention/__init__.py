"""Graph-composable linear-attention CUDA primitives."""

from cubic_serve.ops.cuda.linear_attention.gated_delta_decode import (
    INDEX_OWNERSHIP_CONTRACT,
    CausalConv1DDecodePrimitive,
    GatedDeltaDecodePrimitive,
)

__all__ = [
    "INDEX_OWNERSHIP_CONTRACT",
    "CausalConv1DDecodePrimitive",
    "GatedDeltaDecodePrimitive",
]
