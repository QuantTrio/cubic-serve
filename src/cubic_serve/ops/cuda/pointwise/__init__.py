"""Graph-composable caller-owned CUDA pointwise primitives."""

from cubic_serve.ops.cuda.pointwise.qwen_decode import (
    BF16PackedGDNProjectionSplitPrimitive,
    BF16PackedSwiGLUPrimitive,
    BF16ResidualAddPrimitive,
    BF16SigmoidOutputGatePrimitive,
    BF16SwiGLUPrimitive,
    GatedRMSNormPrimitive,
    PackedGDNProjectionSplitPrimitive,
    PackedSwiGLUPrimitive,
    QwenAttentionQGateSplitPrimitive,
    ResidualAddPrimitive,
    SigmoidOutputGatePrimitive,
    SwiGLUPrimitive,
)

__all__ = (
    "BF16PackedGDNProjectionSplitPrimitive",
    "BF16PackedSwiGLUPrimitive",
    "BF16ResidualAddPrimitive",
    "BF16SigmoidOutputGatePrimitive",
    "BF16SwiGLUPrimitive",
    "GatedRMSNormPrimitive",
    "PackedGDNProjectionSplitPrimitive",
    "PackedSwiGLUPrimitive",
    "QwenAttentionQGateSplitPrimitive",
    "ResidualAddPrimitive",
    "SigmoidOutputGatePrimitive",
    "SwiGLUPrimitive",
)
