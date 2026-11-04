"""BF16 linear CUDA primitive and qualification candidate."""

from typing_extensions import TypeAliasType

from typing import Literal

from torch import Tensor

from cubic_serve.ops.cuda.linear.candidate import BF16LinearGraphCandidate
from cubic_serve.ops.cuda.linear.kernel import (
    BF16_LINEAR_OPERATOR_ID,
    BF16_LINEAR_REDUCTION_ABI,
    BF16LinearArtifactIdentity,
    BF16LinearLaunchReceipt,
    BF16LinearMeta,
    BF16LinearPrimitive,
    BF16LinearTensorBinding,
    bf16_linear_fingerprint_payload,
)
from cubic_serve.ops.cuda.linear.mmvf import (
    BF16_MMVF_OPERATOR_ID,
    BF16_MMVF_REDUCTION_ABI,
    BF16MMVFArtifactIdentity,
    BF16MMVFLaunchReceipt,
    BF16MMVFMeta,
    BF16MMVFPrimitive,
    BF16MMVFTensorBinding,
)

BF16LinearKernel = TypeAliasType("BF16LinearKernel", BF16LinearPrimitive | BF16MMVFPrimitive)


def bind_bf16_linear_kernel(
    x: Tensor,
    weight: Tensor,
    output: Tensor,
    active_rows: Tensor,
    *,
    vendor: BF16LinearPrimitive | None = None,
    reduction: Literal["canonical", "fixed_tile"] = "canonical",
) -> BF16LinearKernel:
    """Bind a reduction plan that remains identical across physical row buckets."""

    return BF16MMVFPrimitive(x, weight, output, active_rows, reduction=reduction)



def launch_bf16_linear_kernel(
    kernel: BF16LinearKernel,
    x: Tensor,
    weight: Tensor,
    output: Tensor,
    active_rows: Tensor,
) -> None:
    """Launch one bound BF16 linear kernel through a uniform operator surface."""

    if isinstance(kernel, BF16MMVFPrimitive):
        kernel.launch(x, weight, output, active_rows)
    else:
        kernel.launch(x, weight, output)

__all__ = [
    "BF16_LINEAR_OPERATOR_ID",
    "BF16_LINEAR_REDUCTION_ABI",
    "BF16_MMVF_OPERATOR_ID",
    "BF16_MMVF_REDUCTION_ABI",
    "BF16LinearArtifactIdentity",
    "BF16LinearGraphCandidate",
    "BF16LinearKernel",
    "BF16LinearLaunchReceipt",
    "BF16LinearMeta",
    "BF16LinearPrimitive",
    "BF16LinearTensorBinding",
    "BF16MMVFArtifactIdentity",
    "BF16MMVFLaunchReceipt",
    "BF16MMVFMeta",
    "BF16MMVFPrimitive",
    "BF16MMVFTensorBinding",
    "bf16_linear_fingerprint_payload",
    "bind_bf16_linear_kernel",
    "launch_bf16_linear_kernel",
]
