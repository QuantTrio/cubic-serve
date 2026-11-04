"""Content-addressed vendor BF16 GEMM dispatch and fixed launch ABI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch
from torch import Tensor

from cubic_serve.ops.cuda.compiled_artifact import (
    CompiledArtifactIdentity,
    LaunchReceipt,
    TensorBinding,
    bind_tensor,
    project_torch_cublas_artifact,
    validate_launch_receipt,
    validate_torch_cublas_runtime,
)

BF16_LINEAR_OPERATOR_ID = "cubic.cuda.bf16-linear.torch-cublas.v1"
BF16_LINEAR_REDUCTION_ABI = {
    "id": "cubic.bf16-linear.fp32-reduction.v1",
    "operands": "bf16",
    "accumulator": "fp32-reassociation",
    "output": "bf16-rne",
    "backend": "torch-cublas",
}


BF16LinearTensorBinding = TensorBinding


@dataclass(frozen=True, slots=True)
class BF16LinearMeta:
    device: torch.device
    x: BF16LinearTensorBinding
    weight: BF16LinearTensorBinding
    output: BF16LinearTensorBinding


BF16LinearArtifactIdentity = CompiledArtifactIdentity
BF16LinearLaunchReceipt = LaunchReceipt


class BF16LinearPrimitive:
    __slots__ = (
        "_artifact",
        "_grid",
        "_last_bindings",
        "_last_stream",
        "_ordinal",
        "_receipt_token",
        "_signature",
        "meta",
    )

    def __init__(self, x: Tensor, weight: Tensor, output: Tensor) -> None:
        _validate_layout(x, weight, output)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.meta = BF16LinearMeta(x.device, *(bind_tensor(item) for item in (x, weight, output)))
        m, k = self.meta.x.shape
        n = self.meta.weight.shape[0]
        self._grid = (1, 1, 1)  # one opaque ATen vendor dispatch, not a claimed CUDA grid
        self._signature = f"m={m},n={n},k={k},backend=torch-cublas,acc=fp32,out=bf16"
        with torch.cuda.device(x.device):
            torch.mm(x, weight.t(), out=output)
            artifact = project_torch_cublas_artifact(
                x, weight, output, operator_id=BF16_LINEAR_OPERATOR_ID,
                launch_signature=self._signature, grid=self._grid,
            )
        self._artifact, self._receipt_token, self._ordinal = artifact, object(), 0
        self._last_stream: int | None = None
        self._last_bindings = (self.meta.x, self.meta.weight, self.meta.output)

    @property
    def launch_signature(self) -> str:
        return self._signature

    @property
    def grid(self) -> tuple[int, int, int]:
        return self._grid

    def project_artifact(self) -> BF16LinearArtifactIdentity:
        return project_bf16_linear_artifact(self)

    def validate_operands(self, x: Tensor, weight: Tensor, output: Tensor) -> None:
        validate_bf16_linear_operands(self, x, weight, output)

    def validate_bound_operands(self, x: Tensor, weight: Tensor, output: Tensor) -> None:
        validate_bf16_linear_bound_operands(self, x, weight, output)

    def launch(self, x: Tensor, weight: Tensor, output: Tensor) -> BF16LinearLaunchReceipt:
        return launch_bf16_linear(self, x, weight, output)

    def validate_receipt(self, receipt: BF16LinearLaunchReceipt) -> None:
        validate_bf16_linear_receipt(self, receipt)


def _require_linear_primitive(primitive: object) -> BF16LinearPrimitive:
    if type(primitive) is not BF16LinearPrimitive:
        raise TypeError("exact BF16 linear primitive is required")
    return primitive


def project_bf16_linear_artifact(
    primitive: BF16LinearPrimitive,
) -> BF16LinearArtifactIdentity:
    primitive = _require_linear_primitive(primitive)
    observed = replace(
        primitive._artifact,
        launch_signature=primitive._signature,
        grid=primitive._grid,
    )
    validate_torch_cublas_runtime(observed)
    return observed


def validate_bf16_linear_operands(
    primitive: BF16LinearPrimitive, x: Tensor, weight: Tensor, output: Tensor,
) -> None:
    primitive = _require_linear_primitive(primitive)
    _validate_layout(x, weight, output)
    bindings = tuple(bind_tensor(item) for item in (x, weight, output))
    expected = (primitive.meta.x, primitive.meta.weight, primitive.meta.output)
    if any(
        (item.shape, item.stride, item.dtype, item.device, item.pointer % 16)
        != (base.shape, base.stride, base.dtype, base.device, base.pointer % 16)
        for item, base in zip(bindings, expected, strict=True)
    ):
        raise ValueError("BF16 linear operands differ from the compiled specialization")


def validate_bf16_linear_bound_operands(
    primitive: BF16LinearPrimitive, x: Tensor, weight: Tensor, output: Tensor,
) -> None:
    primitive = _require_linear_primitive(primitive)
    validate_bf16_linear_operands(primitive, x, weight, output)
    if tuple(bind_tensor(item) for item in (x, weight, output)) != (
        primitive.meta.x,
        primitive.meta.weight,
        primitive.meta.output,
    ):
        raise ValueError("BF16 linear qualification requires its exact bound tensors")


def launch_bf16_linear(
    primitive: BF16LinearPrimitive, x: Tensor, weight: Tensor, output: Tensor,
) -> BF16LinearLaunchReceipt:
    primitive = _require_linear_primitive(primitive)
    validate_bf16_linear_operands(primitive, x, weight, output)
    validate_torch_cublas_runtime(primitive._artifact)
    with torch.cuda.device(primitive.meta.device):
        stream = int(torch.cuda.current_stream(primitive.meta.device).cuda_stream)
        torch.mm(x, weight.t(), out=output)
    primitive._ordinal += 1
    primitive._last_bindings = bind_tensor(x), bind_tensor(weight), bind_tensor(output)
    primitive._last_stream = stream
    return LaunchReceipt(
        id(primitive._artifact), primitive._last_bindings, primitive._signature,
        primitive._grid, stream, primitive._ordinal, primitive._receipt_token,
    )


def validate_bf16_linear_receipt(
    primitive: BF16LinearPrimitive,
    receipt: object,
) -> None:
    primitive = _require_linear_primitive(primitive)
    if primitive._last_stream is None:
        raise ValueError("linear primitive has no completed launch receipt")
    try:
        validate_launch_receipt(
            receipt,
            artifact_object_id=id(primitive._artifact),
            bindings=primitive._last_bindings,
            launch_signature=primitive._signature,
            grid=primitive._grid,
            stream=primitive._last_stream,
            ordinal=primitive._ordinal,
            primitive_token=primitive._receipt_token,
        )
    except ValueError as error:
        raise ValueError(
            f"linear launch receipt does not close this primitive launch: {error}"
        ) from error


def bf16_linear_fingerprint_payload(x: Tensor, weight: Tensor, output: Tensor) -> dict[str, object]:
    primitive = BF16LinearPrimitive(x, weight, output)
    artifact = project_bf16_linear_artifact(primitive)
    return {"artifact": asdict(artifact), "artifact_digest": artifact.artifact_digest}


def _validate_layout(x: Tensor, weight: Tensor, output: Tensor) -> None:
    if any(not isinstance(tensor, Tensor) for tensor in (x, weight, output)):
        raise TypeError("BF16 linear operands must be tensors")
    if (
        x.device.type != "cuda"
        or weight.device != x.device
        or output.device != x.device
        or any(tensor.dtype is not torch.bfloat16 for tensor in (x, weight, output))
        or any(tensor.ndim != 2 or not tensor.is_contiguous() for tensor in (x, weight, output))
        or min(x.shape) <= 0
        or weight.shape[1] != x.shape[1]
        or output.shape != (x.shape[0], weight.shape[0])
        or len({tensor.untyped_storage().data_ptr() for tensor in (x, weight, output)}) != 3
    ):
        raise ValueError("expected non-aliasing contiguous CUDA BF16 [M,K]@[N,K]")


__all__ = (
    "BF16_LINEAR_OPERATOR_ID", "BF16_LINEAR_REDUCTION_ABI", "BF16LinearArtifactIdentity",
    "BF16LinearLaunchReceipt", "BF16LinearMeta", "BF16LinearPrimitive",
    "BF16LinearTensorBinding", "bf16_linear_fingerprint_payload", "launch_bf16_linear",
    "project_bf16_linear_artifact", "validate_bf16_linear_bound_operands",
    "validate_bf16_linear_operands", "validate_bf16_linear_receipt",
)
