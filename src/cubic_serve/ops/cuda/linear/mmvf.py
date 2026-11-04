"""Fixed-address BF16 linear primitives with row-invariant reduction plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor
from triton.compiler.compiler import CompiledKernel  # type: ignore[import-untyped]

from cubic_serve.ops.cuda.compiled_artifact import (
    CompiledArtifactIdentity,
    LaunchReceipt,
    TensorBinding,
    bind_tensor,
    project_triton_artifact,
    validate_launch_receipt,
)
from cubic_serve.ops.cuda.launch_cost import (
    LinearTile,
    MatmulGeometry,
    current_linear_cost,
    linear_geometries,
)
from cubic_serve.ops.strategies import linear_parameters, linear_row_tile

BF16_MMVF_OPERATOR_ID = "cubic.cuda.bf16-mmvf-tiled.triton.v4"
BF16_MMVF_REDUCTION_ABI = {
    "id": "cubic.bf16-mmvf-tiled.fp32-reduction.v4",
    "m": "positive",
    "active_rows": "device-int32-active-prefix",
    "program": "one-output-channel",
    "block_k": "next-power-of-two",
    "num_warps": 8,
    "operands": "bf16",
    "accumulator": "fp32-reassociation",
    "output": "bf16-rne",
    "padding": "inactive-rows-write-bf16-positive-zero",
}
BF16_FIXED_TILE_OPERATOR_ID = "cubic.cuda.bf16-fixed-tile.triton.v1"
BF16_FIXED_TILE_REDUCTION_ABI = {
    "id": "cubic.bf16-fixed-tile.fp32-reduction.v1",
    "m": "positive",
    "active_rows": "device-int32-active-prefix",
    "program": "startup-bound-output-tile",
    "block_k": "startup-bound-fixed-reduction-quantum",
    "num_warps": "startup-bound-launch-signature",
    "operands": "bf16",
    "accumulator": "ordered-k-blocks-bf16-dot-fp32",
    "output": "bf16-rne",
    "padding": "inactive-rows-write-bf16-positive-zero",
}

BF16MMVFTensorBinding = TensorBinding
BF16MMVFArtifactIdentity = CompiledArtifactIdentity
BF16MMVFLaunchReceipt = LaunchReceipt


@triton.jit  # type: ignore[untyped-decorator]
def _bf16_mmvf_small_m(  # type: ignore[no-untyped-def]
    x,
    weight,
    output,
    active_rows,
    N: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    active = tl.load(active_rows)
    valid_active = (active >= 1) & (active <= M)
    valid_k = (offsets < K) & valid_active
    values = tl.load(weight + row * K + offsets, mask=valid_k, other=0.0).to(
        tl.float32
    )
    # Keep short-K reductions one-dimensional: vector layout inference may
    # otherwise redistribute K across warps when physical M changes.
    if BLOCK_K < 2048:
        for lane in tl.static_range(BLOCK_M):
            logical_row = tl.program_id(1) * BLOCK_M + lane
            owned = logical_row < active
            vector = tl.load(
                x + logical_row * K + offsets,
                mask=valid_k & owned,
                other=0.0,
            ).to(tl.float32)
            result = tl.sum(values * vector, axis=0)
            tl.store(
                output + logical_row * N + row,
                tl.where(owned, result, 0.0),
                mask=valid_active & (logical_row < M),
            )
    else:
        logical_rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        owned = logical_rows < active
        vectors = tl.load(
            x + logical_rows[:, None] * K + offsets[None, :],
            mask=valid_k[None, :] & owned[:, None],
            other=0.0,
        ).to(tl.float32)
        results = tl.sum(values[None, :] * vectors, axis=1)
        tl.store(
            output + logical_rows * N + row,
            tl.where(owned, results, 0.0),
            mask=valid_active & (logical_rows < M),
        )



@triton.jit
def _bf16_fixed_tile_at(x, weight, output, active_rows, row_tile, column_tile,
                        N: tl.constexpr, K: tl.constexpr, M: tl.constexpr,
                        BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = column_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = tl.arange(0, BLOCK_K)
    active = tl.load(active_rows)
    valid = (active > 0) & (active <= M)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for block in range(tl.cdiv(K, BLOCK_K)):
        kk = block * BLOCK_K + offsets
        left = tl.load(x + rows.to(tl.int64)[:, None] * K + kk[None, :],
                       mask=(rows[:, None] < active) & valid & (kk[None, :] < K), other=0.0)
        right = tl.load(weight + columns.to(tl.int64)[None, :] * K + kk[:, None],
                        mask=(columns[None, :] < N) & (kk[:, None] < K), other=0.0)
        accumulator = tl.dot(left, right, accumulator)
    tl.store(output + rows.to(tl.int64)[:, None] * N + columns[None, :],
             tl.where(rows[:, None] < active, accumulator, 0.0),
             mask=(rows[:, None] < M) & (columns[None, :] < N) & valid)


@triton.jit
def _bf16_fixed_tile(x, weight, output, active_rows,
                     N: tl.constexpr, K: tl.constexpr, M: tl.constexpr,
                     BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    _bf16_fixed_tile_at(
        x, weight, output, active_rows, tl.program_id(1), tl.program_id(0),
        N, K, M, BLOCK_K, BLOCK_M, BLOCK_N,
    )


@dataclass(frozen=True, slots=True)
class BF16MMVFMeta:
    device: torch.device
    x: BF16MMVFTensorBinding
    weight: BF16MMVFTensorBinding
    output: BF16MMVFTensorBinding
    active_rows: BF16MMVFTensorBinding


class BF16MMVFPrimitive:
    """Reuse each weight row across a bounded active prefix of vectors."""

    __slots__ = (
        "_artifact",
        "_bindings",
        "_block_k",
        "_grid",
        "_last_bindings",
        "_last_stream",
        "_operator_id",
        "_ordinal",
        "_receipt_token",
        "_signature",
        "meta",
    )

    def __init__(
        self,
        x: Tensor,
        weight: Tensor,
        output: Tensor,
        active_rows: Tensor,
        *,
        reduction: Literal["canonical", "fixed_tile"] = "canonical",
    ) -> None:
        if reduction not in ("canonical", "fixed_tile"):
            raise ValueError("unknown BF16 linear reduction plan")
        m, n, k = _validate_layout(x, weight, output, active_rows, inspect_active=True)
        self.meta = BF16MMVFMeta(
            x.device,
            *(bind_tensor(item) for item in (x, weight, output, active_rows)),
        )
        self._bindings = (
            self.meta.x,
            self.meta.weight,
            self.meta.output,
            self.meta.active_rows,
        )
        fixed_tile = reduction == "fixed_tile"
        parameters = linear_parameters(n=n, k=k)
        self._block_k = parameters.fixed_block_k if fixed_tile else 1 << (k - 1).bit_length()
        block_m = (
            linear_row_tile(m, parameters.fixed_block_m, parameters.fixed_max_block_m) if fixed_tile
            else min(triton.next_power_of_2(m), parameters.canonical_max_rows)
        )
        block_n = parameters.fixed_block_n if fixed_tile else 1
        warps = parameters.fixed_warps if fixed_tile else parameters.canonical_warps
        stages = parameters.fixed_stages
        kernel = _bf16_fixed_tile if fixed_tile else _bf16_mmvf_small_m
        self._grid = (triton.cdiv(n, block_n) if fixed_tile else n,
                      triton.cdiv(m, block_m), 1)
        self._operator_id = (
            BF16_FIXED_TILE_OPERATOR_ID if fixed_tile else BF16_MMVF_OPERATOR_ID
        )
        with torch.cuda.device(x.device):
            artifact = kernel.warmup(
                x, weight, output, active_rows, grid=self._grid,
                N=n, K=k, M=m, BLOCK_K=self._block_k, BLOCK_M=block_m,
                num_warps=warps,
                **(
                    {"BLOCK_N": block_n, "num_stages": parameters.fixed_stages}
                    if fixed_tile else {}
                ),
            )
            if not isinstance(artifact, CompiledKernel):
                raise RuntimeError("Triton did not return a compiled BF16 linear artifact")
            artifact._init_handles()
            cost = current_linear_cost() if fixed_tile else None
            if cost is not None:
                baseline = MatmulGeometry(block_m, block_n, self._block_k, warps, stages)
                geometries = linear_geometries(
                    cost.device, rows=m, columns=(n,), reduction=k,
                    element_bytes=x.element_size(), communication=cost.communication,
                    baseline=baseline, parameters=parameters,
                )
                tiles = [LinearTile(artifact, self._grid, block_m, block_n, baseline)]
                for point in geometries[1:]:
                    bm, bn = point.block_m, point.block_n
                    grid = (triton.cdiv(n, bn), triton.cdiv(m, bm), 1)
                    candidate = kernel.warmup(
                        x, weight, output, active_rows, grid=grid,
                        N=n, K=k, M=m, BLOCK_K=self._block_k, BLOCK_M=bm, BLOCK_N=bn,
                        num_warps=point.warps, num_stages=point.stages,
                    )
                    candidate._init_handles()
                    tiles.append(LinearTile(candidate, grid, bm, bn, point))
                selected = cost.select(tiles, x, (weight,), grouped=False)
                artifact, self._grid = selected.kernel, selected.grid
                block_m, block_n = selected.block_m, selected.block_n
                assert selected.geometry is not None
                warps, stages = selected.geometry.warps, selected.geometry.stages
        self._signature = (
            f"m={m},n={n},k={k},block_k={self._block_k},block_m={block_m},"
            f"num_stages={stages if fixed_tile else 3},block_n={block_n},"
            f"reduction={reduction},active_rows=device-int32,num_warps={warps},acc=fp32,out=bf16"
        )
        self._artifact, self._receipt_token, self._ordinal = artifact, object(), 0
        self._last_stream: int | None = None
        self._last_bindings = self._bindings

    @property
    def grid(self) -> tuple[int, int, int]:
        return self._grid

    @property
    def launch_signature(self) -> str:
        return self._signature

    @property
    def row_tile_count(self) -> int:
        """Physical row CTAs declared by this binding, independent of dtype policy."""
        return self._grid[1]

    def project_artifact(self) -> BF16MMVFArtifactIdentity:
        return project_bf16_mmvf_artifact(self)

    def validate_operands(
        self,
        x: Tensor,
        weight: Tensor,
        output: Tensor,
        active_rows: Tensor,
    ) -> None:
        validate_bf16_mmvf_operands(self, x, weight, output, active_rows)

    def validate_bound_operands(
        self,
        x: Tensor,
        weight: Tensor,
        output: Tensor,
        active_rows: Tensor,
    ) -> None:
        validate_bf16_mmvf_bound_operands(self, x, weight, output, active_rows)

    def launch(
        self,
        x: Tensor,
        weight: Tensor,
        output: Tensor,
        active_rows: Tensor,
    ) -> BF16MMVFLaunchReceipt:
        return launch_bf16_mmvf(self, x, weight, output, active_rows)

    def validate_receipt(self, receipt: BF16MMVFLaunchReceipt) -> None:
        validate_bf16_mmvf_receipt(self, receipt)


def _require_mmvf_primitive(primitive: object) -> BF16MMVFPrimitive:
    if type(primitive) is not BF16MMVFPrimitive:
        raise TypeError("exact BF16 MMVF primitive is required")
    return primitive


def project_bf16_mmvf_artifact(
    primitive: BF16MMVFPrimitive,
) -> BF16MMVFArtifactIdentity:
    primitive = _require_mmvf_primitive(primitive)
    return project_triton_artifact(
        primitive._artifact,
        operator_id=primitive._operator_id,
        device=primitive.meta.device,
        launch_signature=primitive._signature,
        grid=primitive._grid,
    )


def validate_bf16_mmvf_operands(
    primitive: BF16MMVFPrimitive,
    x: Tensor,
    weight: Tensor,
    output: Tensor,
    active_rows: Tensor,
) -> None:
    primitive = _require_mmvf_primitive(primitive)
    _validate_layout(x, weight, output, active_rows, inspect_active=False)
    bindings = tuple(bind_tensor(item) for item in (x, weight, output, active_rows))
    if any(
        (item.shape, item.stride, item.dtype, item.device, item.pointer % 16)
        != (base.shape, base.stride, base.dtype, base.device, base.pointer % 16)
        for item, base in zip(bindings, primitive._bindings, strict=True)
    ):
        raise ValueError("BF16 MMVF operands differ from the compiled specialization")


def validate_bf16_mmvf_bound_operands(
    primitive: BF16MMVFPrimitive,
    x: Tensor,
    weight: Tensor,
    output: Tensor,
    active_rows: Tensor,
) -> None:
    primitive = _require_mmvf_primitive(primitive)
    validate_bf16_mmvf_operands(primitive, x, weight, output, active_rows)
    if tuple(bind_tensor(item) for item in (x, weight, output, active_rows)) != (
        primitive._bindings
    ):
        raise ValueError("BF16 MMVF graph launch requires its exact bound tensors")


def launch_bf16_mmvf(
    primitive: BF16MMVFPrimitive,
    x: Tensor,
    weight: Tensor,
    output: Tensor,
    active_rows: Tensor,
) -> BF16MMVFLaunchReceipt:
    primitive = _require_mmvf_primitive(primitive)
    tensors = x, weight, output, active_rows
    # One retained compiled specialization serves every same-layout occurrence.
    # CUDA Graph capture fixes the concrete pointers for each launch node; the
    # stricter exact-binding check remains available to qualification callers.
    validate_bf16_mmvf_operands(primitive, *tensors)
    with torch.cuda.device(primitive.meta.device):
        stream = int(torch.cuda.current_stream(primitive.meta.device).cuda_stream)
        primitive._artifact[primitive._grid](*tensors, stream=stream)
    primitive._ordinal += 1
    primitive._last_bindings = (
        bind_tensor(x),
        bind_tensor(weight),
        bind_tensor(output),
        bind_tensor(active_rows),
    )
    primitive._last_stream = stream
    return LaunchReceipt(
        id(primitive._artifact),
        primitive._last_bindings,
        primitive._signature,
        primitive._grid,
        stream,
        primitive._ordinal,
        primitive._receipt_token,
    )


def validate_bf16_mmvf_receipt(
    primitive: BF16MMVFPrimitive,
    receipt: object,
) -> None:
    primitive = _require_mmvf_primitive(primitive)
    if primitive._last_stream is None:
        raise ValueError("BF16 MMVF primitive has no completed launch receipt")
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
            f"BF16 MMVF launch receipt does not close this primitive launch: {error}"
        ) from error


def _validate_layout(
    x: Tensor,
    weight: Tensor,
    output: Tensor,
    active_rows: Tensor,
    *,
    inspect_active: bool,
    allowed_dtypes: tuple[torch.dtype, ...] = (torch.bfloat16,),
) -> tuple[int, int, int]:
    tensors = x, weight, output, active_rows
    if any(not isinstance(tensor, Tensor) for tensor in tensors):
        raise TypeError("BF16 MMVF operands must be tensors")
    if (
        x.device.type != "cuda"
        or any(tensor.device != x.device for tensor in tensors[1:])
        or x.dtype not in allowed_dtypes
        or weight.dtype is not x.dtype
        or output.dtype is not x.dtype
        or active_rows.dtype is not torch.int32
        or x.ndim != 2
        or x.shape[0] < 1
        or weight.ndim != 2
        or output.ndim != 2
        or active_rows.shape != (1,)
        or any(not tensor.is_contiguous() for tensor in tensors)
        or min(x.shape) <= 0
        or weight.shape[1] != x.shape[1]
        or output.shape != (x.shape[0], weight.shape[0])
        or len({tensor.untyped_storage().data_ptr() for tensor in tensors}) != 4
    ):
        raise ValueError(
            "expected non-aliasing CUDA x[M,K], weight[N,K], output[M,N] "
            "with a permitted matching dtype, "
            "for positive M, "
            "and device int32 active_rows[1]"
        )
    if inspect_active and not 0 < int(active_rows.item()) <= x.shape[0]:
        raise ValueError("BF16 MMVF active_rows must lie within physical M")
    return x.shape[0], weight.shape[0], weight.shape[1]


__all__ = (
    "BF16_MMVF_OPERATOR_ID",
    "BF16_MMVF_REDUCTION_ABI",
    "BF16MMVFArtifactIdentity",
    "BF16MMVFLaunchReceipt",
    "BF16MMVFMeta",
    "BF16MMVFPrimitive",
    "BF16MMVFTensorBinding",
    "launch_bf16_mmvf",
    "project_bf16_mmvf_artifact",
    "validate_bf16_mmvf_bound_operands",
    "validate_bf16_mmvf_operands",
    "validate_bf16_mmvf_receipt",
)
