"""Packed CUDA operators with explicit input dtype and FP32 accumulation.

No full dequantized matrix is allocated. Weight-only operands round to the
activation dtype inside the GEMM tile; FP8 and NVFP4 preserve their own scale
boundaries. Kernel selection is by encoding, never model, device or concurrency.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor
from triton.compiler.compiler import CompiledKernel  # type: ignore[import-untyped]
from triton.runtime.jit import JITFunction  # type: ignore[import-untyped]

from cubic_serve.ops.cuda.compiled_artifact import (
    CompiledArtifactIdentity,
    LaunchReceipt,
    TensorBinding,
    bind_tensor,
    project_triton_artifact,
)
from cubic_serve.ops.cuda.native import NativeLaunch


def concatenate_to_device(
    tensors: tuple[Tensor, ...], device: torch.device, dim: int = 0,
) -> Tensor:
    """Copy host rows directly into their final, ordered device allocation.

    Finish this transfer boundary before releasing any source owner. Nonleading
    concatenation and mixed dtype/device inputs retain Torch's ordinary rules.
    This transports encoded bytes; it does not interpret quantization metadata.
    """
    device = torch.device(device)
    if (len(tensors) == 1 and tensors[0].ndim > 0
            and tensors[0].device.type == "cpu" and device.type == "cuda"):
        return tensors[0].contiguous().to(device)
    if (not tensors or device.type != "cuda" or tensors[0].ndim == 0
            or dim not in (0, -tensors[0].ndim)
            or any(t.device.type != "cpu" or t.layout != torch.strided
                   or t.dtype != tensors[0].dtype
                   or t.ndim != tensors[0].ndim or t.shape[1:] != tensors[0].shape[1:]
                   for t in tensors)):
        return torch.cat(tensors, dim=dim).to(device)
    shape = (sum(t.shape[0] for t in tensors), *tensors[0].shape[1:])
    output = torch.empty(shape, dtype=tensors[0].dtype, device=device)
    offset = 0
    try:
        for tensor in tensors:
            output[offset:offset + tensor.shape[0]].copy_(tensor, non_blocking=True)
            offset += tensor.shape[0]
    finally:
        torch.cuda.current_stream(device).synchronize()
    return output


def constant_identity(value: Any) -> Any:
    """Describe bound JIT callbacks by source closure, never process addresses."""
    if isinstance(value, JITFunction):
        return (value.__module__, value.__name__, value.cache_key)
    if isinstance(value, tuple):
        return tuple(constant_identity(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class _PackedLaunch:
    kernel: Any
    tensors: tuple[Tensor, ...]
    bindings: tuple[TensorBinding, ...]
    grid: tuple[int, int, int]
    signature: str
    operator_id: str
    argument_groups: tuple[int | tuple[int, ...], ...] | None = None

    @property
    def arguments(self) -> tuple[Any, ...]:
        if self.argument_groups is None:
            return self.tensors
        return tuple(
            self.tensors[group] if isinstance(group, int)
            else tuple(self.tensors[index] for index in group)
            for group in self.argument_groups
        )

    @classmethod
    def compile(
        cls,
        kernel: Any,
        tensors: tuple[Tensor | tuple[Tensor, ...], ...],
        constants: tuple[Any, ...],
        grid: tuple[int, ...],
        options: dict[str, Any],
    ) -> _PackedLaunch:
        if not 1 <= len(grid) <= 3:
            raise ValueError("packed launch requires a one-to-three-dimensional grid")
        flat: list[Tensor] = []
        groups: list[int | tuple[int, ...]] = []
        for argument in tensors:
            if isinstance(argument, Tensor):
                groups.append(len(flat))
                flat.append(argument)
            elif isinstance(argument, tuple) and argument and all(
                isinstance(tensor, Tensor) for tensor in argument
            ):
                groups.append(tuple(range(len(flat), len(flat) + len(argument))))
                flat.extend(argument)
            else:
                raise TypeError("packed arguments require tensors or nonempty tensor tuples")
        if not flat:
            raise ValueError("packed launch requires tensor operands")
        argument_groups = tuple(groups) if any(isinstance(g, tuple) for g in groups) else None
        physical_grid = (grid[0], grid[1] if len(grid) > 1 else 1, grid[2] if len(grid) > 2 else 1)
        with torch.cuda.device(flat[0].device):
            compiled = kernel.warmup(*tensors, *constants, grid=physical_grid, **options)
            if not isinstance(compiled, CompiledKernel):
                raise RuntimeError("packed launch requires a retained compiled kernel")
            compiled._init_handles()
        return cls(
            compiled,
            tuple(flat),
            tuple(bind_tensor(t) for t in flat),
            physical_grid,
            f"{kernel.__name__}:{constant_identity(constants)}:{sorted(options.items())}"
            + (f":argument_groups={argument_groups}" if argument_groups is not None else ""),
            f"cubic.cuda.packed.{kernel.__name__}.v1",
            argument_groups,
        )


class PackedOperatorBinding:
    """Startup-bound native launches with owned static activation scratch.

    This is a primitive binding, not a graph qualification or replay lease.
    The graph executor must qualify and exclusively own the entire model bucket.
    """

    def __init__(
        self,
        output: Tensor,
        launches: tuple[_PackedLaunch | NativeLaunch, ...],
        *,
        scratch: tuple[Tensor, ...] = (),
        reusable_scratch: bool = False,
        reusable_scratch_indices: tuple[int, ...] = (),
        row_tile_count: int | None = None,
        row_tile_size: int | None = None,
    ) -> None:
        # Opt-in: independent row tiles and no hidden communication. None
        # deliberately makes no scheduling claim for a new operator/backend.
        if row_tile_count is not None and (type(row_tile_count) is not int or row_tile_count < 1):
            raise ValueError("row tile count must be a positive integer")
        if row_tile_size is not None and (type(row_tile_size) is not int or row_tile_size < 1):
            raise ValueError("row tile size must be a positive integer")
        self.row_tile_count = row_tile_count
        # A format may declare the geometry actually compiled into its kernel.
        # Routing prefixes and audit recipes consume this same declaration.
        self.row_tile_size = row_tile_size
        self.output = output
        self.scratch = scratch
        self.reusable_scratch = reusable_scratch
        if type(reusable_scratch_indices) is not tuple:
            raise ValueError("reusable scratch indices require an explicit tuple")
        indices = ((0,) if reusable_scratch else ()) + reusable_scratch_indices
        if any(type(i) is not int or not 0 <= i < len(scratch) for i in indices):
            raise ValueError("reusable scratch indices differ from owned buffers")
        self.reusable_scratch_indices = tuple(dict.fromkeys(indices))
        self._launches = launches
        self._ordinal = 0
        self._receipt_token = object()

    def rebind_private_scratch(self, scratch: Tensor) -> None:
        """Preserve the original first-buffer opt-in ABI."""
        if not self.reusable_scratch:
            raise ValueError("scratch rebinding requires an unlaunched opt-in binding")
        self.rebind_scratch(0, scratch)

    def rebind_scratch(self, index: int, scratch: Tensor) -> None:
        """Replace a format-declared temporary before any launch or capture.

        The format promises complete production and consumption inside this
        binding, with no exported aliases. Encoded inputs and persistent
        outputs are never implicitly reusable. Concurrent lanes remain disjoint.
        """
        if (type(index) is not int or index not in self.reusable_scratch_indices
                or self._ordinal):
            raise ValueError("scratch rebinding requires an unlaunched opt-in binding")
        old = self.scratch[index]
        if (
            scratch.shape != old.shape
            or scratch.dtype != old.dtype
            or scratch.device != old.device
            or not scratch.is_contiguous()
        ):
            raise ValueError("replacement scratch ABI differs")
        new_pointer = scratch.untyped_storage().data_ptr()
        old_pointer = old.untyped_storage().data_ptr()
        if self.output.untyped_storage().data_ptr() == new_pointer:
            raise ValueError("private scratch must not alias output")
        for item in self._launches:
            for tensor in item.tensors:
                if tensor is not old and tensor.untyped_storage().data_ptr() in (
                    old_pointer,
                    new_pointer,
                ):
                    raise ValueError("private scratch must not alias other bound operands")
        self._launches = tuple(
            item.rebind_scratch(old, scratch) if isinstance(item, NativeLaunch) else replace(
                item,
                tensors=(tensors := tuple(scratch if t is old else t for t in item.tensors)),
                bindings=tuple(bind_tensor(t) for t in tensors),
            )
            for item in self._launches
        )
        self.scratch = (*self.scratch[:index], scratch, *self.scratch[index + 1:])

    def project_artifacts(self) -> tuple[CompiledArtifactIdentity, ...]:
        identities: list[CompiledArtifactIdentity] = []
        for item in self._launches:
            if isinstance(item, NativeLaunch):
                identities.extend(item.project_artifacts())
            else:
                identities.append(project_triton_artifact(
                    item.kernel, operator_id=item.operator_id, device=self.output.device,
                    launch_signature=item.signature, grid=item.grid,
                ))
        return tuple(identities)

    def validate_operands(self) -> None:
        """Preflight a composed binding before any producer or collective executes."""
        for item in self._launches:
            if isinstance(item, NativeLaunch):
                item.validate_operands()
                continue
            if tuple(bind_tensor(t) for t in item.tensors) != item.bindings:
                raise ValueError("packed launch operands changed after binding")

    def launch(self) -> tuple[LaunchReceipt, ...]:
        self.validate_operands()
        self._ordinal += 1
        receipts = []
        with torch.cuda.device(self.output.device):
            stream = int(torch.cuda.current_stream(self.output.device).cuda_stream)
            for item in self._launches:
                if isinstance(item, NativeLaunch):
                    receipts.append(replace(
                        item.launch(), ordinal=self._ordinal, _primitive_token=self._receipt_token
                    ))
                    continue
                item.kernel[item.grid](*item.arguments, stream=stream)
                receipts.append(
                    LaunchReceipt(
                        id(item.kernel),
                        item.bindings,
                        item.signature,
                        item.grid,
                        stream,
                        self._ordinal,
                        self._receipt_token,
                    )
                )
        return tuple(receipts)
