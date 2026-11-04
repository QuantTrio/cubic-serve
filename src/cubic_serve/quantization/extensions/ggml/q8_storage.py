"""One lossless device carrier shared by Q8 projection and lookup consumers.

The carrier is independent of activation precision, graph M, and model type.
It preserves original expert/row/code/FP16-scale identities. Packing has no
numerical operation and is performed before capture; CPU lookup stays raw.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.ops.cuda.compiled_artifact import bind_tensor
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.quantization.common.binding import concatenate_to_device

if TYPE_CHECKING:
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators


@dataclass(frozen=True)
class Q8Storage:
    storage: Tensor
    n: int
    k: int
    experts: int = 1

    def __post_init__(self) -> None:
        if (any(type(v) is not int or not 0 < v < 2**31 for v in (self.n, self.k, self.experts))
                or self.k % 32 or self.storage.dtype != torch.uint8
                or self.storage.ndim != 1 or not self.storage.is_contiguous()
                or not self.storage.is_cuda
                or self.storage.numel() != self.experts * ((self.n + 15) // 16)
                   * (self.k // 32) * 544):
            raise ValueError("Q8 device carrier differs from its logical geometry")

    @property
    def codes(self) -> Tensor:
        """A view; metadata and codes share the same device allocation."""
        return self.storage.view(self.experts, -1)[:, :self.k // 32 * ((self.n + 15) // 16) * 512]

    @property
    def scales(self) -> Tensor:
        return self.storage.view(self.experts, -1)[:, self.k // 32 * ((self.n + 15) // 16) * 512:]


@cache
def sources() -> tuple[tuple[str, bytes], ...]:
    folder = Path(__file__).parent
    return (("entry.cu", (folder / "q8_storage.cu").read_bytes()),
            ("q8_storage.cuh", (folder / "q8_storage.cuh").read_bytes()))


def pack(raw: Tensor, *, k: int, device: torch.device) -> Q8Storage:
    """Consume raw checkpoint bytes, retaining only the prepared allocation.

    A bounded upload is temporary when the checkpoint is CPU-resident, just
    as for Marlin. The caller owns the immutable source and prepared cache.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Q8 storage preparation must precede capture")
    if (raw.dtype != torch.uint8 or raw.ndim not in (2, 3) or not raw.is_contiguous()
            or type(k) is not int or k <= 0 or k % 32 or raw.shape[-1] != k // 32 * 34
            or min(raw.shape) <= 0 or device.type != "cuda"
            or raw.device.type not in ("cpu", "cuda")
            or (raw.is_cuda and raw.device != device)):
        raise ValueError("Q8 preparation requires complete original blocks and rank-local ownership")
    e, n = (1, raw.shape[0]) if raw.ndim == 2 else raw.shape[:2]
    nt, chunks = (n + 15) // 16, k // 32
    if max(e, n, chunks) >= 2**31 or nt * 16 >= 2**31:
        raise ValueError("Q8 storage dimensions exceed the native transport")
    output = torch.empty(e * chunks * nt * 544, dtype=torch.uint8, device=device)
    uploaded = raw.to(device)
    module, entry = compile_native(sources(), device, "ggml_q8_storage")
    count = e * chunks * nt * 128
    module.bind(entry, (uploaded, output, *(CudaScalar("i32", v) for v in (n, chunks, e))),
                operator_id="cubic.cuda.ggml.q8_pack.v1",
                grid=((count + 255) // 256, 1, 1), block=(256, 1, 1)).launch()
    return Q8Storage(output, n, k, e)


def _version(tensor: Tensor) -> int | None:
    try:
        return tensor._version
    except RuntimeError:
        return None  # Inference checkpoint tensors are immutable owners.


@dataclass(frozen=True)
class _Sources:
    weights: tuple[QuantizedMatrix, ...]
    bindings: tuple
    versions: tuple[int | None, ...]

    def matches(self) -> bool:
        return all(binding.matches(weight.payload["weight_packed"])
                   and version == _version(weight.payload["weight_packed"])
                   for weight, binding, version in
                   zip(self.weights, self.bindings, self.versions, strict=True))


def prepare(operators: PackedLinearOperators,
            groups: tuple[tuple[QuantizedMatrix, ...], ...], device: torch.device) -> Q8Storage:
    """Use the resident operator owner's existing prepared-weight cache.

    Groups concatenate projection rows inside each original expert, exactly as
    Marlin's preparation contract. The cache retains one device tensor; original
    CPU checkpoint owners remain CPU-resident. No global cache or M key exists.
    """
    from .checkpoint import kind

    if not groups or any(not group for group in groups):
        raise ValueError("Q8 preparation requires nonempty original owner groups")
    first = groups[0][0]
    experts, k = len(groups[0]), first.shape[1]
    if any(len(group) != experts or any(w.shape != group[0].shape
           or w.shape[1] != k or w.encoding != first.encoding or int(kind(w.encoding)) != 8
           for w in group) for group in groups):
        raise ValueError("Q8 preparation requires matching original expert ownership and encoding")
    n = sum(group[0].shape[0] for group in groups)
    weights = tuple(w for group in groups for w in group)
    key = ("ggml.q8-planar.v1", device, tuple(tuple(id(w) for w in g) for g in groups))
    cached = operators.prepared_weights.get(key)
    if cached is not None and isinstance(cached[0], _Sources) and cached[0].matches():
        return Q8Storage(cached[1], n, k, experts)
    owner = _Sources(weights, tuple(bind_tensor(w.payload["weight_packed"]) for w in weights),
                     tuple(_version(w.payload["weight_packed"]) for w in weights))
    ordered = tuple(group[expert].payload["weight_packed"]
                    for expert in range(experts) for group in groups)
    raw = concatenate_to_device(ordered, device).view(experts, n, k // 32 * 34)
    prepared = pack(raw, k=k, device=device)
    operators.prepared_weights[key] = owner, prepared.storage
    return prepared
