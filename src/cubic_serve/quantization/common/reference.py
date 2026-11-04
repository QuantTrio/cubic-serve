from __future__ import annotations

from typing_extensions import TypeAliasType, TypeVar

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from cubic_serve.models.quantization import (
    QuantizationError,
)

IntegerGeometry = TypeAliasType("IntegerGeometry", Tensor | tuple[Tensor | None, ...])
T = TypeVar("T", bound=IntegerGeometry)


class IntegerGeometryMemo:
    """Bounded immutable integer coordinates, never weights or runtime state."""

    def __init__(self, max_bytes: int = 1 << 20, max_entries: int = 128):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("integer geometry requires a nonnegative byte budget")
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError("integer geometry requires a nonnegative entry budget")
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries: OrderedDict[tuple[object, ...], tuple[int, IntegerGeometry]] = OrderedDict()
        self.bytes = self.peak_bytes = self.hits = self.misses = 0

    def clear(self) -> None:
        self.entries.clear()
        self.bytes = 0

    @staticmethod
    def _capturing(value: IntegerGeometry) -> bool:
        tensors = (value,) if isinstance(value, Tensor) else value
        devices = {t.device for t in tensors if t is not None and t.is_cuda}
        for device in devices:
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    return True
        return False

    def get(self, key: tuple[object, ...], build: Callable[[], T]) -> T:
        prior = self.entries.get(key)
        if prior is not None and not self._capturing(prior[1]):
            self.entries.move_to_end(key)
            self.hits += 1
            return cast(T, prior[1])
        self.misses += 1
        value = build()
        tensors = (value,) if isinstance(value, Tensor) else value
        if any(t is not None and (not isinstance(t, Tensor)
               or t.dtype not in (torch.int32, torch.int64, torch.bool)) for t in tensors):
            raise TypeError("reference geometry contains a non-integer tensor")
        size = sum(t.numel() * t.element_size() for t in tensors if t is not None)
        if (self.max_bytes and self.max_entries and size <= self.max_bytes
                and not self._capturing(value)):
            while self.entries and (self.bytes + size > self.max_bytes
                                    or len(self.entries) >= self.max_entries):
                _, (evicted, _) = self.entries.popitem(last=False)
                self.bytes -= evicted
            self.entries[key] = size, value
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        return value


_INTEGER_GEOMETRY: ContextVar[IntegerGeometryMemo | None] = ContextVar(
    "cubic_reference_integer_geometry", default=None)


@contextmanager
def integer_geometry_scope(memo: IntegerGeometryMemo) -> Iterator[None]:
    token = _INTEGER_GEOMETRY.set(memo)
    try:
        yield
    finally:
        _INTEGER_GEOMETRY.reset(token)


def integer_geometry(
    key: tuple[object, ...], build: Callable[[], T],
) -> T:
    memo = _INTEGER_GEOMETRY.get()
    return build() if memo is None else memo.get(key, build)


def _tensor(
    payload: Mapping[str, Tensor],
    key: str,
    dtype: torch.dtype,
    shape: tuple[int, ...] | None = None,
) -> Tensor:
    value = payload.get(key)
    if (
        not isinstance(value, Tensor)
        or value.dtype is not dtype
        or (shape is not None and tuple(value.shape) != shape)
    ):
        raise QuantizationError(f"invalid quantization tensor {key!r}: expected {dtype}, {shape}")
    return value


def _positive_finite(value: Tensor, label: str, *, zero: bool = False) -> None:
    f = value.float()
    if not bool(torch.isfinite(f).all()) or bool((f < 0 if zero else f <= 0).any()):
        raise QuantizationError(
            f"{label} must be finite and {'nonnegative' if zero else 'positive'}"
        )


@dataclass(frozen=True, slots=True)
class ActivationQuantization:
    codes: Tensor
    scales: Tensor
    dequantized: Tensor


def wna8_codes_and_scale(inputs: Tensor) -> tuple[Tensor, Tensor]:
    """Recover the original signed per-token codes and FP32 scale."""
    values = inputs.float()
    scale = (values.abs().amax(-1, keepdim=True).double() / 127.).float()
    scale = torch.where(scale > 0, scale, 1.)
    inverse = (1. / scale.double()).float()
    normalized = torch.where(
        scale >= torch.finfo(torch.float32).tiny,
        values * inverse,
        (values.double() / scale.double()).float(),
    )
    codes = normalized.round().clamp(-127, 127).to(torch.int8)
    return codes, scale


def wna8_activation(inputs: Tensor) -> Tensor:
    """Independently reconstruct symmetric per-token A8 as FP32."""
    codes, scale = wna8_codes_and_scale(inputs)
    return codes.float() * scale
