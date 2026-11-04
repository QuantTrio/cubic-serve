"""Format and grouping contracts for independently encoded K and V.

This module describes storage, not an execution fallback. A language graph must
separately qualify the writer, reader and transactional tail for its format.
"""

from __future__ import annotations

from dataclasses import dataclass
from cubic_serve._compat import StrEnum
from math import lcm, prod

import torch


def slot_storage_contiguous(value: torch.Tensor) -> bool:
    """Dense inner payloads, with an explicit nonoverlapping outer slot stride."""
    return (value.ndim >= 2 and value.shape[0] > 0 and value[0].is_contiguous()
            and value.stride(0) >= prod(value.shape[1:]))


def storage_regions_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Conservative byte-overlap check, including interleaved slot fields.

    Equal-period, disjoint inner fields can share backing without aliasing.
    Different resource classes sharing bytes remain overlapping: their live
    ownership must be established by the cache transaction, not this helper.
    """
    if not left.numel() or not right.numel():
        return False
    def span(value):
        if any(stride < 0 for stride in value.stride()):
            raise ValueError("cache storage requires nonnegative strides")
        size = (1 + sum((n - 1) * stride for n, stride in
                        zip(value.shape, value.stride(), strict=True))) * value.element_size()
        return value.data_ptr(), value.data_ptr() + size
    first, last = span(left), span(right)
    if first[1] <= last[0] or last[1] <= first[0]:
        return False
    if slot_storage_contiguous(left) and slot_storage_contiguous(right):
        pitch = left.stride(0) * left.element_size()
        if pitch == right.stride(0) * right.element_size():
            offset = (right.data_ptr() - left.data_ptr()) % pitch
            if (offset >= prod(left.shape[1:]) * left.element_size()
                    and offset + prod(right.shape[1:]) * right.element_size() <= pitch):
                return False
    return True


class CacheDType(StrEnum):
    AUTO = "auto"
    BF16 = "bfloat16"
    FP16 = "float16"
    FP8 = "fp8"  # E4M3FN, not the incompatible FNUZ encoding.
    INT8 = "int8"
    INT6 = "int6"
    INT4 = "int4"

    @property
    def quantized(self) -> bool:
        return self in (self.FP8, self.INT8, self.INT6, self.INT4)

    @property
    def bits(self) -> int:
        if self is self.AUTO:
            raise ValueError("auto cache dtype must be resolved against the model")
        return {
            self.BF16: 16,
            self.FP16: 16,
            self.FP8: 8,
            self.INT8: 8,
            self.INT6: 6,
            self.INT4: 4,
        }[self]

    @property
    def quant_max(self) -> float:
        if self is self.FP8:
            return 448.0
        if self in (self.INT8, self.INT6, self.INT4):
            return float((1 << (self.bits - 1)) - 1)
        raise ValueError("unquantized cache has no quantization range")


class CacheGroupAxis(StrEnum):
    TOKENS = "tokens"  # Column groups: fixed head/channel, consecutive tokens.
    CHANNELS = "channels"  # Row groups: fixed token/head, consecutive channels.


@dataclass(frozen=True, slots=True)
class CacheStorageSpec:
    dtype: CacheDType
    axis: CacheGroupAxis
    group_size: int = 32

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, CacheDType) or not isinstance(self.axis, CacheGroupAxis):
            raise TypeError("cache dtype and group axis require typed values")
        if type(self.group_size) is not int or self.group_size <= 0:
            raise ValueError("cache group size must be a positive integer")

    def resolve(self, model_dtype: torch.dtype) -> CacheStorageSpec:
        if self.dtype is not CacheDType.AUTO:
            return self
        if model_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("auto cache storage requires a BF16 or FP16 model dtype")
        return CacheStorageSpec(
            CacheDType.BF16 if model_dtype is torch.bfloat16 else CacheDType.FP16,
            self.axis,
            self.group_size,
        )

    @property
    def identity(self) -> tuple[str, str, str, int]:
        # Scale, rounding, integer packing and FP8 encoding are versioned ABI.
        return ("sym-fp32-scale-rne-lsb-v1", self.dtype.value, self.axis.value, self.group_size)


@dataclass(frozen=True, slots=True)
class KVCacheStorageSpec:
    key: CacheStorageSpec = CacheStorageSpec(CacheDType.AUTO, CacheGroupAxis.TOKENS)
    value: CacheStorageSpec = CacheStorageSpec(CacheDType.AUTO, CacheGroupAxis.CHANNELS)
    enable_rotation: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.key, CacheStorageSpec) or not isinstance(
            self.value, CacheStorageSpec
        ):
            raise TypeError("K and V require independent storage specs")
        if type(self.enable_rotation) is not bool:
            raise TypeError("cache rotation requires a boolean")

    def resolve(self, model_dtype: torch.dtype) -> KVCacheStorageSpec:
        return KVCacheStorageSpec(
            self.key.resolve(model_dtype), self.value.resolve(model_dtype), self.enable_rotation
        )

    @property
    def quantized(self) -> bool:
        return self.key.dtype.quantized or self.value.dtype.quantized

    @property
    def identity(self) -> tuple[object, ...]:
        return ("committed-prefix-current-native-v1", self.key.identity,
                self.value.identity, self.enable_rotation)

    @property
    def column_group_alignment(self) -> int:
        """Engine page alignment; a column group never spans physical owners."""
        return lcm(
            *(
                spec.group_size
                for spec in (self.key, self.value)
                if spec.dtype.quantized and spec.axis is CacheGroupAxis.TOKENS
            )
        )


@dataclass(frozen=True, slots=True)
class GroupedCacheLayout:
    """Logical [token, head, channel] mapped to contiguous encoded groups.

    Groups never cross a head. A final partial group has zero padding, which is
    storage only: it has no request identity and cannot influence absmax.
    Column-tail lifetime is separate from this immutable sealed-group layout.
    """

    spec: CacheStorageSpec
    tokens: int
    heads: int
    head_dim: int

    def __post_init__(self) -> None:
        if not isinstance(self.spec, CacheStorageSpec):
            raise TypeError("cache layout requires a storage spec")
        if any(type(n) is not int or n <= 0 for n in (self.tokens, self.heads, self.head_dim)):
            raise ValueError("cache tensor dimensions must be positive integers")
        if self.spec.dtype is CacheDType.AUTO:
            raise ValueError("cache layout requires a resolved storage dtype")

    @property
    def logical_shape(self) -> tuple[int, int, int]:
        return (self.tokens, self.heads, self.head_dim)

    @property
    def group_shape(self) -> tuple[int, int, int]:
        g = self.spec.group_size
        if self.spec.axis is CacheGroupAxis.TOKENS:
            return (self.heads, self.head_dim, (self.tokens + g - 1) // g)
        return (self.tokens, self.heads, (self.head_dim + g - 1) // g)

    @property
    def group_count(self) -> int:
        return prod(self.group_shape)

    @property
    def packed_group_bytes(self) -> int:
        return (self.spec.group_size * self.spec.dtype.bits + 7) // 8

    @property
    def payload_bytes(self) -> int:
        if not self.spec.dtype.quantized:
            return prod(self.logical_shape) * 2
        return self.group_count * self.packed_group_bytes

    @property
    def scale_bytes(self) -> int:
        return self.group_count * 4 if self.spec.dtype.quantized else 0

    @property
    def storage_bytes(self) -> int:
        return self.payload_bytes + self.scale_bytes

    def group_location(self, token: int, head: int, channel: int) -> tuple[int, int]:
        """Return flat group id and member index; independent of cache encoding."""
        if any(
            type(i) is not int or not 0 <= i < n
            for i, n in zip((token, head, channel), self.logical_shape, strict=True)
        ):
            raise IndexError("cache coordinate is outside its logical extent")
        if self.spec.axis is CacheGroupAxis.TOKENS:
            return (
                (head * self.head_dim + channel) * self.group_shape[-1]
                + token // self.spec.group_size,
                token % self.spec.group_size,
            )
        return (
            (token * self.heads + head) * self.group_shape[-1] + channel // self.spec.group_size,
            channel % self.spec.group_size,
        )

    def groups_for_tokens(
        self, tokens: tuple[int, ...], *, complete: bool = False
    ) -> tuple[int, ...]:
        """Physical group authority; a column write owns every real member.

        Selecting a closing token alone never grants writes to earlier tokens'
        packed bytes or their shared scale. The transaction must name the group.
        """
        if (type(tokens) is not tuple or any(type(t) is not int for t in tokens)
                or tuple(sorted(set(tokens))) != tokens
                or any(not 0 <= t < self.tokens for t in tokens)):
            raise ValueError("cache token offsets must be unique sorted logical positions")
        if not self.spec.dtype.quantized:
            raise ValueError("plain cache storage has token rows, not encoded groups")
        if self.spec.axis is CacheGroupAxis.CHANNELS:
            per_token = self.heads * self.group_shape[-1]
            return tuple(t * per_token + i for t in tokens for i in range(per_token))
        group = self.spec.group_size
        selected = set(tokens)
        columns = tuple(g for g in sorted({t // group for t in tokens})
                        if not complete or all(t in selected for t in
                            range(g * group, min((g + 1) * group, self.tokens))))
        per_column = self.group_shape[-1]
        return tuple(i * per_column + g for i in range(self.heads * self.head_dim) for g in columns)

    def valid_code_groups(self, payload: torch.Tensor) -> torch.Tensor:
        """Code-domain and canonical padding checks, without floating decode."""
        if (not self.spec.dtype.quantized or payload.dtype is not torch.uint8
                or tuple(payload.shape[-2:]) != (self.group_count, self.packed_group_bytes)):
            raise ValueError("encoded cache payload differs from its code-group ABI")
        members = torch.arange(self.spec.group_size, device=payload.device)
        if self.spec.dtype is CacheDType.FP8:
            codes = payload
            valid = (codes & 127) != 127  # E4M3FN has two NaN codes.
        else:
            bits = self.spec.dtype.bits
            byte, shift = members * bits // 8, members * bits % 8
            words = payload[..., byte].to(torch.int32)
            if bits == 6:
                upper = payload[..., (byte + 1).clamp_max(self.packed_group_bytes - 1)]
                words |= torch.where(shift + bits > 8, upper.to(torch.int32) << 8, 0)
            codes = (words >> shift) & ((1 << bits) - 1)
            valid = codes != (1 << (bits - 1))  # Symmetric negative endpoint is reserved.
        extent = self.tokens if self.spec.axis is CacheGroupAxis.TOKENS else self.head_dim
        if extent % self.spec.group_size:
            groups = self.group_shape[-1]
            last = torch.arange(self.group_count, device=payload.device) % groups == groups - 1
            padding = last[:, None] & (members >= extent % self.spec.group_size)
            valid &= ~padding | (codes == 0)
        complete = valid.all(dim=-1)
        used_bits = self.spec.group_size * self.spec.dtype.bits % 8
        if used_bits:
            # Final-byte slack is storage padding, never another logical code.
            complete &= payload[..., -1] >> used_bits == 0
        return complete
