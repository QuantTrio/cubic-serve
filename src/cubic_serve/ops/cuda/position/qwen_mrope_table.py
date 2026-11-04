"""Readiness-built FP16/BF16 M-RoPE tables and graph-composable gather.

Semantics follow llama.cpp c841aeeb, vLLM 73fac1e1, and SGLang ba433bb4. GraphProgram owns
range validation and warmup; replay gathers sealed buffers without trigonometry or clamping.
"""

from __future__ import annotations

from typing_extensions import TypeAliasType

import hashlib
import math
from collections.abc import Sequence
from typing import NamedTuple

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor

_TensorABI = TypeAliasType("_TensorABI", tuple[tuple[int, ...], tuple[int, ...], torch.dtype, torch.device])


class MRoPETableEvidence(NamedTuple):
    max_position: int
    rotary_dim: int
    section: tuple[int, int, int]
    rope_theta_hex: str
    table_digest: str


# fmt: off
@triton.jit  # type: ignore[untyped-decorator]
def _gather(  # type: ignore[no-untyped-def]
    position_ids, cosine_table, sine_table, out_cosine, out_sine,
    C: tl.constexpr, R: tl.constexpr, HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr, BLOCK: tl.constexpr,
):
# fmt: on
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    frequency = columns % (R // 2)
    height = (frequency % 3 == 1) & (frequency < HEIGHT * 3)
    width = (frequency % 3 == 2) & (frequency < WIDTH * 3)
    axis = tl.where(height, 1, tl.where(width, 2, 0)).to(tl.int64)
    position = tl.load(position_ids + axis * C + row).to(tl.int64)
    offsets = position * R + columns
    valid = columns < R
    tl.store(
        out_cosine + row * R + columns, tl.load(cosine_table + offsets, mask=valid), mask=valid
    )
    tl.store(out_sine + row * R + columns, tl.load(sine_table + offsets, mask=valid), mask=valid)


def _spec(rotary_dim: int, section: Sequence[int]) -> tuple[int, int, int]:
    if type(rotary_dim) is not int or rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive and even")
    if len(section) != 3 or any(type(count) is not int for count in section):
        raise ValueError("section must contain exactly three integers")
    resolved = (section[0], section[1], section[2])
    half = rotary_dim // 2
    if (
        any(count <= 0 for count in resolved)
        or sum(resolved) != half
        or 3 * resolved[1] - 2 >= half
        or 3 * resolved[2] - 1 >= half
    ):
        raise ValueError("section does not fit Qwen interleaving")
    return resolved


def _digest(cosine: Tensor, sine: Tensor, identity: object) -> str:
    hasher = hashlib.blake2s(repr(identity).encode())
    for table in (cosine, sine):
        hasher.update(table.detach().view(torch.uint8).cpu().numpy().tobytes())
    return "b2s256:" + hasher.hexdigest()


# fmt: off
def build_bf16_mrope_tables(*, max_position: int, device: torch.device,
    rotary_dim: int = 64, section: Sequence[int] = (11, 11, 10),
    rope_theta: float = 10_000_000.0,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Tensor, Tensor, MRoPETableEvidence]:
# fmt: on
    """Build canonical tables in the requested storage dtype once on their target CUDA device."""

    if type(max_position) is not int or max_position <= 0:
        raise ValueError("max_position must be a positive integer")
    if device.type != "cuda":
        raise ValueError("M-RoPE tables must be built on their target CUDA device")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("M-RoPE table storage requires FP16 or BF16")
    if isinstance(rope_theta, bool) or not isinstance(rope_theta, int | float):
        raise TypeError("rope_theta must be a real number")
    theta = float(rope_theta)
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("rope_theta must be finite and positive")
    resolved = _spec(rotary_dim, section)
    exponent = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / float(
        rotary_dim
    )
    theta_tensor = torch.tensor(theta, dtype=torch.float32, device=device)
    inverse_frequency = torch.reciprocal(torch.pow(theta_tensor, exponent))
    positions = torch.arange(max_position, dtype=torch.float32, device=device)
    frequencies = positions.unsqueeze(-1) * inverse_frequency
    doubled = torch.cat((frequencies, frequencies), dim=-1)
    cosine, sine = doubled.cos().to(dtype), doubled.sin().to(dtype)
    identity = max_position, rotary_dim, resolved, theta.hex()
    evidence = MRoPETableEvidence(*identity, _digest(cosine, sine, identity))
    return cosine, sine, evidence


def _abi(tensor: Tensor) -> _TensorABI:
    if not isinstance(tensor, Tensor):
        raise TypeError("M-RoPE gather operands must be tensors")
    return tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device


class MRoPECosSinGatherPrimitive:
    """Gather three-axis positions from sealed caller-owned FP16/BF16 base tables."""

    # fmt: off
    def __init__(
        self, position_ids: Tensor, cosine_table: Tensor, sine_table: Tensor,
        out_cosine: Tensor, out_sine: Tensor, *, evidence: MRoPETableEvidence,
    ) -> None:
    # fmt: on
        tensors = position_ids, cosine_table, sine_table, out_cosine, out_sine
        abis = tuple(_abi(tensor) for tensor in tensors)
        capacity = position_ids.shape[1] if position_ids.ndim == 2 else -1
        if not isinstance(evidence, MRoPETableEvidence):
            raise TypeError("evidence must be MRoPETableEvidence")
        section = _spec(evidence.rotary_dim, evidence.section)
        device = position_ids.device
        max_position, rotary = evidence.max_position, evidence.rotary_dim
        dtype = cosine_table.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("M-RoPE table storage requires FP16 or BF16")
        expected = (
            ((3, capacity), (capacity, 1), torch.int64, device),
            ((max_position, rotary), (rotary, 1), dtype, device),
            ((max_position, rotary), (rotary, 1), dtype, device),
            ((capacity, rotary), (rotary, 1), dtype, device),
            ((capacity, rotary), (rotary, 1), dtype, device),
        )
        valid = capacity > 0 and max_position > 0 and device.type == "cuda" and abis == expected
        if not valid:
            raise ValueError("operands do not match fixed CUDA M-RoPE table/gather ABI")
        if _digest(cosine_table, sine_table, evidence[:-1]) != evidence.table_digest:
            raise ValueError("M-RoPE table content does not match its evidence digest")
        self._abis = abis
        self._meta = capacity, rotary, section, evidence.table_digest
        self._block = triton.next_power_of_2(rotary)

    @property
    def launch_signature(self) -> str:
        capacity, rotary, section, digest = self._meta
        return f"c={capacity},r={rotary},section={section},table={digest}"

    # fmt: off
    def launch(self, position_ids: Tensor, cosine_table: Tensor, sine_table: Tensor,
        out_cosine: Tensor, out_sine: Tensor,
    ) -> None:
    # fmt: on
        tensors = position_ids, cosine_table, sine_table, out_cosine, out_sine
        if tuple(_abi(tensor) for tensor in tensors) != self._abis:
            raise ValueError("M-RoPE gather operands changed after fixed-ABI binding")
        capacity, rotary, section, _ = self._meta
        _gather[(capacity,)](
            *tensors,
            C=capacity,
            R=rotary,
            HEIGHT=section[1],
            WIDTH=section[2],
            BLOCK=self._block,
            num_warps=4,
        )
