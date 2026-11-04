"""Shared packed tile schema, independent of any quantization encoding."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class PackedMatmulParameters:
    """Execution tiles only; checkpoint scales and rounding are not tunables."""

    fixed_block_m: int = 16
    fixed_block_n: int = 32
    fixed_block_k: int = 32
    fixed_warps: int = 4
    fixed_stages: int = 3


@dataclass(frozen=True)
class PackedLinearParameters(PackedMatmulParameters):
    # Backends with their own placement selector opt out in their owner package.
    generic_geometry: ClassVar[bool] = True
    embedding_block_k: int = 128


@dataclass(frozen=True)
class PackedMoEParameters(PackedMatmulParameters):
    grouped_projections: bool = True
