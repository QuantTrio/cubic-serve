"""Marlin launch units and resource equations, independent of checkpoint math."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from cubic_serve.quantization.common.policy import validate_tiles


@dataclass(frozen=True)
class MarlinParameters:
    marlin_enabled: bool = True
    marlin_block_m: int = 16
    marlin_block_n: int = 64
    marlin_block_k: int = 128
    marlin_warps: int = 4
    marlin_stages: int = 4
    # Zero uses actual compiled-kernel residency; positive values cap it.
    marlin_blocks_per_sm: int = 0
    marlin_auto_geometry: bool = True
    marlin_max_candidates: int = 5
    marlin_calibration_replays: int = 10
    marlin_max_error_percent: int = 20
    marlin_min_gain_percent: int = 3
    marlin_cache_fraction_min_percent: int = 50


def validate_marlin(
    base_schema: type[Any],
    values: object,
    forbidden: dict[str, str],
    *,
    min_block_m: int = 16,
) -> tuple[tuple[str, int | bool | float | str], ...]:
    if not isinstance(values, dict):
        raise ValueError("packed parameters must be an object")
    native = {k: v for k, v in values.items() if k.startswith("marlin_")}
    validate_tiles(base_schema, {k: v for k, v in values.items() if k not in native}, forbidden)
    choices = {
        "marlin_block_m": tuple(m for m in (8, 16, 32, 64) if m >= min_block_m),
        "marlin_block_n": (64, 128, 256, 512),
        "marlin_block_k": (16, 32, 64, 128, 256),
        "marlin_warps": (4, 8),
        "marlin_stages": (2, 3, 4),
    }
    for key, value in native.items():
        if key in ("marlin_enabled", "marlin_auto_geometry"):
            valid = type(value) is bool
        elif key == "marlin_blocks_per_sm":
            valid = type(value) is int and value >= 0
        elif key in ("marlin_max_candidates", "marlin_calibration_replays",
                     "marlin_max_error_percent", "marlin_min_gain_percent"):
            valid = type(value) is int and value > 0
        elif key == "marlin_cache_fraction_min_percent":
            valid = type(value) is int and 0 < value <= 100
        else:
            valid = key in choices and type(value) is int and value in choices[key]
        if not valid:
            raise ValueError(f"unsupported Marlin parameter {key}={value!r}")
    return tuple(sorted(values.items()))


def pipeline_compatible(
    params: MarlinParameters, *, group: int, single_subtile: bool = False,
) -> bool:
    """Check fragment and group lifetimes; formats qualify single-subtile use."""
    bn, bk = params.marlin_block_n, params.marlin_block_k
    threads = params.marlin_warps * 32
    return (min(bn, bk, threads, group) > 0
            and ((single_subtile and bk * bn == threads * 32)
                 or bk * bn % (threads * 64) == 0)
            and max(group, bk) % min(group, bk) == 0
            and (group <= bk or (group % bk == 0 and params.marlin_stages % (group // bk) == 0)))


def shared_bytes(
    params: MarlinParameters, *, bits: int, group: int, metadata_planes: int = 0,
    row_metadata: bool = True, zero_points: bool = True,
) -> int:
    """Vendor pipeline, reduction, scales and row metadata; no SKU constants.

    Activations use 16 bits; scale storage uses at most 16 bits.
    Column scales use one group covering K. This conservative envelope also
    covers an accumulator epilogue that needs less reduction scratch.
    """
    bm, bn, bk = params.marlin_block_m, params.marlin_block_n, params.marlin_block_k
    stages = params.marlin_stages
    a = stages * bm * bk * 2
    b = stages * bk * bn * bits // 8
    reduction = max(16, bm) * (bn + 8) * 2
    temporary = max(max(b, reduction), min(b, reduction) + bn * 2)
    scale = stages * math.ceil(bk / group) * bn * 2
    # Legacy/routed callers retain the conservative envelope. A Dense binding
    # can declare identity rows and a zero-point-free carrier explicitly.
    # These are semantic storage requirements, not device tuning cutoffs.
    metadata = scale * (2 * metadata_planes if metadata_planes else 1)
    zeros = scale * (4 if bits <= 4 else 8) // 16 if zero_points else 0
    rows = bm * 16 if row_metadata else 0
    return math.ceil((temporary + a + metadata + zeros + rows) / 256) * 256
