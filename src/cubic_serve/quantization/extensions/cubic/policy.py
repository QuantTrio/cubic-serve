"""Format-owned parameter schema and semantic override constraints."""

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar, Literal

from cubic_serve.quantization.common.marlin.parameters import MarlinParameters, validate_marlin
from cubic_serve.quantization.common.parameters import PackedLinearParameters, PackedMoEParameters


def supports_compute_mode(encoding, operation: str, mode: str) -> bool:
    if mode == "wna16":
        return True
    return (operation == "linear" and mode == "wna8"
            and 2 <= encoding.bits <= 8)


@dataclass(frozen=True)
class CubicLinearParameters(PackedLinearParameters, MarlinParameters):
    marlin_enabled: bool = True
    marlin_stages: int = 2
    marlin_max_candidates: int = 3
    marlin_max_k_partitions: int = 8
    marlin_cta_supply_factor: float = 2.0
    wna8_max_k_partitions: int = 8
    wna8_cta_supply_factor: float = 2.0
    generic_geometry: ClassVar[bool] = False
    fixed_max_block_m: int = 64
    reuse_block_n: int = 64
    reuse_warps: int = 4
    fixed_block_n: int = 16
    fixed_block_k: int = 64
    fixed_split_k: int = 8
    decode_scalar_ops: int = 32
    mma_scalar_ratio: int = 16
    level_table_max_bits: int = 8
    level_table_max_bytes: int = 0
    level_table_max_packed_ratio: float = 1.0
    level_word_lookup: bool = True
    row_reuse_max_tiles: int = 2

    def decode_mode(
        self,
        *,
        bits: int,
        metadata: str,
        n: int,
        k: int,
        group_in: int,
        group_out: int,
        element_bytes: int,
    ) -> Literal["direct", "levels", "words"]:
        """Choose an implemented decoder before allocating its resident table.

        Budgets are per local matrix and compute dtype, not per graph bucket.
        Zero explicitly disables a budget. The default bounds table storage by
        the original packed codes, using actual group geometry. These bounds limit
        memory, not latency: wider codebooks are not assumed to be faster.
        """
        if min(n, k, group_in, group_out, element_bytes) <= 0 or not 1 <= bits <= 8:
            raise ValueError("Cubic decode selection requires valid local weight geometry")
        if (
            metadata != "int8-scale-int4-ab-global-fp32"
            or element_bytes != 2
            or not 2 <= bits <= self.level_table_max_bits
        ):
            # W1 is bipolar, not the signed magnitude codebook used by W2-W8.
            return "direct"
        groups = ((n + group_out - 1) // group_out) * ((k + group_in - 1) // group_in)
        table_bytes = groups * (1 << (bits - 1)) * element_bytes
        packed_bytes = n * ((k * bits + 7) // 8)
        if self.level_table_max_bytes and table_bytes > self.level_table_max_bytes:
            return "direct"
        if (
            self.level_table_max_packed_ratio
            and table_bytes / packed_bytes > self.level_table_max_packed_ratio
        ):
            return "direct"
        # This is the existing kernel's capability, not a tunable bit-width
        # cutoff. Its group/K-tile alignment fallback remains inside the loader.
        return "words" if bits == 4 and self.level_word_lookup else "levels"

    def output_tiles(
        self, m: int, n: int, k: int, split_k: int, multiprocessors: int
    ) -> tuple[tuple[float, int, int, int], ...]:
        """Estimate device waves of decode, operand loading and MMA work.

        Grow both output dimensions together to preserve the MMA layout aspect
        ratio. K partitioning never participates in tile selection. Coefficients
        express relative work, not a measured model/device/C timing table.
        The binder additionally accounts for actual compiled register/shared
        memory pressure and spills before sealing the chosen launch.
        """
        if min(m, n, k, split_k, multiprocessors) <= 0:
            raise ValueError("output tile selection requires positive extents")
        candidates = []
        bm = self.fixed_block_m
        while bm <= max(self.fixed_block_m, self.fixed_max_block_m):
            bn = (
                self.fixed_block_n
                if bm == self.fixed_block_m
                else min(self.reuse_block_n, self.fixed_block_n * bm // self.fixed_block_m)
            )
            warps = self.fixed_warps if bm == self.fixed_block_m else self.reuse_warps
            cost = self.tile_cost(m, n, k, split_k, multiprocessors, bm, bn)
            candidates.append((cost, bm, bn, warps))
            bm *= 2
        return tuple(sorted(candidates))

    def tile_cost(
        self, m: int, n: int, k: int, split_k: int, multiprocessors: int, rows: int, columns: int
    ) -> float:
        """Decoded weight work is shared over the effective output-row extent."""
        grid = ((m + rows - 1) // rows) * ((n + columns - 1) // columns) * split_k
        waves = (grid + multiprocessors - 1) // multiprocessors
        extent_k = (
            (k + self.fixed_block_k * split_k - 1) // (self.fixed_block_k * split_k)
        ) * self.fixed_block_k
        return (
            waves
            * extent_k
            * (
                columns * self.decode_scalar_ops
                + 2 * rows * columns / self.mma_scalar_ratio
                + 2 * rows
            )
        )


@dataclass(frozen=True)
class CubicMoEParameters(PackedMoEParameters, MarlinParameters):
    marlin_stages: int = 2
    marlin_max_candidates: int = 3


PARAMETERS = {"linear": CubicLinearParameters, "moe": CubicMoEParameters}
FORBIDDEN_OVERRIDES: dict[str, str] = {}


def validate(operation: str, values: object) -> tuple[tuple[str, int | bool | float | str], ...]:
    from cubic_serve.quantization.common.policy import validate_tiles

    if operation == "moe":
        return validate_marlin(PackedMoEParameters, values, FORBIDDEN_OVERRIDES)
    if operation == "linear" and isinstance(values, dict):
        custom: dict[str, int | bool | float] = {}
        if "wna8_max_k_partitions" in values:
            value = values["wna8_max_k_partitions"]
            if type(value) is not int or value < 1:
                raise ValueError("wna8_max_k_partitions must be a positive integer")
            custom["wna8_max_k_partitions"] = value
        if "wna8_cta_supply_factor" in values:
            value = values["wna8_cta_supply_factor"]
            if type(value) not in (int, float) or not isfinite(value) or value < 1:
                raise ValueError("wna8_cta_supply_factor must be finite and at least one")
            custom["wna8_cta_supply_factor"] = float(value)
        native = {k: v for k, v in values.items() if k.startswith("marlin_")}
        partition_limit = native.pop("marlin_max_k_partitions", None)
        if partition_limit is not None:
            if type(partition_limit) is not int or partition_limit < 1:
                raise ValueError("marlin_max_k_partitions must be a positive integer")
            custom["marlin_max_k_partitions"] = partition_limit
        supply_factor = native.pop("marlin_cta_supply_factor", None)
        if supply_factor is not None:
            if (
                type(supply_factor) not in (int, float)
                or not isfinite(supply_factor)
                or supply_factor < 1
            ):
                raise ValueError("marlin_cta_supply_factor must be finite and at least one")
            custom["marlin_cta_supply_factor"] = float(supply_factor)
        custom.update(dict(validate_marlin(PackedLinearParameters, native, FORBIDDEN_OVERRIDES)))
        if "level_table_max_bytes" in values:
            value = values["level_table_max_bytes"]
            if type(value) is not int or value < 0:
                raise ValueError("Cubic level_table_max_bytes must be a nonnegative integer")
            custom["level_table_max_bytes"] = value
        if "level_table_max_packed_ratio" in values:
            value = values["level_table_max_packed_ratio"]
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(
                    "Cubic level_table_max_packed_ratio must be finite and nonnegative"
                )
            custom["level_table_max_packed_ratio"] = float(value)
        if "level_word_lookup" in values:
            value = values["level_word_lookup"]
            if type(value) is not bool:
                raise ValueError("Cubic level_word_lookup must be boolean")
            custom["level_word_lookup"] = value
        if "row_reuse_max_tiles" in values:
            value = values["row_reuse_max_tiles"]
            if type(value) is not int or value not in (1, 2):
                raise ValueError("Cubic row_reuse_max_tiles must be 1 or 2")
            custom["row_reuse_max_tiles"] = value
        if "fixed_max_block_m" in values:
            value = values["fixed_max_block_m"]
            if type(value) is not int or not 16 <= value <= 256 or value & (value - 1):
                raise ValueError("Cubic fixed_max_block_m must be a power of two from 16 to 256")
            custom["fixed_max_block_m"] = value
        if "level_table_max_bits" in values:
            value = values["level_table_max_bits"]
            if type(value) is not int or value not in (0, 2, 3, 4, 5, 6, 7, 8):
                raise ValueError("Cubic level_table_max_bits must be 0 or 2 through 8")
            custom["level_table_max_bits"] = value
        for key in ("fixed_split_k", "decode_scalar_ops", "mma_scalar_ratio"):
            if key not in values:
                continue
            value = values[key]
            if (
                type(value) is not int
                or value <= 0
                or (key == "fixed_split_k" and value not in (1, 2, 4, 8, 16, 32))
            ):
                raise ValueError(f"invalid Cubic work parameter {key}={value}")
            custom[key] = value
        other = {key: value for key, value in values.items() if key not in custom}
        return tuple(
            sorted(
                (
                    *validate_tiles(PARAMETERS[operation], other, FORBIDDEN_OVERRIDES),
                    *custom.items(),
                )
            )
        )
    return validate_tiles(PARAMETERS[operation], values, FORBIDDEN_OVERRIDES)
