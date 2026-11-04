"""Format-owned parameter schema and semantic override constraints."""

from dataclasses import dataclass

from cubic_serve.quantization.common.marlin.parameters import MarlinParameters, validate_marlin
from cubic_serve.quantization.common.parameters import PackedLinearParameters, PackedMoEParameters


@dataclass(frozen=True)
class NVFP4LinearParameters(PackedLinearParameters, MarlinParameters):
    # One ordered partial per independently scaled group; not a tuning knob.
    fixed_block_k: int = 16
    marlin_block_n: int = 256
    marlin_block_k: int = 64
    marlin_stages: int = 3


@dataclass(frozen=True)
class NVFP4MoEParameters(PackedMoEParameters, MarlinParameters):
    fixed_block_k: int = 16
    # N warp ownership is required by the ordered K16 rule, not by a GPU SKU.
    marlin_block_n: int = 256
    marlin_block_k: int = 64
    marlin_stages: int = 3


PARAMETERS = {"linear": NVFP4LinearParameters, "moe": NVFP4MoEParameters}
FORBIDDEN_OVERRIDES = {
    "fixed_block_k": "NVFP4 group-16 accumulation is a format boundary, not a K override"
}


def validate(operation: str, values: object) -> tuple[tuple[str, int | bool | float | str], ...]:
    from cubic_serve.quantization.common.policy import validate_tiles

    if operation in ("linear", "moe"):
        base = PackedMoEParameters if operation == "moe" else PackedLinearParameters
        result = validate_marlin(base, values, FORBIDDEN_OVERRIDES)
        if isinstance(values, dict):
            params = (NVFP4MoEParameters(**values) if operation == "moe"
                      else NVFP4LinearParameters(**values))
            if params.marlin_block_n != params.marlin_warps * 64:
                raise ValueError("NVFP4 ordered K16 requires all warps along N")
        return result
    return validate_tiles(PARAMETERS[operation], values, FORBIDDEN_OVERRIDES)
