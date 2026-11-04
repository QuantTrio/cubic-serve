"""AWQ owns numerical eligibility; launch units are shared Marlin parameters."""

from dataclasses import dataclass

from cubic_serve.quantization.common.marlin.parameters import MarlinParameters, validate_marlin
from cubic_serve.quantization.common.parameters import PackedLinearParameters, PackedMoEParameters
from cubic_serve.quantization.common.policy import validate_tiles


@dataclass(frozen=True)
class AwqMoEParameters(PackedMoEParameters, MarlinParameters):
    pass


PARAMETERS = {"linear": PackedLinearParameters, "moe": AwqMoEParameters}
FORBIDDEN_OVERRIDES: dict[str, str] = {}


def validate(operation: str, values: object) -> tuple[tuple[str, int | bool | float | str], ...]:
    if operation == "moe":
        return validate_marlin(PackedMoEParameters, values, FORBIDDEN_OVERRIDES)
    return validate_tiles(PARAMETERS[operation], values, FORBIDDEN_OVERRIDES)
