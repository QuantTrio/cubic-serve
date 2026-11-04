"""Static signed-offset integer codes, with explicit serialization provenance."""

from collections.abc import Mapping

from cubic_serve.models.quantization import ActivationFormat, QuantizationError, QuantizationScheme


def parse_group(group: Mapping[str, object], weights: Mapping[str, object]) -> QuantizationScheme:
    if (
        weights.get("type") != "int"
        or weights.get("dynamic", False) is not False
        or type(weights.get("symmetric")) is not bool
        or weights.get("actorder") not in (None, False)
        or group.get("input_activations") is not None
    ):
        raise QuantizationError(
            "INT packed weights require static integers, identity activation and no actorder"
        )
    strategy = weights.get("strategy")
    size = weights.get("group_size")
    if strategy == "channel" and size in (None, -1):
        size = 2**63 - 1  # channel group is bounded to logical K by the layout
    elif strategy != "group" or type(size) is not int or size <= 0:
        raise QuantizationError("INT supports group and channel weight scales")
    packing = weights.get("packing", "auto")
    scale = weights.get("scale_dtype", "auto")
    if not isinstance(scale, str):
        raise QuantizationError("INT scale dtype must be a string")
    scale = scale.removeprefix("torch.")
    scale = {"float16": "F16", "bfloat16": "BF16", "float32": "F32"}.get(scale, scale)
    symmetry = "sym" if weights["symmetric"] else "asym"
    bits = weights.get("num_bits")
    if type(bits) is not int or type(size) is not int:
        raise QuantizationError("INT bit width and group size must be integers")
    return QuantizationScheme(
        "int-pack-v1",
        bits,
        size,
        metadata=f"{packing}:{symmetry}:{scale}",
    )


def fields(encoding: QuantizationScheme) -> list[str]:
    return encoding.metadata.split(":")


def validate_scheme(encoding: QuantizationScheme) -> None:
    parts = fields(encoding)
    if (
        encoding.bits not in range(2, 9)
        or encoding.group_out != 1
        or encoding.activation != ActivationFormat.IDENTITY
        or len(parts) != 3
        or parts[0] not in ("auto", "dense", "legacy")
        or parts[1] not in ("sym", "asym")
        or parts[2] not in ("auto", "F16", "BF16", "F32")
    ):
        raise QuantizationError("invalid INT encoding")
