"""Checkpoint declarations and validation owned by this extension."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cubic_serve.models.quantization import EncodingRule, QuantizationScheme

from cubic_serve.models.quantization import (
    ActivationFormat,
    QuantizationError,
    QuantizationScheme,
    WeightFormat,
    _positive_int,
    _strings,
)


def parse_group(group: Mapping[str, object], weights: Mapping[str, object]) -> QuantizationScheme:
    activation = group.get("input_activations")
    if weights.get("dynamic") is not False or weights.get("symmetric") is not True:
        raise QuantizationError("checkpoint weights require static symmetric encoding")
    if activation is not None:
        raise QuantizationError("Cubic runtime activation transforms require a separate policy")
    size = weights.get("group_size")
    if isinstance(size, list) and len(size) == 2:
        group_out, group_in = map(_positive_int, size)
    else:
        group_out, group_in = 1, _positive_int(size)
    bits = _positive_int(weights.get("num_bits"))
    if weights.get("packing") != "little-endian-bitstream":
        raise QuantizationError("Cubic requires little-endian bitstream packing")
    if weights.get("reserved_code") != ("binary" if bits == 1 else "zero"):
        raise QuantizationError("Cubic reserved-code convention differs")
    metadata = weights.get("metadata_format", "float32-scale-float16-ab")
    dtypes = {
        "float32-scale-float16-ab": (
            ("torch.float32", "float32"),
            ("torch.float16", "float16"),
        ),
        "int8-scale-int4-ab-global-fp32": (("torch.int8", "int8"), ("packed_int4",)),
        "e5m9-scale-curve2-tensor-fp16": (("packed_e5m9",), ("torch.float16", "float16")),
    }
    if (
        metadata not in dtypes
        or weights.get("scale_dtype") not in dtypes[metadata][0]
        or weights.get("param_dtype") not in dtypes[metadata][1]
    ):
        raise QuantizationError("Cubic metadata dtype declaration differs from its format")
    if metadata == "e5m9-scale-curve2-tensor-fp16" and weights.get("curve_count") != 4:
        raise QuantizationError("Cubic curve2 requires four curves")
    return QuantizationScheme(WeightFormat.CUBIC, bits, group_in, group_out, metadata)


def parse_config(config: Mapping[str, object]) -> tuple[tuple[EncodingRule, ...], tuple[str, ...]]:
    from cubic_serve.quantization.common.checkpoint import parse_groups

    if config.get("format") != "cubic-pack-quantized":
        raise QuantizationError("Cubic requires its explicit packed format")
    return parse_groups(config, parse_group), _strings(config.get("ignore", []))


def validate_scheme(self: QuantizationScheme) -> None:
    if self.bits not in (1, 2, 3, 4, 5, 6, 7, 8):
        raise QuantizationError("unsupported Cubic bit width")
    if self.metadata not in (
        "float32-scale-float16-ab",
        "int8-scale-int4-ab-global-fp32",
        "e5m9-scale-curve2-tensor-fp16",
    ):
        raise QuantizationError("unsupported Cubic metadata")

    if self.activation not in (ActivationFormat.IDENTITY,):
        raise QuantizationError("activation encoding is incompatible with weight format")
