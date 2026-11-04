"""Checkpoint declarations and validation owned by this extension."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cubic_serve.models.quantization import QuantizationScheme

from collections.abc import Mapping

from cubic_serve.models.quantization import (
    ActivationFormat,
    QuantizationError,
    QuantizationScheme,
    WeightFormat,
)


def parse_group(group: Mapping[str, object], weights: Mapping[str, object]) -> QuantizationScheme:
    activation = group.get("input_activations")
    if weights.get("dynamic") is not False or weights.get("symmetric") is not True:
        raise QuantizationError("checkpoint weights require static symmetric encoding")
    if (
        weights.get("num_bits") != 8
        or weights.get("type") != "float"
        or weights.get("strategy") != "channel"
    ):
        raise QuantizationError("FP8 adapter currently requires channel-scaled E4M3 weights")
    mode = ActivationFormat.IDENTITY
    if activation is not None:
        if (
            not isinstance(activation, Mapping)
            or activation.get("dynamic") is not True
            or activation.get("num_bits") != 8
            or activation.get("type") != "float"
            or activation.get("strategy") != "tensor"
            or activation.get("symmetric") is not True
        ):
            raise QuantizationError("unsupported FP8 activation declaration")
        # CT W8A8 maps dynamic tensor activation configuration to its
        # dynamic-token quantization key; this is exporter-specific.
        mode = ActivationFormat.FP8_TOKEN
    return QuantizationScheme(WeightFormat.FP8, 8, 1, activation=mode)


def validate_scheme(self: QuantizationScheme) -> None:
    if self.bits != 8 or self.group_out != 1:
        raise QuantizationError("FP8 encoding requires 8-bit channel weights")

    if self.activation not in (ActivationFormat.IDENTITY, ActivationFormat.FP8_TOKEN):
        raise QuantizationError("activation encoding is incompatible with weight format")
