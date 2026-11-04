"""Original GGML block identities; the GGUF container selects each tensor's type."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING

from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

from cubic_serve.models.quantization import ActivationFormat, QuantizationError

if TYPE_CHECKING:
    from cubic_serve.models.quantization import EncodingRule, QuantizationScheme

FORMAT_ID = "ggml-block-v1"
_TYPES = {"Q8_0": 8, "Q6_K": 6, "IQ3_S": 3, "IQ4_NL": 4, "IQ4_XS": 4}


def kind(encoding: QuantizationScheme) -> GGMLQuantizationType:
    try:
        return GGMLQuantizationType[encoding.metadata.split(":", 1)[0]]
    except KeyError as error:
        raise QuantizationError("GGML encoding requires an explicit original block type") from error


def validate_scheme(encoding: QuantizationScheme) -> None:
    quant = kind(encoding)
    if (quant.name not in _TYPES or encoding.format != FORMAT_ID
            or encoding.activation != ActivationFormat.IDENTITY):
        raise QuantizationError("unsupported GGML block or activation encoding")
    block, _ = GGML_QUANT_SIZES[kind(encoding)]
    column_offset(encoding)
    if (encoding.bits != _TYPES[quant.name] or encoding.group_in != block
            or encoding.group_out != 1):
        raise QuantizationError("GGML block geometry cannot be overridden")


def column_offset(encoding: QuantizationScheme) -> int:
    parts = encoding.metadata.split(":")
    if len(parts) == 1:
        return 0
    if (len(parts) != 2 or not parts[1].isascii() or not parts[1].isdecimal()
            or str(int(parts[1])) != parts[1]
            or not 0 < int(parts[1]) < GGML_QUANT_SIZES[kind(encoding)][0]):
        raise QuantizationError("GGML logical input offset must lie inside its original block")
    return int(parts[1])


@lru_cache(maxsize=128)
def scheme(type_id: int, offset: int = 0) -> QuantizationScheme:
    from cubic_serve.models.quantization import QuantizationScheme

    quant = GGMLQuantizationType(type_id)
    if quant.name not in _TYPES:
        raise QuantizationError(f"GGML weight encoding {quant.name} is not implemented")
    return QuantizationScheme(FORMAT_ID, _TYPES[quant.name], GGML_QUANT_SIZES[quant][0],
                              metadata=quant.name + (f":{offset}" if offset else ""))


def parse_config(config: Mapping[str, object]) -> tuple[tuple[EncodingRule, ...], tuple[str, ...]]:
    # Encoding declarations are physical tensor headers, never an inherited
    # checkpoint-wide nominal bit width or filename quantization label.
    if dict(config) != {"quant_method": "gguf"}:
        raise QuantizationError("GGUF encoding declarations must come from admitted tensor headers")
    return (), ()
