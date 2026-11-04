"""Registration entry point for fp8; imports no execution backend."""

from cubic_serve.quantization.api import QuantizationExtension

EXTENSION = QuantizationExtension(
    "fp8", "fp8-e4m3-channel-v1", __package__, (), ("float-quantized",)
)
