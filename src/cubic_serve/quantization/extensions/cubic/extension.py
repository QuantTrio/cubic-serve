"""Registration entry point for cubic; imports no execution backend."""

from cubic_serve.quantization.api import QuantizationExtension

EXTENSION = QuantizationExtension("cubic", "cubic-bitstream-v1", __package__, ("cubic",), ())
