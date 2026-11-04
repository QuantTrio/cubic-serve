"""Registration entry point for nvfp4; imports no execution backend."""

from cubic_serve.quantization.api import QuantizationExtension

EXTENSION = QuantizationExtension(
    "nvfp4", "nvfp4-compressed-tensors-v1", __package__, ("modelopt",), ("nvfp4-pack-quantized",)
)
