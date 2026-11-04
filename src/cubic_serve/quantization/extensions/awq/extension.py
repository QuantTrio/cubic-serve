"""Registration entry point for awq; imports no execution backend."""

from cubic_serve.quantization.api import QuantizationExtension

EXTENSION = QuantizationExtension(
    "awq", "awq-gemm-v1", __package__, ("awq",), (), ignore_substrings=True
)
