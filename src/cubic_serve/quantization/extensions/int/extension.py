from cubic_serve.quantization.api import QuantizationExtension

EXTENSION = QuantizationExtension("int", "int-pack-v1", __package__, (), ("pack-quantized",))
