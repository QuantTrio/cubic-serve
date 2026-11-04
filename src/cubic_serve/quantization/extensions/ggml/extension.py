from cubic_serve.quantization.api import QuantizationExtension

EXTENSION = QuantizationExtension("ggml", "ggml-block-v1", __package__,
                                 checkpoint_methods=("gguf",))
