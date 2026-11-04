"""Independent mathematical reference; it is never a public serving executor."""

from pathlib import Path

import torch

from cubic_serve.models.quantized_checkpoint import QuantizedCheckpoint
from cubic_serve.models.qwen3_5.model import Qwen3_5ReferenceModel
from cubic_serve.ops.eager.quantization import QuantizedReferenceOperators
from cubic_serve.runtime.quantized import load_quantized_model


def load_quantized_reference(
    model_root: str | Path,
    *,
    device: torch.device | str = "cuda:0",
    dtype: str = "bf16",
    recurrent_state_dtype: torch.dtype = torch.bfloat16,
    output_rows: int = 256,
    include_mtp: bool = True,
    compute_mode: str = "wna16",
) -> tuple[Qwen3_5ReferenceModel, QuantizedCheckpoint]:
    return load_quantized_model(
        model_root,
        operators=QuantizedReferenceOperators(output_rows, compute_mode=compute_mode),
        device=device,
        dtype=dtype,
        recurrent_state_dtype=recurrent_state_dtype,
        include_mtp=include_mtp,
    )
