"""Native attention candidate bound to the existing validated cache/graph ABI."""

from pathlib import Path

import torch

from cubic_serve.ops.cuda.compiled_artifact import bind_tensor
from cubic_serve.ops.cuda.integer_division import invariant_divisor
from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.runtime.cache.quantization import CacheDType, CacheGroupAxis


def bind_native_decode(primitive, query, key, value, table, lengths, output, *,
                       partial=None, statistics=None, arrivals=None, selected_indices=None):
    """Use the common validator and launch portfolio, with no runtime capture."""
    primitive.validate_bound_operands(query, key, value, table, lengths, output)
    if primitive._attention_precision != "auto" or primitive._encoding is None:
        raise ValueError("native encoded attention requires native QKV MMA operands")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("native attention must be bound before graph capture")
    encoding, window = primitive._encoding, primitive._window
    _, layers, page, heads, dim = encoding.validate(key, value)
    padded_dim = (dim + 15) // 16 * 16
    shared = 48 * padded_dim + 6080
    rotate_value = encoding.enable_rotation and encoding.value.spec.dtype.quantized
    if rotate_value:
        # Fixed merge coefficients occupy 1024 floats. The independent inverse
        # uses a disjoint FP32 tile before the only native output rounding.
        shared = max(shared, 4288 + 64 * padded_dim)
    if shared > torch.cuda.get_device_properties(query.device).shared_memory_per_block_optin:
        raise ValueError("native attention tile exceeds the device shared-memory limit")
    kind = {"fp32": 0, "fp16": 1, "bf16": 2}[str(primitive._activation_type)]
    k, v = encoding.key.spec, encoding.value.spec
    definitions = dict(
        INPUT_BF16=int(query.dtype == torch.bfloat16), ACT_KIND=kind,
        CACHE_WINDOW=int(window is not None), KEY_ENCODED=int(k.dtype.quantized),
        CACHE_ACTIVITY=int(window is not None and window.activity is not None),
        VALUE_ENCODED=int(v.dtype.quantized), KEY_COLUMN=int(k.axis is CacheGroupAxis.TOKENS),
        VALUE_ROTATION=int(rotate_value),
        VALUE_COLUMN=int(v.axis is CacheGroupAxis.TOKENS),
        KEY_INTEGER=int(k.dtype in (CacheDType.INT8, CacheDType.INT6, CacheDType.INT4)),
        VALUE_INTEGER=int(v.dtype in (CacheDType.INT8, CacheDType.INT6, CacheDType.INT4)),
        KEY_BITS=k.dtype.bits if k.dtype.quantized else 8,
        VALUE_BITS=v.dtype.bits if v.dtype.quantized else 8,
        # Coarse power-of-two fragment capacities; actual D, M and context
        # stay runtime values, so one image covers neighboring head dimensions.
        REGISTER_TILES=1 << (((padded_dim + 63) // 64 - 1).bit_length()),
    )
    prefix = "".join(f"#define {name} {value}\n" for name, value in definitions.items())
    source = Path(__file__)
    module, entry = compile_native(
        (("entry.cu", prefix.encode() + source.with_suffix(".cu").read_bytes()),
         ("activation_native.cuh", (source.parents[1] / "activation_native.cuh").read_bytes()),
         ("mma.cuh", (source.parents[1] / "mma.cuh").read_bytes())),
        query.device, "paged_decode_native", minimum_isa=90 if kind == 2 else 80,
    )
    workspace = tuple(t for t in (partial, statistics, arrivals) if t is not None)
    selection = () if selected_indices is None else (selected_indices,)
    if (tuple(bind_tensor(t) for t in workspace) != primitive._workspace
            or tuple(bind_tensor(t) for t in selection) != primitive._selection):
        raise ValueError("native attention scratch/selection differ from the validated binding")
    selected = selected_indices
    scalar_values = (layers, primitive.layer, page, table.shape[1], query.shape[1], heads,
                     dim, padded_dim, primitive.grid[2],
                     selected.shape[1] if selected is not None else 0,
                     k.group_size if k.dtype.quantized else 0,
                     v.group_size if v.dtype.quantized else 0,
                     int(k.axis is CacheGroupAxis.TOKENS), int(v.axis is CacheGroupAxis.TOKENS),
                     int(window is not None))
    launch = module.bind(
        entry,
        (query, key, value, table, lengths, output, partial, statistics, arrivals, selected,
         encoding.key_scales, encoding.value_scales,
         *((None,) * 7 if window is None else (
             window.key, window.value, window.key_tail, window.value_tail,
             window.start_positions, window.start_rows, window.sources)),
         None if window is None else window.activity,
         *(CudaScalar("i32", x) for x in scalar_values), CudaScalar("f32", primitive.scale),
         *(CudaScalar("i32", x) for divisor in (
             page, padded_dim, k.group_size if k.dtype.quantized else 1,
             v.group_size if v.dtype.quantized else 1,
         ) for x in invariant_divisor(divisor)),
         *(CudaScalar("i64", stride) for stride in (
             key.stride(0) * key.element_size(), value.stride(0) * value.element_size(),
             0 if encoding.key_scales is None else encoding.key_scales.stride(0),
             0 if encoding.value_scales is None else encoding.value_scales.stride(0),
             0 if window is None or window.key_tail is None else window.key_tail.stride(0),
             0 if window is None or window.value_tail is None else window.value_tail.stride(0),
         ))),
        operator_id="cubic.cuda.encoded-paged-gqa.mma16816.v1", grid=primitive.grid,
        block=(128, 1, 1), shared_bytes=shared,
    )
    return launch
