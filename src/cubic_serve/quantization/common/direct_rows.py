"""NInfer's complete direct split-four contraction with format-owned operands.

This is a pre-readiness qualification primitive, not a serving selector. It
keeps the upstream execution organization while original compact formats own
decoding and rounding. A block epilogue must preserve each storage boundary.
"""
from pathlib import Path

import torch

from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.ops.cuda.activation import activation_type
from cubic_serve.quantization.common.binding import PackedOperatorBinding
from cubic_serve.quantization.common.small_rows import SmallRowsWeight


def bind(x: torch.Tensor, weight: SmallRowsWeight, output: torch.Tensor, *,
         convolution: tuple[torch.Tensor, ...] | None = None,
         joined: tuple[SmallRowsWeight, torch.Tensor] | None = None) -> PackedOperatorBinding:
    """Optionally bind NInfer's segmented projection/convolution epilogue.

    Convolution operands are (weight, state, sources, destinations, segment
    lengths, logical row activity, output). The graph owner must validate all
    index generations/write ownership and lease every checkpoint destination.
    This primitive never chooses an operator, captures, or admits requests.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("direct rows binding must precede capture")
    n, k = weight.shape
    if (not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16)
            or x.ndim != 2 or x.shape[1] != k or not 1 <= x.shape[0] <= 32
            or k % 128 or not 2 <= weight.bits <= 8 or weight.group < 32
            or weight.group % 32 or k % weight.group
            or weight.codes.dtype != torch.uint8 or weight.codes.shape != (n, k*weight.bits//8)
            or weight.metadata.dtype != torch.uint8 or weight.metadata.ndim != 1
            or output.shape != (x.shape[0], n) or output.dtype != x.dtype
            or any(not t.is_contiguous() or t.device != x.device or t.data_ptr()%16
                   for t in (x, weight.codes, weight.metadata, output))):
        raise ValueError("direct rows operands do not satisfy the packed ABI")
    for t in (x, weight.codes, weight.metadata):
        if (output.data_ptr() < t.data_ptr()+t.numel()*t.element_size()
                and t.data_ptr() < output.data_ptr()+output.numel()*output.element_size()):
            raise ValueError("direct rows output aliases an input")
    definitions = dict(OUTPUT_N=str(n), INPUT_K=str(k), ROWS=str(x.shape[0]),
                       BITS=str(weight.bits), GI=str(weight.group), TILE_K=str(weight.group),
                       ELEMENTS="8", IS_BF16=str(int(x.dtype == torch.bfloat16)),
                       FUSE_CONV=str(int(convolution is not None)), JOINED=str(int(joined is not None)))
    minimum_isa = 80
    extra_sources = ()
    if convolution is not None:
        if len(convolution) != 7:
            raise ValueError("direct convolution requires seven explicit operands")
        conv_weight, state, sources, destinations, segments, activity, convolved = convolution
        if (conv_weight.ndim != 2 or conv_weight.shape[0] != n
                or not 2 <= conv_weight.shape[1] <= 32
                or state.ndim != 3 or state.shape[1:] != (n, conv_weight.shape[1]-1)
                or state.shape[0] < 1 or convolved.shape != output.shape
                or any(t.dtype != x.dtype for t in (conv_weight, state, convolved))
                or any(t.dtype != torch.int32 or t.shape != (x.shape[0],)
                       for t in (sources, destinations, segments, activity))
                or any(t.device != x.device or not t.is_contiguous() for t in convolution)):
            raise ValueError("direct convolution differs from segmented checkpoint ABI")
        writable = (output, state, convolved)
        readable = (x, weight.codes, weight.metadata, conv_weight, sources,
                    destinations, segments, activity)
        for index, target in enumerate(writable):
            for other in (*readable, *writable[index+1:]):
                if (target.data_ptr() < other.data_ptr()+other.numel()*other.element_size()
                        and other.data_ptr() < target.data_ptr()+target.numel()*target.element_size()):
                    raise ValueError("direct convolution has overlapping storage")
        arithmetic = str(activation_type(x.dtype, x.device))
        kind = {"fp32": 0, "fp16": 1, "bf16": 2}[arithmetic]
        minimum_isa = 90 if kind == 2 else 80
        definitions.update(CONV_WIDTH=str(conv_weight.shape[1]), ACT_KIND=str(kind))
        header = Path(__file__).parents[2]/"ops"/"cuda"/"activation_native.cuh"
        extra_sources = (("activation_native.cuh", header.read_bytes()),)
    if set(definitions).intersection(dict(weight.constants)):
        raise ValueError("format cannot redefine the direct contraction")
    definitions.update(weight.constants)
    other_arguments: tuple[torch.Tensor, ...] = ()
    total_columns = n
    if joined is not None:
        other, other_output = joined
        other_n, other_k = other.shape
        first_constants, second_constants = dict(weight.constants), dict(other.constants)
        variable_storage = {"DESCRIPTOR_COUNT", "RAW_OFFSET", "ZERO_OFFSET"}
        if (other_k != k or other_n < 1 or other.bits != weight.bits or other.group != weight.group
                or other.decoder != weight.decoder
                or {key:value for key,value in first_constants.items() if key not in variable_storage}
                   != {key:value for key,value in second_constants.items() if key not in variable_storage}
                or other.codes.shape != (other_n,k*other.bits//8) or other.codes.dtype!=torch.uint8
                or other.metadata.ndim!=1 or other.metadata.dtype!=torch.uint8
                or other_output.shape!=(x.shape[0],other_n) or other_output.dtype!=x.dtype
                or any(t.device!=x.device or not t.is_contiguous() or t.data_ptr()%16
                       for t in (other.codes,other.metadata,other_output))):
            raise ValueError("joined projections require compatible original format arithmetic")
        writable = (output, other_output, *((state, convolved) if convolution is not None else ()))
        readable = (x, weight.codes, weight.metadata, other.codes, other.metadata,
                    *((conv_weight,sources,destinations,segments,activity) if convolution is not None else ()))
        for index,target in enumerate(writable):
            for source in (*readable,*writable[index+1:]):
                if (target.data_ptr()<source.data_ptr()+source.numel()*source.element_size()
                        and source.data_ptr()<target.data_ptr()+target.numel()*target.element_size()):
                    raise ValueError("joined projection output aliases an operand")
        if "DESCRIPTOR_COUNT" in first_constants:
            # The owning terminal descriptor always exceeds its local column:
            # additional bounded comparisons stay on that descriptor, without
            # reading another tensor's raw metadata or changing partition order.
            definitions["DESCRIPTOR_COUNT"] = str(max(int(first_constants["DESCRIPTOR_COUNT"]),
                                                       int(second_constants["DESCRIPTOR_COUNT"])))
        definitions.update(OTHER_N=str(other_n), OTHER_RAW_OFFSET=second_constants.get("RAW_OFFSET","0"),
                           OTHER_ZERO_OFFSET=second_constants.get("ZERO_OFFSET","0"))
        other_arguments=(other.codes,other.metadata,other_output)
        total_columns+=other_n
    definitions["packed_direct_rows"] = (
        f"packed_direct_rows_n{n}_k{k}_m{x.shape[0]}_b{weight.bits}_g{weight.group}"
        + (f"_join{joined[0].shape[0]}" if joined is not None else "")
        + ("_conv" if convolution is not None else ""))
    prefix = "".join(f"#define {key} {value}\n" for key, value in definitions.items()).encode()
    module, entry = compile_native((("entry.cu", prefix+Path(__file__).with_suffix(".cu").read_bytes()),
                                    ("decoder.cuh", weight.decoder), *extra_sources),
                                   x.device, "packed_direct_rows", minimum_isa=minimum_isa)
    launch = module.bind(entry, (x, weight.codes, weight.metadata, output, *other_arguments, *(convolution or ())),
                         operator_id=("cubic.cuda.packed.direct_rows_conv.v1" if convolution is not None
                                      else "cubic.cuda.packed.direct_rows.v1"),
                         grid=(total_columns, 1, 1), block=(128, 1, 1))
    return PackedOperatorBinding(output, (launch,), row_tile_count=1)
