"""Independent code extraction, affine reconstruction and scale-dtype rounding."""

from collections.abc import Mapping

import torch
from torch import Tensor

from cubic_serve.models.quantization import QuantizationError, QuantizationScheme
from cubic_serve.quantization.common.reference import integer_geometry
from cubic_serve.quantization.common.reference import wna8_activation as wna8_operand

from .checkpoint import fields


def _word_geometry(device, bits, count, packing, offset, words):
    def build():
        pos = torch.arange(offset, offset + count, device=device, dtype=torch.int64)
        bit = (pos * bits if packing == "dense" else
               (pos // (32 // bits)) * 32 + (pos % (32 // bits)) * bits)
        word, shift = bit // 32, bit % 32
        direct = bits > 0 and (packing != "dense" or 32 % bits == 0)
        return (
            word, shift, shift.to(torch.int32),
            None if direct else torch.clamp(word + 1, max=words - 1),
            None if direct else 32 - shift,
            None if direct else shift + bits > 32,
        )

    return integer_geometry(
        ("int.words", str(device), bits, count, packing, offset, words), build)


def _group_indices(device, count, width):
    return integer_geometry(
        ("int.groups", str(device), count, width),
        lambda: torch.arange(count, device=device) // width)


def unpack(
    packed: Tensor, bits: int, count: int, axis: int = 1, packing: str = "dense", offset: int = 0
) -> Tensor:
    """Extract signed codes directly from their original word coordinates."""
    source = packed.transpose(0, 1) if axis == 0 else packed
    word, shift, shift32, high_word, high_shift, carries = _word_geometry(
        source.device, bits, count, packing, offset, source.shape[1])
    if bits > 0 and (packing != "dense" or 32 % bits == 0):
        low = source[:, word]
        if low.dtype is torch.int32 and bits < 32:
            # The final code mask removes arithmetic sign extension: every
            # selected code is wholly within its original 32-bit word.
            value = low >> shift32
        else:
            value = (low.to(torch.int64) & 0xFFFFFFFF) >> shift
    else:
        assert high_word is not None and high_shift is not None and carries is not None
        low = source[:, word].to(torch.int64) & 0xFFFFFFFF
        high = (
            source[:, high_word].to(torch.int64)
            & 0xFFFFFFFF
        )
        value = (low >> shift) | torch.where(carries, high << high_shift, 0)
    value = ((value & ((1 << bits) - 1)) - (1 << (bits - 1))).to(torch.int8)
    return value.transpose(0, 1).contiguous() if axis == 0 else value


def decode(
    payload: Mapping[str, Tensor],
    encoding: QuantizationScheme,
    shape: tuple[int, int],
    partitions: tuple[int, ...] | None = None,
) -> Tensor:
    packing, symmetry, _ = fields(encoding)
    if packing == "auto":
        raise QuantizationError("INT packing must be resolved before decoding")
    n, k = shape
    q = unpack(payload["weight_packed"], encoding.bits, k, packing=packing).float()
    scales = payload["weight_scale"]
    if not bool((torch.isfinite(scales) & (scales > 0)).all()):
        raise QuantizationError("INT scales must be positive and finite")
    if symmetry == "asym":
        zero = unpack(payload["weight_zero_point"], encoding.bits, n, 0, packing).float()
        q = q - zero[:, _group_indices(q.device, k, min(k, encoding.group_in))]

    group = _group_indices(q.device, k, min(k, encoding.group_in))
    return (q * scales.float()[:, group]).to(scales.dtype).float()


def wna8_weight(matrix) -> Tensor:
    """Independent original-code/group-scale weight for the A8 reduction."""
    encoding, payload = matrix.encoding, matrix.payload
    packing, symmetry, _ = fields(encoding)
    n, k = matrix.shape
    codes = unpack(payload["weight_packed"], encoding.bits, k, packing=packing).float()
    groups = _group_indices(codes.device, k, min(k, encoding.group_in))
    if symmetry == "asym":
        zero = unpack(payload["weight_zero_point"], encoding.bits, n, 0, packing).float()
        codes = codes - zero[:, groups]
    scales = payload["weight_scale"].float()
    if not bool((torch.isfinite(scales) & (scales > 0)).all()):
        raise QuantizationError("INT scales must be positive and finite")
    return codes * scales[:, groups]


def wna8_output_row_batch_size(inputs: Tensor, matrix, *, byte_budget: int) -> int | None:
    """Bound an arithmetic-invariant output-row batch for integer group dots.

    Every integer product and partial sum is exact in the selected FP32/FP64
    dot when its absolute sum fits the significand. Changing independent N
    rows cannot change that dot, the fixed FP32 group-add order or final RNE.
    Other formats retain their canonical GEMM tile unless they own this proof.
    """
    k = matrix.shape[1]
    group = min(k, matrix.encoding.group_in)
    _, symmetry, _ = fields(matrix.encoding)
    maximum = ((1 << matrix.encoding.bits) - 1 if symmetry == "asym"
               else 1 << (matrix.encoding.bits - 1))
    if group * 127 * maximum > 1 << 53:
        return None
    rows = inputs.numel() // k
    groups = (k + group - 1) // group
    # Include packed-code/index/cast temporaries and full group-dot/results.
    per_output_row = 64 * k + 16 * rows * groups + 32 * rows
    return max(1, byte_budget // per_output_row)


def wna8_linear(inputs: Tensor, matrix) -> Tensor:
    """Evaluate the original INT codes with one integer dot per scale group.

    This is an independent eager arithmetic reference. FP32 rounding occurs
    after each group dot and scale multiplication, then at each group addition
    and the final token-scale multiplication. Packed-code extraction remains
    independent of the native Marlin layout and kernel.
    """
    from cubic_serve.executor.runner import require_reference_execution
    from cubic_serve.quantization.common.reference import wna8_codes_and_scale

    require_reference_execution()
    n, k = matrix.shape
    if inputs.shape[-1] != k:
        raise QuantizationError("quantized linear input differs from logical K")
    packing, symmetry, _ = fields(matrix.encoding)
    if packing == "auto":
        raise QuantizationError("INT packing must be resolved before decoding")
    codes, token_scales = wna8_codes_and_scale(inputs)
    rows = codes.reshape(-1, k)
    payload = matrix.payload
    q = unpack(payload["weight_packed"], matrix.encoding.bits, k, packing=packing)
    group_size = min(k, matrix.encoding.group_in)
    group_count = (k + group_size - 1) // group_size
    if symmetry == "asym":
        zero = unpack(
            payload["weight_zero_point"], matrix.encoding.bits, n, 0, packing
        ).to(torch.int16)
        q = q.to(torch.int16) - zero[:, torch.arange(k, device=q.device) // group_size]
    scales = payload["weight_scale"].float()
    if scales.shape != (n, group_count) or not bool(
        (torch.isfinite(scales) & (scales > 0)).all()
    ):
        raise QuantizationError("INT group scales must be positive, finite and match K")

    padded_k = group_count * group_size
    if padded_k != k:
        q = torch.nn.functional.pad(q, (0, padded_k - k))
        rows = torch.nn.functional.pad(rows, (0, padded_k - k))
    # Every FP32 partial is an exact integer when this conservative bound fits
    # its 24-bit significand. Larger groups use FP64 for the integer dot, then
    # explicitly round to FP32 at the native int32-to-float boundary.
    maximum_code = (1 << matrix.encoding.bits) - 1 if symmetry == "asym" else (
        1 << (matrix.encoding.bits - 1)
    )
    exact_fp32_dot = group_size * 127 * maximum_code <= 1 << 24
    if inputs.is_cuda and torch.backends.cuda.matmul.allow_tf32:
        exact_fp32_dot = False
    dot_dtype = torch.float32 if exact_fp32_dot else torch.float64
    a = rows.reshape(-1, group_count, group_size).permute(1, 0, 2).to(dot_dtype)
    b = q.reshape(n, group_count, group_size).permute(1, 2, 0).to(dot_dtype)
    dots = torch.bmm(a, b)
    accumulated = torch.zeros((rows.shape[0], n), device=inputs.device, dtype=torch.float32)
    for group in range(group_count):
        accumulated = accumulated + dots[group].float() * scales[:, group]
    result = accumulated * token_scales.reshape(-1, 1)
    return result.reshape(*inputs.shape[:-1], n).to(inputs.dtype)
