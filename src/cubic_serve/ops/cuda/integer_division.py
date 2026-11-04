"""Exact invariant-divisor parameters for nonnegative signed-32-bit indices.

The device quotient is (umulhi(n, multiplier) + n) >> shift, with
0 <= n <= INT32_MAX and 1 <= divisor <= INT32_MAX. This is the bounded
integer rule used by Torch ATen/cuda/detail/IntegerDivider.cuh and llama.cpp
839a572a, ggml/src/ggml-cuda/common.cuh; it changes no floating arithmetic.
"""

def invariant_divisor(divisor: int) -> tuple[int, int]:
    """Return multiplier bits as i32 plus shift for the sealed operand ABI."""
    if type(divisor) is not int or not 1 <= divisor < (1 << 31):
        raise ValueError("invariant division requires a positive signed-32-bit divisor")
    shift = (divisor - 1).bit_length()
    multiplier = ((1 << 32) * ((1 << shift) - divisor)) // divisor + 1
    if not 0 < multiplier < (1 << 32):
        raise ValueError("invariant divisor multiplier exceeds its transport ABI")
    return multiplier if multiplier < (1 << 31) else multiplier - (1 << 32), shift
