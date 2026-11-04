"""Portable integer operands retain exact address arithmetic without CUDA division."""

import random

import pytest

from cubic_serve.ops.cuda.integer_division import invariant_divisor


def test_invariant_division_matches_integer_quotient_and_remainder():
    # Include power transitions, non-powers, and signed ABI multiplier bits.
    rng = random.Random(1729)
    divisors = {1, 3, 5, 7, 31, 64, 128, 192, 320, 768, (1 << 31) - 1}
    divisors.update(d for p in range(1, 31) for d in ((1 << p) - 1, 1 << p, (1 << p) + 1))
    divisors.update(rng.randrange(1, 1 << 31) for _ in range(128))
    for divisor in divisors:
        multiplier, shift = invariant_divisor(divisor)
        multiplier &= (1 << 32) - 1
        values = {0, 1, divisor - 1, divisor, (1 << 31) - 1}
        values.update(rng.randrange(1 << 31) for _ in range(256))
        for q in (1, 2, 3, 7, 31, ((1 << 31) - 1) // divisor):
            values.update(q * divisor + d for d in (-1, 0, 1)
                          if 0 <= q * divisor + d < (1 << 31))
        for value in values:
            quotient = (((value * multiplier) >> 32) + value) >> shift
            assert quotient == value // divisor
            assert value - quotient * divisor == value % divisor


@pytest.mark.parametrize("divisor", [0, -1, 1 << 31, 1.0, True])
def test_invariant_division_rejects_values_outside_declared_abi(divisor):
    with pytest.raises(ValueError):
        invariant_divisor(divisor)


def test_int_marlin_recipe_reuses_runtime_dimensions_and_group_geometry():
    from cubic_serve.quantization.common.marlin.parameters import MarlinParameters
    from cubic_serve.quantization.extensions.int.marlin_source import _source

    params = MarlinParameters()
    def source(n, k, group, partitions):
        return _source(params, "kBFloat16", 4, group, 16, "F32", False,
                       partitions, n=n, k=k)
    assert source(320, 2560, 128, 4) == source(768, 3072, 192, 2)
