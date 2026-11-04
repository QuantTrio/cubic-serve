"""GGML serialization tables, shared as data rather than decoder implementations."""

from functools import cache

import numpy as np


@cache
def iq3_grid() -> np.ndarray:
    from gguf.quants import IQ3_S

    IQ3_S.init_grid()  # type: ignore[no-untyped-call]
    assert IQ3_S.grid is not None
    grid = np.array(IQ3_S.grid.reshape(512, 4), dtype=np.uint8, copy=True)
    grid.flags.writeable = False
    return grid
