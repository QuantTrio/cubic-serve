"""Process-local exclusion for startup transfers and isolated calibration.

CPU checkpoint reads remain concurrent. CUDA capture and timing must not overlap
another thread's allocation, synchronization or transfer on the same device.
This is pre-readiness coordination, never a serving replay lease.
"""
from functools import cache
from threading import RLock
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_WEIGHT_IDENTITY = ContextVar("startup_weight_identity", default=None)
_ALLOW_CALIBRATION = ContextVar("startup_allow_calibration", default=False)


@contextmanager
def startup_weight_identity(identity, *, allow_calibration=True):
    """Bind tuning with real weights to one admitted startup source/ownership."""
    token = _WEIGHT_IDENTITY.set(identity)
    calibration_token = _ALLOW_CALIBRATION.set(allow_calibration)
    try:
        yield
    finally:
        _WEIGHT_IDENTITY.reset(token)
        _ALLOW_CALIBRATION.reset(calibration_token)


def require_calibration_permission(schema):
    if not _ALLOW_CALIBRATION.get():
        raise RuntimeError(f"operator candidate calibration requires explicit qualification ({schema})")


def calibration_allowed():
    """Only explicit tuning may search candidates; cache misses are ordinary."""
    return _ALLOW_CALIBRATION.get()


def weight_bound_calibration_key(key):
    # Runtime weights are not an operator implementation. Their encoding and
    # operand geometry belong to the caller's entry, not a model-wide key.
    return None if _WEIGHT_IDENTITY.get() is None else key


@cache
def _device_lock(index: int):
    return RLock()


def startup_cuda_lock(device: torch.device | str):
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('startup CUDA exclusion requires a CUDA device')
    return _device_lock(torch.cuda.current_device() if device.index is None else device.index)
