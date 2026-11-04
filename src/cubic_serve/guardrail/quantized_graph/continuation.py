"""Bounded live audit prefixes, never qualification receipts or math memo entries.

Only completed, independently witnessed transitions may issue a prefix. Native,
causal and free-reference states are owned separately. Restoring returns fresh
copies, so a rejected speculative branch cannot mutate its accepted predecessor.
"""
from dataclasses import fields, is_dataclass
import weakref

import torch


def _tree(value, tensor):
    if isinstance(value, torch.Tensor):
        return tensor(value)
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{f.name: _tree(getattr(value, f.name), tensor) for f in fields(value)})
    if type(value) is tuple:
        return tuple(_tree(v, tensor) for v in value)
    if type(value) is list:
        return [_tree(v, tensor) for v in value]
    if type(value) is dict and all(type(k) in (str, int) for k in value):
        return {k: _tree(v, tensor) for k, v in value.items()}
    raise TypeError("unsupported live audit state schema")


def _signature(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", id(value), tuple(value.shape), tuple(value.stride()),
                value.dtype, value._version)
    if value is None or type(value) in (bool, int, float, str, bytes):
        return (type(value), value)
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value), tuple((f.name, _signature(getattr(value, f.name)))
                                   for f in fields(value)))
    if type(value) in (tuple, list):
        return (type(value), tuple(_signature(v) for v in value))
    if type(value) is dict:
        return (dict, tuple((k, _signature(v)) for k, v in value.items()))
    raise TypeError("unsupported live audit state schema")


class AuditPrefix:
    __slots__ = ("_store", "_binding", "_value", "_signature", "_tensors", "bytes", "_release", "__weakref__")

    def __init__(self, store, binding, value, size):
        self._store, self._binding, self.bytes = store, binding, size
        self._tensors = []
        def freeze(tensor):
            if tensor.layout is not torch.strided:
                raise TypeError("live audit prefix requires strided tensor storage")
            with torch.inference_mode(False), torch.no_grad():
                image = tensor.detach().to("cpu", copy=True).contiguous()
            self._tensors.append((image, image._version, tuple(image.shape), image.dtype))
            return image
        self._value = _tree(value, freeze)
        self._signature = _signature(self._value)
        self._release = weakref.finalize(self, store._retire, size)

    def restore(self, store, binding, device):
        if store is not self._store or binding != self._binding or not self._release.alive:
            raise ValueError("live audit prefix source or lifetime differs")
        for tensor, version, shape, dtype in self._tensors:
            if tensor._version != version or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError("immutable live audit prefix was modified")
        if _signature(self._value) != self._signature:
            raise ValueError("immutable live audit prefix schema was modified")
        with torch.inference_mode(False), torch.no_grad():
            return _tree(self._value, lambda t: t.to(device, copy=True))

    def close(self):
        if self._release.alive:
            self._release()
        self._value = None
        self._tensors.clear()


class AuditPrefixStore:
    """Account every retained prefix before copying any of its tensor payload."""
    def __init__(self, byte_budget):
        if type(byte_budget) is not int or byte_budget <= 0:
            raise ValueError("live audit prefix budget must be positive")
        self.byte_budget, self.bytes, self.peak_bytes = byte_budget, 0, 0

    def _retire(self, size):
        self.bytes -= size
        if self.bytes < 0:
            raise RuntimeError("live audit prefix ownership underflow")

    def freeze(self, binding, value):
        size = 0
        def count(tensor):
            nonlocal size
            if tensor.layout is not torch.strided:
                raise TypeError("live audit prefix requires strided tensor storage")
            size += tensor.numel() * tensor.element_size()
            return tensor
        _tree(value, count)
        if self.bytes + size > self.byte_budget:
            raise MemoryError("live audit prefixes exceed admitted host witness memory")
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        try:
            return AuditPrefix(self, binding, value, size)
        except BaseException:
            self._retire(size)
            raise
