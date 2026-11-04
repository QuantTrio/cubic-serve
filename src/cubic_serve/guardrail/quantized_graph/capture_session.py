"""Bounded, exclusively borrowed observation graphs for one offline audit.

Only construction is reused. Every borrow resets the full owned state and runs
all required causal/free-reference proofs again on the actual current inputs.
No numerical verdict, trajectory or state snapshot is cached here.
"""
from collections import OrderedDict


OBSERVATION_FIELDS = ('stage', 'records', 'linear_scales', 'ordered_events', 'normalizations', 'index_query_norms', 'index_key_norms', 'pre_routes', 'route_events', 'recorder', 'attention_events', 'cache_events', 'gate_events', 'selector_events', 'gdn_norm_events', 'gdn_events', 'conv_events', 'scaled_events', 'mix_events', 'combine_events', 'moe_events', 'moe_combine_events', 'signed_events', 'dilated_events', 'captured_experts')


class AuditCaptures:
    def __init__(self, byte_budget, *, reserve_bytes=0):
        if type(byte_budget) is not int or byte_budget < 0:
            raise ValueError("audit capture budget must be nonnegative")
        self.byte_budget = byte_budget
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError("audit capture reserve must be nonnegative")
        self.reserve_bytes = reserve_bytes
        self.bytes = self.peak_bytes = self.hits = self.misses = 0
        self.entries = OrderedDict()
        self.borrowed = set()

    def prepare(self, device):
        """Retire only completed audit captures when construction needs room."""
        import torch
        from cubic_serve.runtime.quantized_qualification import _draft_reference_headroom
        expected = max((size for _, size in self.entries.values()), default=0)
        while self.entries and _draft_reference_headroom(device) < self.reserve_bytes + expected:
            key = next((key for key in self.entries if key not in self.borrowed), None)
            if key is None:
                raise MemoryError("all audit observation graphs are exclusively borrowed")
            self.discard(key)
            torch.cuda.empty_cache()

    def acquire(self, binding):
        if binding in self.borrowed:
            raise RuntimeError("audit observation graph is already exclusively borrowed")
        entry = self.entries.get(binding)
        if entry is None:
            self.misses += 1
            return None
        self.entries.move_to_end(binding)
        self.borrowed.add(binding)
        self.hits += 1
        return entry[0]

    @staticmethod
    def _close(payload):
        stage = payload['stage']
        try:
            stage.close()
        finally:
            if 'launch' in getattr(stage, '__dict__', {}):
                del stage.launch
            payload.clear()

    def retain(self, binding, payload, size):
        if type(size) is not int or size < 0:
            raise ValueError("audit capture size must be nonnegative")
        if binding in self.entries or binding in self.borrowed:
            raise ValueError("audit capture binding is already owned")
        if size == 0 or size > self.byte_budget:
            return False
        while self.entries and self.bytes + size > self.byte_budget:
            key = next((key for key in self.entries if key not in self.borrowed), None)
            if key is None:
                return False
            self.discard(key)
        self.entries[binding] = payload, size
        self.borrowed.add(binding)
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        return True

    def release(self, binding):
        if binding not in self.borrowed:
            raise RuntimeError("audit observation graph has no exclusive borrower")
        self.borrowed.remove(binding)

    def discard(self, binding):
        self.borrowed.discard(binding)
        entry = self.entries.pop(binding, None)
        if entry is not None:
            payload, size = entry
            self.bytes -= size
            self._close(payload)

    def clear(self):
        if self.borrowed:
            raise RuntimeError("cannot retire an in-use audit observation graph")
        for key in tuple(self.entries):
            self.discard(key)
