import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from cubic_serve.runtime.model_state import CONV_STATE_ID, RECURRENT_STATE_ID
from tools.qwen3_5.concurrent_audit import freeze_review_prefix, restore_frozen_prefix


@dataclass
class Layer:
    key: object = None
    value: object = None
    convolution: object = None
    recurrent: object = None


@dataclass
class Cache:
    layers: tuple
    layer_start: int = 0
    sequence_length: int = 0


class ConcurrentAuditTests(unittest.TestCase):
    def fixture(self, position=3):
        key = torch.arange(6 * 1 * 4 * 2 * 3).reshape(6, 1, 4, 2, 3).bfloat16()
        conv = torch.arange(2 * 1 * 3 * 3).reshape(2, 1, 3, 3).bfloat16()
        recurrent = conv.unsqueeze(-1).clone()
        body = SimpleNamespace(
            hidden=torch.empty(1),
            arena=SimpleNamespace(
                key=key,
                value=-key,
                written=torch.ones(6, 1, 4, dtype=torch.bool),
                states={CONV_STATE_ID: conv, RECURRENT_STATE_ID: recurrent},
            ),
        )
        # The destination page is still poison until the review graph performs COW.
        head = SimpleNamespace(
            position=position, page_table=(5, 4), page_source=1, state_source_slot=1
        )
        return body, head

    def test_snapshot_uses_cow_source_and_survives_owner_overwrite(self):
        body, head = self.fixture()
        expected = body.arena.key[1, :, :3].clone()
        state = body.arena.states[RECURRENT_STATE_ID][1].clone()
        prefix = freeze_review_prefix(body, head)
        body.arena.key.fill_(float("nan"))
        body.arena.states[RECURRENT_STATE_ID].zero_()
        self.assertTrue(torch.equal(prefix["kv_key"], expected))
        self.assertTrue(torch.equal(prefix["recurrent"], state))

    def test_page_boundary_does_not_replace_completed_prefix_page(self):
        body, head = self.fixture(position=4)
        prefix = freeze_review_prefix(body, head)
        self.assertTrue(torch.equal(prefix["kv_key"], body.arena.key[5]))

    def test_restore_preserves_each_requests_state_and_cache_axis(self):
        body, head = self.fixture()
        prefix = freeze_review_prefix(body, head)
        second = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in prefix.items()}
        second["recurrent"].add_(7)
        members = [({"prefix": x, "rows": [head]}, [], []) for x in (prefix, second)]
        model = SimpleNamespace(empty_cache=lambda batch: Cache((Layer(), Layer())))
        captured = {"full_layer_ids": [1], "gdn_layer_ids": [0]}
        cache = restore_frozen_prefix(model, captured, members, 3, "cpu")
        self.assertEqual(cache.sequence_length, 3)
        self.assertTrue(torch.equal(cache.layers[0].recurrent[1], second["recurrent"][0]))
        self.assertTrue(torch.equal(cache.layers[1].key[0], prefix["kv_key"][0].transpose(0, 1)))
        cache.layers[0].recurrent.zero_()
        self.assertTrue(torch.count_nonzero(second["recurrent"]))
        second["length"] = 2
        with self.assertRaisesRegex(RuntimeError, "HARNESS:"):
            restore_frozen_prefix(model, captured, members, 3, "cpu")
        second["length"] = 3
        second["kv_written"].zero_()
        with self.assertRaisesRegex(RuntimeError, "EXECUTION:"):
            restore_frozen_prefix(model, captured, members, 3, "cpu")
