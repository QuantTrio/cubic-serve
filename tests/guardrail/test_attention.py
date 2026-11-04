from __future__ import annotations

import os
import unittest
from collections.abc import Callable
from unittest import mock

import torch
from torch import Tensor

from cubic_serve.guardrail.attention import (
    AttentionQualificationError,
    VerifiedAttentionReductionWitness,
    qualify_paged_gqa_attention,
    validate_paged_gqa_attention_witness,
)
from cubic_serve.ops.cuda.attention import paged_decode
from cubic_serve.ops.cuda.attention.paged_decode import (
    BF16PagedGQADecodePrimitive,
    paged_gqa_kv_splits,
)

RUN_REAL = os.environ.get("CUBIC_RUN_CUDA_ATTENTION_GUARDRAIL_TESTS") == "1"
C, PAGES, BLOCK, KVH, QH, D, TABLE = 4, 36, 16, 4, 24, 256, 8
LENGTHS = (17, 48, 95, 128)


def _layout(logical_m: int) -> tuple[int | None, ...]:
    return {
        1: (None, 0, None, None),
        2: (1, None, None, 0),
        4: (2, 0, 3, 1),
    }[logical_m]


def _active_rows(logical_m: int) -> tuple[int, ...]:
    layout = _layout(logical_m)
    return tuple(next(row for row, logical in enumerate(layout) if logical == item)
                 for item in range(logical_m))


def _canonical(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    table: Tensor,
    lengths: Tensor,
    rows: tuple[int, ...],
) -> Tensor:
    result = torch.zeros_like(query)
    for row in rows:
        length = int(lengths[row])
        keys = torch.stack(
            [key[int(table[row, token // BLOCK]), 0, token % BLOCK] for token in range(length)]
        )
        values = torch.stack(
            [value[int(table[row, token // BLOCK]), 0, token % BLOCK]
             for token in range(length)]
        )
        for head in range(QH):
            kv_head = head // (QH // KVH)
            scores = (keys[:, kv_head].float() * query[row, head].float()).sum(-1) * D**-0.5
            probability = torch.softmax(scores, 0, dtype=torch.float32)
            result[row, head] = (values[:, kv_head].float() * probability[:, None]).sum(0)
    return result


@unittest.skipUnless(
    RUN_REAL, "set CUBIC_RUN_CUDA_ATTENTION_GUARDRAIL_TESTS=1 for real CUDA qualification"
)
class AttentionCompositeGuardrailTests(unittest.TestCase):
    def setUp(self) -> None:
        from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

        # This witness is specifically the historical FP32 reassociation rule.
        self.enterContext(operator_strategy_scope(
            OperatorConfig(attention_dtype="fp32", activation_dtype="fp32"),
            device="", model="",
        ))

    def test_qwen_multi_m_dual_poison_and_fail_closed_injections(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        if torch.cuda.get_device_capability(device) != (12, 0):
            self.skipTest("this real qualification is pinned to SM120")

        query = torch.empty((C, QH, D), device=device, dtype=torch.bfloat16)
        key = torch.empty((PAGES, 1, BLOCK, KVH, D), device=device, dtype=torch.bfloat16)
        value = torch.empty_like(key)
        table = torch.empty((C, TABLE), device=device, dtype=torch.int32)
        lengths = torch.empty((C,), device=device, dtype=torch.int32)
        output = torch.empty_like(query)
        splits = paged_gqa_kv_splits(C)
        partial = torch.empty((C, QH, splits, D), device=device, dtype=torch.float32)
        statistics = torch.empty((C, QH, splits, 2), device=device, dtype=torch.float32)
        arrivals = torch.zeros((C, QH), device=device, dtype=torch.int32)
        primitive = BF16PagedGQADecodePrimitive(
            query,
            key,
            value,
            table,
            lengths,
            output,
            layer=0,
            partial=partial,
            statistics=statistics,
            arrivals=arrivals,
        )
        logical_tables = tuple(
            tuple(range(TABLE * item, TABLE * item + TABLE)) for item in range(C)
        )
        scratch_pages = tuple(range(TABLE * C, PAGES))

        def seed(seed: int) -> tuple[Tensor, Tensor, Tensor]:
            torch.manual_seed(seed)
            key_seed = torch.randn_like(key)
            value_seed = torch.randn_like(value)
            logical_query = torch.randn((C, QH, D), device=device, dtype=torch.bfloat16)
            return key_seed, value_seed, logical_query

        current = seed(1987)

        def fill(logical_m: int, poison: float) -> None:
            key.copy_(current[0])
            value.copy_(current[1])
            for physical, logical in enumerate(_layout(logical_m)):
                if logical is None:
                    page = scratch_pages[physical]
                    key[page].fill_(poison)
                    value[page].fill_(-poison)
                    query[physical].fill_(poison)
                    table[physical].fill_(page)
                    lengths[physical] = 1
                else:
                    query[physical].copy_(current[2][logical])
                    table[physical].copy_(
                        torch.tensor(logical_tables[logical], device=device, dtype=torch.int32)
                    )
                    lengths[physical] = LENGTHS[logical]

        fill(4, 37.0)
        stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.device(device), torch.cuda.stream(stream):
            primitive.launch(query, key, value, table, lengths, output)
            primitive.launch(query, key, value, table, lengths, output)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            primitive.launch(query, key, value, table, lengths, output)
        stream.synchronize()

        poison_results: dict[tuple[int, int], Tensor] = {}
        saw_reduction_only_delta = False
        last: (
            tuple[Tensor, Tensor, tuple[int, ...], VerifiedAttentionReductionWitness] | None
        ) = None
        cases = ((1987, 1), (1987, 2), (3, 4), (233, 4), (1987, 4))
        for seed_value, logical_m in cases:
            current = seed(seed_value)
            for poison_index, poison in enumerate((37.0, -73.0)):
                fill(logical_m, poison)
                graph.replay()
                torch.cuda.synchronize(device)
                rows = _active_rows(logical_m)
                expected = _canonical(query, key, value, table, lengths, rows)
                witness = qualify_paged_gqa_attention(
                    expected,
                    output,
                    query=query,
                    key=key,
                    value=value,
                    page_table=table,
                    seq_lens=lengths,
                    active_rows=rows,
                    layer=0,
                )
                self.assertLess(witness.max_allowed_error, 0.02)
                saw_reduction_only_delta |= not torch.equal(
                    output[list(rows)], expected[list(rows)]
                )
                poison_results[(logical_m, poison_index)] = output[list(rows)].clone()
                last = expected, output.clone(), rows, witness
            self.assertTrue(
                torch.equal(poison_results[(logical_m, 0)], poison_results[(logical_m, 1)])
            )
        self.assertTrue(saw_reduction_only_delta)

        assert last is not None
        expected, actual, rows, witness = last
        with self.assertRaises(TypeError):
            VerifiedAttentionReductionWitness()  # type: ignore[call-arg]

        original_probe = paged_decode.paged_gqa_scalar_probe

        def scalar_fault(
            column: int,
        ) -> Callable[[torch.device], tuple[tuple[Tensor, Tensor, Tensor], Tensor]]:
            def run(
                probe_device: torch.device,
            ) -> tuple[tuple[Tensor, Tensor, Tensor], Tensor]:
                operands, observed = original_probe(probe_device)
                corrupted = observed.clone()
                corrupted[0, column] = torch.nextafter(
                    corrupted[0, column], torch.tensor(float("inf"), device=probe_device)
                )
                return operands, corrupted
            return run

        for name, column in (("exp", 0), ("div", 1), ("leaf-fma", 3)):
            with (
                self.subTest(name=name),
                mock.patch.object(paged_decode, "paged_gqa_scalar_probe", scalar_fault(column)),
                self.assertRaisesRegex(AttentionQualificationError, "scalar ABI"),
            ):
                qualify_paged_gqa_attention(
                    expected, actual, query=query, key=key, value=value,
                    page_table=table, seq_lens=lengths, active_rows=rows, layer=0,
                )

        bad_output = actual.clone()
        bad_output[rows[0], 0, 0] += 0.125
        with self.assertRaisesRegex(AttentionQualificationError, "exceeds"):
            qualify_paged_gqa_attention(
                expected, bad_output, query=query, key=key, value=value,
                page_table=table, seq_lens=lengths, active_rows=rows, layer=0,
            )

        changed_key, changed_value, changed_table = key.clone(), value.clone(), table.clone()
        changed_key[[0, 1]] = changed_key[[1, 0]]
        changed_value[[0, 1]] = changed_value[[1, 0]]
        changed_table[rows[0], :2] = changed_table[rows[0], :2].flip(0)
        with self.assertRaisesRegex(AttentionQualificationError, "different content"):
            validate_paged_gqa_attention_witness(
                witness, expected, actual, query=query, key=changed_key, value=changed_value,
                page_table=changed_table, seq_lens=lengths, active_rows=rows, layer=0,
            )

        original_fingerprint = paged_decode.paged_gqa_fingerprint_payload

        def changed_fingerprint(
            query_arg: Tensor,
            key_arg: Tensor,
            page_table_arg: Tensor,
            *,
            layer: int,
            scale: float,
        ) -> dict[str, object]:
            payload = original_fingerprint(
                query_arg, key_arg, page_table_arg, layer=layer, scale=scale
            )
            payload["tree"] = {"id": "injected-tree"}
            return payload

        with (
            mock.patch.object(
                paged_decode, "paged_gqa_fingerprint_payload", changed_fingerprint
            ),
            self.assertRaisesRegex(AttentionQualificationError, "different content"),
        ):
            validate_paged_gqa_attention_witness(
                witness, expected, actual, query=query, key=key, value=value,
                page_table=table, seq_lens=lengths, active_rows=rows, layer=0,
            )

if __name__ == "__main__":
    unittest.main()
