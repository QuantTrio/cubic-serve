from __future__ import annotations

import math
import os
import unittest
from contextlib import ExitStack
from unittest import mock

import torch
from torch import Tensor

from cubic_serve.ops.cuda.attention.paged_decode import (
    BF16PagedGQADecodePrimitive,
    PageCOWCopyPrimitive,
    PageTokenWritePrimitive,
    paged_gqa_kv_splits,
)
from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

RUN_REAL = os.environ.get("CUBIC_RUN_CUDA_PAGED_DECODE_TESTS") == "1"
CAPACITY, LAYERS, PAGE_BLOCK = 4, 3, 16
KV_HEADS, QUERY_HEADS, HEAD_DIM, TABLE_PAGES = 4, 24, 256, 8
PAGE_COUNT = CAPACITY * (TABLE_PAGES + 2) + CAPACITY
ATTENTION_LAYER = 1
SEQ_LENS = (15, 16, 17, 128)


def _rows(logical_m: int) -> tuple[int | None, ...]:
    return {
        1: (None, 0, None, None),
        2: (1, None, None, 0),
        4: (2, 0, 3, 1),
    }[logical_m]


def _logical_pages(logical: int) -> tuple[tuple[int, ...], int, int]:
    base = logical * (TABLE_PAGES + 2)
    table = list(range(base, base + TABLE_PAGES))
    source, destination = base + TABLE_PAGES, base + TABLE_PAGES + 1
    table[(SEQ_LENS[logical] - 1) // PAGE_BLOCK] = destination
    return tuple(table), source, destination


def _value(logical: int, token: int) -> Tensor:
    values = torch.empty((KV_HEADS, HEAD_DIM), dtype=torch.bfloat16)
    length = SEQ_LENS[logical]
    for head in range(KV_HEADS):
        for dim in range(HEAD_DIM):
            base = (token + 2 * head + dim) % 5 - 2
            values[head, dim] = length * base
    return values


def _seed() -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
    torch.manual_seed(953)
    key = torch.randn(
        (PAGE_COUNT, LAYERS, PAGE_BLOCK, KV_HEADS, HEAD_DIM), dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    written = torch.rand((PAGE_COUNT, LAYERS, PAGE_BLOCK)) > 0.5
    queries: list[Tensor] = []
    uniform_keys: list[Tensor] = []
    for logical, length in enumerate(SEQ_LENS):
        table, source, _ = _logical_pages(logical)
        query = torch.randint(-2, 3, (QUERY_HEADS, HEAD_DIM)).to(torch.bfloat16)
        uniform = torch.randint(-2, 3, (KV_HEADS, HEAD_DIM)).to(torch.bfloat16)
        queries.append(query)
        uniform_keys.append(uniform)
        last_page = (length - 1) // PAGE_BLOCK
        for position in range(length):
            page_slot, token = divmod(position, PAGE_BLOCK)
            page = source if page_slot == last_page else table[page_slot]
            key[page, ATTENTION_LAYER, token].copy_(uniform)
            value[page, ATTENTION_LAYER, token].copy_(_value(logical, position))
            written[page, ATTENTION_LAYER, token] = position != length - 1
    return key, value, written, tuple(queries), tuple(uniform_keys)


def _oracle(
    key_seed: Tensor,
    value_seed: Tensor,
    written_seed: Tensor,
    query: Tensor,
    page_table: Tensor,
    seq_lens: Tensor,
    source: Tensor,
    destination: Tensor,
    token_offsets: Tensor,
    layer_values: tuple[tuple[Tensor, Tensor], ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    key, value, written = key_seed.clone(), value_seed.clone(), written_seed.clone()
    for row in range(CAPACITY):
        src, dst = int(source[row]), int(destination[row])
        key[dst].copy_(key[src])
        value[dst].copy_(value[src])
        written[dst].copy_(written[src])
    for layer, (new_key, new_value) in enumerate(layer_values):
        for row in range(CAPACITY):
            dst, token = int(destination[row]), int(token_offsets[row])
            key[dst, layer, token].copy_(new_key[row])
            value[dst, layer, token].copy_(new_value[row])
            written[dst, layer, token] = True

    output = torch.empty_like(query)
    group = QUERY_HEADS // KV_HEADS
    scale = HEAD_DIM**-0.5
    for row in range(CAPACITY):
        length = int(seq_lens[row])
        for query_head in range(QUERY_HEADS):
            kv_head = query_head // group
            logits: list[float] = []
            vectors: list[Tensor] = []
            for position in range(length):
                page = int(page_table[row, position // PAGE_BLOCK])
                token = position % PAGE_BLOCK
                q = query[row, query_head].float().tolist()
                k = key[page, ATTENTION_LAYER, token, kv_head].float().tolist()
                logits.append(math.fsum(a * b for a, b in zip(q, k, strict=True)) * scale)
                vectors.append(value[page, ATTENTION_LAYER, token, kv_head].float())
            maximum = max(logits)
            weights = [math.exp(logit - maximum) for logit in logits]
            denominator = math.fsum(weights)
            for dim in range(HEAD_DIM):
                terms = [weight * float(vector[dim]) / denominator for weight, vector in zip(
                    weights, vectors, strict=True
                )]
                output[row, query_head, dim] = math.fsum(terms)
    return key, value, written, output


@unittest.skipUnless(
    RUN_REAL, "set CUBIC_RUN_CUDA_PAGED_DECODE_TESTS=1 for real CUDA qualification"
)
class BF16PagedDecodeTests(unittest.TestCase):
    def setUp(self) -> None:
        # Historical midpoint/reduction corpus qualifies the explicit FP32
        # activation mode. Native low-precision arithmetic has its own audit.
        self.enterContext(operator_strategy_scope(
            OperatorConfig(activation_dtype="fp32", attention_dtype="fp32"), device="", model=""
        ))

    def test_identical_attention_rows_are_bit_exact_across_graph_capacities(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            self._qualify_attention_rows(dtype)

    def _qualify_attention_rows(self, dtype: torch.dtype) -> None:
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(122)
        key = torch.randn((9, 1, 16, 1, 256), device=device, generator=generator).to(dtype)
        value = torch.randn(key.shape, device=device, generator=generator).to(dtype)
        owned_query = torch.randn(
            (1, 4, 256), device=device, generator=generator
        ).to(dtype)
        baselines = {}
        split_counts = set()
        for capacity in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            splits = paged_gqa_kv_splits(capacity)
            split_counts.add(splits)
            query = owned_query.repeat(capacity, 1, 1)
            output = torch.empty_like(query)
            table = torch.arange(9, dtype=torch.int32, device=device).repeat(capacity, 1)
            lengths = torch.full((capacity,), 4, dtype=torch.int32, device=device)
            partial = torch.empty((capacity, 4, splits, 256), dtype=torch.float32, device=device)
            statistics = torch.empty((capacity, 4, splits, 2), dtype=torch.float32, device=device)
            arrivals = torch.zeros((capacity, 4), dtype=torch.int32, device=device)
            primitive = BF16PagedGQADecodePrimitive(
                query, key, value, table, lengths, output, layer=0,
                partial=partial, statistics=statistics, arrivals=arrivals,
            )
            for length in (4, 17, 129):
                lengths.fill_(length)
                primitive.launch(query, key, value, table, lengths, output)
                if capacity == 1:
                    baselines[length] = output.clone()
                self.assertTrue(torch.equal(output, baselines[length].expand_as(output)))
                self.assertFalse(bool(arrivals.count_nonzero()))
        self.assertEqual(len(split_counts), 1)

    def test_one_outer_graph_covers_multi_m_cow_two_layers_and_padding(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            self._qualify_outer_graph(dtype)

    def _qualify_outer_graph(self, dtype: torch.dtype) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        if torch.cuda.get_device_capability(device) != (12, 0):
            self.skipTest("this real qualification is pinned to SM120")

        key_seed, value_seed, written_seed, logical_queries, logical_keys = _seed()
        key_seed, value_seed, written_seed = (
            tensor.to(device, dtype=dtype if tensor.is_floating_point() else tensor.dtype)
            for tensor in (key_seed, value_seed, written_seed)
        )
        key, value, written = key_seed.clone(), value_seed.clone(), written_seed.clone()
        query = torch.empty((CAPACITY, QUERY_HEADS, HEAD_DIM), device=device,
                            dtype=dtype)
        output = torch.empty_like(query)
        splits = paged_gqa_kv_splits(CAPACITY)
        partial = torch.empty(
            (CAPACITY, QUERY_HEADS, splits, HEAD_DIM), device=device, dtype=torch.float32
        )
        statistics = torch.empty(
            (CAPACITY, QUERY_HEADS, splits, 2), device=device, dtype=torch.float32
        )
        arrivals = torch.zeros((CAPACITY, QUERY_HEADS), device=device, dtype=torch.int32)
        page_table = torch.empty((CAPACITY, TABLE_PAGES), device=device, dtype=torch.int32)
        seq_lens = torch.empty((CAPACITY,), device=device, dtype=torch.int32)
        source = torch.empty_like(seq_lens)
        destination = torch.empty_like(seq_lens)
        token_offsets = torch.empty_like(seq_lens)
        new_key_0 = torch.empty((CAPACITY, KV_HEADS, HEAD_DIM), device=device,
                                dtype=dtype)
        new_value_0 = torch.empty_like(new_key_0)
        new_key_1 = torch.empty_like(new_key_0)
        new_value_1 = torch.empty_like(new_key_0)

        copy = PageCOWCopyPrimitive(key, value, written, source, destination)
        write_0 = PageTokenWritePrimitive(
            key, value, written, destination, token_offsets, new_key_0, new_value_0,
            layer=0,
        )
        write_1 = PageTokenWritePrimitive(
            key, value, written, destination, token_offsets, new_key_1, new_value_1,
            layer=ATTENTION_LAYER,
        )
        attention = BF16PagedGQADecodePrimitive(
            query, key, value, page_table, seq_lens, output, layer=ATTENTION_LAYER,
            partial=partial, statistics=statistics, arrivals=arrivals,
        )
        with self.assertRaisesRegex(ValueError, "finite positive"):
            BF16PagedGQADecodePrimitive(
                query, key, value, page_table, seq_lens, output,
                layer=ATTENTION_LAYER, scale=0.0,
        )
        for primitive in (copy, write_0, write_1, attention):
            values = tuple(vars(primitive).values()) if hasattr(primitive, "__dict__") else tuple(
                getattr(primitive, name) for name in type(primitive).__slots__
            )
            self.assertFalse(any(isinstance(item, Tensor) for item in values))

        scratch = tuple(range(CAPACITY * (TABLE_PAGES + 2), PAGE_COUNT))
        active_pages = {
            page
            for logical in range(CAPACITY)
            for page in (*_logical_pages(logical)[0], *_logical_pages(logical)[1:])
        }
        self.assertEqual(len(scratch), len(set(scratch)))
        self.assertTrue(set(scratch).isdisjoint(active_pages))
        self.assertIn(PAGE_BLOCK, SEQ_LENS)
        self.assertIn(PAGE_BLOCK + 1, SEQ_LENS)

        def fill(logical_m: int, poison: float) -> None:
            key.copy_(key_seed)
            value.copy_(value_seed)
            written.copy_(written_seed)
            for physical, logical in enumerate(_rows(logical_m)):
                if logical is None:
                    scratch_page = scratch[physical]
                    key[scratch_page].fill_(poison)
                    value[scratch_page].fill_(-poison)
                    written[scratch_page].fill_(poison > 0)
                    query[physical].fill_(poison)
                    page_table[physical].fill_(scratch_page)
                    seq_lens[physical] = 1
                    source[physical] = destination[physical] = scratch_page
                    token_offsets[physical] = physical % PAGE_BLOCK
                    for tensor, scalar in (
                        (new_key_0, poison), (new_value_0, -poison),
                        (new_key_1, poison + 1), (new_value_1, -poison - 1),
                    ):
                        tensor[physical].fill_(scalar)
                    continue
                table, src, dst = _logical_pages(logical)
                length = SEQ_LENS[logical]
                query[physical].copy_(logical_queries[logical])
                page_table[physical].copy_(torch.tensor(table, device=device, dtype=torch.int32))
                seq_lens[physical], source[physical], destination[physical] = length, src, dst
                token_offsets[physical] = (length - 1) % PAGE_BLOCK
                new_key_0[physical].fill_(10 + logical)
                new_value_0[physical].fill_(-10 - logical)
                new_key_1[physical].copy_(logical_keys[logical])
                new_value_1[physical].copy_(_value(logical, length - 1).to(device))

        fill(4, 37.0)
        stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream(device))

        def launch() -> None:
            copy.launch(key, value, written, source, destination)
            write_0.launch(key, value, written, destination, token_offsets,
                           new_key_0, new_value_0)
            write_1.launch(key, value, written, destination, token_offsets,
                           new_key_1, new_value_1)
            attention.launch(query, key, value, page_table, seq_lens, output)

        with torch.cuda.device(device), torch.cuda.stream(stream):
            launch()
            launch()
        stream.synchronize()

        def forbidden(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("primitive launch performed a forbidden host operation")

        with ExitStack() as stack:
            for owner, names in (
                (torch, ("empty", "empty_like", "empty_strided", "zeros", "ones", "full",
                         "tensor", "clone", "compile")),
                (Tensor, ("clone", "item", "new_empty", "new_zeros", "new_full")),
                (torch.cuda, ("synchronize",)),
                (torch.cuda.Stream, ("synchronize",)),
            ):
                for name in names:
                    stack.enter_context(mock.patch.object(owner, name, forbidden))
            with torch.cuda.device(device), torch.cuda.stream(stream):
                launch()
        stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launch()
        stream.synchronize()
        tensors = (
            key, value, written, query, output, page_table, seq_lens, source,
            destination, token_offsets, new_key_0, new_value_0, new_key_1, new_value_1,
            partial, statistics, arrivals,
        )
        addresses = tuple(tensor.data_ptr() for tensor in tensors)
        results: dict[tuple[int, float], tuple[Tensor, ...]] = {}

        for logical_m in (1, 2, 4):
            for poison in (37.0, -73.0):
                fill(logical_m, poison)
                oracle = _oracle(
                    key.cpu(), value.cpu(), written.cpu(), query.cpu(),
                    page_table.cpu(), seq_lens.cpu(), source.cpu(), destination.cpu(),
                    token_offsets.cpu(),
                    ((new_key_0.cpu(), new_value_0.cpu()),
                     (new_key_1.cpu(), new_value_1.cpu())),
                )
                expected_key, expected_value, expected_written, expected_output = oracle
                graph.replay()
                torch.cuda.synchronize(device)
                logical_rows = [
                    next(index for index, logical in enumerate(_rows(logical_m)) if logical == i)
                    for i in range(logical_m)
                ]
                actual_output = output[logical_rows].clone()
                expected_active = expected_output[logical_rows].to(device)
                # Uniform logits and length-scaled integral V make every arithmetic
                # result exact. This does not qualify generic REDUCTION_ORDER drift.
                self.assertTrue(torch.equal(expected_active, actual_output))
                active_destinations = tuple(int(destination[row]) for row in logical_rows)
                self.assertTrue(torch.equal(key, expected_key.to(device)))
                self.assertTrue(torch.equal(value, expected_value.to(device)))
                self.assertTrue(torch.equal(written, expected_written.to(device)))
                for row, dst in zip(logical_rows, active_destinations, strict=True):
                    token = int(token_offsets[row])
                    self.assertTrue(torch.equal(key[dst, 0, token], new_key_0[row]))
                    self.assertTrue(torch.equal(value[dst, 0, token], new_value_0[row]))
                    self.assertTrue(torch.equal(key[dst, ATTENTION_LAYER, token], new_key_1[row]))
                    self.assertTrue(
                        torch.equal(value[dst, ATTENTION_LAYER, token], new_value_1[row])
                    )
                    self.assertTrue(bool(written[dst, 0, token]))
                    self.assertTrue(bool(written[dst, ATTENTION_LAYER, token]))
                    self.assertNotEqual(int(source[row]), dst)
                    last_slot = (int(seq_lens[row]) - 1) // PAGE_BLOCK
                    self.assertEqual(int(page_table[row, last_slot]), dst)
                results[(logical_m, poison)] = (
                    actual_output,
                    *(key[dst].clone() for dst in active_destinations),
                    *(value[dst].clone() for dst in active_destinations),
                    *(written[dst].clone() for dst in active_destinations),
                )

            positive, negative = results[(logical_m, 37.0)], results[(logical_m, -73.0)]
            self.assertTrue(
                all(torch.equal(left, right) for left, right in zip(
                    positive, negative, strict=True
                ))
            )

        self.assertEqual(addresses, tuple(tensor.data_ptr() for tensor in tensors))


@unittest.skipUnless(RUN_REAL, "CUDA opt-in")
class NativeQKVTests(unittest.TestCase):
    def test_mma_operands_reference_and_cross_m_graph(self) -> None:
        from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
        from cubic_serve.ops.eager.attention import qkv_attention_row

        configure_reference_math()
        for dtype in (torch.bfloat16, torch.float16):
            for heads in (6, 8, 17):
                with self.subTest(dtype=dtype, heads=heads), operator_strategy_scope(
                    OperatorConfig(attention_dtype="auto", softmax_dtype="fp32"),
                    device="", model="",
                ):
                    torch.manual_seed(923)
                    dim, length = 64, 73
                    key = torch.randn(6, 1, 16, 1, dim, device="cuda", dtype=dtype)
                    value = torch.randn_like(key)
                    key[5].fill_(torch.nan)
                    value[5].fill_(torch.nan)
                    query = torch.randn(33, heads, dim, device="cuda", dtype=dtype)
                    common = None
                    for capacity in (1, 8, 64):
                        logical = min(capacity, 33)
                        indices = torch.randperm(capacity, device="cuda")[:logical]
                        q = torch.full((capacity, heads, dim), torch.nan,
                                       device="cuda", dtype=dtype)
                        q[indices] = query[:logical]
                        table = torch.full((capacity, 5), 5, device="cuda", dtype=torch.int32)
                        table[indices] = torch.arange(5, device="cuda", dtype=torch.int32)
                        lengths = torch.ones(capacity, device="cuda", dtype=torch.int32)
                        lengths[indices] = length
                        output = torch.empty_like(q)
                        arrivals = torch.zeros(capacity, heads, device="cuda", dtype=torch.int32)
                        workspace = dict(
                            partial=torch.empty(capacity, heads, 32, dim, device="cuda"),
                            statistics=torch.empty(capacity, heads, 32, 2, device="cuda"),
                            arrivals=arrivals,
                        )
                        primitive = BF16PagedGQADecodePrimitive(
                            q, key, value, table, lengths, output, layer=0, **workspace
                        )
                        ptx = primitive._artifact.asm["ptx"]
                        suffix = "bf16.bf16.f32" if dtype is torch.bfloat16 else "f16.f16.f32"
                        self.assertIn(suffix, ptx)
                        self.assertIn("mma.sync.aligned", ptx)
                        primitive.launch(q, key, value, table, lengths, output)
                        torch.cuda.synchronize()
                        eager_launch = output[indices].clone()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            primitive.launch(q, key, value, table, lengths, output)
                        for _ in range(3):
                            graph.replay()
                        torch.cuda.synchronize()
                        self.assertTrue(torch.equal(output[indices], eager_launch))
                        self.assertTrue(torch.isfinite(eager_launch).all().item())
                        self.assertEqual(torch.count_nonzero(arrivals).item(), 0)
                        if common is not None:
                            self.assertTrue(torch.equal(common, eager_launch[0]))
                        common = eager_launch[0].clone()
                    with ReferenceExecutionContext():
                        expected = torch.stack([
                            qkv_attention_row(
                                qh, key[:5, 0, :, 0].reshape(-1, dim)[:length],
                                value[:5, 0, :, 0].reshape(-1, dim)[:length], scale=dim**-0.5,
                            ) for qh in query[0]
                        ])
                    # Diagnostic bound includes final storage RNE, not an issued
                    # floating-point qualification or an acceptance-rate oracle.
                    torch.testing.assert_close(common, expected, rtol=0,
                                               atol=2 * torch.finfo(dtype).eps)

    def test_fp16_products_do_not_round_before_fp32_accumulation(self) -> None:
        with operator_strategy_scope(OperatorConfig(attention_dtype="auto", softmax_dtype="fp32"),
                                     device="", model=""):
            # Products exceed FP16 range but the dot and softmax remain finite.
            q = torch.full((1, 1, 16), 256.0, device="cuda", dtype=torch.float16)
            key = torch.full((1, 1, 16, 1, 16), 256.0, device="cuda", dtype=q.dtype)
            value = torch.full_like(key, 3.0)
            table = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
            length = torch.tensor([16], device="cuda", dtype=torch.int32)
            output = torch.empty_like(q)
            primitive = BF16PagedGQADecodePrimitive(q, key, value, table, length, output, layer=0)
            primitive.launch(q, key, value, table, length, output)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(output, torch.full_like(output, 3.0)))

    def test_long_uniform_context_keeps_fp32_denominator(self) -> None:
        with operator_strategy_scope(OperatorConfig(attention_dtype="auto", softmax_dtype="fp32"),
                                     device="", model=""):
            for dtype in (torch.float16, torch.bfloat16):
                for length in (65536, 131072):
                    with self.subTest(dtype=dtype, length=length):
                        q = torch.zeros(1, 1, 16, device="cuda", dtype=dtype)
                        key = torch.zeros(length // 16, 1, 16, 1, 16,
                                          device="cuda", dtype=dtype)
                        value = torch.full_like(key, 3.0)
                        table = torch.arange(length // 16, device="cuda", dtype=torch.int32)[None]
                        lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
                        output = torch.empty_like(q)
                        workspace = dict(partial=torch.empty(1, 1, 32, 16, device="cuda"),
                                         statistics=torch.empty(1, 1, 32, 2, device="cuda"),
                                         arrivals=torch.zeros(1, 1, device="cuda",
                                                              dtype=torch.int32))
                        primitive = BF16PagedGQADecodePrimitive(
                            q, key, value, table, lengths, output, layer=0, **workspace
                        )
                        primitive.launch(q, key, value, table, lengths, output)
                        torch.cuda.synchronize()
                        self.assertTrue(torch.equal(output, torch.full_like(output, 3.0)))

    def test_policy_is_independent_and_legacy_witness_fails_closed(self) -> None:
        from cubic_serve.cli import build_parser
        from cubic_serve.guardrail.attention import (
            AttentionQualificationError,
            qualify_paged_gqa_attention,
        )
        from cubic_serve.ops.strategies import attention_precision, softmax_precision

        args = build_parser().parse_args(
            ["serve", "/checkpoint", "--attention-dtype", "auto", "--softmax-dtype", "fp32"]
        )
        self.assertEqual(args.attention_dtype, "auto")
        self.assertEqual(args.softmax_dtype, "fp32")
        config = OperatorConfig.from_dict({"version": 1, "attention_dtype": "auto",
                                           "activation_dtype": "auto", "softmax_dtype": "fp32"})
        with operator_strategy_scope(config, device="", model=""):
            self.assertEqual(attention_precision(), "auto")
            self.assertEqual(softmax_precision(), "fp32")
            empty = torch.empty(0)
            with self.assertRaisesRegex(AttentionQualificationError, "P-rounding witness"):
                qualify_paged_gqa_attention(empty, empty, query=empty, key=empty, value=empty,
                                           page_table=empty, seq_lens=empty,
                                           active_rows=(), layer=0)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(RUN_REAL, "explicit CUDA audit required")
class PageFieldCopyTests(unittest.TestCase):
    def test_graph_copy_retains_integer_bits_prefix_and_padding(self):
        key = torch.zeros((6, 1, 4, 1, 2), device="cuda", dtype=torch.bfloat16)
        value = torch.zeros_like(key)
        written = torch.zeros((6, 1, 4), device="cuda", dtype=torch.bool)
        fields = (
            torch.zeros((6, 1, 2, 3), device="cuda", dtype=torch.bfloat16),
            torch.zeros((6, 2), device="cuda", dtype=torch.int64),
            torch.zeros((6, 2), device="cuda", dtype=torch.bool),
        )
        source = torch.tensor([0, 0, 5], device="cuda", dtype=torch.int32)
        destination = torch.tensor([2, 3, 5], device="cuda", dtype=torch.int32)
        copy = PageCOWCopyPrimitive(key, value, written, source, destination, page_fields=fields)
        copy.launch(key, value, written, source, destination)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            copy.launch(key, value, written, source, destination)
        for field, number in zip(fields, (3.25, 2**54 + 1, True), strict=True):
            field.fill_(0)
            field[0].fill_(number)
            field[5].fill_(1)
        expected = tuple(t.cpu().clone() for t in fields)
        graph.replay()
        for field, before in zip(fields, expected, strict=True):
            after = field.cpu()
            assert torch.equal(after[0], before[0])
            assert torch.equal(after[2], before[0])
            assert torch.equal(after[3], before[0])
            assert torch.equal(after[1], before[1])
            assert torch.equal(after[4:], before[4:])
