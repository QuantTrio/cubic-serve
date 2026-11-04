import os

import pytest
import torch

from cubic_serve.executor.runner import configure_reference_math
from cubic_serve.ops.cuda.attention.paged_decode import (
    PagedGQADecodePrimitive,
    paged_gqa_kv_splits,
)
from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

pytestmark = pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)


def test_resource_partition_plan_tracks_head_sharding_and_token_domain():
    for sms in (16, 80, 108, 156, 240):
        for heads in (1, 6, 12, 24, 128):
            for tokens in (1, 15, 16, 63, 128, 2051, 8192):
                plans = []
                for m in (1, 2, 8, 32):
                    splits = paged_gqa_kv_splits(
                        m, max_tokens=tokens, head_programs=heads, multiprocessors=sms
                    )
                    plans.append(splits)
                    assert splits > 0 and splits & (splits - 1) == 0
                    assert splits <= min(32, max(1, tokens // 16))
                    if tokens // 16 >= 32:
                        assert splits * heads * m >= sms or splits == 32
                assert plans == sorted(plans, reverse=True)
    for kwargs in (
        {"head_programs": 6}, {"multiprocessors": 80},
        {"head_programs": 0, "multiprocessors": 80},
        {"head_programs": 6, "multiprocessors": -1},
    ):
        with pytest.raises(ValueError, match="head programs and SM count"):
            paged_gqa_kv_splits(1, max_tokens=2051, **kwargs)


@pytest.mark.parametrize("heads", (6, 12))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("split_override", (None, 1, 2, 4, 8, 16, 32))
def test_resource_splits_independent_intervals_and_reused_padded_graph(
    heads, dtype, split_override,
):
    from unittest.mock import patch

    from cubic_serve.guardrail.quantized_graph import attention_bounds

    configure_reference_math()
    torch.manual_seed(912)
    width, dim, page_size = 2051, 256, 16
    positions = torch.arange(width, device="cuda")
    positions = positions // 4 * 8 + positions % 4
    pages = (int(positions[-1]) + page_size) // page_size
    page_order = torch.randperm(pages, device="cuda").int()
    key = torch.randn(pages, 1, page_size, 1, dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    corpus = torch.randn(1, heads, dim, device="cuda", dtype=dtype)
    selected_k = key[page_order[positions // page_size].long(), 0, positions % page_size]
    selected_v = value[page_order[positions // page_size].long(), 0, positions % page_size]
    common = {}
    with operator_strategy_scope(
        OperatorConfig(attention_dtype="fp32", softmax_dtype="fp32"), device="", model=""
    ):
        for capacity in (1, 8):
            query = torch.empty(capacity, heads, dim, device="cuda", dtype=dtype)
            output = torch.empty_like(query)
            table = page_order[None].expand(capacity, -1).contiguous()
            selected = positions.int()[None].expand(capacity, -1).contiguous()
            counts = torch.ones(capacity, device="cuda", dtype=torch.int32)
            splits = paged_gqa_kv_splits(
                capacity, max_tokens=width, head_programs=heads,
                multiprocessors=torch.cuda.get_device_properties(query.device).multi_processor_count,
            )
            if split_override is not None:
                splits = split_override
            workspace = dict(
                partial=torch.empty(capacity, heads, splits, dim, device="cuda"),
                statistics=torch.empty(capacity, heads, splits, 2, device="cuda"),
                arrivals=torch.zeros(capacity, heads, device="cuda", dtype=torch.int32),
            )
            primitive = PagedGQADecodePrimitive(
                query, key, value, table, counts, output, layer=0,
                selected_indices=selected, **workspace,
            )
            query.zero_()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                primitive.launch(query, key, value, table, counts, output)
            # Reuse captured scratch across long-to-short transitions and row
            # permutations. Candidate partials never feed the interval oracle.
            for length in (2051, 7, 513):
                row = (length % capacity)
                query.fill_(float("nan"))
                query[row].copy_(corpus[0])
                counts.fill_(1)
                counts[row] = length
                workspace["partial"].fill_(float("nan"))
                workspace["statistics"].fill_(float("nan"))
                graph.replay()
                result = output[row:row + 1].cpu()
                assert not workspace["arrivals"].count_nonzero()
                bounds = []
                check = attention_bounds.check

                def retain_bounds(actual, lo, hi, *, saved=bounds, compare=check):
                    saved.append((lo.clone(), hi.clone()))
                    return compare(actual, lo, hi)

                # Retain independently computed CPU bounds after the forward;
                # no candidate partials participate in the oracle.
                with patch.object(attention_bounds, "check", retain_bounds):
                    proof = attention_bounds.enclose(
                        corpus.cpu(), selected_k[None, :length].cpu(),
                        selected_v[None, :length].cpu(), result,
                        scale=dim**-0.5, splits=splits, probability_dtype=torch.float32,
                    )
                assert proof["outside_enclosure"] == 0, proof
                lo, hi = bounds.pop()
                if length in common:
                    previous, previous_lo, previous_hi, previous_splits = common[length]
                    # The common logical row has identical Q/K/V and selected
                    # positions. Compare its actual cross-M delta against both
                    # declared reduction trees, rather than a blanket tolerance.
                    delta = result[0].double() - previous[0].double()
                    pair = check(delta, lo - previous_hi, hi - previous_lo)
                    assert pair["outside_enclosure"] == 0, (previous_splits, splits, pair)
                common[length] = result, lo, hi, splits


@pytest.mark.parametrize("split_kv", ("none", "bounded", "full"))
def test_selected_paged_gqa_equals_compact_cache_and_uniform_oracle(split_kv):
    configure_reference_math()
    torch.manual_seed(87)
    # Original positions cross disjoint page boundaries; trailing selectors are
    # invalid poison and must never become an attention operand.
    positions = torch.tensor([0, 2, 7, 13, 16, 29, 34, 51, 57, 62, 71, 79, 91], device="cuda")
    page_order = torch.tensor([8, 1, 10, 3, 0, 7, 4, 9, 5, 11, 2, 6], device="cuda")
    for precision in ("fp32", "auto"):
        with operator_strategy_scope(
            OperatorConfig(attention_dtype=precision, softmax_dtype="fp32"), device="", model=""
        ):
            for dtype in (torch.float16, torch.bfloat16):
                key = torch.randn(12, 2, 8, 2, 64, device="cuda", dtype=dtype)
                value = (torch.arange(key.numel(), device="cuda") % 17).reshape_as(key).to(dtype)
                q_corpus = torch.randn(17, 4, 64, device="cuda", dtype=dtype)
                compact_key, compact_value = (
                    torch.empty(2, 2, 8, 2, 64, device="cuda", dtype=dtype),
                    torch.empty(2, 2, 8, 2, 64, device="cuda", dtype=dtype),
                )
                compact_key.fill_(float("nan"))
                compact_value.fill_(float("nan"))
                selected_k = key[page_order[positions // 8], 1, positions % 8]
                selected_v = value[page_order[positions // 8], 1, positions % 8]
                for i in range(13):
                    compact_key[i // 8, 1, i % 8] = selected_k[i]
                    compact_value[i // 8, 1, i % 8] = selected_v[i]
                common = None
                for capacity, logical in ((1, 1), (8, 8), (32, 17)):
                    query = torch.empty(capacity, 4, 64, device="cuda", dtype=dtype)
                    output, expected = torch.empty_like(query), torch.empty_like(query)
                    table = page_order.int()[None].expand(capacity, -1).contiguous()
                    compact_table = (
                        torch.tensor([0, 1], device="cuda", dtype=torch.int32)[None]
                        .expand(capacity, -1)
                        .contiguous()
                    )
                    counts = torch.full((capacity,), 13, device="cuda", dtype=torch.int32)
                    selection = torch.full((capacity, 24), -999, device="cuda", dtype=torch.int32)
                    selection[:, :13].copy_(positions.int())

                    def workspace(capacity=capacity):
                        if split_kv == "none":
                            return {}
                        splits = paged_gqa_kv_splits(
                            capacity, max_tokens=2051 if split_kv == "bounded" else None
                        )
                        return dict(
                            partial=torch.empty(capacity, 4, splits, 64, device="cuda"),
                            statistics=torch.empty(capacity, 4, splits, 2, device="cuda"),
                            arrivals=torch.zeros(capacity, 4, dtype=torch.int32, device="cuda"),
                        )

                    # The primitive binds addresses; the caller owns workspace
                    # storage through every graph replay.
                    indexed_workspace, compact_workspace = workspace(), workspace()
                    indexed = PagedGQADecodePrimitive(
                        query,
                        key,
                        value,
                        table,
                        counts,
                        output,
                        layer=1,
                        selected_indices=selection,
                        **indexed_workspace,
                    )
                    compact = PagedGQADecodePrimitive(
                        query, compact_key, compact_value, compact_table, counts, expected,
                        layer=1, **compact_workspace,
                    )
                    indexed.launch(query, key, value, table, counts, output)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        indexed.launch(query, key, value, table, counts, output)
                    for reverse in (False, True):
                        query.fill_(float("nan"))
                        data = q_corpus[:logical].flip(0) if reverse else q_corpus[:logical]
                        query[:logical].copy_(data)
                        graph.replay()
                        receipt = compact.launch(
                            query, compact_key, compact_value, compact_table, counts, expected
                        )
                        compact.validate_receipt(receipt)
                        assert torch.equal(output[:logical], expected[:logical])
                        canonical = output[:logical].flip(0) if reverse else output[:logical]
                        if common is not None:
                            assert torch.equal(canonical[:1], common)
                        common = canonical[:1].clone()
                    query.zero_()
                    for count in (1, 7, 13):
                        counts.fill_(count)
                        graph.replay()
                        average = (
                            selected_v[:count].float().mean(0).repeat_interleave(2, 0).to(dtype)
                        )
                        assert torch.equal(output, average[None].expand_as(output))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("precision", ("fp32", "auto"))
def test_bounded_splits_uneven_lengths_against_uniform_mean(dtype, precision):
    configure_reference_math()
    torch.manual_seed(471)
    width, page_size, heads, dim = 2051, 16, 6, 64
    positions = torch.arange(width, device="cuda")
    positions = positions // 4 * 8 + positions % 4
    pages = (int(positions[-1]) + page_size) // page_size
    page_order = torch.randperm(pages, device="cuda").int()
    key = torch.randn(pages, 1, page_size, 1, dim, dtype=dtype, device="cuda")
    value = (torch.arange(key.numel(), device="cuda") % 17).reshape_as(key).to(dtype)
    selected_values = value[page_order[positions // page_size].long(), 0,
                            positions % page_size]
    common = {}
    with operator_strategy_scope(
        OperatorConfig(attention_dtype=precision, softmax_dtype="fp32"), device="", model=""
    ):
        for capacity in (1, 8, 32):
            query = torch.zeros(capacity, heads, dim, dtype=dtype, device="cuda")
            output = torch.empty_like(query)
            table = page_order[None].expand(capacity, -1).contiguous()
            selected = positions.int()[None].expand(capacity, -1).contiguous()
            counts = torch.ones(capacity, dtype=torch.int32, device="cuda")
            splits = paged_gqa_kv_splits(
                capacity, max_tokens=width,
                head_programs=1 if precision == "auto" else heads,
                multiprocessors=torch.cuda.get_device_properties(query.device).multi_processor_count,
            )
            partial = torch.empty(capacity, heads, splits, dim, device="cuda")
            statistics = torch.empty(capacity, heads, splits, 2, device="cuda")
            arrivals = torch.zeros(capacity, heads, dtype=torch.int32, device="cuda")
            primitive = PagedGQADecodePrimitive(
                query, key, value, table, counts, output, layer=0,
                partial=partial, statistics=statistics, arrivals=arrivals,
                selected_indices=selected,
            )
            assert primitive.reduction_abi["kv_splits"] == splits
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                primitive.launch(query, key, value, table, counts, output)
            for length in (1, 7, 13, 127, 128, 129, 511, 513, width):
                counts.fill_(length)
                graph.replay()
                mean = selected_values[:length].float().mean(0).expand(heads, dim).to(dtype)
                assert torch.equal(output, mean[None].expand_as(output))
                if length in common:
                    assert torch.equal(output[:1], common[length])
                else:
                    common[length] = output[:1].clone()
                assert torch.count_nonzero(arrivals) == 0


def test_full_context_partitions_follow_resources_and_preserve_selected_policy():
    for sms in (16, 80, 170, 256):
        for heads in (1, 4, 24):
            for tokens in (31, 1024, 32768, 262144):
                values = [paged_gqa_kv_splits(m, max_tokens=tokens,
                    head_programs=heads, multiprocessors=sms, full_context=True)
                    for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)]
                assert values == sorted(values, reverse=True)
                assert all(1 <= v <= 32 and v & (v-1) == 0 for v in values)
                assert values == [paged_gqa_kv_splits(m, max_tokens=tokens,
                    head_programs=heads*2, multiprocessors=sms*2, full_context=True)
                    for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)]
    # The long-context envelope retains multiple waves; one wave regressed.
    assert paged_gqa_kv_splits(32, max_tokens=262144, head_programs=4,
                             multiprocessors=170, full_context=True) == 32
    assert paged_gqa_kv_splits(1, max_tokens=262144, head_programs=4,
                             multiprocessors=170, full_context=True) == 32
    for kwargs in ({'full_context': True}, {'full_context': True, 'max_tokens': 1024},
                   {'full_context': 1, 'max_tokens': 1024}):
        with pytest.raises(ValueError):
            paged_gqa_kv_splits(1, **kwargs)
