import os

import pytest
import torch

from cubic_serve.ops.cuda.embedding.token import bind_scaled_sharded_embedding

pytestmark = pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)


@pytest.mark.parametrize("weight_dtype", [torch.float8_e4m3fn, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("host", [False, True])
def test_scaled_sharded_lookup_integer_ids_ownership_cross_m_and_padding(weight_dtype, dtype, host):
    torch.manual_seed(328)
    width = 160
    if weight_dtype == torch.float8_e4m3fn:
        # Cover every finite byte pattern, including signed zero and subnormals.
        codes = torch.arange(256).to(torch.uint8)
        codes[127] = 0
        codes[255] = 128
        weight = (
            codes.repeat((11 * width + 255) // 256)[: 11 * width]
            .view(weight_dtype)
            .reshape(11, width)
        )
    else:
        weight = torch.randn(11, width).to(weight_dtype)
    base = 2**40 + 7
    starts = (base, base + 7, base + 25)
    place = (lambda x: x.pin_memory()) if host else (lambda x: x.cuda())
    shard_weights = tuple(place(x) for x in (weight[:3], weight[3:8], weight[8:]))
    inventory = tuple(zip(starts, shard_weights, strict=True))
    scale = torch.tensor(0.09375, dtype=torch.bfloat16, device="cuda")
    corpus = torch.tensor(
        [
            [base, base + 2, base + 7, base + 11],
            [base + 25, base + 27, base + 1, base + 9],
            [base - 1, base + 3, base + 12, base + 28],
        ],
        dtype=torch.int64,
    )
    expected = torch.zeros((3, 4, width), dtype=dtype)
    for row in range(3):
        for head in range(4):
            token = int(corpus[row, head])
            for start, part in inventory:
                if start <= token < start + part.shape[0]:
                    expected[row, head] = (
                        part[token - start].cpu().float() * float(scale.item())
                    ).to(dtype)
    expected = expected.flatten(1)
    common = {}
    for capacity in (1, 8, 32):
        ids = torch.empty((capacity, 4), dtype=torch.int64, device="cuda")
        active = torch.zeros(capacity, dtype=torch.int32, device="cuda")
        output = torch.empty((capacity, 4 * width), dtype=dtype, device="cuda")
        op = bind_scaled_sharded_embedding(ids, inventory, scale, active, output)
        op.launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op.launch()
        mappings = [(0,), (1,), (2,)] if capacity == 1 else [(0, 1, 2), (2, 0, 1)]
        for mapping in mappings:
            ids.fill_(-(2**60))
            active.zero_()
            output.fill_(77)
            for physical, logical in enumerate(mapping):
                ids[physical].copy_(corpus[logical])
                active[physical] = 1
            graph.replay()
            actual = output.cpu()
            for physical, logical in enumerate(mapping):
                assert torch.equal(actual[physical], expected[logical])
                if logical in common:
                    assert torch.equal(actual[physical], common[logical])
                common[logical] = actual[physical].clone()
            assert (actual[len(mapping) :] == 77).all()


def test_sharded_lookup_rejects_overlapping_ownership():
    ids = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
    weight = torch.ones((4, 8), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((1, 8), dtype=torch.bfloat16, device="cuda")
    scale = torch.ones((), dtype=torch.float32, device="cuda")
    active = torch.ones(1, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="disjoint"):
        bind_scaled_sharded_embedding(
            ids, ((0, weight), (3, weight.clone())), scale, active, output
        )


def test_host_lookup_rejects_pageable_storage():
    ids = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
    output = torch.empty((1, 8), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="pinned"):
        bind_scaled_sharded_embedding(
            ids, ((0, torch.ones(4, 8, dtype=torch.bfloat16)),),
            torch.ones((), device="cuda"), torch.ones(1, dtype=torch.int32, device="cuda"),
            output,
        )
