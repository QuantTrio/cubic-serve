import torch

from cubic_serve.executor.runner import ReferenceExecutionContext
from cubic_serve.ops.eager.hyperconnection import combine, injection, mix


def test_zero_gates_define_half_mean_and_unit_injection():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    with ReferenceExecutionContext():
        mixed, normalized = mix(
            x, torch.zeros(6), torch.zeros(2, 6), torch.zeros(6, 2), streams=3, eps=1e-6
        )
        gates = injection(normalized, torch.zeros(3, 6), streams=3)
        result = combine(x, mixed, gates)
    groups = x.reshape(1, 3, 2)
    expected = groups / (groups.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    torch.testing.assert_close(mixed, expected.mean(1) / 2, rtol=1e-6, atol=0)
    assert torch.equal(gates, torch.ones(1, 3))
    assert torch.equal(result, x + mixed.repeat(1, 3))


def test_mix_row_permutation_and_partition_identity():
    torch.manual_seed(6)
    x = torch.randn(7, 24).bfloat16()
    norm = torch.randn(24).bfloat16()
    down, up = torch.randn(5, 24).bfloat16(), torch.randn(24, 5).bfloat16()
    order = torch.tensor([6, 1, 4, 0, 3, 2, 5])
    with ReferenceExecutionContext():
        together, _ = mix(x, norm, down, up, streams=3, eps=1e-6)
        permuted, _ = mix(x[order], norm, down, up, streams=3, eps=1e-6)
        singles = torch.cat([mix(row[None], norm, down, up, streams=3, eps=1e-6)[0] for row in x])
    assert torch.equal(together, singles)
    assert torch.equal(together[order], permuted)


def test_grouped_norm_cuda_rows_do_not_change_the_reduction_tree():
    import os

    import pytest

    from cubic_serve.ops.eager.hyperconnection import grouped_rms_norm

    if os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1":
        pytest.skip("explicit CUDA audit required")
    with ReferenceExecutionContext():
        torch.manual_seed(17)
        for width in (1920, 2560, 5120):
            x = torch.randn(5, width * 4, device="cuda").bfloat16()
            weight = torch.randn(width * 4, device="cuda").bfloat16()
            whole = grouped_rms_norm(x, weight, group_size=width, eps=1e-6)
            split = torch.cat(
                [grouped_rms_norm(z, weight, group_size=width, eps=1e-6) for z in (x[:3], x[3:])]
            )
            assert torch.equal(whole, split)


def test_grouped_cuda_norm_matches_independent_groups_and_padded_graph_rows():
    import os

    import pytest

    from cubic_serve.ops.cuda.normalization import ZeroCenteredRMSNormPrimitive
    from cubic_serve.ops.eager.hyperconnection import grouped_rms_norm

    if os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1":
        pytest.skip("explicit CUDA audit required")
    torch.manual_seed(38)
    for dtype in (torch.float16, torch.bfloat16):
        for width, groups in ((96, 3), (2560, 4), (5120, 1)):
            # Exact variance and square root give an independent arithmetic
            # witness without treating a reduction-order tolerance as proof.
            signs = torch.arange(33 * width * groups, device="cuda").reshape(33, -1)
            corpus = torch.where(signs % 3 == 0, -1.0, 1.0).to(dtype)
            weight = (torch.arange(width * groups, device="cuda") % 8).to(dtype) / 8
            with ReferenceExecutionContext():
                expected = grouped_rms_norm(corpus, weight, group_size=width, eps=3.0)
            for capacity, logical in ((1, 1), (8, 8), (64, 33)):
                x = torch.empty(capacity, width * groups, dtype=dtype, device="cuda")
                output = torch.empty_like(x)
                primitive = ZeroCenteredRMSNormPrimitive(
                    x, weight, output, epsilon=3.0, group_size=width
                )
                primitive.launch(x, weight, output)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    primitive.launch(x, weight, output)
                for reverse in (False, True):
                    x.fill_(float("nan"))
                    source = corpus[:logical].flip(0) if reverse else corpus[:logical]
                    x[:logical].copy_(source)
                    graph.replay()
                    actual = output[:logical].flip(0) if reverse else output[:logical]
                    assert torch.equal(actual, expected[:logical])
                # Random inputs must preserve exactly the established standalone
                # per-group primitive's reduction and affine contract.
                x.normal_()
                graph.replay()
                for group in range(groups):
                    lo, hi = group * width, (group + 1) * width
                    single = x[:, lo:hi].contiguous()
                    affine = weight[lo:hi].contiguous()
                    separate = torch.empty_like(single)
                    ZeroCenteredRMSNormPrimitive(single, affine, separate, epsilon=3.0).launch(
                        single, affine, separate
                    )
                    assert torch.equal(output[:, lo:hi], separate)
