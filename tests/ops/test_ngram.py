"""Causal chunking, EOS reset and signed-wrap witnesses for lexical state."""

import torch

from cubic_serve.executor.runner import ReferenceExecutionContext
from cubic_serve.ops.eager.ngram import dilated_depthwise_conv, hash_indices


def test_hash_wrap_eos_and_chunk_identity():
    tokens = torch.tensor([[7, 9, 99, 4, 2, 8], [99, 3, 8, 99, 99, 6]])
    history = torch.full((2, 2), 99)
    multipliers = torch.tensor([9223372036854775783, 4611686018427387903, 10007])
    sizes = torch.tensor([101, 103, 107, 109])
    offsets = torch.tensor([0, 101, 204, 311])
    with ReferenceExecutionContext():
        whole, final = hash_indices(tokens, history, multipliers, sizes, offsets, eos_token_id=99)
        state = history
        chunks = []
        for start, stop in ((0, 1), (1, 4), (4, 6)):
            ids, state = hash_indices(
                tokens[:, start:stop], state, multipliers, sizes, offsets, eos_token_id=99
            )
            chunks.append(ids)
    assert torch.equal(whole, torch.cat(chunks, 1))
    assert torch.equal(state, final)
    assert torch.equal(history, torch.full((2, 2), 99))

    # Independent Python arbitrary-precision witness, explicitly wrapping signed 64-bit.
    def signed(n):
        return (n + 2**63) % 2**64 - 2**63

    expected = []
    for row in tokens.tolist():
        context = [99, 99]
        output = []
        for token in row:
            prior = list(reversed(context))
            if 99 in prior:
                first = prior.index(99)
                prior[first:] = [99] * (len(prior) - first)
            mixed = signed(token * int(multipliers[0]))
            indices = []
            for lag, previous in enumerate(prior, 1):
                mixed ^= signed(previous * int(multipliers[lag]))
                for head in range(2):
                    h = (lag - 1) * 2 + head
                    indices.append(mixed % int(sizes[h]) + int(offsets[h]))
            output.append(indices)
            context = [context[-1], token]
        expected.append(output)
    assert torch.equal(whole, torch.tensor(expected))


def test_dilated_conv_chunk_identity_and_immutable_state():
    torch.manual_seed(42)
    x = torch.randn(2, 11, 5).bfloat16()
    weight = torch.randn(5, 1, 4).bfloat16()
    initial = torch.zeros(2, 9, 5, dtype=x.dtype)
    with ReferenceExecutionContext():
        whole, final = dilated_depthwise_conv(x, initial, weight, dilation=3)
        state = initial
        rows = []
        for t in range(x.shape[1]):
            out, state = dilated_depthwise_conv(x[:, t : t + 1], state, weight, dilation=3)
            rows.append(out)
    assert torch.equal(whole, torch.cat(rows, 1))
    assert torch.equal(final, state)
    assert torch.equal(final, x[:, -9:])
    assert not bool(initial.any())
