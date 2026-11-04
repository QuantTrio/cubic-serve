import os

import pytest
import torch

from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
from cubic_serve.ops.cuda.embedding.ngram import bind_ngram_hash
from cubic_serve.ops.eager.ngram import hash_indices

pytestmark = pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)


@pytest.mark.parametrize("strided", [False, True])
def test_ngram_graph_segments_overflow_eos_prefix_and_padding(strided):
    configure_reference_math()
    eos = 31
    sequences = ([2**62 + 3, eos, 7], [9, 12], [20])
    initial = torch.tensor([[2**54 + 1, 2**61 + 3]], dtype=torch.int64)
    multipliers = torch.tensor([2**62 + 57, 1000000007, 97], dtype=torch.int64)
    vocabs = torch.tensor([11, 13, 17, 19], dtype=torch.int64)
    offsets = torch.tensor([0, 11, 24, 41], dtype=torch.int64)
    expected, histories = {}, {}
    with ReferenceExecutionContext():
        for owner, sequence in enumerate(sequences):
            history = initial if owner != 1 else torch.full_like(initial, eos)
            for step, token in enumerate(sequence):
                ids, history = hash_indices(
                    torch.tensor([[token]]),
                    history,
                    multipliers,
                    vocabs,
                    offsets,
                    eos_token_id=eos,
                )
                expected[owner, step] = ids[0, 0]
                histories[owner, step] = history[0]
    for capacity in (8, 16):
        tokens = torch.zeros(capacity, device="cuda", dtype=torch.int64)
        backing = torch.full((20, 4 if strided else 2), 987654321,
                             device="cuda", dtype=torch.int64)
        states = backing[:, :2]
        source, destination, segments, activity = [
            torch.zeros(capacity, device="cuda", dtype=torch.int32) for _ in range(4)
        ]
        output = torch.empty(capacity, 4, device="cuda", dtype=torch.int64)
        gpu_metadata = tuple(t.cuda() for t in (multipliers, vocabs, offsets))
        binding = bind_ngram_hash(
            tokens,
            states,
            source,
            destination,
            segments,
            activity,
            *gpu_metadata,
            output,
            eos_token_id=eos,
        )
        binding.launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            binding.launch()
        for order in ((0, 1, 2), (2, 0, 1)):
            tokens.fill_(-999)
            source.fill_(19)
            destination.fill_(19)
            segments.fill_(999)
            activity.zero_()
            output.fill_(-123)
            states.fill_(987654321)
            states[0].copy_(initial[0])
            row = 0
            mapping = []
            for owner in order:
                sequence = sequences[owner]
                for step, token in enumerate(sequence):
                    tokens[row] = token
                    source[row] = -1 if owner == 1 else 0
                    destination[row] = 1 + owner * 3 + step
                    segments[row] = len(sequence) if step == 0 else 0
                    activity[row] = 1
                    mapping.append((row, owner, step))
                    row += 1
            graph.replay()
            result, after = output.cpu(), states.cpu()
            for physical, owner, step in mapping:
                assert torch.equal(result[physical], expected[owner, step])
                assert torch.equal(after[1 + owner * 3 + step], histories[owner, step])
            assert torch.equal(after[0], initial[0])
            assert (result[row:] == -123).all()
            assert (after[10:] == 987654321).all()
            assert (backing[:, 2:] == 987654321).all()


def test_ngram_m1_in_place_continuation_and_new_sequence():
    configure_reference_math()
    eos = 31
    token = torch.empty(1, device="cuda", dtype=torch.int64)
    state = torch.empty(1, 2, device="cuda", dtype=torch.int64)
    source = torch.full((1,), -1, device="cuda", dtype=torch.int32)
    destination = torch.zeros_like(source)
    segment = torch.ones_like(source)
    active = torch.ones_like(source)
    multipliers = torch.tensor([2**61 + 1, 100003, 97], device="cuda", dtype=torch.int64)
    vocab = torch.tensor([11, 13], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([0, 11], device="cuda", dtype=torch.int64)
    output = torch.empty(1, 2, device="cuda", dtype=torch.int64)
    op = bind_ngram_hash(
        token,
        state,
        source,
        destination,
        segment,
        active,
        multipliers,
        vocab,
        offsets,
        output,
        eos_token_id=eos,
    )
    token.zero_()
    op.launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.launch()
    expected_state = torch.full((1, 2), eos, dtype=torch.int64)
    with ReferenceExecutionContext():
        for i, value in enumerate((7, 13, eos, 2**62 + 1, 9)):
            token.fill_(value)
            source.fill_(-1 if i == 0 else 0)
            graph.replay()
            expected, expected_state = hash_indices(
                torch.tensor([[value]]),
                expected_state,
                multipliers.cpu(),
                vocab.cpu(),
                offsets.cpu(),
                eos_token_id=eos,
            )
            assert torch.equal(output.cpu(), expected[:, 0])
            assert torch.equal(state.cpu(), expected_state)
