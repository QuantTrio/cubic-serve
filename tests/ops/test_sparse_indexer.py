import torch

from cubic_serve.executor.runner import ReferenceExecutionContext
from cubic_serve.ops.eager.sparse_indexer import select_blocks


def test_causal_complete_blocks_and_incomplete_tail():
    q = torch.tensor([[[[1.0, 0.0], [-1.0, 0.0]]]]).expand(1, 5, 2, 2)
    keys = torch.tensor([[[1.0, 0.0], [10.0, 0.0], [100.0, 0.0]]])
    lengths = torch.tensor([[1, 4, 7, 8, 11]])
    with ReferenceExecutionContext():
        result = select_blocks(q, keys, lengths, compression=4, token_budget=4)
    assert result.tolist() == [
        [
            [0, -1, -1, -1, -1, -1, -1],
            [0, 1, 2, 3, -1, -1, -1],
            [0, 1, 2, 3, 4, 5, 6],
            [4, 5, 6, 7, -1, -1, -1],
            [4, 5, 6, 7, 8, 9, 10],
        ]
    ]


def test_ties_use_original_block_order():
    with ReferenceExecutionContext():
        result = select_blocks(
            torch.zeros(1, 1, 2, 3),
            torch.zeros(1, 4, 3),
            torch.tensor([[8]]),
            compression=2,
            token_budget=4,
        )
    assert result.tolist() == [[[0, 1, 2, 3, -1]]]
