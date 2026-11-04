"""Independent portable greedy decisions, padding, ownership and graph lifetime."""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cubic_serve.generation.tp_sampling import (
    ShardedSamplingParameters,
    ShardedSamplingWorkspace,
    TensorParallelSamplingError,
    sample_sharded_broadcast_tp,
)
from cubic_serve.models.parallel_weights import AxisSpan
from cubic_serve.ops.cuda.sampling.greedy import CapturedGreedySampler

RUN_REAL = os.environ.get("CUBIC_RUN_CUDA_GREEDY_TESTS") == "1"


def _check_qualification_protected_padding(plan: CapturedGreedySampler) -> None:
    from contextlib import nullcontext
    from unittest.mock import patch

    from cubic_serve.guardrail.quantized_graph import byte_exact

    original = [t.clone() for t in plan._operands]
    memory = torch.cuda.mem_get_info
    replay = plan._replay

    def corrupt_inactive_operand(*args, **kwargs):
        result = replay(*args, **kwargs)
        # The first fixture uses one active row. Every PP/TP rank owns these
        # parameters, including ranks without a final vocabulary shard.
        plan._call["parameters"].ordinals.view(torch.uint8)[-1].bitwise_xor_(1)
        return result

    for cpu_image in (False, True):
        admission = (patch.object(torch.cuda, "mem_get_info",
                                 side_effect=lambda device: (0, memory(device)[1]))
                     if cpu_image else nullcontext())
        plan._replay = corrupt_inactive_operand
        plan.qualified = False
        try:
            with admission, pytest.raises(TensorParallelSamplingError,
                                          match="changed protected operands"):
                plan.qualify()
            assert not plan.qualified
            assert all(byte_exact(t, before)
                       for t, before in zip(plan._operands, original, strict=True))
        finally:
            plan._replay = replay
    plan.qualify()


def _worker(rank: int, tp: int, pp: int, rendezvous: str, evidence_dir: str) -> None:
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=tp * pp,
        timeout=timedelta(seconds=120),
    )
    final_ranks = tuple(range(tp * (pp - 1), tp * pp))
    group = dist.new_group(list(final_ranks), backend="nccl")
    final = rank in final_ranks
    root = rank == final_ranks[0]
    receipts = []
    # Sampling borrows the already-owned model stream. Destroying its graphs
    # must leave the model stream and graph usable.
    model_stream = torch.cuda.Stream(device=device)
    model_state = torch.zeros(1, device=device)
    model_stream.wait_stream(torch.cuda.current_stream(device))
    model_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(model_graph, stream=model_stream):
        model_state.add_(1)
    torch.cuda.current_stream(device).wait_stream(model_stream)
    try:
        # TP4 deliberately includes two completely padded vocab shards.
        width, vocab = (33, 65) if tp == 4 else (4123, 4123 * tp - 7)
        spans = tuple(AxisSpan(i * width, (i + 1) * width) for i in range(tp))
        for dtype in (torch.bfloat16, torch.float16):
            corpus = ((torch.arange(3 * width).reshape(3, width) * 3) % 29 - 14).to(dtype)
            reference = []
            for row in range(3):
                candidates = []
                for span in spans:
                    visible = max(0, min(vocab, span.stop) - span.start)
                    if visible:
                        scores = corpus[row, :visible].float()
                        value, token = scores.max(dim=0)
                        candidates.append((float(value), int(token) + span.start))
                maximum = max(x[0] for x in candidates)
                reference.append(min(x[1] for x in candidates if x[0] == maximum))
            for capacity in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                params = ShardedSamplingParameters(
                    torch.zeros(capacity, device=device),
                    torch.full((capacity,), 20, dtype=torch.int64, device=device),
                    torch.full((capacity,), 0.95, device=device),
                    20,
                    torch.ones(capacity, dtype=torch.int64, device=device),
                    torch.zeros(capacity, dtype=torch.int64, device=device),
                )
                logits = (
                    torch.empty((capacity, width), dtype=dtype, device=device) if final else None
                )
                workspace = (
                    ShardedSamplingWorkspace(
                        torch.empty((capacity, 20), dtype=dtype, device=device),
                        torch.empty((capacity, 20), dtype=torch.int64, device=device),
                        torch.empty((tp, capacity, 20), dtype=dtype, device=device),
                        torch.empty((tp, capacity, 20), dtype=torch.int64, device=device),
                    )
                    if final
                    else None
                )
                kwargs = dict(
                    local_logits=logits,
                    local_vocab_span=spans[final_ranks.index(rank)] if final else None,
                    final_ranks=final_ranks,
                    final_vocab_spans=spans,
                    true_vocab_size=vocab,
                    token_ids=torch.empty(capacity, dtype=torch.int64, device=device),
                    parameters=params,
                    workspace=workspace,
                    final_tp_group=group if final else None,
                    world_group=dist.group.WORLD,
                    root_valid=torch.empty(capacity, dtype=torch.bool, device=device)
                    if root
                    else None,
                )
                plan = CapturedGreedySampler(**kwargs, capture_stream=model_stream)
                try:
                    assert plan.stream is model_stream
                    with pytest.raises(TensorParallelSamplingError, match="qualification"):
                        plan.replay(1, False)
                    plan.qualify()
                    if capacity == 4 and dtype == torch.bfloat16:
                        _check_qualification_protected_padding(plan)
                    for count in range(1, capacity + 1):
                        corpus_rows = tuple(row % len(reference) for row in range(count))
                        for permutation in (corpus_rows, tuple(reversed(corpus_rows))):
                            if logits is not None:
                                logits.fill_(float("nan"))
                                visible = max(
                                    0,
                                    min(vocab, kwargs["local_vocab_span"].stop)
                                    - kwargs["local_vocab_span"].start,
                                )
                                for destination, source in enumerate(permutation):
                                    logits[destination, :visible].copy_(corpus[source, :visible])
                            output = sample_sharded_broadcast_tp(
                                **kwargs,
                                device=device,
                                active_rows=count,
                                all_greedy_rows=True,
                                greedy_plan=plan,
                            )
                            expected = [reference[row] for row in permutation] + [-1] * (
                                capacity - count
                            )
                            assert output.cpu().tolist() == expected
                    for buffer, value in (
                        (params.temperatures, 1),
                        (params.top_ps, 0),
                        (params.top_ks, 21),
                        (params.ordinals, -1),
                    ):
                        saved = buffer.clone()
                        buffer[0] = value
                        with pytest.raises(TensorParallelSamplingError, match="invalid"):
                            plan.replay(count, False)
                        buffer.copy_(saved)
                    replacement = dict(kwargs, token_ids=kwargs["token_ids"].clone(), device=device)
                    with pytest.raises(TensorParallelSamplingError, match="startup binding"):
                        plan.validate_call(**replacement)
                    receipts.append(
                        dict(
                            dtype=str(dtype),
                            evidence=plan.evidence,
                            permuted_cross_m_tokens=True,
                            capacity=capacity,
                            logical_ms=list(range(1, capacity + 1)),
                            invalid_parameters=True,
                            foreign_buffer_rejected=True,
                        )
                    )
                finally:
                    plan.close()
                plan.close()
                with pytest.raises(TensorParallelSamplingError, match="qualification"):
                    plan.replay(1, False)
                before = model_state.item()
                model_graph.replay()
                assert model_state.item() == before + 1
        destination = Path(os.environ.get("CUBIC_GREEDY_EVIDENCE_DIR", evidence_dir))
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"greedy-TP{tp}-PP{pp}-rank{rank}.json").write_text(
            json.dumps(receipts, indent=2)
        )
    finally:
        torch.cuda.synchronize(device)
        model_graph.reset()
        dist.destroy_process_group()


@pytest.mark.skipif(not RUN_REAL, reason="explicit real CUDA qualification")
@pytest.mark.parametrize(("tp", "pp"), [(1, 1), (2, 1), (4, 1), (2, 2)])
def test_greedy_graph_numerical_ownership_and_lifetime(tp: int, pp: int) -> None:
    if torch.cuda.device_count() < tp * pp:
        pytest.skip("insufficient real CUDA devices")
    with TemporaryDirectory() as folder:
        mp.spawn(
            _worker, args=(tp, pp, f"file://{folder}/rendezvous", folder), nprocs=tp * pp, join=True
        )


def _receipt_reuse_worker(rank: int, rendezvous: str, folder: str) -> None:
    from cubic_serve.guardrail.compare import stable_digest
    from cubic_serve.runtime import quantized_qualification as q

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=4,
                            device_id=device)
    capacity, width, vocab = 4, 65, 253
    spans = tuple(AxisSpan(i * width, (i + 1) * width) for i in range(4))
    stream = torch.cuda.Stream(device=device)
    plans = []

    def consensus(valid):
        flag = torch.tensor([int(valid)], dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def make():
        parameters = ShardedSamplingParameters(
            torch.zeros(capacity, device=device),
            torch.full((capacity,), 20, dtype=torch.int64, device=device),
            torch.full((capacity,), .95, device=device), 20,
            torch.ones(capacity, dtype=torch.int64, device=device),
            torch.zeros(capacity, dtype=torch.int64, device=device))
        logits = (torch.arange(width, device=device).float() + rank * width).to(torch.bfloat16)
        logits = logits.expand(capacity, -1).clone()
        logits[:, max(0, vocab - rank * width):] = float("nan")
        workspace = ShardedSamplingWorkspace(
            torch.empty((capacity, 20), dtype=torch.bfloat16, device=device),
            torch.empty((capacity, 20), dtype=torch.int64, device=device),
            torch.empty((4, capacity, 20), dtype=torch.bfloat16, device=device),
            torch.empty((4, capacity, 20), dtype=torch.int64, device=device))
        kwargs = dict(local_logits=logits, local_vocab_span=spans[rank],
                      final_ranks=(0, 1, 2, 3), final_vocab_spans=spans,
                      true_vocab_size=vocab,
                      token_ids=torch.empty(capacity, dtype=torch.int64, device=device),
                      parameters=parameters, workspace=workspace,
                      final_tp_group=dist.group.WORLD, world_group=dist.group.WORLD,
                      root_valid=torch.empty(capacity, dtype=torch.bool, device=device)
                      if rank == 0 else None)
        plan = CapturedGreedySampler(**kwargs, capture_stream=stream)
        plans.append(plan)
        return plan, kwargs

    key = stable_digest(("sampler-receipt-reuse-unit", q._composition_source_identity(),
                         capacity, width, vocab, 4))
    try:
        first, old_inputs = make()
        initial = q.StartupQualification(key, rank, recording=True,
            root=Path(folder) / "receipts", consensus=consensus)
        signature = tuple(call.signature for call in first.calls)
        initial.component(first, signature, attributes=("qualified", "evidence"))
        initial.finish()
        first.close()

        current, inputs = make()
        assert inputs["token_ids"].data_ptr() != old_inputs["token_ids"].data_ptr()
        assert tuple(call.signature for call in current.calls) == signature
        def forbidden():
            raise AssertionError("matching complete sampler proof was recomputed")
        current.qualify = forbidden
        reused = q.StartupQualification(key, rank, recording=True,
            root=Path(folder) / "receipts", consensus=consensus)
        reused.component(current, signature, attributes=("qualified", "evidence"))
        reused.finish()
        assert reused.hits == 1 and current.qualified
        assert current.evidence == first.evidence
        inputs["token_ids"].fill_(-17)
        inputs["local_logits"][3].fill_(float("nan"))
        inactive = inputs["local_logits"][3].view(torch.uint8).clone()
        output = current.replay(3, False).cpu().tolist()
        assert output == [252, 252, 252, -1], output
        assert torch.equal(inputs["local_logits"][3].view(torch.uint8), inactive)
        with pytest.raises(TensorParallelSamplingError, match="startup binding"):
            current.validate_call(**dict(inputs, token_ids=inputs["token_ids"].clone(), device=device))
    finally:
        for plan in plans:
            plan.close()
        dist.destroy_process_group()


@pytest.mark.skipif(not RUN_REAL, reason="explicit real CUDA qualification")
def test_recording_reuses_complete_sampler_proof_with_fresh_live_bindings() -> None:
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four CUDA devices")
    with TemporaryDirectory() as folder:
        mp.spawn(_receipt_reuse_worker, args=(f"file://{folder}/rendezvous", folder),
                 nprocs=4, join=True)


from cubic_serve.guardrail.quantized_graph.speculative import philox_uniform as _philox_uniform


def _candidate_worker(rank, tp, pp, rendezvous, masked=False):
    import math

    from cubic_serve.ops.cuda.sampling import RootTopKTopPSamplingPrimitive

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=tp * pp, device_id=device
    )
    final_ranks = tuple(range(tp * (pp - 1), tp * pp))
    group = dist.new_group(list(final_ranks), backend="nccl")
    final, root = rank in final_ranks, rank == final_ranks[0]
    capacity, width, vocab = 4, 29, min(29 * tp - 7, 65)
    spans = tuple(AxisSpan(i * width, (i + 1) * width) for i in range(tp))
    stream = torch.cuda.Stream(device=device)
    try:
        for dtype in (torch.bfloat16, torch.float16):
            p = ShardedSamplingParameters(
                torch.zeros(capacity, device=device),
                torch.full((capacity,), 20, dtype=torch.int64, device=device),
                torch.ones(capacity, device=device),
                20,
                torch.ones(capacity, dtype=torch.int64, device=device),
                torch.zeros(capacity, dtype=torch.int64, device=device),
            )
            logits = torch.empty((capacity, width), device=device, dtype=dtype) if final else None
            w = (
                ShardedSamplingWorkspace(
                    torch.empty((capacity, 20), device=device, dtype=dtype),
                    torch.empty((capacity, 20), device=device, dtype=torch.int64),
                    torch.empty((tp, capacity, 20), device=device, dtype=dtype),
                    torch.empty((tp, capacity, 20), device=device, dtype=torch.int64),
                )
                if final
                else None
            )
            tokens = torch.empty(capacity, device=device, dtype=torch.int64)
            valid = torch.empty(capacity, device=device, dtype=torch.bool) if root else None
            arena = None
            token_mask = None
            if masked:
                from types import SimpleNamespace
                from cubic_serve.runtime.performance import _ConstraintSamplingArena

                arena = _ConstraintSamplingArena(SimpleNamespace(
                    captured_m=capacity, vocab_size=vocab, _rank=rank,
                    slot=SimpleNamespace(logits=logits, stream=stream,
                                         vocab_span=spans[rank - final_ranks[0]] if final else None),
                    _sampling=SimpleNamespace(temperatures=p.temperatures),
                ), capacity)
                token_mask = arena.mask
                stream = arena.capture_stream
            plan = CapturedGreedySampler(
                local_logits=logits,
                local_vocab_span=spans[rank - final_ranks[0]] if final else None,
                final_ranks=final_ranks,
                final_vocab_spans=spans,
                true_vocab_size=vocab,
                token_ids=tokens,
                parameters=p,
                workspace=w,
                final_tp_group=group if final else None,
                world_group=dist.group.WORLD,
                root_valid=valid,
                capture_stream=stream,
                token_mask=token_mask,
                masked_logits=arena.logits_view(logits) if arena is not None else None,
                graph_pool=arena.graph_pool if arena is not None else None,
            )
            primitive = (
                RootTopKTopPSamplingPrimitive(
                    w.gathered_scores,
                    w.gathered_token_ids,
                    p.temperatures,
                    p.top_ps,
                    p.top_ks,
                    p.seeds,
                    p.ordinals,
                    tokens,
                    valid,
                    allow_underfilled=masked,
                )
                if root
                else None
            )

            def body(plan=plan, primitive=primitive):
                plan.prepare_candidates()
                plan.sample_candidates(primitive)
                plan._broadcast()

            try:
                plan.qualify()
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    body()
                    body()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    body()
                stream.synchronize()
                for count, permutation, bad in [
                    (n, order, bad)
                    for n in (1, 3, 4)
                    for order in (tuple(range(n)), tuple(reversed(range(n))))
                    for bad in (False, True)
                ]:
                    # Distinct exactly representable scores; explicit physical
                    # row permutation with RNG and data following logical owner.
                    corpus = torch.full((capacity, width), float("nan"), dtype=dtype)
                    expected = []
                    for row, owner in enumerate(permutation):
                        temp = 0.0 if owner % 2 == 0 else 0.8
                        seed, ordinal = 0x123456789 + owner, (1 << 34) + owner
                        p.temperatures[row] = temp
                        p.seeds[row] = seed
                        p.ordinals[row] = ordinal
                        p.top_ps[row] = 0.95
                        p.top_ks[row] = 20
                        columns = [(i * 7 + owner * 3) % vocab / 16 for i in range(vocab)]
                        if final:
                            span = spans[rank - final_ranks[0]]
                            visible = max(0, min(span.stop, vocab) - span.start)
                            corpus[row, :visible] = torch.tensor(
                                columns[span.start : span.start + visible], dtype=dtype
                            )
                        allowed = list(range(vocab))
                        if masked:
                            allowed = [i for i in range(vocab) if (i + row) % 5 == 0][:3]
                            mask_row = torch.zeros((vocab + 31) // 32, dtype=torch.int32)
                            for token in allowed:
                                value = int(mask_row[token // 32]) & 0xFFFFFFFF
                                value |= 1 << (token % 32)
                                mask_row[token // 32] = value if value < 2**31 else value - 2**32
                            token_mask[row].copy_(mask_row)
                        ordered = sorted(allowed, key=lambda i: (-columns[i], i))[:20]
                        if temp == 0:
                            expected.append(ordered[0])
                            continue
                        weights = [
                            math.exp((columns[i] - columns[ordered[0]]) / temp) for i in ordered
                        ]
                        probabilities = [v / sum(weights) for v in weights]
                        kept = []
                        cdf = 0.0
                        for i, prob in zip(ordered, probabilities, strict=False):
                            if cdf <= float(p.top_ps[row]):
                                kept.append((i, prob))
                            cdf += prob
                        u = _philox_uniform(seed ^ (seed >> 32) ^ (ordinal >> 32), ordinal)
                        threshold = u * sum(prob for _, prob in kept)
                        cdf = 0.0
                        for i, prob in kept:
                            cdf += prob
                            if cdf > threshold:
                                expected.append(i)
                                break
                    if logits is not None:
                        if bad and root:
                            corpus[0, 0] = float("nan")
                        logits.copy_(corpus)
                    protected = [
                        v.clone() for v in (p.temperatures, p.top_ps, p.top_ks, p.seeds, p.ordinals)
                    ]
                    plan.metadata[0] = count
                    plan.metadata[1] = 0
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        graph.replay()
                    stream.synchronize()
                    assert bool(plan.final_valid) == (not bad)
                    if not bad:
                        assert tokens.cpu().tolist() == expected + [-1] * (capacity - count)
                    else:
                        assert tokens[0].item() == -1
                    # Exercise the uncaptured full-physical-row reference used
                    # by resident stochastic composition qualification. The
                    # independent CPU distribution above witnesses both paths.
                    reference_rejected = False
                    try:
                        plan._reference_stochastic(count, primitive)
                    except TensorParallelSamplingError:
                        reference_rejected = True
                    assert reference_rejected == bad
                    if not bad:
                        assert tokens.cpu().tolist() == expected + [-1] * (capacity - count)
                    for actual, before in zip(
                        (p.temperatures, p.top_ps, p.top_ks, p.seeds, p.ordinals),
                        protected,
                        strict=False,
                    ):
                        assert torch.equal(actual, before)
                graph.reset()
            finally:
                plan.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not RUN_REAL, reason="explicit real CUDA qualification")
@pytest.mark.parametrize(("tp", "pp"), [(1, 1), (2, 1), (4, 1), (2, 2)])
def test_captured_candidates_keep_sampling_rng_activity_and_tp_pp_ownership(tp, pp):
    if torch.cuda.device_count() < tp * pp:
        pytest.skip("insufficient CUDA devices")
    with TemporaryDirectory() as folder:
        mp.spawn(
            _candidate_worker,
            args=(tp, pp, f"file://{folder}/rendezvous"),
            nprocs=tp * pp,
            join=True,
        )


@pytest.mark.skipif(not RUN_REAL, reason="explicit real CUDA qualification")
@pytest.mark.parametrize(("tp", "pp"), [(1, 1), (2, 1), (4, 1), (2, 2)])
def test_masked_captured_candidates_match_independent_philox_distribution(tp, pp):
    if torch.cuda.device_count() < tp * pp:
        pytest.skip("insufficient CUDA devices")
    with TemporaryDirectory() as folder:
        mp.spawn(
            _candidate_worker,
            args=(tp, pp, f"file://{folder}/rendezvous", True),
            nprocs=tp * pp,
            join=True,
        )
