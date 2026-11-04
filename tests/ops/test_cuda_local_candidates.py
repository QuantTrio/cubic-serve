from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import torch

from cubic_serve.ops.cuda.sampling.local_candidates import LocalCandidatePlan


def reference(logits, temperatures, activity, count, width, start, all_random=False):
    """Independent CPU witness, with unchanged top-k as a producer boundary."""
    # A tied cutoff belongs to the unchanged CUDA top-k producer. Witness its
    # selected values/indices, rather than substituting a CPU tie algorithm.
    values, indices = torch.topk(logits[:, :width], count, dim=-1)
    values, indices = values.cpu(), indices.cpu()
    source = logits[:, :width].cpu()
    temps, active = temperatures.cpu(), activity.cpu()
    scores = torch.full_like(values, float('nan'))
    ids = torch.full_like(indices, -1)
    for row in range(len(source)):
        if not active[row]:
            continue
        if not bool(torch.isfinite(source[row]).all()):
            ids[row].fill_(-2)
        elif temps[row] == 0:
            if not all_random:
                best = max(float(v) for v in source[row])
                index = next(i for i, v in enumerate(source[row]) if float(v) == best)
                scores[row, 0] = source[row, index]
                ids[row, 0] = index + start
        else:
            scores[row] = values[row]
            ids[row] = indices[row] + start
            # Independently witness score/ID provenance and the top-k cutoff.
            self_values = source[row, indices[row]]
            torch.testing.assert_close(self_values, values[row], rtol=0, atol=0)
            assert bool((source[row] <= values[row, -1]).sum() >= width - count + 1)
    return scores, ids


@unittest.skipUnless(
    os.environ.get("CUBIC_RUN_CUDA_LOCAL_CANDIDATE_TESTS") == "1",
    "explicit real CUDA candidate audit",
)
class LocalCandidateTests(unittest.TestCase):
    def test_terminal_mask_consumes_only_active_visible_logits(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            rows, width, stride, count, start = 4, 67, 73, 20, 17
            original = (((torch.arange(rows * stride).reshape(rows, stride) * 7) % 131)
                        .float() * 0.125).to(dtype)
            logits = original.cuda()
            activity = torch.tensor([True, False, True, True], device="cuda")
            metadata = torch.tensor([3], dtype=torch.int32, device="cuda")
            mask = torch.full((rows, 3), -1, dtype=torch.int32, device="cuda")
            scores = torch.empty(rows, count, device="cuda", dtype=dtype)
            ids = torch.empty(rows, count, device="cuda", dtype=torch.int64)
            plan = LocalCandidatePlan(local_logits=logits,
                workspace=SimpleNamespace(local_scores=scores, local_token_ids=ids),
                active_rows=rows, visible_width=width, local_count=count, vocab_start=start,
                row_activity=activity, metadata=metadata, token_mask=mask, consume_logits=True)
            self.assertEqual(plan.masked_logits.data_ptr(), logits.data_ptr())
            self.assertEqual(plan.masked_logits.stride(), logits.stride())
            temps = torch.tensor([0.7, 0.7, 0.0, 0.7], device="cuda")
            plan.launch(temperatures=temps)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                plan.launch(temperatures=temps)
            for shift in (0, 1):
                host_mask = torch.full((rows, 3), -1, dtype=torch.int32)
                expected = original.clone()
                for row in range(rows):
                    for column in range(width):
                        if column % 3 == (row + shift) % 3:
                            token = start + column
                            host_mask[row, token // 32] &= ~(1 << (token % 32))
                            if row in (0, 2):
                                expected[row, column] = -float("inf")
                logits.copy_(original)
                mask.copy_(host_mask)
                graph.replay()
                self.assertTrue(torch.equal(logits.cpu().view(torch.uint8),
                                            expected.view(torch.uint8)))
                for row in (0, 2):
                    values, indices = torch.topk(expected[row, :width], count)
                    take = 1 if row == 2 else count
                    self.assertTrue(torch.equal(scores[row, :take].cpu(), values[:take]))
                    self.assertTrue(torch.equal(ids[row, :take].cpu(), indices[:take] + start))
                    if take == 1:
                        self.assertTrue(bool(torch.isnan(scores[row, 1:]).all()))
                        self.assertTrue(bool((ids[row, 1:] == -1).all()))
                self.assertTrue(bool((ids[[1, 3]] == -1).all()))
            logits.copy_(original)
            excluded = next(j for j in range(width) if j % 3 == 1)
            logits[0, excluded] = float("nan")
            graph.replay()
            self.assertTrue(bool((ids[0] == -2).all()))
            graph.reset()

    def test_qualification_restores_external_storage_without_full_logits_gpu_backup(self):
        rows, width, count = 4, 62080, 20
        logits = torch.empty((rows, width + 7), dtype=torch.float32, device="cuda")
        # Include signed zero and noncanonical NaN payloads in the storage that
        # qualification temporarily borrows, including the invisible tail.
        payload = torch.arange(logits.numel(), dtype=torch.int32).reshape(logits.shape)
        payload[:, ::3] = -2147483648
        payload[:, 1::3] = 2143289345
        logits.view(torch.int32).copy_(payload)
        activity = torch.tensor([False, True, False, True], device="cuda")
        metadata = torch.tensor([2, 81], dtype=torch.int32, device="cuda")
        scores = torch.full((rows + 1, count + 3), -0.0, device="cuda")
        ids = torch.full(scores.shape, 991, dtype=torch.int64, device="cuda")
        plan = LocalCandidatePlan(
            local_logits=logits,
            workspace=SimpleNamespace(local_scores=scores, local_token_ids=ids),
            active_rows=rows, visible_width=width, local_count=count,
            vocab_start=13, row_activity=activity, metadata=metadata,
        )
        tensors = (logits, scores, ids, activity, metadata)
        originals = [t.cpu().view(torch.uint8).clone() for t in tensors]
        # The unchanged torch.topk producer has its own temporary allocation;
        # measure that separately from the qualification backup obligation.
        temperatures = torch.full((rows,), 0.7, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        plan.launch(temperatures=temperatures)
        torch.cuda.synchronize()
        producer_extra = torch.cuda.max_memory_allocated() - before
        for tensor, original in zip(tensors, originals, strict=True):
            tensor.view(torch.uint8).copy_(original)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        plan.qualify()
        extra = torch.cuda.max_memory_allocated() - before
        self.assertLess(extra - producer_extra, logits.numel() * logits.element_size() // 2)
        for tensor, original in zip(tensors, originals, strict=True):
            self.assertTrue(torch.equal(tensor.cpu().view(torch.uint8), original))
        self.assertTrue(plan.evidence["unchanged_rng"])
        launch = plan.launch

        def corrupt(*, temperatures):
            launch(temperatures=temperatures)
            ids[0, 0] = 2**40

        plan.launch = corrupt
        with self.assertRaisesRegex(RuntimeError, "selection boundary audit failed"):
            plan.qualify()
        for tensor, original in zip(tensors, originals, strict=True):
            self.assertTrue(torch.equal(tensor.cpu().view(torch.uint8), original))

    def test_captured_grammar_masks_preserve_shards_padding_and_nonfinite_guards(self):
        rows, width, start, count = 4, 67, 13, 20
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            logits = torch.arange(rows * width, device="cuda").reshape(rows, width).to(dtype)
            original = logits.clone()
            mask = torch.full((rows, 3), -1, dtype=torch.int32, device="cuda")
            activity = torch.tensor([True, True, True, False], device="cuda")
            temps = torch.tensor([0.0, 0.7, 0.7, 0.7], device="cuda")
            scores = torch.empty((rows, count), dtype=dtype, device="cuda")
            ids = torch.empty((rows, count), dtype=torch.int64, device="cuda")
            plan = LocalCandidatePlan(
                local_logits=logits,
                workspace=SimpleNamespace(local_scores=scores, local_token_ids=ids),
                active_rows=rows,
                visible_width=width,
                local_count=count,
                vocab_start=start,
                row_activity=activity,
                token_mask=mask,
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                plan.launch(temperatures=temps)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    plan.launch(temperatures=temps)
            torch.cuda.current_stream().wait_stream(stream)
            host_mask = torch.zeros((rows, 3), dtype=torch.int32)
            for row, allowed in enumerate(((14, 78), (15, 31, 79), (), ())):
                for token in allowed:
                    word = int(host_mask[row, token // 32]) & 0xFFFFFFFF
                    word |= 1 << (token % 32)
                    host_mask[row, token // 32] = word if word < 2**31 else word - 2**32
            mask.copy_(host_mask)
            logits[3].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(ids[0, 0].item(), 78)
            self.assertTrue(bool((ids[0, 1:] == -1).all()))
            self.assertEqual(ids[1, :3].tolist(), [79, 31, 15])
            self.assertTrue(bool((ids[1, 3:] == -1).all()))
            self.assertTrue(bool((ids[2:] == -1).all()))
            self.assertTrue(bool(torch.isnan(scores[2:]).all()))
            torch.testing.assert_close(logits[:3], original[:3], rtol=0, atol=0)
            # An excluded NaN is still an invalid model operand, never hidden by a mask.
            logits[1, 0] = float("nan")
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(bool((ids[1] == -2).all()))
            graph.reset()

    def test_borrowed_masked_logits_scratch_is_fixed_and_non_aliased(self):
        width, count = 67, 20
        scratch = torch.empty((8, width), dtype=torch.bfloat16, device="cuda")
        pool, stream = torch.cuda.graph_pool_handle(), torch.cuda.Stream()
        plans, graphs, results = [], [], []
        # Includes target/provider-sized plans and repeated equal capacities.
        for rows in (8, 1, 4, 8):
            logits = torch.randn((rows, width), device="cuda", dtype=scratch.dtype)
            mask = torch.full((rows, 3), -1, dtype=torch.int32, device="cuda")
            scores = torch.empty((rows, count), dtype=logits.dtype, device="cuda")
            ids = torch.empty((rows, count), dtype=torch.int64, device="cuda")
            kwargs = dict(local_logits=logits,
                          workspace=SimpleNamespace(local_scores=scores, local_token_ids=ids),
                          active_rows=rows, visible_width=width, local_count=count,
                          vocab_start=0, token_mask=mask)
            for bad in (logits, scratch[:rows, :-1], scratch[:rows].float()):
                with self.assertRaisesRegex(ValueError, "non-aliased"):
                    LocalCandidatePlan(**kwargs, masked_logits=bad)
            plan = LocalCandidatePlan(**kwargs, masked_logits=scratch[:rows])
            self.assertEqual(plan.masked_logits.data_ptr(), scratch.data_ptr())
            temperatures = torch.ones(rows, device="cuda")
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                plan.launch(temperatures=temperatures)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream, pool=pool):
                    plan.launch(temperatures=temperatures)
            torch.cuda.current_stream().wait_stream(stream)
            plans.append((plan, temperatures, logits.clone()))
            graphs.append(graph)
        # Replay out of capture order; outputs remain private after another
        # graph overwrites the shared temporary. Original logits stay immutable.
        for index in (1, 0, 3, 2, 0, 1):
            plan, temperatures, original = plans[index]
            scratch.fill_(float("nan"))
            expected = torch.topk(original, count, dim=-1)
            graphs[index].replay()
            torch.testing.assert_close(plan.outputs[0], expected.values, rtol=0, atol=0)
            self.assertTrue(torch.equal(plan.outputs[1], expected.indices))
            torch.testing.assert_close(plan.logits, original, rtol=0, atol=0)
            if results:
                old_plan, old_scores, old_ids = results[-1]
                if old_plan is not plan:
                    torch.testing.assert_close(old_plan.outputs[0], old_scores, rtol=0, atol=0)
                    self.assertTrue(torch.equal(old_plan.outputs[1], old_ids))
            results.append((plan, plan.outputs[0].clone(), plan.outputs[1].clone()))
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()

    def test_rank_arena_orders_shared_mask_copies_and_preserves_private_results(self):
        from cubic_serve.runtime.performance import _ConstraintSamplingArena

        width, count = 67, 20
        prototype = torch.empty((8, width), dtype=torch.bfloat16, device="cuda")
        arena = _ConstraintSamplingArena(SimpleNamespace(
            captured_m=8, vocab_size=width, _rank=0,
            slot=SimpleNamespace(logits=prototype, vocab_span=SimpleNamespace(start=0, stop=width),
                                 stream=torch.cuda.Stream()),
            _sampling=SimpleNamespace(temperatures=torch.ones(8, device="cuda")),
        ), 8)
        with self.assertRaisesRegex(RuntimeError, "geometry"):
            arena.logits_view(prototype.float())
        plans, graphs = [], []
        for rows in (8, 1, 4, 8):
            logits = torch.arange(width, device="cuda", dtype=prototype.dtype).repeat(rows, 1)
            scores = torch.empty((rows, count), dtype=logits.dtype, device="cuda")
            ids = torch.empty((rows, count), dtype=torch.int64, device="cuda")
            plan = LocalCandidatePlan(
                local_logits=logits,
                workspace=SimpleNamespace(local_scores=scores, local_token_ids=ids),
                active_rows=rows, visible_width=width, local_count=count, vocab_start=0,
                row_activity=arena.activity[:rows], token_mask=arena.mask[:rows],
                masked_logits=arena.logits_view(logits),
            )
            temps = torch.ones(rows, device="cuda")
            arena.capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(arena.capture_stream):
                plan.launch(temperatures=temps)
                arena.capture_stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=arena.capture_stream, pool=arena.graph_pool):
                    plan.launch(temperatures=temps)
            torch.cuda.current_stream().wait_stream(arena.capture_stream)
            plans.append(plan)
            graphs.append(graph)
        snapshots = []
        # No host or device-wide wait between launches. A shared-mask H2D must
        # wait for the previous graph reader and immutable result snapshots.
        for step in range(64):
            index = (1, 0, 3, 2)[step % 4]
            rows = plans[index].capacity
            allowed = ((step + 3) % width, (step + 29) % width)
            mask = torch.zeros((width + 31) // 32, dtype=torch.int32)
            for token in allowed:
                word = (int(mask[token // 32]) & 0xffffffff) | (1 << (token % 32))
                mask[token // 32] = word if word < 2**31 else word - 2**32
            owners = [row != rows - 1 or step % 2 == 0 for row in range(rows)]
            arena.bind([mask] * rows, owners)
            torch.cuda._sleep(100000)
            graphs[index].replay()
            snapshots.append((plans[index].outputs[1].clone(), owners, sorted(allowed, reverse=True)))
        torch.cuda.synchronize()
        for result, owners, allowed in snapshots:
            expected = [allowed + [-1] * (count - len(allowed)) if active else [-1] * count
                        for active in owners]
            self.assertEqual(result.cpu().tolist(), expected)
        self.assertEqual(len({p.masked_logits.untyped_storage().data_ptr() for p in plans}), 1)
        self.assertEqual(len({p.token_mask.untyped_storage().data_ptr() for p in plans}), 1)
        for graph in graphs:
            graph.reset()

    def test_owned_padding_nonfinite_and_tied_greedy(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for width in (31, 4097, 62080):
                with self.subTest(dtype=dtype, width=width):
                    gen = torch.Generator(device='cuda').manual_seed(78)
                    logits = torch.randn((8, width + 13), generator=gen,
                                         device='cuda', dtype=dtype)
                    logits[0, :width] = -2
                    logits[0, 3] = logits[0, width - 1] = 4
                    logits[2, 0] = float('nan')
                    logits[3, 17] = float('inf')
                    logits[4].fill_(float('nan'))  # Ownerless poison.
                    logits[:, width:].fill_(float('nan'))  # Vocabulary padding.
                    temperatures = torch.tensor([0., .7, .7, 0., .7, .7, 0., .7],
                                                device='cuda')
                    activity = torch.tensor([True, True, True, True, False, True, True, True],
                                            device='cuda')
                    count = 20
                    for all_random in (False, True):
                        scores = torch.full((9, count + 3), -9., dtype=dtype, device='cuda')
                        ids = torch.full((9, count + 3), 993, dtype=torch.int64, device='cuda')
                        workspace = SimpleNamespace(local_scores=scores, local_token_ids=ids)
                        expected = reference(logits, temperatures, activity, count, width,
                                             62080, all_random)
                        plan = LocalCandidatePlan(
                            local_logits=logits, workspace=workspace,
                            active_rows=8,
                            visible_width=width, local_count=count, vocab_start=62080,
                            row_activity=activity, all_random_rows=all_random)
                        plan.launch(temperatures=temperatures)
                        torch.testing.assert_close(scores[:8, :count].cpu(), expected[0],
                                                   rtol=0, atol=0, equal_nan=True)
                        self.assertTrue(torch.equal(ids[:8, :count].cpu(), expected[1]))
                        self.assertTrue(bool(torch.isnan(scores[:8, count:]).all()))
                        self.assertTrue(bool((ids[:8, count:] == -1).all()))
                        self.assertTrue(bool((scores[8] == -9).all()))
                        self.assertTrue(bool((ids[8] == 993).all()))

    def test_captured_dynamic_activity_and_permuted_rows(self):
        logits = torch.randn((4, 62080), dtype=torch.bfloat16, device='cuda')
        temperatures = torch.tensor([.7, 0., .7, 0.], device='cuda')
        activity = torch.ones(4, dtype=torch.bool, device='cuda')
        metadata = torch.tensor([4, 0], dtype=torch.int32, device='cuda')
        workspace = SimpleNamespace(
            local_scores=torch.empty((4, 20), dtype=logits.dtype, device='cuda'),
            local_token_ids=torch.empty((4, 20), dtype=torch.int64, device='cuda'))
        args = dict(local_logits=logits, workspace=workspace,
                    active_rows=4, visible_width=62080, local_count=20,
                    vocab_start=13, row_activity=activity, metadata=metadata)
        plan = LocalCandidatePlan(**args)
        plan.qualify()
        self.assertTrue(plan.evidence['unchanged_rng'])
        plan.launch(temperatures=temperatures)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan.launch(temperatures=temperatures)
        for mask in ([True, True, False, False], [False, True, False, True], [True] * 4):
            activity.copy_(torch.tensor(mask, device='cuda'))
            logits.copy_(logits.flip(0).clone())
            expected = reference(logits, temperatures, activity, 20, 62080, 13)
            graph.replay()
            torch.testing.assert_close(workspace.local_scores.cpu(), expected[0],
                                       rtol=0, atol=0, equal_nan=True)
            self.assertTrue(torch.equal(workspace.local_token_ids.cpu(), expected[1]))
        graph.reset()

    def test_same_corpus_across_capacity_and_physical_row_maps(self):
        width, count = 62080, 20
        corpus = ((torch.arange(3 * width).reshape(3, width) % 31) / 32).to(
            device='cuda', dtype=torch.bfloat16)
        # Distinct cutoff values witness cross-M identity independently of
        # torch.topk's unspecified cutoff-tie choice. Greedy ties remain exact.
        corpus[:, :count] = torch.arange(4, 4 + count, device='cuda')
        corpus[0, 3] = corpus[0, 42] = 32
        temps = torch.tensor([0., .7, .9], device='cuda')
        expected = reference(corpus, temps, torch.ones(3, dtype=torch.bool, device='cuda'),
                             count, width, 17)
        for capacity in (1, 2, 4, 8, 16, 32, 64, 128):
            n = min(3, capacity)
            maps = [tuple(range(n)), tuple(reversed(range(capacity - n, capacity)))]
            for physical in maps:
                with self.subTest(capacity=capacity, rows=physical):
                    logits = torch.full((capacity, width), float('nan'),
                                        device='cuda', dtype=corpus.dtype)
                    temperatures = torch.zeros(capacity, device='cuda')
                    activity = torch.zeros(capacity, dtype=torch.bool, device='cuda')
                    for owner, row in enumerate(physical):
                        logits[row].copy_(corpus[owner])
                        temperatures[row] = temps[owner]
                        activity[row] = True
                    workspace = SimpleNamespace(
                        local_scores=torch.empty((capacity, count), device='cuda',
                                                 dtype=corpus.dtype),
                        local_token_ids=torch.empty((capacity, count), device='cuda',
                                                    dtype=torch.int64))
                    plan = LocalCandidatePlan(
                        local_logits=logits, workspace=workspace, active_rows=capacity,
                        visible_width=width, local_count=count, vocab_start=17,
                        row_activity=activity)
                    plan.launch(temperatures=temperatures)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        plan.launch(temperatures=temperatures)
                    graph.replay()
                    torch.testing.assert_close(workspace.local_scores[list(physical)].cpu(),
                                               expected[0][:n], rtol=0, atol=0, equal_nan=True)
                    self.assertTrue(torch.equal(
                        workspace.local_token_ids[list(physical)].cpu(), expected[1][:n]))
                    self.assertTrue(bool(torch.isnan(workspace.local_scores[~activity]).all()))
                    self.assertTrue(bool((workspace.local_token_ids[~activity] == -1).all()))
                    graph.reset()


if __name__ == '__main__':
    unittest.main()
