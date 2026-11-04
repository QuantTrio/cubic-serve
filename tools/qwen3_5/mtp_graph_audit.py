"""Real TP4 eager/CUDA-Graph audit for the Qwen3.5 MTP layer."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import torch
import torch.distributed as dist

from cubic_serve.contracts.execution import ParallelTopology
from cubic_serve.executor.mtp_runner import Qwen3_5TPMTPBody, Qwen3_5TPMTPGraphSlot
from cubic_serve.generation.sampler import SamplingParameters
from cubic_serve.generation.tp_sampling import sample_sharded_broadcast_tp
from cubic_serve.models.checkpoint import inspect_checkpoint
from cubic_serve.models.qwen3_5 import (
    load_qwen3_5_rank_weights,
    parse_qwen3_5_config_json,
    plan_qwen3_5_checkpoint_weights,
    plan_qwen3_5_dense_weights,
)
from cubic_serve.ops.cuda.sampling import RootTopKTopPSamplingPrimitive
from cubic_serve.runtime.continuous_batch import BatchRow
from cubic_serve.runtime.performance import _SamplingState


def _snapshot(body: Qwen3_5TPMTPBody) -> dict[str, torch.Tensor]:
    return {
        "hidden": body.buffers.norm.clone(),
        "logits": body.buffers.logits.clone(),
        "key": body.arena.key[0].clone(),
        "value": body.arena.value[0].clone(),
        "written": body.arena.written[0].clone(),
    }


def _snapshot_rows(body: Qwen3_5TPMTPBody, pages: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "hidden": body.buffers.norm.clone(),
        "logits": body.buffers.logits.clone(),
        "key": body.arena.key.index_select(0, pages).clone(),
        "value": body.arena.value.index_select(0, pages).clone(),
        "written": body.arena.written.index_select(0, pages).clone(),
    }


def _bind_c1_step(
    body: Qwen3_5TPMTPBody,
    *,
    token_id: int,
    position: int,
    target_hidden: torch.Tensor,
) -> None:
    page = position // 16
    body.bind_token_ids(
        torch.tensor((token_id,), dtype=torch.int64, device=body.buffers.norm.device)
    )
    body.buffers.target_hidden.copy_(target_hidden.unsqueeze(0))
    body.buffers.position_ids.fill_(position)
    body.buffers.page_destination.fill_(page)
    body.buffers.token_offsets.fill_(position % 16)
    body.buffers.page_table.zero_()
    body.buffers.page_table[0, : page + 1] = torch.arange(
        page + 1, dtype=torch.int32, device=body.buffers.norm.device
    )
    body.buffers.sequence_lengths.fill_(position + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        root = args.checkpoint.resolve(strict=True)
        config = parse_qwen3_5_config_json((root / "config.json").read_bytes())
        inventory = inspect_checkpoint(root)
        weight_plan = plan_qwen3_5_checkpoint_weights(config, inventory)
        topology = ParallelTopology.single_node(
            topology_id="qwen3_5.mtp.audit.tp4",
            device_indices=(0, 1, 2, 3),
            tp_size=4,
            pp_size=1,
        )
        parallel_plan = plan_qwen3_5_dense_weights(config, weight_plan, topology, include_mtp=True)
        weights = load_qwen3_5_rank_weights(
            inventory=inventory,
            weight_plan=weight_plan,
            parallel_plan=parallel_plan,
            rank=rank,
            device=torch.device("cuda", rank),
        )
        tp_group = dist.new_group((0, 1, 2, 3), backend="nccl")
        body = Qwen3_5TPMTPBody(
            config=config,
            parallel_plan=parallel_plan,
            weights=weights,
            tp_group=tp_group,
            captured_m=1,
            table_pages=8,
            page_capacity=128,
            max_proposal_tokens=8,
        )
        token = torch.tensor([config.graph.modality_tokens.vision_end], device=weights.device)
        body.bind_token_ids(token)
        body.buffers.position_ids.fill_(75)
        body.buffers.page_destination.fill_(4)
        body.buffers.token_offsets.fill_(11)
        body.buffers.page_table.zero_()
        body.buffers.page_table[0, :5] = torch.arange(5, dtype=torch.int32, device=weights.device)
        body.buffers.sequence_lengths.fill_(76)
        generator = torch.Generator(device=weights.device).manual_seed(20260901)
        body.buffers.target_hidden.copy_(
            torch.randn(
                body.buffers.target_hidden.shape, generator=generator, device=weights.device
            ).to(torch.bfloat16)
        )
        body.arena.key.zero_()
        body.arena.value.zero_()
        body.arena.written.zero_()
        dist.barrier()
        body.launch()
        torch.cuda.synchronize()
        eager = _snapshot(body)

        sequence_length = 128
        sequence_pages = torch.arange(8, dtype=torch.int64, device=weights.device)
        sequence_hidden = torch.randn(
            (sequence_length, config.graph.text.hidden_size),
            generator=torch.Generator(device=weights.device).manual_seed(20260903),
            device=weights.device,
        ).to(torch.bfloat16)
        sequence_tokens = tuple(
            (config.graph.modality_tokens.vision_end + step) % config.graph.text.vocabulary_size
            for step in range(sequence_length)
        )
        body.arena.key.zero_()
        body.arena.value.zero_()
        body.arena.written.zero_()
        eager_sequence: list[tuple[torch.Tensor, torch.Tensor]] = []
        for position, token_id in enumerate(sequence_tokens):
            _bind_c1_step(
                body,
                token_id=token_id,
                position=position,
                target_hidden=sequence_hidden[position],
            )
            body.launch()
            torch.cuda.synchronize()
            eager_sequence.append(
                (body.buffers.norm.cpu().clone(), body.buffers.logits.cpu().clone())
            )
        eager_sequence_state = {
            name: value.cpu() for name, value in _snapshot_rows(body, sequence_pages).items()
        }

        _bind_c1_step(
            body,
            token_id=int(token.item()),
            position=75,
            target_hidden=sequence_hidden[0],
        )
        body.buffers.target_hidden.copy_(
            torch.randn(
                body.buffers.target_hidden.shape,
                generator=torch.Generator(device=weights.device).manual_seed(20260901),
                device=weights.device,
            ).to(torch.bfloat16)
        )
        body.arena.key.zero_()
        body.arena.value.zero_()
        body.arena.written.zero_()
        slot = Qwen3_5TPMTPGraphSlot.capture(body)
        body.arena.key.zero_()
        body.arena.value.zero_()
        body.arena.written.zero_()
        with torch.cuda.stream(slot.stream):
            slot.replay()
        slot.synchronize()
        graph = _snapshot(body)
        exact = all(torch.equal(eager[name], graph[name]) for name in eager)

        body.arena.key.zero_()
        body.arena.value.zero_()
        body.arena.written.zero_()
        sequence_exact = True
        for position, token_id in enumerate(sequence_tokens):
            with torch.cuda.stream(slot.stream):
                _bind_c1_step(
                    body,
                    token_id=token_id,
                    position=position,
                    target_hidden=sequence_hidden[position],
                )
                slot.replay()
            slot.synchronize()
            expected_hidden, expected_logits = eager_sequence[position]
            sequence_exact = sequence_exact and torch.equal(
                body.buffers.norm.cpu(), expected_hidden
            )
            sequence_exact = sequence_exact and torch.equal(
                body.buffers.logits.cpu(), expected_logits
            )
        graph_sequence_state = {
            name: value.cpu() for name, value in _snapshot_rows(body, sequence_pages).items()
        }
        sequence_exact = sequence_exact and all(
            torch.equal(eager_sequence_state[name], graph_sequence_state[name])
            for name in eager_sequence_state
        )
        warm_body = Qwen3_5TPMTPBody(
            config=config,
            parallel_plan=parallel_plan,
            weights=weights,
            tp_group=tp_group,
            captured_m=64,
            table_pages=4,
            page_capacity=128,
            max_proposal_tokens=8,
            arena=body.arena,
            packed_gate_up=body._gate_up,
            mrope_tables=body.mrope_tables,
        )
        warm_body.bind_token_ids(torch.zeros(64, dtype=torch.int64, device=weights.device))
        warm_body.buffers.target_hidden.zero_()
        warm_body.buffers.position_ids.zero_()
        warm_body.buffers.token_offsets.zero_()
        warm_body.buffers.sequence_lengths.fill_(1)
        for index in range(64):
            warm_body.buffers.page_destination[index].fill_(64 + index)
            warm_body.buffers.page_table[index].fill_(64 + index)
        warm_slot = Qwen3_5TPMTPGraphSlot.capture(warm_body)
        with torch.cuda.stream(warm_slot.stream):
            warm_slot.replay()
        warm_slot.synchronize()
        sampled = torch.empty(1, dtype=torch.int64, device=weights.device)
        sampling = _SamplingState(
            1,
            tp_size=4,
            device=weights.device,
            final_rank=True,
            sampling_root=rank == 0,
            token_ids=sampled,
        )
        row = BatchRow(
            "audit",
            1,
            75,
            0,
            0,
            0,
            4,
            (0, 1, 2, 3, 4),
            sampling=SamplingParameters(1.0, 0.95, 20, 20260901),
        )
        sampling.bind((row,))
        if rank == 0:
            assert sampling.root_valid is not None
            sampling.root_sampler = RootTopKTopPSamplingPrimitive(
                sampling.workspace.gathered_scores,
                sampling.workspace.gathered_token_ids,
                sampling.temperatures,
                sampling.top_ps,
                sampling.top_ks,
                sampling.seeds,
                sampling.ordinals,
                sampled,
                sampling.root_valid,
            )
        spans = tuple(
            next(
                binding.spans[0]
                for binding in parallel_plan.bindings_for_rank(owner)
                if binding.tensor_name == "lm_head.weight"
            )
            for owner in range(4)
        )
        slot.replay()
        sample_sharded_broadcast_tp(
            local_logits=slot.logits,
            local_vocab_span=slot.vocab_span,
            final_ranks=(0, 1, 2, 3),
            final_vocab_spans=spans,
            true_vocab_size=config.graph.text.vocabulary_size,
            active_rows=1,
            token_ids=sampled,
            parameters=sampling.parameters,
            workspace=sampling.workspace,
            final_tp_group=tp_group,
            world_group=dist.group.WORLD,
            device=weights.device,
            root_sampler=sampling.root_sampler,
            root_valid=sampling.root_valid,
            groups_prevalidated=True,
            prevalidated_rank=rank,
        )
        torch.cuda.synchronize()
        body32 = Qwen3_5TPMTPBody(
            config=config,
            parallel_plan=parallel_plan,
            weights=weights,
            tp_group=tp_group,
            captured_m=32,
            table_pages=8,
            page_capacity=64,
            max_proposal_tokens=8,
            packed_gate_up=body._gate_up,
            mrope_tables=body.mrope_tables,
        )
        body32.bind_token_ids(
            torch.full(
                (32,),
                config.graph.modality_tokens.vision_end,
                dtype=torch.int64,
                device=weights.device,
            )
        )
        body32.buffers.position_ids.fill_(75)
        body32.buffers.token_offsets.fill_(11)
        body32.buffers.sequence_lengths.fill_(76)
        pages32 = torch.arange(4, 36, dtype=torch.int64, device=weights.device)
        body32.buffers.page_destination.copy_(pages32.to(torch.int32))
        body32.buffers.page_table.zero_()
        body32.buffers.page_table[:, :4] = torch.arange(4, dtype=torch.int32, device=weights.device)
        body32.buffers.page_table[:, 4] = pages32.to(torch.int32)
        generator32 = torch.Generator(device=weights.device).manual_seed(20260902)
        body32.buffers.target_hidden.copy_(
            torch.randn(
                body32.buffers.target_hidden.shape,
                generator=generator32,
                device=weights.device,
            ).to(torch.bfloat16)
        )
        body32.arena.key.zero_()
        body32.arena.value.zero_()
        body32.arena.written.zero_()
        dist.barrier()
        body32.launch()
        torch.cuda.synchronize()
        eager32 = _snapshot_rows(body32, pages32)
        body32.arena.key.zero_()
        body32.arena.value.zero_()
        body32.arena.written.zero_()
        slot32 = Qwen3_5TPMTPGraphSlot.capture(body32)
        body32.arena.key.zero_()
        body32.arena.value.zero_()
        body32.arena.written.zero_()
        with torch.cuda.stream(slot32.stream):
            slot32.replay()
        slot32.synchronize()
        graph32 = _snapshot_rows(body32, pages32)
        exact32 = all(torch.equal(eager32[name], graph32[name]) for name in eager32)

        gathered: list[tuple[bool, bool, bool] | None] = [None] * 4
        dist.all_gather_object(gathered, (exact, exact32, sequence_exact))
        if rank == 0:
            print(
                {
                    "mtp_tp4_c1_eager_graph_exact": all(
                        result is not None and result[0] for result in gathered
                    ),
                    "mtp_tp4_c32_eager_graph_exact": all(
                        result is not None and result[1] for result in gathered
                    ),
                    "mtp_tp4_c1_128_step_eager_graph_exact": all(
                        result is not None and result[2] for result in gathered
                    ),
                    "sampled_token": int(sampled.item()),
                },
                flush=True,
            )
        if not exact or not exact32 or not sequence_exact:
            raise RuntimeError("MTP CUDA Graph differs from same-body eager")
        del slot32, body32, warm_slot, warm_body, slot, body, sampling, weights
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
