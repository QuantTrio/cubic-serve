"""Capture identity for stages with independently named typed state fields.

These bounded boundary-row checks compare the same optimized body before and
inside capture. They neither establish independent numerical correctness nor
replace cross-M, prefix/COW, transaction-write-set or topology qualification.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from cubic_serve.guardrail.compare import stable_digest, tensor_digest
from cubic_serve.runtime.cache.quantization import CacheGroupAxis, KVCacheStorageSpec


def state_audit_horizon(
    model_steps: int, maximum_steps: int, storage: KVCacheStorageSpec,
) -> int:
    """Cover declared model state and a closed column group plus its next read.

    Channel groups close within one token and do not extend the time axis.
    Audit callers and the independent oracle must prepare the same extent.
    """
    if any(type(value) is not int or value <= 0
           for value in (model_steps, maximum_steps)):
        raise ValueError("state audit requires positive integer extents")
    if not isinstance(storage, KVCacheStorageSpec):
        raise TypeError("state audit requires a resolved cache storage contract")
    # Each side must cross its own first sealing boundary. Page allocation's
    # common alignment is not a time horizon: coprime groups must not multiply
    # the number of independent model steps.
    column_steps = max((
        side.group_size + 1 for side in (storage.key, storage.value)
        if side.dtype.quantized and side.axis is CacheGroupAxis.TOKENS
    ), default=0)
    return min(maximum_steps, max(model_steps, column_steps))


def prepare_conditioning_inputs(
    stage: Any, logical_m: int, *, position: int = 0, positions: tuple[int, ...] | None = None,
) -> None:
    """Bind the same finite external feedback corpus at every physical bucket.

    Round capture may leave padding poison in a provider's conditioning input.
    It is not durable state: production binds it before every proposal, and an
    owner-migration witness must likewise supply its own input on every step.
    Only live physical rows are initialized; padding poison remains observable.
    """
    feedback = getattr(stage.buffers, "target_hidden", None)
    if feedback is None:
        return
    if (
        not isinstance(feedback, Tensor)
        or feedback.ndim != 2
        or feedback.shape[0] != stage.captured_m
        or not feedback.is_floating_point()
        or not 0 < logical_m <= stage.captured_m
    ):
        raise ValueError("conditioning input differs from the declared stage capacity")
    columns = torch.arange(feedback.shape[1], dtype=torch.float32, device=feedback.device)
    if positions is None:
        values = columns[None] + position * 3
    else:
        if len(positions) != logical_m or any(type(p) is not int or p < 0 for p in positions):
            raise ValueError("conditioning positions differ from logical row ownership")
        values = columns[None] + torch.tensor(
            positions, device=feedback.device, dtype=torch.float32
        )[:, None] * 3
    feedback[:logical_m].copy_(((values.remainder(17) - 8) * (2.0**-5)).to(feedback.dtype))


def prepare_rows(stage: Any, logical_m: int, *, poison: bool = False) -> None:
    """Prepare cold one-token owners, keeping padding in disjoint private scratch.

    A review bucket can exceed the persistent request-owner budget. Its cold
    capture witness borrows that stage's private scratch for the extra live
    rows; this does not grant those rows persistent serving ownership.
    """
    if not 0 < logical_m <= stage.captured_m:
        raise ValueError("logical rows exceed the fixed capture bucket")
    arena, buffers = stage.arena, stage.buffers
    if arena.spec.page_capacity < logical_m:
        raise ValueError("capture witness has insufficient distinct live pages")
    for slots in buffers.state_slot_maps:
        if slots.shape != (2, 1, stage.captured_m):
            raise ValueError("typed capture witness requires independent state fields")
    stage.prepare_scratch()
    owners = buffers.state_slot_maps[0][1, 0, :logical_m].long().clone()
    persistent = min(logical_m, arena.spec.state_slot_capacity)
    storage=getattr(arena,"shared_storage",None)
    owners[:persistent] = torch.arange(
        persistent, device=arena.device, dtype=torch.int64)
    if storage is not None:
        owners[:persistent] = arena.spec.state_slot_capacity-1-owners[:persistent]
        owner_bytes=persistent*(0 if storage.state_layout is None else storage.state_layout.entry_bytes)
        if logical_m*storage.page_layout.entry_bytes+owner_bytes>storage.budget_bytes:
            raise ValueError("capture witness exceeds shared byte ownership")
    pages = torch.arange(logical_m, device=arena.device, dtype=torch.int64)
    arena.reset_state_rows(owners)
    arena.reset_page_rows(pages)
    buffers.row_activity[:logical_m].fill_(1)
    buffers.segment_lengths[:logical_m].fill_(1)
    buffers.page_source[:logical_m].copy_(pages)
    buffers.page_destination[:logical_m].copy_(pages)
    buffers.page_table[:logical_m].copy_(pages[:, None])
    for slots in buffers.state_slot_maps:
        slots[:, :, :logical_m].copy_(owners[None, None])
    from cubic_serve.models.builtin import builtin_text_config

    vocab = builtin_text_config(stage.config).vocabulary_size
    tokens = torch.arange(stage.captured_m, device=arena.device, dtype=torch.int64)
    tokens.mul_(7919).add_(17).remainder_(vocab)
    stage.fill_token_ids(tokens)
    stage.hidden.zero_()
    prepare_conditioning_inputs(stage, logical_m)
    if poison and logical_m < stage.captured_m:
        # Unmasked rotary/table indexing still requires valid metadata. Poison
        # only inactive data, never reinterpret activity as an indexing mask.
        stage.hidden[logical_m:].fill_(float("nan"))
        feedback = getattr(buffers, "target_hidden", None)
        if feedback is not None:
            feedback[logical_m:].fill_(float("nan"))
        private_states = buffers.state_slot_maps[0][1, 0, logical_m:].long()
        private_pages = buffers.page_destination[logical_m:].long()
        for value in arena.states.values():
            if value.is_floating_point():
                value.index_fill_(0, private_states, float("nan"))
        for value in (arena.key, arena.value, *arena.page_fields.values()):
            if value.is_floating_point():
                value.index_fill_(0, private_pages, float("nan"))
    stage.set_active_rows(logical_m)


def boundary_snapshot(stage: Any, logical_m: int) -> dict[str, Tensor]:
    """Copy one completed row and every declared field of its durable owners.

    Call only after the forward has completed on all participating ranks. This
    function is not an observer callback and deliberately performs host copies.
    """
    if not 0 < logical_m <= stage.captured_m:
        raise ValueError("snapshot row exceeds the fixed capture bucket")
    row = logical_m - 1
    b, arena = stage.buffers, stage.arena
    page = int(b.page_destination[row].item())
    slots = {int(mapping[1, 0, row].item()) for mapping in b.state_slot_maps}
    if len(slots) != 1:
        raise ValueError("typed capture snapshot requires one shared state owner")
    owner = slots.pop()
    values = {
        "hidden": stage.hidden[row : row + 1],
        "key": arena.key[page : page + 1],
        "value": arena.value[page : page + 1],
        "written": arena.written[page : page + 1],
    }
    if stage.logits is not None:
        values["logits"] = stage.logits[row : row + 1]
    feedback = getattr(stage, "draft_feedback", None)
    if feedback is not None:
        values["draft_feedback"] = feedback[row : row + 1]
    values.update({f"state:{name}": v[owner : owner + 1] for name, v in arena.states.items()})
    values.update({f"page:{name}": v[page : page + 1] for name, v in arena.page_fields.items()})
    return {name: value.detach().to("cpu", copy=True) for name, value in values.items()}


def compare_snapshot(reference: dict[str, Tensor], candidate: dict[str, Tensor]) -> str:
    """Fail on any typed bit difference or nonfinite active-state value."""
    if reference.keys() != candidate.keys():
        raise ValueError("capture identity field set changed")
    differences = []
    for name, left in reference.items():
        right = candidate[name]
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or (right.is_floating_point() and not bool(torch.isfinite(right).all()))
            or not torch.equal(
                left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
            )
        ):
            differences.append(name)
    if differences:
        raise ValueError(f"capture identity failed for fields: {', '.join(differences)}")
    return stable_digest({name: tensor_digest(value) for name, value in candidate.items()})


def audit_owner_migration(executors: dict[int, Any], *, max_sequence_tokens: int) -> str:
    """Check one owner's real batch path across buckets, COW and a page boundary.

    This bounded startup check does not substitute for concurrent cross-M or
    prefix-fork acceptance. Every forward completes before snapshots are read.
    COW and owned controls follow the same bucket schedule: this isolates lossless
    ownership changes from permitted differences in bucket reduction arithmetic.
    """
    import torch.distributed as dist

    from cubic_serve.runtime.continuous_batch import BatchRow

    capacities = tuple(sorted(executors))
    if not capacities or capacities[0] != 1:
        raise ValueError("typed owner migration requires a permanent physical M1 bucket")
    base = executors[1].body
    arena = base.arena
    if any(executor.body.arena is not arena for executor in executors.values()):
        raise ValueError("bucket migration requires one shared durable owner arena")
    block = arena.layout.block_size_tokens
    horizon = min(max_sequence_tokens, block + 2)
    pages = (horizon + block - 1) // block
    if horizon < 1 or arena.spec.state_slot_capacity < 2 or arena.spec.page_capacity < 2 * pages:
        raise ValueError("typed owner witness has insufficient context or COW storage")
    expected: list[str] = []
    witnesses = []
    for mode in ("m1_owned", "migrating_owned", "migrating_cow"):
        for position in range(horizon):
            capacity = 1 if mode == "m1_owned" else capacities[position % len(capacities)]
            executor = executors[capacity]
            stage = executor.body
            cow = mode == "migrating_cow"
            destination = position % 2 if cow else 0
            source = 1 - destination if cow and position else destination
            page_index, offset = divmod(position, block)
            page = 2 * page_index + destination
            page_source = 2 * page_index + (source if cow and offset else destination)
            table = (
                *(2 * previous + (((previous + 1) * block - 1) % 2 if cow else 0)
                  for previous in range(page_index)),
                page,
            )
            from cubic_serve.models.builtin import builtin_text_config

            token = (position * 7919 + 17) % builtin_text_config(stage.config).vocabulary_size
            row = BatchRow(
                "typed-capture-owner", token, position, source, destination,
                page_source, page, table, reset_state=position == 0,
                new_pages=(page,) if offset == 0 else (), is_decode=True,
            )
            prepare_conditioning_inputs(stage, 1, position=position)
            executor.replay_batch((row,))
            stage.synchronize()
            dist.barrier()
            snapshot = boundary_snapshot(stage, 1)
            try:
                digest = compare_snapshot(snapshot, snapshot)
            except ValueError as error:
                raise ValueError(
                    f"typed owner finite check failed: mode={mode} M={capacity} "
                    f"position={position}: {error}"
                ) from error
            if mode == "migrating_owned":
                expected.append(digest)
            elif mode == "migrating_cow" and digest != expected[position]:
                raise ValueError(
                    f"typed owner identity failed: mode={mode} M={capacity} position={position}"
                )
            witnesses.append((mode, capacity, position, digest))
    return stable_digest({"schema": "cubic.typed-owner-migration.v2", "witnesses": witnesses})


def audit_prefix_forks(executors: dict[int, Any], *, max_sequence_tokens: int) -> str:
    """Prove typed segmented prefix ownership and two immutable-source forks.

    This is a bounded execution/state witness, not an independent numerical
    oracle. It uses the serving row ABI and completes every replay before reads.
    """
    import torch.distributed as dist

    from cubic_serve.runtime.continuous_batch import BatchRow

    capacities = tuple(sorted(executors))
    base = executors[1].body
    arena = base.arena
    block = arena.layout.block_size_tokens
    if max_sequence_tokens <= block:
        # No page-aligned prefix can leave room for a generated token.
        return stable_digest({"schema": "cubic.typed-prefix-fork.v1", "eligible": False})
    steps = min(2, max_sequence_tokens - block)
    if arena.spec.state_slot_capacity < 4 or arena.spec.page_capacity < 6:
        raise ValueError("typed prefix witness requires four owners and six pages")
    owners = torch.arange(4, device=arena.device, dtype=torch.int64)
    pages = torch.arange(6, device=arena.device, dtype=torch.int64)

    def snapshot(owner: int, page: int, *, output: bool = False) -> dict[str, Tensor]:
        values = {
            "key": arena.key[page], "value": arena.value[page], "written": arena.written[page],
            **{f"state:{name}": value[owner] for name, value in arena.states.items()},
            **{f"page:{name}": value[page] for name, value in arena.page_fields.items()},
        }
        if output:
            values["hidden"] = base.hidden[0]
            if base.logits is not None:
                values["logits"] = base.logits[0]
        return {name: value.detach().to("cpu", copy=True) for name, value in values.items()}

    def replay(rows: tuple[BatchRow, ...]) -> None:
        capacity = next(m for m in capacities if m >= len(rows))
        executor = executors[capacity]
        prepare_conditioning_inputs(
            executor.body, len(rows), positions=tuple(row.position for row in rows)
        )
        executor.replay_batch(rows)
        executor.slot.synchronize()
        dist.barrier()

    def token(position: int) -> int:
        from cubic_serve.models.builtin import builtin_text_config

        return (101 + position) % builtin_text_config(base.config).vocabulary_size

    try:
        arena.reset_state_rows(owners)
        arena.reset_page_rows(pages)
        chunk = min(block, max(capacities))
        for start in range(0, block, chunk):
            stop = min(start + chunk, block)
            replay(tuple(
                BatchRow(
                    "typed-prefix", token(position), position, 0 if start == 0 else 1,
                    1, 1, 1, (1,), reset_state=position == 0,
                    new_pages=(1,) if position == 0 else (),
                    segment_length=stop - start if position == start else 0,
                    is_decode=False,
                )
                for position in range(start, stop)
            ))
        immutable = snapshot(1, 1)
        # Match the cached prefix's segmentation and capacity. Serial M1 and
        # segmented prefill may legitimately use different reduction plans;
        # that numerical comparison belongs to the independent cross-M audit.
        for start in range(0, block, chunk):
            stop = min(start + chunk, block)
            replay(tuple(
                BatchRow(
                    "typed-owned", token(position), position, 0, 0, 4, 4, (4,),
                    reset_state=position == 0,
                    new_pages=(4,) if position == 0 else (),
                    segment_length=stop - start if position == start else 0,
                    is_decode=False,
                )
                for position in range(start, stop)
            ))
        compare_snapshot(immutable, snapshot(0, 4))
        expected = []
        for position in range(block, block + steps):
            page = 5
            replay((BatchRow(
                "typed-owned", token(position), position, 0, 0, page, page,
                (4,) if position < block else (4, 5), reset_state=position == 0,
                new_pages=(page,) if position % block == 0 else (), is_decode=True,
            ),))
            expected.append(snapshot(0, 5, output=True))
        witnesses = []
        for destination in (2, 3):
            for step in range(steps):
                position = block + step
                replay((BatchRow(
                    "typed-fork", token(position), position,
                    1 if step == 0 else destination, destination, destination, destination,
                    (1, destination), new_pages=(destination,) if step == 0 else (),
                    is_decode=True,
                ),))
                witnesses.append(compare_snapshot(
                    expected[step], snapshot(destination, destination, output=True)
                ))
                compare_snapshot(immutable, snapshot(1, 1))
        return stable_digest({
            "schema": "cubic.typed-prefix-fork.v2", "prefix_tokens": block,
            "continuation_tokens": steps, "prefill_chunk": chunk, "witnesses": witnesses,
        })
    finally:
        arena.reset_state_rows(owners)
        arena.reset_page_rows(pages)
        torch.cuda.synchronize(arena.device)
