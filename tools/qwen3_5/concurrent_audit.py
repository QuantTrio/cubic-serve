"""Capture real serving review frames for an independent concurrent target audit.

Diagnostic launcher only: snapshots synchronize and invalidate timing results.
The target oracle freezes the actual pre-review BF16 cache/state and recorded
speculative tokens. Cold-prefix recomputation is a separate diagnostic mode. It does not qualify
the proposal model, verify/repair, padding poison, or cross-M metamorphic identity.
Missing coverage and unexplained numerical differences always remain unqualified.
The caller must correlate captured request IDs with the public load trace: file
labels count participating requests, and do not establish public concurrency alone.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path


def freeze_review_prefix(body, head):
    """Copy the source page of an unfinished COW and the pre-review owner state."""
    import torch

    from cubic_serve.runtime import performance as p

    length = head.position
    block = body.arena.key.shape[2]
    pages = list(head.page_table[: (length + block - 1) // block])
    if length % block:
        pages[-1] = head.page_source
    index = torch.tensor(pages, dtype=torch.long, device=body.hidden.device)

    def kv(value):
        return value.index_select(0, index).transpose(0, 1).flatten(1, 2)[
            :, :length
        ].clone()

    return {
        "length": length,
        "kv_key": kv(body.arena.key),
        "kv_value": kv(body.arena.value),
        "kv_written": kv(body.arena.written),
        "conv": body.arena.states[p.CONV_STATE_ID][head.state_source_slot].clone(),
        "recurrent": body.arena.states[p.RECURRENT_STATE_ID][head.state_source_slot].clone(),
    }


def install_capture() -> None:
    import torch

    from cubic_serve.runtime import performance as p

    root = Path(os.environ["CUBIC_CONCURRENT_AUDIT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    (root / f"capture-source-pid{os.getpid()}.py").write_bytes(Path(__file__).read_bytes())
    wanted = set(map(int, os.environ.get("CUBIC_CONCURRENT_AUDIT_C", "8,32,33").split(",")))
    prefix_limit = int(os.environ.get("CUBIC_CONCURRENT_AUDIT_PREFIX_LIMIT", "2048"))
    reserve_original = p._graph_runtime_reserve_bytes

    def reserve(bodies, wave_lanes):
        body = next(iter(bodies.values()))
        state_bytes = sum(v[0].numel() * v.element_size() for v in body.arena.states.values())
        token_bytes = sum(
            v[0, :, 0].numel() * v.element_size()
            for v in (body.arena.key, body.arena.value, body.arena.written)
        )
        # Diagnostic snapshots consume real memory. Reserve their declared bound
        # before KV admission instead of stealing the serving runtime's reserve.
        snapshots = (max(wanted) + 1) * (state_bytes + prefix_limit * token_bytes)
        return reserve_original(bodies, wave_lanes) + snapshots + (256 << 20)

    p._graph_runtime_reserve_bytes = reserve
    done: set[int] = set()
    current = None
    consumed: dict[str, dict[int, tuple[int, tuple[int, ...] | None]]] = {}

    def append(path, value):
        with path.open("a") as stream:
            stream.write(json.dumps(value, separators=(",", ":")) + "\n")

    submit_original = p._ProcessController.submit

    def submit(self, command, output):
        result = submit_original(self, command, output)
        append(
            root / "requests.jsonl",
            {
                "request_id": command.request_id,
                "input_ids": command.input_ids,
                "positions": command.positions,
                "sampling": dataclasses.asdict(command.sampling),
                "max_tokens": command.max_tokens,
            },
        )
        return result

    p._ProcessController.submit = submit
    receive_original = p._ProcessController._receive_batch

    def receive(self, command):
        result = receive_original(self, command)
        append(root / "results.jsonl", dataclasses.asdict(result))
        return result

    p._ProcessController._receive_batch = receive
    deserialize_original = p._deserialize_worker_command

    def deserialize(*args, **kwargs):
        nonlocal current
        current = deserialize_original(*args, **kwargs)
        return current

    p._deserialize_worker_command = deserialize
    bind_original = p._CapturedGraphRank.bind_batch

    def bind(self, rows):
        result = bind_original(self, rows)
        self._concurrent_audit_rows = None
        if not isinstance(current, p._BatchCommand):
            return result
        for row in rows:
            consumed.setdefault(row.request_id, {})[row.position] = (
                row.token_id,
                row.position_axes,
            )
        # Record the actual review tokens after proposal substitution, not the
        # controller's placeholder tokens. No model callback or graph is active.
        segments = p._batch_segments(rows)
        count = len(segments)
        if count in wanted and count not in done and all(r.is_decode for r in rows):
            heads = [rows[start] for start, stop in segments]
            if min(r.sample_ordinal for r in heads) >= 8 and (
                count != 33 or len(rows) < self.captured_m
            ):
                if max(head.position for head in heads) > prefix_limit:
                    raise RuntimeError("HARNESS: review prefix exceeds reserved snapshot bound")
                self._concurrent_audit_rows = tuple(rows)
                # Freeze the inputs of this review transaction, before its graph
                # can overwrite an exclusively owned recurrent slot or KV page.
                # Keep stream-ordered immutable device copies until completion.
                self._concurrent_audit_prefixes = [
                    freeze_review_prefix(self.body, head) for head in heads
                ]
        return result

    p._CapturedGraphRank.bind_batch = bind
    complete_original = p._CapturedGraphRank.complete_bound

    def complete(self, *, replay):
        result = complete_original(self, replay=replay)
        rows = getattr(self, "_concurrent_audit_rows", None)
        if rows is None or not replay:
            return result
        self._concurrent_audit_rows = None
        segments = p._batch_segments(rows)
        count = len(segments)
        body = self.body
        # The entire graph has completed on the current stream before any D2H
        # or diagnostic collective. Check the ordered rank schedule first.
        schedule = (
            current.batch_id,
            self.captured_m,
            [(r.request_id, r.token_id, r.position, r.segment_length) for r in rows],
        )
        schedules = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(schedules, schedule)
        if any(peer != schedule for peer in schedules):
            raise RuntimeError("HARNESS: concurrent review rank schedules differ")
        frames = []
        for (start, stop), prefix in zip(
            segments, self._concurrent_audit_prefixes, strict=True
        ):
            segment_rows = rows[start:stop]
            last = segment_rows[-1]
            length = last.position + 1
            block = body.arena.key.shape[2]
            pages = list(last.page_table[: (length + block - 1) // block])
            index = torch.tensor(pages, dtype=torch.long, device=body.hidden.device)

            def kv(value, index=index, length=length):
                selected = value.index_select(0, index)
                return selected.transpose(0, 1).flatten(1, 2)[:, :length].cpu()

            frames.append(
                {
                    "rows": segment_rows,
                    "consumed": dict(consumed[last.request_id]),
                    "prefix": {
                        name: value.cpu() if isinstance(value, torch.Tensor) else value
                        for name, value in prefix.items()
                    },
                    "logits": None
                    if body.logits is None
                    else body.logits[start:stop].cpu().clone(),
                    "kv_key": kv(body.arena.key),
                    "kv_value": kv(body.arena.value),
                    "kv_written": kv(body.arena.written),
                    "conv": body.arena.states[p.CONV_STATE_ID][last.state_destination_slot]
                    .cpu()
                    .clone(),
                    "recurrent": body.arena.states[p.RECURRENT_STATE_ID][
                        last.state_destination_slot
                    ]
                    .cpu()
                    .clone(),
                }
            )
        report = {
            "batch_id": current.batch_id,
            "concurrency_in_frame": count,
            "logical_m": len(rows),
            "physical_m": self.captured_m,
            "padding_rows": self.captured_m - len(rows),
            "rank": body.rank,
            "full_layer_ids": body.full_layer_ids,
            "gdn_layer_ids": body.gdn_layer_ids,
            "vocab_span": body.vocab_span,
            "frames": frames,
            "qualification": "captured_only",
        }
        torch.save(report, root / f"c{count}-rank{body.rank}.pt")
        del self._concurrent_audit_prefixes
        done.add(count)
        return result

    p._CapturedGraphRank.complete_bound = complete


def validate_consumed_prefix(frame, request, batches, batch_id):
    """Witness input provenance with committed tokens, never as a float oracle."""
    head = frame["rows"][0]
    prompt_length = len(request["input_ids"])
    remaining = sum(position < prompt_length for position in frame["consumed"])
    generated = []
    for result in batches:
        if result["batch_id"] >= batch_id:
            break
        indices = [i for i, rid in enumerate(result["request_ids"]) if rid == head.request_id]
        if not indices:
            continue
        if remaining:
            remaining -= len(indices)
            if remaining < 0:
                raise RuntimeError("HARNESS: prefill consumption does not match captured rows")
            if remaining == 0:
                generated.append(result["token_ids"][indices[-1]])
        else:
            segment = next(
                (s for s in result["segment_results"] if s["request_id"] == head.request_id), None
            )
            if segment is not None:
                generated.extend(segment["output_token_ids"])
            else:
                generated.append(result["token_ids"][indices[-1]])
    if not generated or len(generated) != head.sample_ordinal or generated[-1] != head.token_id:
        raise RuntimeError("HARNESS: committed sampling trace differs from review root")
    exact = request["input_ids"] + generated[:-1] + [r.token_id for r in frame["rows"]]
    for position, (token, _) in frame["consumed"].items():
        if position < len(exact) and token != exact[position]:
            raise RuntimeError("HARNESS: consumed prefix differs from committed token trace")


def restore_frozen_prefix(reference, captured, members, head, device):
    """Restore the transaction's exact rank-local inputs into an independent cache."""
    import torch

    for frame, _, _ in members:
        if frame["prefix"]["length"] != head or frame["rows"][0].position != head:
            raise RuntimeError("HARNESS: frozen prefix length differs from review position")
    cache = reference.empty_cache(len(members))
    full = {layer: i for i, layer in enumerate(captured["full_layer_ids"])}
    gdn = {layer: i for i, layer in enumerate(captured["gdn_layer_ids"])}
    restored = []
    for layer, empty in enumerate(cache.layers, start=cache.layer_start):
        def stack(name, local):
            return torch.stack(
                [member[0]["prefix"][name][local] for member in members]
            ).to(device)

        if layer in full:
            local = full[layer]
            if not all(
                bool(member[0]["prefix"]["kv_written"][local].all())
                for member in members
            ):
                raise RuntimeError("EXECUTION: frozen prefix KV is unwritten")
            restored.append(dataclasses.replace(
                empty,
                key=stack("kv_key", local).transpose(1, 2).contiguous(),
                value=stack("kv_value", local).transpose(1, 2).contiguous(),
            ))
        else:
            local = gdn[layer]
            restored.append(dataclasses.replace(
                empty,
                convolution=stack("conv", local),
                recurrent=stack("recurrent", local),
            ))
    cache = dataclasses.replace(
        cache, sequence_length=head, layers=tuple(restored)
    )
    return cache


def reference_worker(
    rank, init_method, directory, admission, concurrencies=(8, 32, 33), prefix_mode="frozen"
):
    import traceback

    import torch

    from cubic_serve.executor.distributed_reference import Qwen3_5DistributedReferenceModel
    from cubic_serve.executor.parallel import initialize_eager_nccl
    from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
    from cubic_serve.models.qwen3_5_moe.distributed import Qwen3_5MoeDistributedReferenceModel
    from cubic_serve.runtime import performance as p

    root = Path(directory)
    output = root / f"reference-rank{rank}.json"
    reports = []
    try:
        torch.set_num_threads(1)
        configure_reference_math()
        requests = {
            r["request_id"]: r
            for r in map(json.loads, (root / "requests.jsonl").read_text().splitlines())
        }
        batches = list(map(json.loads, (root / "results.jsonl").read_text().splitlines()))
        with initialize_eager_nccl(
            admission.sealed, rank=rank, init_method=init_method, timeout_seconds=600
        ) as runtime:
            moe = isinstance(admission.config, p.Qwen3_5MoeConfig)
            loader = p.load_qwen3_5_moe_rank_weights if moe else p.load_qwen3_5_rank_weights
            weights = loader(
                inventory=admission.inventory,
                weight_plan=admission.weight_plan,
                parallel_plan=admission.parallel_plan,
                rank=rank,
                device=runtime.device,
            )
            cls = Qwen3_5MoeDistributedReferenceModel if moe else Qwen3_5DistributedReferenceModel
            reference = cls(
                config=admission.config,
                weights=weights,
                parallel_plan=admission.parallel_plan,
                runtime=runtime,
            )
            reference._canonical.recurrent_state_dtype = torch.bfloat16
            for concurrency in concurrencies:
                path = root / f"c{concurrency}-rank{rank}.pt"
                if not path.exists():
                    reports.append({"concurrency": concurrency, "status": "coverage_missing"})
                    continue
                captured = torch.load(path, weights_only=False, map_location="cpu")
                groups = {}
                for frame in captured["frames"]:
                    rows = frame["rows"]
                    req = requests[rows[0].request_id]
                    validate_consumed_prefix(frame, req, batches, captured["batch_id"])
                    length = rows[-1].position + 1
                    tokens = list(req["input_ids"])
                    axes = [tuple(v) for v in req["positions"]] or [
                        (i, i, i) for i in range(len(tokens))
                    ]
                    tokens.extend([-1] * max(0, length - len(tokens)))
                    axes.extend([(i, i, i) for i in range(len(axes), length)])
                    for position, (token, position_axes) in frame["consumed"].items():
                        if position < length:
                            tokens[position] = token
                            axes[position] = position_axes or (position,) * 3
                    if any(token < 0 for token in tokens[:length]):
                        raise RuntimeError("HARNESS: consumed prefix has an unrecorded token")
                    if prefix_mode == "frozen" and "prefix" not in frame:
                        raise RuntimeError("HARNESS: frozen pre-review prefix snapshot is missing")
                    head = rows[0].position if prefix_mode == "frozen" else 0
                    groups.setdefault((length, head), []).append(
                        (frame, tokens[:length], axes[:length])
                    )
                comparison = []
                oracle_frames = []
                all_members = [member for members in groups.values() for member in members]
                shared = 0
                for position in range(
                    min(m[0]["rows"][0].position for m in all_members)
                    if prefix_mode == "cold" else 0
                ):
                    if len({(m[1][position], tuple(m[2][position])) for m in all_members}) != 1:
                        break
                    shared += 1
                prefix_cache = None
                if shared:
                    first = all_members[0]
                    with ReferenceExecutionContext():
                        prefix = reference.forward(
                            torch.tensor([first[1][:shared]], device=runtime.device),
                            position_ids=torch.tensor(
                                [first[2][:shared]], device=runtime.device
                            ).permute(2, 0, 1),
                            epoch_id=f"concurrent-c{concurrency}-independent-common-prefix",
                        )
                    prefix_cache = prefix.cache
                    del prefix
                for (length, head), members in sorted(groups.items()):
                    print(
                        f"reference rank={rank} C={concurrency} B={len(members)} "
                        f"T={length} independent_prefix={shared}",
                        flush=True,
                    )
                    input_ids = torch.tensor([m[1] for m in members], device=runtime.device)
                    positions = torch.tensor(
                        [m[2] for m in members], device=runtime.device
                    ).permute(2, 0, 1)
                    cache = None
                    if prefix_mode == "frozen":
                        shared = head
                        cache = restore_frozen_prefix(
                            reference, captured, members, head, runtime.device
                        )
                    if prefix_cache is not None:
                        # Share only independently recomputed, identical token/position
                        # prefixes. Each request gets its own rounded state copy.
                        batch_size = len(members)

                        def expand(value, batch_size=batch_size):
                            if value is None:
                                return None
                            return value.expand(batch_size, *value.shape[1:]).clone()

                        cache = dataclasses.replace(
                            prefix_cache,
                            batch_size=len(members),
                            layers=tuple(
                                dataclasses.replace(
                                    layer,
                                    key=expand(layer.key),
                                    value=expand(layer.value),
                                    convolution=expand(layer.convolution),
                                    recurrent=expand(layer.recurrent),
                                )
                                for layer in prefix_cache.layers
                            ),
                        )
                    with ReferenceExecutionContext():
                        step = reference.forward(
                            input_ids[:, shared:],
                            position_ids=positions[:, :, shared:],
                            cache=cache,
                            epoch_id=f"concurrent-c{concurrency}-length{length}",
                            return_all_logits=True,
                        )
                    for batch, (frame, _, _) in enumerate(members):
                        differences = {}
                        checked = []
                        expected_fields = {}

                        def compare(
                            name,
                            candidate,
                            expected,
                            checked=checked,
                            differences=differences,
                            expected_fields=expected_fields,
                        ):
                            expected = expected.detach().cpu()
                            expected_fields[name] = expected
                            checked.append(name)
                            if (
                                candidate.shape != expected.shape
                                or candidate.dtype != expected.dtype
                            ):
                                raise RuntimeError(
                                    f"HARNESS: {name} ABI differs: "
                                    f"{candidate.shape}/{candidate.dtype} vs "
                                    f"{expected.shape}/{expected.dtype}"
                                )
                            candidate_bytes = candidate.contiguous().view(torch.uint8)
                            expected_bytes = expected.contiguous().view(torch.uint8)
                            if not torch.equal(candidate_bytes, expected_bytes):
                                delta = (candidate.float() - expected.float()).abs()
                                differences[name] = {
                                    "unequal": int((candidate != expected).sum()),
                                    "unequal_bytes": int((candidate_bytes != expected_bytes).sum()),
                                    "elements": candidate.numel(),
                                    "max_abs": float(delta.max()),
                                }

                        for local, layer in enumerate(captured["full_layer_ids"]):
                            cache = step.cache.layers[layer - step.cache.layer_start]
                            compare(
                                f"layer{layer}.key",
                                frame["kv_key"][local],
                                cache.key[batch].transpose(0, 1),
                            )
                            compare(
                                f"layer{layer}.value",
                                frame["kv_value"][local],
                                cache.value[batch].transpose(0, 1),
                            )
                            if not bool(frame["kv_written"][local].all()):
                                raise RuntimeError(
                                    "EXECUTION: committed prefix KV has an unwritten token"
                                )
                        for local, layer in enumerate(captured["gdn_layer_ids"]):
                            cache = step.cache.layers[layer - step.cache.layer_start]
                            compare(
                                f"layer{layer}.conv", frame["conv"][local], cache.convolution[batch]
                            )
                            compare(
                                f"layer{layer}.recurrent",
                                frame["recurrent"][local],
                                cache.recurrent[batch],
                            )
                        if frame["logits"] is not None:
                            span = captured["vocab_span"]
                            count = len(frame["rows"])
                            compare(
                                "review_logits",
                                frame["logits"],
                                step.verification_logits[batch, -count:, span.start : span.stop],
                            )
                        comparison.append(
                            {
                                "request_id": frame["rows"][0].request_id,
                                "checked": checked,
                                "differences": differences,
                            }
                        )
                        oracle_frames.append(
                            {
                                "request_id": frame["rows"][0].request_id,
                                "fields": expected_fields,
                            }
                        )
                    del step
                torch.save(oracle_frames, root / f"c{concurrency}-oracle-rank{rank}.pt")
                reports.append(
                    {
                        "concurrency": concurrency,
                        "logical_m": captured["logical_m"],
                        "physical_m": captured["physical_m"],
                        "padding_rows": captured["padding_rows"],
                        "status": "checked_target_fields_exact"
                        if all(not r["differences"] for r in comparison)
                        else "numerical_unqualified",
                        "scope": (
                            f"one actual review frame; {prefix_mode} prefix; "
                            "independent target math; frozen proposal tokens"
                        ),
                        "comparisons": comparison,
                    }
                )
                output.write_text(json.dumps({"complete": False, "reports": reports}, indent=2))
        output.write_text(
            json.dumps(
                {"complete": True, "reports": reports, "full_speculative_qualification": False},
                indent=2,
            )
        )
    except Exception as error:
        output.write_text(
            json.dumps(
                {
                    "complete": False,
                    "reports": reports,
                    "failure_kind": "execution"
                    if str(error).startswith("EXECUTION:")
                    else "harness",
                    "error": traceback.format_exc(),
                },
                indent=2,
            )
        )
        raise


def run_reference() -> int:
    import argparse
    import tempfile

    import torch.multiprocessing

    from cubic_serve.runtime import performance as p

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--concurrency", default="8,32,33")
    parser.add_argument("--prefix-mode", choices=("frozen", "cold"), default="frozen")
    args = parser.parse_args(sys.argv[2:])
    concurrencies = tuple(map(int, args.concurrency.split(",")))
    if not concurrencies or any(c <= 0 for c in concurrencies):
        parser.error("concurrency must contain positive integers")
    missing = [
        str(args.directory / f"c{c}-rank{rank}.pt")
        for c in concurrencies
        for rank in range(4)
        if not (args.directory / f"c{c}-rank{rank}.pt").exists()
    ]
    if missing:
        parser.error(f"HARNESS: incomplete rank capture coverage: {missing}")
    (args.directory / "reference-source.py").write_bytes(Path(__file__).read_bytes())
    admission = p._admit(
        args.checkpoint, (0, 1, 2, 3), 2048, 256, tp_size=4, pp_size=1, ep_size=args.ep_size
    )
    with tempfile.TemporaryDirectory(prefix="cubic-concurrent-reference-") as rendezvous:
        torch.multiprocessing.spawn(
            reference_worker,
            nprocs=4,
            args=(
                f"file://{rendezvous}/nccl",
                str(args.directory),
                admission,
                concurrencies,
                args.prefix_mode,
            ),
        )
    reports = [
        json.loads((args.directory / f"reference-rank{rank}.json").read_text()) for rank in range(4)
    ]
    qualified = all(
        report.get("complete")
        and len(report["reports"]) == len(concurrencies)
        and all(r["status"] == "checked_target_fields_exact" for r in report["reports"])
        for report in reports
    )
    print(
        json.dumps(
            {"checked_target_fields_qualified": qualified, "full_speculative_qualification": False}
        ),
        flush=True,
    )
    return 0 if qualified else 1


if os.environ.get("CUBIC_CONCURRENT_AUDIT_DIR") and "reference" not in sys.argv:
    install_capture()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "reference":
        raise SystemExit(run_reference())
    else:
        from cubic_serve.cli import main

        main()
