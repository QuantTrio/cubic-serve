"""Compose independently witnessed model transitions using the shared draft rule.

This is an offline oracle adapter, never an execution mode or a serving fallback.
The adapter receives completed causal/free transition witnesses. Tokens, K and RNG
are controlled by the existing independent pipelined transaction reference.
"""
import math
import weakref

import torch

from cubic_serve.executor.draft import DraftStep


def philox_uniform(seed: int, ordinal: int) -> float:
    """Independent scalar Philox4x32-10 with the declared CUDA RNG coordinates."""
    mask = 0xFFFFFFFF
    counter = [ordinal & mask, 0, 0, 0]
    keys = [seed & mask, (seed >> 32) & mask]
    for _ in range(10):
        a, b = 0xD2511F53 * counter[0], 0xCD9E8D57 * counter[2]
        counter = [(b >> 32) ^ counter[1] ^ keys[0], b & mask,
                   (a >> 32) ^ counter[3] ^ keys[1], a & mask]
        counter = [v & mask for v in counter]
        keys = [(keys[0] + 0x9E3779B9) & mask, (keys[1] + 0xBB67AE85) & mask]
    signed = counter[0] if counter[0] < 0x80000000 else counter[0] - 0x100000000
    signed = -signed - 1 if signed < 0 else signed
    return float(torch.tensor(signed, dtype=torch.float32)
                 * torch.tensor(4.6566127342e-10, dtype=torch.float32))


class SamplingReference:
    """CPU sampling math for an explicitly fixed carried-round RNG trace.

    This is an independent decision witness, not an engine sampler. Its double
    probability calculation does not certify all device rounding boundaries;
    callers must retain actual producer witnesses and report unmatched traces.
    """
    def __init__(self, sampling, *, position=0):
        from cubic_serve.generation.sampler import SamplingParameters
        if not isinstance(sampling, SamplingParameters) or type(position) is not int or position < 0:
            raise ValueError("sampling reference requires fixed parameters and position")
        self.sampling, self.position = sampling, position
        self.temperature = float(torch.tensor(sampling.temperature, dtype=torch.float32))
        self.top_p = float(torch.tensor(sampling.top_p, dtype=torch.float32))
        self.events = []

    def distribution(self, logits):
        values = logits.detach().cpu().flatten()
        if not values.numel() or not bool(torch.isfinite(values).all()):
            raise ValueError("sampling reference requires finite vocabulary logits")
        if self.temperature == 0.:
            return {int(values.argmax()): 1.}
        ordered = sorted(enumerate(values.tolist()), key=lambda item: (-item[1], item[0]))
        ordered = ordered[:self.sampling.top_k]
        scaled = [score / self.temperature for _, score in ordered]
        weights = [math.exp(score - scaled[0]) for score in scaled]
        total, cumulative, kept = sum(weights), 0., {}
        for (token, _), weight in zip(ordered, weights, strict=True):
            probability = weight / total
            if cumulative <= self.top_p:
                kept[token] = probability
            cumulative += probability
        mass = sum(kept.values())
        return {token: value / mass for token, value in kept.items()}

    @staticmethod
    def categorical(probabilities, uniform):
        cumulative = 0.
        for token, probability in probabilities.items():
            cumulative += probability
            if cumulative > uniform:
                return token
        raise ValueError("sampling reference uniform lies outside probability mass")

    def propose(self, logits, depth):
        ordinal = self.position + 1 + depth
        seed = self.sampling.seed
        uniform = (0. if self.temperature == 0. else
                   philox_uniform(seed ^ (seed >> 32) ^ (ordinal >> 32), ordinal))
        distribution = self.distribution(logits)
        token = self.categorical(distribution, uniform)
        self.events.append(dict(kind="proposal", position=self.position, ordinal=ordinal,
                                seed=seed, rng_used=self.temperature != 0., uniform=uniform,
                                distribution=distribution, token=token))
        return token

    def verify(self, proposals, proposal_logits, target_logits):
        from cubic_serve.generation.tp_sampling import SpeculativeSamplingDecision
        if not proposals or len(proposal_logits) != len(proposals) or len(target_logits) != len(proposals) + 1:
            raise ValueError("sampling reference requires one complete fixed proposal/review trace")
        target = tuple(self.distribution(value) for value in target_logits)
        accepted = 0
        key = (self.sampling.seed ^ (self.sampling.seed >> 32) ^ 0x9E3779B9) & 0xFFFFFFFF
        for depth, (token, logits) in enumerate(zip(proposals, proposal_logits, strict=True)):
            q, p = self.distribution(logits), target[depth]
            if token not in q or q[token] <= 0:
                raise ValueError("fixed proposal has no producing q probability")
            ordinal = self.position + 1 + depth
            u = 0. if self.temperature == 0. else philox_uniform(key, ordinal * 2)
            v = 0. if self.temperature == 0. else philox_uniform(key ^ 0x85EBCA6B, ordinal * 2 + 1)
            ratio = min(1., p.get(token, 0.) / q[token])
            retain = token == next(iter(p)) if self.temperature == 0. else u < ratio
            self.events.append(dict(kind="verification", position=self.position, ordinal=ordinal,
                seed=self.sampling.seed, proposal=token, q=q, p=p,
                rng_used=self.temperature != 0., acceptance_uniform=u,
                sample_uniform=v, ratio=ratio, accepted=retain))
            if not retain:
                residual = {t: max(0., value - q.get(t, 0.)) for t, value in p.items()}
                mass = sum(residual.values())
                if mass <= 0:
                    raise ValueError("rejected proposal has no residual probability mass")
                bonus = self.categorical({t: value / mass for t, value in residual.items()}, v)
                break
            accepted += 1
        else:
            bonus = self.categorical(target[-1], v)
        self.position += accepted + 1
        return SpeculativeSamplingDecision(accepted, bonus)


class SpeculativeReference:
    """Keep candidate and independent feedback provenance separate across forks.

    ``target_transition`` and ``provider_transition`` return completed witnesses
    with owned native ``prefix``, CPU ``boundaries`` and independent feedback.
    Each transition must finish its independent model/state proof before return.
    """
    def __init__(self, spec, *, target_transition, provider_transition, device):
        self.spec, self.device = spec, torch.device(device)
        self._target, self._provider = target_transition, provider_transition
        self._feedback = {}
        self.reports = []

    def external_feedback(self, value):
        """Declare the identical external initial boundary for both trajectories."""
        if (value.shape != (1, 1, self.spec.feedback_width)
                or value.dtype is not torch.bfloat16 or value.device != self.device
                or not bool(torch.isfinite(value).all())):
            raise ValueError("external speculative feedback differs from its contract")
        with torch.inference_mode(False), torch.no_grad():
            actual, independent = value.clone(), value.clone()
        self._remember(actual, independent)
        return actual

    def _remember(self, actual, independent):
        identity = id(actual)
        def retire(reference):
            entry = self._feedback.get(identity)
            if entry is not None and entry[0] is reference:
                self._feedback.pop(identity)
        self._feedback[identity] = (weakref.ref(actual, retire), actual._version,
                                    independent, independent._version)

    def _step(self, transition, tokens, state, axes, feedback=None):
        if (tokens.dtype is not torch.int64 or tokens.shape != (1, 1)
                or tokens.device != self.device or axes is None
                or axes.dtype is not torch.int64 or axes.shape != (3, 1, 1)
                or axes.device != self.device):
            raise ValueError("speculative numerical step differs from its fixed input trace")
        independent_feedback = None
        if feedback is not None:
            entry = self._feedback.get(id(feedback))
            if entry is None or entry[0]() is not feedback:
                raise ValueError("speculative feedback has no witnessed producer")
            if feedback._version != entry[1] or entry[2]._version != entry[3]:
                raise ValueError("immutable speculative feedback was modified")
            independent_feedback = entry[2]
        witness = transition(tokens, state, axes, feedback, independent_feedback)
        report = witness['report']
        if (not report['candidate_boundary_continuation_state_and_logits_exact']
                or not report['independent_trajectory_diagnostic']['route_branch_schedule_exact']
                or witness['token'] != int(tokens.item())):
            raise AssertionError("incomplete independent speculative numerical transition")
        with torch.inference_mode(False), torch.no_grad():
            actual = witness['boundaries']['draft_feedback'].to(
                self.device, copy=True).reshape(1, 1, -1)
            independent = witness['independent_feedback'].to(
                self.device, copy=True).reshape(1, 1, -1)
        if actual.shape != independent.shape or actual.shape[-1] != self.spec.feedback_width:
            raise ValueError("speculative feedback producer width differs")
        self._remember(actual, independent)
        self.reports.append(dict(position=witness['position'], token=witness['token'],
                                 role='target' if transition is self._target else 'provider',
                                 report=report))
        # The transition adapter assembles vocabulary shards after all model
        # observers and numerical proofs have completed.
        return DraftStep(witness['logits'], actual, witness['prefix'])

    def target(self, tokens, state, position_ids):
        return self._step(self._target, tokens, state, position_ids)

    def forward(self, tokens, feedback, state=None, *, position_ids=None):
        return self._step(self._provider, tokens, state, position_ids, feedback)


def _assemble_logits(parts, vocabulary_size):
    """Close exact TP vocabulary ownership before any sampling decision."""
    offset, values = 0, []
    for first, last, value in sorted(parts, key=lambda p: p[0]):
        if (first != offset or last <= first or not isinstance(value, torch.Tensor)
                or value.shape != (1, last - first) or not value.is_floating_point()
                or (values and (value.dtype != values[0].dtype or value.device != values[0].device))):
            raise RuntimeError("Numerical vocabulary ownership or ABI differs")
        offset = last
        values.append(value)
    if not values or offset != vocabulary_size:
        raise RuntimeError("Incomplete numerical vocabulary")
    return torch.cat(values, -1).reshape(1, 1, -1)


def _native_prefix_state(value, arena):
    """Normalize the existing typed and positional prefix witness schemas."""
    if "native" in value:
        native = value["native"]
        if (not {"key", "value", "written"} <= native.keys()
                or set(native.get("states", {})) != set(arena.states)):
            raise RuntimeError("Numerical prefix state inventory differs")
        return {**native, "states": native.get("states", {})}
    storage = value["storage"]
    if len(storage) != 3 + len(arena.page_fields) + len(arena.states):
        raise RuntimeError("Numerical prefix storage inventory differs")
    return dict(key=storage[0], value=storage[1], written=storage[2],
                states={name: storage[3 + len(arena.page_fields) + i][0]
                        for i, name in enumerate(arena.states)})


def _pooled_packed_target(source, pages):
    """Isolate one numerical owner while retaining physical-M math and pitches."""
    from dataclasses import replace
    from .text import _audit_cache_arena

    arena = _audit_cache_arena(source, pages + 1, scratch_capacity=source.captured_m)
    if arena is None:
        return None
    resource = replace(source.resource,
        graph_instance_id="audit:" + source.resource.graph_instance_id,
        stream_lane_id="audit:" + source.resource.stream_lane_id,
        buffer_set_id="audit:" + source.resource.buffer_set_id,
        workspace_arena_id="audit:" + source.resource.workspace_arena_id)
    # Reuse the stage's existing reconstruction contract, including its model
    # constructor, format operators and collectives; replace only owned storage.
    options = dict(source._overlap_factory)
    options.update(captured_m=source.captured_m, table_pages=pages,
                   arena=arena, state_slots=1, resource=resource)
    stage = type(source)(**options)
    try:
        stage.capture()
    except BaseException:
        stage.close()
        raise
    return stage


def audit_carried_draft_numerics(
    round_key,
    round_,
    rounds,
    providers,
    targets,
    hidden,
    admission,
    *,
    draft_provider,
    groups,
    rank,
    sampler_rank,
    sampling=None,
    reference_math_memo=None,
    completed_round=None,
):
    """Complete one declared C1/K/policy three-round composition, without admission.

    Native p/q snapshots belong to unchanged sealed ordinary graphs. Every
    transition independently witnesses original-weight math and rounded state.
    The existing resident/full-state/write-set integration fixture also runs.
    This result covers only the named K, policy and topology. Other K values,
    permutations and portfolio acceptance remain separate.
    """

    import copy
    import gc
    import time
    import torch.distributed as dist
    from cubic_serve.runtime import quantized_qualification as q
    from cubic_serve.guardrail.draft_control import DraftRoundLogits
    from cubic_serve.guardrail.quantized_graph.text import audit_original as typed_original
    from cubic_serve.guardrail.quantized_graph import audit_original as packed_original
    from cubic_serve.guardrail.quantized_graph.packed_mtp import audit_original as packed_provider_original
    from cubic_serve.guardrail.quantized_graph.text_cross_m import (
        _attention_arithmetic_plan,
        _attention_arithmetic_scope,
    )
    from cubic_serve.guardrail.quantized_graph.continuation import AuditPrefixStore
    from cubic_serve.guardrail.quantized_graph.capture_session import AuditCaptures
    from cubic_serve.guardrail.quantization import ReferenceMathMemo
    from cubic_serve.quantization.common.small_rows_cost import row_arithmetic_plan
    from cubic_serve.executor.draft import reference_pending_draft, reference_pipelined_draft_round
    from cubic_serve.generation.sampler import GREEDY_SAMPLING

    depth = round_.depth
    cache = {}
    refs = {}
    native = []
    prefixes = weakref.WeakSet()
    captures = AuditCaptures(1 << 30, reserve_bytes=512 << 20)
    store = AuditPrefixStore(512 << 20)
    owns_memo = reference_math_memo is None
    memo = ReferenceMathMemo() if owns_memo else reference_math_memo
    if completed_round is not None and not callable(completed_round):
        raise ValueError("completed round consumer must be callable")
    start = time.monotonic()
    if round_.C != 1 or round_key not in rounds or rounds[round_key] is not round_:
        raise ValueError("Numerical composition requires its explicit bound C1 round")
    if round_key != (round_.M, depth):
        raise ValueError("Numerical composition round geometry differs from its portfolio key")
    provider = draft_provider
    if sampling is not None and sampling.temperature <= 0.:
        raise ValueError("explicit stochastic audit policy requires positive temperature")
    if provider is None or not 1 <= depth <= provider.spec.max_depth:
        raise ValueError("Numerical composition requires its explicit checkpoint provider")
    typed = q._has_typed_boundary(round_.target.body)
    # Typed observations select the actual arithmetic plan explicitly. The
    # packed reference observes the actual physical bucket directly instead.
    source_target = targets[1].body if typed else round_.target.body
    source_provider = round_.head.body
    if source_provider is not providers[1].body:
        raise ValueError("Numerical proposal owner differs from its explicit C1 provider")

    def freeze(t):
        return t.detach().to("cpu", copy=True)

    trace_pages = (4 * round_.stride + round_.target._block_size - 1) // round_.target._block_size
    owner_ids, _ = q._draft_fixture_state_map(round_, 1, trace_pages)
    owner = owner_ids[0]
    target_arena, provider_arena = round_.target.body.arena, round_.align.body.arena
    snapshot_tensors = [
        hidden[owner],
        *(t[owner] for t in target_arena.states.values()),
        *(t[owner] for t in provider_arena.states.values()),
        *(t[owner] for t in round_.pool.backup.values()),
        *(
            t[:trace_pages]
            for arena in (target_arena, provider_arena)
            for t in (arena.key, arena.value, arena.written)
        ),
    ]
    snapshot_bytes = 3 * sum(t.numel() * t.element_size() for t in snapshot_tensors)
    if snapshot_bytes > 512 << 20:
        raise MemoryError("Speculative numerical host snapshots exceed their byte budget")

    class Recorder(DraftRoundLogits):
        def after_forward(self, kind):
            super().after_forward(kind)
            if kind == "resident":
                return
            if len(native) < 3:
                ta, pa = round_.target.body.arena, round_.align.body.arena
                native.append(
                    dict(
                        metadata=copy.deepcopy(round_.pool.metadata[owner]),
                        decision=tuple(round_.host_result[0].tolist()),
                        target_states={n: freeze(v[owner]) for n, v in ta.states.items()},
                        provider_states={n: freeze(v[owner]) for n, v in pa.states.items()},
                        backup={n: freeze(v[owner]) for n, v in round_.pool.backup.items()},
                        feedback=freeze(hidden[owner]),
                        target_key=freeze(ta.key[:trace_pages]),
                        target_value=freeze(ta.value[:trace_pages]),
                        provider_key=freeze(pa.key[:trace_pages]),
                        provider_value=freeze(pa.value[:trace_pages]),
                        target_written=freeze(ta.written[:trace_pages]),
                        provider_written=freeze(pa.written[:trace_pages]),
                        p=self.events[-1]["p"].clone(),
                        q=[self.events[-1]["q" + str(d)].clone() for d in range(depth)],
                    )
                )

    trace = Recorder(round_, forward_capacity=4, candidates=sampling is not None)
    try:
        unit = q._qualify_draft_window_case(
            round_key[0],
            round_,
            round_._resident_window if sampling is None else round_._candidate_window,
            providers,
            targets,
            hidden,
            admission,
            (0,),
            rank=rank,
            sampler_rank=sampler_rank,
            sampling=sampling,
            original_cache=cache,
            reference_cache=refs,
            input_observer=trace,
        )
    finally:
        trace.close()
        refs.clear()
        image = cache.pop("receipt", None)
        if image is not None:
            try:
                image.restore()
            finally:
                image.close()
        gc.collect()
        torch.cuda.empty_cache()
    if len(native) != 3:
        raise RuntimeError("Missing complete native three-round trace")
    native_seconds = time.monotonic() - start
    records = []
    packed_target = None
    # q logits are produced by the proposal head. Alignment may bind a
    # different physical-M reduction and does not own these logits. Its final
    # state remains independently compared with the reference composition.
    plans = {} if not typed else {
        role: row_arithmetic_plan(source.operators, source.captured_m)
        for role, source in [("target", round_.target.body), ("provider", source_provider)]
    }
    attention = {} if not typed else {
        role: _attention_arithmetic_plan(source)
        for role, source in [("target", round_.target.body), ("provider", source_provider)]
    }

    def adapter(role, source):
        def transition(tokens, state, axes, feedback, independent_feedback):
            got = []
            opts = dict(
                token_ids=(int(tokens.item()),),
                position_axes=(tuple(int(x) for x in axes.flatten().tolist()),),
                prefix_store=store,
                prefix=state,
                completed_transition=got.append,
            )
            if role == "provider":
                opts.update(
                    draft_provider=provider,
                    feedback_rows=(feedback,),
                    independent_feedback_rows=(independent_feedback,),
                )
            if typed:
                opts.update(diagnose_independent_trajectory=True, reference_math_memo=memo,
                            row_plan=plans[role], capture_pool=captures)
                if role == "provider":
                    opts["column_completion_names"] = provider.reference_columns
                with _attention_arithmetic_scope(attention[role]):
                    typed_original(source, admission, groups, **opts)
            elif role == "provider":
                packed_provider_original(source, source_target, admission, groups,
                                         reference_math_memo=memo, **opts)
            else:
                nonlocal packed_target
                if packed_target is None:
                    packed_target = _pooled_packed_target(
                        source, math.ceil(max(16, 4 * round_.stride)
                                          / source.arena.layout.block_size_tokens))
                packed_original(source if packed_target is None else packed_target,
                                admission, groups, prefix_length=0, logical_m=1,
                                steps=1, prefix_capacity_tokens=max(16, 4 * round_.stride), **opts)
            if len(got) != 1:
                raise RuntimeError("Missing numerical transition")
            w = got[0]
            prefixes.add(w["prefix"])
            span = getattr(source, "vocab_span", None) or source._vocab_span
            part = (span.start, span.stop, w["boundaries"]["logits"])
            parts = [part]
            if source.topology.tp_size > 1:
                group = groups.get("tp")
                if group is None:
                    raise ValueError("Numerical vocabulary requires its explicit TP group")
                parts = [None] * dist.get_world_size(group)
                dist.all_gather_object(parts, part, group=group)
            w["logits"] = _assemble_logits(parts, provider.spec.vocabulary_size)
            return w

        return transition

    ref = SpeculativeReference(
        provider.spec,
        target_transition=adapter("target", source_target),
        provider_transition=adapter("provider", source_provider),
        device=source_target.hidden.device,
    )

    sampling_reference = SamplingReference(GREEDY_SAMPLING if sampling is None else sampling)
    propose, verify = sampling_reference.propose, sampling_reference.verify

    def exact(actual, expected, label):
        if (
            actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or not torch.equal(
                actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
            )
        ):
            details = dict(actual_shape=tuple(actual.shape), expected_shape=tuple(expected.shape),
                           actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype))
            if actual.shape == expected.shape and actual.dtype == expected.dtype:
                details["unequal_bytes"] = int((actual.contiguous().view(torch.uint8)
                                               != expected.contiguous().view(torch.uint8)).sum())
                details["max_abs"] = float((actual.double() - expected.double()).abs().max())
                if actual.ndim == 2 and actual.is_floating_point():
                    details["actual_local_top1"] = actual.argmax(-1).tolist()
                    details["reference_local_top1"] = expected.argmax(-1).tolist()
            raise RuntimeError(
                "Independent speculative boundary differs:"
                + label + ": " + repr(details) + "; no floating-point verdict"
            )

    def native_prefix(prefix, source):
        return _native_prefix_state(prefix.restore(store, prefix._binding, "cpu"), source.arena)

    root_token = torch.tensor(
        [[101 % provider.spec.vocabulary_size]], dtype=torch.int64, device=ref.device
    )
    feedback = ref.external_feedback(
        torch.zeros(1, 1, provider.spec.feedback_width, dtype=torch.bfloat16, device=ref.device)
    )
    result_report = None
    try:
        pending = reference_pending_draft(
            provider=ref,
            root_token=root_token,
            feedback=feedback,
            draft_state=None,
            position=0,
            depth=depth,
            propose=propose,
        )
        target_state = None
        for index, actual in enumerate(native):
            before = time.monotonic()
            result = reference_pipelined_draft_round(
                provider=ref,
                target=ref.target,
                pending=pending,
                target_state=target_state,
                device=ref.device,
                next_depth=depth,
                propose=propose,
                verify=verify,
            )
            check_error = None
            try:
                span = getattr(source_target, "vocab_span", None) or source_target._vocab_span
                review = torch.cat([step.logits.reshape(1, -1)[:, span.start:span.stop]
                                    for step in result.target_steps], 0)
                # Establish the producing p values before judging a token decision.
                exact(actual["p"][:round_.stride], review, "target review logits before decision")
                if (
                    result.decision.accepted_draft_tokens != actual["decision"][0]
                    or result.decision.next_root_token_id != actual["decision"][1]
                ):
                    raise RuntimeError("Unmatched actual-logit verification trace: " + repr(dict(
                        round=index, native=actual["decision"],
                        reference=(result.decision.accepted_draft_tokens,
                                   result.decision.next_root_token_id))) + "; no floating-point verdict")
                if (
                    result.pending.position,
                    result.pending.root_token_id,
                    result.pending.proposal_tokens,
                ) != (actual["metadata"][1], actual["metadata"][2], actual["metadata"][6]):
                    raise AssertionError("Independent pending/K/position trace differs")
                expected_target = native_prefix(result.target_state, source_target)
                expected_provider = native_prefix(result.pending.root_step.state, source_provider)
                expected_backup = native_prefix(result.draft_state, source_provider)
                for label, expected, values in [
                    ("target", expected_target, actual["target_states"]),
                    ("provider", expected_provider, actual["provider_states"]),
                    ("backup", expected_backup, actual["backup"]),
                ]:
                    for name, value in values.items():
                        exact(value, expected["states"][name], label + ":" + name)
                exact(actual["feedback"], result.feedback.detach().cpu().reshape(-1), "target feedback")
                for d, step in enumerate(result.pending.proposal_steps):
                    exact(
                        actual["q"][d],
                        step.logits.reshape(1, -1)[:, span.start : span.stop],
                        "provider proposal logits depth" + str(d),
                    )
                for label, expected, length in [
                    ("target", expected_target, result.pending.position),
                    ("provider", expected_provider, result.pending.position + 1),
                ]:
                    for side in ("key", "value", "written"):
                        # Prefix rows include the page/layer dimensions; compare semantic token
                        # order, ignoring only suffix writes separately covered by the full unit.
                        lhs = actual[label + "_" + side].transpose(0, 1).flatten(1, 2)[:, :length]
                        rhs = expected[side].transpose(0, 1).flatten(1, 2)[:, :length]
                        exact(lhs, rhs, label + ":" + side)
            except Exception as exc:
                check_error = (type(exc).__name__, str(exc))
            check_errors = [check_error]
            if dist.is_initialized():
                check_errors = [None] * dist.get_world_size()
                dist.all_gather_object(check_errors, check_error)
            if any(check_errors):
                raise RuntimeError("Unmatched speculative composition trace; no floating-point verdict: "
                                   + repr(check_errors))
            record = dict(
                round=index,
                seconds=time.monotonic() - before,
                accepted=result.decision.accepted_draft_tokens,
                root=result.pending.root_token_id,
                position=result.pending.position,
                proposal=result.pending.proposal_tokens,
                prefix_live_bytes=store.bytes,
                prefix_peak_bytes=store.peak_bytes,
                model_steps=len(ref.reports),
                capture_hits=captures.hits,
                capture_misses=captures.misses,
                capture_bytes=captures.bytes,
            )
            records.append(record)
            target_state = result.target_state
            pending = result.pending
            if completed_round is not None:
                completed_round(index + 1, 3)
            # Carry only the committed target and pending provider prefixes.
            # Retaining every old review branch doubles the next round peak.
            del expected_target, expected_provider, expected_backup, result
        result_report = dict(
            scope="independent actual-logit C1 carried-round numerical/state composition",
            sampling_policy="greedy" if sampling is None else "stochastic",
            sampling_parameters=None if sampling is None else dict(
                temperature=sampling.temperature, top_p=sampling.top_p,
                top_k=sampling.top_k, seed=sampling.seed),
            sampling_trace=sampling_reference.events,
            depth=depth,
            physical_owner=owner,
            native_seconds=native_seconds,
            total_seconds=time.monotonic() - start,
            rounds=records,
            transitions=ref.reports,
            complete_portfolio_qualification=False,
            composition_identity=unit,
            host_snapshot_bytes=snapshot_bytes,
        )
        return result_report
    finally:
        for prefix in tuple(prefixes):
            prefix.close()
        if packed_target is not None:
            packed_target.close()
        captures.clear()
        if owns_memo:
            memo.clear()
        gc.collect()
        torch.cuda.empty_cache()
        if result_report is not None:
            result_report["total_seconds"] = time.monotonic() - start


def draft_numerical_case_plan(rounds, admission, draft_provider):
    """Declare C1 arithmetic/composition witnesses beside native multi-M proofs."""
    from cubic_serve.generation.sampler import SamplingParameters

    maximum = (admission.dynamic_max_speculative_tokens if admission.dynamic_speculation
               else admission.num_speculative_tokens)
    initial = admission.num_speculative_tokens
    if (draft_provider is None or not 1 <= initial <= maximum <= draft_provider.spec.max_depth):
        raise ValueError("independent draft audit requires its declared provider/depths")
    expected = set(range(1, maximum + 1)) if admission.dynamic_speculation else {initial}
    selected = [(key, round_) for key, round_ in sorted(rounds.items()) if round_.C == 1]
    by_depth = {round_.depth: (key, round_) for key, round_ in selected}
    if len(selected) != len(by_depth) or set(by_depth) != expected:
        raise ValueError("independent draft audit lacks the complete bound C1 depth portfolio")
    cases = []
    for depth, (key, round_) in sorted(by_depth.items()):
        if key != (round_.M, depth) or round_.stride != depth + 1:
            raise ValueError("independent draft audit round geometry differs")
        cases.extend((key, round_, policy) for policy in (
            None, SamplingParameters(.6, .95, 20, 0x123456789)))
        if depth == 1:
            cases.append((key, round_, SamplingParameters(.6, .95, 20, 0x12345678b)))
    return tuple(cases)


def audit_draft_portfolio_numerics(
    rounds, providers, targets, hidden, admission, *, draft_provider, groups,
    rank, sampler_rank, cases=None, completed_round=None,
):
    """Run every declared independent unit; never cache a numerical verdict.

    Native full/padded/permuted ownership and target/provider cross-M obligations
    remain separate. Only exact-input pure arithmetic is shared between units.
    """
    from cubic_serve.guardrail.compare import stable_digest
    from cubic_serve.guardrail.quantization import ReferenceMathMemo

    declared = draft_numerical_case_plan(rounds, admission, draft_provider)
    if cases is not None and tuple(cases) != declared:
        raise ValueError("independent draft numerical schedule differs")
    memo, witnesses = ReferenceMathMemo(), []
    try:
        for unit, (key, round_, sampling) in enumerate(declared):
            def progressed(completed, total):
                if total != 3 or not 0 < completed <= total:
                    raise RuntimeError("independent draft completed-round schedule differs")
                if completed_round is not None:
                    completed_round(unit * 3 + completed, len(declared) * 3)
            report = audit_carried_draft_numerics(
                key, round_, rounds, providers, targets, hidden, admission,
                draft_provider=draft_provider, groups=groups, rank=rank,
                sampler_rank=sampler_rank, sampling=sampling,
                reference_math_memo=memo, completed_round=progressed)
            if len(report["rounds"]) != 3 or not report["transitions"]:
                raise RuntimeError("independent draft numerical unit is incomplete")
            for transition in report["transitions"]:
                proof = transition["report"]
                if (not proof["candidate_boundary_continuation_state_and_logits_exact"]
                        or not proof["independent_trajectory_diagnostic"]["route_branch_schedule_exact"]):
                    raise RuntimeError("independent draft numerical transition is incomplete")
            witnesses.append(dict(
                key=key, policy=report["sampling_policy"],
                parameters=report["sampling_parameters"], witness=stable_digest(report),
                model_steps=len(report["transitions"]), seconds=report["total_seconds"],
                accepted=tuple(item["accepted"] for item in report["rounds"])))
        return tuple(witnesses)
    finally:
        memo.clear()
