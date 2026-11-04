"""Independent native packed-provider witnesses through its original model math.

The existing provider owns construction and checkpoint semantics. This adapter
records actual native inputs, completes the forward and topology schedule, then
uses the existing projection, normalization, attention and collective contracts.
No upstream output or candidate-only trajectory is a numerical oracle.
"""
from contextlib import ExitStack
from types import MethodType
from unittest.mock import patch

import torch
import torch.distributed as dist

from cubic_serve.executor.runner import ReferenceExecutionContext
from cubic_serve.guardrail.quantization import with_reference_integer_geometry
from cubic_serve.ops.strategies import operator_strategy_scope
from cubic_serve.runtime.text_qualification import prepare_rows, boundary_snapshot

from .linear import Recorder, same
from .nonlinear import Recorder as NonlinearRecorder, NonlinearReplay
from .replay import AttentionReplay, Reference, ordered_sum, reference_model_for_token_domain


class ProviderAuditHarnessError(RuntimeError):
    pass


def _check_event_consumption(causal, nonlin, recorder):
    """Each event belongs to exactly its projection/attention/nonlinear layer."""
    if (len(causal.used) != sum(map(len, recorder.events.values()))
            or causal.embedding_used != len(recorder.embeddings)
            or causal.attention_used != len(recorder.attention_events)
            or any(nonlin.indices[k] != len(nonlin.r.events[k])
                   for k in nonlin.indices if k != 'attention')):
        raise ProviderAuditHarnessError('provider numerical event inventory is incomplete')


def _raise_errors(errors, phase):
    failures = [error for error in errors if error is not None]
    if failures:
        kind = ('harness' if any(k == 'harness' for k, _ in failures) else
                'execution' if any(k == 'execution' for k, _ in failures) else 'numerical')
        exception = AssertionError if kind == 'numerical' else RuntimeError
        raise exception('provider '+phase+' '+kind+' failure: '+str(failures))


@with_reference_integer_geometry
def audit_original(
    source, target_source, admission, groups, *, draft_provider, token_ids,
    feedback_rows, position_axes, independent_feedback_rows=(),
    prefix_store=None, prefix=None, completed_transition=None,
    feedback_recurrence=False, reference_math_memo=None,
):
    """Witness one bounded provider path, retaining separate causal/free state."""
    config, weights, topology = source.config, source.weights, source.topology
    text, device = config.graph.text, source.hidden.device
    if (topology.pp_size != 1 or topology.ep_size != 1 or text.moe is not None
            or source.arena.layout.storage.quantized):
        raise ValueError("packed provider oracle requires its declared dense native-KV block")
    horizon = len(token_ids)
    if (not horizon or len(feedback_rows) != horizon or len(position_axes) != horizon
            or any(type(t) is not int or not 0 <= t < draft_provider.spec.vocabulary_size
                   for t in token_ids)):
        raise ValueError("packed provider corpus differs from its declared token extent")
    for row in (*feedback_rows, *independent_feedback_rows):
        if (row.shape != (1, 1, draft_provider.spec.feedback_width)
                or row.dtype is not torch.bfloat16 or row.device != device
                or not bool(torch.isfinite(row).all())):
            raise ValueError("packed provider feedback differs from its producing boundary")
    if independent_feedback_rows and len(independent_feedback_rows) != horizon:
        raise ValueError("independent provider feedback extent differs")
    if any(len(axes) != 3 or any(type(p) is not int or not 0 <= p < admission.max_sequence_tokens
                                for p in axes) for axes in position_axes):
        raise ValueError("packed provider position axes exceed admitted context")
    if (prefix is not None or prefix_store is not None) and (
            completed_transition is None or prefix_store is None):
        raise ValueError("live provider prefixes require a completed numerical consumer")
    if completed_transition is not None and (
            not callable(completed_transition) or prefix_store is None):
        raise ValueError("completed provider transition requires an admitted prefix store")
    binding = (id(source), id(target_source), id(weights), repr(admission.operator_config))
    restored = None if prefix is None else prefix.restore(prefix_store, binding, device)
    start = 0 if restored is None else restored['position']
    if start + horizon > admission.max_sequence_tokens:
        raise ValueError("provider live prefix exceeds admitted context")
    block = source.arena.layout.block_size_tokens
    pages = (start + horizon + block - 1) // block
    recorder, nonlinear = Recorder(), NonlinearRecorder()
    nonlinear.norm_epsilon = text.rms_norm_epsilon
    nonlinear.events['attention'] = []
    nonlinear.cache_reads = []
    collective_events = []
    stage = draft_provider.build_stage(weights, topology, captured_m=1,
        tp_group=groups['tp'], pp_group=None, table_pages=pages, page_size=block,
        operator_config=admission.operator_config, mrope_tables=source.mrope_tables)
    body = stage.body
    launch_linear, tp_sum = body._launch_linear, body._tp_sum
    normalizer = type(body._final_norm)
    gate_type, swiglu_type, attention_type = (
        type(body._attention_gate), type(body._swiglu), type(body._attention))
    norm_launch, gate_launch, swiglu_launch, attention_launch = (
        normalizer.launch, gate_type.launch, swiglu_type.launch, attention_type.launch)
    original_launch = stage.launch

    def linear(self, primitive, inputs, weight, output):
        result = launch_linear(primitive, inputs, weight, output)
        members = weight if isinstance(weight, tuple) else (weight,)
        outputs = primitive.parts if isinstance(weight, tuple) else (output,)
        for member, value in zip(members, outputs, strict=True):
            actual = self._packed_operators.projection_input(inputs, member)
            encoded = (self._packed_operators.int8_input(actual)
                       if actual.dtype is torch.int8 else None)
            recorder.record_projection(member, actual, value,
                int8_token_scales=None if encoded is None else encoded.scales)
        return result

    def summed(self, value):
        before = value.detach().clone()
        result = tp_sum(value)
        collective_events.append((before, value.detach().clone()))
        return result

    def norm(op, x, w, y):
        before = x.detach().clone(), w.detach().clone()
        result = norm_launch(op, x, w, y)
        nonlinear.events['zero'].append((*before, y.detach().clone()))
        return result

    def gate(op, x, value, y):
        before = x.detach().clone(), value.detach().clone()
        result = gate_launch(op, x, value, y)
        nonlinear.events['attention_gate'].append((*before, y.detach().clone()))
        return result

    def swiglu(op, inputs, output):
        result = swiglu_launch(op, inputs, output)
        recorder.record_activation(*body._gate_up_linear.parts, output)
        return result

    def attention(op, q, k, v, table, lengths, output):
        indices = table[:, :pages].long()
        before = (q.detach().clone(),
            k[indices, 0].reshape(1, -1, *k.shape[-2:]).detach().clone(),
            v[indices, 0].reshape(1, -1, *v.shape[-2:]).detach().clone(),
            lengths.detach().clone())
        result = attention_launch(op, q, k, v, table, lengths, output)
        nonlinear.events['attention'].append((*before, output.detach().clone()))
        nonlinear.cache_reads.append(None)
        return result

    def observed_launch(*, produce_logits=True):
        if not produce_logits:
            # The auxiliary state-only graph must not replace the full forward's
            # static immutable observer destinations or numerical event inventory.
            from .text import preserve_observer_collections
            with preserve_observer_collections((recorder.linear, recorder.embeddings,
                    recorder.latest, recorder.pending, recorder.shared, recorder.int8_scales,
                    collective_events, *nonlinear.events.values(), nonlinear.cache_reads)):
                return original_launch(produce_logits=False)
        recorder.linear.clear(); recorder.embeddings.clear(); recorder.latest.clear()
        recorder.pending.clear(); recorder.shared.clear(); recorder.int8_scales.clear()
        collective_events.clear()
        for events in nonlinear.events.values():
            events.clear()
        nonlinear.cache_reads.clear()
        return original_launch()

    stage.launch = observed_launch
    try:
        with ExitStack() as contexts:
            for context in (patch.object(body, '_launch_linear', MethodType(linear, body)),
                    patch.object(body, '_tp_sum', MethodType(summed, body)),
                    patch.object(normalizer, 'launch', norm),
                    patch.object(gate_type, 'launch', gate),
                    patch.object(swiglu_type, 'launch', swiglu),
                    patch.object(attention_type, 'launch', attention), recorder.observe()):
                contexts.enter_context(context)
            stage.capture()
        prepare_rows(stage, 1)
        stage.arena.reset_page_rows(torch.arange(pages, device=device))
        stage.arena.key.zero_(); stage.arena.value.zero_(); stage.arena.written.zero_()
        causal_state = free_state = previous_feedback = free_feedback = None
        if restored is not None:
            native = restored['native']
            for side in ('key', 'value', 'written'):
                value = native[side]
                getattr(stage.arena, side)[:len(value)].copy_(value)
            causal_state, free_state = restored['causal'], restored['independent']
            previous_feedback, free_feedback = restored['feedback'], restored['independent_feedback']
            del restored, native
        nonlinear.attention_meta = [(body._attention.meta.scale, body._attention._grid[-1],
                                     body._attention.reduction_abi['block_n'])]
        names = {id(weight): name for name, weight in body._weights.items()}
        base_model = target_source.body.model
        fc = body._weights['mtp.fc.weight']
        row_weights = body._packed_operators.row_weights

        class Original(Reference):
            def linear(self, x, weight):
                result = super().linear(x, weight)
                if weight is fc:
                    full = torch.zeros((*result.shape[:-1], text.hidden_size),
                                       dtype=result.dtype, device=result.device)
                    full[..., body._fc_span.start:body._fc_span.stop] = result
                    return ordered_sum(full, groups['tp'])
                return result

        class Causal(AttentionReplay):
            def embedding(self, ids, weight, *, dtype):
                span = body.embedding_span
                owned = (ids >= span.start) & (ids < span.stop)
                local = torch.where(owned, ids - span.start, 0)
                value = super().embedding(local, weight, dtype=dtype)
                before, after = collective_events[0]
                if not same(value * owned[..., None], before):
                    raise AssertionError('provider embedding mask producer differs')
                return after.clone().reshape_as(value)

            def linear(self, x, weight):
                value = super().linear(x, weight)
                index = (1 if weight is fc else 2 if names[id(weight)].endswith(
                    '.self_attn.o_proj.weight') else 3 if names[id(weight)].endswith(
                    '.mlp.down_proj.weight') else None)
                if index is None:
                    return value
                before, after = collective_events[index]
                if weight is fc:
                    wanted = torch.zeros_like(before)
                    wanted[:, body._fc_span.start:body._fc_span.stop] = value.reshape(1, -1)
                else:
                    wanted = value.reshape_as(before)
                if not same(wanted, before):
                    raise AssertionError('provider TP producer/scatter ownership differs')
                return after.clone().reshape(*x.shape[:-1], after.shape[-1])

        reports = []
        for ordinal, token in enumerate(token_ids):
            position = start + ordinal
            b = stage.buffers
            stage.fill_token_ids(torch.tensor([token], device=device, dtype=torch.int64))
            b.position_ids.copy_(torch.tensor(position_axes[ordinal], device=device)[:, None])
            b.sequence_lengths.fill_(position + 1); b.token_offsets.fill_(position % block)
            b.page_source.fill_(position // block); b.page_destination.fill_(position // block)
            b.page_table[0].copy_(torch.arange(pages, device=device))
            feedback = previous_feedback if feedback_recurrence and ordinal else feedback_rows[ordinal]
            independent_input = (free_feedback if feedback_recurrence and ordinal else
                independent_feedback_rows[ordinal] if independent_feedback_rows else feedback)
            b.target_hidden.copy_(feedback.reshape_as(b.target_hidden))
            stage.feature_rows.zero_(); stage.embedding_mask.copy_(body._embedding_mask)
            before_native = tuple(t.detach().clone() for t in (
                stage.arena.key, stage.arena.value, stage.arena.written))
            stage.replay(); stage.synchronize()
            schedule = (tuple((names[id(w)], tuple(x.shape), tuple(y.shape), str(x.dtype))
                              for w, x, y in recorder.linear),
                        tuple((kind, len(events)) for kind, events in nonlinear.events.items()),
                        tuple((tuple(x.shape), tuple(y.shape)) for x, y in collective_events))
            schedules = [None] * dist.get_world_size(groups['tp'])
            dist.all_gather_object(schedules, schedule, group=groups['tp'])
            if any(item != schedule for item in schedules):
                raise RuntimeError('provider completed-forward rank event schedules differ')
            if len(collective_events) != 4:
                raise RuntimeError('provider dense collective inventory differs')
            completions = [ordered_sum(before, groups['tp']) for before, _ in collective_events]
            error = None
            try:
                if any(not same(expected, after) for expected, (_, after)
                       in zip(completions, collective_events, strict=True)):
                    raise AssertionError('provider stored TP completion differs')
                wanted_written = torch.zeros_like(stage.arena.written)
                wanted_written[:position // block, 0].fill_(True)
                wanted_written[position // block, 0, :position % block + 1] = True
                if not same(wanted_written, stage.arena.written):
                    raise AssertionError('provider native KV write-set differs')
                allowed = torch.zeros_like(stage.arena.written)
                allowed[position // block, 0, position % block] = True
                for current, old in zip((stage.arena.key, stage.arena.value), before_native[:2], strict=True):
                    mask = ~allowed[..., None, None].expand_as(current)
                    if not same(current[mask], old[mask]):
                        raise AssertionError('provider changed protected cache or prefix bytes')
            except Exception as exc:
                error = ('harness' if isinstance(exc, ProviderAuditHarnessError) else
                         'numerical' if isinstance(exc, AssertionError) else 'execution', str(exc))
            errors = [None] * dist.get_world_size(groups['tp'])
            dist.all_gather_object(errors, error, group=groups['tp'])
            _raise_errors(errors, 'completed collective/state proof')
            recorder.freeze(); nonlinear.freeze()
            recorder.attention_events = nonlinear.events['attention']
            recorder.attention_meta = nonlinear.attention_meta
            recorder.cache_reads = nonlinear.cache_reads
            causal, nonlin = Causal(recorder, names), NonlinearReplay(nonlinear)
            axes = torch.tensor(position_axes[ordinal], device=device).reshape(3, 1, 1)
            ids = torch.tensor([[token]], device=device, dtype=torch.int64)
            model = reference_model_for_token_domain(
                base_model, draft_provider.spec.vocabulary_size, operators=causal)
            error = None
            try:
                with operator_strategy_scope(admission.operator_config, device='', model=''), \
                        ReferenceExecutionContext(), torch.inference_mode(), nonlin.install():
                    witnessed = model.mtp_forward(ids, feedback, position_ids=axes, cache=causal_state)
                _check_event_consumption(causal, nonlin, recorder)
                if not same(stage.logits, witnessed.logits) or not same(stage.draft_feedback,
                                                                                witnessed.hidden_states):
                    raise AssertionError('provider causal output/feedback differs')
                causal_state = witnessed.cache
                for side in ('key', 'value'):
                    native = getattr(stage.arena, side)[:pages, 0].flatten(0, 1)[:position + 1]
                    expected = getattr(causal_state.layer, side)[0].transpose(0, 1)
                    if not same(native, expected):
                        raise AssertionError('provider causal durable '+side+' differs')
            except Exception as exc:
                error = ('harness' if isinstance(exc, ProviderAuditHarnessError) else
                         'numerical' if isinstance(exc, AssertionError) else 'execution', str(exc))
            errors = [None] * dist.get_world_size(groups['tp'])
            dist.all_gather_object(errors, error, group=groups['tp'])
            _raise_errors(errors, 'original-weight causal proof')
            original = Original(row_weights, body.embedding_span, groups['tp'])
            free_model = reference_model_for_token_domain(
                base_model, draft_provider.spec.vocabulary_size, operators=original)
            with operator_strategy_scope(admission.operator_config, device='', model=''), \
                    ReferenceExecutionContext(), torch.inference_mode():
                independent = free_model.mtp_forward(ids, independent_input,
                                                     position_ids=axes, cache=free_state)
            free_state, free_feedback = independent.cache, independent.hidden_states
            previous_feedback = stage.draft_feedback.detach().clone().reshape(1, 1, -1)
            from .text import _TrajectoryDifferences
            delta = _TrajectoryDifferences()
            diagnostics = dict(route_branch_schedule_exact=True,
                logits=delta(stage.logits, independent.logits),
                feedback=delta(previous_feedback, free_feedback))
            delta.flush()
            boundaries = boundary_snapshot(stage, 1)
            from cubic_serve.guardrail.compare import tensor_digest
            report = dict(candidate_boundary_continuation_state_and_logits_exact=True,
                independent_trajectory_diagnostic=diagnostics,
                linear_proofs=causal.reports, activation_proofs=causal.activation_proofs,
                attention_proofs=causal.attention_proofs, nonlinear_proofs=nonlin.proved,
                collective_proofs=4, position=position,
                boundary_digests={n:tensor_digest(v) for n,v in boundaries.items()})
            reports.append(report)
            if completed_transition is not None:
                image = prefix_store.freeze(binding, dict(position=position + 1,
                    native={side:getattr(stage.arena, side)[:pages]
                            for side in ('key','value','written')},
                    causal=causal_state, independent=free_state,
                    feedback=previous_feedback, independent_feedback=free_feedback))
                completed_transition(dict(position=position,token=token,prefix=image,
                    boundaries=boundaries,report=report,
                    independent_logits=independent.logits.detach().to('cpu',copy=True),
                    independent_feedback=free_feedback.detach().to('cpu',copy=True)))
        return dict(independent_trajectory=True,steps=reports,
                    cross_step_error_propagation_qualified=len(reports)>1)
    finally:
        stage.close()
        del stage.launch
