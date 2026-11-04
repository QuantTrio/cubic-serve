"""Current-step producing-value identity, never a numerical verdict cache.

Use only after completed native forwards and rank schedule agreement. The
caller keeps both observation graphs exclusively borrowed and retires this
view before either graph advances. Values are immutable observer destinations;
no model execution, reference math or CPU read belongs to a forward callback.
Different observations require their own independent oracle, not rejection.
"""
from dataclasses import fields, is_dataclass
import struct
import logging

import torch
import torch.distributed as dist

from cubic_serve.guardrail.quantization import ReferenceMathMemo
from .capture_session import OBSERVATION_FIELDS
from .linear import same


class ObservationUnavailable(RuntimeError):
    """The complete exact-input proof transport is unavailable, not an FP verdict."""


def _schedule(payload):
    # Partition count is recorded and independently bounded at the attention
    # operator. It is execution metadata, not a different mathematical input.
    return tuple(event[:-1] if event[0] == 'attention' else event
                 for event in payload['ordered_events'])


def agree_schedules(payloads, group):
    schedules = tuple(_schedule(payload) for payload in payloads)
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, schedules, group=group)
        if any(value != schedules for value in gathered):
            raise RuntimeError('current observation rank event schedules differ')
    return schedules


def _flatten(value, path, names, scalars, tensors):
    if id(value) in names:
        scalars[path] = ('weight', names[id(value)], id(value))
    elif isinstance(value, torch.Tensor):
        tensors[path] = value
    elif type(value) is float:
        scalars[path] = ('float', struct.pack('>d', value))
    elif value is None or type(value) in (int, bool, str, bytes):
        scalars[path] = (type(value).__name__, value)
    elif isinstance(value, torch.dtype):
        scalars[path] = ('dtype', str(value))
    elif isinstance(value, (tuple, list, set, frozenset)):
        scalars[path] = ('sequence', type(value).__name__, len(value))
        for i, item in enumerate(sorted(value, key=repr) if isinstance(value, (set, frozenset))
                                 else value):
            _flatten(item, path + '/' + str(i), names, scalars, tensors)
    elif isinstance(value, dict):
        keys = tuple(sorted((names.get(key, key) for key in value), key=repr))
        scalars[path] = ('dict', keys)
        for key, item in value.items():
            _flatten(item, path + '/' + str(names.get(key, key)), names, scalars, tensors)
    elif is_dataclass(value):
        scalars[path] = ('dataclass', type(value).__qualname__)
        for field in fields(value):
            _flatten(getattr(value, field.name), path + '/' + field.name, names, scalars, tensors)
    else:
        raise ObservationUnavailable('unmapped producing observation schema: ' + path)


def reference_family(stage, names):
    """The independent free trajectory's rule must match as well as its values."""
    return (type(stage), id(stage.weights), repr(stage.config), repr(stage.operators.config),
            stage.recurrent_state_dtype, repr(stage.arena.layout.storage),
            tuple(sorted(names[key] for key in stage.helper.local_tp_routers)))


def producing_values(payload, names, *, position, tensor_rank, tensor_parts, compression):
    if set(payload) != set(OBSERVATION_FIELDS):
        raise ObservationUnavailable('complete producing observation inventory differs')
    if type(position) is not int or position < 0 or not 0 <= tensor_rank < tensor_parts:
        raise ValueError('invalid current observation position or ownership')
    stage = payload['stage']
    if stage.captured_m != 1 or stage.arena.layout.storage.quantized:
        raise ObservationUnavailable('current observation transport requires native-KV M1')
    scalars, tensors = {}, {}
    for name, value in payload.items():
        if name in ('stage', 'recorder', 'captured_experts'):
            continue
        if name == 'ordered_events':
            value = _schedule(payload)
        elif name == 'attention_events':
            value = [event[:-1] for event in value]
        elif name == 'normalizations':
            value = list(value)
            for i, offsets in payload['index_key_norms'].items():
                x, weight, y, epsilon, group = value[i]
                owned = ((offsets // compression).remainder(tensor_parts) == tensor_rank)
                value[i] = (x, weight, torch.where(
                    owned.reshape(-1, *([1] * (y.ndim - 1))), y, torch.zeros_like(y)),
                    epsilon, group)
        elif name == 'selector_events':
            # No score/order is produced or consumed by the no-ranking path.
            # Keep every query/key, chosen ID/count, activity and input descriptor.
            value = [(*event[:-1], event[-1] if (position + 1) // event[0]
                      > event[1] // event[0] else None) for event in value]
        _flatten(value, name, names, scalars, tensors)
    _flatten(payload['captured_experts'], 'captured_experts', names, scalars, tensors)
    recorder = payload['recorder']
    for name in ('embeddings', 'linear', 'int8_scales', 'shared'):
        _flatten(getattr(recorder, name), 'recorder/' + name, names, scalars, tensors)
    _flatten({name: value[0] for name, value in stage.arena.states.items()},
             'native/states', names, scalars, tensors)
    for name in ('key', 'value', 'written'):
        _flatten(getattr(stage.arena, name), 'native/' + name, names, scalars, tensors)
    _flatten(stage.arena.page_fields, 'native/pages', names, scalars, tensors)
    _flatten(stage.logits, 'native/logits', names, scalars, tensors)
    _flatten(stage.draft_feedback, 'native/feedback', names, scalars, tensors)
    return scalars, tensors


def weight_bindings(payload, names, *, include_experts=True):
    """Stamp actual producers; unnamed Tensor operands are compared in full.

    A derived immutable weight operand need not have a checkpoint-name entry. Its
    exact typed contents remain in producing_values, and its own version must
    stay unchanged throughout the consumer. Unknown encoded objects cannot
    obtain this transport proof by being treated as anonymous tensors.
    """
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    selected = {}
    def use(weight, site):
        if id(weight) in names:
            key = names[id(weight)]
        elif isinstance(weight, (torch.Tensor, QuantizedMatrix)) and not site.startswith('captured_experts/'):
            key = ('complete_weight_operand', site)
        else:
            raise ObservationUnavailable('unbound producing weight at ' + site + ': ' +
                                         type(weight).__qualname__)
        stamp = ReferenceMathMemo.stamp(weight)
        if stamp is None:
            raise ObservationUnavailable('current weight has no immutable version binding')
        selected[key] = stamp
    recorder = payload['recorder']
    for label, events in (('records', payload['records']),
                          ('recorder/linear', recorder.linear),
                          ('recorder/embeddings', recorder.embeddings)):
        for i, (weight, *_) in enumerate(events):
            use(weight, label + '/' + str(i))
    for i, (_, weight, *_) in enumerate(payload['normalizations']):
        use(weight, 'normalizations/' + str(i))
    for i, (_, weight, *_) in enumerate(payload['conv_events']):
        use(weight, 'conv_events/' + str(i))
    for i, (_, _, weight, *_) in enumerate(payload['dilated_events']):
        use(weight, 'dilated_events/' + str(i))
    for label in ('gdn_events', 'gdn_norm_events'):
        for i, (weight, *_) in enumerate(payload[label]):
            use(weight, label + '/' + str(i))
    if include_experts:
        for i, (groups, _, ids, *_) in enumerate(payload['captured_experts']):
            for expert in set(ids.cpu().flatten().tolist()):
                for member, group in enumerate(groups):
                    use(group[expert], f'captured_experts/{i}/{expert}/{member}')
    return selected


def compare_values(left, right):
    """All required producing bits; a mismatch requests fresh original math."""
    a, x = left
    b, y = right
    differences = []
    if a != b:
        differences.append('typed_scalar_or_weight_binding')
    if x.keys() != y.keys():
        differences.append('producing_tensor_inventory')
    size = 0
    for name in x.keys() & y.keys():
        first, second = x[name], y[name]
        size += first.numel() * first.element_size()
        if first.shape != second.shape or first.dtype != second.dtype:
            differences.append(name)
        elif first.is_contiguous() and second.is_contiguous():
            if not same(first, second):differences.append(name)
        else:
            from . import byte_exact
            if not byte_exact(first, second):differences.append(name)
    return dict(exact=not differences, tensor_count=len(x), bytes=size,
                differences=tuple(sorted(differences)))


def attention_witnesses(payload, *, position, attention_dtype):
    """Re-evaluate each actual partition's own independent reduction enclosure."""
    from .attention_bounds import enclose
    reports = []
    for layer, scale, query, key, value, table, counts, output, chosen, splits in (
            payload['attention_events']):
        selected = chosen[0, :int(counts[0])].long()
        length = position + 1
        if (selected.numel() == 0 or bool(((selected < 0) | (selected >= length)).any())
                or selected.unique().numel() != selected.numel()):
            raise RuntimeError('current attention selected-position ownership differs')
        selected = selected.sort().values
        keys = key[table[0, selected // key.shape[2]].long(), layer,
                   selected % key.shape[2]][None]
        values = value[table[0, selected // value.shape[2]].long(), layer,
                       selected % value.shape[2]][None]
        proof = enclose(query, keys, values, output, scale=scale, splits=splits,
                        probability_dtype=torch.float32 if attention_dtype == 'fp32' else query.dtype)
        if proof['outside_enclosure']:
            raise AssertionError('current attention producing value is outside its own enclosure')
        reports.append(proof)
    return reports


class CurrentPlanWitnesses:
    """Transport one completed step's operator math to exact native consumers.

    This owns no independent state trajectory and caches no model verdict.
    A consumer keeps its own native state. Initial state, every producing value,
    every rounded transition and the weight versions must match the current
    original witness; changed attention reductions are independently enclosed.
    Unsupported/different consumers return to their complete original audit.
    """
    _INPUTS = ('token_ids', 'position_ids', 'page_source', 'page_destination',
               'token_offsets', 'page_table', 'sequence_lengths', 'segment_lengths',
               'row_activity', 'state_slot_maps', 'target_hidden')

    def __init__(self, pool, bindings, names, group, *, input_byte_budget=64 << 20):
        self.pool, self.bindings, self.names, self.group = pool, dict(bindings), names, group
        self.active, self.steps, self.unavailable = {}, {}, {}
        self.source = None
        if type(input_byte_budget) is not int or input_byte_budget < 0:
            raise ValueError('current input witness budget must be nonnegative')
        self.input_byte_budget = input_byte_budget
        self.inputs, self.input_position, self.input_peak_bytes = None, None, 0

    def _gather(self, value):
        if not dist.is_initialized():
            return [value]
        gathered = [None] * dist.get_world_size(self.group)
        dist.all_gather_object(gathered, value, group=self.group)
        return gathered

    @staticmethod
    def _state(stage):
        return dict(states={name: value[0] for name, value in stage.arena.states.items()},
                    pages=stage.arena.page_fields, key=stage.arena.key,
                    value=stage.arena.value, written=stage.arena.written)

    def _state_values(self, stage):
        scalars, tensors = {}, {}
        _flatten(self._state(stage), 'initial', {}, scalars, tensors)
        return scalars, tensors

    def _retire(self, name, reason):
        binding = self.active.pop(name, None)
        if binding is not None:
            self.pool.release(binding)
        self.steps.pop(name, None)
        self.unavailable[name] = reason
        logging.getLogger(__name__).info("current operator proof transport unavailable: %s", reason)

    def start(self, stage, payload):
        from cubic_serve.runtime.text_qualification import prepare_rows
        if self.active:
            raise RuntimeError('previous current operator witness is still borrowed')
        self.source, self.steps, self.unavailable = stage, {}, {}
        self.inputs, self.input_position = None, None
        family = reference_family(stage, self.names)
        for name, binding in self.bindings.items():
            present = binding in self.pool.entries and binding not in self.pool.borrowed
            if not all(self._gather(present)):
                self.unavailable[name] = 'private observed capture unavailable'
                continue
            other = self.pool.acquire(binding)
            self.active[name] = binding
            candidate = other['stage']
            eligible = (reference_family(candidate, self.names) == family
                        and _schedule(other) == _schedule(payload))
            if not all(self._gather(eligible)):
                self._retire(name, 'different independent rule or semantic event schedule')
                continue
            prepare_rows(candidate, 1)
            candidate.arena.reset_page_rows(torch.arange(candidate.arena.spec.page_capacity,
                                                         device=candidate.hidden.device))
            result = compare_values(self._state_values(stage), self._state_values(candidate))
            if not all(self._gather(result['exact'])):
                self._retire(name, 'initial native state requires its own original witness')
                continue
            self.steps[name] = []

    @classmethod
    def _input_values(cls, stage):
        return dict(**{name: getattr(stage.buffers, name) for name in cls._INPUTS},
                    feature_rows=stage.feature_rows, embedding_mask=stage.embedding_mask)

    @staticmethod
    def _flat_inputs(inputs):
        scalars, tensors = {}, {}
        _flatten(inputs, 'inputs', {}, scalars, tensors)
        return scalars, tensors

    def _protected_inputs(self, stage):
        values = self._input_values(stage)
        expected = {name: value for name, value in self.inputs.items()
                    if name not in ('target_hidden', 'feature_rows')}
        actual = {name: values[name] for name in expected}
        return compare_values(self._flat_inputs(expected), self._flat_inputs(actual))['exact']

    def before_step(self, stage, position):
        from .continuation import _tree
        if not self.active:
            return
        expected = set(self._INPUTS) | {'norm', 'hidden'}
        if {field.name for field in fields(stage.buffers)} != expected:
            raise ObservationUnavailable('native input metadata schema is unmapped')
        if stage is not self.source or (self.input_position is not None
                                       and position != self.input_position + 1):
            raise RuntimeError('current operator input step or owner differs')
        values, size = self._input_values(stage), 0
        def count(tensor):
            nonlocal size
            size += tensor.numel() * tensor.element_size()
            return tensor
        _tree(values, count)
        if not all(self._gather(size <= self.input_byte_budget)):
            for name in tuple(self.active):
                self._retire(name, 'bounded current input witness unavailable')
            return
        with torch.inference_mode(False), torch.no_grad():
            self.inputs = _tree(values, lambda tensor: tensor.detach().clone())
        self.input_position = position
        self.input_peak_bytes = max(self.input_peak_bytes, size)
        for name, binding in tuple(self.active.items()):
            candidate = self.pool.entries[binding][0]['stage']
            error = None
            try:
                destinations = self._input_values(candidate)
                for field, value in self.inputs.items():
                    destination = destinations[field]
                    if value is None:
                        if destination is not None:
                            raise ValueError('native conditioning presence differs')
                        continue
                    pairs = zip(value, destination, strict=True) if isinstance(value, tuple) else ((value, destination),)
                    for source, target in pairs:
                        if source.shape != target.shape or source.dtype != target.dtype:
                            raise ValueError('native input geometry or precision differs')
                        target.copy_(source)
                if not compare_values(self._flat_inputs(self.inputs),
                                      self._flat_inputs(destinations))['exact']:
                    raise ValueError('native input copy changed an actual input bit')
            except Exception as exc:
                error = str(exc)
            errors = self._gather(error)
            if any(errors):
                raise RuntimeError('current observation input binding failed: ' + str(errors))

    def completed_step(self, position, payload, report):
        import copy
        source = self.source
        if self.active and (payload['stage'] is not source or self.input_position != position):
            raise RuntimeError('current operator completed step differs from its input owner')
        if self.active and not all(self._gather(self._protected_inputs(source))):
            raise RuntimeError('source forward changed protected discrete input metadata')
        options = dict(position=position, tensor_rank=source.tensor_rank,
                       tensor_parts=source.topology.tp_size, compression=source.indexer_plan.compression)
        for name, binding in tuple(self.active.items()):
            other = self.pool.entries[binding][0]
            # No reference work, D2H read or assertion runs inside either graph.
            candidate = other['stage']
            stamp_error, before, other_before = None, None, None
            try:
                before = weight_bindings(payload, self.names)
                other_before = {key: value for key, value in weight_bindings(
                    other, self.names, include_experts=False).items() if isinstance(key, tuple)}
            except ObservationUnavailable as exc:
                stamp_error = str(exc)
            stamp_errors = self._gather(stamp_error)
            if any(stamp_errors):
                self._retire(name, 'original weight binding unavailable: ' + str(stamp_errors))
                continue
            candidate.replay()
            candidate.synchronize()
            agree_schedules((payload, other), self.group)
            error, result, attention = None, None, None
            try:
                result = compare_values(producing_values(payload, self.names, **options),
                                        producing_values(other, self.names, **options))
                if before != weight_bindings(payload, self.names):
                    raise RuntimeError('actual current weight version binding changed')
                if not self._protected_inputs(candidate):
                    raise RuntimeError('candidate forward changed protected discrete input metadata')
                other_after = weight_bindings(other, self.names)
                if other_before != {key: value for key, value in other_after.items() if isinstance(key, tuple)}:
                    raise RuntimeError('anonymous producing Tensor version binding changed')
                if result['exact']:
                    if ({key: value for key, value in before.items() if not isinstance(key, tuple)}
                            != {key: value for key, value in other_after.items() if not isinstance(key, tuple)}):
                        raise ObservationUnavailable('different actual selected checkpoint weight binding')
                    attention = attention_witnesses(other, position=position,
                                                     attention_dtype=candidate.operators.config.attention_dtype)
            except ObservationUnavailable as exc:
                error = ('unavailable', str(exc))
            except AssertionError as exc:
                error = ('numerical', str(exc))
            except Exception as exc:
                error = ('execution', str(exc))
            errors = self._gather(error)
            failures = [value for value in errors if value is not None]
            if failures:
                if any(kind != 'unavailable' for kind, _ in failures):
                    exception = (AssertionError if any(kind == 'numerical' for kind, _ in failures)
                                 else RuntimeError)
                    raise exception('current operator witness failed: ' + str(errors))
                self._retire(name, 'complete original math required: ' + str(errors))
                continue
            if not all(self._gather(result['exact'])):
                self._retire(name, 'different actual producing values require original math: ' +
                             repr(result['differences'][:8]))
                continue
            # Reuse current exact-operand mathematical certificates. Recompute
            # candidate acceptance facts after the full owned-state/operand proof;
            # no saved model verdict or prior trajectory is consulted.
            step = {key: copy.deepcopy(value) for key, value in report.items()
                    if key != 'candidate_boundary_continuation_state_and_logits_exact'}
            step['attention_reduction_proofs'] = attention
            step['candidate_boundary_continuation_state_and_logits_exact'] = True
            step['current_operator_math_transport'] = dict(
                position=position, tensor_count=result['tensor_count'], bytes=result['bytes'],
                initial_native_prefix_exact=True, actual_weight_versions_exact=True,
                source='completed current original-weight causal and free-reference step',
                own_attention_reduction_enclosures=True,
                input_witness_peak_bytes=self.input_peak_bytes,
                discrete_input_protection_exact=True)
            import hashlib
            step['current_operator_math_transport']['weight_binding_digest'] = (
                hashlib.sha256(repr(before).encode()).hexdigest())
            self.steps[name].append(step)

    def report(self, name, original):
        import copy
        if name in self.unavailable or name not in self.steps:
            return None
        steps = self.steps[name]
        if len(steps) != len(original['steps']) or len(steps) < 2:
            return None
        result = {key: copy.deepcopy(value) for key, value in original.items()
                  if key not in ('steps', 'cross_step_error_propagation_qualified')}
        result['steps'] = steps
        result['cross_step_error_propagation_qualified'] = all(
            step['candidate_boundary_continuation_state_and_logits_exact']
            and step['independent_trajectory_diagnostic']['route_branch_schedule_exact']
            for step in steps)
        result['current_operator_math_transport'] = True
        return result

    def finish(self):
        for name, binding in tuple(self.active.items()):
            self.pool.release(binding)
            del self.active[name]
        self.source = self.inputs = self.input_position = None


def current_plan_reports(members, bindings, pool, names, group, audit):
    """Witness every plan, using fresh original anchors for distinct values.

    Equivalence is established on completed current operands and transitions,
    separately for each corpus. A failed identity never becomes an FP failure
    or an inferred pass: its member later runs the complete independent audit.
    No grouping survives this invocation and no model/device policy is fitted.
    """
    if not members:
        raise ValueError('current plan proof requires its complete member inventory')
    remaining, reports = list(members), {}
    resources = dict(complete_original_reports=0, transported_reports=0,
                     input_witness_peak_bytes=0, fallback_reasons={}, groups=0)
    while remaining:
        anchor = remaining.pop(0)
        session = (CurrentPlanWitnesses(pool, {identity: bindings[identity]
                   for identity in remaining if identity in bindings}, names, group)
                   if remaining else None)
        try:
            report = (audit(members[anchor]) if session is None else
                      audit(members[anchor], operator_observers=session))
        finally:
            if session is not None:
                session.finish()
        reports[anchor] = report
        resources['complete_original_reports'] += 1
        resources['groups'] += 1
        if session is None:
            continue
        resources['input_witness_peak_bytes'] = max(
            resources['input_witness_peak_bytes'], session.input_peak_bytes)
        for reason in session.unavailable.values():
            reasons = resources['fallback_reasons']
            reasons[reason] = reasons.get(reason, 0) + 1
        for identity in tuple(remaining):
            candidate = session.report(identity, report)
            if candidate is not None:
                reports[identity] = candidate
                remaining.remove(identity)
                resources['transported_reports'] += 1
    if reports.keys() != members.keys():
        raise RuntimeError('current plan proof lost a required member')
    return reports, resources
