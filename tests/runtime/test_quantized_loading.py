"""Private construction overlaps reads; execution still joins the source transaction."""
from threading import Event

import pytest
import torch
from safetensors.torch import save_file

from cubic_serve.models.parallel_weights import AxisSpan, LogicalWeightSlice
from cubic_serve.models.quantization import QuantizationError
from cubic_serve.models.quantized_checkpoint import CheckpointWeights, LogicalWeight, QuantizedMatrix
from cubic_serve.runtime.quantized import QuantizedWeights, _PendingRankWeights


def test_residency_walk_handles_optional_and_encoded_prepared_entries():
    from cubic_serve.models.quantization import QuantizationScheme, WeightFormat

    matrix = QuantizedMatrix(
        {"qweight": torch.zeros(256, 2, dtype=torch.int32),
         "qzeros": torch.zeros(2, 2, dtype=torch.int32),
         "scales": torch.ones(2, 16, dtype=torch.float16)},
        QuantizationScheme(WeightFormat.AWQ, 4, 128), (16, 256),
    )
    weights = QuantizedWeights.__new__(QuantizedWeights)
    weights._weights = {"projection.weight": matrix}
    weights.prepared_weights = {("no_zero_correction",): (matrix, None),
                                ("grouped_checkpoint",): (matrix, matrix)}
    weights.materialized_weights = {("alias",): (matrix, matrix)}
    named = weights.named_tensors()
    assert len(named) == 9 and all(isinstance(t, torch.Tensor) for _, t in named)
    assert weights.storage_bytes == dict(
        device=0, pinned_host=0,
        pageable_host=sum(t.untyped_storage().nbytes() for t in matrix.payload.values()),
    )
    assert weights.resident_bytes == 0


def test_residency_counts_backing_stores_once_and_keeps_mapped_host_on_cpu(monkeypatch):
    plain = torch.empty(16, dtype=torch.bfloat16)
    pinned = torch.empty(32, dtype=torch.uint8)
    pinned_pointer = pinned.untyped_storage().data_ptr()
    # Exercise placement accounting without allocating CUDA/pinned pages in a CPU test.
    monkeypatch.setattr(torch.Tensor, "is_pinned",
                        lambda t: t.untyped_storage().data_ptr() == pinned_pointer)
    weights = QuantizedWeights.__new__(QuantizedWeights)
    weights._weights = {"plain": plain[3:], "table": pinned[4:]}
    weights.prepared_weights = {("alias",): (plain, plain[:2])}
    weights.materialized_weights = {}
    assert weights.storage_bytes == dict(device=0, pinned_host=32, pageable_host=32)


def fixture(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    data = {f'{name}.weight': torch.arange(32, dtype=torch.bfloat16).reshape(4, 8) + i
            for i, name in enumerate(('a', 'b', 'c'))}
    save_file(data, tmp_path / 'model.safetensors')
    plan = CheckpointWeights.admit(tmp_path, tuple(
        LogicalWeight(name, (4, 8), 'BF16', 'language', 'tensor') for name in data),
        verify_payloads=False, require_quantization=False)
    owners = tuple(LogicalWeightSlice(name, 0, (AxisSpan(0, 4),), (4, 8)) for name in data)
    return plan, owners, data


def test_private_operands_are_available_while_remaining_read_is_blocked(tmp_path, monkeypatch):
    plan, owners, expected = fixture(tmp_path)
    entered, release = Event(), Event()
    original = CheckpointWeights.materialize
    def read(self, weight, **kwargs):
        if weight.logical.name == 'b.weight':
            entered.set()
            assert release.wait(5), 'test producer never released'
        return original(self, weight, **kwargs)
    monkeypatch.setattr(CheckpointWeights, 'materialize', read)
    weights = QuantizedWeights(plan, device='cpu', compute_dtype=torch.bfloat16,
                               owner_bindings=owners, _overlap_loading=True)
    try:
        assert entered.wait(5)
        assert torch.equal(weights['a.weight'], expected['a.weight'])
        assert isinstance(weights._weights, _PendingRankWeights)
        assert not weights._weights._done
    finally:
        release.set()
        weights.finish_loading()
    assert isinstance(weights._weights, dict)
    for name in expected:
        assert torch.equal(weights[name], expected[name])


def test_final_source_failure_is_not_hidden_by_early_construction(tmp_path, monkeypatch):
    import os
    plan, owners, _ = fixture(tmp_path)
    entered, release = Event(), Event()
    original = CheckpointWeights.materialize
    def read(self, weight, **kwargs):
        if weight.logical.name == 'b.weight':
            entered.set()
            assert release.wait(5)
        return original(self, weight, **kwargs)
    monkeypatch.setattr(CheckpointWeights, 'materialize', read)
    weights = QuantizedWeights(plan, device='cpu', compute_dtype=torch.bfloat16,
                               owner_bindings=owners, _overlap_loading=True)
    try:
        assert entered.wait(5)
        weights['a.weight']  # a private operand is not final execution admission
        path = tmp_path / 'model.safetensors'
        stamp = path.stat()
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1000000))
        release.set()
        with pytest.raises(QuantizationError, match='changed during rank'):
            weights.finish_loading()
    finally:
        release.set()
        weights.finish_loading(cancel=True)


def test_reader_error_wakes_consumer_and_cancel_joins(tmp_path, monkeypatch):
    plan, owners, _ = fixture(tmp_path)
    def fail(*a, **kw):
        raise OSError('read failed')
    monkeypatch.setattr(CheckpointWeights, 'materialize', fail)
    weights = QuantizedWeights(plan, device='cpu', compute_dtype=torch.bfloat16,
                               owner_bindings=owners, _overlap_loading=True)
    pending = weights._weights
    with pytest.raises(OSError, match='read failed'):
        weights['c.weight']
    weights.finish_loading(cancel=True)
    assert not pending._thread.is_alive()


def test_demand_changes_only_read_order_not_owner_spans(tmp_path):
    plan, owners, _ = fixture(tmp_path)
    requests = tuple((w, 0, (AxisSpan(0, 4),)) for w in plan.weights)
    pending = _PendingRankWeights(requests, torch.device('cpu'))
    pending._demand.append('c.weight')
    order = tuple(pending.order())
    assert order[0][0].logical.name == 'c.weight'
    assert set(order) == set(requests)
    for bad in (order[:-1], (*order, order[0]),
                ((order[0][0], 1, order[0][2]), *order[1:])):
        with pytest.raises(QuantizationError, match='pending load order'):
            plan.materialize_many(requests, device='cpu', compute_dtype=torch.bfloat16,
                                  _request_order=bad)


@pytest.mark.parametrize('bits', [4, 5, 8])
def test_encoded_owner_payloads_remain_bit_exact(tmp_path, bits):
    import json
    from tests.ops.test_int_quantization import fixture as integer_fixture
    matrix, _ = integer_fixture(bits, n=32, k=64, group=32, sym=True)
    (tmp_path / 'config.json').write_text(json.dumps({'quantization_config': {
        'quant_method': 'compressed-tensors', 'format': 'pack-quantized',
        'config_groups': {'int': {'targets': ['linear'], 'weights': {
            'num_bits': bits, 'type': 'int', 'strategy': 'group',
            'group_size': 32, 'symmetric': True}}}}}))
    save_file({'linear.' + suffix: value for suffix, value in matrix.payload.items()},
              tmp_path / 'model.safetensors')
    plan = CheckpointWeights.admit(tmp_path, (
        LogicalWeight('linear.weight', (32, 64), 'BF16', 'language', 'linear'),),
        verify_payloads=False)
    for axis, span in ((0, AxisSpan(8, 24)), (1, AxisSpan(32, 64))):
        owners = (LogicalWeightSlice('linear.weight', axis, (span,), (32, 64)),)
        direct = QuantizedWeights(plan, device='cpu', compute_dtype=torch.bfloat16,
                                  owner_bindings=owners)
        candidate = QuantizedWeights(plan, device='cpu', compute_dtype=torch.bfloat16,
                                     owner_bindings=owners, _overlap_loading=True)
        candidate.finish_loading()
        actual, expected = candidate['linear.weight'], direct['linear.weight']
        assert actual.encoding == expected.encoding
        assert actual.shape == expected.shape
        for key in expected.payload:
            assert torch.equal(actual.payload[key].view(torch.uint8),
                               expected.payload[key].view(torch.uint8))


def test_cuda_reader_event_precedes_binding_and_allocation_owners_survive(tmp_path):
    import os
    if not os.environ.get('CUBIC_TEST_QUANT_CUDA'):
        pytest.skip('explicit CUDA storage audit')
    plan, owners, expected = fixture(tmp_path)
    device = torch.device('cuda', 0)
    weights = QuantizedWeights(plan, device=device, compute_dtype=torch.bfloat16,
                               owner_bindings=owners, _overlap_loading=True)
    try:
        # Consume before joining: producer and constructor have distinct streams.
        copies = {name: weights[name].clone() for name in expected}
        weights.finish_loading()
        torch.cuda.synchronize(device)
        for name, value in copies.items():
            assert torch.equal(value.cpu(), expected[name])
    finally:
        weights.finish_loading(cancel=True)


def test_cuda_calibration_exclusion_preserves_global_capture(tmp_path):
    import os
    if not os.environ.get('CUBIC_TEST_QUANT_CUDA'):
        pytest.skip('explicit CUDA capture/transfer audit')
    from cubic_serve.ops.cuda.startup import startup_cuda_lock
    plan, owners, expected = fixture(tmp_path)
    device = torch.device('cuda', 0)
    source = torch.ones(32, device=device)
    output = torch.empty_like(source)
    weights = None
    try:
        with startup_cuda_lock(device):
            weights = QuantizedWeights(plan, device=device, compute_dtype=torch.bfloat16,
                                       owner_bindings=owners, _overlap_loading=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):  # keep the ordinary global safety mode
                output.copy_(source)
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.equal(output, source)
        weights.finish_loading()
        for name in expected:
            assert torch.equal(weights[name].cpu(), expected[name])
    finally:
        if weights is not None:
            weights.finish_loading(cancel=True)
