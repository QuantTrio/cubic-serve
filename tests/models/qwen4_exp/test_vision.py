"""Shared vision math and request transport at multimodal boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cubic_serve.contracts.request import DecodePositionPlan, TokenPosition
from cubic_serve.inputs import EncodedFeatureBundle
from cubic_serve.models.protocol import PreparedInput
from cubic_serve.models.qwen3_5.multimodal import (
    Qwen3_5VisionWeights,
    QwenVisionReferenceModel,
    _bilinear_position_inputs,
    _position_ids,
    VisionTensorParallel,
    vision_weight_requests,
)
from cubic_serve.models.qwen4_exp import parse_qwen4_exp_config
from cubic_serve.models.qwen4_exp.weights import _add_vision
from cubic_serve.runtime.continuous_batch import prepared_input_fields
from tests.models.qwen3_5.test_model_multimodal import _prepared_and_store
from tests.models.qwen4_exp.test_config import _fixture


@pytest.mark.parametrize("metadata", (
    torch.tensor([[1, 2, 4]], dtype=torch.int64),
    torch.tensor([0.1, 1.25], dtype=torch.float64),
    torch.zeros((0, 3), dtype=torch.int32), None,
))
def test_common_encoder_transport_preserves_model_metadata(metadata):
    from cubic_serve.executor.vision import _pack_encoder_metadata, _unpack_encoder_metadata
    actual = _unpack_encoder_metadata(_pack_encoder_metadata(metadata))
    if metadata is None:
        assert actual is None
    else:
        assert actual.device.type == "cpu" and actual.dtype == metadata.dtype
        assert actual.shape == metadata.shape and torch.equal(actual, metadata)


@pytest.mark.parametrize("size", (1, 2, 4, 8))
def test_vision_tp_complete_head_order_and_paired_channel_coverage(size):
    from cubic_serve.models.quantized_checkpoint import EncodedWeight, LogicalWeight
    config = parse_qwen4_exp_config(_fixture())
    vision = replace(config.vision, layer_count=1, hidden_size=32, head_count=8,
                     intermediate_size=35, output_hidden_size=24)
    bindings = []
    def add(name, shape, node, partition, dtype="BF16"):
        bindings.append(EncodedWeight(LogicalWeight(name, shape, dtype, partition), None, ()))
    _add_vision(add, replace(config, vision=vision))
    checkpoint = SimpleNamespace(weights=tuple(bindings))
    owners = [dict((w.logical.name, (axis, spans)) for w, axis, spans in
                   vision_weight_requests(checkpoint, vision, size=size, rank=rank)[0])
              for rank in range(size)]
    qkv = "model.visual.blocks.0.attn.qkv.weight"
    for part in range(3):
        actual = []
        for owned in owners:
            spans = owned[qkv][1]
            if size == 1:
                actual.extend(range(part * 32, (part + 1) * 32))
            else:
                span = spans[part]
                actual.extend(range(span.start, span.stop))
        assert actual == list(range(part * 32, (part + 1) * 32))
    for prefix, width in (("model.visual.blocks.0.mlp", 35), ("model.visual.merger", 128)):
        seen = []
        for owned in owners:
            up = owned[prefix + ".linear_fc1.weight"]
            down = owned[prefix + ".linear_fc2.weight"]
            if size > 1:
                assert up[0] == 0 and down[0] == 1 and up[1] == down[1]
            for span in up[1]:
                seen.extend(range(span.start, span.stop))
        assert seen == list(range(width))


def test_vision_row_parallel_bias_is_added_after_reduction_once():
    config = parse_qwen4_exp_config(_fixture())
    vision = replace(config.vision, layer_count=1, hidden_size=8, head_count=2,
                     intermediate_size=16, output_hidden_size=16)
    tensors = {}
    def add(name, shape, *args, **kwargs):
        tensors[name] = torch.zeros(shape, dtype=torch.bfloat16)
    _add_vision(add, replace(config, vision=vision))
    name = "model.visual.blocks.0.attn.proj"
    tensors[name + ".weight"] = torch.ones(8, 4, dtype=torch.bfloat16)
    tensors[name + ".bias"].fill_(7)
    weights = Qwen3_5VisionWeights(tensors=tensors, device=torch.device("cpu"),
        plan=SimpleNamespace(checkpoint_header_digest="x", semantic_plan_digest="y"))
    observed = []
    def reduce(value):
        observed.append(value.clone())
        return value * 2
    model = QwenVisionReferenceModel(config=replace(config, vision=vision), weights=weights,
        parallel=VisionTensorParallel(2, 0, frozenset((name,))), reduce=reduce)
    output = model.project(torch.ones(3, 4, dtype=torch.bfloat16), name)
    assert torch.equal(observed[0], torch.full((3, 8), 4, dtype=torch.bfloat16))
    assert torch.equal(output, torch.full((3, 8), 15, dtype=torch.bfloat16))


@pytest.mark.parametrize("image_grid", ((2, 2), (34, 34)))
def test_visual_padding_and_frame_isolation_preserve_logical_features(image_grid):
    config = parse_qwen4_exp_config(_fixture())
    v = replace(config.vision, layer_count=1, hidden_size=8, intermediate_size=16,
                output_hidden_size=16, head_count=2, patch_size=2,
                temporal_patch_size=2, spatial_merge_size=2, position_count=4,
                deepstack_layer_indexes=())
    config = replace(config, vision=v)
    tensors = {}
    generator = torch.Generator().manual_seed(118)
    def add(name, shape, *args, **kwargs):
        tensors[name] = torch.randn(shape, generator=generator).to(torch.bfloat16) * 0.1
    _add_vision(add, config)
    weights = Qwen3_5VisionWeights(tensors=tensors, device=torch.device('cpu'),
                                   plan=SimpleNamespace(checkpoint_header_digest='x',
                                                        semantic_plan_digest='y'))
    model = QwenVisionReferenceModel(config=config, weights=weights)
    image_rows = image_grid[0] * image_grid[1]
    logical_rows = image_rows + 8
    capacity = max(16, 1 << (logical_rows - 1).bit_length())
    padding = capacity - logical_rows
    grid = torch.tensor([[1, *image_grid], [1, 2, 4]])
    pixels = torch.randn((logical_rows, 24), generator=generator)
    expected = model.forward(pixels, grid)
    positions = _position_ids(grid, 2)
    indices, fractions = _bilinear_position_inputs(grid, num_grid_per_side=2,
                                                  spatial_merge_size=2)
    frame = torch.tensor([0]*image_rows + [1]*8 + [2]*padding)
    padded_pixels = torch.cat((pixels, torch.full((padding, 24), 73.0)))
    actual = model.encode(padded_pixels, torch.cat((positions, torch.zeros(padding, 2))),
                         torch.cat((indices, torch.zeros(4, padding, dtype=torch.int64)), 1),
                         torch.cat((fractions, torch.zeros(4, padding)), 1), None,
                         frame[:, None] == frame[None, :])
    assert torch.equal(actual[:logical_rows // 4], expected)
    changed = pixels.clone()
    changed[image_rows:] += 19
    assert torch.equal(model.forward(changed, grid)[:image_rows // 4],
                       expected[:image_rows // 4])
    assert not torch.equal(model.forward(changed, grid)[image_rows // 4:],
                           expected[image_rows // 4:])


def test_feature_transport_preserves_width_bytes_digest_and_positions():
    prepared, store = _prepared_and_store()
    plan = prepared.input_plan
    positions, first, step, rows, digest = prepared_input_fields(
        plan.token_ids, prepared, hidden_size=6)
    assert positions == tuple(p.axes for p in plan.positions)
    assert (first, step) == (plan.decode_positions.first.axes, plan.decode_positions.step_axes)
    assert rows[0] == b'' and len(rows[1]) == 12
    assert torch.equal(torch.frombuffer(bytearray(rows[1]), dtype=torch.bfloat16),
                       torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.bfloat16))
    assert digest == prepared.feature_bundle.semantic_digest
    with pytest.raises(ValueError, match='width'):
        prepared_input_fields(plan.token_ids, prepared, hidden_size=24)
    with pytest.raises(ValueError, match='token ids'):
        prepared_input_fields((9, 9), prepared, hidden_size=6)


def test_request_admission_bounds_multimodal_axes_and_decode_continuation():
    prepared, _ = _prepared_and_store()
    plan = replace(
        prepared.input_plan,
        positions=(TokenPosition((0, 0, 0)), TokenPosition((1, 8, 2))),
        decode_positions=DecodePositionPlan(TokenPosition((2, 9, 3)), (1, 1, 1)),
    )
    feature = prepared.feature_bundle.bindings[0].feature
    prepared = PreparedInput(plan, EncodedFeatureBundle.from_input_plan(plan, (feature,)))
    prepared_input_fields(plan.token_ids, prepared, hidden_size=6,
                          max_position=10, max_tokens=1)
    # Token count fits the context, but a spatial axis would leave the RoPE table.
    with pytest.raises(ValueError, match='position axes limit'):
        prepared_input_fields(plan.token_ids, prepared, hidden_size=6,
                              max_position=10, max_tokens=2)
    plan = replace(plan, positions=(TokenPosition((0, 0, 0)), TokenPosition((1, 10, 2))))
    prepared = PreparedInput(plan, EncodedFeatureBundle.from_input_plan(plan, (feature,)))
    with pytest.raises(ValueError, match='position axes limit'):
        prepared_input_fields(plan.token_ids, prepared, hidden_size=6,
                              max_position=10, max_tokens=1)


@pytest.mark.parametrize("tp,pp", ((1,1),(2,1),(4,1),(2,2),(4,2)))
def test_owned_encoder_uses_first_stage_rank_order_and_local_memory_reserve(tp, pp):
    from unittest.mock import MagicMock, Mock, patch
    from cubic_serve.executor.vision import prepare_owned_input_encoder
    placements = tuple(SimpleNamespace(rank=r, pipeline_stage=r//tp)
                       for r in range(tp*pp))
    topology = SimpleNamespace(placements=placements)
    group = object()
    for rank in range(tp*pp):
        encoder = SimpleNamespace(max_input_rows=8, max_dynamic_input_rows=16,
                                  peak_input_bytes=8192,
                                  resident_graph_bytes=1024,
                                  resources=SimpleNamespace(staging_bytes=4096))
        factory, connection = Mock(return_value=encoder), Mock()
        admission = SimpleNamespace(input_encoder_factory=factory,
            input_encoder_reserves={}, input_encoder_connection=connection)
        device = torch.device('cuda', rank)
        with patch('torch.distributed.new_group', return_value=group) as make_group, \
             patch('torch.distributed.get_group_rank', return_value=rank):
            owned = prepare_owned_input_encoder(admission, topology=topology,
                                                 rank=rank, device=device)
        make_group.assert_called_once_with(list(range(tp)))
        if rank < tp:
            assert owned == (encoder, rank)
            factory.assert_called_once_with(device=device, tp_rank=rank, tp_group=group)
            assert admission.input_encoder_reserves == {rank: 8192}
            connection.send.assert_called_once_with(('ready', 8, 4096, 1024, 16, 8192))
        else:
            assert owned is None and admission.input_encoder_reserves == {}
            factory.assert_not_called()
            connection.send.assert_not_called()


def test_borrowed_encoder_workers_preserve_request_outputs_and_close_without_terminating():
    import multiprocessing
    import threading
    from contextlib import nullcontext
    from unittest.mock import MagicMock, Mock, patch
    from cubic_serve.executor.vision import InputEncoderGroup, serve_input_encoder

    def segment(pixels, metadata, capacity):
        for start in range(0, len(pixels), capacity):
            yield pixels[start:start+capacity], metadata

    group = InputEncoderGroup(None, devices=(2,3), config=None, semantic_digest='test',
        segmenter=segment, module='test_input', input_width=3, output_width=3,
        merge_extent=1, colocated=True, admission_item_label='image or video frame')
    lease = MagicMock()
    group.execution_lease = lease
    parents, children = zip(*(multiprocessing.Pipe() for _ in group.devices))
    workers = [Mock(is_alive=Mock(return_value=True)) for _ in group.devices]
    threads=[]
    observed=[[],[]]
    metadata=torch.tensor([[2**54+1]],dtype=torch.int64)
    with patch('torch.cuda.set_device'), patch('torch.cuda.stream',return_value=nullcontext()):
        for rank, child in enumerate(children):
            def run(rank=rank, child=child):
                child.send(('ready',4,4096,1024,16,8192))
                def forward(pixels, actual):
                    observed[rank].append(actual.clone())
                    return (pixels*2).to(torch.bfloat16)
                encoder=SimpleNamespace(stream=None, weights=SimpleNamespace(device='cpu'),
                                        forward=forward)
                serve_input_encoder(encoder,child,rank)
            thread=threading.Thread(target=run)
            thread.start();threads.append(thread)
        try:
            group.attach_main_workers(workers,parents)
            pixels=torch.arange(15).reshape(5,3).float()
            first=group.forward(pixels,metadata)
            second=group.forward(-pixels[:3],metadata)
            group._segmenter=lambda pixels, metadata, capacity: iter(((pixels,metadata),))
            oversize=torch.arange(21).reshape(7,3).float()
            third=group.forward(oversize,metadata)
            with pytest.raises(ValueError, match=r'image or video frame is too large.*admitted_input_rows=16'):
                group.forward(torch.zeros(17,3),metadata)
            fourth=group.forward(-pixels[:3],metadata)
            assert torch.equal(first,(pixels*2).to(torch.bfloat16))
            assert torch.equal(second,(-pixels[:3]*2).to(torch.bfloat16))
            assert torch.equal(third,(oversize*2).to(torch.bfloat16))
            assert torch.equal(fourth,second)
            lease.__enter__.assert_called_once_with()
            lease.__exit__.assert_called_once_with(None, None, None)
            assert torch.equal(first,(pixels*2).to(torch.bfloat16))
            assert all(torch.equal(item,metadata) for records in observed for item in records)
        finally:
            group.close()
            for thread in threads:
                thread.join(timeout=5)
                assert not thread.is_alive()
            for child in children:child.close()
    for worker in workers:
        worker.terminate.assert_not_called()
        worker.join.assert_not_called()
