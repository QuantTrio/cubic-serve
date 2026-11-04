"""Serialization transforms, independent of model execution and CUDA timing."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from gguf import GGUFValueType, GGUFWriter

from cubic_serve.models.gguf import inspect_gguf
from cubic_serve.models.gguf_weights import GGUFPart, GGUFWeightView, validate_gguf_coverage
from cubic_serve.models.parallel_weights import AxisSpan
from cubic_serve.models.quantization import QuantizationError
from cubic_serve.models.quantized_checkpoint import LogicalWeight
from cubic_serve.models.qwen4_exp.gguf import _head_order, _log_decay, _zero_centered


def _bundle(path, values, *, raw_type=None):
    writer = GGUFWriter(path, "test")
    for name, value in values.items():
        writer.add_tensor(name, value, raw_dtype=raw_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    return inspect_gguf(path)


def test_inverse_gdn_head_permutation_preserves_original_encoded_blocks(tmp_path):
    from gguf import GGMLQuantizationType

    # Six value heads, two original scale blocks per head.
    original = np.arange(12 * 36, dtype=np.uint8).reshape(12, 36)
    export = original.reshape(2, 3, 2, 36).transpose(1, 0, 2, 3).reshape(12, 36)
    inv = _bundle(tmp_path / "heads.gguf", {"w": export},
                  raw_type=GGMLQuantizationType.IQ4_NL)
    weight = LogicalWeight("logical.weight", (12, 64), "BF16", "language", "linear")
    view = GGUFWeightView(weight, (GGUFPart(inv, "w"),), row_order=_head_order(2, 6, 2))
    view.validate()
    np.testing.assert_array_equal(view.read().numpy(), original)
    np.testing.assert_array_equal(view.read((slice(3, 9), slice(18, 36))).numpy(),
                                  original[3:9, 18:])
    validate_gguf_coverage((inv,), (view,))


def test_original_block_reads_keep_mutation_tracking_inside_inference(tmp_path):
    from gguf import GGMLQuantizationType

    raw = np.arange(4 * 36, dtype=np.uint8).reshape(4, 36)
    inv = _bundle(tmp_path / "tracked.gguf", {"w": raw},
                  raw_type=GGMLQuantizationType.IQ4_NL)
    logical = LogicalWeight("logical.weight", (4, 64), "BF16", "language", "linear")
    view = GGUFWeightView(logical, (GGUFPart(inv, "w"),))
    view.validate()
    with torch.inference_mode():
        value = view.read()
        assert not value.is_inference() and not value.requires_grad
        version = value._version
        np.testing.assert_array_equal(value.numpy(), raw)
        value[0, 0] = 17
        assert value._version > version
    # The returned tensor owns its bytes; a caller cannot mutate the file view.
    np.testing.assert_array_equal(view.read().numpy(), raw)


def test_inverse_gdn_input_column_permutation_keeps_scale_ownership(tmp_path):
    from gguf import GGMLQuantizationType

    original = np.arange(3 * 6 * 18, dtype=np.uint8).reshape(3, 6, 18)
    export = original.reshape(3, 2, 3, 18).transpose(0, 2, 1, 3).reshape(3, -1)
    inv = _bundle(tmp_path / "columns.gguf", {"w": export},
                  raw_type=GGMLQuantizationType.IQ4_NL)
    weight = LogicalWeight("logical.weight", (3, 192), "BF16", "language", "linear")
    view = GGUFWeightView(weight, (GGUFPart(inv, "w"),), column_order=_head_order(2, 6, 32))
    view.validate()
    np.testing.assert_array_equal(view.read().numpy(), original.reshape(3, -1))
    np.testing.assert_array_equal(view.read((slice(1, 3), slice(36, 72))).numpy(),
                                  original[1:3, 2:4].reshape(2, -1))
    bad_order = list(range(192))
    bad_order[1], bad_order[33] = bad_order[33], bad_order[1]
    with pytest.raises(QuantizationError, match="scale blocks"):
        replace(view, column_order=tuple(bad_order)).validate()


def test_split_join_and_metadata_do_not_change_existing_weight_shapes(tmp_path):
    from cubic_serve.models.gguf_weights import GGUFMetadataTensor

    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = a + 20
    inv = _bundle(tmp_path / "plain.gguf", {"a": a, "b": b})
    weight = LogicalWeight("existing.indexer.weight", (6, 4), "BF16", "language", "linear")
    view = GGUFWeightView(weight, tuple(GGUFPart(inv, key) for key in ("a", "b")))
    validate_gguf_coverage((inv,), (view,))
    assert torch.equal(view.read(), torch.from_numpy(np.concatenate((a, b))))
    assert torch.equal(view.read((slice(1, 5), slice(2, 4))),
                       torch.from_numpy(np.concatenate((a, b))[1:5, 2:4]))
    # Integer metadata never takes a float32 round trip.
    writer = GGUFWriter(tmp_path / "meta.gguf", "test")
    writer.add_key_value("test.multipliers", [23703573157769, 20109073645365],
                         GGUFValueType.ARRAY, sub_type=GGUFValueType.INT64)
    writer.add_tensor("dummy", np.ones((1, 1), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    meta = inspect_gguf(tmp_path / "meta.gguf")
    declared = LogicalWeight("multipliers", (2,), "I64", "language")
    integer = GGUFWeightView(declared, metadata=GGUFMetadataTensor(meta, "test.multipliers"))
    integer.validate()
    assert integer.read().tolist() == [23703573157769, 20109073645365]


def test_rectangular_ownership_closes_both_rows_and_columns(tmp_path):
    inv = _bundle(tmp_path / "source.gguf", {"w": np.ones((4, 8), dtype=np.float32)})
    views = tuple(GGUFWeightView(
        LogicalWeight(str(i), (4, 4), "BF16", "language", "linear"),
        (GGUFPart(inv, "w", columns=AxisSpan(i * 4, (i + 1) * 4)),)) for i in range(2))
    validate_gguf_coverage((inv,), views)
    with pytest.raises(QuantizationError, match="unowned columns"):
        validate_gguf_coverage((inv,), views[:1])
    with pytest.raises(QuantizationError, match="overlapping"):
        validate_gguf_coverage((inv,), (*views, replace(views[0], logical=replace(
            views[0].logical, name="duplicate"))))
    with pytest.raises(QuantizationError, match="unowned rows"):
        validate_gguf_coverage((inv,), ())


def test_plain_integer_payload_never_takes_a_float_round_trip(tmp_path):
    from gguf import GGMLQuantizationType

    values = np.array([23703573157769, -20109073645365], dtype=np.int64)
    inv = _bundle(tmp_path / "integers.gguf", {"exact": values},
                  raw_type=GGMLQuantizationType.I64)
    logical = LogicalWeight("hash.constants", (2,), "I64", "language")
    view = GGUFWeightView(logical, (GGUFPart(inv, "exact"),))
    validate_gguf_coverage((inv,), (view,))
    assert view.read().tolist() == values.tolist()
    with pytest.raises(QuantizationError, match="dtype contract"):
        replace(view, logical=replace(logical, dtype="BF16")).validate()


def test_plain_inverse_export_math_and_temporal_patch_join(tmp_path):
    gamma = torch.tensor([-0.125, 0.0, 0.0625], dtype=torch.bfloat16)
    assert torch.equal(_zero_centered(gamma.float() + 1).bfloat16(), gamma)
    decay_log = torch.arange(16, dtype=torch.float32).div(8).bfloat16()
    assert torch.equal(_log_decay(-decay_log.float().exp()).bfloat16(), decay_log)
    with pytest.raises(QuantizationError, match="strictly negative"):
        _log_decay(torch.tensor([0.0]))
    first = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    second = first + 1000
    inv = _bundle(tmp_path / "vision.gguf", {"p0": first, "p1": second})
    weight = LogicalWeight("unchanged.patch.weight", (2, 3, 2, 4, 4), "BF16", "vision")
    view = GGUFWeightView(weight, tuple(GGUFPart(inv, name, reshape=(2, 3, 1, 4, 4))
                                       for name in ("p0", "p1")), join_axis=2)
    validate_gguf_coverage((inv,), (view,))
    assert torch.equal(view.read(), torch.from_numpy(np.stack((first, second), axis=2)))


def test_shared_materialization_reads_existing_owner_spans(tmp_path):
    from gguf import GGMLQuantizationType

    from cubic_serve.models.gguf_weights import GGUFCheckpointWeights
    from cubic_serve.quantization.extensions.ggml.extension import EXTENSION
    from cubic_serve.quantization.registry import extensions, register

    if not any(e.format_id == EXTENSION.format_id for e in extensions()):
        register(EXTENSION)
    (tmp_path / "config.json").write_text("{}")
    original = np.arange(6 * 72, dtype=np.uint8).reshape(6, 72)
    inv = _bundle(tmp_path / "w.gguf", {"w": original}, raw_type=GGMLQuantizationType.IQ4_NL)
    weight = LogicalWeight("existing.proj.weight", (6, 128), "BF16", "language", "linear")
    source = GGUFCheckpointWeights.from_views(tmp_path, (inv,), (
        GGUFWeightView(weight, (GGUFPart(inv, "w"),)),))
    source.validate_for(tmp_path, (weight,))
    bound = source.weights[0]
    # A TP input boundary inside a scale block retains both covering blocks;
    # the logical offset is metadata, never a requantized or expanded matrix.
    result = source.materialize(bound, device="cpu", axis=1, spans=(AxisSpan(40, 104),))
    assert result.shape == (6, 64)
    assert result.encoding.metadata == "IQ4_NL:8"
    assert torch.equal(result.payload["weight_packed"], torch.from_numpy(original[:, 18:72].copy()))
    nested = result.columns(9, 50)
    assert nested.shape == (6, 41)
    assert nested.encoding.metadata == "IQ4_NL:17"
    rows = source.materialize_many(((bound, 0, (AxisSpan(1, 3), AxisSpan(4, 6))),),
                                   device="cpu", compute_dtype=torch.bfloat16)
    assert torch.equal(rows[weight.name].payload["weight_packed"],
                       torch.from_numpy(original[[1, 2, 4, 5]]))
    with pytest.raises(QuantizationError, match="one contiguous span"):
        source.materialize(bound, device="cpu", axis=1,
                           spans=(AxisSpan(0, 17), AxisSpan(96, 128)))
    with pytest.raises(QuantizationError, match="complete admission"):
        source.payload_view().validate_for(tmp_path, (weight,))
