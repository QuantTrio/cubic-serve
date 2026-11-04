import json
import os
import tempfile
import unittest
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from cubic_serve.models.checkpoint import CheckpointInventoryError, inspect_checkpoint
from cubic_serve.models.parallel_weights import AxisSpan
from cubic_serve.models.quantization import (
    ActivationFormat,
    QuantizationConfig,
    QuantizationError,
    QuantizationScheme,
    WeightFormat,
)
from cubic_serve.models.quantized_checkpoint import (
    CheckpointWeights,
    EncodedWeight,
    LogicalWeight,
    QuantizedCheckpoint,
    QuantizedMatrix,
    StackedMatrixSlice,
)
from cubic_serve.ops.eager.quantization import decode_weight
from tests.models.test_quantization import fp8_config


class BatchSourceTransactionTests(unittest.TestCase):
    def test_borrowed_readers_keep_source_checks_at_the_batch_boundary(self):
        import cubic_serve.models.parallel_weights as paths
        from tests.ops.test_int_quantization import fixture

        for integer, partition in ((False, "language"), (False, "vision"),
                                   (True, "language"), (True, "vision")):
            with self.subTest(integer=integer, partition=partition), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                matrix, _ = fixture(8, n=16, k=64, group=32, sym=True)
                names = ("first", "second", "third")
                config = {} if not integer else {"quantization_config": {
                    "quant_method": "compressed-tensors", "format": "pack-quantized",
                    "config_groups": {"integer": {"targets": list(names), "weights": {
                        "num_bits": 8, "type": "int", "strategy": "group",
                        "group_size": 32, "symmetric": True,
                    }}},
                }}
                (root / "config.json").write_text(json.dumps(config))
                payload = {name + "." + key: value.clone() for name in names
                           for key, value in (matrix.payload.items() if integer else
                                              (("weight", torch.ones(16, 64, dtype=torch.bfloat16)),))}
                shard = root / "model.safetensors"
                save_file(payload, shard)
                logical = tuple(LogicalWeight(name + ".weight", (16, 64), "BF16",
                                               partition, "linear") for name in names)
                plan = CheckpointWeights.admit(root, logical, verify_payloads=False,
                                               require_quantization=integer)
                requests = tuple((w, 0, (AxisSpan(0, 16),)) for w in plan.weights)
                with patch.object(paths, "confined_checkpoint_path",
                                  wraps=paths.confined_checkpoint_path) as resolve, patch(
                    "cubic_serve.models.quantized_checkpoint.confined_checkpoint_path", resolve,
                ):
                    loaded = plan.materialize_many(requests, device="cpu",
                                                   compute_dtype=torch.bfloat16)
                self.assertEqual(set(loaded), {w.name for w in logical})
                self.assertEqual(resolve.call_count, 3)  # attest, open, pre-return check
                original = CheckpointWeights.materialize
                touched = False

                def mutate_after_read(self, *args, **kwargs):
                    nonlocal touched
                    value = original(self, *args, **kwargs)
                    if not touched:
                        stamp = shard.stat()
                        os.utime(shard, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1000000))
                        touched = True
                    return value

                progress = []
                with patch.object(CheckpointWeights, "materialize", mutate_after_read), \
                        self.assertRaisesRegex(QuantizationError, "changed during rank"):
                    plan.materialize_many(requests, device="cpu", compute_dtype=torch.bfloat16,
                                          progress=lambda done, total: progress.append((done, total)))
                self.assertNotIn((len(requests), len(requests)), progress)


def test_host_allocator_defaults_preserve_user_settings_and_host_resource_limits():
    from cubic_serve.models.quantized_checkpoint import _configure_host_allocator

    cases = (
        ("max_split_size_mb:256", 32, True,
         "max_split_size_mb:256,pinned_use_cuda_host_register:True,pinned_num_register_threads:8"),
        ("", 16, True, "pinned_use_cuda_host_register:True,pinned_num_register_threads:4"),
        ("", 8, True, None),
        ("", 32, False, None),
        ("pinned_use_cuda_host_register:False", 32, True, None),
        ("pinned_num_register_threads:1", 32, True, None),
    )
    for config, cpus, supported, expected in cases:
        _configure_host_allocator.cache_clear()
        try:
            with patch("torch._C._accelerator_getAllocatorSettings", return_value=config), \
                    patch("torch._C._accelerator_setAllocatorSettings") as update, \
                    patch("os.sched_getaffinity", return_value=set(range(cpus))), \
                    patch("torch.distributed.is_initialized", return_value=True), \
                    patch("torch.distributed.get_world_size", return_value=4), \
                    patch("torch.cuda.device_count", return_value=2), \
                    patch("cubic_serve.ops.cuda.embedding.immutable.host_registration_supported",
                          return_value=supported):
                _configure_host_allocator()
                _configure_host_allocator()
                if expected is None:
                    update.assert_not_called()
                else:
                    update.assert_called_once_with(expected)
        finally:
            _configure_host_allocator.cache_clear()


def test_host_storage_fallback_warns_once_on_stdout_and_only_for_unsupported_device(capsys):
    from cubic_serve.models.quantized_checkpoint import (
        _pinned_host_storage_supported,
        use_pinned_host_storage,
    )

    _pinned_host_storage_supported.cache_clear()
    try:
        with patch("cubic_serve.ops.cuda.embedding.immutable.mapped_host_memory_supported",
                   return_value=False) as query:
            assert not use_pinned_host_storage(torch.device("cuda:3"))
            assert not use_pinned_host_storage(torch.device("cuda:3"))
            assert not use_pinned_host_storage(torch.device("cpu"))
            query.assert_called_once_with(torch.device("cuda:3"))
        output = capsys.readouterr()
        assert output.out.count("WARNING:") == 1
        assert "cuda:3" in output.out and "falling back to GPU storage" in output.out
        assert not output.err
        _pinned_host_storage_supported.cache_clear()
        with patch("cubic_serve.ops.cuda.embedding.immutable.mapped_host_memory_supported",
                   return_value=True):
            assert use_pinned_host_storage(torch.device("cuda:1"))
        assert not capsys.readouterr().out
        _pinned_host_storage_supported.cache_clear()
        with patch("torch.cuda.current_device", side_effect=[1, 3]), patch(
            "cubic_serve.ops.cuda.embedding.immutable.mapped_host_memory_supported",
            side_effect=lambda device: device.index == 1,
        ):
            assert use_pinned_host_storage(torch.device("cuda"))
            assert not use_pinned_host_storage(torch.device("cuda"))
        assert "cuda:3" in capsys.readouterr().out
        _pinned_host_storage_supported.cache_clear()
        with patch("cubic_serve.ops.cuda.embedding.immutable.mapped_host_memory_supported",
                   side_effect=RuntimeError("driver query failed")), \
                unittest.TestCase().assertRaisesRegex(RuntimeError, "driver query failed"):
            use_pinned_host_storage(torch.device("cuda:1"))
        assert not capsys.readouterr().out
    finally:
        _pinned_host_storage_supported.cache_clear()


@unittest.skipUnless(os.environ.get("CUBIC_TEST_QUANT_CUDA") == "1", "explicit CUDA audit")
class HostEmbeddingLoadingTests(unittest.TestCase):
    def test_int_owner_transport_keeps_execution_device_and_original_lookup_on_fallback(self):
        from cubic_serve.models.quantized_checkpoint import _pinned_host_storage_supported
        from cubic_serve.ops.cuda.embedding.token import bind_scaled_sharded_embedding
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
        from tests.ops.test_int_quantization import fixture

        matrix, expected = fixture(8, n=16, k=64, group=32, sym=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"quantization_config": {
                "quant_method": "compressed-tensors", "format": "pack-quantized",
                "config_groups": {"integer": {"targets": ["lookup"], "weights": {
                    "num_bits": 8, "type": "int", "strategy": "group",
                    "group_size": 32, "symmetric": True,
                }}},
            }}))
            save_file({"lookup." + key: value for key, value in matrix.payload.items()},
                      root / "model.safetensors")
            logical = LogicalWeight("lookup.weight", matrix.shape, "BF16", "language",
                                    "embedding", storage="pinned_host")
            plan = CheckpointWeights.admit(root, (logical,), verify_payloads=False)
            device = torch.device("cuda:0")
            try:
                for supported in (True, False):
                    _pinned_host_storage_supported.cache_clear()
                    with patch(
                        "cubic_serve.ops.cuda.embedding.immutable.mapped_host_memory_supported",
                        return_value=supported,
                    ):
                        loaded = plan.materialize_many(
                            ((plan.weights[0], 0, (AxisSpan(0, 16),)),), device=device,
                            packed_device="cpu", compute_dtype=torch.bfloat16,
                        )[logical.name]
                        assert all(t.is_pinned() == supported for t in loaded.payload.values())
                        for key, value in loaded.payload.items():
                            assert torch.equal(value, matrix.payload[key])
                        if not supported:
                            loaded = PackedLinearOperators().materialize_weight(loaded, device)
                        ids = torch.tensor([[0], [15], [7], [3]], device=device)
                        output = torch.empty((4, 64), device=device, dtype=torch.bfloat16)
                        op = bind_scaled_sharded_embedding(
                            ids, ((0, loaded),), torch.ones(1, device=device),
                            torch.ones(4, device=device, dtype=torch.int32), output,
                        )
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            op.launch()
                        graph.replay()
                        torch.cuda.synchronize()
                        assert torch.equal(output.cpu(), expected[ids.cpu().flatten()].bfloat16())
            finally:
                _pinned_host_storage_supported.cache_clear()

    def test_unsupported_device_preserves_original_rows_on_all_loading_paths(self):
        from cubic_serve.models.quantized_checkpoint import _pinned_host_storage_supported
        from cubic_serve.runtime.quantized import QuantizedWeights

        _pinned_host_storage_supported.cache_clear()
        try:
            with tempfile.TemporaryDirectory() as directory, patch(
                "cubic_serve.ops.cuda.embedding.immutable.mapped_host_memory_supported",
                return_value=False,
            ), patch("builtins.print") as warning:
                root = Path(directory)
                (root / "config.json").write_text("{}")
                original = torch.arange(96).reshape(16, 6).to(torch.bfloat16)
                save_file({"lookup.weight": original}, root / "model.safetensors")
                logical = LogicalWeight("lookup.weight", (16, 6), "BF16", "language",
                                        "embedding", storage="pinned_host")
                plan = CheckpointWeights.admit(root, (logical,), require_quantization=False,
                                               verify_payloads=False)
                spans = (AxisSpan(3, 11),)
                direct = plan.materialize(plan.weights[0], spans=spans, device="cuda:0")
                owner = OwnerBinding(logical.name, 0, spans, logical.shape, (8, 6))
                sliced = QuantizedWeights(plan, device="cuda:0", compute_dtype=torch.bfloat16,
                                         owner_bindings=(owner,))[logical.name]
                whole = QuantizedWeights(plan, device="cuda:0", compute_dtype=torch.bfloat16)
                for value in (direct, sliced, whole[logical.name]):
                    assert value.is_cuda
                    expected = original if value.shape == original.shape else original[3:11]
                    assert torch.equal(value.cpu(), expected)
                assert whole.resident_bytes == original.numel() * original.element_size()
                warning.assert_called_once()
        finally:
            _pinned_host_storage_supported.cache_clear()

    def test_declared_owner_loads_directly_to_host_without_cuda_payload(self):
        from cubic_serve.runtime.quantized import QuantizedWeights

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            original = torch.arange(96).reshape(16, 6).to(torch.bfloat16)
            save_file({"lookup.weight": original}, root / "model.safetensors")
            logical = LogicalWeight("lookup.weight", (16, 6), "BF16", "language",
                                    "embedding", storage="pinned_host")
            plan = CheckpointWeights.admit(root, (logical,), require_quantization=False,
                                           verify_payloads=False)
            value = plan.materialize(plan.weights[0], axis=0, spans=(AxisSpan(3, 11),),
                                     device="cuda:0")
            assert value.device.type == "cpu" and value.is_pinned()
            assert torch.equal(value, original[3:11])
            weights = QuantizedWeights(plan, device="cuda:0", compute_dtype=torch.bfloat16)
            assert weights["lookup.weight"].is_pinned() and weights.resident_bytes == 0
            # The independent eager path gathers original rows before transferring them.
            from cubic_serve.ops.eager.quantization import QuantizedReferenceOperators
            ids = torch.tensor([[0, 15], [7, 3]], device="cuda")
            actual = QuantizedReferenceOperators().embedding(
                ids, weights["lookup.weight"], dtype=torch.bfloat16)
            assert torch.equal(actual.cpu(), original[ids.cpu()])


class QuantizedCheckpointTests(unittest.TestCase):
    def test_startup_reuse_preserves_plan_and_rejects_changed_sources(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as other:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            value = torch.ones(8, 6, dtype=torch.bfloat16)
            save_file({"projection.weight": value}, root / "weights.safetensors")
            index = root / "model.safetensors.index.json"
            original_index = json.dumps({"metadata": {"total_size": 96},
                                         "weight_map": {"projection.weight": "weights.safetensors"}})
            index.write_text(original_index)
            logical = (LogicalWeight("projection.weight", (8, 6), "BF16", "language", "linear"),)
            plan = CheckpointWeights.admit(root, logical, verify_payloads=False,
                                           require_quantization=False)
            with patch("cubic_serve.models.quantized_checkpoint.inspect_checkpoint",
                       side_effect=AssertionError("must reuse admitted headers")):
                plan.validate_for(root, logical)
            import pickle
            view = pickle.loads(pickle.dumps(plan.payload_view()))
            self.assertEqual(view.semantic_digest, plan.semantic_digest)
            self.assertEqual(view.inventory.inventory_digest, plan.inventory.inventory_digest)
            self.assertFalse(hasattr(view.inventory, "tensors"))
            self.assertTrue(torch.equal(view.materialize(view.weights[0], device="cpu"), value))
            with self.assertRaisesRegex(QuantizationError, "cannot replace complete"):
                view.validate_for(root, logical)
            with self.assertRaisesRegex(QuantizationError, "another model root"):
                plan.validate_for(other, logical)
            with self.assertRaisesRegex(QuantizationError, "logical weight plan"):
                plan.validate_for(root, (replace(logical[0], shape=(4, 6)),))
            (root / "config.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(QuantizationError, "metadata changed"):
                plan.validate_for(root, logical)
            (root / "config.json").write_text("{}")
            index.write_text(original_index + " ")
            with self.assertRaisesRegex(QuantizationError, "metadata changed"):
                plan.validate_for(root, logical)
            index.write_text(original_index)
            save_file({"projection.weight": value * 2}, root / "weights.safetensors")
            with self.assertRaisesRegex(QuantizationError, "shard changed"):
                plan.validate_for(root, logical)
            with self.assertRaisesRegex(QuantizationError, "shard changed"):
                view.materialize(view.weights[0], device="cpu")

    def test_payload_view_preserves_partition_and_global_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            value = torch.arange(48, dtype=torch.bfloat16).reshape(8, 6)
            save_file({"vision.weight": value}, root / "vision.safetensors")
            save_file({"language.weight": value + 1}, root / "language.safetensors")
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 192}, "weight_map": {
                    "vision.weight": "vision.safetensors", "language.weight": "language.safetensors"}}))
            logical = tuple(LogicalWeight(name + ".weight", (8, 6), "BF16", name, "linear")
                            for name in ("vision", "language"))
            plan = CheckpointWeights.admit(root, logical, verify_payloads=False,
                                           require_quantization=False)
            view = plan.payload_view(partitions=("vision",))
            self.assertEqual(len(view.weights), 1)
            self.assertIs(view.weights[0], plan.weights[1])
            self.assertEqual(view.semantic_digest, plan.semantic_digest)
            self.assertEqual(view.inventory.inventory_digest, plan.inventory.inventory_digest)
            self.assertEqual([file.relative_path for file in view.inventory.shard_files],
                             ["vision.safetensors"])
            self.assertTrue(torch.equal(view.materialize(view.weights[0]), value))
            with self.assertRaisesRegex(QuantizationError, "another checkpoint plan"):
                view.materialize(plan.weights[0])
            with self.assertRaisesRegex(QuantizationError, "no admitted logical weights"):
                plan.payload_view(partitions=("absent",))

    def test_plain_checkpoint_uses_the_same_closed_payload_and_owner_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            value = torch.arange(48, dtype=torch.bfloat16).reshape(8, 6)
            save_file({"projection.weight": value}, root / "model.safetensors")
            logical = (LogicalWeight("projection.weight", (8, 6), "BF16", "language", "linear"),)
            with self.assertRaisesRegex(QuantizationError, "requires declared quantization"):
                QuantizedCheckpoint.admit(root, logical, verify_payloads=False)
            plan = CheckpointWeights.admit(
                root, logical, verify_payloads=False, require_quantization=False
            )
            self.assertIsNone(plan.quantization)
            self.assertIsNone(plan.weights[0].encoding)
            actual = plan.materialize(
                plan.weights[0], compute_dtype=torch.bfloat16,
                axis=1, spans=(AxisSpan(2, 5),),
            )
            self.assertTrue(torch.equal(actual, value[:, 2:5]))
            save_file(
                {"projection.weight": value, "extra": value.clone()}, root / "model.safetensors"
            )
            with self.assertRaisesRegex(QuantizationError, "unowned tensors"):
                CheckpointWeights.admit(
                    root, logical, verify_payloads=False, require_quantization=False
                )

    def test_stacked_encoded_projection_and_tp_views(self):
        # Every Cubic bit width and metadata ABI uses the same loader geometry.
        for matrix in PackedPartitionTests().matrices():
            if matrix.encoding.format != WeightFormat.CUBIC:
                continue
            with (
                self.subTest(bits=matrix.encoding.bits, metadata=matrix.encoding.metadata),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                (root / "config.json").write_text(json.dumps({"quantization_config": fp8_config()}))
                physical = {
                    "experts.gate_up_" + key.removeprefix("weight_"): torch.stack((value, value))
                    for key, value in matrix.payload.items()
                }
                save_file(physical, root / "model.safetensors")
                logical = tuple(
                    LogicalWeight(
                        f"experts.{expert}.{part}.weight",
                        (8, 259),
                        "BF16",
                        "language",
                        "linear",
                        stacked_source=StackedMatrixSlice(
                            "experts.gate_up",
                            (2, 16, 259),
                            expert,
                            AxisSpan(start, start + 8),
                            (8, 8),
                        ),
                    )
                    for expert in range(2)
                    for part, start in (("gate", 0), ("up", 8))
                )
                with patch.object(QuantizationConfig, "resolve", return_value=matrix.encoding):
                    plan = QuantizedCheckpoint.admit(root, logical, verify_payloads=False)
                    with self.assertRaisesRegex(QuantizationError, "unowned|missing owners"):
                        QuantizedCheckpoint.admit(root, logical[:-1], verify_payloads=False)
                from cubic_serve.runtime.quantized import QuantizedWeights

                loaded = QuantizedWeights(plan, device="cpu", compute_dtype=torch.float16)
                for weight in plan.weights:
                    view = weight.stacked_source
                    expected = matrix.rows(view.rows.start, view.rows.stop)
                    for key, actual in loaded[weight.logical.name].payload.items():
                        self.assertTrue(torch.equal(actual, expected.payload[key]), key)
                    for key, actual in plan.read(weight).items():
                        self.assertTrue(torch.equal(actual, expected.payload[key]), key)
                    for axis, span in (
                        (0, AxisSpan(2, 6)),
                        (1, AxisSpan(32, 64)),
                        (1, AxisSpan(8, 24)),
                    ):
                        want = (
                            expected.rows(span.start, span.stop)
                            if axis == 0
                            else expected.columns(span.start, span.stop)
                        )
                        for dtype in (torch.float16, torch.bfloat16):
                            actual = plan.materialize(
                                weight, compute_dtype=dtype, axis=axis, spans=(span,)
                            )
                            self.assertEqual(actual.shape, want.shape)
                            for key in want.payload:
                                self.assertTrue(
                                    torch.equal(actual.payload[key], want.payload[key]), key
                                )
                                self.assertEqual(actual.payload[key].dtype, want.payload[key].dtype)

    def test_stacked_plain_views_close_ownership_and_compose_tp_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "quantization_config": fp8_config(),
                    }
                )
            )
            tensor = torch.arange(2 * 8 * 6, dtype=torch.float16).reshape(2, 8, 6)
            save_file({"experts.gate_up": tensor}, root / "model.safetensors")
            logical = tuple(
                LogicalWeight(
                    f"experts.{e}.{part}.weight",
                    (4, 6),
                    "BF16",
                    "language",
                    "tensor",
                    stacked_source=StackedMatrixSlice(
                        "experts.gate_up", (2, 8, 6), e, AxisSpan(start, start + 4)
                    ),
                )
                for e in range(2)
                for part, start in (("gate", 0), ("up", 4))
            )
            plan = QuantizedCheckpoint.admit(root, logical, verify_payloads=False)
            progress = []
            requests = tuple(
                (weight, 0, (AxisSpan(0, weight.logical.shape[0]),))
                for weight in plan.weights
            )
            loaded = plan.materialize_many(
                requests, device="cpu", compute_dtype=torch.float16,
                progress=lambda done, total: progress.append((done, total)),
            )
            self.assertEqual(progress[0], (0, len(requests)))
            self.assertEqual(progress[-1], (len(requests), len(requests)))
            self.assertEqual(set(loaded), {w.logical.name for w in plan.weights})
            failed_progress = []
            with patch.object(
                QuantizedCheckpoint, "materialize", side_effect=RuntimeError("read failed")
            ), self.assertRaisesRegex(RuntimeError, "read failed"):
                plan.materialize_many(
                    requests, device="cpu", compute_dtype=torch.float16,
                    progress=lambda done, total: failed_progress.append((done, total)),
                )
            self.assertEqual(failed_progress, [(0, len(requests))])
            for weight in plan.weights:
                view = weight.stacked_source
                expected = tensor[view.index, view.rows.start : view.rows.stop]
                self.assertTrue(torch.equal(plan.read(weight)["weight"], expected))
                for axis in (0, 1):
                    actual = plan.materialize(
                        weight, compute_dtype=torch.float16, axis=axis, spans=(AxisSpan(1, 3),)
                    )
                    want = expected[1:3] if axis == 0 else expected[:, 1:3]
                    self.assertTrue(torch.equal(actual, want))
                    self.assertEqual(actual.dtype, torch.float16)
            with self.assertRaisesRegex(QuantizationError, "unowned|missing owners"):
                QuantizedCheckpoint.admit(root, logical[:-1], verify_payloads=False)
            duplicate = replace(logical[0], name="duplicate.weight")
            with self.assertRaisesRegex(QuantizationError, "overlapping"):
                QuantizedCheckpoint.admit(root, (*logical, duplicate), verify_payloads=False)

    def fixture(self, root, *, extra=False):
        (root / "config.json").write_text(json.dumps({"quantization_config": fp8_config()}))
        tensors = {
            "model.embed_tokens.weight": torch.ones((8, 4)).to(torch.float8_e4m3fn),
            "model.embed_tokens.weight_scale": torch.ones((8, 1)),
            "lm_head.weight": torch.ones((8, 4)).to(torch.float8_e4m3fn),
            "lm_head.weight_scale": torch.ones((8, 1)),
        }
        if extra:
            tensors["unowned.weight"] = torch.ones(1)
        save_file(tensors, root / "model-00001-of-00001.safetensors")
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 999999},
                    "weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors},
                }
            )
        )
        return (
            LogicalWeight("model.embed_tokens.weight", (8, 4), "BF16", "language", "embedding"),
            LogicalWeight("lm_head.weight", (8, 4), "BF16", "lm_head", "linear"),
        )

    def test_advisory_size_does_not_relax_physical_closure_or_strict_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logical = self.fixture(root)
            plan = QuantizedCheckpoint.admit(root, logical, verify_payloads=False)
            self.assertEqual(plan.declared_total_size, 999999)
            self.assertEqual(plan.inventory.total_tensor_bytes, 128)
            self.assertEqual(plan.weights[1].encoding.activation, ActivationFormat.IDENTITY)
            self.assertEqual(plan.weights[0].encoding.activation, ActivationFormat.FP8_TOKEN)
            self.assertEqual(plan.read(plan.weights[0])["weight"].dtype, torch.float8_e4m3fn)
            with self.assertRaisesRegex(CheckpointInventoryError, "total_size"):
                inspect_checkpoint(root, verify_payloads=False)
            other = QuantizedCheckpoint.admit(root, logical, verify_payloads=False)
            with self.assertRaisesRegex(QuantizationError, "another checkpoint"):
                plan.read(other.weights[0])

    def test_unowned_payload_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logical = self.fixture(root, extra=True)
            with self.assertRaisesRegex(QuantizationError, "unowned"):
                QuantizedCheckpoint.admit(root, logical, verify_payloads=False)


class PackedPartitionTests(unittest.TestCase):
    def matrices(self):
        n, k = 16, 256
        generator = torch.Generator().manual_seed(91)
        yield QuantizedMatrix(
            {
                "qweight": torch.randint(
                    -(2**31), 2**31 - 1, (k, n // 8), generator=generator, dtype=torch.int32
                ),
                "qzeros": torch.full((2, n // 8), 0x77777777, dtype=torch.int32),
                "scales": torch.arange(1, 33, dtype=torch.float16).reshape(2, n) / 128,
            },
            QuantizationScheme(WeightFormat.AWQ, 4, 128),
            (n, k),
        )
        yield QuantizedMatrix(
            {
                "weight": torch.randn(n, k, generator=generator).to(torch.float8_e4m3fn),
                "weight_scale": torch.arange(1, n + 1).float().reshape(n, 1) / 16,
            },
            QuantizationScheme(WeightFormat.FP8, 8, 1, activation=ActivationFormat.FP8_TOKEN),
            (n, k),
        )
        yield QuantizedMatrix(
            {
                "weight_packed": torch.randint(
                    0, 256, (n, k // 2), generator=generator, dtype=torch.uint8
                ),
                "weight_scale": torch.arange(1, 17).repeat(n, 1).to(torch.float8_e4m3fn),
                "weight_global_scale": torch.tensor([32.0]),
                "input_global_scale": torch.tensor([2.0]),
            },
            QuantizationScheme(WeightFormat.NVFP4, 4, 16, activation=ActivationFormat.NVFP4_LOCAL),
            (n, k),
        )
        for bits in range(1, 9):
            for metadata in (
                "float32-scale-float16-ab",
                "int8-scale-int4-ab-global-fp32",
                "e5m9-scale-curve2-tensor-fp16",
            ):
                # Include a partial final group and a non-byte-aligned final code.
                width = 259
                payload = {
                    "weight_packed": torch.randint(
                        0, 256, (n, (width * bits + 7) // 8), generator=generator, dtype=torch.uint8
                    )
                }
                if metadata.startswith("float32"):
                    payload.update(
                        weight_scale=torch.rand(n // 2, 9, generator=generator),
                        weight_a=torch.full((n // 2, 9), 0.75, dtype=torch.float16),
                        weight_b=torch.full((n // 2, 9), 0.125, dtype=torch.float16),
                    )
                elif metadata.startswith("int8"):
                    payload.update(
                        weight_scale=torch.randint(
                            1, 100, (n // 2, 9), generator=generator, dtype=torch.int8
                        ),
                        weight_ab=torch.full((n // 2, 9), 0x23, dtype=torch.uint8),
                        weight_scale_global=torch.tensor([0.01, 0.02]),
                        weight_a_global=torch.tensor([0.125, 0.0625]),
                        weight_b_global=torch.tensor([0.03125, 0.015625]),
                    )
                else:
                    payload.update(
                        weight_metadata=(
                            torch.randint(0, 0x3C00, (n // 2, 9), generator=generator)
                            | (torch.randint(0, 4, (n // 2, 9), generator=generator) << 14)
                        ).to(torch.uint16),
                        weight_curve_a=torch.full((2, 4), 0.75, dtype=torch.float16),
                        weight_curve_b=torch.full((2, 4), 0.125, dtype=torch.float16),
                    )
                yield QuantizedMatrix(
                    payload,
                    QuantizationScheme(
                        WeightFormat.CUBIC, bits, 32, group_out=2, metadata=metadata
                    ),
                    (n, width),
                    (8, 8),
                )

    @staticmethod
    def decoded(matrix):
        return decode_weight(
            matrix.payload, matrix.encoding, matrix.shape, partitions=matrix.partitions or None
        )

    def test_input_and_output_shards_preserve_original_decoded_values(self):
        for matrix in self.matrices():
            with self.subTest(encoding=matrix.encoding):
                expected = self.decoded(matrix)
                halves = (matrix.columns(0, 128), matrix.columns(128, matrix.shape[1]))
                self.assertTrue(
                    torch.equal(torch.cat([self.decoded(x) for x in halves], dim=1), expected)
                )
                for part in halves:
                    self.assertTrue(
                        torch.equal(self.decoded(part.rows(8, 16)), self.decoded(part)[8:16])
                    )
                # Original Q/K/V partition globals survive K slicing verbatim.
                for key, value in matrix.payload.items():
                    if key.endswith("_global") or key in (
                        "weight_global_scale",
                        "input_global_scale",
                        "weight_curve_a",
                        "weight_curve_b",
                    ):
                        self.assertEqual(halves[1].payload[key].data_ptr(), value.data_ptr())
                self.assertEqual(halves[1].encoding, matrix.encoding)

    def test_single_original_group_subshards_preserve_decoded_weights(self):
        for matrix in self.matrices():
            if matrix.encoding.format != WeightFormat.CUBIC:
                continue
            original = self.decoded(matrix)
            for start, stop in ((0, 16), (8, 24), (40, 56), (256, 259)):
                part = matrix.columns(start, stop)
                self.assertEqual(part.encoding, matrix.encoding)
                self.assertTrue(torch.equal(self.decoded(part), original[:, start:stop]))

    def test_misaligned_group_origin_fails_before_execution(self):
        for matrix in self.matrices():
            if matrix.encoding.format is WeightFormat.FP8:
                part = matrix.columns(3, 131)
                self.assertTrue(torch.equal(self.decoded(part), self.decoded(matrix)[:, 3:131]))
            else:
                with self.assertRaisesRegex(QuantizationError, "original scale groups"):
                    matrix.columns(1, 129)


@dataclass(frozen=True)
class OwnerBinding:
    tensor_name: str
    shard_axis: int
    spans: tuple[AxisSpan, ...]
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]


class QuantizedOwnerLoadingTests(unittest.TestCase):
    def test_dense_and_moe_owner_plans_across_tp_pp_ep(self):
        from cubic_serve.contracts.execution import ParallelTopology
        from cubic_serve.models.qwen3_5 import parse_qwen3_5_config
        from cubic_serve.models.qwen3_5.weights import quantized_owner_slices as dense_owners
        from cubic_serve.models.qwen3_5_moe import parse_qwen3_5_moe_config
        from cubic_serve.models.qwen3_5_moe.weights import quantized_owner_slices as moe_owners
        from tests.models.qwen3_5.test_config import _fixture as dense_fixture
        from tests.models.qwen3_5_moe.test_config import _fixture as moe_fixture

        for config, planner in (
            (parse_qwen3_5_config(dense_fixture()), dense_owners),
            (parse_qwen3_5_moe_config(moe_fixture()), moe_owners),
        ):
            is_moe = config.graph.text.moe is not None
            for tp, pp in ((1, 1), (2, 1), (4, 1), (1, 2), (1, 4), (2, 2)):
                for ep in [e for e in range(1, tp + 1) if tp % e == 0] if is_moe else [1]:
                    topology = ParallelTopology.single_node(
                        topology_id=f"owner-tp{tp}-pp{pp}-ep{ep}",
                        device_indices=tuple(range(tp * pp)),
                        tp_size=tp,
                        pp_size=pp,
                        ep_size=ep,
                    )
                    owners = [planner(config, topology, rank) for rank in range(tp * pp)]
                    per_stage = len(config.graph.text.layer_schedule) // pp
                    for rank, bindings in enumerate(owners):
                        stage = rank // tp
                        for b in bindings:
                            if b.tensor_name.startswith("model.language_model.layers."):
                                layer = int(b.tensor_name.split(".layers.")[1].split(".")[0])
                                self.assertEqual(layer // per_stage, stage)
                            if b.tensor_name == "model.language_model.embed_tokens.weight":
                                self.assertEqual(stage, 0)
                            if b.tensor_name == "lm_head.weight":
                                self.assertEqual(stage, pp - 1)
                            if ".experts." in b.tensor_name:
                                expert = int(b.tensor_name.split(".experts.")[1].split(".")[0])
                                expert_owner, expert_lane = topology.expert_coordinates(rank)
                                self.assertEqual(
                                    expert // (config.graph.text.moe.expert_count // ep),
                                    expert_owner,
                                )
                                axis = 1 if b.tensor_name.endswith("down_proj.weight") else 0
                                self.assertEqual(b.shard_axis, axis)
                                self.assertEqual(
                                    b.local_shape[axis] * topology.expert_tensor_size,
                                    b.global_shape[axis],
                                )
                                self.assertEqual(
                                    b.spans[0].start, expert_lane * b.local_shape[axis]
                                )
                    # Every GDN shard keeps independent Q/K/V output spans.
                    name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
                    qkv = [
                        next(b for b in bindings if b.tensor_name == name)
                        for bindings in owners[:tp]
                    ]
                    for part in range(3):
                        spans = [b.spans[part] for b in qkv]
                        self.assertTrue(all(a.stop == b.start for a, b in pairwise(spans)))
                    if pp == 1:
                        mtp = planner(config, topology, 0, include_mtp=True)
                        self.assertTrue(any(b.tensor_name.startswith("mtp.") for b in mtp))
                    else:
                        with self.assertRaisesRegex(ValueError, "MTP"):
                            planner(config, topology, 0, include_mtp=True)

    def fixture(self, root, matrix):
        tensors = {f"matrix.{k}": v for k, v in matrix.payload.items()}
        tensors["norm.weight"] = torch.arange(16, dtype=torch.bfloat16)
        tensors["decay.weight"] = torch.arange(4, dtype=torch.float32)
        shard = "model-00001-of-00001.safetensors"
        save_file(tensors, root / shard)
        (root / "config.json").write_text("{}")
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "total_size": sum(t.numel() * t.element_size() for t in tensors.values())
                    },
                    "weight_map": {name: shard for name in tensors},
                }
            )
        )
        encoded = EncodedWeight(
            LogicalWeight(
                "matrix.weight",
                matrix.shape,
                "BF16",
                "language",
                "linear",
                output_partitions=matrix.partitions,
            ),
            matrix.encoding,
            tuple((k, f"matrix.{k}", shard) for k in matrix.payload),
        )
        plain = EncodedWeight(
            LogicalWeight("norm.weight", (16,), "BF16", "language"),
            None,
            (("weight", "norm.weight", shard),),
        )
        decay = EncodedWeight(
            LogicalWeight("decay.weight", (4,), "F32", "language"),
            None,
            (("weight", "decay.weight", shard),),
        )
        plan = QuantizedCheckpoint(
            inspect_checkpoint(root, verify_payloads=False),
            QuantizationConfig.from_model_config({"quantization_config": fp8_config()}),
            (decay, encoded, plain),
            "test-owned-spans",
            None,
        )
        return plan, encoded, plain, decay

    def test_original_codes_metadata_and_plain_dtype_across_owner_spans(self):
        from cubic_serve.runtime.quantized import QuantizedWeights

        for matrix in PackedPartitionTests().matrices():
            with self.subTest(encoding=matrix.encoding), tempfile.TemporaryDirectory() as directory:
                plan, encoded, plain, decay = self.fixture(Path(directory), matrix)
                for axis, spans in (
                    (0, (AxisSpan(8, 16),)),
                    (1, (AxisSpan(128, matrix.shape[1]),)),
                    (0, (AxisSpan(0, 8), AxisSpan(8, 16))),
                ):
                    loaded = plan.materialize(encoded, axis=axis, spans=spans)
                    expected = PackedPartitionTests.decoded(matrix)
                    expected = torch.cat(
                        [expected.narrow(axis, span.start, span.size) for span in spans], dim=axis
                    )
                    self.assertTrue(torch.equal(PackedPartitionTests.decoded(loaded), expected))
                    self.assertEqual(loaded.encoding, matrix.encoding)
                    for value in loaded.payload.values():
                        self.assertTrue(value.is_contiguous())
                # PP selection and TP spans use the same loader; no unowned weight is loaded.
                owner = OwnerBinding(
                    "matrix.weight", 0, (AxisSpan(8, 16),), matrix.shape, (8, matrix.shape[1])
                )
                resident = QuantizedWeights(
                    plan, device="cpu", compute_dtype=torch.float16, owner_bindings=(owner,), rank=2
                )
                self.assertEqual(resident.rank, 2)
                self.assertEqual(resident.tensor_count, 1)
                self.assertEqual(resident["matrix.weight"].shape, owner.local_shape)
                with self.assertRaises(KeyError):
                    resident["norm.weight"]
                for dtype in (torch.float16, torch.bfloat16):
                    self.assertEqual(plan.materialize(plain, compute_dtype=dtype).dtype, dtype)
                    self.assertEqual(
                        plan.materialize(decay, compute_dtype=dtype).dtype, torch.float32
                    )

    def test_invalid_ownership_rejected_before_payload_io(self):
        matrix = next(PackedPartitionTests().matrices())
        with tempfile.TemporaryDirectory() as directory:
            plan, encoded, _, _ = self.fixture(Path(directory), matrix)
            with patch("cubic_serve.models.quantized_checkpoint.safe_open") as reader:
                for axis, spans in (
                    (0, (AxisSpan(1, 9),)),
                    (1, (AxisSpan(1, 129),)),
                    (0, (AxisSpan(8, 16), AxisSpan(0, 8))),
                    (0, ()),
                    (0, (AxisSpan(0, 32),)),
                ):
                    with self.assertRaises(QuantizationError):
                        plan.materialize(encoded, axis=axis, spans=spans)
                reader.assert_not_called()
