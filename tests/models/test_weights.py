from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

from cubic_serve.contracts.model import ScalarDType, TensorRole, WeightEncoding
from cubic_serve.models import (
    EXECUTABLE_WEIGHT_READBACK_WINDOW_BYTES,
    CheckpointWeightLoadEvidence,
    ExecutableWeightSealRejected,
    FullBF16WeightPolicy,
    WeightIntegrityError,
    WeightLoadEvidence,
    WeightTensorEvidence,
    checkpoint_weight_manifest,
    collect_full_bf16_weight_evidence,
    inspect_checkpoint,
    seal_attested_full_bf16_weight_set,
    seal_executable_full_bf16_weight_set,
    seal_full_bf16_weight_set,
)
from cubic_serve.models.weights import (
    EXECUTABLE_WEIGHT_HEADER_MAX_BYTES,
    _allocate_executable_source_buffer,
    _compare_executable_tensor_bytes,
    _parse_executable_safetensors_header,
    _reject_executable_storage_overlap,
)


def _weights() -> list[tuple[str, object]]:
    return [
        ("embed.weight", torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)),
        ("layer.a_log", torch.tensor([0.25], dtype=torch.float32)),
        ("router.indices", torch.tensor([3, 1], dtype=torch.int64)),
    ]


def _policy() -> FullBF16WeightPolicy:
    return FullBF16WeightPolicy(
        expected_names=("embed.weight", "layer.a_log", "router.indices"),
        fp32_stability_names=("layer.a_log",),
        int64_metadata_names=("router.indices",),
    )


class WeightEvidenceTests(unittest.TestCase):
    def test_executable_seal_raw_offsets_cover_scalar_through_3d(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            tensors = {
                "scalar": torch.tensor(3, dtype=torch.bfloat16),
                "vector": torch.arange(7, dtype=torch.bfloat16),
                "matrix": torch.arange(15, dtype=torch.bfloat16).reshape(3, 5),
                "volume": torch.arange(42, dtype=torch.bfloat16).reshape(2, 3, 7),
            }
            save_file(tensors, root / "model.safetensors")
            inventory = inspect_checkpoint(root)
            names = tuple(sorted(tensors))
            sealed = seal_executable_full_bf16_weight_set(
                lambda: tuple((name, tensors[name].clone()) for name in names),
                inventory=inventory,
                policy=FullBF16WeightPolicy(expected_names=names),
                window_bytes=5,
            )
            self.assertEqual(tuple(item.name for item in sealed.evidence.tensors), names)

    def test_executable_seal_streams_c_order_with_nonoverlapping_readers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps({"model_type": "fixture"}), encoding="utf-8"
            )
            source = torch.arange(42, dtype=torch.bfloat16).reshape(2, 3, 7)
            save_file({"matrix": source}, root / "model.safetensors")
            inventory = inspect_checkpoint(root)
            policy = FullBF16WeightPolicy(expected_names=("matrix",))
            live = source.clone()

            def provider() -> tuple[tuple[str, torch.Tensor], ...]:
                return (("matrix", live),)

            real_empty = torch.empty
            with (
                mock.patch(
                    "cubic_serve.models.weights.safe_open",
                    side_effect=AssertionError("safe_open must not read executable bytes"),
                    create=True,
                ),
                mock.patch(
                    "cubic_serve.models.weights.torch.empty", wraps=real_empty
                ) as allocated,
                mock.patch(
                    "cubic_serve.models.weights._allocate_executable_source_buffer",
                    wraps=_allocate_executable_source_buffer,
                ) as allocated_source,
            ):
                sealed = seal_executable_full_bf16_weight_set(
                    provider,
                    inventory=inventory,
                    policy=policy,
                    window_bytes=5,
                )

            self.assertIsInstance(sealed.evidence, WeightLoadEvidence)
            self.assertNotIsInstance(sealed.evidence, CheckpointWeightLoadEvidence)
            expected = source.reshape(-1).view(torch.uint8).numpy().tobytes()
            self.assertEqual(
                sealed.evidence.tensors[0].bytes_digest,
                f"b2s256:{hashlib.blake2s(expected).hexdigest()}",
            )
            allocated.assert_called_once_with(
                5,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=False,
            )
            allocated_source.assert_called_once_with(5)
            sealed.verify_unchanged(boundary="executable test")

    def test_executable_seal_rejects_byte_drift_and_noncontiguous_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            source = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
            save_file({"weight": source}, root / "model.safetensors")
            inventory = inspect_checkpoint(root)
            policy = FullBF16WeightPolicy(expected_names=("weight",))

            changed = source.clone()
            changed[0, 0] += 1
            with self.assertRaisesRegex(WeightIntegrityError, "bytes differ"):
                seal_executable_full_bf16_weight_set(
                    lambda: (("weight", changed),),
                    inventory=inventory,
                    policy=policy,
                    window_bytes=4,
                )

            noncontiguous = source.t().contiguous().t()
            self.assertFalse(noncontiguous.is_contiguous())
            with self.assertRaisesRegex(WeightIntegrityError, "contiguous storage"):
                seal_executable_full_bf16_weight_set(
                    lambda: (("weight", noncontiguous),),
                    inventory=inventory,
                    policy=policy,
                )

    def test_executable_seal_window_is_positive_and_capped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            tensor = torch.ones(1, dtype=torch.bfloat16)
            save_file({"weight": tensor}, root / "model.safetensors")
            inventory = inspect_checkpoint(root)
            policy = FullBF16WeightPolicy(expected_names=("weight",))
            def provider() -> tuple[tuple[str, torch.Tensor], ...]:
                return (("weight", tensor),)

            for window in (0, -1, EXECUTABLE_WEIGHT_READBACK_WINDOW_BYTES + 1):
                with self.subTest(window=window), self.assertRaisesRegex(
                    WeightIntegrityError, "positive value at most 20 MiB"
                ):
                    seal_executable_full_bf16_weight_set(
                        provider,
                        inventory=inventory,
                        policy=policy,
                        window_bytes=window,
                    )

    def test_executable_seal_rejects_replaced_inventory_shard_with_typed_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            original = torch.arange(8, dtype=torch.bfloat16)
            replacement = original + 32
            shard = root / "model.safetensors"
            save_file({"weight": original}, shard)
            inventory = inspect_checkpoint(root)
            replacement_path = root / "replacement.safetensors"
            save_file({"weight": replacement}, replacement_path)
            os.replace(replacement_path, shard)

            with self.assertRaisesRegex(
                ExecutableWeightSealRejected, "digest differs from inventory"
            ):
                seal_executable_full_bf16_weight_set(
                    lambda: (("weight", replacement.clone()),),
                    inventory=inventory,
                    policy=FullBF16WeightPolicy(expected_names=("weight",)),
                    window_bytes=4,
                )

    def test_executable_seal_rejects_midstream_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            source = torch.arange(16, dtype=torch.bfloat16)
            shard = root / "model.safetensors"
            save_file({"weight": source}, shard)
            inventory = inspect_checkpoint(root)
            replacement = root / "replacement.safetensors"
            save_file({"weight": source.clone()}, replacement)
            replaced = False

            def replacing_compare(*args: object, **kwargs: object) -> str:
                nonlocal replaced
                result = _compare_executable_tensor_bytes(*args, **kwargs)
                if not replaced:
                    os.replace(replacement, shard)
                    replaced = True
                return result

            with (
                mock.patch(
                    "cubic_serve.models.weights._compare_executable_tensor_bytes",
                    side_effect=replacing_compare,
                ),
                self.assertRaisesRegex(
                    ExecutableWeightSealRejected, "changed during byte sealing"
                ),
            ):
                seal_executable_full_bf16_weight_set(
                    lambda: (("weight", source.clone()),),
                    inventory=inventory,
                    policy=FullBF16WeightPolicy(expected_names=("weight",)),
                    window_bytes=4,
                )

    def test_executable_seal_rejects_live_storage_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            source = torch.arange(8, dtype=torch.bfloat16)
            save_file(
                {"first": source.clone(), "second": source.clone()},
                root / "model.safetensors",
            )
            inventory = inspect_checkpoint(root)
            shared = source.clone()
            with self.assertRaisesRegex(
                ExecutableWeightSealRejected, "overlap live storage"
            ):
                seal_executable_full_bf16_weight_set(
                    lambda: (("first", shared), ("second", shared.view_as(shared))),
                    inventory=inventory,
                    policy=FullBF16WeightPolicy(expected_names=("first", "second")),
                    window_bytes=4,
                )

    def test_executable_seal_rejects_overlapping_distinct_storages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            source = torch.arange(4, dtype=torch.bfloat16)
            save_file(
                {"first": source.clone(), "second": source.clone()},
                root / "model.safetensors",
            )
            inventory = inspect_checkpoint(root)
            backing = bytearray(16)
            first = torch.frombuffer(backing, dtype=torch.bfloat16, count=4, offset=0)
            second = torch.frombuffer(
                memoryview(backing)[2:], dtype=torch.bfloat16, count=4, offset=0
            )
            self.assertNotEqual(first.untyped_storage()._cdata, second.untyped_storage()._cdata)
            self.assertLess(second.data_ptr(), first.data_ptr() + first.nbytes)
            with self.assertRaisesRegex(
                ExecutableWeightSealRejected, "overlap live storage"
            ):
                seal_executable_full_bf16_weight_set(
                    lambda: (("first", first), ("second", second)),
                    inventory=inventory,
                    policy=FullBF16WeightPolicy(expected_names=("first", "second")),
                )

        _reject_executable_storage_overlap(
            (
                ("cpu", torch.empty(1, dtype=torch.bfloat16)),
                ("meta", torch.empty(1, dtype=torch.bfloat16, device="meta")),
            )
        )

    def test_executable_seal_rejects_symlinked_shard_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            payload = root / "payload"
            payload.mkdir()
            source = torch.arange(4, dtype=torch.bfloat16)
            save_file({"weight": source}, payload / "real.bin")
            (root / "model.safetensors").symlink_to(payload / "real.bin")
            inventory = inspect_checkpoint(root)
            with self.assertRaisesRegex(
                ExecutableWeightSealRejected, "cannot contain symlinks"
            ):
                seal_executable_full_bf16_weight_set(
                    lambda: (("weight", source.clone()),),
                    inventory=inventory,
                    policy=FullBF16WeightPolicy(expected_names=("weight",)),
                )

    def test_executable_safetensors_header_rejects_malformed_offsets_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            tensors = {
                "first": torch.arange(4, dtype=torch.bfloat16),
                "second": torch.arange(4, dtype=torch.bfloat16),
            }
            valid_path = root / "model.safetensors"
            save_file(tensors, valid_path)
            inventory = inspect_checkpoint(root)
            raw = valid_path.read_bytes()
            original_header_bytes = int.from_bytes(raw[:8], "little")
            original_header = json.loads(raw[8 : 8 + original_header_bytes])
            payload = raw[8 + original_header_bytes :]

            def parse(encoded: bytes, *, declared: int | None = None) -> None:
                candidate = root / "candidate.bin"
                candidate.write_bytes(
                    (len(encoded) if declared is None else declared).to_bytes(8, "little")
                    + encoded
                    + payload
                )
                descriptor = os.open(candidate, os.O_RDONLY)
                try:
                    _parse_executable_safetensors_header(
                        descriptor,
                        file_bytes=candidate.stat().st_size,
                        records=inventory.tensors,
                    )
                finally:
                    os.close(descriptor)

            def encoded(header: object) -> bytes:
                return json.dumps(header, separators=(",", ":")).encode()

            mutations: tuple[tuple[str, object, str], ...] = (
                ("missing name", {"second": original_header["second"]}, "missing tensor"),
                (
                    "dtype",
                    {
                        **original_header,
                        "first": {**original_header["first"], "dtype": "F16"},
                    },
                    "dtype differs",
                ),
                (
                    "shape",
                    {
                        **original_header,
                        "first": {**original_header["first"], "shape": [2, 2]},
                    },
                    "shape differs",
                ),
                (
                    "byte count",
                    {
                        **original_header,
                        "first": {**original_header["first"], "data_offsets": [0, 6]},
                    },
                    "byte count differs",
                ),
                (
                    "out of file",
                    {
                        **original_header,
                        "first": {**original_header["first"], "data_offsets": [1000, 1008]},
                    },
                    "extends beyond",
                ),
                (
                    "overlap",
                    {
                        **original_header,
                        "first": {**original_header["first"], "data_offsets": [0, 8]},
                        "second": {**original_header["second"], "data_offsets": [0, 8]},
                    },
                    "overlap",
                ),
                (
                    "metadata",
                    {**original_header, "__metadata__": {"bad": 1}},
                    "metadata",
                ),
            )
            for label, header, message in mutations:
                with self.subTest(label=label), self.assertRaisesRegex(ValueError, message):
                    parse(encoded(header))

            duplicate_entry = json.dumps(original_header["first"], separators=(",", ":"))
            duplicate = (
                f'{{"first":{duplicate_entry},"first":{duplicate_entry},'
                f'"second":{json.dumps(original_header["second"], separators=(",", ":"))}}}'
            ).encode()
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                parse(duplicate)
            with self.assertRaisesRegex(ValueError, "header budget"):
                parse(b"{}", declared=EXECUTABLE_WEIGHT_HEADER_MAX_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "extends beyond its shard"):
                parse(b"{}", declared=1024)

    def test_executable_seal_wraps_provider_and_cleanup_failures(self) -> None:
        with self.assertRaises(ExecutableWeightSealRejected):
            seal_executable_full_bf16_weight_set(  # type: ignore[arg-type]
                None,
                inventory=object(),  # type: ignore[arg-type]
                policy=object(),  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ExecutableWeightSealRejected, "provider exploded"):
            seal_executable_full_bf16_weight_set(
                lambda: (_ for _ in ()).throw(KeyError("provider exploded")),
                inventory=object(),  # type: ignore[arg-type]
                policy=FullBF16WeightPolicy(expected_names=("weight",)),
            )
        existing = ExecutableWeightSealRejected("already typed")
        with self.assertRaises(ExecutableWeightSealRejected) as observed:
            seal_executable_full_bf16_weight_set(
                lambda: (_ for _ in ()).throw(existing),
                inventory=object(),  # type: ignore[arg-type]
                policy=FullBF16WeightPolicy(expected_names=("weight",)),
            )
        self.assertIs(observed.exception, existing)

        class StopAdmission(BaseException):
            pass

        with self.assertRaises(StopAdmission):
            seal_executable_full_bf16_weight_set(
                lambda: (_ for _ in ()).throw(StopAdmission()),
                inventory=object(),  # type: ignore[arg-type]
                policy=FullBF16WeightPolicy(expected_names=("weight",)),
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            source = torch.arange(4, dtype=torch.bfloat16)
            save_file({"weight": source}, root / "model.safetensors")
            inventory = inspect_checkpoint(root)
            real_close = os.close

            def close_then_fail(file_descriptor: int) -> None:
                real_close(file_descriptor)
                raise OSError("cleanup exploded")

            with (
                mock.patch("cubic_serve.models.weights.os.close", side_effect=close_then_fail),
                self.assertRaisesRegex(ExecutableWeightSealRejected, "cleanup exploded"),
            ):
                seal_executable_full_bf16_weight_set(
                    lambda: (("weight", source.clone()),),
                    inventory=inventory,
                    policy=FullBF16WeightPolicy(expected_names=("weight",)),
                )

    def test_executable_seal_accepts_disjoint_offset_and_rejects_nonfinite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            source = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
            save_file({"weight": source}, root / "model.safetensors")
            inventory = inspect_checkpoint(root)
            backing = torch.empty(10, dtype=torch.bfloat16)
            backing[2:8].copy_(source.reshape(-1))
            offset = backing[2:8].reshape(2, 3)
            self.assertGreater(offset.storage_offset(), 0)
            seal_executable_full_bf16_weight_set(
                lambda: (("weight", offset),),
                inventory=inventory,
                policy=FullBF16WeightPolicy(expected_names=("weight",)),
                window_bytes=5,
            )

        for raw in (0x7F80, 0xFF80, 0x7FC1):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config.json").write_text("{}", encoding="utf-8")
                nonfinite = torch.tensor([raw], dtype=torch.uint16).view(torch.bfloat16)
                save_file({"weight": nonfinite}, root / "model.safetensors")
                inventory = inspect_checkpoint(root)
                with self.assertRaisesRegex(
                    ExecutableWeightSealRejected, "only finite values"
                ):
                    seal_executable_full_bf16_weight_set(
                        lambda value=nonfinite: (("weight", value.clone()),),
                        inventory=inventory,
                        policy=FullBF16WeightPolicy(expected_names=("weight",)),
                    )

    def test_checkpoint_attested_seal_avoids_payload_snapshot_and_guards_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps({"model_type": "fixture"}),
                encoding="utf-8",
            )
            tensors = {
                "embed.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
                "state.scale": torch.tensor([0.25, 0.5], dtype=torch.float32),
            }
            save_file(tensors, root / "model.safetensors")
            inventory = inspect_checkpoint(root)
            policy = FullBF16WeightPolicy(
                expected_names=("embed.weight", "state.scale"),
                fp32_stability_names=("state.scale",),
            )
            plan_digest = f"b2s256:{'1' * 64}"
            manifest = checkpoint_weight_manifest(
                inventory,
                policy=policy,
                semantic_plan_digest=plan_digest,
            )
            loaded: list[tuple[str, object]] = [
                (name, tensors[name].clone()) for name in policy.expected_names
            ]
            sealed = seal_attested_full_bf16_weight_set(
                lambda: loaded,
                inventory=inventory,
                policy=policy,
                semantic_plan_digest=plan_digest,
            )

            self.assertIsInstance(sealed.evidence, CheckpointWeightLoadEvidence)
            self.assertEqual(sealed.evidence.manifest, manifest)
            self.assertEqual(sealed.evidence.inventory_digest, inventory.inventory_digest)
            self.assertEqual(
                sealed.evidence.placements,
                (("embed.weight", "cpu"), ("state.scale", "cpu")),
            )
            sealed.verify_unchanged(boundary="attested test")
            bound_weight = loaded[0][1]
            assert isinstance(bound_weight, torch.Tensor)
            binding = sealed.tensor_binding_digest(
                "embed.weight", bound_weight, boundary="attested test"
            )
            self.assertTrue(binding.startswith("b2s256:"))
            loaded[0][1][0, 0] = torch.tensor(9.0, dtype=torch.bfloat16)
            with self.assertRaisesRegex(WeightIntegrityError, "drifted.*attested test"):
                sealed.verify_unchanged(boundary="attested test")

    def test_sealed_weight_set_uses_versioned_storage_guard_after_startup(self) -> None:
        weight = torch.ones((2, 2), dtype=torch.bfloat16)
        inventory: list[tuple[str, object]] = [("weight", weight)]
        sealed = seal_full_bf16_weight_set(
            lambda: inventory,
            policy=FullBF16WeightPolicy(expected_names=("weight",)),
        )

        sealed.verify_unchanged(boundary="test")
        weight[0, 0] = torch.tensor(2.0, dtype=torch.bfloat16)
        with self.assertRaisesRegex(WeightIntegrityError, "drifted.*test"):
            sealed.verify_unchanged(boundary="test")

    def test_live_tensor_binding_comes_from_sealed_bytes_and_exact_object(self) -> None:
        weight = torch.ones((2, 2), dtype=torch.bfloat16)
        inventory: list[tuple[str, object]] = [("weight", weight)]
        sealed = seal_full_bf16_weight_set(
            lambda: inventory,
            policy=FullBF16WeightPolicy(expected_names=("weight",)),
        )

        first = sealed.tensor_binding_digest("weight", weight, boundary="qualification")
        second = sealed.tensor_binding_digest("weight", weight, boundary="qualification")
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("b2s256:"))
        with self.assertRaisesRegex(WeightIntegrityError, "not the sealed live tensor"):
            sealed.tensor_binding_digest("weight", weight.clone(), boundary="qualification")
        with self.assertRaisesRegex(WeightIntegrityError, "absent"):
            sealed.tensor_binding_digest("missing", weight, boundary="qualification")

    def test_sealed_weight_set_rejects_tensor_replacement_without_rehashing(self) -> None:
        inventory: list[tuple[str, object]] = [("weight", torch.ones((2, 2), dtype=torch.bfloat16))]
        sealed = seal_full_bf16_weight_set(
            lambda: inventory,
            policy=FullBF16WeightPolicy(expected_names=("weight",)),
        )
        inventory[0] = ("weight", torch.ones((2, 2), dtype=torch.bfloat16))

        with self.assertRaisesRegex(WeightIntegrityError, "drifted.*prefix acquire"):
            sealed.verify_unchanged(boundary="prefix acquire")

    def test_sealed_weight_provider_must_use_canonical_inventory_order(self) -> None:
        with self.assertRaisesRegex(WeightIntegrityError, "canonical policy order"):
            seal_full_bf16_weight_set(
                lambda: (
                    ("z", torch.ones(1, dtype=torch.bfloat16)),
                    ("a", torch.ones(1, dtype=torch.bfloat16)),
                ),
                policy=FullBF16WeightPolicy(expected_names=("a", "z")),
            )

    def test_actual_tensors_issue_full_bf16_manifest_and_exact_records(self) -> None:
        source = _weights()
        evidence = collect_full_bf16_weight_evidence(source, policy=_policy())

        self.assertTrue(evidence.manifest.is_full_bf16_weights)
        self.assertEqual(evidence.manifest.encoding, WeightEncoding.DENSE)
        self.assertEqual(evidence.manifest.manifest_digest, evidence.semantic_digest)
        self.assertEqual(evidence.manifest.tensor_count, 3)
        self.assertEqual(evidence.manifest.total_numel, 5)
        self.assertEqual(
            tuple(tensor.name for tensor in evidence.tensors),
            _policy().expected_names,
        )
        embed = evidence.tensors[0]
        self.assertEqual(embed.dtype, ScalarDType.BF16)
        self.assertEqual(embed.shape, (1, 2))
        self.assertEqual(embed.device, "cpu")
        self.assertEqual(embed.byte_count, 4)
        expected_bytes = source[0][1].reshape(-1).view(torch.uint8).numpy().tobytes()
        self.assertEqual(
            embed.bytes_digest,
            f"b2s256:{hashlib.blake2s(expected_bytes).hexdigest()}",
        )
        self.assertEqual(
            tuple((item.role, item.dtype) for item in evidence.manifest.dtype_overrides),
            (
                (TensorRole.STABILITY_PARAMETER, ScalarDType.FP32),
                (TensorRole.METADATA, ScalarDType.INT64),
            ),
        )

    def test_input_order_is_irrelevant_but_actual_bytes_change_identity(self) -> None:
        source = _weights()
        first = collect_full_bf16_weight_evidence(source, policy=_policy())
        reordered = collect_full_bf16_weight_evidence(reversed(source), policy=_policy())
        self.assertEqual(first, reordered)

        changed = _weights()
        changed[0][1][0, 0] = torch.tensor(9.0, dtype=torch.bfloat16)
        second = collect_full_bf16_weight_evidence(changed, policy=_policy())
        self.assertNotEqual(first.tensors[0].bytes_digest, second.tensors[0].bytes_digest)
        self.assertNotEqual(first.semantic_digest, second.semantic_digest)
        self.assertNotEqual(first.evidence_digest, second.evidence_digest)

    def test_duplicate_non_tensor_and_empty_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            collect_full_bf16_weight_evidence(
                [
                    ("embed.weight", torch.ones(1, dtype=torch.bfloat16)),
                    ("embed.weight", torch.ones(1, dtype=torch.bfloat16)),
                ],
                policy=FullBF16WeightPolicy(expected_names=("embed.weight",)),
            )
        with self.assertRaisesRegex(TypeError, "not a tensor"):
            collect_full_bf16_weight_evidence(
                [("embed.weight", "bf16")],
                policy=FullBF16WeightPolicy(expected_names=("embed.weight",)),
            )
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            collect_full_bf16_weight_evidence([], policy=_policy())

    def test_inventory_must_match_independent_core_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing=.*router.indices"):
            collect_full_bf16_weight_evidence(_weights()[:-1], policy=_policy())
        with self.assertRaisesRegex(ValueError, "unexpected=.*extra"):
            collect_full_bf16_weight_evidence(
                [*_weights(), ("extra", torch.ones(1, dtype=torch.bfloat16))],
                policy=_policy(),
            )

    def test_full_bf16_policy_checks_actual_storage_dtypes(self) -> None:
        wrong_default = _weights()
        wrong_default[0] = ("embed.weight", torch.ones((1, 2), dtype=torch.float16))
        with self.assertRaisesRegex(ValueError, "expected bf16, found fp16"):
            collect_full_bf16_weight_evidence(wrong_default, policy=_policy())

        wrong_stability = _weights()
        wrong_stability[1] = ("layer.a_log", torch.ones(1, dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "expected fp32, found bf16"):
            collect_full_bf16_weight_evidence(wrong_stability, policy=_policy())

        wrong_metadata = _weights()
        wrong_metadata[2] = ("router.indices", torch.ones(2, dtype=torch.int8))
        with self.assertRaisesRegex(ValueError, "expected int64, found int8"):
            collect_full_bf16_weight_evidence(wrong_metadata, policy=_policy())

    def test_empty_tensor_and_unsupported_dtype_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            collect_full_bf16_weight_evidence(
                [("embed.weight", torch.empty(0, dtype=torch.bfloat16))],
                policy=FullBF16WeightPolicy(expected_names=("embed.weight",)),
            )
        with self.assertRaisesRegex(TypeError, "unsupported dtype"):
            collect_full_bf16_weight_evidence(
                [("embed.weight", torch.ones(1, dtype=torch.int32))],
                policy=FullBF16WeightPolicy(expected_names=("embed.weight",)),
            )

    def test_nonfinite_bf16_and_fp32_weights_fail_startup_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            collect_full_bf16_weight_evidence(
                [("weight", torch.tensor([float("nan")], dtype=torch.bfloat16))],
                policy=FullBF16WeightPolicy(expected_names=("weight",)),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            collect_full_bf16_weight_evidence(
                [
                    ("stability", torch.tensor([float("inf")], dtype=torch.float32)),
                    ("weight", torch.ones(1, dtype=torch.bfloat16)),
                ],
                policy=FullBF16WeightPolicy(
                    expected_names=("stability", "weight"),
                    fp32_stability_names=("stability",),
                ),
            )

    def test_evidence_records_cannot_be_manually_claimed(self) -> None:
        with self.assertRaisesRegex(TypeError, "actual tensor"):
            WeightTensorEvidence(
                name="fake",
                dtype=ScalarDType.BF16,
                shape=(1,),
                device="cpu",
                numel=1,
                byte_count=2,
                bytes_digest=f"b2s256:{'0' * 64}",
                semantic_digest=f"b2s256:{'1' * 64}",
                evidence_digest=f"b2s256:{'2' * 64}",
            )

        issued = collect_full_bf16_weight_evidence(_weights(), policy=_policy())
        with self.assertRaisesRegex(TypeError, "actual tensors"):
            WeightLoadEvidence(
                manifest=issued.manifest,
                tensors=issued.tensors,
                policy_digest=issued.policy_digest,
                semantic_digest=issued.semantic_digest,
                evidence_digest=issued.evidence_digest,
            )


if __name__ == "__main__":
    unittest.main()
