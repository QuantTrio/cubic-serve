from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import cubic_serve.models.checkpoint as checkpoint_module
from cubic_serve.models import (
    SAFETENSORS_OFFSET_EVIDENCE,
    CheckpointInventoryError,
    CheckpointLayout,
    FullBF16WeightPolicy,
    inspect_checkpoint,
)


def _write_config(root: Path, value: object | None = None) -> Path:
    path = root / "config.json"
    path.write_text(
        json.dumps({"model_type": "fixture"} if value is None else value),
        encoding="utf-8",
    )
    return path


def _tensor_bytes(tensors: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())


def _write_index(
    root: Path,
    weight_map: dict[str, str],
    total_size: int | float,
    *,
    extra: dict[str, object] | None = None,
) -> Path:
    value: dict[str, object] = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    if extra:
        value.update(extra)
    path = root / "model.safetensors.index.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _content_digest(path: Path) -> str:
    return f"b2s256:{hashlib.blake2s(path.read_bytes()).hexdigest()}"


class CheckpointInventoryTests(unittest.TestCase):
    def test_relaxed_index_summary_reuses_headers_without_weakening_strict_calls(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as cache_directory,
            patch.dict(os.environ, {"CUBIC_SERVE_CACHE_DIR": cache_directory}),
        ):
            root = Path(directory)
            _write_config(root)
            shard = root / "model-00001-of-00001.safetensors"
            save_file({"weight": torch.ones((2, 3), dtype=torch.bfloat16)}, shard)
            _write_index(root, {"weight": shard.name}, 999)
            original = inspect_checkpoint(root, verify_payloads=False, require_index_size=False)
            cache = json.loads(checkpoint_module._inventory_cache_path(root).read_text())
            self.assertNotEqual(cache["version"], 1)
            with patch(
                "cubic_serve.models.checkpoint._read_shard_tensors",
                side_effect=AssertionError("unchanged cached headers were reopened"),
            ):
                self.assertEqual(
                    inspect_checkpoint(root, verify_payloads=False, require_index_size=False),
                    original,
                )
                with self.assertRaisesRegex(CheckpointInventoryError, "metadata.total_size"):
                    inspect_checkpoint(root, verify_payloads=False)
            save_file({"weight": torch.zeros((2, 3), dtype=torch.bfloat16)}, shard)
            changed = inspect_checkpoint(root, verify_payloads=False, require_index_size=False)
            self.assertNotEqual(changed.inventory_digest, original.inventory_digest)

    def test_header_inventory_cache_hits_and_invalidates_on_source_change(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as cache_directory,
            patch.dict(os.environ, {"CUBIC_SERVE_CACHE_DIR": cache_directory}),
        ):
            root = Path(directory)
            _write_config(root)
            shard = root / "model.safetensors"
            save_file({"weight": torch.ones((2, 3), dtype=torch.bfloat16)}, shard)

            original = inspect_checkpoint(root, verify_payloads=False)
            with patch(
                "cubic_serve.models.checkpoint._read_shard_tensors",
                side_effect=AssertionError("cache hit reopened a safetensors header"),
            ):
                cached = inspect_checkpoint(root, verify_payloads=False)
            self.assertEqual(cached, original)

            save_file({"weight": torch.zeros((2, 3), dtype=torch.bfloat16)}, shard)
            with patch(
                "cubic_serve.models.checkpoint._read_shard_tensors",
                wraps=checkpoint_module._read_shard_tensors,
            ) as read_shard:
                changed = inspect_checkpoint(root, verify_payloads=False)
            read_shard.assert_called_once()
            self.assertNotEqual(changed.inventory_digest, original.inventory_digest)

    def test_header_only_inventory_skips_payload_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_config(root)
            save_file(
                {"weight": torch.ones((2, 3), dtype=torch.bfloat16)},
                root / "model.safetensors",
            )

            with patch("cubic_serve.models.checkpoint._hash_file") as hash_file:
                inventory = inspect_checkpoint(root, verify_payloads=False)

            hash_file.assert_not_called()
            self.assertFalse(inventory.payload_verified)
            self.assertEqual(inventory.expected_names, ("weight",))

    def test_single_file_inventory_is_header_derived_and_policy_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _write_config(root)
            tensors = {
                "embed.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
                "state.scale": torch.tensor([0.25, 0.5], dtype=torch.float32),
            }
            shard_path = root / "model.safetensors"
            save_file(tensors, shard_path)

            inventory = inspect_checkpoint(root)

            self.assertTrue(inventory.payload_verified)
            self.assertEqual(inventory.layout, CheckpointLayout.SINGLE_FILE)
            self.assertIsNone(inventory.index_file)
            self.assertEqual(
                inventory.expected_names,
                ("embed.weight", "state.scale"),
            )
            self.assertEqual(inventory.tensor_count, 2)
            self.assertEqual(inventory.total_numel, 8)
            self.assertEqual(inventory.total_tensor_bytes, _tensor_bytes(tensors))
            self.assertEqual(inventory.config_file.content_digest, _content_digest(config_path))
            self.assertEqual(inventory.shard_files[0].content_digest, _content_digest(shard_path))
            self.assertEqual(inventory.offset_evidence, SAFETENSORS_OFFSET_EVIDENCE)
            by_name = {tensor.name: tensor for tensor in inventory.tensors}
            self.assertEqual(by_name["embed.weight"].dtype, "BF16")
            self.assertEqual(by_name["embed.weight"].shape, (2, 3))
            self.assertEqual(by_name["state.scale"].dtype, "F32")
            with safe_open(shard_path, framework="pt", device="cpu") as reader:
                expected_offset_order = tuple(reader.offset_keys())
            actual_offset_order = tuple(
                tensor.name
                for tensor in sorted(inventory.tensors, key=lambda tensor: tensor.offset_ordinal)
            )
            self.assertEqual(actual_offset_order, expected_offset_order)

            policy = FullBF16WeightPolicy(
                expected_names=inventory.expected_names,
                fp32_stability_names=("state.scale",),
            )
            self.assertEqual(policy.expected_names, inventory.expected_names)

    def test_sharded_index_closes_mapping_hashes_and_integral_float_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _write_config(root)
            first = {"embed.weight": torch.ones((2, 2), dtype=torch.bfloat16)}
            second = {"head.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)}
            first_name = "model-00001-of-00002.safetensors"
            second_name = "model.safetensors-00002-of-00002.safetensors"
            save_file(first, root / first_name)
            save_file(second, root / second_name)
            total_size = _tensor_bytes(first) + _tensor_bytes(second)
            index_path = _write_index(
                root,
                {
                    "head.weight": second_name,
                    "embed.weight": first_name,
                },
                float(total_size),
            )

            inventory = inspect_checkpoint(root)

            self.assertEqual(inventory.layout, CheckpointLayout.SHARDED_INDEX)
            self.assertEqual(inventory.expected_names, ("embed.weight", "head.weight"))
            self.assertEqual(inventory.total_numel, 10)
            self.assertEqual(inventory.total_tensor_bytes, total_size)
            self.assertEqual(inventory.config_file.content_digest, _content_digest(config_path))
            self.assertIsNotNone(inventory.index_file)
            assert inventory.index_file is not None
            self.assertEqual(inventory.index_file.content_digest, _content_digest(index_path))
            self.assertEqual(
                tuple(file.relative_path for file in inventory.shard_files),
                tuple(sorted((first_name, second_name))),
            )
            for evidence in inventory.shard_files:
                self.assertEqual(
                    evidence.content_digest, _content_digest(root / evidence.relative_path)
                )
            self.assertEqual(
                {tensor.name: tensor.shard_path for tensor in inventory.tensors},
                {"embed.weight": first_name, "head.weight": second_name},
            )

    def test_data_config_and_index_tampering_changes_sealed_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_config(root)
            shard_name = "model-00001-of-00001.safetensors"
            shard_path = root / shard_name
            tensors = {"embed.weight": torch.ones(4, dtype=torch.bfloat16)}
            save_file(tensors, shard_path)
            index_path = _write_index(
                root,
                {"embed.weight": shard_name},
                _tensor_bytes(tensors),
            )
            original = inspect_checkpoint(root)

            shard_bytes = bytearray(shard_path.read_bytes())
            shard_bytes[-1] ^= 0x01
            shard_path.write_bytes(shard_bytes)
            changed_shard = inspect_checkpoint(root)
            self.assertEqual(original.tensors, changed_shard.tensors)
            self.assertNotEqual(
                original.shard_files[0].content_digest,
                changed_shard.shard_files[0].content_digest,
            )
            self.assertNotEqual(original.inventory_digest, changed_shard.inventory_digest)

            _write_config(root, {"model_type": "fixture", "revision": 2})
            changed_config = inspect_checkpoint(root)
            self.assertNotEqual(
                changed_shard.config_file.content_digest,
                changed_config.config_file.content_digest,
            )
            self.assertNotEqual(changed_shard.inventory_digest, changed_config.inventory_digest)

            index_value = json.loads(index_path.read_text(encoding="utf-8"))
            index_path.write_text(json.dumps(index_value, indent=2), encoding="utf-8")
            changed_index = inspect_checkpoint(root)
            assert changed_config.index_file is not None
            assert changed_index.index_file is not None
            self.assertNotEqual(
                changed_config.index_file.content_digest,
                changed_index.index_file.content_digest,
            )
            self.assertNotEqual(changed_config.inventory_digest, changed_index.inventory_digest)

    def test_corrupt_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_config(root)
            (root / "model.safetensors").write_bytes(b"not a safetensors file")
            with self.assertRaisesRegex(CheckpointInventoryError, "invalid safetensors"):
                inspect_checkpoint(root)

    def test_index_path_escape_is_rejected_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "model"
            root.mkdir()
            _write_config(root)
            outside = base / "outside.safetensors"
            save_file({"embed.weight": torch.ones(1, dtype=torch.bfloat16)}, outside)
            _write_index(root, {"embed.weight": "../outside.safetensors"}, 2)

            with self.assertRaisesRegex(CheckpointInventoryError, "canonical POSIX relative path"):
                inspect_checkpoint(root)

            _write_index(root, {"embed.weight": "C:/outside.safetensors"}, 2)
            with self.assertRaisesRegex(CheckpointInventoryError, "canonical POSIX relative path"):
                inspect_checkpoint(root)

    def test_missing_wrong_and_unknown_index_mappings_are_rejected(self) -> None:
        cases = (
            (
                {
                    "first.weight": "model-00001-of-00002.safetensors",
                    "ghost.weight": "model-00002-of-00002.safetensors",
                },
                "missing from checkpoint index",
            ),
            (
                {
                    "first.weight": "model-00002-of-00002.safetensors",
                    "second.weight": "model-00001-of-00002.safetensors",
                },
                "maps to",
            ),
            (
                {
                    "first.weight": "model-00001-of-00002.safetensors",
                    "second.weight": "model-00002-of-00002.safetensors",
                    "ghost.weight": "model-00002-of-00002.safetensors",
                },
                "missing_from_headers",
            ),
        )
        for weight_map, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_config(root)
                first = {"first.weight": torch.ones(1, dtype=torch.bfloat16)}
                second = {"second.weight": torch.ones(2, dtype=torch.bfloat16)}
                save_file(first, root / "model-00001-of-00002.safetensors")
                save_file(second, root / "model-00002-of-00002.safetensors")
                _write_index(root, weight_map, _tensor_bytes(first) + _tensor_bytes(second))
                with self.assertRaisesRegex(CheckpointInventoryError, message):
                    inspect_checkpoint(root)

    def test_unknown_and_duplicate_shards_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_config(root)
            tensors = {"embed.weight": torch.ones(1, dtype=torch.bfloat16)}
            shard_name = "model-00001-of-00001.safetensors"
            save_file(tensors, root / shard_name)
            save_file({"extra": torch.ones(1)}, root / "unindexed.safetensors")
            _write_index(root, {"embed.weight": shard_name}, _tensor_bytes(tensors))
            with self.assertRaisesRegex(CheckpointInventoryError, "unknown=.*unindexed"):
                inspect_checkpoint(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_config(root)
            shard_name = "model-00001-of-00001.safetensors"
            save_file(
                {
                    "embed.weight": torch.ones(1, dtype=torch.bfloat16),
                    "head.weight": torch.ones(1, dtype=torch.bfloat16),
                },
                root / shard_name,
            )
            alias = "model-alias.safetensors"
            (root / alias).symlink_to(shard_name)
            _write_index(
                root,
                {"embed.weight": shard_name, "head.weight": alias},
                4,
            )
            with self.assertRaisesRegex(CheckpointInventoryError, "duplicate shard paths"):
                inspect_checkpoint(root)

    def test_duplicate_tensor_names_across_shards_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_config(root)
            first_name = "model-00001-of-00002.safetensors"
            second_name = "model-00002-of-00002.safetensors"
            first = {"duplicate.weight": torch.ones(1, dtype=torch.bfloat16)}
            second = {
                "duplicate.weight": torch.ones(1, dtype=torch.bfloat16),
                "unique.weight": torch.ones(1, dtype=torch.bfloat16),
            }
            save_file(first, root / first_name)
            save_file(second, root / second_name)
            _write_index(
                root,
                {"duplicate.weight": first_name, "unique.weight": second_name},
                _tensor_bytes(first) + _tensor_bytes(second),
            )
            with self.assertRaisesRegex(CheckpointInventoryError, "duplicate tensor name"):
                inspect_checkpoint(root)

    def test_strict_json_schema_and_total_size_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_config(root)
            tensors = {"embed.weight": torch.ones(1, dtype=torch.bfloat16)}
            shard_name = "model-00001-of-00001.safetensors"
            save_file(tensors, root / shard_name)

            invalid_indexes = (
                (
                    {
                        "metadata": {"total_size": 2},
                        "weight_map": {"embed.weight": shard_name},
                        "extra": True,
                    },
                    "exactly metadata and weight_map",
                ),
                (
                    {
                        "metadata": {"total_size": 2, "extra": 0},
                        "weight_map": {"embed.weight": shard_name},
                    },
                    "only total_size",
                ),
                (
                    {
                        "metadata": {"total_size": 2.5},
                        "weight_map": {"embed.weight": shard_name},
                    },
                    "integral number",
                ),
                (
                    {
                        "metadata": {"total_size": True},
                        "weight_map": {"embed.weight": shard_name},
                    },
                    "integral number",
                ),
                (
                    {"metadata": {"total_size": 2}, "weight_map": {}},
                    "non-empty object",
                ),
                (
                    {
                        "metadata": {"total_size": 4},
                        "weight_map": {"embed.weight": shard_name},
                    },
                    "differs from header-derived",
                ),
            )
            index_path = root / "model.safetensors.index.json"
            for value, message in invalid_indexes:
                with self.subTest(message=message):
                    index_path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaisesRegex(CheckpointInventoryError, message):
                        inspect_checkpoint(root)

            index_path.write_text(
                '{"metadata":{"total_size":2},'
                f'"weight_map":{{"embed.weight":"{shard_name}",'
                f'"embed.weight":"{shard_name}"}}}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CheckpointInventoryError, "duplicate JSON key"):
                inspect_checkpoint(root)

    def test_config_must_be_strict_json_object_and_layout_must_be_unambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_config(root, [])
            save_file(
                {"embed.weight": torch.ones(1, dtype=torch.bfloat16)},
                root / "model.safetensors",
            )
            with self.assertRaisesRegex(CheckpointInventoryError, "root must be a JSON object"):
                inspect_checkpoint(root)

            _write_config(root)
            _write_index(root, {"embed.weight": "model.safetensors"}, 2)
            with self.assertRaisesRegex(CheckpointInventoryError, "ambiguous"):
                inspect_checkpoint(root)


if __name__ == "__main__":
    unittest.main()
