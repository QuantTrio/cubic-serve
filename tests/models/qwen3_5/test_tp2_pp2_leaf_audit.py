from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import cast

import torch

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools" / "qwen3_5"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

import tp2_pp2_leaf_audit as leaf  # noqa: E402

from cubic_serve.executor import model_runner as parallel  # noqa: E402
from cubic_serve.guardrail.oracle import verify_observed_bf16_linear_sample  # noqa: E402
from cubic_serve.models.qwen3_5 import parse_qwen3_5_config  # noqa: E402
from tests.models.qwen3_5.test_config import _fixture  # noqa: E402

_DIGEST = "b2s256:" + "1" * 64


class Qwen3_5TP2PP2LeafAuditTests(unittest.TestCase):
    def test_gdn_abz_pack_preserves_unequal_semantic_segments(self) -> None:
        class Weights:
            device = torch.device("cpu")

            def __init__(self) -> None:
                self.values: dict[str, torch.Tensor] = {}
                for layer in (0, 2):
                    prefix = f"model.language_model.layers.{layer}.linear_attn"
                    for suffix, width, value in (
                        ("in_proj_a.weight", 2, layer + 1),
                        ("in_proj_b.weight", 3, layer + 11),
                        ("in_proj_z.weight", 5, layer + 21),
                    ):
                        self.values[f"{prefix}.{suffix}"] = torch.full(
                            (width, 4), value, dtype=torch.bfloat16
                        )

            def __getitem__(self, name: str) -> torch.Tensor:
                return self.values[name]

            def rebind_semantic_view(self, name: str, value: torch.Tensor) -> None:
                self.values[name] = value

        source = Weights()
        packed, widths = parallel._pack_gdn_abz_weights(
            cast(parallel._StageWeights, source), (0, 2)
        )
        self.assertEqual(widths, (2, 3, 5))
        self.assertEqual(packed.shape, (2, 10, 4))
        self.assertTrue(packed.is_contiguous())
        for ordinal, layer in enumerate((0, 2)):
            prefix = f"model.language_model.layers.{layer}.linear_attn"
            self.assertTrue(torch.equal(packed[ordinal, :2], source[f"{prefix}.in_proj_a.weight"]))
            self.assertTrue(torch.equal(packed[ordinal, 2:5], source[f"{prefix}.in_proj_b.weight"]))
            self.assertTrue(torch.equal(packed[ordinal, 5:], source[f"{prefix}.in_proj_z.weight"]))
            for suffix in ("in_proj_a.weight", "in_proj_b.weight", "in_proj_z.weight"):
                self.assertEqual(
                    source[f"{prefix}.{suffix}"].untyped_storage().data_ptr(),
                    packed.untyped_storage().data_ptr(),
                )

    def test_gate_up_pack_is_gate_first_persistent_bf16(self) -> None:
        class Weights:
            device = torch.device("cpu")

            def __init__(self) -> None:
                self.values: dict[str, torch.Tensor] = {}
                for layer in range(2):
                    prefix = f"model.language_model.layers.{layer}.mlp"
                    self.values[f"{prefix}.gate_proj.weight"] = torch.full(
                        (3, 4), layer + 1, dtype=torch.bfloat16
                    )
                    self.values[f"{prefix}.up_proj.weight"] = torch.full(
                        (3, 4), layer + 11, dtype=torch.bfloat16
                    )

            def __getitem__(self, name: str) -> torch.Tensor:
                return self.values[name]

            def rebind_semantic_view(self, name: str, value: torch.Tensor) -> None:
                self.values[name] = value

        source = Weights()
        packed = parallel._pack_gate_up_weights(cast(parallel._StageWeights, source), 2)
        self.assertEqual(packed.shape, (2, 6, 4))
        self.assertEqual(packed.dtype, torch.bfloat16)
        self.assertTrue(packed.is_contiguous())
        for layer in range(2):
            prefix = f"model.language_model.layers.{layer}.mlp"
            self.assertTrue(torch.equal(packed[layer, :3], source[f"{prefix}.gate_proj.weight"]))
            self.assertTrue(torch.equal(packed[layer, 3:], source[f"{prefix}.up_proj.weight"]))
            self.assertEqual(
                source[f"{prefix}.gate_proj.weight"].untyped_storage().data_ptr(),
                packed.untyped_storage().data_ptr(),
            )
            self.assertEqual(
                source[f"{prefix}.up_proj.weight"].untyped_storage().data_ptr(),
                packed.untyped_storage().data_ptr(),
            )

    def test_observer_consumes_immutable_snapshots_only_after_forward(self) -> None:
        consumed: list[int] = []

        def consume(event: object, ordinal: int) -> None:
            assert isinstance(event, parallel.Qwen3_5LinearLeafEvent)
            self.assertEqual(event.linear_input.item(), ordinal)
            consumed.append(ordinal)
            if ordinal == 1:
                raise ValueError("bad leaf")

        observer = leaf._DeferredObserver(consume)
        reused = torch.zeros((1, 1))
        for ordinal in range(4):
            reused.fill_(ordinal)
            observer(
                parallel.Qwen3_5LinearLeafEvent(
                    global_layer=ordinal,
                    weight_name="projection.weight",
                    weight_shape=(1, 1),
                    artifact_digest="synthetic artifact",
                    launch_specialization_digest="synthetic launch",
                    linear_input=reused,
                    local_weight=torch.ones((1, 1)),
                    local_output=reused,
                    tp_partial=None,
                    tp_reduced=None,
                )
            )
        reused.fill_(-1)
        self.assertEqual(observer.count, 4)
        self.assertEqual(consumed, [])
        with self.assertRaisesRegex(ValueError, "bad leaf"):
            observer.finish(4)
        self.assertEqual(consumed, [0, 1])

    def test_snapshot_failure_is_deferred_until_all_callbacks_finish(self) -> None:
        consumed: list[int] = []
        observer = leaf._DeferredObserver(lambda event, ordinal: consumed.append(ordinal))
        for _ in range(4):
            observer(object())
        self.assertEqual(observer.count, 4)
        self.assertEqual(consumed, [])
        with self.assertRaisesRegex(RuntimeError, "deferred through 4 callbacks") as raised:
            observer.finish(4)
        self.assertIsInstance(raised.exception.__cause__, AttributeError)

    def test_expected_program_covers_every_stage_linear_occurrence(self) -> None:
        config = parse_qwen3_5_config(_fixture())
        first = leaf._expected_weights(config, 0)
        second = leaf._expected_weights(config, 1)
        self.assertEqual((len(first), len(second)), (248, 249))
        self.assertEqual(first[0], "model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
        self.assertEqual(second[-1], "lm_head.weight")
        for stage in (first, second):
            row = tuple(
                name
                for name in stage
                if name.endswith(
                    (
                        "linear_attn.out_proj.weight",
                        "self_attn.o_proj.weight",
                        "mlp.down_proj.weight",
                    )
                )
            )
            self.assertEqual(len(row), 64)

        tp4_pp1 = leaf._expected_weights(config, 0, pp_size=1)
        self.assertEqual(len(tp4_pp1), 497)
        self.assertEqual(tp4_pp1[0], first[0])
        self.assertEqual(tp4_pp1[-1], "lm_head.weight")
        self.assertEqual(
            sum(
                name.endswith(
                    (
                        "linear_attn.out_proj.weight",
                        "self_attn.o_proj.weight",
                        "mlp.down_proj.weight",
                    )
                )
                for name in tp4_pp1
            ),
            128,
        )

    def test_segmented_local_columns_map_to_global_semantic_columns(self) -> None:
        self.assertEqual(
            leaf._global_columns(((0, 2), (4, 6), (8, 10)), (0, 1, 2, 3, 4, 5)),
            (0, 1, 4, 5, 8, 9),
        )
        with self.assertRaises(ValueError):
            leaf._global_columns(((0, 2),), (2,))

    def test_sample_witness_adapter_preserves_full_k_and_launch_binding(self) -> None:
        x = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        weight = torch.tensor([[3.0, 4.0], [2.0, 1.0]], dtype=torch.bfloat16)
        output = torch.mm(x.float(), weight.float().t()).to(torch.bfloat16)
        witness = verify_observed_bf16_linear_sample(
            x=x,
            weight=weight,
            output=output,
            input_rows=(0,),
            output_columns=(0, 1),
            operand_binding_digest=_DIGEST,
            artifact_digest=_DIGEST,
            launch_specialization_digest=_DIGEST,
        )
        payload = leaf._witness_payload(witness)
        self.assertEqual(payload["reduction_length"], 2)
        self.assertEqual(payload["sampled_element_count"], 2)
        self.assertEqual(payload["launch_specialization_digest"], _DIGEST)

    def test_partitioned_row_witness_binds_both_complete_k_shards(self) -> None:
        x0 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        x1 = torch.tensor([1.0, 1.0], dtype=torch.bfloat16)
        w0 = torch.tensor([[3.0, 4.0], [2.0, 1.0]], dtype=torch.bfloat16)
        w1 = torch.tensor([[1.0, 2.0], [3.0, 1.0]], dtype=torch.bfloat16)
        partial0 = torch.mv(w0.float(), x0.float()).to(torch.bfloat16)
        partial1 = torch.mv(w1.float(), x1.float()).to(torch.bfloat16)
        reduced = (partial0.float() + partial1.float()).to(torch.bfloat16)

        def record(
            rank: int,
            x: torch.Tensor,
            weight: torch.Tensor,
            partial: torch.Tensor,
        ) -> dict[str, object]:
            return {
                "ordinal": 0,
                "global_layer": 0,
                "weight_name": "model.language_model.layers.0.mlp.down_proj.weight",
                "global_shape": (2, 4),
                "layout": "row",
                "sample_local_columns": (0, 1),
                "sample_global_columns": (0, 1),
                "artifact_digest": _DIGEST,
                "launch_specialization_digest": _DIGEST,
                "operand_binding_digest": "b2s256:" + str(rank + 2) * 64,
                "local_k_witness": {"witness_digest": "b2s256:" + str(rank + 4) * 64},
                "spans": ((rank * 2, rank * 2 + 2),),
                "row_x": x,
                "row_weight": weight,
                "tp_partial": partial,
                "tp_reduced": reduced.clone(),
            }

        result = leaf._row_witness(record(0, x0, w0, partial0), record(1, x1, w1, partial1))
        self.assertEqual(result["reduction_length"], 4)
        self.assertTrue(str(result["witness_digest"]).startswith("b2s256:"))

    def test_partitioned_row_witness_binds_four_complete_k_shards(self) -> None:
        xs = [torch.tensor([rank + 1.0], dtype=torch.bfloat16) for rank in range(4)]
        weights = [torch.tensor([[rank + 2.0]], dtype=torch.bfloat16) for rank in range(4)]
        partials = [
            torch.mv(weight.float(), x.float()).to(torch.bfloat16)
            for x, weight in zip(xs, weights, strict=True)
        ]
        reduced = sum((partial.float() for partial in partials), torch.zeros(1)).to(torch.bfloat16)

        def record(rank: int) -> dict[str, object]:
            return {
                "ordinal": 0,
                "global_layer": 0,
                "weight_name": "model.language_model.layers.0.mlp.down_proj.weight",
                "global_shape": (1, 4),
                "layout": "row",
                "sample_local_columns": (0,),
                "sample_global_columns": (0,),
                "artifact_digest": _DIGEST,
                "launch_specialization_digest": _DIGEST,
                "operand_binding_digest": "b2s256:" + str(rank + 2) * 64,
                "local_k_witness": {"witness_digest": "b2s256:" + str(rank + 4) * 64},
                "spans": ((rank, rank + 1),),
                "row_x": xs[rank],
                "row_weight": weights[rank],
                "tp_partial": partials[rank],
                "tp_reduced": reduced.clone(),
            }

        result = leaf._row_witness(*(record(rank) for rank in range(4)))
        self.assertEqual(result["reduction_length"], 4)
        self.assertTrue(str(result["witness_digest"]).startswith("b2s256:"))


if __name__ == "__main__":
    unittest.main()
