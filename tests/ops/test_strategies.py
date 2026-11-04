"""Configuration precedence, process transport and operator ownership contracts."""

import os
import pickle
import tempfile
import unittest
from pathlib import Path

from cubic_serve.ops.strategies import (
    LinearParameters,
    MoEParameters,
    OperatorConfig,
    gdn_parameters,
    launch_parameters,
    linear_parameters,
    moe_parameters,
    operator_strategy_scope,
)


class OperatorStrategyTests(unittest.TestCase):
    def test_transport_calibration_controls_and_explicit_override(self) -> None:
        values = {"transport_calibration": False, "transport_calibration_bytes": 131072,
                  "transport_calibration_replays": 3, "transport_min_gain_percent": 7,
                  "transport_max_fit_error_percent": 10, "exchange_latency_bytes": 123456}
        config = OperatorConfig.from_dict({"version": 1, "overrides": {"collective": values}})
        resolved = pickle.loads(pickle.dumps(config)).resolve("collective", device="", model="")
        for key, value in values.items():
            self.assertEqual(getattr(resolved, key), value)
        for key, value in (("transport_calibration", 1), ("transport_calibration_bytes", 0),
                           ("transport_calibration_replays", 0),
                           ("transport_min_gain_percent", 100),
                           ("transport_max_fit_error_percent", -1)):
            with self.assertRaises(ValueError):
                OperatorConfig.from_dict({"version": 1, "overrides": {"collective": {key: value}}})

    def test_shared_expert_overlap_is_a_scoped_boolean(self) -> None:
        for field in ("shared_expert_overlap", "projection_group_overlap"):
            self.assertTrue(getattr(launch_parameters(), field))
            config = OperatorConfig.from_dict({
                "version": 1, "overrides": {"launch": {field: False}},
            })
            with operator_strategy_scope(config, device="", model=""):
                self.assertFalse(getattr(launch_parameters(), field))
            self.assertTrue(getattr(launch_parameters(), field))
            for invalid in (0, 1, "true"):
                with self.assertRaises(ValueError):
                    OperatorConfig.from_dict({
                        "version": 1, "overrides": {"launch": {field: invalid}},
                    })

    def test_sparse_route_grid_prices_candidate_blocks(self) -> None:
        policy = MoEParameters()
        for rows, experts, top_k, tile in ((1, 256, 8, 16), (13, 47, 3, 32), (33, 47, 3, 16)):
            self.assertTrue(policy.route_indexed(rows, experts, top_k, tile))
            self.assertFalse(MoEParameters(sparse_route_grid_percent=0).route_indexed(
                rows, experts, top_k, tile,
            ))
        self.assertFalse(policy.route_indexed(8, 64, 8, 16))
        self.assertFalse(policy.route_indexed(17, 47, 3, 32))
        for value in (0, 50, 100, 1000):
            OperatorConfig.from_dict({
                "version": 1, "overrides": {"moe": {"sparse_route_grid_percent": value}},
            })
        for value in (-1, True, "100"):
            with self.assertRaises(ValueError):
                OperatorConfig.from_dict({
                    "version": 1, "overrides": {"moe": {"sparse_route_grid_percent": value}},
                })

    def test_overlap_partitions_follow_row_coverage_and_budget(self) -> None:
        from cubic_serve.ops.strategies import CollectiveParameters

        params = CollectiveParameters()
        self.assertEqual(params.overlap_partitions(128, 1), ())
        self.assertEqual(params.overlap_partitions(256, 2), ((128, 128),))
        self.assertEqual(
            params.overlap_partitions(512, 4),
            ((256, 256), (256, 128, 128), (128, 128, 128, 128)),
        )
        # Same row coverage on a different geometry gives the scaled plan.
        self.assertEqual(
            params.overlap_partitions(64, 4), ((32, 32), (32, 16, 16), (16, 16, 16, 16))
        )
        self.assertEqual(
            CollectiveParameters(decode_overlap_max_parts=1).overlap_partitions(512, 4), ()
        )
        for rows, tiles in ((0, 1), (33, 4), (64, 0)):
            with self.assertRaises(ValueError):
                params.overlap_partitions(rows, tiles)
        for field in ("decode_overlap_max_parts", "decode_overlap_samples"):
            with self.assertRaises(ValueError):
                OperatorConfig.from_dict({"version": 1, "overrides": {"collective": {field: 0}}})

    def test_route_completion_parameters_roundtrip_and_reject_invalid_costs(self) -> None:
        values = {"route_completion": False, "route_completion_calibration_bytes": 65536,
                  "route_completion_calibration_replays": 7,
                  "route_completion_min_gain_percent": 9,
                  "route_completion_max_fit_error_percent": 12}
        config = OperatorConfig.from_dict({"version": 1, "overrides": {"collective": values}})
        resolved = pickle.loads(pickle.dumps(config)).resolve(
            "collective", device="arbitrary", model="arbitrary"
        )
        for key, value in values.items():
            self.assertEqual(getattr(resolved, key), value)
        for key, value in (("route_completion", 1), ("route_completion_calibration_bytes", 0),
                           ("route_completion_calibration_replays", -1),
                           ("route_completion_min_gain_percent", 100),
                           ("route_completion_max_fit_error_percent", -1)):
            with self.assertRaises(ValueError):
                OperatorConfig.from_dict({"version": 1, "overrides": {"collective": {key: value}}})

    def test_cubic_decode_default_dispatch_and_memory_boundaries(self) -> None:
        from dataclasses import replace

        from cubic_serve.quantization.extensions.cubic.policy import CubicLinearParameters

        compact = "int8-scale-int4-ab-global-fp32"
        params = CubicLinearParameters()
        geometry = dict(n=64, k=128, group_in=32, group_out=1, element_bytes=2)
        # Explicitly unbounded tables exercise every supported width and metadata
        # kind; wider tables never inherit the W4 word-loader capability.
        for metadata in (compact, "e5m9-scale-curve2-tensor-fp16", "fp16"):
            for limit in (0, 2, 3, 4, 5, 6, 7, 8):
                for words in (False, True):
                    tuned = replace(params, level_table_max_bits=limit, level_word_lookup=words,
                                    level_table_max_packed_ratio=0.0)
                    for bits in range(1, 9):
                        expected = "direct"
                        if metadata == compact and 2 <= bits <= limit:
                            expected = "words" if bits == 4 and words else "levels"
                        self.assertEqual(
                            tuned.decode_mode(bits=bits, metadata=metadata, **geometry), expected
                        )
        # W4: 256 groups * 8 half values = 4096 bytes, equal to packed payload.
        for cap, ratio, expected in (
            (4096, 1.0, "words"),
            (4095, 0.0, "direct"),
            (0, 0.99, "direct"),
            (0, 1.0, "words"),
        ):
            tuned = replace(params, level_table_max_bytes=cap, level_table_max_packed_ratio=ratio)
            self.assertEqual(tuned.decode_mode(bits=4, metadata=compact, **geometry), expected)
        # Ceil partial K groups and byte-packed rows independently. W6 table
        # is 256 bytes, packed payload 147 bytes for this local shard.
        tuned = replace(params, level_table_max_bits=8, level_table_max_bytes=256,
                        level_table_max_packed_ratio=0.0)
        tail = dict(n=3, k=65, group_in=64, group_out=2, element_bytes=2)
        self.assertEqual(tuned.decode_mode(bits=6, metadata=compact, **tail), "levels")
        self.assertEqual(
            replace(tuned, level_table_max_bytes=255).decode_mode(bits=6, metadata=compact, **tail),
            "direct",
        )
        # Default admission is a storage ratio, not a bit-width cutoff. The same
        # width can fit or not fit depending on checkpoint group geometry.
        for bits in range(2, 9):
            for group in (16, 32, 64, 128, 256, 512):
                table = 64 * (512 // group) * (1 << (bits - 1)) * 2
                packed = 64 * (512 * bits // 8)
                expected = ("words" if bits == 4 else "levels") if table <= packed else "direct"
                self.assertEqual(params.decode_mode(bits=bits, metadata=compact,
                    n=64, k=512, group_in=group, group_out=1, element_bytes=2), expected)

    def test_cubic_memory_controls_use_existing_configuration_transport(self) -> None:
        config = OperatorConfig.from_dict(
            {
                "version": 1,
                "overrides": {
                    "cubic_linear": {
                        "level_table_max_bytes": 1048576,
                        "level_table_max_packed_ratio": 1.5,
                    }
                },
            }
        )
        params = pickle.loads(pickle.dumps(config)).resolve(
            "cubic_linear", device="any", model="any", n=64, k=128
        )
        self.assertEqual(params.level_table_max_bytes, 1048576)
        self.assertEqual(params.level_table_max_packed_ratio, 1.5)
        for key, values in (
            ("level_table_max_bytes", (-1, True, 1.5)),
            ("level_table_max_packed_ratio", (-1, True, float("nan"), float("inf"))),
        ):
            for value in values:
                with self.assertRaises(ValueError):
                    OperatorConfig.from_dict(
                        {
                            "version": 1,
                            "overrides": {"cubic_linear": {key: value}},
                        }
                    )

    def test_cubic_level_policy_is_format_owned_and_has_an_explicit_control(self) -> None:
        for limit in (0, 2, 4, 8):
            config = OperatorConfig.from_dict(
                {"version": 1, "overrides": {"cubic_linear": {"level_table_max_bits": limit}}}
            )
            params = config.resolve("cubic_linear", device="any", model="any", n=256, k=128)
            self.assertEqual(params.level_table_max_bits, limit)
            for family in ("awq_linear", "fp8_linear", "nvfp4_linear", "linear"):
                self.assertFalse(
                    hasattr(
                        config.resolve(family, device="any", model="any", n=256, k=128),
                        "level_table_max_bits",
                    )
                )
        for limit in (-1, 1, 9, True):
            with self.assertRaises(ValueError):
                OperatorConfig.from_dict(
                    {"version": 1, "overrides": {"cubic_linear": {"level_table_max_bits": limit}}}
                )

    def test_cubic_bit_selection_and_row_reuse_controls(self) -> None:
        for words in (False, True):
            for tiles in (1, 2):
                config = OperatorConfig.from_dict(
                    {
                        "version": 1,
                        "overrides": {
                            "cubic_linear": {
                                "level_word_lookup": words,
                                "row_reuse_max_tiles": tiles,
                            }
                        },
                    }
                )
                params = config.resolve("cubic_linear", device="any", model="any", n=256, k=128)
                self.assertEqual(params.level_word_lookup, words)
                self.assertEqual(params.row_reuse_max_tiles, tiles)
                for family in ("linear", "awq_linear", "fp8_linear", "nvfp4_linear", "cubic_moe"):
                    other = config.resolve(
                        family,
                        device="any",
                        model="any",
                        n=256,
                        k=128,
                        **({"experts": 4, "top_k": 2} if family == "cubic_moe" else {}),
                    )
                    self.assertFalse(hasattr(other, "row_reuse_max_tiles"))
        for key, value in (
            ("level_word_lookup", 1),
            ("row_reuse_max_tiles", True),
            ("row_reuse_max_tiles", 0),
            ("row_reuse_max_tiles", 3),
        ):
            with self.assertRaises(ValueError):
                OperatorConfig.from_dict(
                    {"version": 1, "overrides": {"cubic_linear": {key: value}}}
                )

    def test_cubic_work_estimate_preserves_overrides_and_small_m(self) -> None:
        from cubic_serve.quantization.extensions.cubic.policy import CubicLinearParameters

        params = CubicLinearParameters()
        for processors in (40, 84, 142, 240):
            choices = params.output_tiles(128, 5120, 5120, 8, processors)
            self.assertEqual({c[1:] for c in choices}, {(16, 16, 4), (32, 32, 4), (64, 64, 4)})
            self.assertEqual(params.output_tiles(1, 5120, 5120, 8, processors)[0][1:], (16, 16, 4))
        fixed = CubicLinearParameters(fixed_max_block_m=16)
        self.assertEqual(len(fixed.output_tiles(512, 5120, 5120, 8, 142)), 1)
        config = OperatorConfig.from_dict(
            {
                "version": 1,
                "overrides": {
                    "cubic_linear": {"reuse_block_n": 32, "reuse_warps": 8, "decode_scalar_ops": 23}
                },
            }
        )
        tuned = config.resolve("cubic_linear", device="any", model="any", n=5120, k=5120)
        self.assertEqual(tuned.decode_scalar_ops, 23)
        self.assertIn((64, 32, 8), {c[1:] for c in tuned.output_tiles(512, 5120, 5120, 8, 142)})

    def test_collective_cost_has_no_large_payload_reversal(self) -> None:
        from cubic_serve.ops.strategies import CollectiveParameters

        params = CollectiveParameters()
        for ranks in (2, 3, 4, 6, 8):
            choices = [params.partitioned_sum(2**exponent, ranks) for exponent in range(10, 31)]
            self.assertEqual(choices, sorted(choices))
        self.assertFalse(params.partitioned_sum(1 << 30, 2))
        self.assertFalse(params.partitioned_sum(320 * 1024, 4))
        self.assertTrue(params.partitioned_sum(5 * 1024 * 1024, 4))
        config = OperatorConfig.from_dict(
            {"version": 1, "overrides": {"collective": {"exchange_latency_bytes": 1234567}}}
        )
        self.assertEqual(
            config.resolve("collective", device="any", model="any").exchange_latency_bytes, 1234567
        )

    def test_row_tile_uses_capacity_without_changing_explicit_minimum(self) -> None:
        from cubic_serve.ops.strategies import linear_row_tile

        for capacity in range(1, 514):
            for minimum in (16, 32, 64):
                for maximum in (16, 32, 64):
                    tile = linear_row_tile(capacity, minimum, maximum)
                    self.assertGreaterEqual(tile, minimum)
                    self.assertLessEqual(tile, max(minimum, maximum))
                    self.assertTrue(tile == minimum or tile <= capacity)
                    self.assertEqual(tile & (tile - 1), 0)
        self.assertEqual(linear_row_tile(1, 16, 32), 16)
        self.assertEqual(linear_row_tile(31, 16, 32), 16)
        self.assertEqual(linear_row_tile(32, 16, 32), 32)
        self.assertEqual(linear_row_tile(256, 16, 16), 16)
        with self.assertRaises(ValueError):
            linear_row_tile(0, 16, 32)

    def test_cubic_split_parameters_are_format_owned(self) -> None:
        from cubic_serve.quantization.extensions.cubic.policy import CubicLinearParameters

        config = OperatorConfig.from_dict(
            {
                "version": 1,
                "overrides": {"cubic_linear": {"fixed_split_k": 4}},
            }
        )
        params = config.resolve("cubic_linear", device="any", model="any", n=1024, k=5120)
        self.assertIsInstance(params, CubicLinearParameters)
        self.assertEqual(params.fixed_split_k, 4)
        for family in ("awq_linear", "fp8_linear", "nvfp4_linear"):
            resolved = config.resolve(family, device="any", model="any", n=1024, k=5120)
            self.assertFalse(hasattr(resolved, "fixed_split_k"))
        for value in (0, 3, 64, True):
            with self.assertRaises(ValueError):
                OperatorConfig.from_dict(
                    {
                        "version": 1,
                        "overrides": {"cubic_linear": {"fixed_split_k": value}},
                    }
                )

    def test_wna8_supply_controls_validate_and_roundtrip(self) -> None:
        config = OperatorConfig.from_dict({
            "version": 1, "compute_mode": "wna8",
            "overrides": {"cubic_linear": {
                "wna8_max_k_partitions": 5, "wna8_cta_supply_factor": 1.5,
            }},
        })
        params = config.resolve("cubic_linear", device="any", model="any", n=4352, k=5120)
        self.assertEqual(params.wna8_max_k_partitions, 5)
        self.assertEqual(params.wna8_cta_supply_factor, 1.5)
        self.assertEqual(pickle.loads(pickle.dumps(config)), config)
        for key, values in (
            ("wna8_max_k_partitions", (0, -1, True, 1.5)),
            ("wna8_cta_supply_factor", (0.5, True, float("nan"), float("inf"))),
        ):
            for value in values:
                with self.assertRaises(ValueError):
                    OperatorConfig.from_dict({"version": 1, "overrides": {
                        "cubic_linear": {key: value},
                    }})

    def test_packed_rank_preserves_profile_identity_and_operator_families(self) -> None:
        from types import SimpleNamespace

        import torch

        from cubic_serve.executor.quantized_model_runner import _RankOperators
        from cubic_serve.models.quantization import (
            ActivationFormat,
            QuantizationScheme,
            WeightFormat,
        )
        from cubic_serve.models.quantized_checkpoint import QuantizedMatrix

        config = OperatorConfig.from_dict(
            {
                "version": 1,
                "profiles": [
                    {
                        "operator": family,
                        "device": "audit-gpu",
                        "model": "audit-model",
                        "shape": {"n": 16, "k": 128},
                        "parameters": {"fixed_block_n": width},
                    }
                    for family, width in (("fp8_linear", 64), ("fp8_moe", 128))
                ],
            }
        )
        weight = QuantizedMatrix(
            {
                "weight": torch.zeros(16, 128, dtype=torch.float8_e4m3fn),
                "weight_scale": torch.ones(16, 1),
            },
            QuantizationScheme(WeightFormat.FP8, 8, 1, activation=ActivationFormat.FP8_TOKEN),
            (16, 128),
        )
        for model, expected in (("audit-model", 64), ("other-model", 32)):
            operators = _RankOperators(
                SimpleNamespace(bindings=(), prepared_weights={}, materialized_weights={}),
                None,
                pickle.loads(pickle.dumps(config)),
                device="audit-gpu",
                model=model,
            )
            self.assertEqual(operators._parameters(weight).fixed_block_n, expected)
            self.assertEqual(operators._parameters(torch.empty(16, 128)), LinearParameters())
            routed = operators.config.resolve(
                "fp8_moe",
                device=operators.device_identity,
                model=operators.model_identity,
                n=16,
                k=128,
                experts=8,
                top_k=2,
            )
            self.assertEqual(routed.fixed_block_n, 128 if model == "audit-model" else 32)

    def test_gdn_coefficient_reuse_boolean_override(self) -> None:
        self.assertTrue(
            gdn_parameters(q_heads=4, value_heads=12, key_dim=128, value_dim=128).reuse_coefficients
        )
        config = OperatorConfig.from_dict(
            {"version": 1, "overrides": {"gdn": {"reuse_coefficients": False}}}
        )
        with operator_strategy_scope(config, device="", model=""):
            self.assertFalse(
                gdn_parameters(
                    q_heads=4, value_heads=12, key_dim=128, value_dim=128
                ).reuse_coefficients
            )
            self.assertEqual(linear_parameters(n=128, k=128), LinearParameters())
        for value in (0, 1, "true"):
            with self.assertRaises(ValueError):
                OperatorConfig.from_dict(
                    {"version": 1, "overrides": {"gdn": {"reuse_coefficients": value}}}
                )

    def test_gdn_tiles_are_shape_bound_and_do_not_change_other_families(self) -> None:
        config = OperatorConfig.from_dict(
            {
                "version": 1,
                "profiles": [
                    {
                        "operator": "gdn",
                        "shape": {"value_dim": 128},
                        "parameters": {"native_value_tile": 32},
                    }
                ],
                "overrides": {"gdn": {"native_value_tile": 64}},
            }
        )
        with operator_strategy_scope(pickle.loads(pickle.dumps(config)), device="gpu", model=""):
            self.assertEqual(
                gdn_parameters(
                    q_heads=4, value_heads=12, key_dim=128, value_dim=128
                ).native_value_tile,
                64,
            )
            self.assertEqual(linear_parameters(n=128, k=128), LinearParameters())
        for value in (8, 256, True):
            with self.assertRaises(ValueError):
                OperatorConfig.from_dict(
                    {"version": 1, "overrides": {"gdn": {"native_value_tile": value}}}
                )

    def test_quantized_families_do_not_change_plain_linear_or_moe(self) -> None:
        from cubic_serve.quantization.extensions.fp8.policy import Fp8LinearParameters

        config = OperatorConfig.from_dict(
            {
                "version": 1,
                "profiles": [
                    {
                        "operator": "cubic_linear",
                        "shape": {"k": 128},
                        "parameters": {"fixed_block_n": 64},
                    }
                ],
                "overrides": {"awq_linear": {"fixed_block_k": 64}},
            }
        )
        for family in ("fp8_linear",):
            self.assertEqual(
                config.resolve(family, device="", model="", n=64, k=128), Fp8LinearParameters()
            )
        self.assertEqual(
            config.resolve("nvfp4_linear", device="", model="", n=64, k=128).fixed_block_k, 16
        )
        with self.assertRaisesRegex(ValueError, "format boundary"):
            OperatorConfig.from_dict(
                {"version": 1, "overrides": {"nvfp4_linear": {"fixed_block_k": 32}}}
            )
        self.assertEqual(
            config.resolve("linear", device="", model="", n=64, k=128), LinearParameters()
        )
        self.assertEqual(
            config.resolve("cubic_linear", device="", model="", n=64, k=128).fixed_block_n, 64
        )
        self.assertEqual(
            config.resolve("awq_linear", device="", model="", n=64, k=128).fixed_block_k, 64
        )
        self.assertEqual(
            config.resolve(
                "moe",
                device="",
                model="",
                hidden=128,
                intermediate=64,
                experts=4,
                global_experts=8,
                top_k=2,
            ),
            MoEParameters(),
        )
        for parameter in ("bits", "group_size", "activation_dtype", "grouped_projections"):
            with self.assertRaises(ValueError):
                OperatorConfig.from_dict(
                    {"version": 1, "overrides": {"cubic_linear": {parameter: 4}}}
                )

    def test_projection_grouping_is_typed_and_linear_only(self) -> None:
        config = OperatorConfig.from_dict(
            {"version": 1, "overrides": {"linear": {"grouped_projections": True}}}
        )
        with operator_strategy_scope(
            pickle.loads(pickle.dumps(config)), device="gpu", model="model"
        ):
            self.assertTrue(linear_parameters(n=256, k=5120).grouped_projections)
            self.assertEqual(
                moe_parameters(hidden=128, intermediate=64, experts=4, global_experts=8, top_k=2),
                MoEParameters(),
            )
        for family, value in (("moe", True), ("linear", 1), ("linear", "true")):
            with self.assertRaises(ValueError):
                OperatorConfig.from_dict(
                    {"version": 1, "overrides": {family: {"grouped_projections": value}}}
                )

    def test_defaults_and_family_isolation(self) -> None:
        config = OperatorConfig.from_dict(
            {"version": 1, "overrides": {"linear": {"fixed_block_n": 32}}}
        )
        with operator_strategy_scope(config, device="device", model="model"):
            self.assertEqual(linear_parameters(n=256, k=5120).fixed_block_n, 32)
            self.assertEqual(
                moe_parameters(hidden=128, intermediate=64, experts=4, global_experts=8, top_k=2),
                MoEParameters(),
            )
        self.assertEqual(linear_parameters(n=256, k=5120), LinearParameters())

    def test_precedence_shape_and_spawn_serialization(self) -> None:
        config = OperatorConfig.from_dict(
            {
                "version": 1,
                "profiles": [
                    {"operator": "linear", "model": "dense", "parameters": {"fixed_block_n": 128}},
                    {"operator": "linear", "device": "gpu", "parameters": {"fixed_block_n": 32}},
                    {"operator": "linear", "parameters": {"fixed_block_n": 16}},
                    {
                        "operator": "linear",
                        "model": "dense",
                        "device": "gpu",
                        "shape": {"n": 256, "k": 5120},
                        "parameters": {"fixed_block_n": 64},
                    },
                ],
                "overrides": {"linear": {"fixed_warps": 8, "fixed_stages": 2}},
            }
        )
        transferred = pickle.loads(pickle.dumps(config))
        self.assertEqual(config.digest, transferred.digest)
        for device, model, n, expected in [
            ("gpu", "dense", 256, 64),
            ("gpu", "dense", 512, 128),
            ("gpu", "moe", 256, 32),
            ("other", "moe", 256, 16),
        ]:
            result = transferred.resolve("linear", device=device, model=model, n=n, k=5120)
            self.assertEqual(
                (result.fixed_block_n, result.fixed_warps, result.fixed_stages),
                (expected, 8, 2),
            )

    def test_rejects_typos_semantic_switches_and_m_dependent_reductions(self) -> None:
        invalid = [
            {"version": True},
            {"version": 1, "profiles": [{"operator": "linear_fp8"}]},
            {"version": 1, "profiles": [{"operator": "moe_w4a16"}]},
            {"version": 1, "overides": {}},
            {"version": 1, "profiles": [{"operator": "linear", "shape": {"m": 1}}]},
            {"version": 1, "profiles": [{"operator": "moe", "shape": {"n": 1}}]},
            {"version": 1, "profiles": [{"operator": "linear", "shape": {"n": True}}]},
            {"version": 1, "overrides": {"linear": {"fixed_block_n": 0}}},
            {"version": 1, "overrides": {"linear": {"fixed_block_k": True}}},
            {"version": 1, "overrides": {"linear": {"fixed_stages": 0}}},
            {"version": 1, "overrides": {"linear": {"fixed_stages": 5}}},
            {"version": 1, "overrides": {"linear": {"reduction": "vendor"}}},
            {"version": 1, "overrides": {"moe": {"expert_remap": 1}}},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                OperatorConfig.from_dict(value)

    def test_duplicate_keys_fail_before_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operators.json"
            path.write_text('{"version": 1, "version": 1}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                OperatorConfig.from_file(path)

    def test_changed_configuration_invalidates_identity_and_scope_restores(self) -> None:
        default = OperatorConfig()
        changed = OperatorConfig.from_dict({"version": 1, "overrides": {"moe": {"block_n": 32}}})
        self.assertNotEqual(default.digest, changed.digest)
        try:
            with operator_strategy_scope(changed, device="gpu", model="moe"):
                raise RuntimeError("capture failed")
        except RuntimeError:
            pass
        self.assertEqual(
            moe_parameters(hidden=128, intermediate=64, experts=4, global_experts=8, top_k=2),
            MoEParameters(),
        )

    @unittest.skipUnless(
        os.environ.get("CUBIC_RUN_CUDA_LINEAR_TESTS") == "1", "requires CUDA operator audit"
    )
    def test_grouped_narrow_binding_receipt_graph_and_oracle(self) -> None:
        import torch

        from cubic_serve.ops.cuda.linear.grouped import bind_grouped_linear

        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(249)
        for dtype in (torch.bfloat16, torch.float16):
            weights = tuple(torch.randint(-3, 4, (n, 5120), device=device,
                                          generator=generator).to(dtype) for n in (12, 24))
            owned = torch.randint(-3, 4, (4, 5120), device=device,
                                  generator=generator).to(dtype)
            for m in (4, 17, 256):
                x = torch.full((m, 5120), float("nan"), device=device, dtype=dtype)
                x[:4].copy_(owned)
                active = torch.tensor([4], device=device, dtype=torch.int32)
                outputs = tuple(torch.empty((m, w.shape[0]), device=device,
                                            dtype=dtype) for w in weights)
                primitive = bind_grouped_linear(x, weights, outputs, active)
                self.assertIsNotNone(primitive)
                self.assertIn("fp32-row-column-tree", primitive.launch_signature)
                receipt = primitive.launch(x, weights, outputs, active)
                primitive.validate_receipt(receipt)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    primitive.launch(x, weights, outputs, active)
                for output in outputs:
                    output.fill_(float("nan"))
                graph.replay()
                for output, weight in zip(outputs, weights, strict=True):
                    self.assertTrue(torch.equal(output[:4],
                                                (owned.double() @ weight.double().t()).to(dtype)))
                    self.assertEqual(int(output[4:].view(torch.int16).count_nonzero()), 0)
                graph.reset()

    @unittest.skipUnless(
        os.environ.get("CUBIC_RUN_CUDA_LINEAR_TESTS") == "1", "requires CUDA operator audit"
    )
    def test_grouped_row_reuse_preserves_tree_and_poisoned_padding(self) -> None:
        import torch
        import triton

        from cubic_serve.ops.cuda.linear.grouped import _grouped_row_column

        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(248)
        for dtype in (torch.bfloat16, torch.float16):
            for k, ns in ((4096, (16, 12)), (5120, (12, 24))):
                for integral in (True, False):
                    def values(shape, integral=integral, dtype=dtype):
                        if integral:
                            return torch.randint(-3, 4, shape, generator=generator,
                                                 device=device).to(dtype)
                        return torch.randn(shape, generator=generator,
                                           device=device, dtype=dtype)

                    weights = tuple(values((n, k)) for n in ns)
                    owned = values((4, k))
                    reference = None
                    for m in (4, 17, 256):
                        active = torch.tensor([4], device=device, dtype=torch.int32)
                        x = torch.full((m, k), float("nan"), device=device, dtype=dtype)
                        order = torch.tensor([2, 0, 3, 1], device=device)
                        x[:4].copy_(owned[order])
                        for rows in (1, 2, 4, 8):
                            outputs = tuple(torch.full((m, n), float("nan"),
                                                       device=device, dtype=dtype) for n in ns)
                            _grouped_row_column[(sum(ns), triton.cdiv(m, rows))](
                                x, weights, outputs, active, ns, k, m,
                                triton.next_power_of_2(k), rows,
                                num_warps=8, enable_fp_fusion=False,
                            )
                            if reference is None:
                                reference = tuple(y[:4].clone() for y in outputs)
                            for y, w, expected in zip(outputs, weights, reference, strict=True):
                                self.assertTrue(torch.equal(y[:4], expected))
                                self.assertEqual(int(y[4:].view(torch.int16).count_nonzero()), 0)
                                if integral:
                                    oracle = (owned[order].double() @ w.double().t()).to(dtype)
                                    self.assertTrue(torch.equal(y[:4], oracle))

    @unittest.skipUnless(
        os.environ.get("CUBIC_RUN_CUDA_LINEAR_TESTS") == "1", "requires CUDA operator audit"
    )
    def test_grouped_projection_oracle_padding_and_row_permutation(self) -> None:
        import torch

        from cubic_serve.ops.cuda.linear.grouped import bind_grouped_linear

        config = OperatorConfig.from_dict(
            {"version": 1, "overrides": {"linear": {"grouped_projections": True}}}
        )
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(714)
        weights = tuple(
            torch.randint(-4, 5, (n, 128), generator=generator, device=device).bfloat16()
            for n in (65, 37)
        )
        owned = torch.randint(-4, 5, (3, 128), generator=generator, device=device).bfloat16()
        oracles = tuple((owned.double() @ w.double().t()).bfloat16() for w in weights)
        for capacity in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            count = min(3, capacity)
            x = torch.full((capacity, 128), float("nan"), dtype=torch.bfloat16, device=device)
            outputs = tuple(
                torch.empty((capacity, w.shape[0]), dtype=torch.bfloat16, device=device)
                for w in weights
            )
            active = torch.tensor([count], dtype=torch.int32, device=device)
            x[:count].copy_(owned[:count])
            with operator_strategy_scope(config, device="gpu", model="any"):
                primitive = bind_grouped_linear(x, weights, outputs, active)
            self.assertIsNotNone(primitive)
            receipt = primitive.launch(x, weights, outputs, active)
            primitive.validate_receipt(receipt)
            self.assertIn("ns=(65, 37)", primitive.project_artifact().launch_signature)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                primitive.launch(x, weights, outputs, active)
            for order in (
                torch.arange(count, device=device),
                torch.arange(count - 1, -1, -1, device=device),
            ):
                x[:count].copy_(owned[order])
                for output in outputs:
                    output.fill_(float("nan"))
                graph.replay()
                for output, oracle in zip(outputs, oracles, strict=True):
                    self.assertTrue(torch.equal(output[:count], oracle[order]))
                    self.assertFalse(bool(output[count:].view(torch.int16).count_nonzero()))
            graph.reset()

    @unittest.skipUnless(
        os.environ.get("CUBIC_RUN_CUDA_LINEAR_TESTS") == "1", "requires CUDA operator audit"
    )
    def test_bound_linear_keeps_config_after_scope_exit(self) -> None:
        import torch

        from cubic_serve.ops.cuda.linear.mmvf import BF16MMVFPrimitive

        config = OperatorConfig.from_dict(
            {
                "version": 1,
                "overrides": {"linear": {"fixed_block_n": 32, "fixed_warps": 8, "fixed_stages": 2}},
            }
        )
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(91)
        weight = torch.randint(-4, 5, (65, 128), generator=generator, device=device).to(
            torch.bfloat16
        )
        owned = torch.randint(-4, 5, (3, 128), generator=generator, device=device).to(
            torch.bfloat16
        )
        oracle = (owned.double() @ weight.double().t()).to(torch.bfloat16)
        for capacity in (4, 8, 16, 32, 64, 128, 256):
            x = torch.full((capacity, 128), float("nan"), dtype=torch.bfloat16, device=device)
            x[:3].copy_(owned)
            output = torch.empty((capacity, 65), dtype=torch.bfloat16, device=device)
            active = torch.tensor([3], dtype=torch.int32, device=device)
            with operator_strategy_scope(config, device="gpu", model="dense"):
                primitive = BF16MMVFPrimitive(x, weight, output, active, reduction="fixed_tile")
            self.assertIn("block_n=32", primitive.launch_signature)
            self.assertIn("num_warps=8", primitive.launch_signature)
            self.assertIn("num_stages=2", primitive.launch_signature)
            primitive.launch(x, weight, output, active)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                primitive.launch(x, weight, output, active)
            output.fill_(float("nan"))
            graph.replay()
            self.assertTrue(torch.equal(output[:3], oracle))
            self.assertFalse(bool(output[3:].view(torch.int16).count_nonzero()))
            graph.reset()


if __name__ == "__main__":
    unittest.main()
