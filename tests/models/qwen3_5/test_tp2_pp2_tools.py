from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import torch

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "qwen3_5"
sys.path.insert(0, str(TOOLS))

import tp2_pp2_audit as audit  # noqa: E402
import tp2_pp2_bench as bench  # noqa: E402


class Qwen3_5TP2PP2ToolTests(unittest.TestCase):
    def test_dynamic_api_loader_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            audit._load_performance_api("cubic_serve.does_not_exist")

    def test_bucket_parser_rejects_duplicates_and_unknown_sizes(self) -> None:
        self.assertEqual(audit._csv_ints("128,512", allowed=(128, 256, 512)), (128, 512))
        for value in ("", "128,128", "64", "not-an-int"):
            with self.assertRaises(argparse.ArgumentTypeError):
                audit._csv_ints(value, allowed=(128, 256, 512))

    def test_snapshot_compare_is_bit_exact_and_locates_first_difference(self) -> None:
        expected = {"state": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)}
        equal = audit._compare(expected, {"state": expected["state"].clone()})
        self.assertTrue(equal["passed"])
        actual = {"state": torch.tensor([[1.0, 3.0]], dtype=torch.bfloat16)}
        different = audit._compare(expected, actual)
        self.assertFalse(different["passed"])
        field = different["fields"]["state"]
        self.assertFalse(field["bit_exact"])
        self.assertEqual(field["first_difference"]["index"], (0, 1))

    def test_aggregate_throughput_uses_global_slowest_rank_latency(self) -> None:
        reports = []
        for rank, latency in enumerate((10.0, 11.0, 12.0, 11.5)):
            reports.append(
                {
                    "rank": rank,
                    "results": [
                        {
                            "status": "measured",
                            "logical_m": 128,
                            "captured_m": 128,
                            "bucket_logical_ms": [64, 128],
                            "context_tokens": 128,
                            "latency_ms": latency,
                            "capture_seconds": 2.0 + rank,
                            "memory": {"peak_allocated_gib": 20.0 + rank},
                        }
                    ],
                }
            )
        result = bench._aggregate(reports, (128,))[0]
        self.assertEqual(result["latency_ms"], 12.0)
        self.assertEqual(result["aggregate_decode_tokens_per_second"], 10666.67)
        self.assertFalse(result["fallback"])
        self.assertEqual(len(result["rank_memory"]), 4)

    def test_serving_benchmark_routes_every_logical_m_to_power_of_two_bucket(self) -> None:
        self.assertEqual(
            tuple(sorted({bench._captured_m(value) for value in bench._LOGICAL_MS})),
            bench._CAPTURED_MS,
        )
        self.assertEqual(bench._captured_m(1), 1)
        for logical_m in bench._LOGICAL_MS:
            captured_m = bench._captured_m(logical_m)
            self.assertGreaterEqual(captured_m, logical_m)
            self.assertLess(captured_m, 2 * logical_m)


if __name__ == "__main__":
    unittest.main()
