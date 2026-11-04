from __future__ import annotations

import os
import unittest
from pathlib import Path

from cubic_serve.generation.sampler import GREEDY_SAMPLING
from cubic_serve.models.builtin import admit_qwen3_5_reference
from cubic_serve.runtime import performance as serving
from cubic_serve.runtime.token_stream import TokenStream

CHECKPOINT = os.getenv("CUBIC_QWEN3_5_CHECKPOINT")


@unittest.skipUnless(CHECKPOINT, "set CUBIC_QWEN3_5_CHECKPOINT for the real TP4 audit")
class RaggedMTPDeviceTests(unittest.TestCase):
    def test_fixed_and_mixed_request_k_are_greedy_exact(self) -> None:
        checkpoint = Path(CHECKPOINT)
        weights = admit_qwen3_5_reference(checkpoint, verify_payloads=False).checkpoint_weights
        admission = serving._admit(
            checkpoint,
            (4, 5, 6, 7),
            4096,
            256,
            tp_size=4,
            pp_size=1,
            ep_size=4,
            num_speculative_tokens=1,
            max_speculative_tokens=8,
            dynamic_max_speculative_tokens=7,
            dynamic_speculation=True,
            use_prefix_cache=True,
            checkpoint_weights=weights,
        )
        controller = serving._ProcessController(
            admission,
            startup_timeout_seconds=1800.0,
            worker_timeout_seconds=600.0,
        )
        try:
            def run_group(
                label: str, depths: tuple[int, ...]
            ) -> tuple[tuple[tuple[int, ...], ...], tuple[int, int, int]]:
                with controller._condition:
                    before = (
                        controller._serving_stats.mtp_accepted_total,
                        controller._serving_stats.mtp_proposed_total,
                        controller._serving_stats.mtp_rounds_total,
                    )
                streams: list[TokenStream] = []
                with controller._condition:
                    for index, draft_tokens in enumerate(depths):
                        request_id = f"ragged:{label}:{index}:k{draft_tokens}"
                        stream = TokenStream()
                        controller.submit(
                            serving._GenerateCommand(
                                request_id,
                                tuple(range(1, 33)),
                                16,
                                (),
                                GREEDY_SAMPLING,
                            ),
                            stream,
                        )
                        # This is an execution audit, not a policy test: pin
                        # each request's K so the dynamic planner cannot
                        # normalize the mixed group back to its initial K1.
                        controller._requests[request_id].draft_policy = None
                        controller._requests[request_id].draft_tokens = draft_tokens
                        streams.append(stream)
                outputs = tuple(tuple(stream) for stream in streams)
                with controller._condition:
                    after = (
                        controller._serving_stats.mtp_accepted_total,
                        controller._serving_stats.mtp_proposed_total,
                        controller._serving_stats.mtp_rounds_total,
                    )
                witness = tuple(
                    end - start for start, end in zip(before, after, strict=True)
                )
                print(
                    f"RAGGED_DEVICE_SUMMARY {label} "
                    + " ".join(
                        f"k{depth}={output[:16]}"
                        for depth, output in zip(depths, outputs, strict=True)
                    )
                    + f" acceptance={witness}",
                    flush=True,
                )
                return outputs, witness

            k0, k0_witness = run_group("fixed-k0", (0, 0))
            k1, k1_witness = run_group("fixed-k1", (1, 1))
            k3, k3_witness = run_group("fixed-k3", (3, 3))
            mixed, mixed_witness = run_group("mixed", (0, 1, 3))

            oracle = k0[0]
            self.assertTrue(oracle)
            for group in (k0, k1, k3, mixed):
                for output in group:
                    self.assertEqual(output, oracle)
            self.assertEqual(k0_witness, (0, 0, 0))
            self.assertTrue(all(value % 2 == 0 for value in (*k1_witness, *k3_witness)))
            accepted, proposed, rounds = mixed_witness
            self.assertLessEqual(accepted, proposed)
            self.assertGreater(proposed, 0)
            self.assertGreater(rounds, 0)
            self.assertEqual(len(controller.rank_cross_row_state_evidence_ids), 4)
            self.assertTrue(all(controller.rank_cross_row_state_evidence_ids))
            self.assertGreater(controller._serving_stats.prefix_hit_tokens_total, 0)
            self.assertTrue(controller.ready)
        finally:
            controller.close()


if __name__ == "__main__":
    unittest.main()
