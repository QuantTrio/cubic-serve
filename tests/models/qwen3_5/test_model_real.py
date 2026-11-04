from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import cast

import torch

from cubic_serve.contracts.execution import BatchPhase
from cubic_serve.contracts.request import (
    ContentPart,
    DecodePositionPlan,
    InputPlan,
    TokenEventKind,
    TokenPosition,
)
from cubic_serve.models.builtin import admit_qwen3_5_reference
from cubic_serve.models.protocol import PreparedInput
from cubic_serve.runtime.engine import EngineStep, ReferenceEngine

_CHECKPOINT = Path("/pubdata/llm_models/Qwen/Qwen3.8-27B")
_RUN_REAL = os.environ.get("CUBIC_RUN_REAL_QWEN") == "1"


@unittest.skipUnless(_RUN_REAL and _CHECKPOINT.is_dir(), "real Qwen smoke test is opt-in")
class Qwen3_5RealCheckpointTests(unittest.TestCase):
    def test_descriptor_declares_executable_visual_inputs(self) -> None:
        descriptor = admit_qwen3_5_reference(
            _CHECKPOINT,
            verify_payloads=False,
        ).adapter.descriptor
        modalities = {item.name: item for item in descriptor.model_spec.modalities}

        self.assertEqual(set(modalities), {"text", "image", "video"})
        self.assertEqual(modalities["image"].mime_prefixes, ("image/",))
        self.assertEqual(modalities["video"].mime_prefixes, ("video/",))
        self.assertEqual(modalities["image"].feature_hidden_size, 5120)
        self.assertEqual(modalities["video"].feature_hidden_size, 5120)
        self.assertEqual(modalities["image"].encoder_id, modalities["video"].encoder_id)
        self.assertTrue(
            modalities["image"].encoder_id.startswith("qwen3_5.vision.reference@b2s256:")
        )

    @staticmethod
    def token_ids(steps: tuple[EngineStep, ...], request_id: str) -> tuple[int, ...]:
        return tuple(
            cast(int, event.token_id)
            for step in steps
            for event in step.events
            if event.request_id == request_id and event.kind is TokenEventKind.TOKEN
        )

    @staticmethod
    def prepared_input(
        engine: ReferenceEngine,
        request_id: str,
        token_ids: tuple[int, ...],
    ) -> PreparedInput:
        descriptor = engine.loaded_adapter.descriptor
        return PreparedInput(
            InputPlan(
                request_id=request_id,
                model_id=descriptor.model_spec.model_id,
                processor_recipe_digest=f"b2s256:{'a' * 64}",
                weight_variant=descriptor.weight_variant,
                content_parts=(ContentPart(0, "text", text="pretokenized fixture"),),
                token_ids=token_ids,
                positions=tuple(TokenPosition((index,)) for index in range(len(token_ids))),
                decode_positions=DecodePositionPlan(
                    first=TokenPosition((len(token_ids),)),
                    step_axes=(1,),
                ),
            )
        )

    @staticmethod
    def output_logits(steps: tuple[EngineStep, ...], request_id: str) -> torch.Tensor:
        rows = tuple(
            row.tensor for step in steps for row in step.logits_rows if row.request_id == request_id
        )
        if len(rows) != 1:
            raise AssertionError(f"expected one output row, found {len(rows)}")
        return rows[0]

    def test_bf16_eager_scheduler_prefix_and_official_oracle(self) -> None:
        admission = admit_qwen3_5_reference(_CHECKPOINT)
        engine = ReferenceEngine(
            admission.adapter,
            kv_layout=admission.kv_layout,
            weight_policy=admission.weight_policy,
            checkpoint_weights=admission.checkpoint_weights,
            model_root=admission.model_root,
            owned_layer_ids=admission.owned_layer_ids,
            device="cuda:0",
            token_budget=16,
            prefill_chunk_tokens=16,
        )
        prompt = (
            814,
            20139,
            3069,
            279,
            12515,
            369,
            6105,
            303,
            799,
            11316,
            13,
        )

        oracle = self.prepared_input(engine, "oracle", prompt)
        engine.admit(oracle, max_new_tokens=2, use_prefix_cache=False)
        oracle_steps = engine.run_until_drained()
        self.assertEqual(self.token_ids(oracle_steps, "oracle"), (271, 760))

        long_prompt = prompt + prompt[:6]
        cold = self.prepared_input(engine, "prefix-cold", long_prompt)
        self.assertFalse(engine.admit(cold, max_new_tokens=1).hit)
        cold_steps = engine.run_until_drained()
        publications = tuple(
            publication for step in cold_steps for publication in step.prefix_publications
        )
        self.assertEqual(len(publications), 1)
        self.assertEqual(publications[0].snapshot.physical_block_boundary, 16)
        self.assertTrue(all(step.post_commit_ok for step in cold_steps))

        lease = engine.prefix_cache.acquire(
            publications[0].snapshot.prefix_key,
            source_rank=0,
        )
        self.assertIsNotNone(lease)
        assert lease is not None
        with lease:
            snapshot = lease.snapshot
            self.assertEqual(len(snapshot.recurrent_states), 96)
            self.assertEqual({state.position for state in snapshot.recurrent_states}, {16})
            self.assertEqual(
                {state.value.dtype for state in snapshot.recurrent_states},
                {torch.bfloat16, torch.float32},
            )

        warm = self.prepared_input(engine, "prefix-warm", long_prompt)
        warm_admission = engine.admit(warm, max_new_tokens=1)
        self.assertEqual(
            (
                warm_admission.hit,
                warm_admission.token_count,
                warm_admission.resume_token_count,
            ),
            (True, 16, 16),
        )
        warm_steps = engine.run_until_drained()
        self.assertEqual(
            self.token_ids(warm_steps, "prefix-warm"),
            self.token_ids(cold_steps, "prefix-cold"),
        )
        self.assertTrue(
            torch.equal(
                self.output_logits(warm_steps, "prefix-warm"),
                self.output_logits(cold_steps, "prefix-cold"),
            )
        )
        cold_digest = engine.terminal_digest("prefix-cold")
        warm_digest = engine.terminal_digest("prefix-warm")
        self.assertEqual(warm_digest.cache_digest, cold_digest.cache_digest)
        self.assertEqual(warm_digest.state_digest, cold_digest.state_digest)

        table = engine.allocator.page_table("prefix-warm")
        key, value = engine.store.read_kv(
            "prefix-warm",
            table.logical_to_physical[0],
            admission.kv_layout.layer_ids[0],
            tuple(range(16)),
        )
        self.assertIs(key.dtype, torch.bfloat16)
        self.assertIs(value.dtype, torch.bfloat16)

        # Exercise a real mixed schedule whose first plan fills M=16 with an
        # 11-token prompt and the first 5 rows of a 7-token prompt.  Each
        # request must remain bitwise-equivalent to running in isolation.
        concurrent_prompts = (prompt, prompt[:7])
        isolated: list[tuple[tuple[int, ...], tuple[str, str]]] = []
        for index, candidate in enumerate(concurrent_prompts):
            request_id = f"isolated-{index}"
            engine.admit(
                self.prepared_input(engine, request_id, candidate),
                max_new_tokens=2,
                use_prefix_cache=False,
            )
            steps = engine.run_until_drained()
            digest = engine.terminal_digest(request_id)
            isolated.append(
                (
                    self.token_ids(steps, request_id),
                    (digest.cache_digest, digest.state_digest),
                )
            )

        concurrent_ids = tuple(f"concurrent-{index}" for index in range(len(concurrent_prompts)))
        for request_id, candidate in zip(concurrent_ids, concurrent_prompts, strict=True):
            engine.admit(
                self.prepared_input(engine, request_id, candidate),
                max_new_tokens=2,
                use_prefix_cache=False,
            )
        concurrent_steps = engine.run_until_drained()
        self.assertTrue(
            any(
                step.batch_plan.shape.logical_m == 16 and len(step.batch_plan.segments) == 2
                for step in concurrent_steps
            )
        )
        self.assertTrue(any(step.batch_plan.phase is BatchPhase.MIXED for step in concurrent_steps))
        for request_id, (expected_tokens, expected_digests) in zip(
            concurrent_ids,
            isolated,
            strict=True,
        ):
            self.assertEqual(self.token_ids(concurrent_steps, request_id), expected_tokens)
            actual_digest = engine.terminal_digest(request_id)
            self.assertEqual(actual_digest.cache_digest, expected_digests[0])
            self.assertEqual(actual_digest.state_digest, expected_digests[1])


if __name__ == "__main__":
    unittest.main()
