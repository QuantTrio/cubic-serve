from __future__ import annotations

import hashlib
import os
import unittest
from dataclasses import replace
from unittest import mock

import torch

from cubic_serve.guardrail import attention_artifact as artifact_guardrail
from cubic_serve.guardrail.attention_artifact import (
    PAGED_GQA_ARTIFACT_SEMANTICS,
    AttentionArtifactRejected,
    BF16PagedGQAArtifactManifest,
    ReleaseBF16PagedGQAArtifact,
    StructuralAttentionArtifactWitness,
    verify_bf16_paged_gqa_artifact_launch,
)
from cubic_serve.ops.cuda.attention import (
    BF16_PAGED_GQA_OPERATOR_ID,
    BF16PagedGQAArtifactIdentity,
    BF16PagedGQADecodePrimitive,
    BF16PagedGQALaunchReceipt,
    BF16PagedGQATensorBinding,
    paged_decode,
)
from cubic_serve.ops.cuda.compiled_artifact import COMPILED_ARTIFACT_SCHEMA_ID

RUN_REAL = os.environ.get("CUBIC_RUN_CUDA_ATTENTION_ARTIFACT_TESTS") == "1"


def _digest(label: str) -> str:
    return "b2s256:" + hashlib.blake2s(label.encode()).hexdigest()


def _identity() -> BF16PagedGQAArtifactIdentity:
    return BF16PagedGQAArtifactIdentity(
        schema_id=COMPILED_ARTIFACT_SCHEMA_ID,
        operator_id=BF16_PAGED_GQA_OPERATOR_ID,
        compiled_hash=_digest("triton-cache-key"),
        source_hash=_digest("compiled-source-with-token-zero"),
        stage_hashes=tuple(
            (name, _digest(name)) for name in ("ttir", "ttgir", "llir", "ptx", "cubin")
        ),
        compiler_metadata_json='[["num_warps",4]]',
        specialization_json=(
            '[["LAYER",1],["SCALE","0x1.0000000000000p-2"],["TABLE_PAGES",2]]'
        ),
        target=("cuda", 120, 32),
        driver_version=13020,
        sm=(12, 0),
        compiler_flags=(
            ("debug", "false"),
            ("enable_fp_fusion", "true"),
            ("enable_reflect_ftz", "true"),
            ("num_ctas", "1"),
            ("num_stages", "2"),
            ("num_warps", "4"),
            ("sanitize_overflow", "true"),
        ),
        runtime=(("torch", "test"), ("triton", "test"), ("cuda", "test")),
        launch_signature=(
            "capacity=4,layers=2,layer=1,page_block=4,table_pages=2,"
            "query_heads=4,kv_heads=2,head_dim=16,block_d=16,block_n=16,"
            "scale=0x1.0000000000000p-2,mask=seq_len,acc=fp32,out=bf16"
        ),
        grid=(4, 4, 1),
    )


def _approval(identity: BF16PagedGQAArtifactIdentity) -> ReleaseBF16PagedGQAArtifact:
    return ReleaseBF16PagedGQAArtifact(
        "release:test",
        BF16_PAGED_GQA_OPERATOR_ID,
        identity.artifact_digest,
        identity.launch_specialization_digest,
        PAGED_GQA_ARTIFACT_SEMANTICS,
    )


def _manifest(identity: BF16PagedGQAArtifactIdentity) -> BF16PagedGQAArtifactManifest:
    return BF16PagedGQAArtifactManifest("release:test", (_approval(identity),))


class AttentionArtifactManifestTests(unittest.TestCase):
    def test_content_address_covers_source_binary_specialization_and_grid(self) -> None:
        identity = _identity()
        self.assertEqual(
            artifact_guardrail._resolve_release_artifact(_manifest(identity), identity),
            _approval(identity),
        )
        changed_stages = tuple(
            (name, _digest("changed-cubin") if name == "cubin" else digest)
            for name, digest in identity.stage_hashes
        )
        faults = (
            replace(identity, compiled_hash=_digest("unknown-cache-key")),
            replace(identity, source_hash=_digest("compiled-source-omits-token-zero")),
            replace(identity, stage_hashes=changed_stages),
            replace(identity, specialization_json='[["LAYER",0]]'),
            replace(identity, grid=(3, 4, 1)),
        )
        manifest = _manifest(identity)
        for fault in faults:
            with self.subTest(fault=fault.artifact_digest), self.assertRaisesRegex(
                AttentionArtifactRejected, "not release-approved"
            ):
                artifact_guardrail._resolve_release_artifact(manifest, fault)

    def test_release_declares_exactly_structural_facts_and_target(self) -> None:
        identity = _identity()
        for facts in (
            PAGED_GQA_ARTIFACT_SEMANTICS[:-1],
            (*PAGED_GQA_ARTIFACT_SEMANTICS, ("numeric_correctness", "claimed")),
        ):
            approval = ReleaseBF16PagedGQAArtifact(
                "release:test",
                BF16_PAGED_GQA_OPERATOR_ID,
                identity.artifact_digest,
                identity.launch_specialization_digest,
                facts,
            )
            with self.assertRaisesRegex(AttentionArtifactRejected, "operator semantics"):
                artifact_guardrail._resolve_release_artifact(
                    BF16PagedGQAArtifactManifest("release:test", (approval,)), identity
                )
        with self.assertRaisesRegex(ValueError, "active SM"):
            replace(identity, target=("cuda", 90, 32))

    def test_negligible_token_zero_counterexample_is_rejected_by_source_identity(self) -> None:
        logits = torch.tensor([-80.0, 0.0], dtype=torch.float32)
        values = torch.tensor([128.0, 1.0], dtype=torch.float32)
        full = (torch.softmax(logits, 0) * values).sum().to(torch.bfloat16)
        omitted = values[1].to(torch.bfloat16)
        self.assertTrue(torch.equal(full, omitted))

        identity = _identity()
        omitted_source = replace(
            identity,
            source_hash=_digest("compiled-source-omits-token-zero"),
        )
        with self.assertRaisesRegex(AttentionArtifactRejected, "not release-approved"):
            artifact_guardrail._resolve_release_artifact(_manifest(identity), omitted_source)

    def test_fake_primitive_and_witness_reject(self) -> None:
        identity = _identity()

        class FakePrimitive:
            artifact = identity

        cpu_q = torch.zeros((1, 2, 4), dtype=torch.bfloat16)
        cpu_kv = torch.zeros((1, 1, 1, 1, 4), dtype=torch.bfloat16)
        cpu_table = torch.zeros((1, 1), dtype=torch.int32)
        cpu_lens = torch.ones((1,), dtype=torch.int32)
        with self.assertRaisesRegex(AttentionArtifactRejected, "production"):
            verify_bf16_paged_gqa_artifact_launch(
                primitive=FakePrimitive(),  # type: ignore[arg-type]
                query=cpu_q,
                key=cpu_kv,
                value=cpu_kv.clone(),
                page_table=cpu_table,
                seq_lens=cpu_lens,
                output=cpu_q.clone(),
                active_rows=(0,),
                layer=0,
                scale=0.5,
                manifest=_manifest(identity),
            )
        with self.assertRaisesRegex(TypeError, "verifier-issued"):
            StructuralAttentionArtifactWitness(
                _digest("artifact"),
                _digest("approval"),
                _digest("operands"),
                _digest("launch"),
                _digest("domain"),
                _digest("output"),
                object(),
            )

    def test_manifest_subclass_cannot_override_release_admission(self) -> None:
        with self.assertRaisesRegex(TypeError, "cannot be subclassed"):

            class FakeManifest(BF16PagedGQAArtifactManifest):
                pass

    def test_manifest_revalidates_mutated_release_semantics_on_consume(self) -> None:
        identity = _identity()
        manifest = _manifest(identity)
        object.__setattr__(manifest.entries[0], "semantic_facts", ())
        with self.assertRaisesRegex(AttentionArtifactRejected, "semantic facts"):
            artifact_guardrail._resolve_release_artifact(manifest, identity)

    def test_identity_rejects_malformed_digest_stage_flag_target_and_runtime(self) -> None:
        identity = _identity()
        faults = (
            {"source_hash": "raw-hash"},
            {"stage_hashes": identity.stage_hashes[:-1]},
            {"compiler_flags": tuple(reversed(identity.compiler_flags))},
            {"target": ("cuda", 90, 32)},
            {"runtime": (("torch", "test"), ("cuda", "test"))},
        )
        for fault in faults:
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                replace(identity, **fault)  # type: ignore[arg-type]

    def test_receipt_requires_exact_type_and_binds_stream(self) -> None:
        primitive = object.__new__(BF16PagedGQADecodePrimitive)
        primitive._artifact = object()
        primitive._last_bindings = ()
        primitive._signature = "signature"
        primitive._grid = (1, 1, 1)
        primitive._ordinal = 1
        primitive._last_stream = 17
        primitive._receipt_token = object()
        binding = BF16PagedGQATensorBinding(
            1, 1, 1, 0, (1,), (1,), "torch.bfloat16", "cuda:0"
        )
        primitive._last_bindings = (binding,)
        receipt = BF16PagedGQALaunchReceipt(
            id(primitive._artifact),
            primitive._last_bindings,
            primitive._signature,
            primitive._grid,
            primitive._last_stream,
            primitive._ordinal,
            primitive._receipt_token,
        )

        with self.assertRaisesRegex(TypeError, "cannot be subclassed"):

            class FakeReceipt(BF16PagedGQALaunchReceipt):
                pass
        with self.assertRaisesRegex(ValueError, "receipt"):
            primitive.validate_receipt(replace(receipt, stream=19))

    def test_primitive_has_no_instance_method_shadowing_surface(self) -> None:
        primitive = object.__new__(BF16PagedGQADecodePrimitive)
        self.assertFalse(hasattr(primitive, "__dict__"))
        with self.assertRaises(AttributeError):
            primitive.validate_operands = lambda *args: None  # type: ignore[method-assign]


@unittest.skipUnless(RUN_REAL, "set CUBIC_RUN_CUDA_ATTENTION_ARTIFACT_TESTS=1")
class RealAttentionArtifactLaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        self.device = torch.device("cuda:0")
        if torch.cuda.get_device_capability(self.device) != (12, 0):
            self.skipTest("real artifact qualification is pinned to SM120")
        torch.manual_seed(4049)
        self.query = torch.randn((4, 4, 16), device=self.device, dtype=torch.bfloat16)
        self.key = torch.randn((8, 2, 4, 2, 16), device=self.device, dtype=torch.bfloat16)
        self.value = torch.randn_like(self.key)
        self.table = torch.tensor(
            ((0, 1), (2, 3), (4, 5), (6, 7)), device=self.device, dtype=torch.int32
        )
        self.lengths = torch.ones((4,), device=self.device, dtype=torch.int32)
        self.output = torch.empty_like(self.query)
        self.primitive = BF16PagedGQADecodePrimitive(
            self.query,
            self.key,
            self.value,
            self.table,
            self.lengths,
            self.output,
            layer=1,
            scale=0.25,
        )
        self.manifest = _manifest(self.primitive.project_artifact())

    def _verify(self, active_rows: tuple[int, ...]) -> StructuralAttentionArtifactWitness:
        self.lengths.fill_(1)
        for offset, row in enumerate(active_rows):
            self.lengths[row] = (1, 4, 5, 8)[offset]
        return verify_bf16_paged_gqa_artifact_launch(
            primitive=self.primitive,
            query=self.query,
            key=self.key,
            value=self.value,
            page_table=self.table,
            seq_lens=self.lengths,
            output=self.output,
            active_rows=active_rows,
            layer=1,
            scale=0.25,
            manifest=self.manifest,
        )

    def test_retained_artifact_launches_logical_m1_m2_m4_without_jit(self) -> None:
        with mock.patch.object(
            paged_decode._decode,
            "run",
            side_effect=AssertionError("readiness-time JIT path was entered"),
        ):
            witnesses = tuple(self._verify(rows) for rows in ((2,), (0, 3), (0, 1, 2, 3)))
        self.assertTrue(all(witness.witness_digest.startswith("b2s256:") for witness in witnesses))
        self.assertEqual({witness.artifact_digest for witness in witnesses}, {
            self.primitive.project_artifact().artifact_digest
        })
        self.assertFalse(bool(torch.isnan(self.output).any().item()))

    def test_wrong_binding_alias_domain_and_specialization_reject_before_launch(self) -> None:
        with self.assertRaisesRegex(AttentionArtifactRejected, "exact bound"):
            verify_bf16_paged_gqa_artifact_launch(
                primitive=self.primitive,
                query=self.query.clone(),
                key=self.key,
                value=self.value,
                page_table=self.table,
                seq_lens=self.lengths,
                output=self.output,
                active_rows=(0, 1, 2, 3),
                layer=1,
                scale=0.25,
                manifest=self.manifest,
            )
        with self.assertRaisesRegex(ValueError, "non-aliasing"):
            BF16PagedGQADecodePrimitive(
                self.query,
                self.key,
                self.value,
                self.table,
                self.lengths,
                self.query,
                layer=1,
                scale=0.25,
            )

        self.lengths.fill_(1)
        self.table[0, 0] = self.key.shape[0]
        with mock.patch.object(
            self.primitive._artifact,
            "_run",
            side_effect=AssertionError("invalid domain reached launch"),
        ), self.assertRaisesRegex(AttentionArtifactRejected, "out of range"):
            self._verify((0,))
        self.table[0, 0] = 0

        old_grid = self.primitive._grid
        self.primitive._grid = (old_grid[0] - 1, old_grid[1], old_grid[2])
        with mock.patch.object(
            self.primitive._artifact,
            "_run",
            side_effect=AssertionError("unapproved grid reached launch"),
        ), self.assertRaisesRegex(AttentionArtifactRejected, "not release-approved"):
            self._verify((0,))
        self.primitive._grid = old_grid

    def test_omitted_token_source_and_changed_cubin_reject_before_launch(self) -> None:
        artifact = self.primitive._artifact
        source = artifact.asm["source"]
        artifact.asm["source"] = str(source) + "\n# fault: position zero omitted"
        with mock.patch.object(
            artifact,
            "_run",
            side_effect=AssertionError("unapproved source reached launch"),
        ), self.assertRaisesRegex(AttentionArtifactRejected, "not release-approved"):
            self._verify((0,))
        artifact.asm["source"] = source

        cubin = artifact.kernel
        changed = bytes([cubin[0] ^ 1]) + cubin[1:]
        artifact.kernel = changed
        artifact.asm["cubin"] = changed
        with mock.patch.object(
            artifact,
            "_run",
            side_effect=AssertionError("unapproved cubin reached launch"),
        ), self.assertRaisesRegex(AttentionArtifactRejected, "not release-approved"):
            self._verify((0,))
        artifact.kernel = cubin
        artifact.asm["cubin"] = cubin

    def test_stale_output_receipt_and_observation_reject(self) -> None:
        with mock.patch.object(self.primitive._artifact, "_run", new=lambda *args: None), (
            self.assertRaisesRegex(AttentionArtifactRejected, "stale")
        ):
            self._verify((0, 1, 2, 3))

        self.lengths.fill_(1)
        observation = artifact_guardrail._observe_bf16_paged_gqa_artifact_launch(
            primitive=self.primitive,
            query=self.query,
            key=self.key,
            value=self.value,
            page_table=self.table,
            seq_lens=self.lengths,
            output=self.output,
            active_rows=(0, 1, 2, 3),
            layer=1,
            scale=0.25,
            manifest=self.manifest,
        )
        self.output.add_(1.0)
        with self.assertRaisesRegex(AttentionArtifactRejected, "changed after collection"):
            observation.consume()

        receipt = self.primitive.launch(
            self.query, self.key, self.value, self.table, self.lengths, self.output
        )
        torch.cuda.synchronize(self.device)
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.primitive.validate_receipt(replace(receipt, ordinal=receipt.ordinal + 1))


if __name__ == "__main__":
    unittest.main()
