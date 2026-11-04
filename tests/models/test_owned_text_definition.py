from __future__ import annotations

import json
import pickle
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from cubic_serve.executor.quantized_model_runner import QuantizedDecodeStage
from cubic_serve.models.builtin import BuiltinModelConfigError
from cubic_serve.models.loader import load_model_definition
from cubic_serve.runtime import quantized_graph as runtime
from cubic_serve.runtime.quantized_qualification import execution_identity
from tests.models.qwen3_5.test_config import _fixture


class _Checkpoint:
    quantization = None
    semantic_digest = "checkpoint"

    def validate_for(self, root, logical):
        self.logical = logical

    def payload_view(self):
        return SimpleNamespace(semantic_digest=self.semantic_digest, inventory="inventory")


class _Controller:
    ready = True
    _page_capacity = 32
    _block_size = 16
    _processes = ()

    def __init__(self, admission, **kwargs):
        self.admission = admission
        self.max_sequence_tokens = admission.max_sequence_tokens
        self.input_encoder_connections = {}

    def close(self):
        pass


def _executor(root, checkpoint, *, source=None, depth=0, encoder=None, cache_storage=None):
    (root / "generation_config.json").write_text(
        json.dumps({"do_sample": False}),
    )
    with patch.object(runtime, "_ProcessController", _Controller):
        return runtime.TextGraphExecutor(
            str(root), devices=(0,), tp=1, pp=1, ep=1, max_requests=4,
            max_context=64, prefill_chunk=None, operator_config=None,
            eos_token_ids=(), execution_mode="cuda", checkpoint=checkpoint,
            num_speculative_tokens=depth, model_definition_source=source,
            input_encoder=encoder,
            cache_storage=cache_storage,
        )


ENTRY = '''from dataclasses import dataclass, replace
from cubic_serve.models.qwen3_5 import MODEL_DEFINITION as BASE
@dataclass(frozen=True)
class Config:
    identity: object
    graph: object
    @property
    def text_config(self):
        return self.graph.text
    def architecture_plan(self):
        return self.graph.architecture_plan()
def parse(raw):
    native = BASE.parse_config(dict(raw, model_type="qwen3_5"))
    return Config(native.identity, native.graph)
MODEL_DEFINITION = replace(BASE, model_type="owned_dense", parse_config=parse,
    text_model=lambda **kwargs: BASE.text_model(**kwargs))
'''


@pytest.mark.parametrize("depth", (0, 1, 2, 3))
def test_owned_config_planning_and_draft_restore_without_builtin_type_lookup(tmp_path, depth):
    raw = _fixture()
    raw["model_type"] = "owned_dense"
    (tmp_path / "config.json").write_text(json.dumps(raw))
    package = tmp_path / "adapter"
    package.mkdir()
    (package / "__init__.py").write_text(ENTRY)
    with pytest.raises(BuiltinModelConfigError):
        _executor(tmp_path, _Checkpoint(), depth=depth)
    with load_model_definition(package) as source:
        checkpoint = _Checkpoint()
        executor = _executor(tmp_path, checkpoint, source=source, depth=depth)
        admission = executor._controller.admission
        assert admission.config.__class__.__module__ == source.module_name
        assert checkpoint.logical == source.definition.logical_weights(admission.config)
        assert admission.model_program_recipe == source.execution_recipe(admission.config)
        assert executor.semantic_digest != checkpoint.semantic_digest
        assert len(admission.graph_capacities) == (3 if depth == 0 else (4 if depth == 1 else 5))
        if depth:
            assert admission.draft_provider.spec.max_depth == depth
        binding = source.definition.text_graph(
            admission.config, admission.parallel_plan.topology, block_size=16,
            recurrent_state_dtype="bfloat16", definition=source.definition,
        )
        assert isinstance(binding.stage_type, partial)
        assert binding.stage_type.func is QuantizedDecodeStage
        assert binding.stage_type.keywords["model_factory"] is source.definition.text_model
        saved_config = pickle.dumps(admission.config)
        saved_identity = execution_identity(admission).parallel_plan.semantic_plan_digest
        payload = pickle.dumps(source.bind_metadata(admission))
        executor.close()
    restored = pickle.loads(payload)
    with restored.model_definition_source:
        assert pickle.dumps(restored.config) == saved_config
        assert restored.model_program_recipe == restored.model_definition_source.execution_recipe(
            restored.config,
        )
        assert execution_identity(restored).parallel_plan.semantic_plan_digest == saved_identity


def test_default_admission_has_no_source_wrapper_and_retains_original_identity(tmp_path):
    from dataclasses import asdict

    from cubic_serve.guardrail.compare import stable_digest

    (tmp_path / "config.json").write_text(json.dumps(_fixture()))
    executor = _executor(tmp_path, _Checkpoint(), depth=1)
    try:
        admission = executor._controller.admission
        assert not hasattr(admission, "model_definition_source")
        assert not hasattr(admission, "model_program_recipe")
        assert executor.semantic_digest == "checkpoint"
        ownership = tuple(
            tuple(asdict(binding) for binding in rank) for rank in admission.parallel_plan.bindings
        )
        assert execution_identity(admission).parallel_plan.semantic_plan_digest == stable_digest({
            "checkpoint": "checkpoint", "ownership": ownership,
        })
    finally:
        executor.close()


def test_cache_storage_is_shared_engine_identity_and_aligns_pages_without_model_options(tmp_path):
    from cubic_serve.runtime.cache.quantization import (
        CacheDType, CacheGroupAxis, CacheStorageSpec, KVCacheStorageSpec,
    )

    (tmp_path / "config.json").write_text(json.dumps(_fixture()))
    default = _executor(tmp_path, _Checkpoint(), cache_storage=KVCacheStorageSpec(), depth=1)
    fp8_storage = KVCacheStorageSpec(
        CacheStorageSpec(CacheDType.FP8, CacheGroupAxis.TOKENS, 24),
        CacheStorageSpec(CacheDType.INT6, CacheGroupAxis.CHANNELS),
    )
    encoded = _executor(tmp_path, _Checkpoint(), cache_storage=fp8_storage, depth=1)
    try:
        ordinary, quantized = default._controller.admission, encoded._controller.admission
        assert not hasattr(ordinary, "cache_storage")
        assert default.semantic_digest == "checkpoint"
        assert quantized.cache_storage == fp8_storage
        assert quantized.block_size_tokens % 24 == 0
        assert quantized.draft_provider is not None
        assert encoded.semantic_digest != default.semantic_digest
        assert execution_identity(quantized).parallel_plan.semantic_plan_digest != (
            execution_identity(ordinary).parallel_plan.semantic_plan_digest)
        assert execution_identity(quantized).parallel_plan.weight_plan_digest == (
            execution_identity(ordinary).parallel_plan.weight_plan_digest)
        restored = pickle.loads(pickle.dumps(quantized))
        assert restored.cache_storage == quantized.cache_storage
        assert execution_identity(restored).parallel_plan.semantic_plan_digest == (
            execution_identity(quantized).parallel_plan.semantic_plan_digest)
    finally:
        default.close()
        encoded.close()


def test_colocated_encoder_config_is_bound_to_the_same_owned_namespace(tmp_path):
    raw = _fixture()
    raw["model_type"] = "owned_dense"
    (tmp_path / "config.json").write_text(json.dumps(raw))
    package = tmp_path / "adapter"
    package.mkdir()
    (package / "__init__.py").write_text(ENTRY)
    with load_model_definition(package) as source, load_model_definition(package) as other:
        encoder = SimpleNamespace(
            colocated=True, config=source.definition.parse_config(raw), factory=Mock(),
            model_definition_namespace=source.module_name,
            attach_main_workers=Mock(), close=Mock(),
        )
        executor = _executor(tmp_path, _Checkpoint(), source=source, encoder=encoder)
        try:
            admission = executor._controller.admission
            assert admission.input_encoder_model_config is encoder.config
            assert admission.input_encoder_program_recipe == source.execution_recipe(encoder.config)
            encoder.attach_main_workers.assert_called_once_with((), {})
        finally:
            executor.close()
        with pytest.raises(ValueError, match="different model definition"):
            _executor(tmp_path, _Checkpoint(), source=other, encoder=encoder)
