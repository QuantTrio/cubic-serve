from __future__ import annotations

import pickle
import sys
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from cubic_serve.executor import vision
from cubic_serve.models.loader import load_model_definition
from cubic_serve.inputs.multimodal import EncoderWorkingSet

ENTRY = '''from dataclasses import dataclass, replace
from types import SimpleNamespace
from cubic_serve.models.qwen3_5 import MODEL_DEFINITION as BASE
@dataclass(frozen=True)
class Config:
    value: int
class Encoder:
    max_input_rows = 4
    resident_graph_bytes = 128
    max_dynamic_input_rows = 16
    peak_input_bytes = 256
    resources = SimpleNamespace(staging_bytes=64)
    def __init__(self, config):
        self.value = config.value
        self.closed = None
        self.namespace = __name__
    def close(self):
        from .payload import VALUE
        self.closed = VALUE
def factory(*, config, device, tp_rank, tp_group):
    return Encoder(config)
MODEL_DEFINITION = replace(BASE, model_type="owned_encoder",
    parse_config=lambda raw:Config(raw['value']))
'''


def _plain_factory(**kwargs):
    return SimpleNamespace(
        max_input_rows=4, resident_graph_bytes=128, max_dynamic_input_rows=16,
        peak_input_bytes=256, resources=SimpleNamespace(staging_bytes=64),
    )


def _group(factory, config, source=None, *, colocated=False):
    context = Mock()
    context.Pipe.side_effect = lambda: (Mock(), Mock())
    context.Process.return_value.pid = None
    context.Process.return_value.is_alive.return_value = False
    with (
        patch.object(vision.multiprocessing, "get_context", return_value=context),
        patch.object(vision.InputEncoderGroup, "_bind"),
    ):
        return vision.InputEncoderGroup(
            factory, devices=(0,), config=config, semantic_digest="weights",
            segmenter=None, module="fixture", input_width=3, output_width=5,
            merge_extent=1, colocated=colocated, model_definition_source=source,
        )


def _run_worker(metadata_path, connection, serve):
    with ExitStack() as stack:
        for target, name in (
            (vision.signal, "signal"), (vision.torch.cuda, "set_device"),
            (vision.torch.distributed, "init_process_group"),
            (vision.torch.distributed, "destroy_process_group"),
        ):
            stack.enter_context(patch.object(target, name))
        stack.enter_context(
            patch.object(vision.torch.distributed, "is_initialized", return_value=True),
        )
        stack.enter_context(patch.object(vision, "configure_reference_math"))
        stack.enter_context(patch("cubic_serve.runtime.logging.configure_logging"))
        stack.enter_context(patch.object(vision, "serve_input_encoder", side_effect=serve))
        vision._encoder_rank(str(metadata_path), 0, 0, connection, "file:///unused", 1)


@pytest.mark.parametrize("fails", (False, True))
def test_owned_encoder_source_covers_factory_forward_and_lazy_cleanup(tmp_path, fails):
    package = tmp_path / "adapter"
    package.mkdir()
    (package / "__init__.py").write_text(ENTRY)
    (package / "payload.py").write_text("VALUE = 17\n")
    cuda_before = torch.cuda.is_initialized()
    with load_model_definition(package) as source:
        namespace = source.module_name
        config = source.definition.parse_config({"value": 5})
        factory = partial(sys.modules[namespace].factory, config=config)
        group = _group(factory, config, source)
        try:
            assert group.weights.semantic_plan_digest != "weights"
            saved = (Path(group._directory.name) / "encoder.pkl").read_bytes()
        finally:
            group.close()
    metadata = tmp_path / "worker.pkl"
    metadata.write_bytes(saved)
    observed = []

    def serve(encoder, connection, rank):
        assert namespace in sys.modules
        observed.append(encoder)
        assert encoder.value == 5 and encoder.closed is None
        (package / "payload.py").write_text("VALUE = 29\n")
        if fails:
            raise RuntimeError("private encoder diagnostic")

    connection = Mock()
    if fails:
        with pytest.raises(RuntimeError, match="private encoder diagnostic"):
            _run_worker(metadata, connection, serve)
    else:
        _run_worker(metadata, connection, serve)
    assert len(observed) == 1 and observed[0].closed == 17
    assert not any(n == namespace or n.startswith(namespace + ".") for n in sys.modules)
    assert connection.send.call_args_list[0].args[0] == ("ready", 4, 64, 128, 16, 256)
    if fails:
        assert connection.send.call_args_list[-1].args[0][0] == "failure"
    connection.close.assert_called_once_with()
    assert torch.cuda.is_initialized() == cuda_before


def test_native_encoder_metadata_retains_original_direct_factory_bytes(tmp_path):
    factory = partial(_plain_factory, setting=3)
    group = _group(factory, {"value": 5})
    try:
        with open(group._directory.name + "/encoder.pkl", "rb") as stream:
            assert stream.read() == pickle.dumps(factory, protocol=pickle.HIGHEST_PROTOCOL)
        assert group.weights.semantic_plan_digest == "weights"
        assert not hasattr(group, "model_program_recipe")
        assert not hasattr(group, "model_definition_namespace")
    finally:
        group.close()


def test_colocated_source_recipe_mismatch_fails_before_group_or_factory(tmp_path):
    package = tmp_path / "adapter"
    package.mkdir()
    (package / "__init__.py").write_text(ENTRY)
    (package / "payload.py").write_text("VALUE = 17\n")
    with load_model_definition(package) as source:
        config = source.definition.parse_config({"value": 5})
        admission = SimpleNamespace(
            input_encoder_factory=Mock(), model_definition_source=source,
            input_encoder_model_config=config,
            input_encoder_program_recipe=source.execution_recipe(
                source.definition.parse_config({"value": 6}),
            ),
        )
        with (
            patch.object(torch.distributed, "new_group") as group,
            pytest.raises(ValueError, match="recipe"),
        ):
            vision.prepare_owned_input_encoder(admission, topology=None, rank=0, device=None)
        group.assert_not_called()
        admission.input_encoder_factory.assert_not_called()


@pytest.mark.parametrize("requested", [None, 4 << 30])
def test_memory_log_uses_launch_flag_and_reports_effective_bytes(caplog, requested):
    plan = EncoderWorkingSet(65536, 145664, 1, 4, 256)
    with caplog.at_level("INFO", logger=vision.__name__):
        vision._log_input_capacity(plan, 65536, {0: requested},
                                   {0: 1 << 28}, {0: (4 << 30) - (1 << 28)})
    expected = "--multimodal-memory-gib: 4.00 GiB/GPU"
    if requested is None:
        expected += " (Auto)"
    assert caplog.messages == [expected]


def test_explicit_memory_log_retains_requested_envelope(caplog):
    plan = EncoderWorkingSet(65536, 145664, 1, 4, 256)
    with caplog.at_level("INFO", logger=vision.__name__):
        vision._log_input_capacity(plan, 65536, {0: 16 << 30},
                                   {0: 1 << 28}, {0: 13 << 30})
    assert caplog.messages == ["--multimodal-memory-gib: 16.00 GiB/GPU"]


def test_final_frontend_limit_uses_smallest_rank_and_reports_reduced_area(caplog):
    plan = EncoderWorkingSet(65536, 145664, 1, 4, 256)
    group = SimpleNamespace(
        devices=(0, 1), max_dynamic_input_rows=65536,
        input_reserves_by_device={0: 13 << 30, 1: 13 << 30},
        _memory_requests={0: None, 1: None},
        _memory_residents={0: 1 << 28, 1: 1 << 28}, _working_set=plan,
    )
    with caplog.at_level("INFO", logger=vision.__name__):
        vision.InputEncoderGroup.limit_admission(group, {0: (32768, 2 << 30),
                                                        1: (16384, 1 << 30)})
    assert group.max_dynamic_input_rows == 16384
    assert group.input_reserves_by_device == {0: 2 << 30, 1: 1 << 30}
    assert caplog.messages[-1] == "image capacity: 4,194,304 pixels"
