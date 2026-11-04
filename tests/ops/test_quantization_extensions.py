"""An out-of-tree extension must work without editing framework format branches."""

import os
import subprocess
import sys
import textwrap


def test_compute_mode_selection_cannot_silently_change_activation_precision():
    from types import SimpleNamespace
    from unittest.mock import patch

    import pytest

    from cubic_serve.quantization.registry import select_compute_mode

    encoding = SimpleNamespace(format="external-format")
    policy = SimpleNamespace(supports_compute_mode=lambda *args: True)
    extension = SimpleNamespace(module=lambda role: policy)
    geometry = dict(m=3, n=17, k=64, dtype=None, device=None)
    with patch("cubic_serve.quantization.registry.get_extension", return_value=extension):
        for mode in ("wna16", "wna8"):
            policy.select_compute_mode = lambda *args, selected=mode, **kw: selected
            assert select_compute_mode(encoding, "linear", mode, **geometry) == mode
        policy.select_compute_mode = lambda *args, **kw: "wna16"
        with pytest.raises(ValueError, match="undeclared packed compute mode"):
            select_compute_mode(encoding, "linear", "wna8", **geometry)


def test_wna8_reference_uses_format_owned_activation_boundary():
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch

    from cubic_serve.executor.runner import ReferenceExecutionContext
    from cubic_serve.ops.eager.quantization import linear_matrix
    from tests.ops.test_int_quantization import fixture

    # The declared boundary deliberately differs from ordinary token
    # quantization. The engine must invoke it rather than inherit INT math.
    operands = torch.tensor([[0.25, -0.5, 0.75, 1.0]], dtype=torch.float32)
    weight = torch.tensor([[1.0, 2.0, -1.0, 0.5]], dtype=torch.float32)
    reference = SimpleNamespace(wna8_operand=lambda x: operands, wna8_weight=lambda w: weight,
                                ROW_ALIGNMENT=1)
    extension = SimpleNamespace(module=lambda role: reference)
    matrix, _ = fixture(8, n=1, k=4, group=4, sym=True)
    inputs = torch.ones(1, 4, dtype=torch.bfloat16)
    with (
        ReferenceExecutionContext(),
        patch("cubic_serve.quantization.registry.select_compute_mode", return_value="wna8"),
        patch("cubic_serve.ops.eager.quantization.get_extension", return_value=extension),
    ):
        result = linear_matrix(inputs, matrix, compute_mode="wna8")
    assert torch.equal(result, (operands.double() @ weight.double().T).to(inputs.dtype))


def test_marlin_residency_limit_is_shared_and_keeps_format_contracts():
    import pytest

    from cubic_serve.quantization.extensions.awq import policy as awq
    from cubic_serve.quantization.extensions.fp8 import policy as fp8
    from cubic_serve.quantization.extensions.nvfp4 import policy as nvfp4

    for policy, operation in ((awq, "moe"), (fp8, "linear"), (nvfp4, "moe")):
        for value in (0, 1, 3):
            assert dict(policy.validate(operation, {"marlin_blocks_per_sm": value})) == {
                "marlin_blocks_per_sm": value,
            }
        for value in (True, -1, 1.5, "auto"):
            with pytest.raises(ValueError, match="marlin_blocks_per_sm"):
                policy.validate(operation, {"marlin_blocks_per_sm": value})
    with pytest.raises(ValueError, match="ordered K16"):
        nvfp4.validate("moe", {"marlin_blocks_per_sm": 0, "marlin_block_n": 64})


def test_packed_geometry_preserves_format_k_and_explicit_tiles():
    from unittest.mock import patch

    import torch

    from cubic_serve.ops.cuda.launch_cost import DeviceResources
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope
    from cubic_serve.quantization.common.policy import placement_candidates

    config = OperatorConfig.from_dict(
        {
            "version": 1,
            "overrides": {
                "nvfp4_moe": {"fixed_block_m": 32, "fixed_block_n": 64},
            },
        }
    )
    with (
        operator_strategy_scope(config, device="", model=""),
        patch(
            "cubic_serve.ops.cuda.launch_cost.device_resources",
            return_value=DeviceResources(80, 2048, 65536, 102400),
        ),
    ):
        params = config.resolve(
            "nvfp4_moe", device="", model="", n=512, k=1024, experts=64, top_k=8
        )
        points = placement_candidates(
            params,
            family="nvfp4_moe",
            device=torch.device("cuda:0"),
            rows=33,
            columns=512,
            reduction=1024,
            element_bytes=2,
            groups=64,
            experts=64,
            top_k=8,
        )
    assert points and all(p.fixed_block_k == 16 for p in points)
    assert all((p.fixed_block_m, p.fixed_block_n) == (32, 64) for p in points)


def test_installed_extension_owns_loading_reference_and_policy(tmp_path):
    package = tmp_path / "example_quant"
    package.mkdir()
    files = {
        "__init__.py": """
from cubic_serve.quantization.api import QuantizationExtension
EXTENSION = QuantizationExtension("example", "example-int8-v1", __name__, ("example",))
""",
        "checkpoint.py": """
from cubic_serve.models.quantization import EncodingRule, QuantizationScheme

def validate_scheme(scheme):
    if scheme.bits != 8:
        raise ValueError("example requires int8")

def parse_config(config):
    return (EncodingRule("example", ("Linear",), QuantizationScheme("example-int8-v1", 8, 1)),), ()
""",
        "layout.py": """
ROW_ALIGNMENT = 1
CONCAT_AXIS = 0
SCALAR_PAYLOADS = ()
def payload_headers(weight, encoding):
    return (("codes", "I8", weight.shape),)
def slice_index(shape, partitions, encoding, key, axis, start, end):
    index = [slice(None), slice(None)]
    index[axis] = slice(start, end)
    return tuple(index), ()
""",
        "reference.py": """
def decode(payload, encoding, shape, partitions):
    return payload["codes"].float() * 0.5
""",
        "policy.py": """
from cubic_serve.quantization.common.parameters import PackedLinearParameters, PackedMoEParameters
from dataclasses import dataclass
@dataclass(frozen=True)
class ExampleParameters(PackedLinearParameters):
    split_k: int = 1
    algorithm: str = "ordered"
PARAMETERS = {"linear": ExampleParameters, "moe": PackedMoEParameters}
FORBIDDEN_OVERRIDES = {"fixed_block_k": "example fixes K"}
def validate(operation, values):
    from cubic_serve.quantization.common.policy import validate_tiles
    values = dict(values)
    extra = {}
    for key, allowed in (("split_k", (1, 2, 4)), ("algorithm", ("ordered", "tiled"))):
        if key in values:
            extra[key] = values.pop(key)
            if operation != "linear" or extra[key] not in allowed:
                raise ValueError("invalid example parameter")
    checked = validate_tiles(PARAMETERS[operation], values, FORBIDDEN_OVERRIDES)
    return tuple(sorted((*checked, *extra.items())))
""",
    }
    for name, source in files.items():
        (package / name).write_text(textwrap.dedent(source))
    metadata = tmp_path / "example_quant-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: example-quant\nVersion: 1.0\n")
    (metadata / "entry_points.txt").write_text(
        "[cubic_serve.quantization]\nexample = example_quant:EXTENSION\n"
    )
    script = """
import sys, pickle
import torch
from cubic_serve.models.quantization import QuantizationConfig
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.eager.quantization import decode_matrix, embedding_matrix
from cubic_serve.ops.strategies import OperatorConfig
from cubic_serve.quantization.registry import get_extension, register
config = QuantizationConfig.from_model_config({"quantization_config": {"quant_method": "example"}})
scheme = config.resolve("model.projection")
assert scheme.format == "example-int8-v1"
x = torch.arange(32, dtype=torch.int8).reshape(4, 8)
w = QuantizedMatrix({"codes": x}, scheme, (4, 8))
assert torch.equal(decode_matrix(w), x.float() * 0.5)
assert torch.equal(decode_matrix(w.columns(2, 6)), x[:, 2:6].float() * 0.5)
actual = embedding_matrix(torch.tensor([3, 1]), w, dtype=torch.float16)
assert torch.equal(actual, x[[3, 1]].half() * 0.5)
policy = OperatorConfig.from_dict({"version": 1, "overrides": {"example_linear": {
    "fixed_block_n": 64, "split_k": 2, "algorithm": "tiled"}}})
resolved = pickle.loads(pickle.dumps(policy)).resolve(
    "example_linear", device="", model="", n=4, k=8)
assert (resolved.fixed_block_n, resolved.split_k, resolved.algorithm) == (64, 2, "tiled")
try:
    OperatorConfig.from_dict({"version": 1, "overrides": {"example_moe": {"fixed_block_k": 32}}})
except ValueError as e:
    assert "example fixes K" in str(e)
else:
    raise AssertionError("format-owned constraint was lost")
try:
    register(get_extension(scheme.format))
except ValueError:
    pass
else:
    raise AssertionError("duplicate extension silently overwrote a binding")
assert not any(name.startswith("triton") for name in sys.modules)
print("external extension passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={
            **os.environ,
            "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", "src"),
        },
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_callback_identity_is_independent_of_process_addresses():
    from cubic_serve.quantization.common.binding import constant_identity
    from cubic_serve.quantization.extensions.awq.kernels import load_weight as awq
    from cubic_serve.quantization.extensions.cubic.kernels import load_weight as cubic

    first, second = constant_identity((awq,)), constant_identity((cubic,))
    assert first != second
    assert "0x" not in repr(first)
    assert first[0][2] == awq.cache_key


def test_serving_stage_rejects_resource_retarget_and_preserves_pipeline_ownership():
    from dataclasses import replace
    from types import SimpleNamespace

    import pytest

    from cubic_serve.contracts.execution import RankReplayResources
    from cubic_serve.executor.quantized_model_runner import QuantizedDecodeStage
    from cubic_serve.models.parallel_weights import AxisSpan

    stage = object.__new__(QuantizedDecodeStage)
    stage.resource = RankReplayResources(0, "graph", "stream", "buffers", "workspace", "scratch")
    stage.captured_m = 8
    stage._vocab_span = AxisSpan(0, 16)
    stage.first, stage.last = True, False
    stage.embedding_mask = None
    stage.feature_rows = None
    stage._graph = None
    calls = []
    logits = object()
    stage.body = SimpleNamespace(
        prepare_capture_scratch=calls.append, buffers=SimpleNamespace(logits=logits)
    )
    with pytest.raises(ValueError, match="retarget"):
        stage.prepare_scratch(replace(stage.resource, graph_instance_id="another"))
    assert calls == []
    stage.prepare_scratch(stage.resource)
    assert calls == [stage.resource]
    for count in (0, 9, True):
        with pytest.raises(ValueError, match="logical rows"):
            stage.set_active_rows(count)
    stage.set_active_rows(3)
    assert stage.embedding_span == AxisSpan(0, 16)
    assert stage.logits is None and stage.vocab_span is None
    stage.first, stage.last = False, True
    assert stage.embedding_span is None
    assert stage.logits is logits and stage.vocab_span == AxisSpan(0, 16)
    with pytest.raises(ValueError, match="startup-captured"):
        _ = stage.graph


def test_serving_stage_replay_uses_its_bound_stream():
    from contextlib import contextmanager
    from types import SimpleNamespace
    from unittest.mock import patch

    from cubic_serve.executor.quantized_model_runner import QuantizedDecodeStage

    stage = object.__new__(QuantizedDecodeStage)
    stage.stream = object()
    events = []

    @contextmanager
    def stream_scope(stream):
        assert stream is stage.stream
        events.append("enter")
        yield
        events.append("exit")

    stage._graph = SimpleNamespace(replay=lambda: events.append("replay"))
    stage._overlap_resources = None
    stage._use_overlap = False
    with patch("torch.cuda.stream", stream_scope):
        stage.replay()
    assert events == ["enter", "replay", "exit"]
    stage._overlap_resources = (SimpleNamespace(replay=lambda: events.append("overlap")),)
    stage.select_decode_overlap(True)
    with patch("torch.cuda.stream", stream_scope):
        stage.replay()
    assert events[-3:] == ["enter", "overlap", "exit"]
    stage.select_decode_overlap(False)
    with patch("torch.cuda.stream", stream_scope):
        stage.replay()
    assert events[-3:] == ["enter", "replay", "exit"]
