"""Hot route-completion plans bind a final decision without repricing."""
from types import SimpleNamespace

import pytest
import torch

from cubic_serve.ops.cuda import cache_identity
from cubic_serve.ops.cuda.launch_cost import RouteCompletionCost, TransferCost
from cubic_serve.ops.cuda.moe import route_completion as route
from cubic_serve.ops.cuda.startup import startup_weight_identity
from cubic_serve.ops.strategies import CollectiveParameters


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setenv("CUBIC_SERVE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache_identity, "gpu_model", lambda d: ("GPU model",))
    monkeypatch.setattr(cache_identity, "cuda_libraries", lambda names: ("collective-image",))
    monkeypatch.setattr(cache_identity, "module_kernels", lambda m: ("route-image",))
    monkeypatch.setattr(route.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(route.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(route.dist, "all_reduce", lambda *args, **kwargs: None)
    tensor = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs: tensor(
        *args, **dict(kwargs, device="cpu")))
    monkeypatch.setattr(route, "partitioned_transport", lambda *args: True)
    template = SimpleNamespace(shape=(128, 8, 64), dtype=torch.bfloat16,
        device=torch.device("cuda", 0), element_size=lambda: 2)
    return template, object(), CollectiveParameters(route_completion=True)


def test_missing_cold_plan_does_not_benchmark_candidates(setup, monkeypatch):
    monkeypatch.setattr(route, "_calibrate", lambda *args: pytest.fail("normal startup tuned"))
    with startup_weight_identity(("weights",), allow_calibration=False):
        assert not route.RouteCompletionPlanner(*setup, {}).selected
        assert not route.RouteCompletionPlanner(*setup, {}).selected


def test_hot_final_decision_never_refits_or_predicts(setup, monkeypatch):
    cost = RouteCompletionCost(2, TransferCost(10, .001), TransferCost(20, .001),
                              TransferCost(20, .001), TransferCost(12, .0001), .02)
    monkeypatch.setattr(route, "_calibrate", lambda *args: cost)
    with startup_weight_identity(("weights",), allow_calibration=True):
        cold = route.RouteCompletionPlanner(*setup, {})
    monkeypatch.setattr(RouteCompletionCost, "predict", lambda *a, **kw: pytest.fail("hot repriced"))
    monkeypatch.setattr(RouteCompletionCost, "beneficial", lambda *a, **kw: pytest.fail("hot selected"))
    monkeypatch.setattr(route, "_calibrate", lambda *args: pytest.fail("hot tuned"))
    with startup_weight_identity(("other-weights",), allow_calibration=False):
        hot = route.RouteCompletionPlanner(*setup, {})
    assert hot.selected == cold.selected
    assert hot.predicted_us == cold.predicted_us
