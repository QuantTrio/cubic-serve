"""Final row recipes are portable decisions, not warm-start timing fits."""
from types import SimpleNamespace

import pytest
import torch

from cubic_serve.ops.cuda import cache_identity
from cubic_serve.ops.cuda.startup import startup_weight_identity
from cubic_serve.quantization.common import small_rows_cost as selection


@pytest.fixture
def operands(monkeypatch, tmp_path):
    monkeypatch.setenv("CUBIC_SERVE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache_identity, "gpu_model", lambda d: ("GPU model",))
    monkeypatch.setattr(selection, "_recipe_kernels", lambda linear: ("row-kernel", "decoder"))
    monkeypatch.setattr(selection, "_plan_key", lambda *args: ("wna16", "bf16", (64, 128)))
    def forbidden(*args, **kwargs):
        pytest.fail("normal cold/hot startup fitted or predicted competing row kernels")
    monkeypatch.setattr(selection, "_choose_and_record", forbidden)
    monkeypatch.setattr(selection, "_fit", forbidden)
    monkeypatch.setattr(selection, "_predict", forbidden)
    owner = SimpleNamespace(config=SimpleNamespace(compute_mode="wna16"))
    x = SimpleNamespace(device=torch.device("cuda", 0), shape=(1, 128),
                        dtype=torch.bfloat16, data_ptr=lambda: 0)
    linear = SimpleNamespace(row_contraction=SimpleNamespace(
        prepare=lambda *args: "encoded-weight", supported=lambda w: True))
    weight = SimpleNamespace(shape=(64, 128), encoding="int4", partitions=(64,))
    return (owner, x, (weight,), (SimpleNamespace(dtype=x.dtype),),
            SimpleNamespace(marlin_enabled=True), linear)


def test_cold_missing_row_recipe_uses_and_reuses_default_without_search(operands):
    with startup_weight_identity(("weights",), allow_calibration=False):
        assert selection.choose(*operands) is None
        assert selection.choose(*operands) is None
    assert tuple(operands[0]._packed_row_plans.values()) == (("baseline", 0, 0),)


def test_explicit_selection_is_persisted_and_warm_binds_it_without_fit(operands, monkeypatch):
    owner, x, weights, outputs, params, linear = operands
    key = selection._plan_key(owner, x, weights, params)
    def tune(*args):
        owner._packed_row_plans = {(key, 1): ("simt", 4, 2)}
        return "tuned-binding"
    with startup_weight_identity(("weights",), allow_calibration=False):
        assert selection.choose(*operands) is None
    owner = SimpleNamespace(config=owner.config)
    operands = (owner, *operands[1:])
    monkeypatch.setattr(selection, "_choose_and_record", tune)
    with startup_weight_identity(("weights",), allow_calibration=True):
        assert selection.choose(*operands) == "tuned-binding"
    monkeypatch.setattr(selection, "_choose_and_record", lambda *args: pytest.fail("retimed"))
    monkeypatch.setattr(selection, "bind", lambda *a, **kw: (a[1], kw))
    fresh_owner = SimpleNamespace(config=owner.config)
    with startup_weight_identity(("another-checkpoint",), allow_calibration=False):
        result = selection.choose(fresh_owner, x, weights, outputs, params, linear)
    assert result == (("encoded-weight",), {"warps": 4, "partitions": 2})


def test_hot_recipe_does_not_bypass_operand_alignment(operands, monkeypatch):
    owner, x, weights, outputs, params, linear = operands
    key = selection._plan_key(owner, x, weights, params)
    def tune(*args):
        owner._packed_row_plans = {(key, 1): ("simt", 4, 2)}
        return "selected"
    monkeypatch.setattr(selection, "_choose_and_record", tune)
    with startup_weight_identity(("weights",), allow_calibration=True):
        selection.choose(*operands)
    monkeypatch.setattr(selection, "bind", lambda *args, **kw: pytest.fail("unaligned kernel"))
    x.data_ptr = lambda: 2
    fresh_owner = SimpleNamespace(config=owner.config)
    with startup_weight_identity(("weights",), allow_calibration=False):
        assert selection.choose(fresh_owner, x, weights, outputs, params, linear) is None


def test_corrupt_recipe_automatically_rebuilds_only_its_default(operands, tmp_path):
    with startup_weight_identity(("weights",), allow_calibration=False):
        selection.choose(*operands)
        path = next(tmp_path.rglob("*.json"))
        path.write_text("{partial")
        fresh = SimpleNamespace(config=operands[0].config)
        assert selection.choose(fresh, *operands[1:]) is None


@pytest.mark.parametrize("recipe", [("simt", 4., 1), ("simt", 4, True), ("mma", 0., 0)])
def test_noninteger_recipe_fields_are_cache_misses(operands, tmp_path, recipe):
    from cubic_serve.ops.cuda.launch_cost import _write_startup_calibration
    with startup_weight_identity(("weights",), allow_calibration=False):
        selection.choose(*operands)
        path = next(tmp_path.rglob("*.json"))
        _write_startup_calibration(path, "cubic.row-recipe.v1", {"recipe": recipe})
        fresh = SimpleNamespace(config=operands[0].config)
        assert selection.choose(fresh, *operands[1:]) is None


def test_another_rank_cannot_revise_a_bound_owners_arithmetic(operands, monkeypatch, tmp_path):
    from cubic_serve.ops.cuda.launch_cost import _write_startup_calibration
    owner, x, weights, outputs, params, linear = operands
    key = selection._plan_key(owner, x, weights, params)
    def tune(*args):
        owner._packed_row_plans = {(key, 1): ("simt", 4, 1)}
        return "selected"
    monkeypatch.setattr(selection, "_choose_and_record", tune)
    with startup_weight_identity(("weights",), allow_calibration=True):
        selection.choose(*operands)
    path = next(tmp_path.rglob("*.json"))
    _write_startup_calibration(path, "cubic.row-recipe.v1", {"recipe": ("simt", 8, 2)})
    monkeypatch.setattr(selection, "bind", lambda *a, **kw: (kw["warps"], kw["partitions"]))
    with startup_weight_identity(("weights",), allow_calibration=False):
        assert selection.choose(*operands) == (4, 1)
        fresh = SimpleNamespace(config=owner.config)
        assert selection.choose(fresh, x, weights, outputs, params, linear) == (8, 2)


def test_concurrent_cold_owners_share_one_selection(operands, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Event, local
    owner, x, weights, outputs, params, linear = operands
    entered, release = Event(), Event()
    start = Barrier(3)
    both_missed = Barrier(2)
    thread = local()
    read = selection._read_startup_calibration
    def initial_miss(*args):
        value = read(*args)
        if not getattr(thread, "read", False):
            thread.read = True
            both_missed.wait(timeout=3)
        return value
    monkeypatch.setattr(selection, "_read_startup_calibration", initial_miss)
    calls = []
    def tune(own, *args):
        calls.append(own)
        entered.set()
        assert release.wait(3)
        key = selection._plan_key(own, x, weights, params)
        own._packed_row_plans = {(key, 1): ("simt", 4, 2)}
        return "bound"
    monkeypatch.setattr(selection, "_choose_and_record", tune)
    monkeypatch.setattr(selection, "bind", lambda *a, **kw: "bound")
    owners = [SimpleNamespace(config=owner.config) for _ in range(2)]
    def bind(own):
        start.wait()
        with startup_weight_identity(None, allow_calibration=True):
            return selection.choose(own, x, weights, outputs, params, linear)
    with ThreadPoolExecutor(2) as pool:
        pending = [pool.submit(bind, own) for own in owners]
        start.wait()
        assert entered.wait(3)
        release.set()
        assert [future.result(timeout=3) for future in pending] == ["bound", "bound"]
    assert len(calls) == 1
    assert owners[0]._packed_row_plans == owners[1]._packed_row_plans
