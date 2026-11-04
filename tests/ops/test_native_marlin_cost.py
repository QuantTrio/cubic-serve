"""Native placement prediction must price issue supply, cache and uncertainty."""

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
from cubic_serve.ops.strategies import OperatorConfig
from cubic_serve.quantization.common.marlin import cost
from cubic_serve.quantization.common.marlin.parameters import MarlinParameters


@pytest.fixture(autouse=True)
def explicit_cost_qualification():
    """These tests exercise the offline fitter, outside production startup."""
    from cubic_serve.ops.cuda.startup import startup_weight_identity
    with startup_weight_identity(None, allow_calibration=True):
        yield


def test_prebuild_plan_identity_tracks_included_headers_but_not_python(monkeypatch, tmp_path):
    from cubic_serve.ops.cuda.cache_identity import native_headers
    owner = tmp_path/'cost.py'
    owner.write_text("# presentation one\n")
    headers = tmp_path/'native'/'nested'
    headers.mkdir(parents=True)
    header = headers/'kernel.cuh'
    header.write_text("__device__ int value() { return 1; }\n")
    monkeypatch.setattr(cost, "__file__", str(owner))
    monkeypatch.setattr(cost.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="CUDA 13.0"))
    try:
        native_headers.cache_clear();cost._plan_compiler_identity.cache_clear()
        before = cost._plan_compiler_identity()
        owner.write_text("# changed logging\n")
        native_headers.cache_clear();cost._plan_compiler_identity.cache_clear()
        assert cost._plan_compiler_identity() == before
        header.write_text("__device__ int value() { return 2; }\n")
        native_headers.cache_clear();cost._plan_compiler_identity.cache_clear()
        assert cost._plan_compiler_identity() != before
    finally:
        native_headers.cache_clear();cost._plan_compiler_identity.cache_clear()


def test_fixed_partition_supply_prices_tail_work_and_preserves_groups():
    from cubic_serve.quantization.common.marlin.cost import fixed_k_partitions

    args = dict(
        n=4352,
        k=5120,
        group=128,
        block_n=64,
        block_k=128,
        sms=156,
        supply_factor=2.0,
        max_partitions=8,
    )
    # The first divisor above the supply target (5) makes the last SM execute
    # 24 K units instead of 20. Keep 4; a narrower N benefits from 8.
    assert fixed_k_partitions(**args) == 4
    assert fixed_k_partitions(**(args | {"n": 2560})) == 8
    # 136/156 SM coverage is cheaper than three short waves of 5 partitions.
    # A hard one-wave admission floor wrongly excluded this legal candidate.
    assert fixed_k_partitions(**(args | {"group": 512})) == 2
    assert fixed_k_partitions(**(args | {"n": 8704, "sms": 312})) == 4
    for sms in (16, 48, 96, 156, 300):
        for budget in (1, 3, 6, 8):
            p = fixed_k_partitions(**(args | {"sms": sms, "max_partitions": budget}))
            assert p <= budget and args["k"] % (p * args["group"]) == 0


@pytest.mark.parametrize("dense,active", [(False, 8), (True, 1)])
def test_untuned_partition_default_counts_actual_active_expert_supply(monkeypatch, dense, active):
    from cubic_serve.ops.cuda.startup import startup_weight_identity
    cost._select_dense_partition_model.cache_clear()
    monkeypatch.setattr(cost, "_plan_path", lambda *a: None)
    monkeypatch.setattr(cost, "_read_plan", lambda *a: None)
    monkeypatch.setattr(cost, "_write_plan", lambda *a, **kw: None)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: SimpleNamespace(multi_processor_count=156))
    seen = []
    def rule(**kwargs):
        seen.append(kwargs["n"])
        return 2
    monkeypatch.setattr(cost, "fixed_k_partitions", rule)
    spec = replace(cost.ProbeFormat(4, torch.float16, torch.float16, torch.float16, 0, 1, 1),
                   dense_linear=dense)
    with startup_weight_identity(("weights",), allow_calibration=False):
        selected = cost._select_dense_partition_model(((1, "a"), (2, "b")), MarlinParameters(),
            spec, torch.device("cuda", 0), 320, 2560, 128, 3, 128, 8)
    assert selected == 2 and seen == [320 * active]
    cost._select_dense_partition_model.cache_clear()


def test_issue_waves_and_cache_capacity_are_independent_factors():
    model = cost.NativeCost((1.0, 2.0, 3.0), 0.0, 8, 100, 16, 64, 64, 16, 100, 4)
    assert model.features(1, 0, 8) == (1.0, 0.0, 0.0)
    assert model.features(1, 0, 9) == (1.0, 1.0, 0.0)
    assert model.features(2, 0, 8)[2] > 0
    assert replace(model, l2_bytes=200).features(2, 0, 8)[2] == 0
    assert replace(model, sms=16).features(1, 0, 9)[1] == 0


def test_nonnegative_fit_predicts_a_shape_not_in_the_fit():
    points = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (1.0, 2.0, 3.0)]
    coefficients = cost.NativeCost.fit(points, [1.0, 3.0, 4.0, 14.0])
    assert coefficients == pytest.approx((1.0, 2.0, 3.0))
    assert sum(a * b for a, b in zip(coefficients, (1, 4, 2), strict=True)) == pytest.approx(15)
    with pytest.raises(ValueError):
        cost.NativeCost.fit(points, [1.0, 3.0, float("nan"), 14.0])


def test_expected_route_tiles_include_tail_beyond_the_mean():
    active, tiles = cost.expected_row_tiles(16, 16)
    assert active == pytest.approx(1 - math.exp(-16))
    assert 1.4 < tiles < 1.6
    assert cost.expected_row_tiles(0, 16) == (0, 0)
    # Large populations use the bounded rounding overhead, avoiding exp underflow.
    assert cost.expected_row_tiles(10000, 16)[1] == pytest.approx(625.46875)


def test_noisy_fit_protects_sparse_latency_and_is_time_unit_invariant():
    points = [
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 3.0, 0.0),
        (1.0, 7.0, 0.0),
        (1.0, 13.0, 1.0),
        (1.0, 26.0, 1.1),
        (1.0, 39.0, 1.2),
    ]
    times = [0.01, 0.01, 0.03, 0.06, 0.16, 0.2, 0.28]
    beta = cost.NativeCost.fit(points, times)
    assert (
        max(
            abs(sum(a * b for a, b in zip(row, beta, strict=True)) / t - 1)
            for row, t in zip(points, times, strict=True)
        )
        < 0.15
    )
    assert cost.NativeCost.fit(points, [1000 * t for t in times]) == pytest.approx(
        tuple(1000 * b for b in beta)
    )


def test_selector_respects_shape_overrides_and_prediction_uncertainty(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(shared_memory_per_block_optin=200000),
    )

    def calibrated(source, p, *args):
        return cost.NativeCost(
            (10.0 if p.marlin_block_m == 16 else 5.0, 0.0, 0.0),
            uncertainty[0],
            100,
            100000,
            p.marlin_block_m,
            p.marlin_block_n,
            256,
            128,
            100,
            4,
        )

    monkeypatch.setattr(cost, "calibrate", calibrated)
    spec = cost.ProbeFormat(4, torch.float16, torch.float32, torch.float16, 0, 1, 1)
    kw = dict(
        family="awq_moe",
        scope=dict(n=128, k=128, experts=4, top_k=2),
        source=lambda p: "",
        spec=spec,
        device=torch.device("cuda", 0),
        n=256,
        k=128,
        experts=4,
        group=32,
        rows=4,
        routed=True,
    )
    base = MarlinParameters()
    uncertainty = [0.01]
    assert cost.select(base, operators=PackedLinearOperators(), **kw).marlin_block_m == 32
    uncertainty[0] = 0.8
    assert cost.select(base, operators=PackedLinearOperators(), **kw) == base
    config = OperatorConfig.from_dict(
        {
            "version": 1,
            "profiles": [
                {"operator": "awq_moe", "shape": kw["scope"], "parameters": {"marlin_block_m": 16}}
            ],
        }
    )
    uncertainty[0] = 0.01
    assert cost.select(base, operators=PackedLinearOperators(config), **kw).marlin_block_m == 16


def test_full_capacity_gain_must_not_regress_lower_population(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(shared_memory_per_block_optin=200000),
    )

    def calibrated(source, p, *args):
        def predict(active, rows, tiles):
            return 10.0 if p.marlin_block_m == 16 else (5.0 if rows >= 64 else 15.0)

        return SimpleNamespace(error=0.01, predict=predict)

    monkeypatch.setattr(cost, "calibrate", calibrated)
    spec = cost.ProbeFormat(4, torch.float16, torch.float32, torch.float16, 0, 1, 1)
    base = MarlinParameters()
    result = cost.select(
        base,
        operators=PackedLinearOperators(),
        family="awq_moe",
        scope=dict(n=128, k=128),
        source=lambda p: "",
        spec=spec,
        device=torch.device("cuda", 0),
        n=256,
        k=128,
        experts=1,
        group=32,
        rows=64,
        routed=False,
    )
    assert result == base


def test_selector_does_not_underprice_observed_relative_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(shared_memory_per_block_optin=200000))
    base = replace(MarlinParameters(), marlin_stages=2,
                   marlin_max_error_percent=50, marlin_min_gain_percent=0)

    def calibrated(source, params, *args):
        # Candidate center * (1+error) = 9.52 incorrectly beats a 10ms
        # baseline. The observed-relative-error upper bound is 11.33ms.
        return SimpleNamespace(error=0 if params == base else .4,
                               predict=lambda *args: 10 if params == base else 6.8)

    monkeypatch.setattr(cost, "calibrate", calibrated)
    result = cost.select(
        base, operators=PackedLinearOperators(), family="awq_moe", scope=dict(n=256, k=128),
        source=lambda params: "", spec=cost.ProbeFormat(4, torch.float16, torch.float32,
                                                     torch.float16, 0, 1, 1),
        device=torch.device("cuda", 0), n=256, k=128, experts=4, group=32, rows=4, routed=True,
    )
    assert result == base


def test_column_scale_does_not_inherit_cyclic_group_pipeline_limits(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(shared_memory_per_block_optin=200000),
    )

    def calibrated(source, p, *args):
        return cost.NativeCost(
            (5.0 if p.marlin_block_m == 32 else 10.0, 0.0, 0.0),
            0.01,
            100,
            100000,
            p.marlin_block_m,
            p.marlin_block_n,
            256,
            4096,
            100,
            4,
        )

    monkeypatch.setattr(cost, "calibrate", calibrated)
    spec = cost.ProbeFormat(
        8, torch.float16, torch.float32, torch.float16, 0, 1, 1, column_scale=True
    )
    selected = cost.select(
        MarlinParameters(),
        operators=PackedLinearOperators(),
        family="fp8_linear",
        scope=dict(n=256, k=4096),
        source=lambda p: "",
        spec=spec,
        device=torch.device("cuda", 0),
        n=256,
        k=4096,
        experts=1,
        group=4096,
        rows=4,
        routed=False,
    )
    assert selected.marlin_block_m == 32


def test_fixed_partials_price_fractional_supply_and_merge_traffic():
    whole = cost.NativeCost((1.0, 2.0, 3.0), 0.0, 8, 1_000_000, 16, 64, 128, 128, 4096, 2)
    partial = replace(whole, fixed_partitions=4)
    # Two N tiles times four K partitions exactly supply one CTA per SM.
    assert partial.features(1, 1, 1)[1] == 0
    assert partial.features(1, 17, 2)[1] == 1
    assert partial.features(1, 1, 1)[2] > whole.features(1, 1, 1)[2]
    assert replace(partial, sms=6).features(1, 1, 1)[1] == pytest.approx(1 / 3)


@pytest.mark.parametrize("partitions,dense", [(2, False), (0, True), (2, True)])
def test_fixed_dense_capacity_selects_independently_of_smaller_graphs(
    monkeypatch, partitions, dense
):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(shared_memory_per_block_optin=200000),
    )

    def calibrated(source, p, *args):
        def predict(active, rows, tiles):
            return 10.0 if p.marlin_block_m == 16 else (5.0 if rows >= 64 else 15.0)

        return SimpleNamespace(error=0.01, predict=predict)

    monkeypatch.setattr(cost, "calibrate", calibrated)
    spec = cost.ProbeFormat(
        4,
        torch.float16,
        torch.float16,
        torch.float16,
        0,
        1,
        1,
        fixed_partitions=partitions,
        dense_linear=dense,
    )
    base = MarlinParameters(marlin_max_candidates=3)
    kwargs = dict(
        operators=PackedLinearOperators(),
        family="cubic_linear",
        scope=dict(n=256, k=128),
        source=lambda p: "",
        spec=spec,
        device=torch.device("cuda", 0),
        n=256,
        k=128,
        experts=1,
        group=32,
        routed=False,
    )
    assert cost.select(base, rows=64, **kwargs).marlin_block_m != 16
    assert cost.select(base, rows=32, **kwargs).marlin_block_m == 16
    assert cost.select(base, rows=1, **kwargs).marlin_block_m == 16


def test_partial_calibration_needs_within_tile_rows_to_identify_merge_cost():
    model = cost.NativeCost(
        (1.0, 2.0, 3.0),
        0.0,
        156,
        100_000_000,
        64,
        64,
        2560,
        5120,
        13_000_000,
        2,
        fixed_partitions=4,
    )
    boundary = [model.features(1, rows, math.ceil(rows / 64)) for rows in (1, 65, 129, 257)]
    # Boundary-only samples confound output bytes with the number of CTA waves.
    assert torch.linalg.matrix_rank(torch.tensor(boundary, dtype=torch.float64)) == 2
    independent = [*boundary, model.features(1, 64, 1)]
    assert torch.linalg.matrix_rank(torch.tensor(independent, dtype=torch.float64)) == 3


@pytest.mark.parametrize("sms,bits,group", [(24, 4, 128), (156, 8, 128), (96, 5, 80)])
def test_dense_partition_plan_uses_resources_and_original_group(monkeypatch, sms, bits, group):
    hw = SimpleNamespace(
        multi_processor_count=sms,
        max_threads_per_multi_processor=1536,
        shared_memory_per_multiprocessor=100000,
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: hw)
    observed = []

    def calibrated(source, params, spec, device, n, k, experts, quantum, samples):
        p = spec.fixed_partitions
        observed.append(p)
        assert source == f"partition={p}"
        assert spec.storage_bits == bits
        assert k % (p * math.lcm(group, params.marlin_block_k)) == 0
        # Synthetic measured costs have their optimum at a resource-derived
        # full wave. Exercise selection with no GPU calibration dependency.
        value = 1 + abs(math.log((n // params.marlin_block_n) * p / sms))
        return SimpleNamespace(error=0.01, predict=lambda *args: value)

    monkeypatch.setattr(cost, "calibrate", calibrated)
    cost._select_dense_partition_model.cache_clear()
    spec = cost.ProbeFormat(
        4 if bits <= 4 else 8,
        torch.bfloat16,
        torch.bfloat16,
        torch.float16,
        0,
        1,
        1,
        storage_bits=bits,
        dense_linear=True,
        group_metadata=(1, 0),
    )
    kwargs = dict(
        operators=PackedLinearOperators(),
        source=lambda p: f"partition={p}",
        spec=spec,
        device=torch.device("cuda", 0),
        n=320,
        k=10240,
        group=math.gcd(group, 128),
        partition_group=group,
    )
    params = MarlinParameters()
    selected = cost.plan_dense_partitions(params, **kwargs)
    assert len(observed) <= params.marlin_max_candidates
    assert selected == min(observed, key=lambda p: (abs(math.log(5 * p / sms)), p))
    if sms == 156:
        assert selected > 8
    calls = len(observed)
    # A sealed plan is reused, not refitted from a different M or live input.
    assert cost.plan_dense_partitions(params, **kwargs) == selected
    assert len(observed) == calls
    assert cost.plan_dense_partitions(params, max_partitions=1, **kwargs) == 1
    cost._select_dense_partition_model.cache_clear()


def test_dense_partition_uncertainty_and_cross_population_regression(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=16),
    )

    def calibrated(source, params, spec, *args):
        p = spec.fixed_partitions
        return SimpleNamespace(
            error=0.9 if p == 8 else 0.01,
            predict=lambda active, rows, tiles: (10.0 if p == 1 else (6.0 if rows == 1 else 30.0)),
        )

    monkeypatch.setattr(cost, "calibrate", calibrated)
    cost._select_dense_partition_model.cache_clear()
    result = cost._select_dense_partition_model(
        ((1, "one"), (4, "four"), (8, "eight")),
        MarlinParameters(),
        cost.ProbeFormat(
            4, torch.float16, torch.float16, torch.float16, 0, 1, 1, dense_linear=True
        ),
        torch.device("cuda", 0),
        128,
        4096,
        128,
        3,
    )
    assert result == 1
    cost._select_dense_partition_model.cache_clear()


@pytest.mark.parametrize("sms,experts,top_k", [(24, 8, 2), (156, 512, 10), (96, 64, 8)])
def test_routed_partition_plan_prices_expert_supply(monkeypatch, sms, experts, top_k):
    hw = SimpleNamespace(
        multi_processor_count=sms, max_threads_per_multi_processor=1536,
        shared_memory_per_multiprocessor=100000,
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: hw)
    observed = []
    workloads = []

    def calibrated(source, params, spec, device, n, k, count, quantum, samples):
        partitions = spec.fixed_partitions or 1
        observed.append(partitions)
        assert source == f"partition={spec.fixed_partitions}"
        assert count == experts
        assert k % (partitions * math.lcm(quantum, params.marlin_block_k)) == 0
        value = 1 + abs(math.log(n // params.marlin_block_n * top_k * partitions / sms))

        def predict(active, rows, tiles):
            workloads.append((active, rows, tiles))
            assert active == min(experts, rows)
            assert rows % top_k == 0
            assert tiles == active * math.ceil(rows / active / params.marlin_block_m)
            return value

        return SimpleNamespace(error=0.01, predict=predict)

    monkeypatch.setattr(cost, "calibrate", calibrated)
    cost._select_dense_partition_model.cache_clear()
    params = MarlinParameters()
    spec = cost.ProbeFormat(
        4, torch.bfloat16, torch.bfloat16, torch.float16, 0, 1, 1,
        group_metadata=(1,), storage_bits=4,
    )
    kwargs = dict(
        operators=PackedLinearOperators(), source=lambda p: f"partition={p if p > 1 else 0}",
        spec=spec, device=torch.device("cuda", 0), n=320, k=2560, group=128,
        partition_group=128, experts=experts, routing_top_k=top_k,
    )
    chosen = cost.plan_dense_partitions(params, **kwargs)
    assert chosen == min(observed, key=lambda p: (abs(math.log(5 * top_k * p / sms)), p))
    assert len(observed) <= params.marlin_max_candidates
    assert len(set(workloads)) >= 2
    count = len(observed)
    assert cost.plan_dense_partitions(params, **kwargs) == chosen
    assert len(observed) == count
    cost._select_dense_partition_model.cache_clear()


def test_routed_partial_traffic_includes_physical_padding():
    from dataclasses import replace

    dense = cost.NativeCost(
        (0.0, 0.0, 1.0), 0.0, 156, 100_000_000, 8, 64, 320, 2560,
        409600, 2, fixed_partitions=4,
    )
    routed = replace(dense, routed_partials=True)
    extra = routed.features(10, 10, 10)[2] - dense.features(10, 10, 10)[2]
    assert extra == pytest.approx(4 * 4 * (80 - 10) * 320 / 1e8)


@pytest.mark.parametrize(
    "sms,n,experts,top_k,expected",
    [
        (156, 320, 512, 10, {1, 4, 16}),
        (312, 640, 512, 10, {1, 4, 16}),
        (80, 320, 512, 10, {1, 2, 16}),
        (156, 64, 2, 2, {1, 16, 1233}),
    ],
)
def test_routed_first_wave_counts_expert_tiles(monkeypatch, sms, n, experts, top_k, expected):
    observations = []
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=sms),
    )

    def predict(active, rows, tiles):
        observations.append((active, rows, tiles))
        return 1.0

    monkeypatch.setattr(
        cost, "calibrate", lambda *args: SimpleNamespace(error=0.0, predict=predict)
    )
    params = MarlinParameters(marlin_block_m=16, marlin_block_n=64)
    spec = cost.ProbeFormat(4, torch.bfloat16, torch.bfloat16, torch.float16, 0, 1, 1)
    cost._select_dense_partition_model.cache_clear()
    try:
        cost._select_dense_partition_model(
            ((1, "probe"),), params, spec, torch.device("cuda", 0), n, 2560, 128, 1, experts, top_k
        )
    finally:
        cost._select_dense_partition_model.cache_clear()
    assert {rows // top_k for _, rows, _ in observations} == expected
    # The resource anchor really supplies a full wave in the selector's prior.
    covering = max(m for m in expected if m != 16) if len(expected) > 2 else min(expected - {1})
    if covering > 1:

        def supply(m):
            active = min(experts, m * top_k)
            return active * math.ceil(m * top_k / active / 16) * (n // 64)

        assert supply(covering) >= sms
        assert supply(covering - 1) < sms


def test_dense_sm_anchor_is_unchanged(monkeypatch):
    observed = []
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=156),
    )

    def predict(active, rows, tiles):
        observed.append(rows)
        return 1.0

    monkeypatch.setattr(
        cost, "calibrate", lambda *args: SimpleNamespace(error=0.0, predict=predict)
    )
    spec = cost.ProbeFormat(
        4, torch.bfloat16, torch.bfloat16, torch.float16, 0, 1, 1, dense_linear=True
    )
    cost._select_dense_partition_model.cache_clear()
    try:
        cost._select_dense_partition_model(
            ((1, "probe"),),
            MarlinParameters(marlin_block_m=16, marlin_block_n=64),
            spec,
            torch.device("cuda", 0),
            320,
            2560,
            128,
            1,
        )
    finally:
        cost._select_dense_partition_model.cache_clear()
    assert set(observed) == {1, 16, 512}


def test_calibration_receipt_binds_geometry_precision_compilation_and_driver(monkeypatch, tmp_path):
    monkeypatch.setenv('CUBIC_SERVE_CACHE_DIR', str(tmp_path))
    image = SimpleNamespace(source_hash='source', ptx_hash='ptx', cubin_hash='image',
                            compiler_metadata_json='{}', driver_version=1, sm=(8, 0), device='device')
    environment = ['GPU-model-A']
    monkeypatch.setattr('cubic_serve.ops.cuda.cache_identity.gpu_model', lambda device: tuple(environment))
    model = cost.NativeCost((0., 0., 0.), 0., 16, 4096, 16, 64, 128, 256, 8192, 2)
    spec = cost.ProbeFormat(4, torch.bfloat16, torch.bfloat16, torch.float16, 0, 1., 1.)
    args = (image, MarlinParameters(), spec, model, 1, 64, 3)
    path = cost._calibration_cache_path(*args)
    assert path is not None and path.parent.name == 'marlin' and path.is_relative_to(tmp_path / 'operators')
    for field, value in (('cubin_hash', 'different'),):
        changed = SimpleNamespace(**(vars(image) | {field: value}))
        assert cost._calibration_cache_path(changed, *args[1:]) != path
    assert cost._calibration_cache_path(image, args[1], replace(spec, output_dtype=torch.float16),
                                       *args[3:]) != path
    assert cost._calibration_cache_path(image, args[1], spec, replace(model, n=256),
                                       *args[4:]) != path
    assert cost._calibration_cache_path(image, args[1], replace(spec, metadata_packer=object()),
                                       *args[3:]) is None
    environment[0] = 'GPU-model-B'
    assert cost._calibration_cache_path(*args) != path
    environment[0] = 'GPU-model-A'
    monkeypatch.setattr(cost, '_calibration_implementation_identity', lambda: 'changed-launch-glue')
    assert cost._calibration_cache_path(*args) != path


def test_calibration_receipt_requires_integrity_and_original_resource_range(tmp_path):
    import json

    model = cost.NativeCost((0., 0., 0.), 0., 16, 4096, 16, 64, 128, 256, 8192, 2)
    fitted = replace(model, coefficients=(1., 2., 3.), error=.1, l2_bytes=2048)
    path = tmp_path / 'receipt.json'
    cost._write_calibration(path, fitted)
    assert cost._read_calibration(path, model) == fitted
    data = json.loads(path.read_text())
    data['cost']['coefficients'][0] = 9.
    path.write_text(json.dumps(data))
    assert cost._read_calibration(path, model) is None
    cost._write_calibration(path, replace(fitted, l2_bytes=8192))
    assert cost._read_calibration(path, model) is None
    path.write_text('incomplete')
    assert cost._read_calibration(path, model) is None


def test_cached_calibration_returns_original_fit_before_gpu_fixture_or_timing(monkeypatch, tmp_path):
    from contextlib import nullcontext

    device = torch.device("cuda", 0)
    hw = SimpleNamespace(multi_processor_count=16, L2_cache_size=4096)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: hw)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(cost, "compile_module", lambda *args: (object(), "entry"))
    path = tmp_path / "exact-key.json"
    monkeypatch.setattr(cost, "_calibration_cache_path", lambda *args: path)
    model = cost.NativeCost((1., 2., 3.), .125, 16, 2048, 16, 64, 128, 256,
                            128 * 256 // 2 + 128 * 2 * 2, 2, routed_partials=True)
    cost._write_calibration(path, model)

    def forbidden(*args, **kwargs):
        pytest.fail("a valid cost receipt must not allocate or benchmark synthetic fixtures")

    monkeypatch.setattr(torch, "full", forbidden)
    monkeypatch.setattr(cost, "_time_graph", forbidden)
    spec = cost.ProbeFormat(4, torch.bfloat16, torch.bfloat16, torch.float16, 0, 1., 1.)
    # Bypass only the process-local memo: the disk record must be reused with
    # its original coefficients/error, rather than re-fitting or widening it.
    result = cost.calibrate.__wrapped__("source", MarlinParameters(), spec, device,
                                       128, 256, 1, 128, 3)
    assert result == model






@pytest.mark.parametrize('inconclusive', [False, True])
def test_persistent_partition_plan_skips_calibration_and_prediction(monkeypatch, tmp_path, inconclusive):
    monkeypatch.setattr(cost, '_plan_path', lambda *a: tmp_path / 'partition.json')
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda d: SimpleNamespace(multi_processor_count=8))
    monkeypatch.setattr(cost, 'calibrate', lambda source, *a:
                        SimpleNamespace(error=.30 if inconclusive else .01, predict=lambda *a: 1. if source == 'two' else 10.))
    p = MarlinParameters()
    spec = cost.ProbeFormat(4, torch.bfloat16, torch.bfloat16, torch.float16,
                           0, 1., 1., dense_linear=True)
    args = (((1, 'one'), (2, 'two')), p, spec, torch.device('cuda', 0), 128, 512, 128, 3)
    cost._select_dense_partition_model.cache_clear()
    try:
        assert cost._select_dense_partition_model(*args) == (1 if inconclusive else 2)
        cost._select_dense_partition_model.cache_clear()
        def forbidden(*a, **kw):
            pytest.fail('persistent final-plan hit must not measure or predict costs')
        monkeypatch.setattr(cost, 'calibrate', forbidden)
        monkeypatch.setattr(torch.cuda, 'get_device_properties', forbidden)
        assert cost._select_dense_partition_model(*args) == (1 if inconclusive else 2)
    finally:
        cost._select_dense_partition_model.cache_clear()


@pytest.mark.parametrize('inconclusive', [False, True])
def test_persistent_geometry_plan_skips_calibration_and_prediction(monkeypatch, tmp_path, inconclusive):
    from cubic_serve.ops.cuda import launch_cost
    monkeypatch.setenv('CUBIC_SERVE_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(launch_cost, '_launch_calibration_sources', lambda: 'source-v1')
    monkeypatch.setattr('cubic_serve.ops.cuda.cache_identity.gpu_model', lambda d: ('GPU-model-A',))
    monkeypatch.setattr(cost, '_plan_compiler_identity', lambda: ('nvcc', '13.0'))
    monkeypatch.setattr(cost, '_calibration_implementation_identity', lambda: 'math-v1')
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda d: SimpleNamespace(shared_memory_per_block_optin=200000))
    monkeypatch.setattr(cost, 'calibrate', lambda source, p, *a:
                        SimpleNamespace(error=.30 if inconclusive else .01, predict=lambda *a: 1. if p.marlin_block_m == 32 else 10.))
    spec = cost.ProbeFormat(8, torch.bfloat16, torch.bfloat16, torch.float16,
                           0, 1., 1., fixed_partitions=2)
    kw = dict(operators=PackedLinearOperators(), family='awq_moe', scope={'n':256,'k':512},
              source=lambda p: str(p), spec=spec, device=torch.device('cuda', 0),
              n=256, k=512, experts=8, group=128, rows=1., routed=True)
    base = MarlinParameters()
    assert cost.select(base, **kw).marlin_block_m == (16 if inconclusive else 32)
    def forbidden(*a, **kw):
        pytest.fail('persistent final-plan hit must not measure or predict costs')
    monkeypatch.setattr(cost, 'calibrate', forbidden)
    monkeypatch.setattr(cost.NativeCost, 'predict', forbidden)
    assert cost.select(base, **kw).marlin_block_m == (16 if inconclusive else 32)
    path = next(tmp_path.rglob('*.json'))
    path.write_text(path.read_text().replace('"complete": true', '"complete": false'))
    with pytest.raises(pytest.fail.Exception):
        cost.select(base, **kw)


def test_final_plan_key_binds_source_geometry_policy_compiler_and_device(monkeypatch, tmp_path):
    from cubic_serve.ops.cuda import launch_cost
    monkeypatch.setenv('CUBIC_SERVE_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(launch_cost, '_launch_calibration_sources', lambda: 'full-source-v1')
    monkeypatch.setattr('cubic_serve.ops.cuda.cache_identity.gpu_model', lambda d: ('GPU-model-A',))
    monkeypatch.setattr(cost, '_plan_compiler_identity', lambda: ('nvcc', '13.0-v1'))
    monkeypatch.setattr(cost, '_calibration_implementation_identity', lambda: 'math-v1')
    spec = cost.ProbeFormat(4, torch.bfloat16, torch.bfloat16, torch.float16, 0, 1., 1.)
    args = ['geometry', MarlinParameters(), spec, torch.device('cuda', 0), ((1, 'source'),), (128,512,128,4)]
    path = cost._plan_path(*args)
    assert path is not None and cost._plan_path(*args) == path
    for i, value in [(1,replace(args[1],marlin_stages=2)),
                     (2,replace(spec,bits=8)),
                     (4,((1,'changed-source'),)), (5,(128,512,128,8))]:
        changed = list(args);changed[i] = value
        assert cost._plan_path(*changed) != path
    monkeypatch.setattr(cost, '_plan_compiler_identity', lambda: ('nvcc','13.0-v2'))
    assert cost._plan_path(*args) != path
    monkeypatch.setattr(cost, '_plan_compiler_identity', lambda: ('nvcc','13.0-v1'))
    monkeypatch.setattr(launch_cost, '_launch_calibration_sources', lambda: 'unrelated-python-change')
    assert cost._plan_path(*args) == path
    changed = list(args); changed[3] = torch.device('cuda',1)
    assert cost._plan_path(*changed) == path


def test_matching_ratio_has_its_own_complete_held_out_error_and_domain():
    samples = ((1, 1, 12., False), (4, 32, 12., False), (2, 8, 12., True))
    base = cost.NativeCost((10., 0., 0.), .19, 8, 4096, 16, 64, 128, 128, 8192, 2,
                           samples=samples)
    candidate = replace(base, coefficients=(8., 0., 0.), error=.25,
                        samples=tuple((a, r, t * .8, held) for a, r, t, held in samples))
    interval, error = candidate.relative_interval(base, 2, 16, 2)
    assert interval == pytest.approx((.8, .8)) and error == pytest.approx(0.)
    assert candidate.relative_interval(base, 5, 16, 2) is None
    assert candidate.relative_interval(base, 2, 96, 6) is None
    incomplete = replace(candidate, samples=candidate.samples[:-1])
    assert incomplete.relative_interval(base, 2, 16, 2) is None
    assert replace(candidate, block_m=32).relative_interval(base, 2, 16, 2) is None
    # The last independent point can invalidate an apparent shared-error gain.
    changed = replace(candidate, samples=(*candidate.samples[:-1], (2, 8, 16., True)))
    interval, error = changed.relative_interval(base, 2, 16, 2)
    assert interval[1] >= 1 and error > .2
    assert replace(candidate, samples=((1, 1, 9.6, False),)).relative_interval(
        replace(base, samples=((1, 1, 12., False),)), 1, 1, 1,
    ) is None


def test_selector_can_use_qualified_relative_evidence_with_absolute_overlap(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(shared_memory_per_block_optin=200000))
    base = MarlinParameters()
    samples = ((1, 1, 12., False), (4, 32, 12., False), (2, 8, 12., True))
    bad_holdout = [False]

    def calibrated(source, p, *args):
        faster = p.marlin_block_m == 16 and p.marlin_stages == 2 and p.marlin_warps == 4
        values = tuple((a, r, t * (.8 if faster else 1), held)
                       for a, r, t, held in samples)
        if faster and bad_holdout[0]:
            values = (*values[:-1], (2, 8, 16., True))
        return cost.NativeCost((8. if faster else 10., 0., 0.), .25 if faster else .19,
                               8, 4096, p.marlin_block_m, p.marlin_block_n,
                               256, 128, 8192, 2, samples=values)

    monkeypatch.setattr(cost, "calibrate", calibrated)
    kw = dict(operators=PackedLinearOperators(), family="int_moe",
              scope=dict(n=256, k=128, experts=4, top_k=2), source=lambda p: "",
              spec=cost.ProbeFormat(4, torch.bfloat16, torch.bfloat16, torch.float16, 0, 1, 1),
              device=torch.device("cuda", 0), n=256, k=128, experts=4, group=32,
              rows=4., routed=True)
    assert cost.select(base, **kw).marlin_stages == 2
    bad_holdout[0] = True
    assert cost.select(base, **kw) == base


def test_calibration_receipt_binds_all_ratio_measurements(tmp_path):
    import json
    model = cost.NativeCost((1., 0., 0.), .1, 8, 4096, 16, 64, 128, 128, 8192, 2,
                           samples=((1, 1, 1., False), (2, 8, 1.1, True)))
    path = tmp_path / "receipt.json"
    cost._write_calibration(path, model)
    assert cost._read_calibration(path, model) == model
    data = json.loads(path.read_text())
    data["cost"]["samples"][-1][2] = .01
    path.write_text(json.dumps(data))
    assert cost._read_calibration(path, model) is None


@pytest.mark.parametrize('changes,resident,local_bytes,expected', [
    ({}, 2, 0, 64),
    ({}, 1, 0, 16),
    ({}, 0, 0, 16),
    ({}, 2, 16, 16),
    ({'m': 1}, 2, 0, 16),
    ({'m': 96}, 2, 0, 16),
    ({'n': 256}, 2, 0, 16),
    ({'sms': 24}, 2, 0, 64),
])
def test_memory_reuse_default_observes_resource_and_padding_boundaries(
    changes, resident, local_bytes, expected,
):
    base = MarlinParameters()
    wide = replace(base, marlin_block_m=64, marlin_stages=2)
    points = [(base, cost.KernelResources(2, 98, 0)),
              (wide, cost.KernelResources(resident, 244, local_bytes))]
    geometry = dict(m=128, n=62080, k=2560, weight_bytes=161408000,
                    sms=156)
    chosen = cost.memory_reuse_default(base, points, **(geometry | changes))
    assert chosen.marlin_block_m == expected


def test_memory_reuse_prices_cached_weight_transport():
    base = MarlinParameters()
    wide = replace(base, marlin_block_m=64, marlin_stages=2)
    points = [(base, cost.KernelResources(2, 99, 0)),
              (wide, cost.KernelResources(2, 244, 0))]
    # The 5 MiB carrier fits in L2, but each row tile still reads it. Preserve
    # complete-wave supply at the smaller boundary and reuse at larger M.
    geometry = dict(n=10240, k=512, weight_bytes=5406720, sms=156)
    assert cost.memory_reuse_default(base, points, m=64, **geometry) == base
    for m in (128, 256, 512):
        assert cost.memory_reuse_default(base, points, m=m, **geometry) == wide


@pytest.mark.parametrize('origin,basis', [
    ('production_default', 'retained_default'),
    ('resource_default', 'retained_default'),
    (None, 'retained_default'),
    (None, 'timing_policy'),
])
def test_only_unmeasured_production_default_is_reconsidered(monkeypatch, origin, basis):
    monkeypatch.setattr(cost, '_read_startup_calibration', lambda *a:
                        dict(chosen=1, complete=True, origin=origin, basis=basis))
    assert cost._read_plan(None, 2) == 1
    assert cost._read_plan(None, 2, True) == (None if origin == 'production_default' else 1)


def test_resource_default_cache_restores_without_queries_or_timing(monkeypatch, tmp_path):
    from cubic_serve.ops.cuda import launch_cost
    from cubic_serve.ops.cuda.startup import startup_weight_identity
    monkeypatch.setenv('CUBIC_SERVE_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(launch_cost, '_launch_calibration_sources', lambda: 'source-v1')
    monkeypatch.setattr('cubic_serve.ops.cuda.cache_identity.gpu_model', lambda d: ('device',))
    monkeypatch.setattr(cost, '_plan_compiler_identity', lambda: ('nvcc', '13.0'))
    monkeypatch.setattr(cost, '_calibration_implementation_identity', lambda: 'math-v1')
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda d: SimpleNamespace(shared_memory_per_block_optin=200000))
    base = MarlinParameters()
    wide = replace(base, marlin_block_m=64, marlin_stages=2)
    spec = cost.ProbeFormat(8, torch.int8, torch.bfloat16, torch.float16,
                           1, 1., 1., dense_linear=True, single_subtile=True)
    kw = dict(operators=PackedLinearOperators(), family='int_linear', scope={'n':256,'k':512},
              source=lambda p: str(p), spec=spec, device=torch.device('cuda', 0),
              n=256, k=512, experts=1, group=128, rows=128, routed=False,
              alternates=(wide,))
    monkeypatch.setattr(cost, '_native_memory_reuse_default', lambda *a, **kw: wide)
    with startup_weight_identity(('weights',), allow_calibration=False):
        assert cost.select(base, **kw) == wide
        def forbidden(*a, **kw):
            pytest.fail('a complete resource plan must restore without native queries or timing')
        monkeypatch.setattr(cost, '_native_memory_reuse_default', forbidden)
        monkeypatch.setattr(cost, 'calibrate', forbidden)
        assert cost.select(base, **kw) == wide


@pytest.fixture
def capacity_partition_plan(monkeypatch, tmp_path):
    from contextlib import nullcontext

    entries = {}
    monkeypatch.setattr(cost, "_plan_path", lambda *a, **kw: tmp_path / "capacity.json")
    monkeypatch.setattr(cost, "_read_plan", lambda path, count: entries.get(path))
    monkeypatch.setattr(cost, "_write_plan", lambda path, chosen, **kw: entries.__setitem__(path, chosen))
    monkeypatch.setattr(cost, "local_plan_lock", lambda _: nullcontext())
    params = MarlinParameters(marlin_block_m=16, marlin_block_n=64, marlin_block_k=128)

    def plan(*, sms=156, **changes):
        monkeypatch.setattr(torch.cuda, "get_device_properties",
                            lambda _: SimpleNamespace(multi_processor_count=sms))
        kwargs = dict(partitions=40, rows=256, n=320, k=10240, group=128,
                      spec=None, source=lambda p: f"native partition={p}", device="cuda:0")
        return cost.capacity_k_partitions(params, **(kwargs | changes))

    return plan, entries


@pytest.mark.parametrize("rows,sms,expected", [
    (1, 156, 40), (16, 156, 40), (128, 156, 4), (256, 156, 2),
    (128, 20, 1), (128, 72, 2), (128, 204, 8),
])
def test_capacity_partitions_remove_scratch_with_legal_device_supply(
    capacity_partition_plan, rows, sms, expected,
):
    plan, _ = capacity_partition_plan
    actual = plan(rows=rows, sms=sms)
    assert actual == expected
    assert 10240 % (actual * 128) == 0
    assert actual <= 40


def test_capacity_partition_plan_does_not_expand_an_existing_limit(capacity_partition_plan):
    plan, _ = capacity_partition_plan
    assert plan(rows=128, sms=512, partitions=2) == 2


def test_warm_capacity_plan_restores_without_resource_pricing(capacity_partition_plan, monkeypatch):
    plan, _ = capacity_partition_plan
    assert plan() == 2

    def forbidden(*a, **kw):
        pytest.fail("a warm capacity plan repeated resource pricing")

    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    params = MarlinParameters(marlin_block_m=16, marlin_block_n=64, marlin_block_k=128)
    assert cost.capacity_k_partitions(
        params, partitions=40, rows=256, n=320, k=10240, group=128,
        spec=None, source=lambda p: f"native partition={p}", device="cuda:0",
    ) == 2


def test_invalid_cached_capacity_plan_is_rebuilt(capacity_partition_plan):
    plan, entries = capacity_partition_plan
    assert plan() == 2
    entries[next(iter(entries))] = 2  # Three slices cannot divide 80 K groups.
    assert plan() == 2


def test_int_a8_cross_m_anchor_restores_actual_native_reduction_without_repricing():
    from cubic_serve.quantization.common.small_rows_cost import (
        row_arithmetic_plan, row_arithmetic_scope,
    )
    from cubic_serve.quantization.extensions.int.marlin import _a8_dense_arithmetic
    from cubic_serve.quantization.extensions.int.policy import IntegerLinearParameters

    params = IntegerLinearParameters()
    selected = replace(params, marlin_block_m=64, marlin_stages=2)
    matrix = SimpleNamespace(encoding=("sym", 8, 128), shape=(320, 10240), partitions=())
    owner, anchor = SimpleNamespace(), SimpleNamespace()
    x, y = torch.empty(256, 10240), torch.empty(256, 320)
    assert _a8_dense_arithmetic(owner, x, matrix, y, params, lambda: (selected, 2)) == (selected, 2)
    plan = row_arithmetic_plan(owner, 256)

    def forbidden():
        pytest.fail("an independent anchor repriced the candidate as a C1 workload")

    with row_arithmetic_scope(plan):
        assert _a8_dense_arithmetic(
            anchor, x[:1], matrix, y[:1], params, forbidden,
        ) == (selected, 2)
    assert row_arithmetic_plan(anchor, 1) == plan
    with row_arithmetic_scope({}), pytest.raises(ValueError, match="absent"):
        _a8_dense_arithmetic(anchor, x[:1], matrix, y[:1], params, forbidden)
    with row_arithmetic_scope(plan), pytest.raises(ValueError, match="absent"):
        _a8_dense_arithmetic(anchor, x[:1], matrix, y[:1], params, forbidden, grouped=True)
    corrupt = {key: ("unknown", recipe[1], recipe[2]) for key, recipe in plan.items()}
    with row_arithmetic_scope(corrupt), pytest.raises(ValueError, match="unknown reduction"):
        _a8_dense_arithmetic(anchor, x[:1], matrix, y[:1], params, forbidden)


def test_dense_capacity_scope_is_owner_local_nested_and_exception_safe():
    first, second = PackedLinearOperators(), PackedLinearOperators()
    assert not first.dense_capacity_partitions
    with pytest.raises(RuntimeError, match="binding failed"):
        with first.dense_capacity_scope():
            assert first.dense_capacity_partitions
            assert not second.dense_capacity_partitions
            with first.dense_capacity_scope():
                assert first.dense_capacity_partitions
            assert first.dense_capacity_partitions
            raise RuntimeError("binding failed")
    assert not first.dense_capacity_partitions


def test_int_a16_anchor_keeps_complete_group_geometry_and_positive_partitions():
    from cubic_serve.quantization.common.small_rows_cost import (
        row_arithmetic_plan, row_arithmetic_scope,
    )
    from cubic_serve.quantization.extensions.int.marlin import _a16_dense_arithmetic
    from cubic_serve.quantization.extensions.int.policy import IntegerLinearParameters

    params = IntegerLinearParameters()
    matrix = SimpleNamespace(encoding=("sym", 4, 64), shape=(320, 10240), partitions=())
    tail = SimpleNamespace(encoding=matrix.encoding, shape=(64, 10240), partitions=())
    groups = ((matrix,), (tail,))
    owner, anchor = SimpleNamespace(), SimpleNamespace()
    x, y = torch.empty(256, 10240), torch.empty(256, 384)
    assert _a16_dense_arithmetic(owner, x, groups, y, params,
                                lambda: (params, 2)) == (params, 2)
    plan = row_arithmetic_plan(owner, 256)

    def forbidden():
        pytest.fail("A16 independent anchor repriced the candidate")

    with row_arithmetic_scope(plan):
        assert _a16_dense_arithmetic(anchor, x[:1], groups, y[:1], params,
                                     forbidden) == (params, 2)
        changed_tail = SimpleNamespace(encoding=matrix.encoding, shape=(128, 10240), partitions=())
        with pytest.raises(ValueError, match="absent"):
            _a16_dense_arithmetic(anchor, x[:1], ((matrix,), (changed_tail,)),
                                  y[:1], params, forbidden)
    assert row_arithmetic_plan(anchor, 1) == plan
    corrupt = {key: (recipe[0], recipe[1], 0) for key, recipe in plan.items()}
    with row_arithmetic_scope(corrupt), pytest.raises(ValueError, match="malformed"):
        _a16_dense_arithmetic(anchor, x[:1], groups, y[:1], params, forbidden)


@pytest.mark.parametrize("mode,scoped,expected", [
    ("wna16", False, 20), ("wna16", True, 2), ("wna8", True, 20),
])
def test_int_dense_binding_limits_capacity_change_to_declared_wna16_blocks(
    monkeypatch, mode, scoped, expected,
):
    from collections import namedtuple
    from contextlib import nullcontext
    from cubic_serve.quantization.extensions.int import marlin
    from cubic_serve.quantization.extensions.int.policy import IntegerLinearParameters

    owner = PackedLinearOperators(OperatorConfig(compute_mode=mode))
    encoding = namedtuple("Encoding", "bits group_in")(4, 64)
    matrix = SimpleNamespace(encoding=encoding,
                             shape=(320, 10240), partitions=())
    x, y = torch.empty(256, 10240, dtype=torch.bfloat16), torch.empty(256, 320)
    monkeypatch.setattr(marlin, "fields", lambda enc: ("legacy", "sym", "F16"))
    monkeypatch.setattr(marlin, "plan_dense_partitions", lambda *a, **kw: 20)
    monkeypatch.setattr(marlin, "select", lambda p, **kw: p)
    seen = []

    def cap(p, **kwargs):
        seen.append(kwargs["rows"])
        return 2

    class StopBeforeDeviceAllocation(Exception):
        pass

    def done(p, **kwargs):
        raise StopBeforeDeviceAllocation

    monkeypatch.setattr(marlin, "capacity_k_partitions", cap)
    monkeypatch.setattr(marlin, "shared_bytes", done)
    with owner.dense_capacity_scope() if scoped else nullcontext():
        with pytest.raises(StopBeforeDeviceAllocation):
            marlin._bind(owner, x, ((matrix,),), y, IntegerLinearParameters())
    assert seen == ([256] if expected == 2 else [])
    if expected == 2:
        assert next(iter(owner._packed_row_plans.values()))[-1] == expected
