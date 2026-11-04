"""Resource exclusion must be priced without inventing a mixed kernel footprint."""

import os
from unittest.mock import patch

import pytest
import torch

from cubic_serve.ops.cuda.launch_cost import (
    DeviceResources,
    KernelResources,
    MatmulGeometry,
    RouteCompletionCost,
    TransferCost,
    WorkCost,
    device_resources,
    fit_nonnegative_cost,
    graph_kernel_resources,
    matmul_geometries,
    overlapping_cost,
    predicted_gain,
    prediction_interval,
    scratch_ranges_disjoint,
)
from cubic_serve.ops.strategies import LaunchParameters, LinearParameters, linear_overlap_tiles


def test_scratch_range_scan_matches_the_complete_original_predicate():
    import random

    class Span:
        def __init__(self, start, end):
            self.start, self.end = start, end

        def data_ptr(self):
            return self.start

        def numel(self):
            return self.end - self.start

        def element_size(self):
            return 1

    def original(a, b):
        return all(x.start >= y.end or y.start >= x.end for x in a for y in b)

    spans = [Span(a, b) for a in range(6) for b in range(a, 6)]
    for a in spans:
        for b in spans:
            assert scratch_ranges_disjoint((a,), (b,)) == original((a,), (b,))
    rng = random.Random(9821)
    for _ in range(256):
        a = rng.choices(spans, k=rng.randrange(13))
        b = rng.choices(spans, k=rng.randrange(13))
        assert scratch_ranges_disjoint(a, b) == original(a, b)
    # Nested/duplicate intervals within a lane are legal; boundary-touching
    # and empty views retain the original conservative endpoint rule.
    a = [Span(0, 9), Span(1, 3), Span(0, 9), Span(12, 12)]
    assert scratch_ranges_disjoint(a, [Span(9, 12), Span(20, 21)])
    assert not scratch_ranges_disjoint(a, [Span(4, 4)])


def test_scratch_range_scan_reads_each_tensor_extent_once():
    class Span:
        def __init__(self, start):
            self.start, self.calls = start, 0

        def data_ptr(self):
            self.calls += 1
            return self.start

        def numel(self):
            self.calls += 1
            return 8

        def element_size(self):
            self.calls += 1
            return 1

    a = [Span(16 * i) for i in range(128)]
    b = [Span(4096 + 16 * i) for i in range(1024)]
    assert scratch_ranges_disjoint(a, b)
    assert all(t.calls == 3 for t in (*a, *b))


@pytest.fixture(autouse=True)
def explicit_launch_qualification():
    """Cost-search fixtures opt in; serving defaults remain search-free."""
    from cubic_serve.ops.cuda.startup import startup_weight_identity
    with startup_weight_identity(None, allow_calibration=True):
        yield


def test_calibration_comparison_visits_all_strided_elements_with_bounded_leaves():
    from cubic_serve.ops.cuda.launch_cost import _equal_calibration_output

    equal = torch.equal
    actual = torch.arange(2 * 7 * 17).reshape(2, 7, 17).transpose(0, 2)
    expected = actual.clone()
    calls = []

    def bounded(a, b):
        calls.append(a.numel())
        assert a.numel() <= 5
        return equal(a, b)

    with patch("torch.equal", bounded):
        assert _equal_calibration_output(actual, expected, max_elements=5)
        expected[-1, -1, -1] = -1
        assert not _equal_calibration_output(actual, expected, max_elements=5)
    assert sum(calls) >= actual.numel()
    for a, b in ((torch.empty(0), torch.empty(0)),
                 (torch.tensor(1.), torch.tensor(1)),
                 (torch.tensor([-0.]), torch.tensor([0.])),
                 (torch.tensor([float("nan")]), torch.tensor([float("nan")])),
                 (torch.ones(2, 3), torch.ones(3, 2))):
        assert _equal_calibration_output(a, b, max_elements=1) == equal(a, b)


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1",
                    reason="explicit CUDA audit required")
def test_large_cuda_calibration_comparison_has_bounded_temporary_and_tail_check():
    from cubic_serve.ops.cuda.launch_cost import _equal_calibration_output

    value = torch.ones((32, 1024, 1024), dtype=torch.bfloat16, device="cuda")
    expected = value.clone()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    assert _equal_calibration_output(value, expected)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - allocated <= 16 << 20
    expected[-1, -1, -1] = 0
    assert not _equal_calibration_output(value, expected)


def test_shared_fit_recovers_separate_launch_and_work_costs() -> None:
    features = [(1, work, traffic) for work, traffic in
                ((0, 0), (1, 0), (0, 1), (3, 2), (2, 7), (9, 4))]
    times = [2 + 3 * work + 5 * traffic for _, work, traffic in features]
    assert fit_nonnegative_cost(features, times) == pytest.approx((2, 3, 5))
    for features, times in (([], []), ([(1, -1)], [1]),
                            ([(1, 2)], [0]), ([(1, float("nan"))], [1])):
        with pytest.raises(ValueError):
            fit_nonnegative_cost(features, times)


def test_prediction_bounds_invert_observed_relative_error_and_preserve_units() -> None:
    # An observation of 10 and prediction of 8 has 20% relative-to-observed
    # error. Multiplying 8 by 1.2 produces 9.6 and excludes that observation.
    assert prediction_interval(8, .2) == pytest.approx((8 / 1.2, 10))
    assert prediction_interval(8000, .2) == pytest.approx(
        tuple(1000 * v for v in prediction_interval(8, .2)))
    assert prediction_interval(8, 1)[1] == float("inf")
    assert not predicted_gain(prediction_interval(8, .2), prediction_interval(10, 0))
    assert predicted_gain(prediction_interval(7, .1), prediction_interval(10, .1), 5)
    for value, error in ((-1, 0), (1, -1), (float("nan"), 0), (1, float("inf"))):
        with pytest.raises(ValueError):
            prediction_interval(value, error)


def test_shared_fit_is_repeatable_with_dependent_storage_cost_axes() -> None:
    # Three row populations can leave a storage-domain basis rank deficient.
    # Valid equivalent fits still must not make startup predictions fluctuate.
    features = [
        (1., work, work * rows, work * (rows == 2), work * rows * (rows == 4))
        for work in (1., 2.) for rows in (1, 2, 4)
    ]
    times = [40.396, 47.052, 57.804, 69.068, 80.972, 100.808]
    coefficients = fit_nonnegative_cost(features, times)
    assert all(value >= 0 for value in coefficients)
    for _ in range(12):
        assert fit_nonnegative_cost(features, times) == pytest.approx(
            coefficients, rel=1e-11, abs=1e-11)


def test_neighboring_rows_identify_storage_domain_costs() -> None:
    features = [
        (1., work, work * rows, work * (rows in (2, 3)),
         work * rows * (rows == 4))
        for work in (1., 2.) for rows in (1, 2, 3, 4)
        if rows != 3 or work != 2.
    ]
    expected = (10., 2., .5, .3, .2)
    times = [sum(a * b for a, b in zip(expected, point)) for point in features]
    coefficients = fit_nonnegative_cost(features, times)
    assert coefficients == pytest.approx(expected)
    # The held-out row population at a different geometry has its own value.
    point = (1., 1.5, 4.5, 1.5, 0.)
    assert sum(a * b for a, b in zip(coefficients, point)) == pytest.approx(15.7)


def test_complete_transport_uncertainty_prevents_unproved_center_value_switch() -> None:
    from dataclasses import replace

    from cubic_serve.ops.cuda.launch_cost import OrderedCollectiveCost

    def observation(payload):
        base = 10 + .01 * payload
        return payload, (base, .9 * base, .8 * base, .7 * base)

    cost = OrderedCollectiveCost.fit(
        [observation(p) for p in (10, 100, 400, 1000)],
        [observation(p) for p in (50, 250, 600)],
    )
    cost = replace(cost, interval_error_fractions=((.15,) * 3,) * 4)
    args = dict(min_gain_percent=0, max_error_percent=20)
    # Every nominal step is faster and passes the error ceiling, but none
    # beats its actual baseline once both independently measured errors enter.
    assert not cost.select(50, fallback=False, **args)
    assert cost.select(50, fallback=True, **args)
    assert not cost.select_host(50, partitioned=True, **args)
    assert not cost.select_host_partition(50, partitioned=True, host_selected=True, **args)
    # A sufficiently separated end-to-end candidate remains eligible.
    assert cost.select_host_partition(50, partitioned=False, host_selected=False, **args)


def test_paged_cost_respects_measured_domain_and_merge_price() -> None:
    from cubic_serve.ops.cuda.attention.paged_decode import PagedDecodeCost

    # Synthetic independent costs: very expensive publication/merge must beat
    # a coverage-only rule even though all partitions fit in one SM wave.
    cost = PagedDecodeCost((1, 1, 1, 1, 100, 1, 1), .01, 128, 1, 8, 256)
    assert cost.select(1, 256, fallback=16, min_gain_percent=5,
                       max_error_percent=10) == 1
    assert cost.select(9, 256, fallback=16, min_gain_percent=5,
                       max_error_percent=10) == 16
    assert cost.select(1, 257, fallback=16, min_gain_percent=5,
                       max_error_percent=10) == 16
    assert cost.select(1, 256, fallback=16, min_gain_percent=5,
                       max_error_percent=0) == 16
    with pytest.raises(ValueError, match="measured domain"):
        cost.predict(1, 257, 16)
    for rows, tokens, splits in ((0, 256, 16), (1, 0, 16), (1, 256, 3),
                                 (True, 256, 16), (1, 256, 16.0)):
        with pytest.raises(ValueError):
            cost.select(rows, tokens, fallback=splits, min_gain_percent=5,
                        max_error_percent=10)


def test_work_cost_prices_compute_bandwidth_and_supplied_work() -> None:
    cost = WorkCost(2, 100, 10)
    assert cost.predict(1000, 1) == 12
    assert cost.predict(1, 1000) == 102
    assert cost.predict(1000, 1, supply=0.5, launches=3) == 26
    for invalid in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            cost.predict(invalid, 1)
        with pytest.raises(ValueError):
            WorkCost(1, invalid, 1)
    for launches in (True, 1.5, -1):
        with pytest.raises(ValueError):
            cost.predict(1, 1, launches=launches)


@pytest.mark.parametrize("replays", [0, -1, True, 1.5])
def test_linear_calibration_rejects_invalid_repetition_count(replays) -> None:
    from cubic_serve.ops.cuda.launch_cost import LinearLaunchCost

    cost = LinearLaunchCost(DeviceResources(1, 2048, 65536, 65536), 0.0, ())
    with pytest.raises(ValueError, match="positive integer"):
        cost.select([], torch.empty((1, 1)), (), grouped=True, replays=replays)


def test_branch_costs_account_for_shared_device_contention_and_join() -> None:
    cost = WorkCost(1, 100, 10)
    compute, memory = (1000, 0, 1), (0, 100, 1)
    serial, parallel = cost.branch_costs(compute, memory, fork_join_launches=1)
    assert (serial, parallel) == (22, 12)
    # Two compute-bound branches share one compute budget; their work cannot
    # both run at the isolated peak. Join overhead can make overlap lose.
    serial, parallel = cost.branch_costs(compute, compute, fork_join_launches=3)
    assert (serial, parallel) == (22, 23)
    with pytest.raises(ValueError):
        cost.branch_costs(compute, memory, fork_join_launches=float("nan"))


def test_transfer_fit_rejects_unusable_measurements() -> None:
    fit = TransferCost.fit(((100, 12), (300, 16)))
    assert fit.predict(500) == pytest.approx(20)
    for points in (((100, 12), (100, 16)), ((100, 12), (300, 11)),
                   ((100, 12), (float("nan"), 16))):
        with pytest.raises(ValueError):
            TransferCost.fit(points)


def test_ordered_transport_envelope_and_holdout_guard() -> None:
    from cubic_serve.ops.cuda.launch_cost import OrderedCollectiveCost

    def observations(payload):
        return payload, (min(10 + .2 * payload, 30 + .05 * payload),
                         min(20 + .1 * payload, 35 + .02 * payload))

    cost = OrderedCollectiveCost.fit(
        [observations(p) for p in (10, 100, 400, 1000)],
        [observations(p) for p in (50, 600)],
    )
    assert cost.error_fraction < 1e-12
    assert not cost.select(50, fallback=True, min_gain_percent=5, max_error_percent=20)
    assert cost.select(600, fallback=False, min_gain_percent=5, max_error_percent=20)
    # Unmeasured sizes do not extrapolate an assumed NCCL protocol regime.
    assert not cost.select(2000, fallback=False, min_gain_percent=5, max_error_percent=20)
    failed = OrderedCollectiveCost.fit(
        [observations(p) for p in (10, 100, 400, 1000)], [(600, (200, 200))],
    )
    assert not failed.select(600, fallback=False, min_gain_percent=5, max_error_percent=20)
    with pytest.raises(ValueError):
        OrderedCollectiveCost.fit([observations(p) for p in (10, 100, 400, 1000)],
                                  [(600, (float("nan"), 1))])


def test_transport_bandwidth_deterioration_does_not_extrapolate_low_cost() -> None:
    from cubic_serve.ops.cuda.launch_cost import OrderedCollectiveCost

    def observation(payload, host):
        return payload, (30 + .1 * payload, 40 + .08 * payload, host)

    anchors = [observation(p, t) for p, t in
               ((10, 2.1), (100, 3), (400, 26), (1000, 50))]
    holdouts = [observation(p, t) for p, t in
                ((50, 2.5), (250, 14.5), (600, 34))]
    cost = OrderedCollectiveCost.fit(anchors, holdouts)
    assert cost.host is not None
    # The small-payload line predicts 6 at a measured large-payload cost of
    # 26. Taking the global minimum rejects an otherwise accurate local fit.
    assert min(f.predict(400) for f in cost.host) == pytest.approx(6)
    assert cost.host_error_fraction < 1e-12
    args = dict(partitioned=False, min_gain_percent=5, max_error_percent=20)
    assert cost.select_host(50, **args)
    assert cost.select_host(600, **args)
    assert not cost.select_host(2000, **args)
    # An invalid transition declines only that interval; it must not disable
    # independently qualified small and large payloads.
    bad = OrderedCollectiveCost.fit(anchors, [holdouts[0], observation(250, 4), holdouts[2]])
    assert bad.select_host(50, **args)
    assert not bad.select_host(250, **args)
    assert bad.select_host(600, **args)
    missing = OrderedCollectiveCost.fit(anchors, [holdouts[0], holdouts[2]])
    assert not missing.select_host(250, **args)


@pytest.mark.parametrize("ranks", [2, 3, 4, 6, 8])
def test_route_completion_prices_padding_and_reduced_gather(ranks: int) -> None:
    # Synthetic rates express latency/bandwidth competition, not a tuning table.
    cost = RouteCompletionCost(ranks, TransferCost(10, 0.001), TransferCost(20, 0.001),
                               TransferCost(20, 0.001), TransferCost(12, 0.0001), 0.02)
    assert not cost.beneficial(1, 8, 64, 2, partitioned=False,
                               min_gain_percent=5, max_error_percent=20)
    assert cost.beneficial(ranks * 64, 8, 64, 2, partitioned=True,
                           min_gain_percent=5, max_error_percent=20)
    # With top-k=1 the folded gather removes no route dimension.
    old, new = cost.predict(ranks * 64, 1, 64, 2, partitioned=True)
    assert new == old
    full = cost.predict(ranks * 64, 8, 64, 2, partitioned=True)[1]
    padded = cost.predict(ranks * 64 - 1, 8, 64, 2, partitioned=True)[1]
    assert padded > full


def test_route_completion_rejects_failed_holdout_without_batch_patches() -> None:
    args = (4, TransferCost(10, 0.001), TransferCost(20, 0.001),
            TransferCost(20, 0.001), TransferCost(12, 0.0001))
    cost = RouteCompletionCost(*args, error_fraction=0.4)
    assert not cost.beneficial(256, 8, 64, 2, partitioned=True,
                               min_gain_percent=5, max_error_percent=20)
    singleton = RouteCompletionCost(1, *args[1:], error_fraction=0.0)
    assert singleton.predict(256, 8, 64, 2, partitioned=False) == (0, 0)
    assert not singleton.beneficial(256, 8, 64, 2, partitioned=False,
                                    min_gain_percent=5, max_error_percent=20)
    copies_unfitted = RouteCompletionCost(*args, error_fraction=0.01,
                                          padding_error_fraction=0.4)
    assert copies_unfitted.beneficial(256, 8, 64, 2, partitioned=True,
                                      min_gain_percent=5, max_error_percent=20)
    assert not copies_unfitted.beneficial(255, 8, 64, 2, partitioned=True,
                                          min_gain_percent=5, max_error_percent=20)


@pytest.mark.parametrize("device", [
    DeviceResources(14, 1536, 65536, 65536),
    DeviceResources(108, 2048, 65536, 167936),
    DeviceResources(132, 1536, 65536, 102400),
    DeviceResources(148, 2048, 131072, 233472),
])
@pytest.mark.parametrize("shape", [(1, (97,), 65), (33, (1537, 4096), 2048),
                                  (256, (1024,), 6144), (512, (768,), 7168)])
def test_geometry_search_is_resource_bounded_and_preserves_reduction(device, shape) -> None:
    m, ns, k = shape
    base = MatmulGeometry(16, 64, 64, 4, 3)
    points = matmul_geometries(device, rows=m, columns=ns, reduction=k,
                              element_bytes=2, baseline=base)
    assert points[0] == base
    assert len(set(points)) == len(points) <= LaunchParameters().max_candidates
    for p in points[1:]:
        assert p.block_k == base.block_k
        assert (p.block_m + p.block_n) * p.block_k * 2 * p.stages <= device.shared
        assert 32 * p.warps <= device.threads


def test_geometry_respects_explicit_overrides_and_search_budget() -> None:
    device = DeviceResources(108, 2048, 65536, 167936)
    base = MatmulGeometry(32, 128, 128, 8, 2)
    kwargs = dict(rows=128, columns=(4096,), reduction=6144, element_bytes=2, baseline=base)
    assert matmul_geometries(device, **kwargs, limits=LaunchParameters(max_candidates=1)) == (
        base,
    )
    assert matmul_geometries(device, **kwargs,
                            locked=frozenset(("block_m", "block_n", "warps", "stages"))) == (
        base,
    )


def test_pipeline_probe_preserves_placement_reduction_and_explicit_stage_lock() -> None:
    device = DeviceResources(108, 2048, 65536, 167936)
    base = MatmulGeometry(16, 32, 32, 4, 3)
    kwargs = dict(rows=4, columns=(256,), reduction=3072, element_bytes=2,
                  groups=256, baseline=base)
    points = matmul_geometries(device, **kwargs)
    assert MatmulGeometry(16, 32, 32, 4, 1) in points
    assert any(p.block_n != base.block_n for p in points)
    locked = matmul_geometries(device, **kwargs, locked=frozenset(("stages",)))
    assert all(p.stages == base.stages for p in locked)
    assert len(points) <= LaunchParameters().max_candidates


def test_geometry_uses_device_resources_and_local_group_population() -> None:
    base = MatmulGeometry(16, 64, 128, 4, 3)
    kwargs = dict(rows=4, columns=(1024,), reduction=6144, element_bytes=2, baseline=base)
    device = DeviceResources(132, 1536, 65536, 102400)
    sparse = matmul_geometries(device, **kwargs)
    routed = matmul_geometries(device, **kwargs, groups=64)
    assert sparse != routed


def test_prefilter_preserves_a_collective_coexistence_class_within_budget() -> None:
    # A full grid is not necessarily desirable when its shared-memory footprint
    # excludes the collective. The final scorer must receive an alternative.
    device = DeviceResources(100, 2048, 65536, 100000)
    base = MatmulGeometry(32, 64, 64, 4, 3)
    communication = (KernelResources(4, 128, 1024, 90000),)
    kwargs = dict(rows=128, columns=(4096,), reduction=2048,
                  element_bytes=2, baseline=base, limits=LaunchParameters(max_candidates=2))
    independent = matmul_geometries(device, **kwargs)
    joint = matmul_geometries(device, **kwargs, communication=communication)
    assert len(joint) == 2 and joint[0] == base
    assert independent != joint
    point = joint[1]
    blocks = ((128 + point.block_m - 1) // point.block_m
              * ((4096 + point.block_n - 1) // point.block_n))
    assert device.sms - blocks >= communication[0].blocks
    assert point.block_k == base.block_k


def test_resource_exclusion_changes_joint_cost_without_changing_compute() -> None:
    device = DeviceResources(sms=100, threads=2048, registers=65536, shared=100000)
    communication = (KernelResources(4, 512, 48000, 90000),)
    occupied = KernelResources(200, 128, 8192, 20000)
    available = KernelResources(90, 128, 8192, 20000)
    assert overlapping_cost(2, occupied, 3, communication, device) == 5
    assert overlapping_cost(2, available, 3, communication, device) == 3
    larger_shared = DeviceResources(100, 2048, 65536, 200000)
    assert overlapping_cost(2, occupied, 3, communication, larger_shared) == 3
    assert overlapping_cost(2, occupied, 0, (), device) == 2


def test_each_collective_node_keeps_its_own_resource_provenance() -> None:
    device = DeviceResources(100, 2048, 65536, 100000)
    compute = KernelResources(90, 128, 8192, 20000)
    # The large grid fits alongside compute; the exclusive kernel needs only
    # four of the ten free SMs. Combining max(grid) with max(shared) is wrong.
    communication = (KernelResources(1000, 128, 1024, 0),
                     KernelResources(4, 512, 48000, 90000))
    assert overlapping_cost(2, compute, 3, communication, device) == 3


def test_tile_bounds_do_not_change_reduction_or_override_larger_fixed_tiles() -> None:
    parameters = LinearParameters(fixed_block_k=128, fixed_warps=8, fixed_stages=2)
    assert linear_overlap_tiles(256, parameters) == ((32, 64), (64, 128))
    assert linear_overlap_tiles(33, parameters) == ((32, 64), (32, 128))
    assert (parameters.fixed_block_k, parameters.fixed_warps, parameters.fixed_stages) == (
        128, 8, 2,
    )
    fixed = LinearParameters(fixed_max_block_m=64, fixed_block_n=256)
    assert linear_overlap_tiles(256, fixed) == ((64, 256),)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cold_selection_rejects_only_resource_infeasible_candidates() -> None:
    from triton.runtime.errors import OutOfResources

    from cubic_serve.ops.cuda.launch_cost import select_launch

    output = torch.zeros(8, device="cuda")

    def bind(point):
        if point == "oversized":
            raise OutOfResources(200000, 100000, "shared memory")
        if point == "invalid":
            raise ValueError("invalid arithmetic contract")

    assert (
        select_launch(
            ("oversized", "valid"),
            device=output.device,
            bind=bind,
            launch=lambda: output.fill_(2),
            snapshot=lambda: (output,),
        )
        == "valid"
    )
    with pytest.raises(ValueError, match="arithmetic"):
        select_launch(
            ("invalid", "valid"),
            device=output.device,
            bind=bind,
            launch=lambda: output.fill_(2),
            snapshot=lambda: (output,),
        )
    with pytest.raises(RuntimeError, match="resource-qualified"):
        select_launch(
            ("oversized",),
            device=output.device,
            bind=bind,
            launch=lambda: output.fill_(2),
            snapshot=lambda: (output,),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_resource_introspection_reads_a_live_graph_and_preserves_execution() -> None:
    x = torch.ones(4096, device="cuda")
    y = torch.empty_like(x)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    try:
        with torch.cuda.graph(graph):
            torch.add(x, x, out=y)
        resources = graph_kernel_resources(graph)
        assert resources and all(r.blocks > 0 and r.threads > 0 for r in resources)
        device = device_resources(x.device)
        assert device.sms > 0 and device.registers > 0 and device.shared > 0
        graph.instantiate()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, torch.full_like(y, 2))
    finally:
        graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_plain_selection_preserves_dtype_and_irregular_reduction(dtype) -> None:
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.ops.strategies import OperatorConfig

    generator = torch.Generator(device="cuda").manual_seed(71)
    weight = torch.randn((193, 257), device="cuda", dtype=dtype, generator=generator) * 0.01
    fixed = PackedLinearOperators(OperatorConfig.from_dict({
        "version": 1, "overrides": {"launch": {"max_candidates": 1}},
    }))
    estimated = PackedLinearOperators()
    for rows in (1, 33, 128):
        x = torch.randn((rows, 257), device="cuda", dtype=dtype, generator=generator)
        expected = fixed.linear(x, weight)
        actual = estimated.linear(x, weight)
        torch.cuda.synchronize()
        assert actual.dtype == dtype
        assert torch.equal(actual, expected)
        # Rebinding the same shape must reuse the choice without changing bits.
        assert torch.equal(estimated.linear(x, weight), expected)


def test_launch_parameters_are_serialized_and_explicit_choices_are_authoritative() -> None:
    from cubic_serve.ops.strategies import (
        OperatorConfig,
        launch_parameters,
        operator_strategy_scope,
        parameter_overrides,
    )

    config = OperatorConfig.from_dict({
        "version": 1, "overrides": {
            "launch": {"max_candidates": 2, "max_block_n": 128, "calibration_samples": 5},
            "moe": {"gemm_warps": 4},
        },
    })
    with operator_strategy_scope(config, device="any-device", model="any-model"):
        assert launch_parameters().max_candidates == 2
        assert launch_parameters().calibration_samples == 5
        assert parameter_overrides("moe") == frozenset({"gemm_warps"})
        assert parameter_overrides("linear", n=193, k=257) == frozenset()
    assert launch_parameters() == LaunchParameters()


def test_partitioned_host_transport_has_independent_cost_and_error_gate() -> None:
    from dataclasses import replace

    from cubic_serve.ops.cuda.launch_cost import OrderedCollectiveCost

    def observation(payload):
        return payload, (20 + .2 * payload, 30 + .1 * payload,
                         2 + .08 * payload, 10 + .03 * payload)

    cost = OrderedCollectiveCost.fit([observation(p) for p in (10, 100, 400, 1000)],
                                    [observation(p) for p in (50, 600)])
    args = dict(partitioned=False, host_selected=True,
                min_gain_percent=5, max_error_percent=20)
    assert not cost.select_host_partition(50, **args)
    assert cost.select_host_partition(600, **args)
    assert not cost.select_host_partition(2000, **args)
    assert not replace(cost, interval_error_fractions=(
        *cost.interval_error_fractions[:3], (.3,) * 3,
    )).select_host_partition(600, **args)
    assert not replace(cost, host_partition=None).select_host_partition(600, **args)
    # A failed fourth path must not invalidate an otherwise qualified one-shot.
    noisy = OrderedCollectiveCost.fit(
        [(p, (*observation(p)[1][:3], 10.0)) for p in (10, 100, 400, 1000)],
        [(p, (*observation(p)[1][:3], 10.0)) for p in (50, 600)],
    )
    assert noisy.host_partition is None
    assert noisy.select_host(50, partitioned=False, min_gain_percent=5, max_error_percent=20)


def test_unselected_transport_error_does_not_veto_host_candidate() -> None:
    from dataclasses import replace

    from cubic_serve.ops.cuda.launch_cost import OrderedCollectiveCost

    def observation(payload):
        return payload, (20 + .2 * payload, 30 + .1 * payload,
                         2 + .08 * payload, 10 + .03 * payload)

    anchors = [observation(p) for p in (10, 100, 400, 1000)]
    holdouts = [observation(50), (600, (500, 90, 50, 28))]
    cost = OrderedCollectiveCost.fit(anchors, holdouts)
    assert cost.error_fraction > .2
    args = dict(partitioned=True, min_gain_percent=5, max_error_percent=20)
    assert cost.select_host(600, **args)
    assert cost.select_host_partition(600, host_selected=False, **args)
    assert not cost.select_host(600, **dict(args, partitioned=False))
    bad = replace(cost, interval_error_fractions=(
        cost.interval_error_fractions[0], (.3,) * 3, *cost.interval_error_fractions[2:],
    ))
    assert not bad.select_host_partition(600, host_selected=False, **args)
    assert not replace(cost, interval_error_fractions=(
        *cost.interval_error_fractions[:3], (.3,) * 3,
    )).select_host_partition(
        600, host_selected=True, **args)


def test_host_transport_must_beat_selected_legacy_and_pass_holdout() -> None:
    from dataclasses import replace

    from cubic_serve.ops.cuda.launch_cost import OrderedCollectiveCost

    def observation(payload):
        return payload, (20 + .2 * payload, 30 + .05 * payload, 8 + .08 * payload)

    cost = OrderedCollectiveCost.fit([observation(p) for p in (10, 100, 400, 1000)],
                                     [observation(p) for p in (50, 600)])
    assert cost.select_host(50, partitioned=False, min_gain_percent=5, max_error_percent=20)
    assert not cost.select_host(1000, partitioned=True, min_gain_percent=5, max_error_percent=20)
    assert not cost.select_host(2000, partitioned=False, min_gain_percent=5, max_error_percent=20)
    bad = replace(cost, interval_error_fractions=(
        *cost.interval_error_fractions[:2], (.3,) * 3, cost.interval_error_fractions[3],
    ))
    assert not bad.select_host(50, partitioned=False, min_gain_percent=5, max_error_percent=20)
    # A failed third-transport fit leaves the existing two-transport rule intact.
    noisy = OrderedCollectiveCost.fit(
        [(p, (*observation(p)[1][:2], 5.0)) for p in (10, 100, 400, 1000)],
        [(50, (*observation(50)[1][:2], 5.0))])
    assert noisy.host is None
    assert noisy.predict(50) == cost.predict(50)


def test_persistent_launch_choice_binds_source_fixture_device_and_candidate_order(tmp_path, monkeypatch):
    from cubic_serve.ops.cuda import launch_cost as module
    monkeypatch.setenv('CUBIC_SERVE_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(module, '_launch_calibration_sources', lambda: 'implementation-A')
    device = ['GPU-model-A', 120, 156]
    monkeypatch.setattr('cubic_serve.ops.cuda.cache_identity.gpu_model', lambda d: tuple(device))
    points = (MatmulGeometry(16, 16, 128, 4, 3), MatmulGeometry(32, 16, 128, 4, 3))
    key = ('synthetic-recurrence', torch.bfloat16, 4, 24, 128, points)
    path = module._launch_calibration_path(key, range(2), torch.device('cuda', 0))
    assert path is not None
    assert path == module._launch_calibration_path(key, range(2), torch.device('cuda', 0))
    assert path != module._launch_calibration_path((*key, True), range(2), torch.device('cuda', 0))
    assert path != module._launch_calibration_path(key, (1, 0), torch.device('cuda', 0))
    assert module._launch_calibration_path((object(),), range(2), torch.device('cuda', 0)) is None
    device[0] = 'GPU-model-B'
    assert path != module._launch_calibration_path(key, range(2), torch.device('cuda', 0))
    device[0] = 'GPU-model-A'
    monkeypatch.setattr(module, '_launch_calibration_sources', lambda: 'implementation-B')
    assert path != module._launch_calibration_path(key, range(2), torch.device('cuda', 0))


def test_launch_choice_receipt_rejects_corruption_failed_equivalence_and_bad_measurements(tmp_path):
    import hashlib
    import json
    from cubic_serve.ops.cuda import launch_cost as module
    path = tmp_path / ('a' * 64 + '.json')
    module._write_launch_calibration(path, 1, [3., 1., float('inf')])
    assert module._read_launch_calibration(path, 3) == 1
    assert module._read_launch_calibration(path, 2) is None
    original = json.loads(path.read_text())
    for name, value in [('chosen', 0), ('fixture_equivalence', False), ('key', 'wrong'),
                        ('scores', [float('nan'), 1., None]), ('scores', [0., -1., None])]:
        body = dict(original['body']);body[name] = value
        # Even a self-consistent envelope cannot bless invalid measured evidence.
        digest = hashlib.blake2s(json.dumps(body, sort_keys=True).encode()).hexdigest()
        path.write_text(json.dumps(dict(body=body, digest=digest)))
        assert module._read_launch_calibration(path, 3) is None
    path.write_text('{partial')
    assert module._read_launch_calibration(path, 3) is None


@pytest.mark.skipif(os.environ.get('CUBIC_TEST_QUANT_CUDA') != '1', reason='explicit CUDA audit')
def test_launch_choice_hit_skips_retiming_but_preserves_new_instance_bind_and_prepare(tmp_path, monkeypatch):
    from cubic_serve.ops.cuda import launch_cost as module
    path = tmp_path / ('b' * 64 + '.json')
    module._write_launch_calibration(path, 1, [2., 1.])
    monkeypatch.setattr(module, '_launch_calibration_path', lambda *a: path)
    calls = []
    def no_launch():
        raise AssertionError('a complete tuning receipt must not re-run fixture timings')
    choice = module.select_launch((10, 20), device=torch.device('cuda', 0),
        bind=lambda point: calls.append(('bind', point)),
        prepare=lambda: calls.append(('prepare',)), launch=no_launch,
        snapshot=no_launch, cache_key=('source-independent-fixture',))
    assert choice == 20 and calls == [('bind', 20), ('prepare',)]


@pytest.mark.skipif(os.environ.get('CUBIC_TEST_QUANT_CUDA') != '1', reason='explicit CUDA audit')
def test_packed_row_receipt_uses_real_encoding_and_operator_policy_key(tmp_path, monkeypatch):
    from cubic_serve.quantization.common import small_rows_cost as module
    from cubic_serve.quantization.extensions.int import linear
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.quantization.common.marlin.parameters import MarlinParameters
    from tests.ops.test_int_quantization import fixture
    monkeypatch.setenv('CUBIC_SERVE_CACHE_DIR', str(tmp_path))
    module._costs.clear()
    matrix, _ = fixture(8, n=256, k=128, group=64, sym=True)
    x = torch.zeros((1, 128), dtype=torch.bfloat16, device='cuda:0')
    out = torch.empty((1, 256), dtype=x.dtype, device=x.device)
    operators = PackedLinearOperators()
    cost = module.Cost((.001, 0., 0., 0., 0.), (1., 0., 0.), .01, .01, 1, 4,
                       module.Geometry(8, 64, 1, 8), (('full', .01),))
    calls = []
    monkeypatch.setattr(module, '_fit', lambda *a: calls.append('fit') or cost)
    monkeypatch.setattr(module, 'bind', lambda *a, **kw: 'native-binding')
    args = operators, x, (matrix,), (out,), MarlinParameters(), linear
    assert module._choose_simt(*args) == 'native-binding'
    assert calls == ['fit']
    files = list((tmp_path / 'calibration/launch').glob('*.json'))
    assert files, 'real quantization and operator metadata must have a portable cache identity'
    module._costs.clear()
    assert module._choose_simt(*args) == 'native-binding'
    assert calls == ['fit']
def test_normal_linear_selection_never_constructs_candidate_fixture(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from cubic_serve.ops.cuda import cache_identity, launch_cost, startup

    monkeypatch.setenv("CUBIC_SERVE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache_identity, "gpu_model", lambda d: ("GPU model",))
    monkeypatch.setattr(torch.cuda, "device", lambda d: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(startup, "startup_cuda_lock", lambda d: nullcontext())
    monkeypatch.setattr(torch, "randn", lambda *a, **kw: pytest.fail("startup generated a fixture"))
    x, weight = torch.empty(1, 64), torch.empty(64, 64)
    tiles = tuple(SimpleNamespace(kernel=SimpleNamespace(asm={"cubin": image}, n_spills=0))
                  for image in (b"binary-A", b"binary-B"))
    cost = launch_cost.LinearLaunchCost(DeviceResources(80, 2048, 65536, 65536), 0., ())
    with startup.startup_weight_identity(("weights",), allow_calibration=False):
        assert cost.select(tiles, x, (weight,), grouped=False) is tiles[0]
        assert tuple(tmp_path.rglob("*.json"))
        cost.choices.clear()
        assert cost.select(tiles, x, (weight,), grouped=False) is tiles[0]


def test_gdn_restored_geometry_never_allocates_row_sized_calibration_states(monkeypatch):
    from types import SimpleNamespace
    from cubic_serve.ops.cuda.linear_attention import gated_delta_decode as module

    device = torch.device("cuda:0")
    primitive = object.__new__(module.GatedDeltaDecodePrimitive)
    primitive._meta = (1024, 16, 48, 128, 128, 4, device, .125, 1e-6, True)
    primitive._abis = ((None, None, torch.float32),) * 11
    primitive._block_v, primitive._block_k = 16, 128
    primitive._warps, primitive._stages, primitive._activation_type = 4, 3, "fp32"
    choices = {}
    monkeypatch.setattr(module, "parameter_overrides", lambda *a, **kw: {})
    monkeypatch.setattr(module, "current_linear_cost", lambda: None)
    monkeypatch.setattr(module, "device_resources", lambda d: None)
    monkeypatch.setattr(module, "startup_operator_choices", lambda d: choices)
    monkeypatch.setattr(module, "matmul_geometries", lambda *a, **kw: (
        MatmulGeometry(16, 16, 128, 4, 3), MatmulGeometry(32, 16, 128, 4, 3)))
    monkeypatch.setattr(torch, "Generator", lambda *a, **kw: pytest.fail("allocated a fixture"))

    def restore(candidates, **kwargs):
        kwargs["bind"](1)
        kwargs["prepare"]()  # Existing cache-hit callback contract stays intact.
        return 1

    monkeypatch.setattr(module, "select_launch", restore)
    packed = SimpleNamespace(shape=(1024, 10240), dtype=torch.bfloat16,
                             element_size=lambda: 2)
    primitive._select_geometry(packed, SimpleNamespace(dtype=torch.bfloat16))
    assert primitive._block_v == 32 and tuple(choices.values()) == (1,)


@pytest.mark.parametrize("grouped,partitions", [(False, 1), (False, 2), (True, 2)])
def test_linear_price_identity_tracks_executed_images_and_merge_not_neighboring_native_files(
    monkeypatch, grouped, partitions,
):
    from types import SimpleNamespace
    from cubic_serve.ops.cuda import launch_cost as module
    from cubic_serve.ops.cuda.linear import grouped as grouped_module
    from cubic_serve.quantization.common import kernels as plain_module

    def unrelated_scan(*args, **kwargs):
        pytest.fail("linear prices must not depend on unrelated sibling CUDA sources")
    monkeypatch.setattr(module, "module_kernels", unrelated_scan)
    identities = []
    def restored(candidates, **kwargs):
        identities.append(kwargs["kernels"])
        # No random operand allocation, GPU work, timing, or comparison on a hit.
        kwargs["bind"](candidates[-1])
        return candidates[-1]
    monkeypatch.setattr(module, "select_launch", restored)
    def select(image=b"candidate-A", merge="merge-A"):
        monkeypatch.setattr(plain_module, "_merge_split_k", SimpleNamespace(cache_key=merge))
        monkeypatch.setattr(grouped_module, "_grouped_partition_merge",
                            SimpleNamespace(cache_key="grouped-" + merge))
        tiles = [SimpleNamespace(kernel=SimpleNamespace(asm={"cubin": image}, n_spills=0)),
                 SimpleNamespace(kernel=SimpleNamespace(asm={"cubin": b"candidate-B"}, n_spills=0))]
        cost = module.LinearLaunchCost(DeviceResources(96, 2048, 65536, 65536), 0., ())
        x, weight = torch.empty((4, 8)), torch.empty((3, 8))
        assert cost.select(tiles, x, (weight,), grouped=grouped, masked=grouped,
                           partitions=partitions) is tiles[-1]
        return identities[-1]
    original = select()
    assert select() == original
    assert select(image=b"changed executable") != original
    if partitions > 1:
        assert select(merge="changed merge executable") != original
    else:
        assert select(merge="unused merge executable") == original
