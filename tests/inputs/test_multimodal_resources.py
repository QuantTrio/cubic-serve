"""Resource exhaustion and abandoned input-worker lifetimes."""

import asyncio
import threading

import pytest

from cubic_serve.inputs.multimodal import (
    EncoderAdmissionError,
    EncoderMemoryBudget,
    EncoderWorkingSet,
    MultimodalInputPolicy,
    MultimodalPreprocessingGate,
    bounded_encoder_groups,
)


@pytest.mark.parametrize("amount", [0, -1, True, 0.1, float("nan"), float("inf")])
def test_invalid_resource_budget_is_rejected_before_loading(amount):
    with pytest.raises(ValueError, match="budget"):
        MultimodalInputPolicy(encoder_memory_bytes=amount)


@pytest.mark.parametrize("fps", [0, -1, True, float("nan"), float("inf")])
def test_invalid_video_sampling_policy_is_rejected_before_loading(fps):
    with pytest.raises(ValueError, match="second"):
        MultimodalInputPolicy(video_frames_per_second=fps)


@pytest.mark.parametrize("frames", [0, 3, 4.0, True])
def test_video_frame_budget_requires_valid_integer_capacity(frames):
    with pytest.raises(ValueError, match="max frames"):
        MultimodalInputPolicy(video_max_frames=frames)


def test_shared_budget_accounts_multiple_encoders_and_reserved_staging():
    budget = EncoderMemoryBudget(MultimodalInputPolicy(encoder_memory_bytes=1000))
    budget.initialize(2000)
    budget.reserve(resident_bytes=400, staging_bytes=200)
    budget.reserve(resident_bytes=300, staging_bytes=100)
    assert (budget.resident_bytes, budget.staging_bytes) == (700, 200)
    with pytest.raises(RuntimeError, match="budget"):
        budget.reserve(resident_bytes=101, staging_bytes=200)
    assert budget.resident_bytes == 700
    with pytest.raises(ValueError, match="staging"):
        with budget.lease(201):
            pytest.fail("over-budget input reached the device")
    with pytest.raises(RuntimeError, match="caller failed"):
        with budget.lease(200):
            raise RuntimeError("caller failed")
    with budget.lease(200):
        pass


def test_capture_admission_includes_staging_and_other_resident_encoders():
    budget = EncoderMemoryBudget(MultimodalInputPolicy(encoder_memory_bytes=4000))
    budget.initialize(8000)
    budget.reserve(resident_bytes=400, staging_bytes=0)
    capacity = budget.capture_capacity(linear_bytes=10, quadratic_bytes=1,
                                       staging_bytes_per_row=10, minimum=4, declared=128)
    assert capacity == 16
    with pytest.raises(RuntimeError, match="minimum"):
        budget.capture_capacity(linear_bytes=100, quadratic_bytes=10,
                                staging_bytes_per_row=100, minimum=4, declared=128)


def test_dynamic_encoder_item_uses_same_budget_and_exclusive_lease():
    budget = EncoderMemoryBudget(MultimodalInputPolicy(encoder_memory_bytes=1000))
    budget.initialize(2000)
    budget.reserve(resident_bytes=400, staging_bytes=200)
    with budget.lease_dynamic(600):
        pass
    with pytest.raises(EncoderAdmissionError, match="required_bytes=1001 budget_bytes=1000"):
        with budget.lease_dynamic(601):
            pytest.fail("over-budget encoder item ran")


@pytest.mark.parametrize("limit,expected", [(1500, 16), (3500, 32)])
def test_measured_graph_residency_bounds_portfolio_growth(limit, expected):
    budget = EncoderMemoryBudget(MultimodalInputPolicy(encoder_memory_bytes=limit * 4))
    budget.initialize(100000)
    # The analytical estimate would admit twice as many rows. Measured sealed
    # bindings/pool residency must stop growth within the existing byte budget.
    assert budget.capture_capacity(linear_bytes=20, quadratic_bytes=1,
                                   staging_bytes_per_row=10, minimum=4, declared=128,
                                   measured_capacity=16,
                                   measured_resident_bytes=16 * 60 + 16**2) == expected
    assert budget.total_bytes == limit * 4


def test_measured_capture_keeps_other_encoders_and_shared_staging_in_budget():
    budget = EncoderMemoryBudget(MultimodalInputPolicy(encoder_memory_bytes=8000))
    budget.initialize(10000)
    budget.reserve(resident_bytes=400, staging_bytes=200)
    assert budget.capture_capacity(linear_bytes=10, quadratic_bytes=1,
                                   staging_bytes_per_row=10, minimum=4, declared=128,
                                   measured_capacity=16, measured_resident_bytes=1216) == 16
    with pytest.raises(ValueError, match="measured capacity"):
        budget.capture_capacity(linear_bytes=10, quadratic_bytes=1,
                                staging_bytes_per_row=10, minimum=4, declared=128,
                                measured_resident_bytes=1)


def test_auto_encoder_requires_a_model_extent_and_reserves_its_full_working_set():
    budget = EncoderMemoryBudget(MultimodalInputPolicy())
    with pytest.raises(ValueError, match="finite model input limit"):
        budget.initialize(80 << 30)
    plan = EncoderWorkingSet(65536, 145664, 1, 4)
    budget.initialize(80 << 30, working_set=plan, graph_bytes=1 << 30)
    assert budget.total_bytes == (1 << 30) + 13841203200
    assert budget.graph_limit_bytes == 1 << 30
    assert plan.fit(budget.total_bytes - budget.graph_limit_bytes) == 65536


@pytest.mark.parametrize("available", [1 << 30, 4 << 30, 16 << 30])
def test_auto_input_extent_closes_against_media_and_language_cache(available):
    plan = EncoderWorkingSet(65536, 145664, 1, 4)
    costs = dict(cache_page_bytes=393216, cache_page_tokens=16,
                 additional_tokens=1024, owner_bytes=16 << 20)
    rows = plan.fit(available, minimum_rows=2048, **costs)
    def required(n):
        tokens = (n + 3) // 4 + 1024
        return plan.transient_bytes(n) + ((tokens + 15) // 16) * 393216 + (16 << 20)
    assert required(rows) <= available
    assert rows % plan.merge_extent == 0
    assert rows == plan.declared_rows or required(rows + plan.merge_extent) > available
    assert rows <= plan.fit(available)


def test_auto_too_small_for_even_one_graph_and_owner_fails_before_work():
    plan = EncoderWorkingSet(65536, 145664, 1, 4)
    with pytest.raises(RuntimeError, match="minimum media input and language cache"):
        plan.fit(1000, minimum_rows=2048, owner_bytes=2000)


def test_media_admission_preserves_cache_pages_after_working_set_subtraction():
    plan = EncoderWorkingSet(128, 13, 1, 4)
    page_bytes, page_tokens, dummy_bytes, scratch_pages = 256, 16, 48, 2
    for available in range(4000, 4300):
        owner_pool = ((available // page_bytes - scratch_pages) * page_bytes - dummy_bytes)
        rows = plan.fit(owner_pool, cache_page_bytes=page_bytes,
                        cache_page_tokens=page_tokens, additional_tokens=1,
                        owner_bytes=96, working_alignment_bytes=page_bytes)
        remaining = available - plan.transient_bytes(rows)
        actual_pool = ((remaining // page_bytes - scratch_pages) * page_bytes - dummy_bytes)
        tokens = rows // plan.merge_extent + 1
        assert actual_pool >= ((tokens + page_tokens - 1) // page_tokens) * page_bytes + 96


@pytest.mark.parametrize("budget_bytes", [4 << 30, int(76.464 * (1 << 30)) // 20])
@pytest.mark.parametrize("rows", [10080, 16544])
def test_large_image_workspace_survives_worst_admitted_graph_residency(budget_bytes, rows):
    budget = EncoderMemoryBudget(MultimodalInputPolicy(encoder_memory_bytes=budget_bytes))
    budget.initialize(80 << 30)
    # Qwen vision geometry: upload/result + mask + tiled attention + activations.
    transient = rows * (11264 + 512 * 16 * 8 + 4304 * 16) + rows**2
    graph_bytes = budget_bytes // 4
    assert budget.graph_capture_fits(resident_bytes=graph_bytes, staging_bytes=0)
    assert not budget.graph_capture_fits(resident_bytes=graph_bytes + 1, staging_bytes=0)
    budget.reserve(resident_bytes=graph_bytes, staging_bytes=0)
    with budget.lease_dynamic(transient):
        pass


def test_graph_residency_can_explain_wide_image_passing_while_tall_image_fails():
    budget = EncoderMemoryBudget(MultimodalInputPolicy(encoder_memory_bytes=4 << 30))
    budget.initialize(80 << 30)
    old_resident = 7 << 28  # A possible retained portfolio; not an A100 observation.
    assert not budget.graph_capture_fits(resident_bytes=old_resident, staging_bytes=0)
    budget.reserve(resident_bytes=old_resident, staging_bytes=0)
    def transient(rows):
        return rows * (11264 + 512 * 16 * 8 + 4304 * 16) + rows**2
    with budget.lease_dynamic(transient(10080)):
        pass
    with pytest.raises(EncoderAdmissionError):
        with budget.lease_dynamic(transient(16544)):
            pytest.fail("the old graph portfolio left insufficient input working space")


@pytest.mark.parametrize("lengths", [[256]*9, [119, 119, 37, 83, 119]])
def test_segment_groups_preserve_all_rows_and_never_split_declared_segments(lengths):
    groups = bounded_encoder_groups(lengths, 512)
    seen = [i for start, stop in groups for i in range(start, stop)]
    assert seen == list(range(len(lengths)))
    assert all(sum(lengths[start:stop]) <= 512 for start, stop in groups)
    with pytest.raises(ValueError, match="segment"):
        bounded_encoder_groups([4, 513, 4], 512)


def test_oversize_encoder_item_gets_a_standalone_dynamic_group():
    lengths = [119, 119, 7688, 37, 83]
    groups = bounded_encoder_groups(lengths, 512, allow_oversize=True)
    assert groups == ((0, 2), (2, 3), (3, 5))
    assert [lengths[index] for start, stop in groups for index in range(start, stop)] == lengths


def test_cancelled_http_request_keeps_worker_slot_until_real_work_finishes():
    async def scenario():
        gate = MultimodalPreprocessingGate(1)
        entered, release, second_entered = threading.Event(), threading.Event(), threading.Event()

        def first():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test worker did not release")
            raise RuntimeError("abandoned worker failure")

        def second():
            second_entered.set()
            return "audio feature"

        task = asyncio.create_task(gate.run(first))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            successor = asyncio.create_task(gate.run(second))
            assert not await asyncio.to_thread(second_entered.wait, 0.05)
            release.set()
            assert await asyncio.wait_for(successor, 3) == "audio feature"
            assert await gate.run(lambda: "video feature") == "video feature"
        finally:
            release.set()

    asyncio.run(scenario())
