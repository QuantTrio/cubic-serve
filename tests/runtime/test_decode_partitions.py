"""Graph lanes must preserve recurrence segments and transaction ownership."""

from dataclasses import replace

from cubic_serve.runtime.continuous_batch import BatchRow, independent_decode_partitions


def _rows(lengths):
    rows = []
    for owner, length in enumerate(lengths):
        for offset in range(length):
            rows.append(BatchRow(
                request_id=f"r{owner}", token_id=100 + offset, position=32 + offset,
                state_source_slot=owner, state_destination_slot=100 + len(rows),
                page_source=owner, page_destination=owner, page_table=(owner,),
                segment_length=length if offset == 0 else 0, is_decode=True,
            ))
    return tuple(rows)


def test_whole_segments_fit_two_three_or_four_lanes_with_padding():
    for capacities in ((4, 4), (4, 4, 4), (4, 4, 4, 4)):
        rows = _rows((2,) * (sum(capacities) // 2 - 1))
        assert independent_decode_partitions(rows, capacities)
    assert independent_decode_partitions(_rows((3, 1, 2)), (4, 4))
    assert not independent_decode_partitions(_rows((3, 3)), (4, 4))


def test_cross_lane_read_write_aliases_are_rejected():
    rows = _rows((2, 2))
    assert independent_decode_partitions(rows, (2, 2))
    for field, value in (("state_source_slot", 100), ("state_destination_slot", 100),
                         ("page_source", 0), ("page_destination", 0)):
        changed = (*rows[:2], replace(rows[2], **{field: value}), rows[3])
        # Segment metadata requires all rows to share their input state.
        if field == "state_source_slot":
            changed = (*changed[:3], replace(changed[3], **{field: value}))
        assert not independent_decode_partitions(changed, (2, 2))


def test_partition_eligibility_does_not_admit_prefill_or_duplicate_owners():
    rows = _rows((2, 2))
    assert not independent_decode_partitions((replace(rows[0], is_decode=False), *rows[1:]),
                                             (2, 2))
    repeated = (*rows[:2], *(replace(row, request_id="r0") for row in rows[2:]))
    assert not independent_decode_partitions(repeated, (2, 2))
    assert not independent_decode_partitions(rows, (2,))
