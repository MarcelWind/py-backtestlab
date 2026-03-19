"""Tests for generic IS/OOS event partition helpers."""

from stratlab.validation.partition import compute_is_oos_event_counts, split_events_in_sample_out_of_sample


def test_compute_is_oos_event_counts_rounding() -> None:
    is_count, oos_count = compute_is_oos_event_counts(5, out_of_sample_ratio=0.4)
    assert is_count == 3
    assert oos_count == 2


def test_compute_is_oos_event_counts_single_event_raises() -> None:
    try:
        compute_is_oos_event_counts(1, out_of_sample_ratio=0.4)
        raise AssertionError("Expected ValueError for n_events < 2")
    except ValueError as exc:
        assert "at least 2 events" in str(exc)


def test_split_events_is_deterministic_for_seed() -> None:
    events = [f"event-{i}" for i in range(10)]
    part_a = split_events_in_sample_out_of_sample(events, out_of_sample_ratio=0.4, seed=123)
    part_b = split_events_in_sample_out_of_sample(events, out_of_sample_ratio=0.4, seed=123)

    assert part_a.in_sample_event_slugs == part_b.in_sample_event_slugs
    assert part_a.out_of_sample_event_slugs == part_b.out_of_sample_event_slugs
    assert part_a.n_events_in_sample == 6
    assert part_a.n_events_out_of_sample == 4


def test_split_events_covers_all_without_overlap() -> None:
    events = [f"event-{i}" for i in range(7)]
    part = split_events_in_sample_out_of_sample(events, out_of_sample_ratio=0.4, seed=99)

    is_set = set(part.in_sample_event_slugs)
    oos_set = set(part.out_of_sample_event_slugs)
    assert is_set.isdisjoint(oos_set)
    assert is_set.union(oos_set) == set(events)
