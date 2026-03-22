"""Tests for Monte Carlo permutation gate helpers."""

from stratlab.validation.significance import empirical_one_tailed_pvalue, walk_forward_pvalue_from_event_order


def test_empirical_pvalue_returns_one_for_empty_history() -> None:
    assert empirical_one_tailed_pvalue(0.5, []) == 1.0


def test_empirical_pvalue_counts_greater_or_equal() -> None:
    history = [0.1, 0.2, 0.3, 0.4]
    assert empirical_one_tailed_pvalue(0.35, history) == 0.25
    assert empirical_one_tailed_pvalue(0.4, history) == 0.25
    assert empirical_one_tailed_pvalue(0.05, history) == 1.0


def test_walk_forward_pvalue_uses_max_prefix_pvalue() -> None:
    event_scores = {"oos-a": 0.9, "oos-b": 0.5, "oos-c": 0.8}
    ordered_oos = ["oos-a", "oos-b", "oos-c"]
    historical_prefix_scores = {
        1: [0.1, 0.2, 0.3],
        2: [0.2, 0.3, 0.4],
        3: [0.3, 0.4, 0.5],
    }

    pvalue = walk_forward_pvalue_from_event_order(event_scores, ordered_oos, historical_prefix_scores)

    assert pvalue == 0.0


def test_walk_forward_pvalue_breaks_on_missing_event_score() -> None:
    event_scores = {"oos-a": 0.4}
    ordered_oos = ["oos-a", "oos-b"]
    historical_prefix_scores = {
        1: [0.1, 0.2],
        2: [0.1, 0.2],
    }

    pvalue = walk_forward_pvalue_from_event_order(event_scores, ordered_oos, historical_prefix_scores)

    assert pvalue == 0.0


def test_walk_forward_pvalue_is_neutral_without_oos_events() -> None:
    pvalue = walk_forward_pvalue_from_event_order({}, [], {})
    assert pvalue == 0.0
