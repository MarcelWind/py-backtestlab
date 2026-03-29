"""Permutation/significance helpers for Monte Carlo validation gates."""

from __future__ import annotations

import numpy as np


def empirical_one_tailed_pvalue(observed: float, historical_scores: list[float]) -> float:
    """Estimate one-tailed p-value P(null_score >= observed) from historical scores."""
    valid_scores = [float(x) for x in historical_scores if np.isfinite(x)]
    if not valid_scores:
        return 1.0
    greater_or_equal = sum(1 for score in valid_scores if score >= observed)
    return float(greater_or_equal / len(valid_scores))


def walk_forward_pvalue_from_event_order(
    event_scores: dict[str, float],
    ordered_oos_slugs: list[str],
    historical_prefix_scores: dict[int, list[float]],
) -> float:
    """Compute conservative outsample walk-forward p-value using ordered event prefixes."""
    if not ordered_oos_slugs:
        return 0.0

    running_scores: list[float] = []
    prefix_pvalues: list[float] = []
    for idx, slug in enumerate(ordered_oos_slugs, start=1):
        score = event_scores.get(slug)
        if score is None or not np.isfinite(score):
            break
        running_scores.append(float(score))
        prefix_mean = float(np.mean(running_scores))
        prefix_history = historical_prefix_scores.get(idx, [])
        prefix_pvalues.append(empirical_one_tailed_pvalue(prefix_mean, prefix_history))

    if not prefix_pvalues:
        return 1.0
    return float(max(prefix_pvalues))
