"""Cross-sectional signal generators for multi-asset strategies."""

import pandas as pd
import numpy as np


def momentum_score(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Compute momentum (past returns) for each asset.

    Args:
        prices: DataFrame of prices (rows=dates, cols=assets)
        lookback: Number of periods for momentum calculation

    Returns:
        DataFrame of momentum scores (same shape as prices)
    """
    return prices.pct_change(lookback)


def rank_assets(scores: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """
    Rank assets cross-sectionally at each time point.

    Args:
        scores: DataFrame of scores (rows=dates, cols=assets)
        ascending: If False, highest score = rank 1 (default)

    Returns:
        DataFrame of ranks (1 = best)
    """
    return scores.rank(axis=1, ascending=ascending, method="min")


def top_n_mask(scores: pd.DataFrame, n: int, min_score: float | None = None) -> pd.DataFrame:
    """
    Create a boolean mask for top N assets at each time point.

    Args:
        scores: DataFrame of scores (rows=dates, cols=assets)
        n: Number of top assets to select
        min_score: Minimum score threshold (e.g., 0 for positive momentum only)

    Returns:
        Boolean DataFrame (True = selected)
    """
    ranks = rank_assets(scores)
    mask = ranks <= n

    # Apply minimum score filter if specified
    if min_score is not None:
        mask = mask & (scores >= min_score)

    return mask


def equal_weight_allocation(mask: pd.DataFrame) -> pd.DataFrame:
    """
    Compute equal weights for selected assets.

    Args:
        mask: Boolean DataFrame of selected assets

    Returns:
        DataFrame of weights (sums to 1 per row, or 0 if none selected)
    """
    n_selected = mask.sum(axis=1)
    weights = mask.astype(float).div(n_selected.replace(0, np.nan), axis=0)
    return weights.fillna(0)


def apply_signals(
    current_weights: np.ndarray,
    signals: np.ndarray | pd.Series,
    long_only: bool = True,
    max_position: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Apply delta signals to current weights to get new weights.

    Signals represent the CHANGE in position (deltas):
    - Signal = +0.1 means "buy 10% more" (add to current position)
    - Signal = -0.05 means "sell 5%" (subtract from current position)
    - Signal = 0 means "hold current position"

    Args:
        current_weights: Array of current portfolio weights
        signals: Array of delta signals (change in position for each asset)
        long_only: If True, clip negative positions to 0
        max_position: Maximum allowed position size per asset (default 1.0)
        normalize: If True and total exceeds 1.0, scale down proportionally

    Returns:
        Array of new weights after applying signals
    """
    signals = np.array(signals, dtype=float)
    new_weights = current_weights.copy() + signals

    # Clip individual positions to max
    new_weights = np.clip(new_weights, -max_position if not long_only else 0, max_position)

    # Handle long-only constraint
    if long_only:
        new_weights = np.maximum(new_weights, 0)

    # Normalize if total exposure exceeds 1.0 (scale down proportionally)
    if normalize:
        total = np.sum(np.abs(new_weights))
        if total > 1.0:
            new_weights = new_weights / total

    return new_weights


def positions_to_weights(
    positions: np.ndarray | pd.Series,
    normalize: bool = True,
    long_only: bool = True,
    min_weight: float = 0.0,
) -> np.ndarray:
    """
    Convert target positions directly to portfolio weights.
    For delta signals (rate of change), use apply_signals() instead.

    Positions can be:
    - Binary: 1 = buy, 0 = no position, -1 = short (if long_only=False)
    - Scores: any numeric values (will be normalized)
    - Amounts: specific values that get normalized to sum to 1

    Args:
        positions: Array of positions for each asset
        normalize: If True, normalize so weights sum to 1 (or -1 for short)
        long_only: If True, clip negative positions to 0
        min_weight: Minimum weight threshold (smaller weights set to 0)

    Returns:
        Array of weights (same length as positions)
    """
    weights = np.array(positions, dtype=float)

    # Handle long-only constraint
    if long_only:
        weights = np.maximum(weights, 0)

    # Apply minimum weight threshold
    weights[np.abs(weights) < min_weight] = 0

    # Normalize
    if normalize:
        total = np.sum(np.abs(weights))
        if total > 0:
            weights = weights / total

    return weights


def scores_to_weights(
    scores: np.ndarray | pd.Series,
    top_n: int | None = None,
    min_score: float | None = None,
    weight_by_score: bool = False,
) -> np.ndarray:
    """
    Convert scores to weights with optional filtering.

    Args:
        scores: Array of scores for each asset (higher = better)
        top_n: Only allocate to top N assets (None = no limit)
        min_score: Minimum score to be included (None = no threshold)
        weight_by_score: If True, weight proportional to score; if False, equal weight

    Returns:
        Array of weights
    """
    scores = np.array(scores, dtype=float)
    n_assets = len(scores)

    # Create mask for valid assets
    mask = np.ones(n_assets, dtype=bool)

    # Apply minimum score filter
    if min_score is not None:
        mask &= scores >= min_score

    # Apply top-N filter
    if top_n is not None and top_n < np.sum(mask):
        # Get indices of top N scores among valid assets
        valid_scores = np.where(mask, scores, -np.inf)
        top_indices = np.argsort(valid_scores)[-top_n:]
        new_mask = np.zeros(n_assets, dtype=bool)
        new_mask[top_indices] = True
        mask &= new_mask

    # Compute weights
    if not np.any(mask):
        return np.zeros(n_assets)

    if weight_by_score:
        # Weight proportional to score
        weights = np.where(mask, np.maximum(scores, 0), 0)
    else:
        # Equal weight
        weights = mask.astype(float)

    # Normalize
    total = np.sum(weights)
    if total > 0:
        weights = weights / total

    return weights
