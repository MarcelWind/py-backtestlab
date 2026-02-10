"""Statistical methods for interpreting prediction market prices as implied probabilities."""

import numpy as np
import pandas as pd


def brier_score(forecasts: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Compute the Brier score for probabilistic forecasts.

    Measures calibration of prediction market prices against resolved outcomes.
    Lower is better. Range: [0, 1].

    Args:
        forecasts: Predicted probabilities in [0, 1] (market prices)
        outcomes: Binary outcomes (0 or 1)

    Returns:
        Mean squared error between forecasts and outcomes
    """
    forecasts = np.asarray(forecasts, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((forecasts - outcomes) ** 2))


def brier_score_penalized(
    forecasts: np.ndarray,
    outcomes: np.ndarray,
    penalty_weight: float = 0.5,
) -> float:
    """
    Brier score with an overconfidence penalty.

    Adds a term that punishes forecasts far from 0.5 when the prediction
    lands on the wrong side. A market pricing an event at 0.95 that resolves
    to 0 is penalized more than one pricing at 0.55.

    Args:
        forecasts: Predicted probabilities in [0, 1] (market prices)
        outcomes: Binary outcomes (0 or 1)
        penalty_weight: Scaling factor for the overconfidence penalty

    Returns:
        Penalized Brier score
    """
    forecasts = np.asarray(forecasts, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    base = (forecasts - outcomes) ** 2

    wrong_side = (
        ((forecasts > 0.5) & (outcomes == 0))
        | ((forecasts < 0.5) & (outcomes == 1))
    )
    confidence = np.abs(forecasts - 0.5)
    penalty = np.where(wrong_side, confidence * penalty_weight, 0.0)

    return float(np.mean(base + penalty))


def cdf_from_ranges(
    boundaries: np.ndarray,
    prices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an implied CDF from bracket prediction market prices.

    Given N+1 sorted boundaries defining N ranges and N market prices
    representing implied probability per range, returns the cumulative
    distribution evaluated at each boundary.

    Args:
        boundaries: Sorted array of N+1 boundary values defining the ranges
        prices: Array of N market prices (implied probabilities) for each range

    Returns:
        Tuple of (boundaries, cdf_values) where cdf_values[i] = P(X <= boundaries[i])
    """
    boundaries = np.asarray(boundaries, dtype=float)
    prices = np.asarray(prices, dtype=float)

    total = prices.sum()
    probs = prices / total if total > 0 else np.ones_like(prices) / len(prices)

    cdf = np.concatenate(([0.0], np.cumsum(probs)))
    return boundaries, cdf


def midpoints(boundaries: np.ndarray) -> np.ndarray:
    """
    Compute bucket midpoints from N+1 boundaries.

    Pairs with cdf_from_ranges: pass the same boundaries here to get
    representative values for market_mean, vwap_mean, etc.

    Args:
        boundaries: Sorted array of N+1 boundary values

    Returns:
        Array of N midpoint values
    """
    boundaries = np.asarray(boundaries, dtype=float)
    return (boundaries[:-1] + boundaries[1:]) / 2


def crps(
    cdf_values: np.ndarray,
    cdf_points: np.ndarray,
    outcome: float,
) -> float:
    """
    Compute the Continuous Ranked Probability Score.

    Integrates the squared difference between the forecast CDF and the
    Heaviside step function at the observed outcome. Generalises the Brier
    score to continuous outcomes. Lower is better.

    Args:
        cdf_values: Forecast CDF values F(x) at each evaluation point
        cdf_points: Sorted x-values where the CDF is evaluated
        outcome: Observed outcome value

    Returns:
        CRPS value (non-negative, lower is better)
    """
    cdf_points = np.asarray(cdf_points, dtype=float)
    cdf_values = np.asarray(cdf_values, dtype=float)

    heaviside = np.where(cdf_points >= outcome, 1.0, 0.0)
    integrand = (cdf_values - heaviside) ** 2

    return float(np.trapezoid(integrand, cdf_points))


def market_mean(values: np.ndarray, prices: np.ndarray) -> float:
    """
    Expected value of the underlying as implied by market prices alone.

    Treats prices as unnormalised probabilities over outcome values.
    E[X] = sum(value_i * price_i) / sum(price_i).

    Args:
        values: Outcome values of the underlying (e.g. range midpoints)
        prices: Market prices (implied probabilities) for each outcome

    Returns:
        Price-implied expected value of the underlying
    """
    values = np.asarray(values, dtype=float)
    prices = np.asarray(prices, dtype=float)

    total = prices.sum()
    if total == 0:
        return float(np.mean(values))

    return float(np.sum(values * prices) / total)


def vwap_mean(
    values: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
) -> float:
    """
    Expected value of the underlying implied by prices, weighted by volume.

    Like market_mean but additionally weights each contract by its traded
    volume, so heavily-traded outcomes contribute more.
    E[X] = sum(value_i * price_i * volume_i) / sum(price_i * volume_i).

    Args:
        values: Outcome values of the underlying (e.g. range midpoints)
        prices: Market prices (implied probabilities) for each outcome
        volumes: Traded volume for each outcome contract

    Returns:
        Volume-and-price-weighted expected value of the underlying
    """
    values = np.asarray(values, dtype=float)
    prices = np.asarray(prices, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    weights = prices * volumes
    total = weights.sum()
    if total == 0:
        return float(np.mean(values))

    return float(np.sum(values * weights) / total)


def volume_skew(
    values: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
) -> float:
    """
    Compare volume above vs below the market mean of the underlying.

    Positive skew means more volume on outcomes above the price-implied
    expected value (bullish), negative means below (bearish). Range: [-1, 1].

    Args:
        values: Outcome values of the underlying (e.g. range midpoints)
        prices: Market prices (implied probabilities) for each outcome
        volumes: Traded volume for each outcome contract

    Returns:
        Normalised skew: (vol_above - vol_below) / (vol_above + vol_below)
    """
    values = np.asarray(values, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    mm = market_mean(values, prices)
    vol_above = volumes[values > mm].sum()
    vol_below = volumes[values < mm].sum()

    total = vol_above + vol_below
    if total == 0:
        return 0.0

    return float((vol_above - vol_below) / total)


def volume_entropy(prices: np.ndarray, volumes: np.ndarray) -> float:
    """
    Shannon entropy of the price-weighted volume distribution.

    Weights each outcome's volume by its market price before computing
    entropy. High entropy = conviction spread uniformly (uncertainty).
    Low entropy = conviction concentrated at few outcomes (consensus).

    Args:
        prices: Market prices (implied probabilities) for each outcome
        volumes: Traded volume for each outcome contract

    Returns:
        Shannon entropy in nats
    """
    prices = np.asarray(prices, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    weights = prices * volumes
    total = weights.sum()
    if total == 0:
        return 0.0

    probs = weights / total
    probs = probs[probs > 0]

    return float(-np.sum(probs * np.log(probs)))


def price_volume_divergence(
    values: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
) -> float:
    """
    Distance between volume-weighted and price-only implied means.

    Large divergence signals that heavily-traded contracts imply a
    different expected value than prices alone.

    Args:
        values: Outcome values of the underlying (e.g. range midpoints)
        prices: Market prices (implied probabilities) for each outcome
        volumes: Traded volume for each outcome contract

    Returns:
        Signed difference: vwap_mean - market_mean (positive = volume skews higher)
    """
    mm = market_mean(values, prices)
    vm = vwap_mean(values, prices, volumes)

    return vm - mm
