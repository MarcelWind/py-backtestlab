"""Vectorized signal / weight generation for batch permutation tests.

Computes entry signals, position masks, and portfolio weights for all
permutations simultaneously using boolean masks and numpy operations.

This module replicates the core logic of
:class:`~strategies.weather_market_strategy.WeatherMarketImbalanceStrategy`
but operates on 3D indicator arrays produced by
:mod:`~stratlab.strategy.vectorized_indicators`.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Regime classification (vectorised)
# ---------------------------------------------------------------------------

def classify_regime(
    above_mean_pct: np.ndarray,
    above_1sd_pct: np.ndarray,
    below_minus_1sd_pct: np.ndarray,
    within_1sd_pct: np.ndarray,
    mean_reversion: np.ndarray,
    *,
    imbalance_above_mean_threshold: float = 60.0,
    imbalance_above_1sd_threshold: float = 20.0,
    imbalance_up_below_mean_cap: float = 40.0,
    imbalance_below_1sd_threshold: float = 30.0,
    imbalance_down_above_mean_cap: float = 40.0,
    mean_reversion_threshold: float = 0.6,
    balanced_within_1sd_threshold: float = 70.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify market regime for all (perm, bar, asset) simultaneously.

    Returns
    -------
    regime : int array, same shape as inputs
        0 = no tradeable regime, 1 = Imb. Up (long), 2 = Imb. Down (short)
    confidence : float array, same shape as inputs
    """
    shape = above_mean_pct.shape

    regime = np.zeros(shape, dtype=np.int8)
    confidence = np.zeros(shape, dtype=np.float32)

    # Imb. Up
    imb_up = (
        (above_mean_pct > imbalance_above_mean_threshold)
        & (above_1sd_pct > imbalance_above_1sd_threshold)
        & ((100.0 - above_mean_pct) < imbalance_up_below_mean_cap)
    )
    regime[imb_up] = 1
    confidence[imb_up] = np.minimum(
        1.0,
        (above_mean_pct[imb_up] - 50) / 50.0 + above_1sd_pct[imb_up] / 100.0,
    )

    # Imb. Down (only where not already Imb. Up)
    imb_down = (
        ~imb_up
        & (below_minus_1sd_pct > imbalance_below_1sd_threshold)
        & (above_mean_pct < imbalance_down_above_mean_cap)
    )
    regime[imb_down] = 2
    confidence[imb_down] = np.minimum(
        1.0,
        (40 - above_mean_pct[imb_down]) / 50.0 + below_minus_1sd_pct[imb_down] / 100.0,
    )

    return regime, confidence


# ---------------------------------------------------------------------------
# Entry filter chain
# ---------------------------------------------------------------------------

def apply_entry_filters(
    regime: np.ndarray,
    *,
    vwap_slope: np.ndarray | None = None,
    volume_imbalance: np.ndarray | None = None,
    cum_buy_delta: np.ndarray | None = None,
    cum_sell_delta: np.ndarray | None = None,
    cum_buy_delta_mean: np.ndarray | None = None,
    cum_buy_delta_std: np.ndarray | None = None,
    # Filter parameters
    use_vwap_slope_filter: bool = False,
    max_vwap_slope_for_short: float = float("inf"),
    min_vwap_slope_for_long: float = float("-inf"),
    use_volume_imbalance_filter: bool = False,
    max_volume_imbalance_pct_for_short: float = float("inf"),
    min_volume_imbalance_pct_for_long: float = float("-inf"),
    use_buy_cvd_3sd_gate: bool = False,
    use_buy_cvd_filter: bool = False,
    max_buy_cvd_for_short: float = float("inf"),
    min_buy_cvd_for_long: float = float("-inf"),
    use_sell_cvd_filter: bool = False,
    min_sell_cvd_for_short: float = float("-inf"),
    max_sell_cvd_for_long: float = float("inf"),
    # Minimum window size (bars from start before entries are allowed)
    min_bars: int = 2,
) -> np.ndarray:
    """Produce a boolean entry-allowed mask.

    Parameters
    ----------
    regime : (n_perms, n_bars, n_assets)
        1 = long, 2 = short, 0 = no regime.
    ... : optional indicator arrays, same shape.
    min_bars : int
        Bars at the start of the series where entries are suppressed.

    Returns
    -------
    entry_mask : bool array (n_perms, n_bars, n_assets)
        True where an entry is allowed.
    """
    is_long = regime == 1
    is_short = regime == 2
    has_regime = is_long | is_short

    # Start with regime-qualifying entries
    ok = has_regime.copy()

    # Suppress early bars
    ok[:, :min_bars, :] = False

    # VWAP slope filter
    if use_vwap_slope_filter and vwap_slope is not None:
        slope_ok_short = np.isfinite(vwap_slope) & (vwap_slope <= max_vwap_slope_for_short)
        slope_ok_long = np.isfinite(vwap_slope) & (vwap_slope >= min_vwap_slope_for_long)
        slope_ok = np.where(is_short, slope_ok_short, np.where(is_long, slope_ok_long, True))
        ok &= slope_ok

    # Volume imbalance filter
    if use_volume_imbalance_filter and volume_imbalance is not None:
        vim = volume_imbalance
        vim_ok_short = np.isfinite(vim) & (vim <= max_volume_imbalance_pct_for_short)
        vim_ok_long = np.isfinite(vim) & (vim >= min_volume_imbalance_pct_for_long)
        vim_ok = np.where(is_short, vim_ok_short, np.where(is_long, vim_ok_long, True))
        ok &= vim_ok

    # Buy CVD 3-sigma gate
    if use_buy_cvd_3sd_gate and cum_buy_delta is not None and cum_buy_delta_mean is not None and cum_buy_delta_std is not None:
        thr = cum_buy_delta_mean - 3.0 * cum_buy_delta_std
        # Compute band position: nearest_band_index equivalent
        # For the gate, we need band_idx <= -3 (short) or >= 3 (long)
        # Approximate by checking if delta is below -3sd threshold
        gate_short = cum_buy_delta <= thr
        gate_long = cum_buy_delta >= (cum_buy_delta_mean + 3.0 * cum_buy_delta_std)
        gate_ok = np.where(is_short, gate_short, np.where(is_long, gate_long, True))
        ok &= gate_ok

    # Buy CVD filter
    if use_buy_cvd_filter and cum_buy_delta is not None:
        bcd = cum_buy_delta
        bcd_ok_short = np.isfinite(bcd) & (bcd < max_buy_cvd_for_short)
        bcd_ok_long = np.isfinite(bcd) & (bcd > min_buy_cvd_for_long)
        bcd_ok = np.where(is_short, bcd_ok_short, np.where(is_long, bcd_ok_long, True))
        ok &= bcd_ok

    # Sell CVD filter
    if use_sell_cvd_filter and cum_sell_delta is not None:
        scd = cum_sell_delta
        scd_ok_short = np.isfinite(scd) & (scd > min_sell_cvd_for_short)
        scd_ok_long = np.isfinite(scd) & (scd < max_sell_cvd_for_long)
        scd_ok = np.where(is_short, scd_ok_short, np.where(is_long, scd_ok_long, True))
        ok &= scd_ok

    return ok


# ---------------------------------------------------------------------------
# Position management: hold-to-end
# ---------------------------------------------------------------------------

def hold_to_end_positions(
    entry_mask: np.ndarray,
    regime: np.ndarray,
) -> np.ndarray:
    """Compute position mask using hold-to-end logic.

    For each (perm, asset), once the first entry bar is found, the position
    is held until the end of the series.

    Parameters
    ----------
    entry_mask : bool (n_perms, n_bars, n_assets)
    regime : int (n_perms, n_bars, n_assets) — 1=long, 2=short

    Returns
    -------
    position_side : int (n_perms, n_bars, n_assets)
        0 = flat, +1 = long, -1 = short.  Once entered, stays until end.
    """
    n_p, n_b, n_a = entry_mask.shape

    # Side at entry: +1 for long, -1 for short
    side_at_entry = np.where(regime == 1, 1, np.where(regime == 2, -1, 0))
    # Zero out non-entry bars
    side_signal = np.where(entry_mask, side_at_entry, 0)

    # For hold-to-end: find first non-zero bar per (perm, asset) and
    # forward-fill.  np.maximum.accumulate works because once a position
    # is taken, it persists — but we need to handle the sign.
    # Instead, use argmax on the entry mask to find first entry bar.
    position_side = np.zeros((n_p, n_b, n_a), dtype=np.int8)
    for p in range(n_p):
        for a in range(n_a):
            entries = entry_mask[p, :, a]
            if not entries.any():
                continue
            first_bar = int(np.argmax(entries))
            side = side_signal[p, first_bar, a]
            position_side[p, first_bar:, a] = side

    return position_side


def hold_to_end_positions_fast(
    entry_mask: np.ndarray,
    regime: np.ndarray,
) -> np.ndarray:
    """Vectorized hold-to-end using cumulative max trick.

    Faster than :func:`hold_to_end_positions` for large n_perms.
    """
    n_p, n_b, n_a = entry_mask.shape

    # Assign side at each potential entry
    side_at_entry = np.where(regime == 1, 1, np.where(regime == 2, -1, 0))
    side_signal = np.where(entry_mask, side_at_entry, 0).astype(np.int8)

    # Create a bar-index array that's 0 where no entry, bar_index+1 where entry
    bar_idx = np.arange(1, n_b + 1, dtype=np.int32).reshape(1, -1, 1)
    entry_bar_idx = np.where(entry_mask, bar_idx, 0)

    # Find first entry bar per (perm, asset) via argmax
    # argmax returns first occurrence of max value; if all zeros returns 0
    has_any_entry = entry_mask.any(axis=1)  # (n_p, n_a)
    first_entry_bar = np.argmax(entry_mask, axis=1)  # (n_p, n_a) bar indices

    # Extract side at first entry bar
    p_idx = np.arange(n_p).reshape(-1, 1)
    a_idx = np.arange(n_a).reshape(1, -1)
    first_side = side_signal[p_idx, first_entry_bar, a_idx]  # (n_p, n_a)
    first_side = np.where(has_any_entry, first_side, 0)

    # Build output: position_side[p, b, a] = first_side[p, a] if b >= first_entry_bar[p, a]
    bar_range = np.arange(n_b).reshape(1, -1, 1)  # (1, n_b, 1)
    first_bar_3d = first_entry_bar.reshape(n_p, 1, n_a)  # (n_p, 1, n_a)
    in_position = bar_range >= first_bar_3d

    position_side = np.where(
        in_position & has_any_entry.reshape(n_p, 1, n_a),
        first_side.reshape(n_p, 1, n_a),
        0,
    ).astype(np.int8)

    return position_side


# ---------------------------------------------------------------------------
# Portfolio weights from position mask
# ---------------------------------------------------------------------------

def compute_weights(
    position_side: np.ndarray,
) -> np.ndarray:
    """Compute equal-weight portfolio weights from position side array.

    Parameters
    ----------
    position_side : int (n_perms, n_bars, n_assets)
        0 = flat, +1 = long, -1 = short.

    Returns
    -------
    weights : float (n_perms, n_bars, n_assets)
        Equal-weight among active positions, signed by side.
    """
    active = (position_side != 0).astype(np.float32)
    n_active = active.sum(axis=2, keepdims=True)  # (n_p, n_b, 1)
    safe_n = np.maximum(n_active, 1.0)
    unit_weight = 1.0 / safe_n
    return position_side.astype(np.float32) * unit_weight


# ---------------------------------------------------------------------------
# Portfolio returns from weights + asset returns
# ---------------------------------------------------------------------------

def portfolio_returns(
    weights: np.ndarray,
    returns: np.ndarray,
    lookback: int = 1,
) -> np.ndarray:
    """Compute per-bar portfolio returns.

    Parameters
    ----------
    weights : (n_perms, n_bars, n_assets)
    returns : (n_perms, n_bars, n_assets) — per-bar pct-change
    lookback : int — skip first N bars (to match backtester convention)

    Returns
    -------
    port_returns : (n_perms, n_bars)
    """
    port = np.nansum(weights * returns, axis=2)  # (n_perms, n_bars)
    port[:, :lookback] = 0.0
    return port.astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline: indicators → signals → weights → returns
# ---------------------------------------------------------------------------

def batch_strategy_returns(
    indicators: dict[str, np.ndarray],
    returns: np.ndarray,
    params: dict,
) -> np.ndarray:
    """End-to-end vectorized strategy computation.

    Parameters
    ----------
    indicators : dict
        Output of :func:`~stratlab.strategy.vectorized_indicators.compute_all_indicators`.
    returns : (n_perms, n_bars, n_assets)
    params : dict
        Strategy parameters controlling regime thresholds, filters, etc.

    Returns
    -------
    port_returns : (n_perms, n_bars)
    """
    # Regime classification
    regime, confidence = classify_regime(
        above_mean_pct=indicators["band_position_above_mean_pct"],
        above_1sd_pct=indicators["band_position_above_1sd_pct"],
        below_minus_1sd_pct=indicators["band_position_below_minus_1sd_pct"],
        within_1sd_pct=indicators["band_position_within_1sd_pct"],
        mean_reversion=indicators["mean_reversion"],
        imbalance_above_mean_threshold=params.get("imbalance_above_mean_threshold", 60.0),
        imbalance_above_1sd_threshold=params.get("imbalance_above_1sd_threshold", 20.0),
        imbalance_up_below_mean_cap=params.get("imbalance_up_below_mean_cap", 40.0),
        imbalance_below_1sd_threshold=params.get("imbalance_below_1sd_threshold", 30.0),
        imbalance_down_above_mean_cap=params.get("imbalance_down_above_mean_cap", 40.0),
        mean_reversion_threshold=params.get("mean_reversion_threshold", 0.6),
        balanced_within_1sd_threshold=params.get("balanced_within_1sd_threshold", 70.0),
    )

    # Entry filter chain
    entry_mask = apply_entry_filters(
        regime,
        vwap_slope=indicators.get("vwap_slope"),
        volume_imbalance=indicators.get("volume_imbalance"),
        cum_buy_delta=indicators.get("cum_buy_delta"),
        cum_sell_delta=indicators.get("cum_sell_delta"),
        cum_buy_delta_mean=indicators.get("cum_buy_delta_mean"),
        cum_buy_delta_std=indicators.get("cum_buy_delta_std"),
        use_vwap_slope_filter=params.get("use_vwap_slope_filter", False),
        max_vwap_slope_for_short=params.get("max_vwap_slope_for_short", float("inf")),
        min_vwap_slope_for_long=params.get("min_vwap_slope_for_long", float("-inf")),
        use_volume_imbalance_filter=params.get("use_volume_imbalance_filter", False),
        max_volume_imbalance_pct_for_short=params.get("max_volume_imbalance_pct_for_short", float("inf")),
        min_volume_imbalance_pct_for_long=params.get("min_volume_imbalance_pct_for_long", float("-inf")),
        use_buy_cvd_3sd_gate=params.get("use_buy_cvd_3sd_gate", False),
        use_buy_cvd_filter=params.get("use_buy_cvd_filter", False),
        max_buy_cvd_for_short=params.get("max_buy_cvd_for_short", float("inf")),
        min_buy_cvd_for_long=params.get("min_buy_cvd_for_long", float("-inf")),
        use_sell_cvd_filter=params.get("use_sell_cvd_filter", False),
        min_sell_cvd_for_short=params.get("min_sell_cvd_for_short", float("-inf")),
        max_sell_cvd_for_long=params.get("max_sell_cvd_for_long", float("inf")),
        min_bars=params.get("lookback", 2),
    )

    # Positions (hold-to-end)
    position_side = hold_to_end_positions_fast(entry_mask, regime)

    # Weights
    weights = compute_weights(position_side)

    # Portfolio returns
    lookback = params.get("lookback", 1)
    return portfolio_returns(weights, returns, lookback=lookback)
