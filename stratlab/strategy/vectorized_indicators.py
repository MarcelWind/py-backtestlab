"""Vectorized indicator computations for batch permutation tests.

All functions operate on 3D numpy arrays of shape
``(n_perms, n_bars, n_assets)`` and return arrays of the same shape
(or multi-stat variants).  This eliminates the per-bar Python loop and
per-permutation overhead present in the incremental indicator classes.

Convention: axis 0 = permutation, axis 1 = bar, axis 2 = asset.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------------

def typical_price(
    close: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
) -> np.ndarray:
    """Compute typical price ``(H + L + C) / 3`` element-wise.

    Falls back to *close* where *high* or *low* are missing or NaN.
    """
    if high is None or low is None:
        return close.copy()
    valid = np.isfinite(high) & np.isfinite(low)
    out = close.copy()
    out[valid] = (high[valid] + low[valid] + close[valid]) / 3.0
    return out


# ---------------------------------------------------------------------------
# SdBands  —  expanding mean / std per asset
# ---------------------------------------------------------------------------

def sd_bands_expanding(
    prices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute expanding mean and std for every (perm, bar, asset).

    Parameters
    ----------
    prices : (n_perms, n_bars, n_assets)

    Returns
    -------
    mean : (n_perms, n_bars, n_assets)
    std  : (n_perms, n_bars, n_assets)
    """
    # Use float64 for cumulative sums to avoid precision loss over many bars,
    # then cast the result back to float32 to halve downstream memory.
    prices_f64 = prices.astype(np.float64) if prices.dtype != np.float64 else prices
    cumsum = np.nancumsum(prices_f64, axis=1)
    cumsum_sq = np.nancumsum(prices_f64 ** 2, axis=1)
    counts = np.nancumsum(np.isfinite(prices_f64).astype(np.float64), axis=1)
    counts = np.maximum(counts, 1.0)  # avoid division by zero

    mean = (cumsum / counts).astype(np.float32)
    var = np.maximum(cumsum_sq / counts - (cumsum / counts) ** 2, 0.0)
    std = np.sqrt(var).astype(np.float32)
    return mean, std


def sd_band_levels(
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, np.ndarray]:
    """Derive ±{1,2,3}σ bands from mean and std arrays.

    Returns dict with keys: ``mean, +1sd, -1sd, +2sd, -2sd, +3sd, -3sd``.
    Each value has the same shape as *mean*.
    """
    return {
        "mean": mean,
        "+1sd": mean + std,
        "-1sd": mean - std,
        "+2sd": mean + 2.0 * std,
        "-2sd": mean - 2.0 * std,
        "+3sd": mean + 3.0 * std,
        "-3sd": mean - 3.0 * std,
    }


# ---------------------------------------------------------------------------
# Vwap  —  expanding VWAP
# ---------------------------------------------------------------------------

def vwap_expanding(
    price: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Expanding VWAP: cumsum(price*volume) / cumsum(volume).

    Parameters
    ----------
    price  : (n_perms, n_bars, n_assets)  — typically typical_price output
    volume : (n_perms, n_bars, n_assets)

    Returns
    -------
    vwap : (n_perms, n_bars, n_assets)
    """
    pv = np.where(np.isfinite(price) & np.isfinite(volume), price * volume, 0.0)
    v = np.where(np.isfinite(volume), volume, 0.0)
    cum_pv = np.cumsum(pv, axis=1)
    cum_v = np.cumsum(v, axis=1)
    # Where cumulative volume is zero, fall back to price
    safe_cv = np.where(cum_v > 0, cum_v, 1.0)
    vwap = cum_pv / safe_cv
    vwap = np.where(cum_v > 0, vwap, price)
    return vwap


# ---------------------------------------------------------------------------
# CumulativeYesNoDelta
# ---------------------------------------------------------------------------

def cumulative_yes_no_delta(
    yes_volume: np.ndarray,
    no_volume: np.ndarray,
) -> np.ndarray:
    """Expanding cumulative (yes - no) delta, baseline-shifted to start at 0.

    Parameters
    ----------
    yes_volume, no_volume : (n_perms, n_bars, n_assets)

    Returns
    -------
    delta : (n_perms, n_bars, n_assets)
    """
    yes = np.where(np.isfinite(yes_volume), yes_volume, 0.0)
    no = np.where(np.isfinite(no_volume), no_volume, 0.0)
    cum = np.cumsum(yes - no, axis=1)
    # Baseline shift: subtract first bar value
    baseline = cum[:, 0:1, :]  # keep dims for broadcasting
    return cum - baseline


def cumulative_delta_stats(
    delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expanding mean and std of cumulative delta.

    Parameters
    ----------
    delta : (n_perms, n_bars, n_assets)

    Returns
    -------
    mean : (n_perms, n_bars, n_assets)
    std  : (n_perms, n_bars, n_assets)
    """
    delta_f64 = delta.astype(np.float64) if delta.dtype != np.float64 else delta
    cumsum = np.cumsum(delta_f64, axis=1)
    cumsum_sq = np.cumsum(delta_f64 ** 2, axis=1)
    counts = np.arange(1, delta.shape[1] + 1, dtype=np.float64).reshape(1, -1, 1)
    mean = (cumsum / counts).astype(np.float32)
    var = np.maximum(cumsum_sq / counts - (cumsum / counts) ** 2, 0.0)
    std = np.sqrt(var).astype(np.float32)
    return mean, std


# ---------------------------------------------------------------------------
# BandPosition  —  rolling band-position percentages
# ---------------------------------------------------------------------------

def band_position(
    prices: np.ndarray,
    mean: np.ndarray,
    plus_1sd: np.ndarray,
    minus_1sd: np.ndarray,
    lookback_bars: int,
) -> dict[str, np.ndarray]:
    """Rolling band-position statistics.

    Parameters
    ----------
    prices : (n_perms, n_bars, n_assets)
    mean, plus_1sd, minus_1sd : same shape — expanding band levels
    lookback_bars : int — window size

    Returns
    -------
    Dict with keys ``above_mean_pct, above_1sd_pct, below_minus_1sd_pct,
    within_1sd_pct``, each (n_perms, n_bars, n_assets).
    """
    valid = np.isfinite(prices) & np.isfinite(mean)
    above_mean = (valid & (prices > mean)).astype(np.float32)
    above_1sd = (valid & (prices > plus_1sd)).astype(np.float32)
    below_m1sd = (valid & (prices < minus_1sd)).astype(np.float32)
    within_1sd = (valid & (prices >= minus_1sd) & (prices <= plus_1sd)).astype(np.float32)
    valid_f = valid.astype(np.float32)

    def _rolling_sum(arr: np.ndarray) -> np.ndarray:
        cs = np.cumsum(arr, axis=1)
        out = cs.copy()
        out[:, lookback_bars:, :] = cs[:, lookback_bars:, :] - cs[:, :-lookback_bars, :]
        return out

    total = _rolling_sum(valid_f)
    safe_total = np.maximum(total, 1.0)

    return {
        "above_mean_pct": _rolling_sum(above_mean) / safe_total * 100.0,
        "above_1sd_pct": _rolling_sum(above_1sd) / safe_total * 100.0,
        "below_minus_1sd_pct": _rolling_sum(below_m1sd) / safe_total * 100.0,
        "within_1sd_pct": _rolling_sum(within_1sd) / safe_total * 100.0,
    }


# ---------------------------------------------------------------------------
# MeanReversion
# ---------------------------------------------------------------------------

def mean_reversion(
    prices: np.ndarray,
    window: int,
    lookback_bars: int,
) -> np.ndarray:
    """Rolling mean-reversion oscillation score.

    Parameters
    ----------
    prices : (n_perms, n_bars, n_assets)
    window : int — rolling-mean window for deviation computation
    lookback_bars : int — window for counting sign changes

    Returns
    -------
    scores : (n_perms, n_bars, n_assets) — values in [0, 1]
    """
    n_p, n_b, n_a = prices.shape

    # Compute rolling mean via cumsum trick
    cs = np.nancumsum(prices, axis=1)
    counts = np.nancumsum(np.isfinite(prices).astype(np.float32), axis=1)
    safe_counts = np.maximum(counts, 1.0)
    # For bars < window, use expanding mean; else rolling mean
    roll_mean = np.empty_like(prices)
    for b in range(n_b):
        if b < window:
            roll_mean[:, b, :] = cs[:, b, :] / safe_counts[:, b, :]
        else:
            window_sum = cs[:, b, :] - cs[:, b - window, :]
            window_cnt = counts[:, b, :] - counts[:, b - window, :]
            safe_wc = np.maximum(window_cnt, 1.0)
            roll_mean[:, b, :] = window_sum / safe_wc

    # Compute deviation sign
    dev = prices - roll_mean
    signs = np.sign(dev)
    valid = np.isfinite(dev)
    signs[~valid] = 0.0

    # Sign change at each bar (comparing consecutive bars)
    change = np.zeros_like(prices)
    prev_valid = valid[:, :-1, :]
    cur_valid = valid[:, 1:, :]
    sign_changed = signs[:, 1:, :] != signs[:, :-1, :]
    change[:, 1:, :] = (prev_valid & cur_valid & sign_changed).astype(np.float32)

    valid_f = valid.astype(np.float32)

    # Rolling sums over lookback_bars
    def _rolling_sum(arr: np.ndarray) -> np.ndarray:
        cs2 = np.cumsum(arr, axis=1)
        out = cs2.copy()
        out[:, lookback_bars:, :] = cs2[:, lookback_bars:, :] - cs2[:, :-lookback_bars, :]
        return out

    total_valid = _rolling_sum(valid_f)
    total_changes = _rolling_sum(change)

    denom = np.maximum(total_valid - 1.0, 1.0)
    scores = np.clip(total_changes / denom, 0.0, 1.0)
    return scores


# ---------------------------------------------------------------------------
# VwapSlope  —  rolling OLS slope
# ---------------------------------------------------------------------------

def vwap_slope(
    vwap: np.ndarray,
    lookback: int,
    volume: np.ndarray | None = None,
    mode: str = "scaled",
    value_per_point: float = 1e-4,
    scale: float = 1.0,
) -> np.ndarray:
    """Rolling OLS slope of VWAP.

    Uses vectorized closed-form linear regression over a rolling window.
    Bars with zero volume are excluded when *volume* is provided.

    Parameters
    ----------
    vwap : (n_perms, n_bars, n_assets)
    lookback : int
    volume : optional (n_perms, n_bars, n_assets)
    mode : 'raw', 'scaled', or 'angle'
    value_per_point, scale : transformation parameters

    Returns
    -------
    slopes : (n_perms, n_bars, n_assets)
    """
    n_p, n_b, n_a = vwap.shape

    # Build x-coordinate arrays: simple 0..lookback-1 for each window
    # We use the running-sum approach for OLS:
    #   slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    # where x = bar offset within window, y = vwap value

    slopes = np.zeros((n_p, n_b, n_a), dtype=np.float32)

    # For each bar b, the window is [max(0, b-lookback+1), b].
    # We compute this with a simple loop over bars — still vectorized over
    # (n_perms, n_assets) which is where the speed gain is.
    for b in range(1, n_b):
        start = max(0, b - lookback + 1)
        w = b - start + 1
        if w < 2:
            continue

        y_win = vwap[:, start:b + 1, :]  # (n_p, w, n_a)
        x = np.arange(w, dtype=np.float32)  # (w,)

        if volume is not None:
            v_win = volume[:, start:b + 1, :]
            valid = np.isfinite(y_win) & (v_win > 0)
        else:
            valid = np.isfinite(y_win)

        # Masked regression: set invalid entries to 0 contribution
        y_masked = np.where(valid, y_win, 0.0)
        x_3d = x.reshape(1, -1, 1) * valid.astype(np.float32)
        n_valid = valid.astype(np.float32).sum(axis=1)  # (n_p, n_a)

        sx = x_3d.sum(axis=1)         # (n_p, n_a)
        sy = y_masked.sum(axis=1)     # (n_p, n_a)
        sxy = (x_3d * y_masked).sum(axis=1)
        sxx = (x_3d * x_3d).sum(axis=1)

        denom = n_valid * sxx - sx * sx
        safe_denom = np.where(np.abs(denom) > 1e-15, denom, 1.0)
        raw_slope = np.where(
            np.abs(denom) > 1e-15,
            (n_valid * sxy - sx * sy) / safe_denom,
            0.0,
        )
        # Require at least 2 valid points
        raw_slope = np.where(n_valid >= 2, raw_slope, 0.0)
        slopes[:, b, :] = raw_slope

    # Apply transformation
    if mode == "raw":
        return slopes

    vpp = value_per_point if value_per_point != 0.0 else 1.0
    normalized = slopes / vpp

    if mode == "scaled":
        return normalized * scale
    elif mode == "angle":
        return np.degrees(np.arctan(normalized)) * scale

    return slopes


# ---------------------------------------------------------------------------
# VwapVolumeImbalance
# ---------------------------------------------------------------------------

def vwap_volume_imbalance(
    prices: np.ndarray,
    volume: np.ndarray,
    band_mean: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """Rolling (above-mean volume − below-mean volume) / total × 100.

    Parameters
    ----------
    prices : (n_perms, n_bars, n_assets)
    volume : (n_perms, n_bars, n_assets)
    band_mean : (n_perms, n_bars, n_assets) — expanding mean from sd_bands
    lookback : int

    Returns
    -------
    imbalance_pct : (n_perms, n_bars, n_assets)
    """
    valid = np.isfinite(prices) & np.isfinite(volume) & np.isfinite(band_mean) & (volume > 0)
    above = np.where(valid & (prices > band_mean), volume, 0.0)
    below = np.where(valid & (prices <= band_mean), volume, 0.0)

    def _rolling_sum(arr: np.ndarray) -> np.ndarray:
        cs = np.cumsum(arr, axis=1)
        out = cs.copy()
        out[:, lookback:, :] = cs[:, lookback:, :] - cs[:, :-lookback, :]
        return out

    win_above = _rolling_sum(above)
    win_below = _rolling_sum(below)
    total = win_above + win_below
    safe_total = np.maximum(total, 1.0)
    return (win_above - win_below) / safe_total * 100.0


# ---------------------------------------------------------------------------
# Convenience: compute all indicators in one pass
# ---------------------------------------------------------------------------

def compute_all_indicators(
    close: np.ndarray,
    *,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    open_: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    buy_volume_yes: np.ndarray | None = None,
    buy_volume_no: np.ndarray | None = None,
    sell_volume_yes: np.ndarray | None = None,
    sell_volume_no: np.ndarray | None = None,
    lookback_bars: int = 72,
    mean_reversion_window: int = 5,
    vwap_slope_lookback: int = 30,
    vwap_slope_mode: str = "scaled",
    vwap_slope_value_per_point: float = 1e-4,
    vwap_slope_scale: float = 1.0,
    vwap_volume_imbalance_lookback: int = 30,
) -> dict[str, np.ndarray]:
    """Compute the full indicator stack for all permutations in one pass.

    Parameters
    ----------
    close : (n_perms, n_bars, n_assets)
    ... : other field arrays with same leading dims

    Returns
    -------
    dict mapping indicator name to 3D array (or dict of arrays).
    """
    tp = typical_price(close, high, low)

    # SdBands
    band_mean, band_std = sd_bands_expanding(tp)
    bands = sd_band_levels(band_mean, band_std)

    result: dict[str, np.ndarray] = {
        "sd_bands_mean": band_mean,
        "sd_bands_std": band_std,
    }
    for k, v in bands.items():
        result[f"sd_bands_{k}"] = v

    # Vwap
    if volume is not None:
        vwap_arr = vwap_expanding(tp, volume)
        result["vwap"] = vwap_arr

        # VwapSlope
        result["vwap_slope"] = vwap_slope(
            vwap_arr, vwap_slope_lookback,
            volume=volume,
            mode=vwap_slope_mode,
            value_per_point=vwap_slope_value_per_point,
            scale=vwap_slope_scale,
        )
        result["vwap_slope_raw"] = vwap_slope(
            vwap_arr, vwap_slope_lookback,
            volume=volume,
            mode="raw",
        )

        # VwapVolumeImbalance
        result["vwap_volume_imbalance"] = vwap_volume_imbalance(
            tp, volume, band_mean, vwap_volume_imbalance_lookback,
        )
    else:
        result["vwap"] = close.copy()

    # CumulativeYesNoDelta (buy / sell)
    if buy_volume_yes is not None and buy_volume_no is not None:
        buy_delta = cumulative_yes_no_delta(buy_volume_yes, buy_volume_no)
        result["cum_buy_delta"] = buy_delta
        buy_mean, buy_std = cumulative_delta_stats(buy_delta)
        result["cum_buy_delta_mean"] = buy_mean
        result["cum_buy_delta_std"] = buy_std
    if sell_volume_yes is not None and sell_volume_no is not None:
        sell_delta = cumulative_yes_no_delta(sell_volume_yes, sell_volume_no)
        result["cum_sell_delta"] = sell_delta
        sell_mean, sell_std = cumulative_delta_stats(sell_delta)
        result["cum_sell_delta_mean"] = sell_mean
        result["cum_sell_delta_std"] = sell_std

    # BandPosition
    bp = band_position(
        tp, band_mean, bands["+1sd"], bands["-1sd"], lookback_bars,
    )
    for k, v in bp.items():
        result[f"band_position_{k}"] = v

    # MeanReversion
    result["mean_reversion"] = mean_reversion(
        tp, mean_reversion_window, lookback_bars,
    )

    return result
