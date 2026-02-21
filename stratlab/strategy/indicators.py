"""Pre-computable indicator classes for use with the declarative indicator system.

Strategies declare indicators as a class-level `indicator_defs` list. The
Backtester calls `strategy._compute_indicators()` before each `generate_weights`
call, populating `strategy.indicators` with the results keyed by indicator name.

Example
-------
class MyStrategy(Strategy):
    indicator_defs = [
        VwapSlope(vwap=vwap_df, volume=vol_df, lookback=30),
        VwapVolumeImbalance(vwap=vwap_df, volume=vol_df, lookback=60),
        BandPosition(lookback_hours=6.0),
    ]

    def generate_weights(self, prices, returns, index):
        slope    = self.indicators["vwap_slope"]           # pd.Series (per asset)
        imbalance= self.indicators["vwap_volume_imbalance"] # pd.Series (per asset)
        bands    = self.indicators["band_position"]         # pd.DataFrame (stats x assets)
        ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class Indicator(ABC):
    """Abstract base class for all indicators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique key used in the `indicators` dict."""
        ...

    @abstractmethod
    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        index: int,
    ) -> Any:
        """Compute the indicator value at `index`.

        Args:
            prices:  Full price DataFrame (rows=timestamps, cols=assets).
            returns: Full returns DataFrame aligned to prices.
            index:   Current integer bar index.

        Returns:
            Indicator value — typically a pd.Series (per-asset scalar)
            or pd.DataFrame (per-asset multi-stat).
        """
        ...


# ---------------------------------------------------------------------------
# Helpers shared by multiple indicators
# ---------------------------------------------------------------------------

def _polyfit_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return OLS slope from np.polyfit, or 0.0 on degenerate input."""
    if len(x) < 2:
        return 0.0
    x_rel = x - x[0]
    return float(np.polyfit(x_rel, y, 1)[0])


def _lookback_window_start(
    prices: pd.DataFrame,
    index: int,
    lookback_hours: float | None,
    lookback_bars: int | None,
) -> int:
    """Return the start row index for a lookback window ending at *index*.

    Prefer *lookback_hours* (wall-clock) over *lookback_bars*.
    """
    if lookback_hours is not None:
        current_ts = pd.Timestamp(prices.index[index])
        start_ts = current_ts - pd.Timedelta(hours=float(lookback_hours))
        return int(pd.DatetimeIndex(prices.index).searchsorted(start_ts, side="left"))
    bars = lookback_bars if lookback_bars is not None else 1
    return max(0, index - bars + 1)


def _transform_slope(raw: float, mode: str, value_per_point: float, scale: float) -> float:
    """Convert a raw price-per-bar slope to scaled or angle form."""
    if mode == "raw":
        return float(raw)
    vpp = value_per_point if value_per_point != 0.0 else 1.0
    normalized = float(raw / vpp)
    if mode == "scaled":
        return float(normalized * scale)
    if mode == "angle":
        return float(np.degrees(np.arctan(normalized)) * scale)
    raise ValueError(f"Unsupported vwap_slope_mode={mode!r}")


# ---------------------------------------------------------------------------
# VwapSlope
# ---------------------------------------------------------------------------

class VwapSlope(Indicator):
    """OLS slope of VWAP over a rolling lookback window.

    Returns a ``pd.Series`` indexed by asset name. Assets absent from the
    VWAP DataFrame receive slope 0.0.

    Parameters
    ----------
    vwap:
        DataFrame of VWAP prices aligned to the prices/returns DataFrames
        (same index, one column per asset).
    volume:
        Optional volume DataFrame. When provided, bars with zero volume are
        excluded from the regression (matches Sierra Chart behaviour).
    lookback:
        Number of bars to include in the regression window.
    mode:
        ``"raw"`` — price units per bar.
        ``"scaled"`` — normalized by *value_per_point*, then multiplied by *scale*.
        ``"angle"`` — arctan of the normalized slope in degrees × *scale*.
    value_per_point:
        Normalization divisor applied before scaled/angle conversion.
    scale:
        Multiplier applied after normalization.
    """

    def __init__(
        self,
        vwap: pd.DataFrame,
        volume: pd.DataFrame | None = None,
        lookback: int = 30,
        mode: str = "scaled",
        value_per_point: float = 1e-4,
        scale: float = 1.0,
        name: str = "vwap_slope",
    ) -> None:
        self._vwap = vwap
        self._volume = volume
        self.lookback = int(lookback)
        self.mode = mode
        self.value_per_point = float(value_per_point)
        self.scale = float(scale)
        self._indicator_name = name

    @property
    def name(self) -> str:
        return self._indicator_name

    def _slope_for_asset(self, asset: str, index: int) -> float:
        if asset not in self._vwap.columns:
            return 0.0

        start = max(0, index - self.lookback + 1)
        vwap_slice = self._vwap.iloc[start: index + 1][asset]
        bar_offsets = np.arange(start, index + 1, dtype=float)

        if self._volume is not None and asset in self._volume.columns:
            vol_slice = self._volume.iloc[start: index + 1][asset]
            valid = (vol_slice.fillna(0.0) > 0.0) & vwap_slice.notna()
            y = vwap_slice[valid].to_numpy(dtype=float)
            x = bar_offsets[valid.to_numpy(dtype=bool)]
        else:
            valid = vwap_slice.notna()
            y = vwap_slice[valid].to_numpy(dtype=float)
            x = bar_offsets[valid.to_numpy(dtype=bool)]

        raw = _polyfit_slope(x, y)
        return _transform_slope(raw, self.mode, self.value_per_point, self.scale)

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        index: int,
    ) -> pd.Series:
        assets = prices.columns.tolist()
        slopes = {a: self._slope_for_asset(a, index) for a in assets}
        return pd.Series(slopes, dtype=float)


# ---------------------------------------------------------------------------
# VwapVolumeImbalance
# ---------------------------------------------------------------------------

class VwapVolumeImbalance(Indicator):
    """Rolling (above-VWAP volume − below-VWAP volume) / total × 100.

    Returns a ``pd.Series`` indexed by asset name. Assets with insufficient
    data receive ``float("nan")``.

    Parameters
    ----------
    vwap:
        VWAP DataFrame aligned to prices (same index/columns).
    volume:
        Volume DataFrame aligned to prices.
    lookback:
        Rolling window size in bars for the volume sums.
    """

    def __init__(
        self,
        vwap: pd.DataFrame,
        volume: pd.DataFrame,
        lookback: int = 30,
        name: str = "vwap_volume_imbalance",
    ) -> None:
        self._vwap = vwap
        self._volume = volume
        self.lookback = int(lookback)
        self._indicator_name = name

    @property
    def name(self) -> str:
        return self._indicator_name

    def _imbalance_for_asset(self, prices: pd.DataFrame, asset: str, index: int) -> float:
        if (
            asset not in self._vwap.columns
            or asset not in self._volume.columns
            or asset not in prices.columns
        ):
            return float("nan")

        price_hist = prices.iloc[: index + 1][asset]
        vwap_hist = self._vwap.iloc[: index + 1][asset]
        vol_hist = self._volume.iloc[: index + 1][asset].fillna(0.0)

        valid = vwap_hist.notna() & (vol_hist > 0.0)
        vol_above = vol_hist.where(valid & (price_hist > vwap_hist), 0.0)
        vol_below = vol_hist.where(valid & (price_hist < vwap_hist), 0.0)

        roll_window = max(3, self.lookback)
        ra = vol_above.rolling(window=roll_window, min_periods=1).sum()
        rb = vol_below.rolling(window=roll_window, min_periods=1).sum()
        total = ra + rb
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (ra - rb).divide(total.where(total > 0.0)) * 100.0

        try:
            return float(ratio.iloc[-1])
        except Exception:
            return float("nan")

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        index: int,
    ) -> pd.Series:
        assets = prices.columns.tolist()
        values = {a: self._imbalance_for_asset(prices, a, index) for a in assets}
        return pd.Series(values, dtype=float)


# ---------------------------------------------------------------------------
# BandPosition
# ---------------------------------------------------------------------------

class BandPosition(Indicator):
    """Cumulative band-position statistics over a rolling lookback window.

    For each asset, computes what fraction of bars in the window fell above
    the running mean, above +1σ, below −1σ, or within ±1σ.

    Returns a ``pd.DataFrame`` with asset names as columns and the following
    rows as the index:

    * ``above_mean_pct``
    * ``above_1sd_pct``
    * ``below_minus_1sd_pct``
    * ``within_1sd_pct``

    Parameters
    ----------
    lookback_hours:
        Rolling window expressed in wall-clock hours. Takes precedence over
        *lookback_bars* when set.
    lookback_bars:
        Fallback window in bars when *lookback_hours* is ``None``.
    """

    _STATS = ["above_mean_pct", "above_1sd_pct", "below_minus_1sd_pct", "within_1sd_pct"]

    def __init__(
        self,
        lookback_hours: float | None = None,
        lookback_bars: int | None = None,
        name: str = "band_position",
    ) -> None:
        if lookback_hours is None and lookback_bars is None:
            raise ValueError("Provide at least one of lookback_hours or lookback_bars.")
        self.lookback_hours = lookback_hours
        self.lookback_bars = lookback_bars
        self._indicator_name = name

    @property
    def name(self) -> str:
        return self._indicator_name

    @staticmethod
    def _stats_for_window(arr: np.ndarray) -> dict[str, float]:
        if len(arr) == 0:
            return {k: 0.0 for k in BandPosition._STATS}

        p = arr.astype(float)
        n = len(p)
        idx = np.arange(1, n + 1, dtype=float)
        csum = np.cumsum(p)
        csum_sq = np.cumsum(p * p)
        m = csum / idx
        var = np.maximum(csum_sq / idx - m * m, 0.0)
        s = np.sqrt(var)
        up1 = m + s
        dn1 = m - s

        return {
            "above_mean_pct": float((p > m).sum() / n * 100.0),
            "above_1sd_pct": float((p > up1).sum() / n * 100.0),
            "below_minus_1sd_pct": float((p < dn1).sum() / n * 100.0),
            "within_1sd_pct": float(((p >= dn1) & (p <= up1)).sum() / n * 100.0),
        }

    def _window_start(self, prices: pd.DataFrame, index: int) -> int:
        return _lookback_window_start(prices, index, self.lookback_hours, self.lookback_bars)

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        index: int,
    ) -> pd.DataFrame:
        start = self._window_start(prices, index)
        assets = prices.columns.tolist()
        result: dict[str, dict[str, float]] = {}
        for asset in assets:
            window = prices.iloc[start: index + 1][asset].dropna().to_numpy(dtype=float)
            result[asset] = self._stats_for_window(window)
        return pd.DataFrame(result, index=self._STATS)


# ---------------------------------------------------------------------------
# MeanReversion
# ---------------------------------------------------------------------------

class MeanReversion(Indicator):
    """Per-asset mean-reversion oscillation score over a rolling lookback window.

    Measures how frequently price crosses its rolling mean within the window.
    A score of 1.0 means every consecutive bar-pair involves a sign-change in
    the deviation from the rolling mean (maximum oscillation); 0.0 means no
    crossings at all (persistent trend).

    Returns a ``pd.Series`` indexed by asset name with values in ``[0, 1]``.

    Parameters
    ----------
    window:
        Rolling-mean period used to compute deviations.
    lookback_hours:
        Price window expressed in wall-clock hours. Takes precedence over
        *lookback_bars* when set.
    lookback_bars:
        Price window in bars when *lookback_hours* is ``None``.
    name:
        Key used in ``strategy.indicators``. Defaults to ``"mean_reversion"``.
    """

    def __init__(
        self,
        window: int = 5,
        lookback_hours: float | None = None,
        lookback_bars: int | None = None,
        name: str = "mean_reversion",
    ) -> None:
        if lookback_hours is None and lookback_bars is None:
            raise ValueError("Provide at least one of lookback_hours or lookback_bars.")
        self.window = int(window)
        self.lookback_hours = lookback_hours
        self.lookback_bars = lookback_bars
        self._indicator_name = name

    @property
    def name(self) -> str:
        return self._indicator_name

    @staticmethod
    def _score_for_window(arr: np.ndarray, window: int) -> float:
        if len(arr) < window:
            return 0.0
        roll = pd.Series(arr).rolling(window=window, min_periods=1).mean().to_numpy()
        dev = arr - roll
        valid = dev[~np.isnan(dev)]
        if len(valid) <= 1:
            return 0.0
        changes = int(np.sum(np.diff(np.sign(valid)) != 0))
        return float(np.clip(changes / (len(valid) - 1), 0.0, 1.0))

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        index: int,
    ) -> pd.Series:
        start = _lookback_window_start(prices, index, self.lookback_hours, self.lookback_bars)
        assets = prices.columns.tolist()
        scores = {
            asset: self._score_for_window(
                prices.iloc[start: index + 1][asset].dropna().to_numpy(dtype=float),
                self.window,
            )
            for asset in assets
        }
        return pd.Series(scores, dtype=float)
