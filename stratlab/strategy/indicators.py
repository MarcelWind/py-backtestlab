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

import re
import numpy as np
import pandas as pd


def _compute_price(
    close: float,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None, # open is a reserved keyword, so we use open_ in the signature
    method: str = "typical",
) -> float:
    """Compute price based on pricing method.
    
    Parameters
    ----------
    close : float
        Close price (required for all methods).
    high : float, optional
        High price (required for 'typical', 'median', 'close_weighted', 'ohlc_avg').
    low : float, optional
        Low price (required for 'typical', 'median', 'close_weighted', 'ohlc_avg').
    open_ : float, optional
        Open price (required for 'ohlc_avg').
    method : str
        Pricing method: 'close', 'median', 'typical', 'close_weighted', 'ohlc_avg'.
        Default is 'typical' (H+L+C)/3.
        
    Returns
    -------
    float
        Computed price.
    """
    if method == "close":
        return close
    elif method == "median":
        if high is None or low is None:
            return close
        return (high + low) / 2.0
    elif method == "typical":
        if high is None or low is None:
            return close
        return (high + low + close) / 3.0
    elif method == "close_weighted":
        if high is None or low is None:
            return close
        return (high + low + 2.0 * close) / 4.0
    elif method == "ohlc_avg":
        if high is None or low is None or open_ is None:
            return close
        return (open_ + high + low + close) / 4.0
    else:
        return close


class Indicator(ABC):
    """Abstract base class for all indicators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique key used in the `indicators` dict."""
        ...

    @property
    def plot_panel(self) -> int | None:
        """Subplot panel index for plotting (0 = price panel, None = skip).

        Subclasses set ``self._plot_panel`` to control which panel this
        indicator is drawn in when ``plot_market_indicators`` is called.
        """
        return getattr(self, "_plot_panel", None)

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

    def compute_series(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute indicator values for every bar and return as a DataFrame.

        Default implementation iterates over all bar indices calling
        ``compute()``.  The result must be a ``pd.Series`` at each bar;
        raises ``TypeError`` otherwise (override this method for indicators
        that return non-Series values, e.g. ``BandPosition``).

        Args:
            prices:  Full price DataFrame (rows=timestamps, cols=assets).
            returns: Full returns DataFrame aligned to prices.

        Returns:
            DataFrame of shape ``(n_bars, n_assets)`` with the same index
            as ``prices``.
        """
        result = []
        for i in range(len(prices)):
            val = self.compute(prices, returns, i)
            if not isinstance(val, pd.Series):
                raise TypeError(
                    f"{self.__class__.__name__}.compute() returns "
                    f"{type(val).__name__}, not pd.Series — "
                    "override compute_series() or set plot_panel=None."
                )
            result.append(val)
        return pd.DataFrame(result, index=prices.index)


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
# Yes/No cumulative delta helpers
# ---------------------------------------------------------------------------

# regular expression used to detect __yes/__no suffixes in column names
_SUFFIX_RE = re.compile(r"^(.*)__(yes|no)$", re.IGNORECASE)


def select_yes_no_columns(volume_df: pd.DataFrame | None, base: str, market_str: str, suffix_re=_SUFFIX_RE):
    """Return (no_col, yes_col) column names from volume_df for the given base market name.

    Mirrors the selection logic originally defined in the strategy module so
    indicator code can resolve paired yes/no columns reliably.
    """
    if volume_df is None:
        return None, None
    col_list = [str(c) for c in volume_df.columns]
    cand_no = None
    cand_yes = None
    exact_no = f"{base}__no"
    exact_yes = f"{base}__yes"
    if exact_no in col_list:
        cand_no = exact_no
    if exact_yes in col_list:
        cand_yes = exact_yes
    if cand_no is None:
        for c in col_list:
            if c.lower().startswith(base.lower()) and c.lower().endswith("__no"):
                cand_no = c
                break
    if cand_yes is None:
        for c in col_list:
            if c.lower().startswith(base.lower()) and c.lower().endswith("__yes"):
                cand_yes = c
                break
    if (cand_no is None or cand_yes is None) and base in col_list:
        m_m = suffix_re.match(market_str)
        if m_m:
            suffix = m_m.group(2).lower()
            if suffix == "no":
                cand_no = cand_no or base
            elif suffix == "yes":
                cand_yes = cand_yes or base
    return cand_no, cand_yes


def cumulative_yes_no_delta(
    volume_df: pd.DataFrame | None,
    cache: dict[str, tuple[str | None, str | None]],
    asset: str,
    prices: pd.DataFrame,
    index: int,
    suffix_re=_SUFFIX_RE,
) -> float:
    """Return cumulative yes-minus-no delta for one volume matrix/asset.

    The returned value is the running sum of (yes - no) from the beginning
    of ``prices`` through ``index``. The series is baseline-shifted so the
    first available bar is always zero.
    """
    if volume_df is None or volume_df.empty:
        return float("nan")

    m = suffix_re.match(asset)
    base = m.group(1) if m else asset

    no_col, yes_col = cache.get(asset, (None, None))
    if no_col is None or yes_col is None:
        no_col, yes_col = select_yes_no_columns(volume_df, base, asset, suffix_re)
        cache[asset] = (no_col, yes_col)

    if yes_col is None or no_col is None:
        return float("nan")

    yes_series = volume_df[yes_col].reindex(prices.index).fillna(0.0)
    no_series = volume_df[no_col].reindex(prices.index).fillna(0.0)
    delta_series = yes_series - no_series
    cum = delta_series.cumsum()
    if cum.empty:
        return float("nan")
    baseline = float(cum.iloc[0])
    try:
        return float(cum.iloc[index] - baseline)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Vwap
# ---------------------------------------------------------------------------

class Vwap(Indicator):
    """Incremental VWAP indicator with configurable pricing method.

    Computes VWAP = Σ(price × volume) / Σ(volume) over a rolling or expanding
    window, updating O(1) per bar for the expanding case. Supports multiple
    pricing methods (close, median, typical, close_weighted, ohlc_avg).

    Parameters
    ----------
    volume:
        Volume DataFrame aligned to prices (same index/columns).
    window_bars:
        Rolling window in bars. ``None`` (default) → expanding from bar 0.
    pricing_method : str
        Pricing method: 'close', 'median', 'typical' (default), 'close_weighted', 'ohlc_avg'.
    high : pd.DataFrame, optional
        High prices. Required for methods other than 'close'.
    low : pd.DataFrame, optional
        Low prices. Required for methods other than 'close'.
    open_ : pd.DataFrame, optional
        Open prices. Required for 'ohlc_avg' method.
    """

    def __init__(
        self,
        volume: pd.DataFrame,
        window_bars: int | None = None,
        name: str = "vwap",
        plot_panel: int | None = 0,
        pricing_method: str = "typical",
        high: pd.DataFrame | None = None,
        low: pd.DataFrame | None = None,
        open_: pd.DataFrame | None = None,
    ) -> None:
        self._volume = volume
        self.window_bars = window_bars
        self._indicator_name = name
        self._plot_panel = plot_panel
        self.pricing_method = pricing_method
        self._high = high
        self._low = low
        self._open = open_
        self._values: dict[str, list[float]] = {}
        self._ts_lists: dict[str, list] = {}
        # Expanding: scalar running sums.  Rolling: lists of bar-level (p*v, v).
        self._pv_acc: dict[str, Any] = {}
        self._v_acc: dict[str, Any] = {}
        self._next_bar: int = -1

    @property
    def name(self) -> str:
        return self._indicator_name

    def _process_bar(self, prices: pd.DataFrame, i: int) -> None:
        ts = prices.index[i]
        for asset in prices.columns:
            try:
                close = float(prices.iloc[i][asset])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(close):
                continue
            if asset not in self._volume.columns:
                continue
            
            # Extract OHLC for pricing method
            high = None
            low = None
            open_ = None
            if self._high is not None and asset in self._high.columns:
                try:
                    h = float(self._high.iloc[i][asset])
                    if np.isfinite(h):
                        high = h
                except (TypeError, ValueError):
                    pass
            if self._low is not None and asset in self._low.columns:
                try:
                    l = float(self._low.iloc[i][asset])
                    if np.isfinite(l):
                        low = l
                except (TypeError, ValueError):
                    pass
            if self._open is not None and asset in self._open.columns:
                try:
                    o = float(self._open.iloc[i][asset])
                    if np.isfinite(o):
                        open_ = o
                except (TypeError, ValueError):
                    pass
            
            # Compute price based on method
            p = _compute_price(close, high, low, open_, self.pricing_method)
            
            try:
                v = float(self._volume.iloc[i][asset])
            except (TypeError, ValueError):
                v = 0.0
            if not np.isfinite(v):
                v = 0.0

            if self.window_bars is None:
                # Expanding: O(1) running sums
                pv_sum = float(self._pv_acc.get(asset, 0.0)) + p * v
                v_sum  = float(self._v_acc.get(asset, 0.0)) + v
                self._pv_acc[asset] = pv_sum
                self._v_acc[asset]  = v_sum
                vwap_val = pv_sum / v_sum if v_sum > 0.0 else p
            else:
                # Rolling: keep a list of the last window_bars bar-level sums
                pv_hist = self._pv_acc.setdefault(asset, [])
                v_hist  = self._v_acc.setdefault(asset, [])
                pv_hist.append(p * v)  # type: ignore[union-attr]
                v_hist.append(v)       # type: ignore[union-attr]
                w = self.window_bars
                total_v = sum(v_hist[-w:])  # type: ignore[arg-type]
                vwap_val = sum(pv_hist[-w:]) / total_v if total_v > 0.0 else p  # type: ignore[arg-type]

            self._values.setdefault(asset, []).append(vwap_val)
            self._ts_lists.setdefault(asset, []).append(ts)

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        index: int,
    ) -> pd.Series:
        """Return current bar's VWAP per asset as a ``pd.Series``.

        On first call processes all bars from 0 up to *index* (catch-up pass)
        so that ``vwap_slice`` has full history from the start of the series.
        """
        if self._next_bar < 0:
            self._next_bar = 0
        for i in range(self._next_bar, index + 1):
            self._process_bar(prices, i)
        self._next_bar = index + 1
        result = {a: self._values[a][-1] for a in prices.columns if a in self._values}
        return pd.Series(result, dtype=float)

    def vwap_slice(self, asset: str, from_ts, to_ts) -> "pd.Series | None":
        """Return VWAP history for *asset* between *from_ts* and *to_ts* inclusive.

        Uses binary search for O(log n + w) lookup.
        Returns ``None`` if no data is available for the requested range.
        """
        import bisect
        ts_list = self._ts_lists.get(asset)
        vals = self._values.get(asset)
        if not ts_list or not vals:
            return None
        lo = bisect.bisect_left(ts_list, from_ts)
        hi = bisect.bisect_right(ts_list, to_ts)
        if lo >= hi:
            return None
        return pd.Series(vals[lo:hi], index=ts_list[lo:hi], dtype=float)


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
        (same index, one column per asset).  Ignored when *vwap_indicator*
        is provided.
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
    vwap_indicator:
        ``Vwap`` indicator instance.  When set, VWAP values are sourced from
        this indicator's accumulated history (takes precedence over *vwap*).
    """

    def __init__(
        self,
        vwap: pd.DataFrame | None = None,
        volume: pd.DataFrame | None = None,
        lookback: int = 30,
        mode: str = "scaled",
        value_per_point: float = 1e-4,
        scale: float = 1.0,
        name: str = "vwap_slope",
        plot_panel: int | None = 2,
        vwap_indicator: "Vwap | None" = None,
    ) -> None:
        self._vwap = vwap if vwap is not None else pd.DataFrame()
        self._volume = volume
        self.vwap_indicator = vwap_indicator
        self.lookback = int(lookback)
        self.mode = mode
        self.value_per_point = float(value_per_point)
        self.scale = float(scale)
        self._indicator_name = name
        self._plot_panel = plot_panel

    @property
    def name(self) -> str:
        return self._indicator_name

    def _slope_for_asset(self, prices: pd.DataFrame, asset: str, index: int) -> float:
        start = max(0, index - self.lookback + 1)

        if self.vwap_indicator is not None:
            from_ts = prices.index[start]
            to_ts   = prices.index[index]
            vwap_ser = self.vwap_indicator.vwap_slice(asset, from_ts, to_ts)
            if vwap_ser is None or len(vwap_ser) < 2:
                return 0.0
            if self._volume is not None and asset in self._volume.columns:
                vol_slice = self._volume.loc[vwap_ser.index, asset].fillna(0.0)
                valid = (vol_slice > 0.0) & vwap_ser.notna()
                vwap_ser = vwap_ser[valid]
            if len(vwap_ser) < 2:
                return 0.0
            x = np.arange(len(vwap_ser), dtype=float)
            y = vwap_ser.to_numpy(dtype=float)
        else:
            if asset not in self._vwap.columns:
                return 0.0
            vwap_slice = self._vwap.iloc[start: index + 1][asset]
            # bar_offsets must match vwap_slice length (vwap may be shorter than prices)
            bar_offsets = np.arange(start, start + len(vwap_slice), dtype=float)
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
        slopes = {a: self._slope_for_asset(prices, a, index) for a in assets}
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
        VWAP DataFrame aligned to prices (same index/columns).  Ignored when
        *vwap_indicator* is provided.
    volume:
        Volume DataFrame aligned to prices.
    lookback:
        Rolling window size in bars for the volume sums.
    vwap_indicator:
        ``Vwap`` indicator instance.  When set, VWAP values are sourced from
        this indicator's accumulated history (takes precedence over *vwap*).
    """

    def __init__(
        self,
        vwap: pd.DataFrame | None = None,
        volume: pd.DataFrame = pd.DataFrame(),
        lookback: int = 30,
        name: str = "vwap_volume_imbalance",
        plot_panel: int | None = 1,
        vwap_indicator: "Vwap | None" = None,
    ) -> None:
        self._vwap = vwap if vwap is not None else pd.DataFrame()
        self._volume = volume
        self.vwap_indicator = vwap_indicator
        self.lookback = int(lookback)
        self._indicator_name = name
        self._plot_panel = plot_panel

    @property
    def name(self) -> str:
        return self._indicator_name

    def _imbalance_for_asset(self, prices: pd.DataFrame, asset: str, index: int) -> float:
        if asset not in self._volume.columns or asset not in prices.columns:
            return float("nan")

        price_hist = prices.iloc[: index + 1][asset]
        vol_hist = self._volume.iloc[: index + 1][asset].fillna(0.0)

        if self.vwap_indicator is not None:
            from_ts = prices.index[0]
            to_ts   = prices.index[index]
            vwap_ser = self.vwap_indicator.vwap_slice(asset, from_ts, to_ts)
            if vwap_ser is None:
                return float("nan")
            vwap_hist = vwap_ser.reindex(price_hist.index)
        else:
            if asset not in self._vwap.columns:
                return float("nan")
            vwap_hist = self._vwap.iloc[: index + 1][asset]

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
        plot_panel: int | None = None,
        sd_bands: "SdBands | None" = None,
    ) -> None:
        if lookback_hours is None and lookback_bars is None:
            raise ValueError("Provide at least one of lookback_hours or lookback_bars.")
        self.lookback_hours = lookback_hours
        self.lookback_bars = lookback_bars
        self._indicator_name = name
        self._plot_panel = plot_panel
        self.sd_bands = sd_bands

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

    @staticmethod
    def _stats_with_bands(
        prices_arr: np.ndarray,
        mean_vals: np.ndarray,
        up1_vals: np.ndarray,
        dn1_vals: np.ndarray,
    ) -> dict[str, float]:
        valid = np.isfinite(mean_vals) & np.isfinite(up1_vals) & np.isfinite(dn1_vals)
        p = prices_arr[valid]
        m = mean_vals[valid]
        up1 = up1_vals[valid]
        dn1 = dn1_vals[valid]
        n = len(p)
        if n == 0:
            return {k: 0.0 for k in BandPosition._STATS}
        return {
            "above_mean_pct":      float((p > m).sum()  / n * 100.0),
            "above_1sd_pct":       float((p > up1).sum() / n * 100.0),
            "below_minus_1sd_pct": float((p < dn1).sum() / n * 100.0),
            "within_1sd_pct":      float(((p >= dn1) & (p <= up1)).sum() / n * 100.0),
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
        result: dict[str, dict[str, float]] = {}
        for asset in prices.columns.tolist():
            window_ser = prices.iloc[start: index + 1][asset].dropna()
            if self.sd_bands is not None and len(window_ser) > 0:
                bands_df = self.sd_bands.band_slice(
                    asset, window_ser.index[0], window_ser.index[-1]
                )
                if bands_df is not None and len(bands_df) > 0:
                    aligned = bands_df.reindex(window_ser.index)
                    result[asset] = self._stats_with_bands(
                        window_ser.to_numpy(dtype=float),
                        aligned["mean"].to_numpy(dtype=float),
                        aligned["+1sd"].to_numpy(dtype=float),
                        aligned["-1sd"].to_numpy(dtype=float),
                    )
                    continue
            result[asset] = self._stats_for_window(window_ser.to_numpy(dtype=float))
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
        plot_panel: int | None = 3,
    ) -> None:
        if lookback_hours is None and lookback_bars is None:
            raise ValueError("Provide at least one of lookback_hours or lookback_bars.")
        self.window = int(window)
        self.lookback_hours = lookback_hours
        self.lookback_bars = lookback_bars
        self._indicator_name = name
        self._plot_panel = plot_panel

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

# ---------------------------------------------------------------------------
# Cumulative Band Analysis (Market Regime Analysis)
# ---------------------------------------------------------------------------

def sd_bands_expanding(
    prices_series: np.ndarray | pd.Series,
    timestamps: np.ndarray | pd.Index | None = None,
) -> pd.DataFrame:
    """
    Compute cumulative expanding standard deviation bands for a single price series.

    For each bar, computes mean and std of all historical prices up to that bar
    (expanding / full-history). Returns a DataFrame with columns: mean,
    +1sd, -1sd, +2sd, -2sd, +3sd, -3sd.

    Parameters
    ----------
    prices_series:
        1D array of prices.
    timestamps:
        Optional timestamps for the index. If not provided, integer index is used.

    Returns:
        DataFrame with bands (one row per price point).
    """
    if isinstance(prices_series, pd.Series):
        prices = prices_series.to_numpy(dtype=float)
        idx = prices_series.index if timestamps is None else timestamps
    else:
        prices = np.asarray(prices_series, dtype=float)
        idx = timestamps if timestamps is not None else np.arange(len(prices))

    rows = []
    for i in range(len(prices)):
        hist = prices[: i + 1]
        m = float(np.nanmean(hist))
        s = float(np.nanstd(hist, ddof=0))
        rows.append({
            "mean": m,
            "+1sd": m + s,
            "-1sd": m - s,
            "+2sd": m + 2 * s,
            "-2sd": m - 2 * s,
            "+3sd": m + 3 * s,
            "-3sd": m - 3 * s,
        })
    return pd.DataFrame(rows, index=idx)


class SdBands(Indicator):
    """Expanding mean/std band indicator, computed incrementally per bar.

    Maintains running sum/sum_sq/count per asset for O(1) per-bar updates
    (vs. O(n) per bar in the standalone ``sd_bands_expanding`` function).
    Accumulates the full band history accessible via ``band_series``.
    
    Parameters
    ----------
    pricing_method : str
        Pricing method for band computation: 'close', 'median', 'typical' (default),
        'close_weighted', 'ohlc_avg'.
    high : pd.DataFrame, optional
        High prices. Required for methods other than 'close'.
    low : pd.DataFrame, optional
        Low prices. Required for methods other than 'close'.
    open_ : pd.DataFrame, optional
        Open prices. Required for 'ohlc_avg' method.
    """

    @property
    def name(self) -> str:
        return "sd_bands"

    def __init__(
        self,
        pricing_method: str = "typical",
        high: pd.DataFrame | None = None,
        low: pd.DataFrame | None = None,
        open_: pd.DataFrame | None = None,
    ) -> None:
        self.pricing_method = pricing_method
        self._high = high
        self._low = low
        self._open = open_
        self._sums: dict[str, float] = {}
        self._sum_sqs: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._history: dict[str, list[tuple]] = {}
        self._ts_lists: dict[str, list] = {}
        self._next_bar: int = -1

    def _process_bar(self, prices: pd.DataFrame, i: int) -> None:
        ts = prices.index[i]
        for asset in prices.columns:
            try:
                close = float(prices.iloc[i][asset])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(close):
                continue
            
            # Extract OHLC for pricing method
            high = None
            low = None
            open_ = None
            if self._high is not None and asset in self._high.columns:
                try:
                    h = float(self._high.iloc[i][asset])
                    if np.isfinite(h):
                        high = h
                except (TypeError, ValueError):
                    pass
            if self._low is not None and asset in self._low.columns:
                try:
                    l = float(self._low.iloc[i][asset])
                    if np.isfinite(l):
                        low = l
                except (TypeError, ValueError):
                    pass
            if self._open is not None and asset in self._open.columns:
                try:
                    o = float(self._open.iloc[i][asset])
                    if np.isfinite(o):
                        open_ = o
                except (TypeError, ValueError):
                    pass
            
            # Compute price based on method
            price = _compute_price(close, high, low, open_, self.pricing_method)
            
            self._sums[asset] = self._sums.get(asset, 0.0) + price
            self._sum_sqs[asset] = self._sum_sqs.get(asset, 0.0) + price * price
            self._counts[asset] = self._counts.get(asset, 0) + 1
            n = self._counts[asset]
            mean = self._sums[asset] / n
            var = max(self._sum_sqs[asset] / n - mean ** 2, 0.0)
            s = float(np.sqrt(var))
            band_s = pd.Series({
                "mean": mean,
                "+1sd": mean + s,   "-1sd": mean - s,
                "+2sd": mean + 2*s, "-2sd": mean - 2*s,
                "+3sd": mean + 3*s, "-3sd": mean - 3*s,
            })
            self._history.setdefault(asset, []).append((ts, band_s))
            self._ts_lists.setdefault(asset, []).append(ts)

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        index: int,
    ) -> pd.DataFrame:
        """Return current band values as DataFrame with shape (7, n_assets).

        Index rows: ``["mean", "+1sd", "-1sd", "+2sd", "-2sd", "+3sd", "-3sd"]``
        Columns: asset names

        On first call, processes all bars from 0 up to *index* (catch-up pass)
        so that ``band_slice`` has full history from the start of the price series.
        Subsequent calls process exactly one bar per invocation.
        """
        if self._next_bar < 0:
            self._next_bar = 0
        for i in range(self._next_bar, index + 1):
            self._process_bar(prices, i)
        self._next_bar = index + 1
        result = {a: self._history[a][-1][1] for a in prices.columns if a in self._history}
        return pd.DataFrame(result)

    def band_slice(self, asset: str, from_ts, to_ts) -> "pd.DataFrame | None":
        """Return band history for *asset* between *from_ts* and *to_ts* inclusive.

        Uses binary search for O(log n + w) lookup.
        Returns ``None`` if no data is available for the requested range.
        """
        import bisect
        ts_list = self._ts_lists.get(asset)
        hist = self._history.get(asset)
        if not ts_list or not hist:
            return None
        lo = bisect.bisect_left(ts_list, from_ts)
        hi = bisect.bisect_right(ts_list, to_ts)
        if lo >= hi:
            return None
        rows = hist[lo:hi]
        return pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])

    @property
    def band_series(self) -> dict[str, pd.DataFrame]:
        """Full band history keyed by asset name.

        Each value is a DataFrame of shape ``(n_bars, 7)`` indexed by timestamp,
        with columns ``["mean", "+1sd", "-1sd", "+2sd", "-2sd", "+3sd", "-3sd"]``.
        """
        return {
            asset: pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])
            for asset, rows in self._history.items()
            if rows
        }


class VwapBands(Indicator):
    """Expanding mean/std band indicator computed from VWAP values.

    Similar to SdBands, but computes bands using a referenced Vwap indicator's
    output values instead of close prices. Maintains running sum/sum_sq/count
    per asset for O(1) per-bar updates.
    
    Note: VwapBands should be added to indicator_defs AFTER the Vwap indicator
    it references, to ensure Vwap values are computed first.
    
    Parameters
    ----------
    vwap_indicator : Vwap
        Reference to a Vwap indicator instance. Bands are computed from its values.
    """

    @property
    def name(self) -> str:
        return "vwap_bands"

    def __init__(self, vwap_indicator: Vwap) -> None:
        self._vwap_indicator = vwap_indicator
        self._sums: dict[str, float] = {}
        self._sum_sqs: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._history: dict[str, list[tuple]] = {}
        self._ts_lists: dict[str, list] = {}
        self._next_bar: int = -1

    def _process_bar(self, prices: pd.DataFrame, i: int, vwap_values: pd.Series) -> None:
        """Process one bar given VWAP values for that bar."""
        ts = prices.index[i]
        
        for asset in prices.columns:
            if asset not in vwap_values.index:
                continue
            try:
                vwap_val = float(vwap_values.loc[asset])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(vwap_val):
                continue
            
            self._sums[asset] = self._sums.get(asset, 0.0) + vwap_val
            self._sum_sqs[asset] = self._sum_sqs.get(asset, 0.0) + vwap_val * vwap_val
            self._counts[asset] = self._counts.get(asset, 0) + 1
            n = self._counts[asset]
            mean = self._sums[asset] / n
            var = max(self._sum_sqs[asset] / n - mean ** 2, 0.0)
            s = float(np.sqrt(var))
            band_s = pd.Series({
                "mean": mean,
                "+1sd": mean + s,   "-1sd": mean - s,
                "+2sd": mean + 2*s, "-2sd": mean - 2*s,
                "+3sd": mean + 3*s, "-3sd": mean - 3*s,
            })
            self._history.setdefault(asset, []).append((ts, band_s))
            self._ts_lists.setdefault(asset, []).append(ts)

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        index: int,
    ) -> pd.DataFrame:
        """Return current band values as DataFrame with shape (7, n_assets).

        Index rows: ``["mean", "+1sd", "-1sd", "+2sd", "-2sd", "+3sd", "-3sd"]``
        Columns: asset names

        On first call, processes all bars from 0 up to *index* (catch-up pass)
        so that ``band_slice`` has full history from the start of the price series.
        Subsequent calls process exactly one bar per invocation.
        """
        if self._next_bar < 0:
            self._next_bar = 0
        for i in range(self._next_bar, index + 1):
            # Get VWAP values for this bar from the referenced indicator
            vwap_result = self._vwap_indicator.compute(prices, returns, i)
            self._process_bar(prices, i, vwap_result)
        self._next_bar = index + 1
        result = {a: self._history[a][-1][1] for a in prices.columns if a in self._history}
        return pd.DataFrame(result)

    def band_slice(self, asset: str, from_ts, to_ts) -> "pd.DataFrame | None":
        """Return band history for *asset* between *from_ts* and *to_ts* inclusive.

        Uses binary search for O(log n + w) lookup.
        Returns ``None`` if no data is available for the requested range.
        """
        import bisect
        ts_list = self._ts_lists.get(asset)
        hist = self._history.get(asset)
        if not ts_list or not hist:
            return None
        lo = bisect.bisect_left(ts_list, from_ts)
        hi = bisect.bisect_right(ts_list, to_ts)
        if lo >= hi:
            return None
        rows = hist[lo:hi]
        return pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])

    @property
    def band_series(self) -> dict[str, pd.DataFrame]:
        """Full band history keyed by asset name.

        Each value is a DataFrame of shape ``(n_bars, 7)`` indexed by timestamp,
        with columns ``["mean", "+1sd", "-1sd", "+2sd", "-2sd", "+3sd", "-3sd"]``.
        """
        return {
            asset: pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])
            for asset, rows in self._history.items()
            if rows
        }


def market_regimes(
    band_position_stats: dict[str, float],
    mean_reversion_score: float,
    confidence_boost: float = 0.5,
) -> tuple[str, float]:
    """
    Classify market regime based on band position and mean reversion metrics.

    Uses priority rules:
    1. Strong imbalance up (above mean >60%, above 1sd >40%)
    2. Strong imbalance down (below -1sd >40%, above mean <40%)
    3. High mean reversion (score >=0.5)
    4. Balanced (within ±1sd >=70%)
    5. Rotational (default)

    Parameters
    ----------
    band_position_stats:
        Dict with keys: "above_mean_pct", "above_1sd_pct", "below_minus_1sd_pct", "within_1sd_pct"
    mean_reversion_score:
        Oscillation score in [0, 1], where 1 = maximum mean reversion.
    confidence_boost:
        Scaling factor for confidence scores (0-1).

    Returns:
        Tuple of (regime_label, confidence_score).
    """
    above_mean = band_position_stats.get("above_mean_pct", 0.0)
    above_1sd = band_position_stats.get("above_1sd_pct", 0.0)
    below_1sd = band_position_stats.get("below_minus_1sd_pct", 0.0)
    within = band_position_stats.get("within_1sd_pct", 0.0)

    # Priority rules
    if above_mean > 60 and above_1sd > 40:
        conf = min(1.0, (above_mean - 50) / 50 + above_1sd / 100.0)
        return "Imb. Up", conf
    if below_1sd > 40 and above_mean < 40:
        conf = min(1.0, (40 - above_mean) / 50 + below_1sd / 100.0)
        return "Imb. Down", conf
    if mean_reversion_score >= 0.5:
        return "Mean-Reverting", mean_reversion_score
    if within >= 70:
        return "Balanced", within / 100.0
    return "Rotational", 0.5


# ---------------------------------------------------------------------------
# Window-based Band Analysis (comparing windows against full-history reference)
# ---------------------------------------------------------------------------

def analyze_band_position_vs_reference(
    window_prices: np.ndarray | pd.Series,
    full_bands: pd.DataFrame,
    window_timestamps: np.ndarray | pd.Index | None = None,
) -> dict[str, float]:
    """
    Compute band position stats for a window against full-history band reference.
    
    Aligns window timestamps to full-history bands (nearest previous) and computes
    what % of window prices fall above/below the reference bands.
    
    Parameters
    ----------
    window_prices:
        Price series for the window (must have timestamps as index or separate arg).
    full_bands:
        Full-history bands DataFrame from sd_bands_rolling() with band statistics.
        Can have 'timestamp' column or timestamps in index.
    window_timestamps:
        Optional explicit timestamps for window. If None, uses window_prices.index.
    
    Returns:
        Dict with keys: above_mean_pct, above_1sd_pct, below_minus_1sd_pct, within_1sd_pct
    """
    if isinstance(window_prices, pd.Series):
        if window_timestamps is None:
            window_timestamps = window_prices.index
        prices_arr = window_prices.values
    else:
        prices_arr = np.asarray(window_prices, dtype=float)
    
    if len(prices_arr) == 0 or full_bands.empty:
        return {
            "above_mean_pct": 0.0,
            "above_1sd_pct": 0.0,
            "below_minus_1sd_pct": 0.0,
            "within_1sd_pct": 0.0,
        }
    
    # Build window DataFrame
    window_df = pd.DataFrame({
        "timestamp": window_timestamps,
        "price": prices_arr,
    })
    
    # Prepare reference bands - convert index to column if needed
    ref = full_bands.copy()
    if "timestamp" not in ref.columns:
        ref = ref.reset_index()
        if ref.columns[0] != "timestamp":
            ref = ref.rename(columns={ref.columns[0]: "timestamp"})
    
    # Ensure we have the needed columns
    ref = ref[["timestamp", "mean", "+1sd", "-1sd"]].sort_values("timestamp")
    window_df = window_df.sort_values("timestamp")
    
    # Align using nearest previous
    aligned = pd.merge_asof(window_df, ref, on="timestamp", direction="backward")
    aligned = aligned.dropna(subset=["price", "mean", "+1sd", "-1sd"])
    
    if aligned.empty:
        return {
            "above_mean_pct": 0.0,
            "above_1sd_pct": 0.0,
            "below_minus_1sd_pct": 0.0,
            "within_1sd_pct": 0.0,
        }
    
    p = np.asarray(aligned["price"].values, dtype=float)
    m = np.asarray(aligned["mean"].values, dtype=float)
    up1 = np.asarray(aligned["+1sd"].values, dtype=float)
    dn1 = np.asarray(aligned["-1sd"].values, dtype=float)
    n = len(p)
    
    return {
        "above_mean_pct": float((p > m).sum() / n * 100.0),
        "above_1sd_pct": float((p > up1).sum() / n * 100.0),
        "below_minus_1sd_pct": float((p < dn1).sum() / n * 100.0),
        "within_1sd_pct": float(((p >= dn1) & (p <= up1)).sum() / n * 100.0),
    }


def detect_mean_reversion_vs_reference(
    window_prices: np.ndarray | pd.Series,
    full_bands: pd.DataFrame,
    window_timestamps: np.ndarray | pd.Index | None = None,
    window: int = 5,
) -> float:
    """
    Compute mean-reversion score for window using deviations from full-history mean.
    
    Aligns window timestamps to full-history bands and measures oscillation
    around the full-history rolling mean.
    
    Parameters
    ----------
    window_prices:
        Price series for the window.
    full_bands:
        Full-history bands DataFrame with band statistics.
        Can have 'timestamp' column or timestamps in index.
    window_timestamps:
        Optional explicit timestamps for window.
    window:
        Rolling window for mean calculation (not currently used, kept for API).
    
    Returns:
        Score in [0, 1] where 1 = maximum mean reversion.
    """
    if isinstance(window_prices, pd.Series):
        if window_timestamps is None:
            window_timestamps = window_prices.index
        prices_arr = window_prices.values
    else:
        prices_arr = np.asarray(window_prices, dtype=float)
    
    if len(prices_arr) == 0 or full_bands.empty:
        return 0.0
    
    window_df = pd.DataFrame({
        "timestamp": window_timestamps,
        "price": prices_arr,
    })
    
    # Prepare reference bands - convert index to column if needed
    ref = full_bands.copy()
    if "timestamp" not in ref.columns:
        ref = ref.reset_index()
        if ref.columns[0] != "timestamp":
            ref = ref.rename(columns={ref.columns[0]: "timestamp"})
    
    ref = ref[["timestamp", "mean"]].sort_values("timestamp")
    window_df = window_df.sort_values("timestamp")
    
    aligned = pd.merge_asof(window_df, ref, on="timestamp", direction="backward").dropna(
        subset=["price", "mean"]
    )
    
    if len(aligned) < 2:
        return 0.0
    
    dev = np.asarray(aligned["price"].values, dtype=float) - np.asarray(aligned["mean"].values, dtype=float)
    valid = dev[~np.isnan(dev)]
    if len(valid) <= 1:
        return 0.0
    changes = np.sum(np.diff(np.sign(valid)) != 0)
    return float(np.clip(changes / (len(valid) - 1), 0.0, 1.0))


class RegimeClassification(Indicator):
    """
    Classify market regime based on cumulative band position and mean reversion.

    Returns a numeric score (0-1) representing regime strength/confidence.
    Regimes are classified priority-based:
    - Imbalance Up/Down: High imbalance + low mean reversion
    - Mean-Reverting: High oscillation score
    - Balanced: Within ±1σ > 70%
    - Rotational: Default, moderate confidence

    Parameters
    ----------
    band_position_threshold:
        Pct threshold for classifying imbalance (default 60% above/below mean).
    mean_reversion_threshold:
        Score threshold for detecting mean reversion (default 0.5).
    use_cumulative:
        If True, compute band position cumulatively (all history).
        If False, use rolling window over lookback_bars.
    lookback_bars:
        Rolling window size when use_cumulative=False.
    """

    def __init__(
        self,
        band_position_threshold: float = 0.6,
        mean_reversion_threshold: float = 0.5,
        use_cumulative: bool = True,
        lookback_bars: int = 100,
        name: str = "regime_classification",
        plot_panel: int | None = None,
    ) -> None:
        self.band_position_threshold = float(band_position_threshold)
        self.mean_reversion_threshold = float(mean_reversion_threshold)
        self.use_cumulative = bool(use_cumulative)
        self.lookback_bars = int(lookback_bars)
        self._indicator_name = name
        self._plot_panel = plot_panel

    @property
    def name(self) -> str:
        return self._indicator_name

    def _compute_band_position(
        self,
        prices: pd.Series,
        bands: pd.DataFrame,
        index: int,
    ) -> dict[str, float]:
        """Compute band position stats against cumulative or rolling bands."""
        p_slice = prices.iloc[: index + 1]
        bands_slice = bands.iloc[: index + 1]

        if self.use_cumulative:
            # Use all historical data
            p = p_slice.to_numpy(dtype=float)
            m = bands_slice["mean"].to_numpy(dtype=float)
            up1 = bands_slice["+1sd"].to_numpy(dtype=float)
            dn1 = bands_slice["-1sd"].to_numpy(dtype=float)
        else:
            # Use rolling window
            start = max(0, index - self.lookback_bars + 1)
            p = p_slice.iloc[start:].to_numpy(dtype=float)
            m = bands_slice.iloc[start:]["mean"].to_numpy(dtype=float)
            up1 = bands_slice.iloc[start:]["+1sd"].to_numpy(dtype=float)
            dn1 = bands_slice.iloc[start:]["-1sd"].to_numpy(dtype=float)

        n = len(p)
        if n < 2:
            return {
                "above_mean_pct": 0.0,
                "above_1sd_pct": 0.0,
                "below_minus_1sd_pct": 0.0,
                "within_1sd_pct": 0.0,
            }

        return {
            "above_mean_pct": float((p > m).sum() / n * 100.0),
            "above_1sd_pct": float((p > up1).sum() / n * 100.0),
            "below_minus_1sd_pct": float((p < dn1).sum() / n * 100.0),
            "within_1sd_pct": float(((p >= dn1) & (p <= up1)).sum() / n * 100.0),
        }

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        index: int,
    ) -> pd.Series:
        """
        Compute regime confidence scores for each asset.

        Returns a pd.Series indexed by asset name with confidence values [0, 1].
        """
        assets = prices.columns.tolist()
        scores = {}

        for asset in assets:
            # Compute bands for this asset (expanding / full-history bands)
            bands = sd_bands_expanding(prices[asset])

            # Compute band position stats
            band_stats = self._compute_band_position(
                prices[asset],
                bands,
                index,
            )

            # Compute mean reversion for this asset
            hist = prices[asset].iloc[: index + 1].dropna().to_numpy(dtype=float)
            if self.use_cumulative:
                window = max(2, len(hist) // 5)
            else:
                window = max(2, self.lookback_bars // 5)

            mr_score = MeanReversion._score_for_window(hist, window)

            # Classify regime and extract confidence
            regime, confidence = market_regimes(band_stats, mr_score)
            scores[asset] = confidence

        return pd.Series(scores, dtype=float)


# ---------------------------------------------------------------------------
# Volume Analysis Utilities
# ---------------------------------------------------------------------------

def volume_profile(
    price_series: "pd.Series | np.ndarray",
    volume_series: "pd.Series | np.ndarray | None" = None,
    bins: int = 24,
) -> "tuple[np.ndarray, np.ndarray]":
    """Compute a volume-weighted histogram of prices.

    Parameters
    ----------
    price_series:
        Price values (pd.Series or np.ndarray).
    volume_series:
        Volume values aligned to prices. If None, uniform weights are used.
    bins:
        Number of histogram bins (default 24).

    Returns
    -------
    tuple of (bin_centers, histogram) where:
    - bin_centers: centre price of each bin
    - histogram: volume-weighted count in each bin
    """
    p = (price_series.to_numpy(dtype=float)
         if isinstance(price_series, pd.Series)
         else np.asarray(price_series, dtype=float))

    if volume_series is None:
        w = np.ones_like(p)
    else:
        w = (volume_series.to_numpy(dtype=float)
             if isinstance(volume_series, pd.Series)
             else np.asarray(volume_series, dtype=float))

    valid = (~np.isnan(p)) & (~np.isnan(w)) & (w > 0.0)
    if valid.sum() < 2:
        return np.array([]), np.array([])

    p_valid = p[valid]
    w_valid = w[valid]
    pmin = float(np.min(p_valid))
    pmax = float(np.max(p_valid))

    if pmin == pmax:
        edges = np.array([pmin - 1e-6, pmax + 1e-6], dtype=float)
    else:
        edges = np.linspace(pmin, pmax, bins + 1)

    hist, _ = np.histogram(p_valid, bins=edges, weights=w_valid)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, hist.astype(float)



class CumulativeVolumeDelta(Indicator):
    """Per-asset cumulative signed-volume delta indicator.

    This indicator accepts a `volume` DataFrame of signed volumes (positive
    for net buys, negative for net sells) and computes the cumulative sum per
    asset incrementally. The indicator returns a ``pd.Series`` of the current
    cumulative delta for each asset when ``compute`` is called.

    Parameters
    ----------
    volume:
        Volume DataFrame aligned to prices (rows=timestamps, cols=assets). If
        an asset is missing from `volume`, it will not appear in the output
        (or will yield NaN if absent entirely).
    name:
        Indicator name key stored in the strategy's `indicators` map.
    plot_panel:
        Optional plotting panel index.
    """

    def __init__(self, volume: pd.DataFrame, name: str = "cum_vol_delta", plot_panel: int | None = None) -> None:
        self._volume = volume if volume is not None else pd.DataFrame()
        self._indicator_name = name
        self._plot_panel = plot_panel
        self._history: dict[str, list[float]] = {}
        self._ts_lists: dict[str, list] = {}
        self._next_bar: int = -1

    @property
    def name(self) -> str:
        return self._indicator_name

    def _process_bar(self, prices: pd.DataFrame, i: int) -> None:
        ts = prices.index[i]
        for asset in prices.columns:
            if asset not in self._volume.columns:
                # treat missing asset in volume as no data
                continue
            try:
                v = float(self._volume.iloc[i][asset])
            except (TypeError, ValueError, IndexError):
                v = 0.0
            if not np.isfinite(v):
                v = 0.0
            prev = self._history.get(asset, [0.0])[-1] if self._history.get(asset) else 0.0
            cur = prev + v
            self._history.setdefault(asset, []).append(cur)
            self._ts_lists.setdefault(asset, []).append(ts)

    def compute(self, prices: pd.DataFrame, returns: pd.DataFrame, index: int) -> pd.Series:
        if self._next_bar < 0:
            self._next_bar = 0
        for i in range(self._next_bar, index + 1):
            self._process_bar(prices, i)
        self._next_bar = index + 1
        result = {a: (self._history[a][-1] if a in self._history and self._history[a] else float("nan")) for a in prices.columns}
        return pd.Series(result, dtype=float)
    
    

class CumulativeYesNoDelta(Indicator):
    """Incremental cumulative yes-minus-no delta indicator.

    Maintains per-asset running cumulative sums of (yes - no) taken from a
    provided `volume_df` containing paired yes/no columns. The indicator
    exposes `series(asset)` to retrieve the full pd.Series history for an
    asset when needed by downstream indicators.

    Parameters
    ----------
    volume_df:
        DataFrame containing yes/no columns (may be None).
    name:
        Indicator name key stored in the strategy's `indicators` map.
    """

    @property
    def name(self) -> str:
        return self._indicator_name

    def __init__(self, volume_df: pd.DataFrame | None = None, name: str = "cum_yes_no_delta") -> None:
        self._volume = volume_df if volume_df is not None else pd.DataFrame()
        self._indicator_name = name
        self._history: dict[str, list[float]] = {}
        self._ts_lists: dict[str, list] = {}
        self._col_map: dict[str, tuple[str | None, str | None]] = {}
        # running statistics for incremental threshold computation
        self._baseline_map: dict[str, float] = {}
        self._sums: dict[str, float] = {}
        self._sum_sqs: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._thr_history: dict[str, list[float]] = {}
        self._next_bar: int = -1

    def _process_bar(self, prices: pd.DataFrame, i: int) -> None:
        ts = prices.index[i]
        for asset in prices.columns:
            no_col, yes_col = self._col_map.get(asset, (None, None))
            if no_col is None or yes_col is None:
                no_col, yes_col = select_yes_no_columns(self._volume, asset, asset)
                self._col_map[asset] = (no_col, yes_col)

            if yes_col is None or no_col is None or self._volume is None:
                # treat as missing data: carry forward previous cumulative value
                prev = self._history.get(asset, [0.0])[-1] if self._history.get(asset) else 0.0
                cur = prev
                self._history.setdefault(asset, []).append(cur)
                self._ts_lists.setdefault(asset, []).append(ts)
                # ensure baseline exists
                if asset not in self._baseline_map:
                    self._baseline_map[asset] = cur
                # update running stats using baseline-shifted value
                baseline = self._baseline_map[asset]
                shifted = cur - baseline
                self._sums[asset] = self._sums.get(asset, 0.0) + shifted
                self._sum_sqs[asset] = self._sum_sqs.get(asset, 0.0) + shifted * shifted
                self._counts[asset] = self._counts.get(asset, 0) + 1
                n = self._counts[asset]
                mean = self._sums[asset] / n
                var = max(self._sum_sqs[asset] / n - mean * mean, 0.0)
                s = float(np.sqrt(var))
                thr = mean - 3.0 * s
                self._thr_history.setdefault(asset, []).append(thr)
                continue

            try:
                yes = float(self._volume.iloc[i][yes_col])
            except Exception:
                yes = 0.0
            try:
                no = float(self._volume.iloc[i][no_col])
            except Exception:
                no = 0.0
            if not np.isfinite(yes):
                yes = 0.0
            if not np.isfinite(no):
                no = 0.0
            delta = yes - no
            prev = self._history.get(asset, [0.0])[-1] if self._history.get(asset) else 0.0
            cur = prev + delta
            self._history.setdefault(asset, []).append(cur)
            self._ts_lists.setdefault(asset, []).append(ts)
            # set baseline on first observed value so first shifted value is zero
            if asset not in self._baseline_map:
                self._baseline_map[asset] = cur
            baseline = self._baseline_map[asset]
            shifted = cur - baseline
            # update running sums for incremental mean/std of shifted cumulative
            self._sums[asset] = self._sums.get(asset, 0.0) + shifted
            self._sum_sqs[asset] = self._sum_sqs.get(asset, 0.0) + shifted * shifted
            self._counts[asset] = self._counts.get(asset, 0) + 1
            n = self._counts[asset]
            mean = self._sums[asset] / n
            var = max(self._sum_sqs[asset] / n - mean * mean, 0.0)
            s = float(np.sqrt(var))
            thr = mean - 3.0 * s
            self._thr_history.setdefault(asset, []).append(thr)

    def compute(self, prices: pd.DataFrame, returns: pd.DataFrame, index: int) -> pd.Series:
        if self._next_bar < 0:
            self._next_bar = 0
        for i in range(self._next_bar, index + 1):
            self._process_bar(prices, i)
        self._next_bar = index + 1
        result = {a: (self._history[a][-1] if a in self._history and self._history[a] else float("nan")) for a in prices.columns}
        return pd.Series(result, dtype=float)

    def series(self, asset: str) -> "pd.Series | None":
        """Return full cumulative series for *asset* as a pd.Series indexed by timestamps.

        Returns None if no history is available for the asset.
        """
        rows = self._history.get(asset)
        ts = self._ts_lists.get(asset)
        if not rows or not ts:
            return None
        baseline = self._baseline_map.get(asset)
        arr = pd.Series(rows, index=ts, dtype=float)
        if baseline is not None:
            arr = arr - float(baseline)
        return arr

    def threshold_series(self, asset: str) -> "pd.Series | None":
        """Return the per-bar -3sd threshold series for *asset* as a pd.Series indexed by timestamps."""
        rows = self._thr_history.get(asset)
        ts = self._ts_lists.get(asset)
        if not rows or not ts:
            return None
        return pd.Series(rows, index=ts, dtype=float)


class CvdSdThreshold(Indicator):
    """Compute per-asset -3sd threshold from a precomputed cumulative delta indicator.

    This refactored implementation expects a reference to a
    `CumulativeYesNoDelta` indicator instance. It does not re-resolve
    yes/no columns or recompute cumulative sums itself — instead it reads the
    cumulative series produced by the referenced indicator and applies
    `sd_bands_expanding()` to that series to extract the -3sd threshold.

    Parameters
    ----------
    cum_delta_indicator:
        Instance of `CumulativeYesNoDelta` that provides per-asset cumulative
        yes/no delta series via its `series(asset)` method.
    name:
        Optional indicator name.
    """

    @property
    def name(self) -> str:
        return self._indicator_name

    def __init__(self, cum_delta_indicator: "CumulativeYesNoDelta | None" = None, name: str = "cvd_sd_threshold") -> None:
        self._cum_indicator = cum_delta_indicator
        self._indicator_name = name

    def compute(self, prices: pd.DataFrame, returns: pd.DataFrame, index: int) -> pd.Series:
        assets = prices.columns.tolist()
        out: dict[str, float] = {}

        if self._cum_indicator is None:
            for a in assets:
                out[a] = float("nan")
            return pd.Series(out)

        for asset in assets:
            try:
                thr_ser = self._cum_indicator.threshold_series(asset)
                if thr_ser is None or len(thr_ser) == 0 or index >= len(thr_ser):
                    out[asset] = float("nan")
                    continue
                try:
                    out[asset] = float(thr_ser.iloc[index])
                except Exception:
                    out[asset] = float("nan")
            except Exception:
                out[asset] = float("nan")

        return pd.Series(out)


# ---------------------------------------------------------------------------
# Stop-loss and Take-profit helpers
# ---------------------------------------------------------------------------


class StopLossIndicator(Indicator):
    """Compute candidate stop prices based on SD bands or VWAP.

    This helper indicator stores references to a pre-existing `SdBands`
    or `Vwap` indicator instance and can compute an entry-time stop price
    given an entry timestamp and price. The indicator's `compute` method
    returns a per-asset NaN series by default so it integrates cleanly with
    the declarative indicator system; callers should use
    `stop_price_for_entry(...)` to obtain the stop price at a specific
    entry bar.
    """

    def __init__(
        self,
        sd_bands: SdBands | None = None,
        vwap_indicator: Vwap | None = None,
        mode: str = "band",
        band_offset: int = 1,
        vwap_offset_pct: float = 0.0,
        name: str = "stop_loss",
    ) -> None:
        self._sd_bands = sd_bands
        self._vwap_indicator = vwap_indicator
        self.mode = str(mode)
        self.band_offset = int(band_offset)
        self._indicator_name = name

    @property
    def name(self) -> str:
        return self._indicator_name

    def compute(self, prices: pd.DataFrame, returns: pd.DataFrame, index: int) -> pd.Series:
        # Default behaviour: indicator system expects a pd.Series. We don't
        # supply a per-bar stop price by default here — callers should use
        # `stop_price_for_entry` at the exact entry bar. Return NaNs so the
        # indicator map contains the key.
        return pd.Series({a: float("nan") for a in prices.columns}, dtype=float)

    def _nearest_band_index(self, band_row: pd.Series, price: float) -> int | None:
        # Map band labels to integer indexes: mean=0, +1sd=1, -1sd=-1, etc.
        mapping = {
            "mean": 0,
            "+1sd": 1,
            "+2sd": 2,
            "+3sd": 3,
            "-1sd": -1,
            "-2sd": -2,
            "-3sd": -3,
        }
        best = None
        best_idx = None
        for k, idx in mapping.items():
            if k not in band_row.index:
                continue
            try:
                v = float(band_row[k])
            except Exception:
                continue
            d = abs(price - v)
            if best is None or d < best:
                best = d
                best_idx = idx
        return best_idx

    def stop_price_for_entry(
        self,
        prices: pd.DataFrame,
        entry_index: int,
        asset: str,
        entry_price: float,
    ) -> float:
        """Compute an absolute stop price for *asset* at the given entry bar.

        Returns a float stop price or NaN if computation isn't possible.
        """
        # Band-based stop
        if self.mode == "band" and self._sd_bands is not None:
            try:
                ts = prices.index[entry_index]
                bands = self._sd_bands.band_slice(asset, ts, ts)
                if bands is None or bands.empty:
                    return float("nan")
                row = bands.iloc[-1]
                entry_band = self._nearest_band_index(row, float(entry_price))
                if entry_band is None:
                    return float("nan")
                # Move the target band toward the mean relative to the entry band.
                # If the entry is above the mean (positive index), subtract the
                # offset so the stop lies closer to mean/VWAP. If entry is below
                # the mean (negative index), add the offset. If exactly at the
                # mean, default to moving outward by adding the offset.
                if entry_band > 0:
                    target = entry_band - int(self.band_offset)
                elif entry_band < 0:
                    target = entry_band + int(self.band_offset)
                else:
                    target = entry_band + int(self.band_offset)
                # clamp target to [-3,3]
                target = max(-3, min(3, target))
                # map back to label
                label = None
                if target == 0:
                    label = "mean"
                elif target > 0:
                    label = f"+{target}sd"
                else:
                    label = f"{target}sd"
                if label not in row.index:
                    return float("nan")
                stop_val = float(row[label])
                # Ensure stop is not on the wrong side of the entry price.
                # For short positions a stop below the entry would immediately
                # be true on the next tick; avoid that by enforcing a tiny
                # buffer above the entry when the computed stop is <= entry.
                try:
                    ep = float(entry_price)
                except Exception:
                    ep = None
                if ep is not None and np.isfinite(ep) and stop_val <= ep:
                    # use multiplicative buffer so this also works for tiny prices
                    stop_val = ep * 1.000001
                return float(stop_val)
            except Exception:
                return float("nan")

        # VWAP-based stop: stop = VWAP
        if self.mode == "vwap" and self._vwap_indicator is not None:
            try:
                ts = prices.index[entry_index]
                ser = self._vwap_indicator.vwap_slice(asset, ts, ts)
                if ser is None or ser.empty:
                    return float("nan")
                v = float(ser.iloc[-1])
                stop_val = float(v)
                try:
                    ep = float(entry_price)
                except Exception:
                    ep = None
                if ep is not None and np.isfinite(ep) and stop_val <= ep:
                    stop_val = ep * 1.000001
                return float(stop_val)
            except Exception:
                return float("nan")

        return float("nan")


class TakeProfitIndicator(Indicator):
    """Simple take-profit threshold indicator.

    For the current task the TP is a small absolute price threshold near
    zero. The indicator stores a `price_threshold` value and exposes it via
    `threshold_for_entry` so strategies can record the TP price at entry.
    """

    def __init__(self, price_threshold: float = 0.01, name: str = "take_profit") -> None:
        self.price_threshold = float(price_threshold)
        self._indicator_name = name

    @property
    def name(self) -> str:
        return self._indicator_name

    def compute(self, prices: pd.DataFrame, returns: pd.DataFrame, index: int) -> pd.Series:
        # Return the threshold value per asset for integration; strategies
        # will typically read the threshold at entry and store it.
        return pd.Series({a: float(self.price_threshold) for a in prices.columns}, dtype=float)

    def threshold_for_entry(self, prices: pd.DataFrame, entry_index: int, asset: str, entry_price: float) -> float:
        return float(self.price_threshold)



