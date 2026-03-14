import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, cast
import math
import logging as logger

from stratlab.strategy.base import Strategy
from stratlab.strategy.indicators import (
    BandPosition,
    MeanReversion,
    SdBands,
    Vwap,
    VwapSlope,
    VwapVolumeImbalance,
    CumulativeYesNoDelta,
    CvdSdThreshold,
    StopLossIndicator,
    TakeProfitIndicator,
)


# cumulative yes/no delta helpers moved to stratlab.strategy.indicators


# configuration file that holds the profile presets in JSON format
_PRESETS_PATH = Path(__file__).with_name("weather_market_presets.json")


def _load_profile_presets() -> dict[str, dict[str, object]]:
    """Read available trading profiles from the external JSON file.

    Falling back to an empty dict if the file cannot be read keeps the module
    importable for tools that only need the class definitions.
    """
    try:
        text = _PRESETS_PATH.read_text()
        return json.loads(text)
    except FileNotFoundError:
        return {}


PROFILE_PRESETS: dict[str, dict[str, object]] = _load_profile_presets()


class WeatherMarketImbalanceStrategy(Strategy):
    """Short markets classified as imbalanced down in a rolling window.

    The regime logic mirrors stratlab.strategy.indicators.classify_regime() thresholds
    but is evaluated on a rolling lookback window to be deployable in live trading.
    """

    def __init__(
        self,
        lookback: int = 1,
        lookback_hours: float | None = 6.0,
        use_market_regime: bool = True,
        allow_cash: bool = True,
        entry_regime: str = "Imb. Down",
        exit_mode: str = "hold_to_end",
        max_concurrent_positions: int | None = None,
        apply_global_position_cap: bool = False,
        mean_reversion_window: int = 5,
        imbalance_above_mean_threshold: float = 60.0,
        imbalance_above_1sd_threshold: float = 40.0,
        imbalance_below_1sd_threshold: float = 50.0,
        imbalance_down_above_mean_cap: float = 30.0,
        balanced_within_1sd_threshold: float = 70.0,
        mean_reversion_threshold: float = 0.5,
        use_vwap_slope_filter: bool = True,
        use_vwap_volume_imbalance_filter: bool = True,
        max_vwap_volume_imbalance_pct: float = -1.0,
        vwap_volume_imbalance_lookback: int | None = None,
        vwap: pd.DataFrame | None = None,
        volume: pd.DataFrame | None = None,
        high: pd.DataFrame | None = None,
        low: pd.DataFrame | None = None,
        open_: pd.DataFrame | None = None,
        pricing_method: str = "typical",
        vwap_slope_mode: str = "scaled",
        vwap_slope_value_per_point: float = 1e-4,
        vwap_slope_scale: float = 1.0,
        vwap_slope_lookback: int = 15, # in bars, not hours; should be less than lookback if lookback_hours is set
        max_vwap_slope: float = -2.0,
        buy_volume: pd.DataFrame | None = None,
        sell_volume: pd.DataFrame | None = None,
        use_buy_cvd_filter: bool = True,
        max_buy_cvd_for_short: float = 0.0,
        use_sell_cvd_filter: bool = True,
        min_sell_cvd_for_short: float = 0.0,
        use_cvd_sd_gate: bool = False,
        # stop-loss / take-profit controls
        use_stop_loss: bool = False,
        stop_loss_mode: str = "band",
        sl_entry_band_offset: int = 1,
        use_take_profit: bool = False,
        take_profit_price_threshold: float = 0.01,
        use_trailing_stop: bool = False,
        min_sell_price: float | None = 0.05,
        max_buy_price: float | None = 0.95,
    ):
        """Initialize the weather imbalance strategy.

        Tuning guide (most important first):

        Entry strictness / signal quality:
        - lookback_hours: Rolling window used for regime classification.
            Larger -> smoother/slower signals; smaller -> faster/noisier signals.
        - imbalance_below_1sd_threshold: Minimum % of bars below -1sd to call Imb. Down.
            Larger -> stricter downside imbalance requirement.
        - imbalance_down_above_mean_cap: Max % bars above mean for Imb. Down.
            Smaller -> stricter bearish condition.
        - use_vwap_slope_filter: Enables VWAP trend confirmation.
        - vwap_slope_mode: "raw" (price units/bar), "scaled" (normalized),
            or "angle" (degrees of normalized slope).
        - vwap_slope_value_per_point: Normalization factor applied before
            scaled/angle conversion, similar to Sierra Chart value-per-point.
        - vwap_slope_scale: Optional multiplier for scaled/angle output.
        - vwap_slope_lookback: Lookback for slope regression in BARS (not hours).
        - max_vwap_slope: Entry requires slope <= this value.
            More negative -> stricter downward trend requirement.
        - use_vwap_volume_imbalance_filter: Enables VWAP volume-imbalance gate.
        - max_vwap_volume_imbalance_pct: Entry requires imbalance <= this level.
            Example: 50 means skip entries when above-VWAP volume dominates too much.
        - vwap_volume_imbalance_lookback: Lookback in UPDATE BARS for imbalance.
            If None, uses vwap_slope_lookback.

        Buy-volume delta filter:
        - buy_volume: optional DataFrame containing per-bar "yes" and "no"
            buy-volume columns (e.g. "market__yes"/"market__no").
        - use_buy_cvd_filter / max_buy_cvd_for_long: optional negative-cumulative
            buy CVD gate for shorts, requiring cumulative (yes - no) to be below
            the configured threshold.

        Sell-volume delta filter:
        - sell_volume: optional DataFrame containing per-bar "yes" and "no"
            sell-volume columns (e.g. "market__yes"/"market__no").
        - use_sell_cvd_filter / min_sell_cvd_for_short: optional positive-cumulative
            sell CVD gate for shorts, requiring cumulative (yes - no) to be above
            the configured threshold.

        Regime behavior (classification boundaries):
        - imbalance_above_mean_threshold / imbalance_above_1sd_threshold:
            Rules for Imb. Up classification.
        - balanced_within_1sd_threshold: % within +/-1sd to classify as Balanced.
        - mean_reversion_window / mean_reversion_threshold:
            Controls Mean-Reverting classification sensitivity.

        Positioning / risk controls:
        - max_concurrent_positions: Optional global cap on open positions.
            Only active when apply_global_position_cap=True.
        - apply_global_position_cap: If False, each market can still have only one
            open position, but there is no event-wide cap.
        - exit_mode: "hold_to_end" or "take_profit_stop_loss".
        - take_profit / stop_loss: Only used in take-profit/stop-loss mode.
        """
        self.lookback = int(lookback)
        self.lookback_hours = lookback_hours
        self.use_market_regime = bool(use_market_regime)
        self.allow_cash = allow_cash
        self.entry_regime = entry_regime
        self.exit_mode = exit_mode
        self.max_concurrent_positions = max_concurrent_positions
        self.apply_global_position_cap = bool(apply_global_position_cap)

        self.mean_reversion_window = int(mean_reversion_window)
        self.imbalance_above_mean_threshold = float(imbalance_above_mean_threshold)
        self.imbalance_above_1sd_threshold = float(imbalance_above_1sd_threshold)
        self.imbalance_below_1sd_threshold = float(imbalance_below_1sd_threshold)
        self.imbalance_down_above_mean_cap = float(imbalance_down_above_mean_cap)
        self.balanced_within_1sd_threshold = float(balanced_within_1sd_threshold)
        self.mean_reversion_threshold = float(mean_reversion_threshold)

        self.buy_volume = buy_volume if buy_volume is not None else pd.DataFrame()
        self.sell_volume = sell_volume if sell_volume is not None else pd.DataFrame()
        self._buy_delta_cache: dict[str, tuple[str | None, str | None]] = {}
        self._sell_delta_cache: dict[str, tuple[str | None, str | None]] = {}

        # history of (index, asset, delta) whenever ``_buy_delta`` is called
        # during weight generation; useful for tests and debugging.
        self.buy_delta_history: list[tuple[int, str, float]] = []
        self.sell_delta_history: list[tuple[int, str, float]] = []

        self.use_buy_cvd_filter = bool(use_buy_cvd_filter)
        self.max_buy_cvd_for_short = float(max_buy_cvd_for_short)
        self.use_sell_cvd_filter = bool(use_sell_cvd_filter)
        self.min_sell_cvd_for_short = float(min_sell_cvd_for_short)
        self.use_cvd_sd_gate = bool(use_cvd_sd_gate)
        self.use_stop_loss = bool(use_stop_loss)
        self.stop_loss_mode = str(stop_loss_mode)
        self.sl_entry_band_offset = int(sl_entry_band_offset)
        self.use_take_profit = bool(use_take_profit)
        self.use_trailing_stop = bool(use_trailing_stop)
        self.take_profit_price_threshold = float(take_profit_price_threshold)
        self.min_sell_price = None if min_sell_price is None else float(min_sell_price)
        self.max_buy_price = None if max_buy_price is None else float(max_buy_price)

        self.use_vwap_slope_filter = bool(use_vwap_slope_filter)
        self.use_vwap_volume_imbalance_filter = bool(use_vwap_volume_imbalance_filter)
        self.max_vwap_volume_imbalance_pct = float(max_vwap_volume_imbalance_pct)
        self.vwap = vwap
        self.volume = volume
        self.vwap_slope_mode = str(vwap_slope_mode)
        self.vwap_slope_value_per_point = float(vwap_slope_value_per_point)
        self.vwap_slope_scale = float(vwap_slope_scale)
        self.vwap_slope_lookback = int(vwap_slope_lookback)
        if vwap_volume_imbalance_lookback is None:
            self.vwap_volume_imbalance_lookback = int(vwap_slope_lookback)
        else:
            self.vwap_volume_imbalance_lookback = int(vwap_volume_imbalance_lookback)
        self.max_vwap_slope = float(max_vwap_slope)

        _vwap = vwap if vwap is not None else pd.DataFrame()
        _volume = volume if volume is not None else pd.DataFrame()
        _high = high if high is not None else pd.DataFrame()
        _low = low if low is not None else pd.DataFrame()
        _open = open_ if open_ is not None else pd.DataFrame() # open is a reserved keyword, so we use open_ in the signature and rename it here for indicator construction

        # keep high/low on the instance for intrabar checks
        self._high = _high
        self._low = _low

        _sd_bands = SdBands(
            pricing_method=pricing_method,
            high=_high,
            low=_low,
            open_=_open,
        )
        _cum_buy_delta = CumulativeYesNoDelta(volume_df=buy_volume, name="cum_buy_delta")
        _cum_sell_delta = CumulativeYesNoDelta(volume_df=sell_volume, name="cum_sell_delta")
        _cvd_buy_sd_thr = CvdSdThreshold(cum_delta_indicator=_cum_buy_delta, name="_cvd_buy_sd_thr")
        _vwap_indicator = Vwap(
            volume=_volume,
            pricing_method=pricing_method,
            high=_high,
            low=_low,
            open_=_open,
        )
        # Stop-loss and Take-profit indicator instances (helpers)
        _stop_loss_indicator = StopLossIndicator(
            sd_bands=_sd_bands,
            vwap_indicator=_vwap_indicator,
            mode=self.stop_loss_mode,
            band_offset=self.sl_entry_band_offset,
        )
        _take_profit_indicator = TakeProfitIndicator(price_threshold=self.take_profit_price_threshold)
        self.indicator_defs = [
            _sd_bands,
            _cum_buy_delta,
            _cum_sell_delta,
            _cvd_buy_sd_thr,
            _vwap_indicator,
            VwapSlope(
                vwap=_vwap,
                volume=_volume,
                vwap_indicator=_vwap_indicator,
                lookback=self.vwap_slope_lookback,
                mode=self.vwap_slope_mode,
                value_per_point=self.vwap_slope_value_per_point,
                scale=self.vwap_slope_scale,
            ),
            VwapSlope(
                vwap=_vwap,
                volume=_volume,
                vwap_indicator=_vwap_indicator,
                lookback=self.vwap_slope_lookback,
                mode="raw",
                name="vwap_slope_raw",
            ),
            VwapVolumeImbalance(
                vwap=_vwap,
                volume=_volume,
                vwap_indicator=_vwap_indicator,
                lookback=self.vwap_volume_imbalance_lookback,
            ),
            BandPosition(
                lookback_hours=lookback_hours,
                lookback_bars=int(lookback) if lookback_hours is None else None,
                sd_bands=_sd_bands,
            ),
            MeanReversion(
                window=self.mean_reversion_window,
                lookback_hours=lookback_hours,
                lookback_bars=int(lookback) if lookback_hours is None else None,
            ),
            _stop_loss_indicator,
            _take_profit_indicator,
        ]

        self._positions: dict[str, dict[str, object]] = {}
        self.trade_log: list[dict[str, object]] = []


    @classmethod
    def available_profiles(cls) -> list[str]:
        return sorted(PROFILE_PRESETS.keys())

    @classmethod
    def profile_params(cls, profile: str) -> dict[str, object]:
        if profile not in PROFILE_PRESETS:
            valid = ", ".join(cls.available_profiles())
            raise ValueError(f"Unknown profile {profile!r}. Choose one of: {valid}")
        return dict(PROFILE_PRESETS[profile])

    @classmethod
    def from_profile(
        cls,
        profile: str,
        **overrides: object,
    ) -> "WeatherMarketImbalanceStrategy":
        params: dict[str, Any] = cls.profile_params(profile)
        params.update(overrides)
        return cls(**params)

    def _classify_regime(self, asset: str) -> tuple[str, float]:
        indicators = getattr(self, "indicators", {})

        band_df = indicators.get("band_position")
        if band_df is not None and asset in band_df.columns:
            col = band_df[asset]
            above_mean = float(col["above_mean_pct"])
            above_1sd = float(col["above_1sd_pct"])
            below_1sd = float(col["below_minus_1sd_pct"])
            within = float(col["within_1sd_pct"])
        else:
            above_mean = above_1sd = below_1sd = within = 0.0

        mr_series = indicators.get("mean_reversion")
        mean_rev = float(mr_series.get(asset, 0.0)) if mr_series is not None else 0.0

        if (
            above_mean > self.imbalance_above_mean_threshold
            and above_1sd > self.imbalance_above_1sd_threshold
        ):
            return "Imb. Up", min(1.0, (above_mean - 50) / 50 + above_1sd / 100.0)

        if (
            below_1sd > self.imbalance_below_1sd_threshold
            and above_mean < self.imbalance_down_above_mean_cap
        ):
            return "Imb. Down", min(1.0, (40 - above_mean) / 50 + below_1sd / 100.0)

        if mean_rev >= self.mean_reversion_threshold:
            return "Mean-Reverting", mean_rev

        if within >= self.balanced_within_1sd_threshold:
            return "Balanced", within / 100.0

        return "Rotational", 0.5

    @staticmethod
    def _short_trade_return(entry_price: float, current_price: float) -> float:
        if entry_price <= 0:
            return 0.0
        return float(entry_price / current_price - 1.0)

    def _should_exit(self, asset: str, current_price: float, index: int, prices: pd.DataFrame) -> str | None:
        # Only skip exit checks when neither the exit_mode requests TP/SL
        # nor the boolean gates are enabled in the profile.
        if not (
            self.exit_mode == "take_profit_stop_loss"
            or getattr(self, "use_stop_loss", False)
            or getattr(self, "use_take_profit", False)
        ):
            return None

        if asset not in self._positions:
            return None

        position = self._positions.get(asset)
        if position is None:
            return None
        entry_price = float(cast(float, position.get("entry_price", 0.0)))

        # Per-position TP/SL price checks (prefer absolute price thresholds)
        tp_price = position.get("tp_price", float("nan"))
        try:
            tp_val = float(cast(float, tp_price))
        except Exception:
            tp_val = float("nan")
        if math.isfinite(tp_val):
            # short: take profit when price falls to or below tp_price
            # check intrabar low if available, otherwise use close
            low_val = None
            try:
                if hasattr(self, "_low") and asset in self._low.columns:
                    low_val = float(self._low.iloc[index][asset])
            except Exception:
                low_val = None
            if low_val is not None and np.isfinite(low_val):
                if low_val <= tp_val:
                    return "take_profit"
            else:
                if current_price <= tp_val:
                    return "take_profit"

        stop_price = position.get("stop_price", float("nan"))
        try:
            stop_val = float(cast(float, stop_price))
        except Exception:
            stop_val = float("nan")
        if math.isfinite(stop_val):
            # short: stop-loss when price rises to or above stop_price
            # check intrabar high if available, otherwise use close
            high_val = None
            try:
                if hasattr(self, "_high") and asset in self._high.columns:
                    high_val = float(self._high.iloc[index][asset])
            except Exception:
                high_val = None
            if high_val is not None and np.isfinite(high_val):
                if high_val >= stop_val:
                    return "stop_loss"
            else:
                if current_price >= stop_val:
                    return "stop_loss"

        # No percentage-based fallback: only absolute TP/SL are enforced.
        return None

    def _close_position(
        self,
        asset: str,
        current_price: float,
        current_index: int,
        prices: pd.DataFrame,
        reason: str,
    ) -> None:
        position = self._positions.get(asset)
        if position is None:
            return

        entry_price = float(cast(float, position.get("entry_price", 0.0)))
        entry_index = int(cast(int, position.get("entry_index", current_index)))
        confidence = float(cast(float, position.get("confidence", 0.0)))
        vwap_slope = float(cast(float, position.get("vwap_slope", 0.0)))
        vwap_slope_raw = float(cast(float, position.get("vwap_slope_raw", 0.0)))
        vwap_volume_imbalance_pct = float(cast(float, position.get("vwap_volume_imbalance_pct", float("nan"))))
        buy_cvd = float(cast(float, position.get("buy_cvd", float("nan"))))
        sell_cvd = float(cast(float, position.get("sell_cvd", float("nan"))))
        stop_price = float(cast(float, position.get("stop_price", float("nan"))))
        tp_price = float(cast(float, position.get("tp_price", float("nan"))))
        # Prefer intrabar fill price when available for stop_loss / take_profit
        fill_price = float(current_price)
        try:
            if reason == "stop_loss" and hasattr(self, "_high") and asset in self._high.columns:
                hi = float(self._high.iloc[current_index][asset])
                if np.isfinite(hi) and hi > 0:
                    fill_price = hi
        except Exception:
            pass
        try:
            if reason == "take_profit" and hasattr(self, "_low") and asset in self._low.columns:
                lo = float(self._low.iloc[current_index][asset])
                if np.isfinite(lo) and lo > 0:
                    fill_price = lo
        except Exception:
            pass

        pnl = self._short_trade_return(entry_price, fill_price)
        self.trade_log.append(
            {
                "asset": asset,
                "entry_index": entry_index,
                "entry_time": position["entry_time"],
                "entry_price": entry_price,
                "exit_index": int(current_index),
                "exit_time": prices.index[current_index],
                "exit_price": float(fill_price),
                "exit_reason": reason,
                "pnl": pnl,
                "regime": position.get("regime"),
                "confidence": confidence,
                "vwap_slope": vwap_slope,
                "vwap_slope_raw": vwap_slope_raw,
                "vwap_volume_imbalance_pct": vwap_volume_imbalance_pct,
                "buy_cvd": buy_cvd,
                "sell_cvd": sell_cvd,
                "stop_price": stop_price,
                "tp_price": tp_price,
            }
        )
        self._positions.pop(asset, None)

    def finalize(self, prices: pd.DataFrame) -> None:
        """Close any open positions at the final available timestamp."""
        if not self._positions:
            return

        final_index = len(prices) - 1
        if final_index < 0:
            return

        for asset in list(self._positions.keys()):
            if asset not in prices.columns:
                self._positions.pop(asset, None)
                continue
            final_price = float(prices.iloc[final_index][asset])
            if np.isnan(final_price) or final_price <= 0:
                self._positions.pop(asset, None)
                continue
            self._close_position(
                asset=asset,
                current_price=final_price,
                current_index=final_index,
                prices=prices,
                reason="end_of_data",
            )

    def generate_weights(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        index: int,
    ) -> np.ndarray:
        assets = prices.columns.tolist()
        current_ts = pd.Timestamp(prices.index[index])

        # Trailing-stop: recompute stored stop_price for open positions
        # using current bands/VWAP so stops can move as indicators evolve.
        if getattr(self, "use_trailing_stop", False):
            sl_obj = None
            for ind in getattr(self, "indicator_defs", []):
                try:
                    if isinstance(ind, StopLossIndicator):
                        sl_obj = ind
                        break
                except Exception:
                    continue
            if sl_obj is not None:
                for asset in list(self._positions.keys()):
                    if asset not in assets:
                        continue
                    try:
                        cur_price = float(prices.iloc[index][asset])
                    except Exception:
                        continue
                    if np.isnan(cur_price) or cur_price <= 0:
                        continue
                    try:
                        new_stop = sl_obj.stop_price_for_entry(prices, index, asset, cur_price)
                        if np.isfinite(new_stop):
                            self._positions[asset]["stop_price"] = float(new_stop)
                    except Exception:
                        continue

        # Exit checks on existing positions
        to_close = []
        for asset in list(self._positions.keys()):
            if asset not in assets:
                to_close.append((asset, "asset_missing"))
                continue
            current_price = float(prices.iloc[index][asset])
            if np.isnan(current_price) or current_price <= 0:
                continue
            exit_reason = self._should_exit(asset, current_price, index, prices)
            if exit_reason is not None:
                to_close.append((asset, exit_reason))

        for asset, reason in to_close:
            if asset not in assets:
                self._positions.pop(asset, None)
                continue
            close_price = float(prices.iloc[index][asset])
            if np.isnan(close_price) or close_price <= 0:
                self._positions.pop(asset, None)
                continue
            self._close_position(
                asset=asset,
                current_price=close_price,
                current_index=index,
                prices=prices,
                reason=reason,
            )

        # Entry checks on all assets
        ranked_entries: list[tuple[str, float, float, float, float, float, float]] = []
        # compute the window start used for market-regime and other indicators
        # but *do not* bail out early; we still want to observe buy-volume
        # delta on every bar.  previously the lookback logic returned before any
        # deltas were recorded, which delayed entries until the window filled.
        if self.lookback_hours is not None:
            start_ts = current_ts - pd.Timedelta(hours=float(self.lookback_hours))
            index_series = pd.DatetimeIndex(prices.index)
            start = int(index_series.searchsorted(start_ts, side="left"))
            # determine whether the regime window is complete
            need_full_window = start >= index or pd.Timestamp(prices.index[start]) > start_ts
        else:
            start = max(0, index - self.lookback + 1)
            need_full_window = False

        for asset in assets:
            # read precomputed indicators instead of calling the function
            inds = getattr(self, "indicators", {})
            def _read_cum(name: str) -> float:
                series = inds.get(name)
                if series is None:
                    return float("nan")
                try:
                    # DataFrame with time index x asset columns
                    return float(series.iloc[index].get(asset, float("nan")))
                except Exception:
                    try:
                        # Series mapping asset -> value
                        return float(series.get(asset, float("nan")))
                    except Exception:
                        return float("nan")

            buy_delta = _read_cum("cum_buy_delta")
            sell_delta = _read_cum("cum_sell_delta")
            self.buy_delta_history.append((index, asset, buy_delta))
            self.sell_delta_history.append((index, asset, sell_delta))

            asset_window = prices.iloc[start : index + 1][asset].dropna().to_numpy(dtype=float)
            if len(asset_window) < max(2, self.lookback // 2):
                continue

            if self.use_market_regime:
                # if the regime window isn't yet complete, skip asset entirely
                if need_full_window:
                    continue
                regime, confidence = self._classify_regime(asset)
                if regime != self.entry_regime:
                    continue
            else:
                # market‑regime filtering disabled – allow everything through
                # still supply a confidence for ranking purposes
                regime = self.entry_regime
                confidence = 0.5

            slope = 0.0
            raw_slope = 0.0
            if self.use_vwap_slope_filter:
                slope = float(self.indicators["vwap_slope"].get(asset, 0.0))
                raw_slope = float(self.indicators["vwap_slope_raw"].get(asset, 0.0))
                if slope > self.max_vwap_slope:
                    continue

            imbalance_pct = float("nan")
            if self.use_vwap_volume_imbalance_filter:
                imbalance_pct = float(self.indicators["vwap_volume_imbalance"].get(asset, float("nan")))
                if np.isnan(imbalance_pct) or imbalance_pct > self.max_vwap_volume_imbalance_pct:
                    continue

            if self.use_cvd_sd_gate:
                thr_series = getattr(self, "indicators", {}).get("_cvd_buy_sd_thr")
                thr = float("nan")
                if isinstance(thr_series, pd.Series):
                    try:
                        thr = float(thr_series.get(asset, float("nan")))
                    except Exception:
                        thr = float("nan")

                if math.isfinite(thr):
                    if (not math.isfinite(buy_delta)) or buy_delta >= thr:
                        logger.debug(
                            "%s @ %s: buy CVD = %s must be < -3sd=%s – skipping",
                            asset,
                            current_ts,
                            buy_delta,
                            thr,
                        )
                        continue
                else:
                    if (not math.isfinite(buy_delta)) or buy_delta >= self.max_buy_cvd_for_short:
                        logger.debug(
                            "%s @ %s: buy CVD = %s must be < max_buy_cvd_for_short=%s (no sd threshold) – skipping",
                            asset,
                            current_ts,
                            buy_delta,
                            self.max_buy_cvd_for_short,
                        )
                        continue
            
            if self.use_buy_cvd_filter:
                if (not math.isfinite(buy_delta)) or buy_delta >= self.max_buy_cvd_for_short:
                    logger.debug(
                        "%s @ %s: buy CVD = %s must be < max_buy_cvd_for_short=%s – skipping",
                        asset,
                        current_ts,
                        buy_delta,
                        self.max_buy_cvd_for_short,
                    )
                    continue

            if self.use_sell_cvd_filter:
                if (not math.isfinite(sell_delta)) or sell_delta <= self.min_sell_cvd_for_short:
                    logger.debug(
                        "%s @ %s: sell CVD = %s must be > min_sell_cvd_for_short=%s – skipping",
                        asset,
                        current_ts,
                        sell_delta,
                        self.min_sell_cvd_for_short,
                    )
                    continue

            # asset passed all filters – add to ranking list
            ranked_entries.append((asset, confidence, slope, raw_slope, imbalance_pct, buy_delta, sell_delta))

        ranked_entries.sort(key=lambda x: x[1], reverse=True)

        if self.apply_global_position_cap and self.max_concurrent_positions is not None:
            capacity = max(0, self.max_concurrent_positions - len(self._positions))
            ranked_entries = ranked_entries[:capacity]

        for asset, confidence, slope, raw_slope, imbalance_pct, buy_delta_entry, sell_delta_entry in ranked_entries:
            if asset in self._positions:
                continue
            entry_price = float(prices.iloc[index][asset])
            if np.isnan(entry_price) or entry_price <= 0:
                continue
            # Enforce min/max price guards to avoid re-entries at near-zero prices
            if self.min_sell_price is not None:
                # this strategy opens shorts (sells) — skip if price below min_sell_price
                if entry_price < self.min_sell_price:
                    continue
            if self.max_buy_price is not None:
                # if strategy allowed longs, skip if price above max_buy_price
                if entry_price > self.max_buy_price:
                    continue
            logger.debug(
                "opening short %s @ %s price=%.5f buy_cvd=%.2f sell_cvd=%.2f",
                asset,
                current_ts,
                entry_price,
                buy_delta_entry,
                sell_delta_entry,
            )
            # compute stop_price and tp_price at entry time using indicator helpers
            stop_price = float("nan")
            if self.use_stop_loss:
                # find StopLossIndicator instance
                sl_obj = None
                for ind in getattr(self, "indicator_defs", []):
                    try:
                        if isinstance(ind, StopLossIndicator):
                            sl_obj = ind
                            break
                    except Exception:
                        continue
                if sl_obj is not None:
                    try:
                        stop_price = sl_obj.stop_price_for_entry(prices, index, asset, entry_price)
                    except Exception:
                        stop_price = float("nan")

            tp_price = float("nan")
            if self.use_take_profit:
                tp_obj = None
                for ind in getattr(self, "indicator_defs", []):
                    try:
                        if isinstance(ind, TakeProfitIndicator):
                            tp_obj = ind
                            break
                    except Exception:
                        continue
                if tp_obj is not None:
                    try:
                        tp_price = tp_obj.threshold_for_entry(prices, index, asset, entry_price)
                    except Exception:
                        tp_price = float("nan")

            self._positions[asset] = {
                "entry_price": entry_price,
                "entry_index": int(index),
                "entry_time": prices.index[index],
                "regime": self.entry_regime,
                "confidence": float(confidence),
                "vwap_slope": float(slope),
                "vwap_slope_raw": float(raw_slope),
                "vwap_volume_imbalance_pct": float(imbalance_pct),
                "buy_cvd": float(buy_delta_entry),
                "sell_cvd": float(sell_delta_entry),
                "stop_price": stop_price,
                "tp_price": tp_price,
            }

        # Build short-only weight vector
        weights = np.zeros(len(assets), dtype=float)
        open_assets = [a for a in assets if a in self._positions]
        if not open_assets:
            return weights

        short_weight = -1.0 / len(open_assets)
        for i, asset in enumerate(assets):
            if asset in self._positions:
                weights[i] = short_weight

        # allow_cash retained for future behavior; currently unused because
        # short weights are normalized over active shorts only.
        _ = self.allow_cash
        return weights
