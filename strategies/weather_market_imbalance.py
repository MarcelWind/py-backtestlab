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
    cumulative_yes_no_delta,
)


# cumulative yes/no delta helpers moved to stratlab.strategy.indicators


# configuration file that holds the profile presets in JSON format
_PRESETS_PATH = Path(__file__).with_name("weather_market_imbalance_presets.json")


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
        take_profit: float = 0.5,
        stop_loss: float = 0.20,
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
        self.take_profit = float(take_profit)
        self.stop_loss = float(stop_loss)
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

        # *** indicator definitions belong in __init__, not _buy_delta ***
        _vwap = vwap if vwap is not None else pd.DataFrame()
        _volume = volume if volume is not None else pd.DataFrame()
        _high = high
        _low = low
        _open = open_

        _sd_bands = SdBands(
            pricing_method=pricing_method,
            high=_high,
            low=_low,
            open_=_open,
        )
        _vwap_ind = Vwap(
            volume=_volume,
            pricing_method=pricing_method,
            high=_high,
            low=_low,
            open_=_open,
        )
        self.indicator_defs = [
            _sd_bands,
            _vwap_ind,
            VwapSlope(
                vwap=_vwap,
                volume=_volume,
                vwap_indicator=_vwap_ind,
                lookback=self.vwap_slope_lookback,
                mode=self.vwap_slope_mode,
                value_per_point=self.vwap_slope_value_per_point,
                scale=self.vwap_slope_scale,
            ),
            VwapSlope(
                vwap=_vwap,
                volume=_volume,
                vwap_indicator=_vwap_ind,
                lookback=self.vwap_slope_lookback,
                mode="raw",
                name="vwap_slope_raw",
            ),
            VwapVolumeImbalance(
                vwap=_vwap,
                volume=_volume,
                vwap_indicator=_vwap_ind,
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
        ]

        self._positions: dict[str, dict[str, object]] = {}
        self.trade_log: list[dict[str, object]] = []


    def _buy_delta(self, asset: str, prices: pd.DataFrame, index: int) -> float:
        """Return **cumulative** yes-minus-no buy-volume delta for *asset*.

        The returned value is the running sum of (yes - no) from the beginning
        of ``prices`` through ``index``.  The series is shifted so that the
        first available bar always returns zero — this mirrors the behaviour of
        the plotting helper which subtracts ``cum_delta.iloc[0]`` when
        drawing the blue line.

        If the necessary columns cannot be resolved or the value is missing,
        ``nan`` is returned.  The expensive column lookup is cached by asset
        name so repeated calls are cheap.
        """
        return cumulative_yes_no_delta(
            volume_df=self.buy_volume,
            cache=self._buy_delta_cache,
            asset=asset,
            prices=prices,
            index=index,
        )

    def _sell_delta(self, asset: str, prices: pd.DataFrame, index: int) -> float:
        """Return **cumulative** yes-minus-no sell-volume delta for *asset*."""
        return cumulative_yes_no_delta(
            volume_df=self.sell_volume,
            cache=self._sell_delta_cache,
            asset=asset,
            prices=prices,
            index=index,
        )

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

    def _should_exit(self, asset: str, current_price: float) -> str | None:
        if self.exit_mode == "hold_to_end":
            return None
        if self.exit_mode != "take_profit_stop_loss":
            return None

        if asset not in self._positions:
            return None

        entry_price = float(cast(float, self._positions[asset]["entry_price"]))
        pnl = self._short_trade_return(entry_price, current_price)
        if pnl >= self.take_profit:
            return "take_profit"
        if pnl <= -self.stop_loss:
            return "stop_loss"
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
        pnl = self._short_trade_return(entry_price, current_price)
        self.trade_log.append(
            {
                "asset": asset,
                "entry_index": entry_index,
                "entry_time": position["entry_time"],
                "entry_price": entry_price,
                "exit_index": int(current_index),
                "exit_time": prices.index[current_index],
                "exit_price": float(current_price),
                "exit_reason": reason,
                "pnl": pnl,
                "regime": position.get("regime"),
                "confidence": confidence,
                "vwap_slope": vwap_slope,
                "vwap_slope_raw": vwap_slope_raw,
                "vwap_volume_imbalance_pct": vwap_volume_imbalance_pct,
                "buy_cvd": buy_cvd,
                "sell_cvd": sell_cvd,
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

        # Exit checks on existing positions
        to_close = []
        for asset in list(self._positions.keys()):
            if asset not in assets:
                to_close.append((asset, "asset_missing"))
                continue
            current_price = float(prices.iloc[index][asset])
            if np.isnan(current_price) or current_price <= 0:
                continue
            exit_reason = self._should_exit(asset, current_price)
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
        ranked_entries: list[tuple[str, float, float, float, float]] = []
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
            # compute delta immediately; always record history so that we
            # know when CVD first turns negative regardless of lookback.
            buy_delta = self._buy_delta(asset, prices, index)
            sell_delta = self._sell_delta(asset, prices, index)
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
            ranked_entries.append((asset, confidence, slope, raw_slope, imbalance_pct))

        ranked_entries.sort(key=lambda x: x[1], reverse=True)

        if self.apply_global_position_cap and self.max_concurrent_positions is not None:
            capacity = max(0, self.max_concurrent_positions - len(self._positions))
            ranked_entries = ranked_entries[:capacity]

        for asset, confidence, slope, raw_slope, imbalance_pct in ranked_entries:
            if asset in self._positions:
                continue
            entry_price = float(prices.iloc[index][asset])
            if np.isnan(entry_price) or entry_price <= 0:
                continue
            logger.debug(
                "opening short %s @ %s price=%.5f buy_cvd=%.2f sell_cvd=%.2f",
                asset,
                current_ts,
                entry_price,
                buy_delta,
                sell_delta,
            )
            self._positions[asset] = {
                "entry_price": entry_price,
                "entry_index": int(index),
                "entry_time": prices.index[index],
                "regime": self.entry_regime,
                "confidence": float(confidence),
                "vwap_slope": float(slope),
                "vwap_slope_raw": float(raw_slope),
                "vwap_volume_imbalance_pct": float(imbalance_pct),
                "buy_cvd": float(buy_delta),
                "sell_cvd": float(sell_delta),
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
