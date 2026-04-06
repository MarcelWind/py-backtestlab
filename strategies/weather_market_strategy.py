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
    nearest_band_index,
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
        side: str | None = None,
        allow_longs: bool | None = None,
        allow_shorts: bool | None = None,
        exit_mode: str = "hold_to_end",
        max_concurrent_positions: int | None = None,
        apply_global_position_cap: bool = False,
        mean_reversion_window: int = 5,
        imbalance_above_mean_threshold: float = 60.0,
        imbalance_above_1sd_threshold: float = 40.0,
        imbalance_below_1sd_threshold: float = 50.0,
        imbalance_up_below_mean_cap: float = 40.0,
        imbalance_down_above_mean_cap: float = 30.0,
        balanced_within_1sd_threshold: float = 70.0,
        mean_reversion_threshold: float = 0.5,
        use_vwap_slope_filter: bool = True,
        use_vwap_volume_imbalance_filter: bool = True,
        max_vwap_volume_imbalance_pct: float = -1.0,
        max_vwap_volume_imbalance_pct_for_short: float | None = None,
        min_vwap_volume_imbalance_pct_for_long: float | None = None,
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
        max_vwap_slope_for_short: float | None = None,
        min_vwap_slope_for_long: float | None = None,
        buy_volume: pd.DataFrame | None = None,
        sell_volume: pd.DataFrame | None = None,
        use_buy_cvd_filter: bool = True,
        max_buy_cvd_for_short: float = 0.0,
        min_buy_cvd_for_long: float | None = None,
        use_buy_cvd_3sd_gate: bool = False,
        use_sell_cvd_filter: bool = True,
        min_sell_cvd_for_short: float = 0.0,
        max_sell_cvd_for_long: float | None = None,
        # stop-loss / take-profit controls
        use_stop_loss: bool = False,
        stop_loss_mode: str = "band",
        sl_entry_band_offset: int = 1,
        use_take_profit: bool = False,
        take_profit_price_short: float = 0.01,
        take_profit_price_long: float = 0.99,
        # Backward-compatible aliases
        take_profit_price_threshold_short: float | None = None,
        take_profit_price_threshold_long: float | None = None,
        take_profit_price_threshold: float | None = None,
        use_trailing_stop: bool = False,
        min_sell_price: float | None = 0.05,
        max_buy_price: float | None = 0.95,
        track_delta_history: bool = True,
        # Market regime mode: "imbalance" (default) or "rotational"
        market_regime_mode: str = "imbalance",
        # Rotational entry parameters (active when market_regime_mode="rotational")
        rotational_price_source: str = "highlow",
        rotational_entry_band: int = 3,
        rotational_exit_offset: int | str = 1,
        rotational_window_hours: float = 12.0,
        rotational_regimes: list[str] | None = None,
        market_end_time: pd.Timestamp | None = None,
        # Renamed imbalance-mode slope/volume thresholds (preferred canonical names).
        # "min" semantics: "minimum -1.0 or lower" — slope/imbalance must be
        # at least this extreme to confirm the imbalance trend.
        min_vwap_slope_for_short: float | None = None,
        min_vwap_volume_imbalance_pct_for_short: float | None = None,
        # Rotational-mode slope/volume ceilings.
        # "max" semantics: slope/imbalance must not exceed this value,
        # preventing entries when conditions suggest imbalance rather than rotation.
        max_vwap_slope_for_long: float | None = None,
        max_vwap_volume_imbalance_pct_for_short_rot: float | None = None,
        max_vwap_volume_imbalance_pct_for_long_rot: float | None = None,
        max_vwap_slope_for_short_rot: float | None = None,
        max_vwap_slope_for_long_rot: float | None = None,
    ):
        """Initialize the weather imbalance strategy.

        Tuning guide (most important first):

        Entry strictness / signal quality:
        - lookback_hours: Rolling window used for regime classification.
            Larger -> smoother/slower signals; smaller -> faster/noisier signals.
        - imbalance_below_1sd_threshold: Minimum % of bars below -1sd to call Imb. Down.
            Larger -> stricter downside imbalance requirement.
        - imbalance_up_below_mean_cap: Max % bars below mean for Imb. Up.
            Smaller -> stricter upward imbalance requirement.
        - imbalance_down_above_mean_cap: Max % bars above mean for Imb. Down.
            Smaller -> stricter bearish condition.
        - use_vwap_slope_filter: Enables VWAP trend confirmation.
        - vwap_slope_mode: "raw" (price units/bar), "scaled" (normalized),
            or "angle" (degrees of normalized slope).
        - vwap_slope_value_per_point: Normalization factor applied before
            scaled/angle conversion, similar to Sierra Chart value-per-point.
        - vwap_slope_scale: Optional multiplier for scaled/angle output.
        - vwap_slope_lookback: Lookback for slope regression in BARS (not hours).
        - max_vwap_slope_for_short: Short-entry slope ceiling (entry requires slope <= this value).
            More negative -> stricter downward trend requirement.
        - max_vwap_slope: Backward-compatible alias for short slope ceiling.
        - min_vwap_slope_for_long: Long-entry slope floor (entry requires slope >= this value).
            If None, falls back to short slope ceiling for backward compatibility.
        - use_vwap_volume_imbalance_filter: Enables VWAP volume-imbalance gate.
        - max_vwap_volume_imbalance_pct_for_short: Short-entry imbalance ceiling.
        - max_vwap_volume_imbalance_pct: Backward-compatible alias for short imbalance ceiling.
        - min_vwap_volume_imbalance_pct_for_long: Long-entry imbalance floor.
            If None, falls back to short imbalance ceiling for backward compatibility.
        - vwap_volume_imbalance_lookback: Lookback in UPDATE BARS for imbalance.
            If None, uses vwap_slope_lookback.

        Buy-volume delta filter:
        - buy_volume: optional DataFrame containing per-bar "yes" and "no"
            buy-volume columns (e.g. "market__yes"/"market__no").
        - use_buy_cvd_filter / max_buy_cvd_for_short: short gate, requiring
            cumulative (yes - no) to be below threshold.
        - min_buy_cvd_for_long: long gate, requiring cumulative (yes - no) to
            be above threshold. If None, falls back to max_buy_cvd_for_short.

        Sell-volume delta filter:
        - sell_volume: optional DataFrame containing per-bar "yes" and "no"
            sell-volume columns (e.g. "market__yes"/"market__no").
        - use_sell_cvd_filter / min_sell_cvd_for_short: short gate, requiring
            cumulative (yes - no) to be above threshold.
        - max_sell_cvd_for_long: long gate, requiring cumulative (yes - no) to
            be below threshold. If None, falls back to min_sell_cvd_for_short.

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
        self.side = str(side or "short").lower()
        if self.side not in {"short", "long"}:
            raise ValueError("side must be either 'short' or 'long'")

        # New dual-side controls default to both enabled unless explicitly set.
        self.allow_longs = True if allow_longs is None else bool(allow_longs)
        self.allow_shorts = True if allow_shorts is None else bool(allow_shorts)

        # Keep legacy side-only behavior compatible when only side is specified.
        if allow_longs is None and allow_shorts is None and side is not None:
            if self.side == "long":
                self.allow_longs = True
                self.allow_shorts = False
            else:
                self.allow_longs = False
                self.allow_shorts = True

        if not self.allow_longs and not self.allow_shorts:
            raise ValueError("At least one of allow_longs/allow_shorts must be True")

        self.exit_mode = exit_mode
        self.max_concurrent_positions = max_concurrent_positions
        self.apply_global_position_cap = bool(apply_global_position_cap)

        self.mean_reversion_window = int(mean_reversion_window)
        self.imbalance_above_mean_threshold = float(imbalance_above_mean_threshold)
        self.imbalance_above_1sd_threshold = float(imbalance_above_1sd_threshold)
        self.imbalance_below_1sd_threshold = float(imbalance_below_1sd_threshold)
        self.imbalance_up_below_mean_cap = float(imbalance_up_below_mean_cap)
        self.imbalance_down_above_mean_cap = float(imbalance_down_above_mean_cap)
        self.balanced_within_1sd_threshold = float(balanced_within_1sd_threshold)
        self.mean_reversion_threshold = float(mean_reversion_threshold)

        self.buy_volume = buy_volume if buy_volume is not None else pd.DataFrame()
        self.sell_volume = sell_volume if sell_volume is not None else pd.DataFrame()
        self._buy_delta_cache: dict[str, tuple[str | None, str | None]] = {}
        self._sell_delta_cache: dict[str, tuple[str | None, str | None]] = {}
        self.track_delta_history = bool(track_delta_history)

        # history of (index, asset, delta) whenever ``_buy_delta`` is called
        # during weight generation; useful for tests and debugging.
        self.buy_delta_history: list[tuple[int, str, float]] = []
        self.sell_delta_history: list[tuple[int, str, float]] = []

        self.use_buy_cvd_filter = bool(use_buy_cvd_filter)
        self.max_buy_cvd_for_short = float(max_buy_cvd_for_short)
        self.min_buy_cvd_for_long = (
            float(min_buy_cvd_for_long)
            if min_buy_cvd_for_long is not None
            else self.max_buy_cvd_for_short
        )
        self.use_sell_cvd_filter = bool(use_sell_cvd_filter)
        self.min_sell_cvd_for_short = float(min_sell_cvd_for_short)
        self.max_sell_cvd_for_long = (
            float(max_sell_cvd_for_long)
            if max_sell_cvd_for_long is not None
            else self.min_sell_cvd_for_short
        )
        self.use_buy_cvd_3sd_gate = bool(use_buy_cvd_3sd_gate)
        self.use_stop_loss = bool(use_stop_loss)
        self.stop_loss_mode = str(stop_loss_mode)
        self.sl_entry_band_offset = int(sl_entry_band_offset)
        self.use_take_profit = bool(use_take_profit)
        self.use_trailing_stop = bool(use_trailing_stop)
        # Preferred config keys are absolute prices per side:
        # take_profit_price_short / take_profit_price_long.
        # Keep legacy threshold keys as fallbacks for backward compatibility.
        if take_profit_price_threshold is not None:
            tp_short = float(take_profit_price_threshold)
            tp_long = float(take_profit_price_threshold)
        else:
            if take_profit_price_threshold_short is not None:
                tp_short = float(take_profit_price_threshold_short)
            else:
                tp_short = float(take_profit_price_short)
            if take_profit_price_threshold_long is not None:
                tp_long = float(take_profit_price_threshold_long)
            else:
                tp_long = float(take_profit_price_long)
        self.take_profit_price_short = tp_short
        self.take_profit_price_long = tp_long
        self.min_sell_price = None if min_sell_price is None else float(min_sell_price)
        self.max_buy_price = None if max_buy_price is None else float(max_buy_price)

        # Market regime mode
        self.market_regime_mode = str(market_regime_mode)
        if self.market_regime_mode not in {"imbalance", "rotational"}:
            raise ValueError("market_regime_mode must be 'imbalance' or 'rotational'")

        # Rotational entry configuration
        self.rotational_price_source = str(rotational_price_source)
        if self.rotational_price_source not in {"highlow", "open", "close"}:
            raise ValueError(
                "rotational_price_source must be one of: highlow, open, close"
            )
        self.rotational_entry_band = int(rotational_entry_band)
        _be_exit = rotational_exit_offset
        if isinstance(_be_exit, str):
            if _be_exit != "mean":
                raise ValueError("rotational_exit_offset string must be 'mean'")
            self.rotational_exit_offset: int | str = "mean"
        else:
            _be_exit_int = int(_be_exit)
            if _be_exit_int not in {1, 2, 3, 4, 5, 6}:
                raise ValueError("rotational_exit_offset must be 1, 2, 3, 4, 5, 6 or 'mean'")
            self.rotational_exit_offset = _be_exit_int
        self.rotational_window_hours = float(rotational_window_hours)
        self.rotational_regimes: list[str] = (
            list(rotational_regimes) if rotational_regimes is not None
            else ["Balanced", "Rotational"]
        )
        self.market_end_time = (
            pd.Timestamp(market_end_time) if market_end_time is not None else None
        )

        self.use_vwap_slope_filter = bool(use_vwap_slope_filter)
        self.use_vwap_volume_imbalance_filter = bool(use_vwap_volume_imbalance_filter)

        # --- Volume imbalance thresholds ---
        # Imbalance mode: resolve canonical min_* from new or old param name.
        _imb_vol_short = (
            float(min_vwap_volume_imbalance_pct_for_short)
            if min_vwap_volume_imbalance_pct_for_short is not None
            else (
                float(max_vwap_volume_imbalance_pct_for_short)
                if max_vwap_volume_imbalance_pct_for_short is not None
                else float(max_vwap_volume_imbalance_pct)
            )
        )
        self.min_vwap_volume_imbalance_pct_for_short = _imb_vol_short
        # keep legacy attribute names for compatibility with existing callers
        self.max_vwap_volume_imbalance_pct_for_short = _imb_vol_short
        self.max_vwap_volume_imbalance_pct = _imb_vol_short
        self.min_vwap_volume_imbalance_pct_for_long = (
            float(min_vwap_volume_imbalance_pct_for_long)
            if min_vwap_volume_imbalance_pct_for_long is not None
            else _imb_vol_short
        )
        # Rotational mode volume imbalance ceilings
        self.max_vwap_volume_imbalance_pct_for_short_rot = (
            float(max_vwap_volume_imbalance_pct_for_short_rot)
            if max_vwap_volume_imbalance_pct_for_short_rot is not None
            else None
        )
        self.max_vwap_volume_imbalance_pct_for_long_rot = (
            float(max_vwap_volume_imbalance_pct_for_long_rot)
            if max_vwap_volume_imbalance_pct_for_long_rot is not None
            else None
        )

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
        # --- VWAP slope thresholds ---
        # Imbalance mode: resolve canonical min_* from new or old param name.
        _imb_slope_short = (
            float(min_vwap_slope_for_short)
            if min_vwap_slope_for_short is not None
            else (
                float(max_vwap_slope_for_short)
                if max_vwap_slope_for_short is not None
                else float(max_vwap_slope)
            )
        )
        self.min_vwap_slope_for_short = _imb_slope_short
        # keep legacy attribute names for compatibility with existing callers
        self.max_vwap_slope_for_short = _imb_slope_short
        self.max_vwap_slope = _imb_slope_short
        self.min_vwap_slope_for_long = (
            float(min_vwap_slope_for_long)
            if min_vwap_slope_for_long is not None
            else (
                float(max_vwap_slope_for_long)
                if max_vwap_slope_for_long is not None
                else _imb_slope_short
            )
        )
        # Rotational mode slope ceilings
        self.max_vwap_slope_for_short_rot = (
            float(max_vwap_slope_for_short_rot)
            if max_vwap_slope_for_short_rot is not None
            else None
        )
        self.max_vwap_slope_for_long_rot = (
            float(max_vwap_slope_for_long_rot)
            if max_vwap_slope_for_long_rot is not None
            else None
        )

        _vwap = vwap if vwap is not None else pd.DataFrame()
        _volume = volume if volume is not None else pd.DataFrame()
        _high = high if high is not None else pd.DataFrame()
        _low = low if low is not None else pd.DataFrame()
        _open = open_ if open_ is not None else pd.DataFrame() # open is a reserved keyword, so we use open_ in the signature and rename it here for indicator construction

        # keep high/low/open on the instance for intrabar and rotational checks
        self._high = _high
        self._low = _low
        self._open = _open

        _sd_bands = SdBands(
            pricing_method=pricing_method,
            high=_high,
            low=_low,
            open_=_open,
        )
        self._sd_bands = _sd_bands
        _vwap_indicator = Vwap(
            volume=_volume,
            pricing_method=pricing_method,
            high=_high,
            low=_low,
            open_=_open,
        )
        needs_cvd = bool(
            self.use_buy_cvd_3sd_gate
            or self.use_buy_cvd_filter
            or self.use_sell_cvd_filter
            or self.track_delta_history
        )
        needs_sd_bands = bool(self.use_market_regime or self.use_stop_loss or self.use_trailing_stop)
        if self.use_vwap_volume_imbalance_filter:
            needs_sd_bands = True
        if self.market_regime_mode == "rotational":
            needs_sd_bands = True
        needs_vwap = bool(
            self.use_vwap_slope_filter
            or self.use_vwap_volume_imbalance_filter
            or self.use_stop_loss
            or self.use_trailing_stop
        )

        self.indicator_defs = []
        if needs_sd_bands:
            self.indicator_defs.append(_sd_bands)

        _cum_buy_delta = None
        _cum_sell_delta = None
        if needs_cvd:
            _cum_buy_delta = CumulativeYesNoDelta(
                volume_df=buy_volume,
                name="cum_buy_delta",
                open_=_open,
                high=_high,
                low=_low,
                dollar_weighted=True,
            )
            _cum_sell_delta = CumulativeYesNoDelta(
                volume_df=sell_volume,
                name="cum_sell_delta",
                open_=_open,
                high=_high,
                low=_low,
                dollar_weighted=True,
            )
            self.indicator_defs.extend([_cum_buy_delta, _cum_sell_delta])
            if self.use_buy_cvd_3sd_gate and _cum_buy_delta is not None:
                self.indicator_defs.append(
                    CvdSdThreshold(cum_delta_indicator=_cum_buy_delta, name="_cvd_buy_sd_thr")
                )
        self._cum_buy_delta_indicator = _cum_buy_delta
        self._cum_sell_delta_indicator = _cum_sell_delta

        if needs_vwap:
            self.indicator_defs.append(_vwap_indicator)

        if self.use_vwap_slope_filter:
            self.indicator_defs.extend(
                [
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
                ]
            )

        if self.use_vwap_volume_imbalance_filter:
            self.indicator_defs.append(
                VwapVolumeImbalance(
                    volume=_volume,
                    sd_bands=_sd_bands,
                    lookback=self.vwap_volume_imbalance_lookback,
                )
            )

        if self.use_market_regime:
            self.indicator_defs.extend(
                [
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
            )

        if self.use_stop_loss or self.use_trailing_stop:
            self.indicator_defs.append(
                StopLossIndicator(
                    sd_bands=_sd_bands,
                    vwap_indicator=_vwap_indicator,
                    mode=self.stop_loss_mode,
                    band_offset=self.sl_entry_band_offset,
                )
            )

        if self.use_take_profit:
            self.indicator_defs.append(
                TakeProfitIndicator(
                    price_short=self.take_profit_price_short,
                    price_long=self.take_profit_price_long,
                )
            )

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

    def _in_rotational_window(self, current_ts: pd.Timestamp) -> bool:
        """Return True if *current_ts* is before the rotational cutoff.

        Entries are allowed from t_0 until market_end_time − window_hours.
        """
        if self.market_end_time is None:
            return False
        cutoff = self.market_end_time - pd.Timedelta(hours=self.rotational_window_hours)
        return current_ts <= cutoff

    def _rotational_price_source(
        self, index: int, asset: str, prices_row: pd.Series,
        check_side: str = "upper",
    ) -> float:
        """Resolve the price used for ±Nsd band-touch checks.

        When *rotational_price_source* is ``"highlow"``, the bar high is
        returned for upper-band checks and the bar low for lower-band checks.
        """
        src = self.rotational_price_source
        try:
            if src == "close":
                return float(prices_row.get(asset, float("nan")))
            if src == "highlow":
                if check_side == "upper":
                    return float(self._high.iloc[index][asset])
                return float(self._low.iloc[index][asset])
            if src == "open":
                return float(self._open.iloc[index][asset])
        except Exception:
            return float("nan")
        return float("nan")

    def _rotational_exit_target_index(self, entry_band: int, entry_side: str) -> int:
        """Compute the exit-target band index for a rotational position.

        Parameters
        ----------
        entry_band : int
            Absolute band index at entry (positive for short at +Nsd,
            positive for long at -Nsd where we store absolute value).
        entry_side : str
            "short" or "long".

        Returns
        -------
        int
            Band index in [-3, 3] used as the exit target.
        """
        offset = self.rotational_exit_offset
        if offset == "mean":
            return 0
        offset_int = int(offset)
        if entry_side == "short":
            # Entered short at +entry_band; exit target moves toward mean.
            target = entry_band - offset_int
        else:
            # Entered long at -entry_band; exit target moves toward mean.
            target = -(entry_band - offset_int)
        return int(max(-3, min(3, target)))

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
            and (100.0 - above_mean) < self.imbalance_up_below_mean_cap
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
    def _trade_return(entry_price: float, current_price: float, side: str) -> float:
        if entry_price <= 0:
            return 0.0
        if side == "long":
            return float(current_price / entry_price - 1.0)
        return float(entry_price / current_price - 1.0)

    def _price_band_position(
        self,
        prices: pd.DataFrame,
        index: int,
        asset: str,
        price_val: float,
    ) -> float:
        if not np.isfinite(price_val):
            return float("nan")
        if not hasattr(self, "_sd_bands") or self._sd_bands is None:
            return float("nan")
        try:
            ts = prices.index[index]
            bands = self._sd_bands.band_slice(asset, ts, ts)
            if bands is None or bands.empty:
                return float("nan")
            idx = nearest_band_index(bands.iloc[-1], float(price_val))
            if idx is None:
                return float("nan")
            return float(idx)
        except Exception:
            return float("nan")

    def _cvd_band_position(
        self,
        indicator: CumulativeYesNoDelta | None,
        asset: str,
        index: int,
        fallback_value: float,
    ) -> float:
        if indicator is None:
            return float("nan")
        try:
            return float(indicator.band_position_at(asset, index, fallback_value=float(fallback_value)))
        except Exception:
            return float("nan")

    def _should_exit(self, asset: str, current_price: float, index: int, prices: pd.DataFrame) -> str | None:
        # Only skip exit checks when neither the exit_mode requests TP/SL
        # nor the boolean gates are enabled in the profile,
        # AND no rotational exit target is stored on the position.
        has_rotational_exit = (
            asset in self._positions
            and self._positions[asset].get("rotational_exit_target_index") is not None
        )
        if not has_rotational_exit and not (
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
        side = str(position.get("side", "short"))

        # --- Rotational exit: dynamic band-based take-profit ---
        be_target_idx = position.get("rotational_exit_target_index")
        if be_target_idx is not None:
            try:
                ts = prices.index[index]
                label = StopLossIndicator._label_for_band_index(int(cast(int, be_target_idx)))
                band_val = self._sd_bands.band_value_at(asset, ts, label)
                if band_val is not None and math.isfinite(band_val):
                    if side == "short":
                        # Check intrabar low, then close
                        low_val = None
                        try:
                            if hasattr(self, "_low") and asset in self._low.columns:
                                low_val = float(self._low.iloc[index][asset])
                        except Exception:
                            low_val = None
                        check_val = low_val if (low_val is not None and np.isfinite(low_val)) else current_price
                        if check_val <= band_val:
                            return "rotational_tp"
                    else:
                        # Check intrabar high, then close
                        high_val = None
                        try:
                            if hasattr(self, "_high") and asset in self._high.columns:
                                high_val = float(self._high.iloc[index][asset])
                        except Exception:
                            high_val = None
                        check_val = high_val if (high_val is not None and np.isfinite(high_val)) else current_price
                        if check_val >= band_val:
                            return "rotational_tp"
            except Exception:
                pass

        # Per-position TP/SL price checks (prefer absolute price thresholds)
        tp_price = position.get("tp_price", float("nan"))
        try:
            tp_val = float(cast(float, tp_price))
        except Exception:
            tp_val = float("nan")
        if math.isfinite(tp_val):
            # check intrabar extremes when available, otherwise use close
            if side == "long":
                high_val = None
                try:
                    if hasattr(self, "_high") and asset in self._high.columns:
                        high_val = float(self._high.iloc[index][asset])
                except Exception:
                    high_val = None
                if high_val is not None and np.isfinite(high_val):
                    if high_val >= tp_val:
                        return "take_profit"
                else:
                    if current_price >= tp_val:
                        return "take_profit"
            else:
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
            # check intrabar extremes when available, otherwise use close
            if side == "long":
                low_val = None
                try:
                    if hasattr(self, "_low") and asset in self._low.columns:
                        low_val = float(self._low.iloc[index][asset])
                except Exception:
                    low_val = None
                if low_val is not None and np.isfinite(low_val):
                    if low_val <= stop_val:
                        return "stop_loss"
                else:
                    if current_price <= stop_val:
                        return "stop_loss"
            else:
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
        price_band_position_entry = float(
            cast(float, position.get("price_band_position_entry", float("nan")))
        )
        buy_cvd_band_position_entry = float(
            cast(float, position.get("buy_cvd_band_position_entry", float("nan")))
        )
        sell_cvd_band_position_entry = float(
            cast(float, position.get("sell_cvd_band_position_entry", float("nan")))
        )
        stop_price = float(cast(float, position.get("stop_price", float("nan"))))
        tp_price = float(cast(float, position.get("tp_price", float("nan"))))
        side = str(position.get("side", "short"))
        # Prefer intrabar fill price when available for stop_loss / take_profit
        fill_price = float(current_price)
        try:
            if reason == "stop_loss":
                if side == "long" and hasattr(self, "_low") and asset in self._low.columns:
                    lo = float(self._low.iloc[current_index][asset])
                    if np.isfinite(lo) and lo > 0:
                        fill_price = lo
                if side == "short" and hasattr(self, "_high") and asset in self._high.columns:
                    hi = float(self._high.iloc[current_index][asset])
                    if np.isfinite(hi) and hi > 0:
                        fill_price = hi
        except Exception:
            pass
        try:
            if reason == "take_profit":
                if side == "long" and hasattr(self, "_high") and asset in self._high.columns:
                    hi = float(self._high.iloc[current_index][asset])
                    if np.isfinite(hi) and hi > 0:
                        fill_price = hi
                if side == "short" and hasattr(self, "_low") and asset in self._low.columns:
                    lo = float(self._low.iloc[current_index][asset])
                    if np.isfinite(lo) and lo > 0:
                        fill_price = lo
        except Exception:
            pass
        try:
            if reason == "rotational_tp":
                if side == "long" and hasattr(self, "_high") and asset in self._high.columns:
                    hi = float(self._high.iloc[current_index][asset])
                    if np.isfinite(hi) and hi > 0:
                        fill_price = hi
                if side == "short" and hasattr(self, "_low") and asset in self._low.columns:
                    lo = float(self._low.iloc[current_index][asset])
                    if np.isfinite(lo) and lo > 0:
                        fill_price = lo
        except Exception:
            pass

        price_band_position_exit = self._price_band_position(
            prices=prices,
            index=current_index,
            asset=asset,
            price_val=float(current_price),
        )
        buy_cvd_band_position_exit = self._cvd_band_position(
            indicator=getattr(self, "_cum_buy_delta_indicator", None),
            asset=asset,
            index=current_index,
            fallback_value=buy_cvd,
        )
        sell_cvd_band_position_exit = self._cvd_band_position(
            indicator=getattr(self, "_cum_sell_delta_indicator", None),
            asset=asset,
            index=current_index,
            fallback_value=sell_cvd,
        )

        pnl = self._trade_return(entry_price, fill_price, side)
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
                "side": side,
                "regime": position.get("regime"),
                "confidence": confidence,
                "vwap_slope": vwap_slope,
                "vwap_slope_raw": vwap_slope_raw,
                "vwap_volume_imbalance_pct": vwap_volume_imbalance_pct,
                "buy_cvd": buy_cvd,
                "sell_cvd": sell_cvd,
                "price_band_position_entry": price_band_position_entry,
                "price_band_position_exit": price_band_position_exit,
                "buy_cvd_band_position_entry": buy_cvd_band_position_entry,
                "buy_cvd_band_position_exit": buy_cvd_band_position_exit,
                "sell_cvd_band_position_entry": sell_cvd_band_position_entry,
                "sell_cvd_band_position_exit": sell_cvd_band_position_exit,
                "stop_price": stop_price,
                "tp_price": tp_price,
                "rotational_exit_target_index": position.get("rotational_exit_target_index"),
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
        assets_set = set(assets)
        current_ts = pd.Timestamp(prices.index[index])
        prices_row = prices.iloc[index]
        inds = getattr(self, "indicators", {})

        def _indicator_asset_value(indicator_value: object, asset: str) -> float:
            if indicator_value is None:
                return float("nan")
            try:
                return float(cast(pd.Series, indicator_value).get(asset, float("nan")))
            except Exception:
                return float("nan")

        def _read_cum(cum_name: str, asset: str) -> float:
            series = inds.get(cum_name)
            if series is None:
                return float("nan")
            try:
                return float(cast(pd.DataFrame, series).iloc[index].get(asset, float("nan")))
            except Exception:
                try:
                    return float(cast(pd.Series, series).get(asset, float("nan")))
                except Exception:
                    return float("nan")

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
                    if asset not in assets_set:
                        continue
                    try:
                        cur_price = float(prices_row.get(asset, float("nan")))
                    except Exception:
                        continue
                    if np.isnan(cur_price) or cur_price <= 0:
                        continue
                    try:
                        pos_side = str(self._positions.get(asset, {}).get("side", "short"))
                        new_stop = sl_obj.stop_price_for_entry(
                            prices,
                            index,
                            asset,
                            cur_price,
                            side=pos_side,
                        )
                        if np.isfinite(new_stop):
                            self._positions[asset]["stop_price"] = float(new_stop)
                    except Exception:
                        continue

        # Exit checks on existing positions
        to_close = []
        for asset in list(self._positions.keys()):
            if asset not in assets_set:
                to_close.append((asset, "asset_missing"))
                continue
            current_price = float(prices_row.get(asset, float("nan")))
            if np.isnan(current_price) or current_price <= 0:
                continue
            exit_reason = self._should_exit(asset, current_price, index, prices)
            if exit_reason is not None:
                to_close.append((asset, exit_reason))

        for asset, reason in to_close:
            if asset not in assets_set:
                self._positions.pop(asset, None)
                continue
            close_price = float(prices_row.get(asset, float("nan")))
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
        ranked_entries: list[tuple[str, str, str, float, float, float, float, float, float]] = []
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
        prices_window = prices.iloc[start : index + 1]

        vwap_slope_values = inds.get("vwap_slope")
        vwap_slope_raw_values = inds.get("vwap_slope_raw")
        vwap_imbalance_values = inds.get("vwap_volume_imbalance")

        for asset in assets:
            buy_delta = _read_cum("cum_buy_delta", asset)
            sell_delta = _read_cum("cum_sell_delta", asset)
            if self.track_delta_history:
                self.buy_delta_history.append((index, asset, buy_delta))
                self.sell_delta_history.append((index, asset, sell_delta))

            asset_window = prices_window[asset].dropna().to_numpy(dtype=float)
            if len(asset_window) < max(2, self.lookback // 2):
                continue

            if self.use_market_regime:
                # if the regime window isn't yet complete, skip asset entirely
                if need_full_window:
                    continue
                regime, confidence = self._classify_regime(asset)

                if self.market_regime_mode == "rotational":
                    # Rotational mode: only enter on rotational touch in
                    # Balanced/Rotational regimes within the temporal window.
                    if regime not in self.rotational_regimes:
                        continue
                    if not self._in_rotational_window(current_ts):
                        continue
                    ts = prices.index[index]
                    upper_label = f"+{self.rotational_entry_band}sd"
                    lower_label = f"-{self.rotational_entry_band}sd"
                    upper_val = self._sd_bands.band_value_at(asset, ts, upper_label)
                    lower_val = self._sd_bands.band_value_at(asset, ts, lower_label)
                    upper_price = self._rotational_price_source(index, asset, prices_row, check_side="upper")
                    lower_price = self._rotational_price_source(index, asset, prices_row, check_side="lower")
                    if upper_val is not None and math.isfinite(upper_price) and upper_price >= upper_val:
                        entry_side = "short"
                    elif lower_val is not None and math.isfinite(lower_price) and lower_price <= lower_val:
                        entry_side = "long"
                    else:
                        continue  # price not at extreme band
                    entry_regime = f"BandExtreme: {regime}"
                    # override regime variable for downstream position tagging
                    regime = entry_regime
                else:
                    # Imbalance mode (default): Imb. Down -> short, Imb. Up -> long
                    if regime == "Imb. Down":
                        entry_side = "short"
                    elif regime == "Imb. Up":
                        entry_side = "long"
                    else:
                        continue

                if entry_side == "long" and not self.allow_longs:
                    continue
                if entry_side == "short" and not self.allow_shorts:
                    continue
            else:
                # market‑regime filtering disabled – allow everything through
                # still supply a confidence for ranking purposes
                regime = self.entry_regime
                confidence = 0.5
                entry_side = self.side
                if entry_side == "long" and not self.allow_longs:
                    continue
                if entry_side == "short" and not self.allow_shorts:
                    continue

            # --- Rotational price-vs-mean gate ---
            # In rotational mode only allow shorts above mean, longs below mean.
            if self.market_regime_mode == "rotational" and hasattr(self, "_sd_bands"):
                try:
                    _mean_val = self._sd_bands.band_value_at(asset, prices.index[index], "mean")
                    if _mean_val is not None and math.isfinite(_mean_val):
                        _cur_price = float(prices_row.get(asset, float("nan")))
                        if math.isfinite(_cur_price):
                            if entry_side == "short" and _cur_price < _mean_val:
                                continue
                            if entry_side == "long" and _cur_price > _mean_val:
                                continue
                except Exception:
                    pass

            slope = 0.0
            raw_slope = 0.0
            if self.use_vwap_slope_filter:
                slope = _indicator_asset_value(vwap_slope_values, asset)
                raw_slope = _indicator_asset_value(vwap_slope_raw_values, asset)
                if not math.isfinite(slope):
                    slope = 0.0
                if not math.isfinite(raw_slope):
                    raw_slope = 0.0
                if self.market_regime_mode == "rotational":
                    # Rotational: slope must confirm mean-reversion direction.
                    # Shorts expect price rising (positive slope) into resistance;
                    # skip if slope is below threshold (too bearish).
                    # Longs expect price falling (negative slope) into support;
                    # skip if slope is above threshold (too bullish).
                    if entry_side == "short":
                        rot_thr = self.max_vwap_slope_for_short_rot
                        if rot_thr is not None and slope < rot_thr:
                            continue
                    else:
                        rot_thr = self.max_vwap_slope_for_long_rot
                        if rot_thr is not None and slope > rot_thr:
                            continue
                else:
                    # Imbalance: min thresholds require sufficient trend extremity
                    if entry_side == "short":
                        if slope > self.min_vwap_slope_for_short:
                            continue
                    else:
                        if slope < self.min_vwap_slope_for_long:
                            continue

            imbalance_pct = float("nan")
            if self.use_vwap_volume_imbalance_filter:
                imbalance_pct = _indicator_asset_value(vwap_imbalance_values, asset)
                if np.isnan(imbalance_pct):
                    continue
                if self.market_regime_mode == "rotational":
                    # Rotational: volume imbalance must confirm mean-reversion.
                    # Shorts: skip if imbalance too negative (sell-side heavy,
                    # trend continuation rather than reversion).
                    # Longs: skip if imbalance too positive.
                    if entry_side == "short":
                        rot_thr = self.max_vwap_volume_imbalance_pct_for_short_rot
                        if rot_thr is not None and imbalance_pct < rot_thr:
                            continue
                    else:
                        rot_thr = self.max_vwap_volume_imbalance_pct_for_long_rot
                        if rot_thr is not None and imbalance_pct > rot_thr:
                            continue
                else:
                    # Imbalance: min thresholds require sufficient extremity
                    if entry_side == "short":
                        if imbalance_pct > self.min_vwap_volume_imbalance_pct_for_short:
                            continue
                    else:
                        if imbalance_pct < self.min_vwap_volume_imbalance_pct_for_long:
                            continue

            if self.use_buy_cvd_3sd_gate:
                buy_band_idx = self._cvd_band_position(
                    indicator=getattr(self, "_cum_buy_delta_indicator", None),
                    asset=asset,
                    index=index,
                    fallback_value=buy_delta,
                )

                # current default setting
                short_cvd_band_gate = -3.0
                long_cvd_band_gate = 3.0

                if entry_side == "short":
                    if (not math.isfinite(buy_band_idx)) or buy_band_idx > short_cvd_band_gate:
                        logger.debug(
                            "%s @ %s: buy CVD band idx = %s must be <= %s – skipping",
                            asset,
                            current_ts,
                            buy_band_idx,
                            short_cvd_band_gate,
                        )
                        continue
                else:
                    if (not math.isfinite(buy_band_idx)) or buy_band_idx < long_cvd_band_gate:
                        logger.debug(
                            "%s @ %s: buy CVD band idx = %s must be >= %s – skipping",
                            asset,
                            current_ts,
                            buy_band_idx,
                            long_cvd_band_gate,
                        )
                        continue
            if self.use_buy_cvd_filter:
                if entry_side == "short":
                    if (not math.isfinite(buy_delta)) or buy_delta >= self.max_buy_cvd_for_short:
                        logger.debug(
                            "%s @ %s: buy CVD = %s must be < max_buy_cvd_for_short=%s – skipping",
                            asset,
                            current_ts,
                            buy_delta,
                            self.max_buy_cvd_for_short,
                        )
                        continue
                else:
                    if (not math.isfinite(buy_delta)) or buy_delta <= self.min_buy_cvd_for_long:
                        logger.debug(
                            "%s @ %s: buy CVD = %s must be > min_buy_cvd_for_long=%s – skipping",
                            asset,
                            current_ts,
                            buy_delta,
                            self.min_buy_cvd_for_long,
                        )
                        continue

            if self.use_sell_cvd_filter:
                if entry_side == "short":
                    if (not math.isfinite(sell_delta)) or sell_delta <= self.min_sell_cvd_for_short:
                        logger.debug(
                            "%s @ %s: sell CVD = %s must be > min_sell_cvd_for_short=%s – skipping",
                            asset,
                            current_ts,
                            sell_delta,
                            self.min_sell_cvd_for_short,
                        )
                        continue
                else:
                    if (not math.isfinite(sell_delta)) or sell_delta >= self.max_sell_cvd_for_long:
                        logger.debug(
                            "%s @ %s: sell CVD = %s must be < max_sell_cvd_for_long=%s – skipping",
                            asset,
                            current_ts,
                            sell_delta,
                            self.max_sell_cvd_for_long,
                        )
                        continue

            # asset passed all filters – add to ranking list
            ranked_entries.append(
                (
                    asset,
                    entry_side,
                    regime,
                    confidence,
                    slope,
                    raw_slope,
                    imbalance_pct,
                    buy_delta,
                    sell_delta,
                )
            )

        ranked_entries.sort(key=lambda x: x[3], reverse=True)

        if self.apply_global_position_cap and self.max_concurrent_positions is not None:
            capacity = max(0, self.max_concurrent_positions - len(self._positions))
            ranked_entries = ranked_entries[:capacity]

        for (
            asset,
            entry_side,
            entry_regime,
            confidence,
            slope,
            raw_slope,
            imbalance_pct,
            buy_delta_entry,
            sell_delta_entry,
        ) in ranked_entries:
            if asset in self._positions:
                continue
            entry_price = float(prices_row.get(asset, float("nan")))
            # For rotational (band-touch) entries fill at the band value,
            # not the bar close.  The bar's high/low touched the band intrabar
            # so the earliest realistic fill is at the band level itself.
            if str(entry_regime).startswith("BandExtreme:"):
                try:
                    _ts_entry = prices.index[index]
                    if entry_side == "short":
                        _fill_label = f"+{self.rotational_entry_band}sd"
                    else:
                        _fill_label = f"-{self.rotational_entry_band}sd"
                    _band_fill = self._sd_bands.band_value_at(asset, _ts_entry, _fill_label)
                    if _band_fill is not None and math.isfinite(_band_fill) and _band_fill > 0:
                        entry_price = float(_band_fill)
                except Exception:
                    pass
            if np.isnan(entry_price) or entry_price <= 0:
                continue
            # Enforce min/max price guards to avoid re-entries at near-zero prices
            if entry_side == "short" and self.min_sell_price is not None:
                if entry_price < self.min_sell_price:
                    continue
            if entry_side == "long" and self.max_buy_price is not None:
                if entry_price > self.max_buy_price:
                    continue
            logger.debug(
                "opening %s %s @ %s price=%.5f buy_cvd=%.2f sell_cvd=%.2f",
                entry_side,
                asset,
                current_ts,
                entry_price,
                buy_delta_entry,
                sell_delta_entry,
            )
            # compute stop_price and tp_price at entry time using indicator helpers
            stop_price = float("nan")
            sl_obj = None
            if self.use_stop_loss:
                # find StopLossIndicator instance
                for ind in getattr(self, "indicator_defs", []):
                    try:
                        if isinstance(ind, StopLossIndicator):
                            sl_obj = ind
                            break
                    except Exception:
                        continue
                if sl_obj is not None:
                    try:
                        stop_price = sl_obj.stop_price_for_entry(
                            prices,
                            index,
                            asset,
                            entry_price,
                            side=entry_side,
                        )
                    except Exception:
                        stop_price = float("nan")

            tp_price = float("nan")
            tp_obj = None
            if self.use_take_profit:
                for ind in getattr(self, "indicator_defs", []):
                    try:
                        if isinstance(ind, TakeProfitIndicator):
                            tp_obj = ind
                            break
                    except Exception:
                        continue
                if tp_obj is not None:
                    try:
                        tp_price = tp_obj.threshold_for_entry(
                            prices,
                            index,
                            asset,
                            entry_price,
                            side=entry_side,
                        )
                    except Exception:
                        tp_price = float("nan")

            self._positions[asset] = {
                "entry_price": entry_price,
                "entry_index": int(index),
                "entry_time": prices.index[index],
                "side": entry_side,
                "regime": entry_regime,
                "confidence": float(confidence),
                "vwap_slope": float(slope),
                "vwap_slope_raw": float(raw_slope),
                "vwap_volume_imbalance_pct": float(imbalance_pct),
                "buy_cvd": float(buy_delta_entry),
                "sell_cvd": float(sell_delta_entry),
                "price_band_position_entry": self._price_band_position(
                    prices=prices,
                    index=index,
                    asset=asset,
                    price_val=entry_price,
                ),
                "buy_cvd_band_position_entry": self._cvd_band_position(
                    indicator=getattr(self, "_cum_buy_delta_indicator", None),
                    asset=asset,
                    index=index,
                    fallback_value=float(buy_delta_entry),
                ),
                "sell_cvd_band_position_entry": self._cvd_band_position(
                    indicator=getattr(self, "_cum_sell_delta_indicator", None),
                    asset=asset,
                    index=index,
                    fallback_value=float(sell_delta_entry),
                ),
                "stop_price": stop_price,
                "tp_price": tp_price,
                "rotational_exit_target_index": None,
            }

            # Compute rotational exit target for rotational entries
            if str(entry_regime).startswith("BandExtreme:"):
                be_exit_idx = self._rotational_exit_target_index(
                    entry_band=self.rotational_entry_band,
                    entry_side=entry_side,
                )
                self._positions[asset]["rotational_exit_target_index"] = be_exit_idx

        # Build side-aware weight vector
        weights = np.zeros(len(assets), dtype=float)
        open_assets = [a for a in assets if a in self._positions]
        if not open_assets:
            return weights

        unit_weight = 1.0 / len(open_assets)
        for i, asset in enumerate(assets):
            if asset in self._positions:
                pos_side = str(self._positions[asset].get("side", "short"))
                weights[i] = unit_weight if pos_side == "long" else -unit_weight

        # allow_cash retained for future behavior; currently unused because
        # short weights are normalized over active shorts only.
        _ = self.allow_cash
        return weights
