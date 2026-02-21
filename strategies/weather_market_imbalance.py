import numpy as np
import pandas as pd
from typing import Any, cast

from stratlab.strategy.base import Strategy
from stratlab.strategy.indicators import BandPosition, MeanReversion, VwapSlope, VwapVolumeImbalance


PROFILE_PRESETS: dict[str, dict[str, object]] = {
    # Conservative = fewer but stronger signals.
    # - Longer lookback, stricter imbalance thresholds, stricter VWAP down-slope gate.
    # - Good when you want cleaner entries and can tolerate lower trade count.
    "conservative": {
        "lookback_hours": 12.0,
        "max_concurrent_positions": None,
        "imbalance_below_1sd_threshold": 40.0,
        "imbalance_down_above_mean_cap": 35.0,
        "mean_reversion_window": 8,
        "mean_reversion_threshold": 0.55,
        "balanced_within_1sd_threshold": 75.0,
        "use_vwap_slope_filter": True,
        "use_vwap_volume_imbalance_filter": True,
        "max_vwap_volume_imbalance_pct": -1.0,
        "vwap_volume_imbalance_lookback": 30,
        "vwap_slope_mode": "scaled",
        "vwap_slope_value_per_point": 1e-4,
        "vwap_slope_scale": 1.0,
        "vwap_slope_lookback": 30,
        "max_vwap_slope": -1.0,
        "take_profit": 0.35,
        "stop_loss": 0.12,
    },
    # Balanced = default test profile.
    # - Medium lookback and medium thresholds.
    # - Reasonable trade frequency with moderate strictness.
    "balanced": {
        "lookback_hours": 6.0,
        "max_concurrent_positions": None,
        "imbalance_below_1sd_threshold": 40.0,
        "imbalance_down_above_mean_cap": 10.0,
        "mean_reversion_window": 5,
        "mean_reversion_threshold": 0.5,
        "balanced_within_1sd_threshold": 70.0,
        "use_vwap_slope_filter": True,
        "use_vwap_volume_imbalance_filter": True,
        "max_vwap_volume_imbalance_pct": -1.0,
        "vwap_volume_imbalance_lookback": 60,
        "vwap_slope_mode": "scaled",
        "vwap_slope_value_per_point": 1e-3,
        "vwap_slope_scale": 1.0,
        "vwap_slope_lookback": 60,
        "max_vwap_slope": -0.5,
        "take_profit": 0.5,
        "stop_loss": 0.20,
    },
    # Aggressive = fastest/loosest entries.
    # - Short lookback, looser imbalance thresholds, permissive slope gate.
    # - Higher trade count, typically noisier entries.
    "aggressive": {
        "lookback_hours": 3.0,
        "max_concurrent_positions": None,
        "imbalance_below_1sd_threshold": 40.0,
        "imbalance_down_above_mean_cap": 30.0,
        "mean_reversion_window": 3,
        "mean_reversion_threshold": 0.45,
        "balanced_within_1sd_threshold": 65.0,
        "use_vwap_slope_filter": True,
        "use_vwap_volume_imbalance_filter": True,
        "max_vwap_volume_imbalance_pct": -1.0,
        "vwap_volume_imbalance_lookback": 20,
        "vwap_slope_mode": "scaled",
        "vwap_slope_value_per_point": 1e-4,
        "vwap_slope_scale": 1.0,
        "vwap_slope_lookback": 15,
        "max_vwap_slope": -0.18,
        "take_profit": 0.25,
        "stop_loss": 0.25,
    },
}


class WeatherMarketImbalanceStrategy(Strategy):
    """Short markets classified as imbalanced down in a rolling window.

    The regime logic mirrors `market_regime_analysis.py` thresholds but is
    evaluated on a rolling lookback window to be deployable in live trading.
    """

    def __init__(
        self,
        lookback: int = 1,
        lookback_hours: float | None = 6.0,
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
        vwap_slope_mode: str = "scaled",
        vwap_slope_value_per_point: float = 1e-4,
        vwap_slope_scale: float = 1.0,
        vwap_slope_lookback: int = 15, # in bars, not hours; should be less than lookback if lookback_hours is set
        max_vwap_slope: float = -2.0,
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

        self._positions: dict[str, dict[str, object]] = {}
        self.trade_log: list[dict[str, object]] = []

        _vwap = vwap if vwap is not None else pd.DataFrame()
        _volume = volume if volume is not None else pd.DataFrame()
        self.indicator_defs = [
            VwapSlope(
                vwap=_vwap,
                volume=_volume,
                lookback=self.vwap_slope_lookback,
                mode=self.vwap_slope_mode,
                value_per_point=self.vwap_slope_value_per_point,
                scale=self.vwap_slope_scale,
            ),
            VwapSlope(
                vwap=_vwap,
                volume=_volume,
                lookback=self.vwap_slope_lookback,
                mode="raw",
                name="vwap_slope_raw",
            ),
            VwapVolumeImbalance(
                vwap=_vwap,
                volume=_volume,
                lookback=self.vwap_volume_imbalance_lookback,
            ),
            BandPosition(
                lookback_hours=lookback_hours,
                lookback_bars=int(lookback) if lookback_hours is None else None,
            ),
            MeanReversion(
                window=self.mean_reversion_window,
                lookback_hours=lookback_hours,
                lookback_bars=int(lookback) if lookback_hours is None else None,
            ),
        ]

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
        if self.lookback_hours is not None:
            start_ts = current_ts - pd.Timedelta(hours=float(self.lookback_hours))
            index_series = pd.DatetimeIndex(prices.index)
            start = int(index_series.searchsorted(start_ts, side="left"))
            if start >= index:
                return np.zeros(len(assets), dtype=float)
            # Require full lookback horizon before allowing entries.
            if pd.Timestamp(prices.index[start]) > start_ts:
                return np.zeros(len(assets), dtype=float)
        else:
            start = max(0, index - self.lookback + 1)

        for asset in assets:
            asset_window = prices.iloc[start: index + 1][asset].dropna().to_numpy(dtype=float)
            if len(asset_window) < max(2, self.lookback // 2):
                continue

            regime, confidence = self._classify_regime(asset)
            if regime != self.entry_regime:
                continue

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
            self._positions[asset] = {
                "entry_price": entry_price,
                "entry_index": int(index),
                "entry_time": prices.index[index],
                "regime": self.entry_regime,
                "confidence": float(confidence),
                "vwap_slope": float(slope),
                "vwap_slope_raw": float(raw_slope),
                "vwap_volume_imbalance_pct": float(imbalance_pct),
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
