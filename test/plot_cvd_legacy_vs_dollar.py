#!/usr/bin/env python3
"""Plot legacy vs dollarized cumulative delta side-by-side for one event.

This script compares CVD computed from raw yes/no volumes (legacy) against
CVD computed from dollarized volumes where price multiplier is
(O + H + L + C) / 4 per bar.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratlab.config import RESULTS_DIR
from stratlab.strategy.indicators import CumulativeYesNoDelta, sd_bands_expanding
from strategies.weather_backtest import load_event_ohlcv_resampled


def _pick_market(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    explicit_market: str | None,
) -> str:
    if explicit_market:
        if explicit_market not in prices.columns:
            raise ValueError(f"market '{explicit_market}' not found in prices columns")
        return explicit_market

    valid_cols = [c for c in prices.columns if c in volume.columns]
    if valid_cols:
        vol_sum = volume[valid_cols].sum(axis=0).sort_values(ascending=False)
        return str(vol_sum.index[0])

    if len(prices.columns) == 0:
        raise ValueError("no market columns available")
    return str(prices.columns[0])


def _compute_cvd_series(
    prices: pd.DataFrame,
    buy_volume: pd.DataFrame | None,
    sell_volume: pd.DataFrame | None,
    open_: pd.DataFrame | None,
    high: pd.DataFrame | None,
    low: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    legacy_buy = CumulativeYesNoDelta(volume_df=buy_volume, name="legacy_buy")
    legacy_sell = CumulativeYesNoDelta(volume_df=sell_volume, name="legacy_sell")

    dollar_buy = CumulativeYesNoDelta(
        volume_df=buy_volume,
        name="dollar_buy",
        open_=open_,
        high=high,
        low=low,
        close=prices,
        dollar_weighted=True,
    )
    dollar_sell = CumulativeYesNoDelta(
        volume_df=sell_volume,
        name="dollar_sell",
        open_=open_,
        high=high,
        low=low,
        close=prices,
        dollar_weighted=True,
    )

    legacy_buy_df = legacy_buy.compute_series(prices, returns)
    legacy_sell_df = legacy_sell.compute_series(prices, returns)
    dollar_buy_df = dollar_buy.compute_series(prices, returns)
    dollar_sell_df = dollar_sell.compute_series(prices, returns)

    return legacy_buy_df, legacy_sell_df, dollar_buy_df, dollar_sell_df


def _plot_side_by_side(
    index: pd.Index,
    price: pd.Series,
    legacy_buy: pd.Series,
    legacy_sell: pd.Series,
    dollar_buy: pd.Series,
    dollar_sell: pd.Series,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16, 10),
        sharex="col",
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]},
    )

    axes[0, 0].plot(index, price.values, color="black", linewidth=1.2, label="close")
    axes[0, 0].set_title("Legacy input (raw volume)")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[0, 0].grid(alpha=0.25)

    axes[1, 0].plot(index, legacy_buy.values, color="#228B22", linewidth=1.2, label="cum buy delta")
    axes[1, 0].axhline(0.0, color="#777", linewidth=0.8)
    axes[1, 0].legend(loc="upper left", fontsize=8)
    axes[1, 0].grid(alpha=0.25)

    axes[2, 0].plot(index, legacy_sell.values, color="#8A2BE2", linewidth=1.2, label="cum sell delta")
    axes[2, 0].axhline(0.0, color="#777", linewidth=0.8)
    axes[2, 0].legend(loc="upper left", fontsize=8)
    axes[2, 0].grid(alpha=0.25)

    axes[0, 1].plot(index, price.values, color="black", linewidth=1.2, label="close")
    axes[0, 1].set_title("Dollarized input ((O+H+L+C)/4 * volume)")
    axes[0, 1].legend(loc="upper left", fontsize=8)
    axes[0, 1].grid(alpha=0.25)

    axes[1, 1].plot(index, dollar_buy.values, color="#228B22", linewidth=1.2, label="cum buy delta")
    axes[1, 1].axhline(0.0, color="#777", linewidth=0.8)
    axes[1, 1].legend(loc="upper left", fontsize=8)
    axes[1, 1].grid(alpha=0.25)

    axes[2, 1].plot(index, dollar_sell.values, color="#8A2BE2", linewidth=1.2, label="cum sell delta")
    axes[2, 1].axhline(0.0, color="#777", linewidth=0.8)
    axes[2, 1].legend(loc="upper left", fontsize=8)
    axes[2, 1].grid(alpha=0.25)

    for r in range(3):
        for c in range(2):
            axes[r, c].tick_params(axis="x", labelrotation=35, labelsize=8)
            axes[r, c].tick_params(axis="y", labelsize=8)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _mean_price_from_ohlc(
    close: pd.Series,
    open_: pd.Series | None,
    high: pd.Series | None,
    low: pd.Series | None,
) -> pd.Series:
    if open_ is None or high is None or low is None:
        return close.astype(float)
    out = (open_.astype(float) + high.astype(float) + low.astype(float) + close.astype(float)) / 4.0
    return out.where(np.isfinite(out), close.astype(float))


def _coerce_finite_series(series: pd.Series, fallback: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    fb = pd.to_numeric(fallback, errors="coerce")
    out = out.where(np.isfinite(out), fb)
    out = out.ffill().bfill()
    return out.astype(float)


def _session_vwap_from_bar_vwap(
    bar_vwap: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Compute session VWAP from bar VWAP and volume from t=0.

    Equivalent to cumulative weighted form:
    cumsum(bar_vwap * volume) / cumsum(volume),
    implemented via an online weighted mean update for stability.
    """
    vol = pd.to_numeric(volume, errors="coerce").fillna(0.0).astype(float)

    out = np.full(len(bar_vwap), np.nan, dtype=float)
    running_weight = 0.0
    running_mean = np.nan

    pb = pd.to_numeric(bar_vwap, errors="coerce").astype(float).to_numpy()
    vv = vol.to_numpy()
    for i in range(len(pb)):
        p = pb[i]
        w = vv[i]
        if not np.isfinite(w) or w < 0.0:
            w = 0.0

        if np.isfinite(p) and w > 0.0:
            new_weight = running_weight + w
            if new_weight > 0.0:
                if np.isfinite(running_mean):
                    running_mean = running_mean + (w / new_weight) * (p - running_mean)
                else:
                    running_mean = p
                running_weight = new_weight

        out[i] = running_mean if np.isfinite(running_mean) else np.nan

    return pd.Series(out, index=bar_vwap.index, dtype=float)


def _plot_sd_bands_source_comparison(
    index: pd.Index,
    close_series: pd.Series,
    mean_price_series: pd.Series,
    vwap_series: pd.Series,
    mean_price_bands: pd.DataFrame,
    vwap_bands: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 8),
        sharex="col",
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )

    axes[0, 0].plot(index, close_series.values, color="black", linewidth=1.1, label="close")
    axes[0, 0].plot(index, mean_price_bands["mean"].values, color="#1f77b4", linewidth=1.0, label="band mean")
    axes[0, 0].plot(index, mean_price_bands["+1sd"].values, color="#7f7f7f", linestyle="--", linewidth=0.9, label="+1sd")
    axes[0, 0].plot(index, mean_price_bands["-1sd"].values, color="#7f7f7f", linestyle="--", linewidth=0.9, label="-1sd")
    axes[0, 0].set_title("SD bands source: mean price (OHLC/4) | price shown: close")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(index, close_series.values, color="black", linewidth=1.1, label="close")
    axes[0, 1].plot(index, vwap_bands["mean"].values, color="#1f77b4", linewidth=1.0, label="band mean")
    axes[0, 1].plot(index, vwap_bands["+1sd"].values, color="#7f7f7f", linestyle="--", linewidth=0.9, label="+1sd")
    axes[0, 1].plot(index, vwap_bands["-1sd"].values, color="#7f7f7f", linestyle="--", linewidth=0.9, label="-1sd")
    axes[0, 1].set_title("SD bands source: VWAP | price shown: close")
    axes[0, 1].legend(loc="upper left", fontsize=8)
    axes[0, 1].grid(alpha=0.25)

    mean_spread = close_series - mean_price_bands["mean"]
    vwap_spread = close_series - vwap_bands["mean"]
    axes[1, 0].plot(index, mean_spread.values, color="#d62728", linewidth=1.0, label="close - band mean")
    axes[1, 0].axhline(0.0, color="#777", linewidth=0.8)
    axes[1, 0].legend(loc="upper left", fontsize=8)
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(index, vwap_spread.values, color="#d62728", linewidth=1.0, label="close - band mean")
    axes[1, 1].axhline(0.0, color="#777", linewidth=0.8)
    axes[1, 1].legend(loc="upper left", fontsize=8)
    axes[1, 1].grid(alpha=0.25)

    for r in range(2):
        for c in range(2):
            axes[r, c].tick_params(axis="x", labelrotation=35, labelsize=8)
            axes[r, c].tick_params(axis="y", labelsize=8)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy vs dollarized CVD for a weather event")
    parser.add_argument(
        "--event-slug",
        default="highest-temperature-in-atlanta-on-february-22-2026",
        help="Event slug to plot",
    )
    parser.add_argument(
        "--resample-rule",
        default="30min",
        help="Resample rule, e.g. 5min, 15min, 30min; set 'none' for native bars",
    )
    parser.add_argument(
        "--prefer-outcome",
        choices=["yes", "no"],
        default="yes",
        help="Outcome preference for close-price matrix",
    )
    parser.add_argument(
        "--market",
        default=None,
        help="Exact market column to compare; default picks highest-volume visible market",
    )
    parser.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "weather_imbalance_test" / "cvd_compare"),
        help="Output directory",
    )
    args = parser.parse_args()

    resample_rule = None if str(args.resample_rule).lower() == "none" else args.resample_rule

    prices, vwap, volume, buy_volume, sell_volume, high, low, open_ = load_event_ohlcv_resampled(
        args.event_slug,
        resample_rule=resample_rule,
        prefer_outcome=args.prefer_outcome,
    )

    # CVD needs paired yes/no columns; load unfiltered buy/sell matrices when available.
    try:
        (
            _prices_full,
            _vwap_full,
            _volume_full,
            buy_volume_full,
            sell_volume_full,
            _high_full,
            _low_full,
            _open_full,
        ) = load_event_ohlcv_resampled(
            args.event_slug,
            resample_rule=resample_rule,
            prefer_outcome=None,
        )
        if buy_volume_full is not None:
            buy_volume = buy_volume_full
        if sell_volume_full is not None:
            sell_volume = sell_volume_full
    except Exception:
        pass

    market = _pick_market(prices=prices, volume=volume, explicit_market=args.market)

    legacy_buy_df, legacy_sell_df, dollar_buy_df, dollar_sell_df = _compute_cvd_series(
        prices=prices,
        buy_volume=buy_volume,
        sell_volume=sell_volume,
        open_=open_,
        high=high,
        low=low,
    )

    comp = pd.DataFrame(
        {
            "close": prices[market],
            "legacy_cum_buy_delta": legacy_buy_df[market],
            "legacy_cum_sell_delta": legacy_sell_df[market],
            "dollar_cum_buy_delta": dollar_buy_df[market],
            "dollar_cum_sell_delta": dollar_sell_df[market],
        },
        index=prices.index,
    )
    comp["buy_delta_diff"] = comp["dollar_cum_buy_delta"] - comp["legacy_cum_buy_delta"]
    comp["sell_delta_diff"] = comp["dollar_cum_sell_delta"] - comp["legacy_cum_sell_delta"]

    out_dir = Path(args.out_dir)
    safe_slug = args.event_slug.replace("/", "_")
    safe_market = market.replace("/", "_")
    out_png = out_dir / f"{safe_slug}__{safe_market}__legacy_vs_dollar.png"
    out_csv = out_dir / f"{safe_slug}__{safe_market}__legacy_vs_dollar.csv"
    out_bands_png = out_dir / f"{safe_slug}__{safe_market}__sd_bands_vwap_vs_mean_price.png"
    out_bands_csv = out_dir / f"{safe_slug}__{safe_market}__sd_bands_vwap_vs_mean_price.csv"

    title = (
        f"{args.event_slug} | market={market} | resample={resample_rule or 'native'} | "
        f"outcome={args.prefer_outcome}"
    )
    _plot_side_by_side(
        index=comp.index,
        price=comp["close"],
        legacy_buy=comp["legacy_cum_buy_delta"],
        legacy_sell=comp["legacy_cum_sell_delta"],
        dollar_buy=comp["dollar_cum_buy_delta"],
        dollar_sell=comp["dollar_cum_sell_delta"],
        title=title,
        out_path=out_png,
    )

    comp.to_csv(out_csv)

    open_s = open_[market] if open_ is not None and market in open_.columns else None
    high_s = high[market] if high is not None and market in high.columns else None
    low_s = low[market] if low is not None and market in low.columns else None

    mean_price_series = _mean_price_from_ohlc(
        close=prices[market],
        open_=open_s,
        high=high_s,
        low=low_s,
    )
    if vwap is not None and market in vwap.columns:
        raw_vwap_series = pd.to_numeric(vwap[market].reindex(prices.index), errors="coerce")
    else:
        raw_vwap_series = pd.Series(np.nan, index=prices.index, dtype=float)

    volume_s = volume[market] if market in volume.columns else pd.Series(0.0, index=prices.index, dtype=float)
    ohlc_avg_series = _mean_price_from_ohlc(
        close=prices[market],
        open_=open_s,
        high=high_s,
        low=low_s,
    )
    # Bar VWAP input: raw feed where available, OHLC/4 fallback when sparse.
    bar_vwap_input = _coerce_finite_series(raw_vwap_series, ohlc_avg_series)
    # Session VWAP: cumulative from t=0 using bar_vwap_input and volume weights.
    session_vwap_series = _session_vwap_from_bar_vwap(bar_vwap_input, volume_s)
    vwap_series = _coerce_finite_series(session_vwap_series, bar_vwap_input)

    mean_price_bands = sd_bands_expanding(mean_price_series)
    vwap_bands = sd_bands_expanding(vwap_series)

    bands_comp = pd.DataFrame(
        {
            "mean_price": mean_price_series,
            "mean_price_band_mean": mean_price_bands["mean"],
            "mean_price_band_plus_1sd": mean_price_bands["+1sd"],
            "mean_price_band_minus_1sd": mean_price_bands["-1sd"],
            "vwap_bar_input": bar_vwap_input,
            "vwap_session": vwap_series,
            "vwap_band_mean": vwap_bands["mean"],
            "vwap_band_plus_1sd": vwap_bands["+1sd"],
            "vwap_band_minus_1sd": vwap_bands["-1sd"],
        },
        index=prices.index,
    )
    bands_comp["band_mean_diff"] = bands_comp["vwap_band_mean"] - bands_comp["mean_price_band_mean"]
    bands_comp["plus_1sd_diff"] = bands_comp["vwap_band_plus_1sd"] - bands_comp["mean_price_band_plus_1sd"]
    bands_comp["minus_1sd_diff"] = bands_comp["vwap_band_minus_1sd"] - bands_comp["mean_price_band_minus_1sd"]
    bands_comp.to_csv(out_bands_csv)

    bands_title = (
        f"{args.event_slug} | market={market} | resample={resample_rule or 'native'} | "
        "SD bands source comparison"
    )
    _plot_sd_bands_source_comparison(
        index=prices.index,
        close_series=prices[market],
        mean_price_series=mean_price_series,
        vwap_series=vwap_series,
        mean_price_bands=mean_price_bands,
        vwap_bands=vwap_bands,
        title=bands_title,
        out_path=out_bands_png,
    )

    print(f"Selected market: {market}")
    print(f"Rows: {len(comp)}")
    print(
        f"Non-zero bars: "
        f"legacy_buy={(comp['legacy_cum_buy_delta'].diff().fillna(0.0) != 0.0).sum()}, "
        f"dollar_buy={(comp['dollar_cum_buy_delta'].diff().fillna(0.0) != 0.0).sum()}, "
        f"legacy_sell={(comp['legacy_cum_sell_delta'].diff().fillna(0.0) != 0.0).sum()}, "
        f"dollar_sell={(comp['dollar_cum_sell_delta'].diff().fillna(0.0) != 0.0).sum()}"
    )
    print(
        "VWAP coverage: "
        f"raw_finite={int(np.isfinite(raw_vwap_series.to_numpy()).sum())}/{len(raw_vwap_series)}, "
        f"bar_input_finite={int(np.isfinite(bar_vwap_input.to_numpy()).sum())}/{len(bar_vwap_input)}, "
        f"session_finite={int(np.isfinite(session_vwap_series.to_numpy()).sum())}/{len(session_vwap_series)}"
    )
    print(f"Saved plot: {out_png}")
    print(f"Saved data: {out_csv}")
    print(f"Saved SD-bands plot: {out_bands_png}")
    print(f"Saved SD-bands data: {out_bands_csv}")
    print(
        "Last-row snapshot: "
        f"legacy_buy={comp['legacy_cum_buy_delta'].iloc[-1]:.6f}, "
        f"dollar_buy={comp['dollar_cum_buy_delta'].iloc[-1]:.6f}, "
        f"legacy_sell={comp['legacy_cum_sell_delta'].iloc[-1]:.6f}, "
        f"dollar_sell={comp['dollar_cum_sell_delta'].iloc[-1]:.6f}"
    )


if __name__ == "__main__":
    main()
