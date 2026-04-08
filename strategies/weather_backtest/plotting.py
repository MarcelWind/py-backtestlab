from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import re
from matplotlib.lines import Line2D

from stratlab.strategy.indicators import (
    SdBands,
    VwapSlope,
    VolumeImbalance,
    MeanReversion,
    sd_bands_expanding,
    analyze_band_position_vs_reference,
    detect_mean_reversion_vs_reference,
    market_regimes,
    select_yes_no_columns,
)
from stratlab.report.plot import (
    draw_price_sd_volume_panel,
    draw_regime_overlays,
    draw_trade_markers,
    draw_volume_profile_inset,
    draw_volume_imbalance_panel,
    draw_vwap_slope_panel,
    draw_mean_reversion_panel,
)

from .constants import WINDOW_COLORS, WINDOW_HOURS


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _compact_event_slug(event_slug: str) -> str:
    """Compact event slug for display."""
    short = event_slug
    short = short.replace("highest-temperature-in-", "")
    short = short.replace("-on-", " | ")
    short = short.replace("-", " ")
    return short


# ---------------------------------------------------------------------------
# Per-market data preparation
# ---------------------------------------------------------------------------

def _prepare_market_data(
    market: str,
    prices: pd.DataFrame,
    vwap: pd.DataFrame | None,
    volume: pd.DataFrame | None,
    suffix_re,
    sdbands: dict[str, pd.DataFrame] | None = None,
):
    """Extract and pre-compute all data structures needed to draw a market's panels.

    Returns None if the market's price series is empty. Otherwise returns:
    (series, mdf, bar_width_days, bands_full, vwap_s, vol_s, display_market,
     price_arr, timestamps)
    """
    series = prices[market].dropna()
    display_market = (
        suffix_re.sub(r"\1", str(market))
        if suffix_re is not None
        else re.sub(r"__(yes|no)$", "", str(market), flags=re.IGNORECASE)
    )
    if series.empty:
        return None

    mdf = (
        pd.DataFrame({"timestamp": pd.to_datetime(series.index), "price": series.values})
        .sort_values("timestamp")
    )

    try:
        idx_dt = pd.DatetimeIndex(series.index)
        if len(idx_dt) >= 2:
            median_delta = idx_dt.to_series().diff().median().to_timedelta64()
            bar_width_days = float(pd.Timedelta(median_delta) / pd.Timedelta(days=1))
        else:
            bar_width_days = 15.0 / 1440.0
    except Exception:
        bar_width_days = 15.0 / 1440.0

    price_arr = mdf["price"].to_numpy(dtype=float)
    timestamps = mdf["timestamp"].to_numpy()
    if sdbands is not None and market in sdbands:
        bands_df = sdbands[market].copy()
        # Drop duplicate timestamps if they exist (keeping first occurrence)
        if bands_df.index.duplicated().any():
            bands_df = bands_df[~bands_df.index.duplicated(keep='first')]
        bands_full = bands_df.reindex(pd.to_datetime(series.index)).copy()
    else:
        bands_full = sd_bands_expanding(price_arr, timestamps).copy()
    bands_full["price"] = price_arr
    bands_full["timestamp"] = timestamps
    band_cols = ["timestamp", "price", "mean", "-3sd", "-2sd", "-1sd", "+1sd", "+2sd", "+3sd"]
    bands_full = bands_full[[c for c in band_cols if c in bands_full.columns]]

    vwap_s = vwap[market].reindex(series.index) if vwap is not None and market in vwap.columns else None
    vol_s = (
        volume[market].reindex(series.index).fillna(0.0)
        if volume is not None and market in volume.columns
        else None
    )

    return series, mdf, bar_width_days, bands_full, vwap_s, vol_s, display_market, price_arr, timestamps


# ---------------------------------------------------------------------------
# Panel drawing helpers
# ---------------------------------------------------------------------------

def _entry_markers(market_trades: pd.DataFrame, series: pd.Series):
    """Return (entry_times, entry_values) for scatter markers, or (None, None) if no trades."""
    if market_trades.empty:
        return None, None
    times = pd.to_datetime(market_trades["entry_time"])
    return times, series.reindex(times).to_numpy(dtype=float)


def _overlay_entry_markers(
    ax,
    market_trades: pd.DataFrame,
    series: pd.Series,
    zorder: int = 7,
) -> None:
    """Overlay side-aware entry markers using the shared _entry_markers helper."""
    entry_times, entry_vals = _entry_markers(market_trades, series)
    if entry_times is None or entry_vals is None:
        return
    valid = np.isfinite(entry_vals)
    if not valid.any():
        return
    valid_times = pd.DatetimeIndex(entry_times)[valid]
    valid_vals = np.asarray(entry_vals, dtype=float)[valid]

    side_series = (
        market_trades.get("side", pd.Series("short", index=market_trades.index))
        .astype(str)
        .str.lower()
    )
    side_by_time = pd.Series(side_series.values, index=pd.to_datetime(market_trades["entry_time"]))
    sides = side_by_time.reindex(valid_times).fillna("short")

    long_mask = sides.eq("long").to_numpy(dtype=bool)
    short_mask = ~long_mask

    if long_mask.any():
        ax.scatter(
            valid_times[long_mask],
            valid_vals[long_mask],
            marker="^",
            s=22,
            color="#2E8B57",
            edgecolors="black",
            linewidths=0.7,
            zorder=zorder,
        )
    if short_mask.any():
        ax.scatter(
            valid_times[short_mask],
            valid_vals[short_mask],
            marker="v",
            s=22,
            color="#FF0000",
            edgecolors="black",
            linewidths=0.7,
            zorder=zorder,
        )

def _draw_price_panel(
    ax,
    series: pd.Series,
    mdf: pd.DataFrame,
    bands_full: pd.DataFrame,
    vwap_s: pd.Series | None,
    vol_s: pd.Series | None,
    bar_width_days: float,
    price_arr: np.ndarray,
    timestamps: np.ndarray,
    market_trades: pd.DataFrame,
    display_market: str,
    highs: pd.Series | None = None,
    lows: pd.Series | None = None,
    opens: pd.Series | None = None,
) -> None:
    """Draw panel 0: price + SD bands + VWAP + volume + regime overlays + trade markers + title."""
    regime_bar_colors: pd.Series | None = None
    try:
        final_ts = pd.Timestamp(series.index.max())
        regime_bar_colors = pd.Series(index=series.index, dtype="object")
        prev_hours = 0
        for wh in sorted(WINDOW_HOURS):
            start_time = final_ts - pd.Timedelta(hours=wh)
            end_time = final_ts - pd.Timedelta(hours=prev_hours)
            if prev_hours == 0:
                mask = (series.index >= start_time) & (series.index <= end_time)
            else:
                mask = (series.index >= start_time) & (series.index < end_time)
            regime_bar_colors.loc[mask] = WINDOW_COLORS.get(wh, "gray")
            prev_hours = wh
    except Exception:
        regime_bar_colors = None

    # If intrabar highs/lows are provided, render candlesticks. Prefer an
    # explicit `opens` series if supplied; otherwise fall back to prior-close.
    if highs is not None and lows is not None:
        if opens is not None:
            try:
                o_s = opens.reindex(series.index)
                if hasattr(o_s, "columns"):
                    o_s = o_s.iloc[:, 0]
            except Exception:
                o_s = None
        else:
            o_s = None

        # fallback to prior close when explicit opens missing
        if o_s is None:
            try:
                o_s = series.shift(1).reindex(series.index)
                o_s.iloc[0] = series.iloc[0]
            except Exception:
                o_s = series.copy()

        draw_price_sd_volume_panel(
            ax,
            series,
            bands_full,
            vwap_s=vwap_s,
            vol_s=vol_s,
            bar_width_days=bar_width_days,
            use_candles=True,
            opens=o_s,
            highs=highs,
            lows=lows,
            candle_kwargs={
                "up_color": "#2ca02c",
                "down_color": "#d62728",
                "bar_colors": regime_bar_colors,
            },
        )
    else:
        draw_price_sd_volume_panel(ax, series, bands_full, vwap_s=vwap_s, vol_s=vol_s,
                                   bar_width_days=bar_width_days)

    window_hours_dict = {wh: f"T-{wh}h" for wh in WINDOW_HOURS}
    windows_str = draw_regime_overlays(
        ax,
        mdf,
        bands_full,
        window_hours_dict,
        WINDOW_COLORS,
        draw_lines=False,
    )
    # draw trade markers, prefer intrabar highs/lows when available
    trade_info = draw_trade_markers(ax, market_trades, highs=highs, lows=lows)

    pos = analyze_band_position_vs_reference(price_arr, bands_full, timestamps)
    mean_rev_full = detect_mean_reversion_vs_reference(price_arr, bands_full, timestamps, 5)
    full_regime, full_conf = market_regimes(pos, mean_rev_full)

    ax.set_title(
        f"{display_market}  |  {full_regime} ({full_conf:.2f})  |  {windows_str}  |  {trade_info}",
        fontsize=7, loc="left", pad=2,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(alpha=0.3)


_VOL_DELTA_STYLES = {
    "buy":  {"title": "buy vol Δ (yes − no)", "up_label": "▲ YES buys", "dn_label": "▼ NO buys",  "cum_color": "#00008B"},
    "sell": {"title": "sell vol Δ (yes − no)", "up_label": "▲ YES sells", "dn_label": "▼ NO sells", "cum_color": "#4B0082"},
}


def _draw_vol_delta_panel(
    ax,
    vol_df: pd.DataFrame | None,
    series: pd.Series,
    market_trades: pd.DataFrame,
    base_name: str,
    market: str,
    suffix_re,
    bar_width_days: float,
    xlim,
    kind: str = "buy",
) -> bool:
    """Draw a yes−no volume delta panel. Returns True if data was plotted.

    Parameters
    ----------
    kind:
        "buy" or "sell" — controls title text, annotation labels, and cumulative line color.
    """
    style = _VOL_DELTA_STYLES[kind]
    no_col, yes_col = select_yes_no_columns(vol_df, base_name, market, suffix_re)
    if vol_df is None or yes_col is None or no_col is None:
        return False

    yes_ser = vol_df[yes_col].reindex(series.index).fillna(0.0)
    no_ser = vol_df[no_col].reindex(series.index).fillna(0.0)
    delta = yes_ser - no_ser

    colors = ["#00FF00" if v >= 0 else "#FF0000" for v in delta.values]
    ax.bar(delta.index, delta.values, width=bar_width_days, color=colors,
           alpha=0.75, align="center", edgecolor="none")
    ax.set_title(f"{style['title']}  |  line = cumulative Δ", fontsize=7, loc="left", pad=2)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.18)
    ax.axhline(0.0, color="#888888", linestyle="-", linewidth=0.8, alpha=0.6, zorder=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
    ax.set_xlim(xlim)
    ax.text(0.01, 0.97, style["up_label"], transform=ax.transAxes, fontsize=7,
            va="top", ha="left", color="#00FF00", alpha=0.85)
    ax.text(0.01, 0.03, style["dn_label"], transform=ax.transAxes, fontsize=7,
            va="bottom", ha="left", color="#FF0000", alpha=0.85)

    cum_delta = delta.cumsum()
    if len(cum_delta) > 0:
        cum_delta = cum_delta - cum_delta.iloc[0]
    
    cum_color = style["cum_color"]
    ax_cum = ax.twinx()

    # Compute expanding SD bands for cumulative delta and draw them behind the cum line
    try:
        cum_delta_bands = sd_bands_expanding(cum_delta)
        if cum_delta_bands is not None:
            cum_delta_bands = cum_delta_bands.reindex(cum_delta.index)
            has_mean = "mean" in cum_delta_bands.columns
            has_p1 = "+1sd" in cum_delta_bands.columns and "-1sd" in cum_delta_bands.columns
            has_p2 = "+2sd" in cum_delta_bands.columns and "-2sd" in cum_delta_bands.columns
            has_p3 = "+3sd" in cum_delta_bands.columns and "-3sd" in cum_delta_bands.columns

            # shaded ±3sd, ±2sd, and ±1sd regions (subtle alpha) and dashed mean line
            if has_p3:
                ax_cum.fill_between(cum_delta_bands.index, cum_delta_bands["-3sd"], cum_delta_bands["+3sd"],
                                    color=cum_color, alpha=0.03, zorder=0)
            if has_p2:
                ax_cum.fill_between(cum_delta_bands.index, cum_delta_bands["-2sd"], cum_delta_bands["+2sd"],
                                    color=cum_color, alpha=0.05, zorder=1)
            if has_p1:
                ax_cum.fill_between(cum_delta_bands.index, cum_delta_bands["-1sd"], cum_delta_bands["+1sd"],
                                    color=cum_color, alpha=0.08, zorder=2)
            if has_mean:
                ax_cum.plot(cum_delta_bands.index, cum_delta_bands["mean"], color=cum_color, linestyle="--",
                            linewidth=0.9, alpha=0.6, zorder=3)
    except Exception:
        # non-fatal: if band computation fails, continue plotting cum line only
        pass

    # plot cumulative delta on top of bands
    ax_cum.plot(cum_delta.index, cum_delta.values, color=cum_color,
                linewidth=1.4, alpha=0.85, zorder=5)
    ax_cum.axhline(0.0, color=cum_color, linestyle=":", linewidth=0.7, alpha=0.4, zorder=1)
    ax_cum.tick_params(axis="y", labelsize=7, labelcolor=cum_color)
    ax_cum.set_ylabel("cum Δ", fontsize=7, color=cum_color)
    ax_cum.set_xlim(xlim)

    _overlay_entry_markers(ax_cum, market_trades, cum_delta, zorder=8)
    return True


def _draw_vol_imbalance_row(
    ax,
    imbalance_df: pd.DataFrame | None,
    market: str,
    market_trades: pd.DataFrame,
    ind_map: dict,
    xlim,
    magnitude_threshold: float | None = None,
) -> None:
    """Draw volume imbalance % panel (panel 3)."""
    if imbalance_df is None or market not in imbalance_df.columns:
        ax.text(0.5, 0.5, "No VWAP data", ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        return
    ratio_pct = imbalance_df[market]
    entry_times, entry_vals = _entry_markers(market_trades, ratio_pct)
    lookback = (
        "session"
        if "volume_imbalance" in ind_map else "?"
    )
    # Extract signal values at rebalance points (index from imbalance_df)
    signal_times = imbalance_df.index
    signal_vals = imbalance_df[market] if market in imbalance_df.columns else None
    draw_volume_imbalance_panel(ax, ratio_pct, entry_times, entry_vals, 
                                signal_times=signal_times, signal_values=signal_vals, 
                                lookback_label=lookback, magnitude_threshold=magnitude_threshold)
    _overlay_entry_markers(ax, market_trades, ratio_pct, zorder=7)
    ax.set_xlim(xlim)


def _draw_vwap_slope_row(
    ax,
    slope_df: pd.DataFrame | None,
    market: str,
    series: pd.Series,
    market_trades: pd.DataFrame,
    vwap: pd.DataFrame | None,
    volume: pd.DataFrame | None,
    slope_ind: VwapSlope | None,
    max_vwap_slope: float | None,
    vwap_slope_mode: str,
    vwap_slope_lookback: int,
    xlim,
    magnitude_threshold: float | None = None,
) -> None:
    """Draw VWAP slope panel (panel 4)."""
    if slope_df is None or market not in slope_df.columns:
        ax.text(0.5, 0.5, "No VWAP data", ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        return
    slope_series = slope_df[market]
    entry_times, entry_vals = _entry_markers(market_trades, slope_series)

    if slope_ind is not None:
        mode_lbl = slope_ind.mode
        if slope_ind.mode != "raw":
            mode_lbl += f", vpp={slope_ind.value_per_point:g}, scale={slope_ind.scale:g}"
        lb_lbl = slope_ind.lookback
    else:
        mode_lbl = vwap_slope_mode
        lb_lbl = vwap_slope_lookback

    upd_pct = 0.0
    if vwap is not None and market in vwap.columns:
        vwap_check = vwap[market].reindex(series.index)
        if volume is not None and market in volume.columns:
            vol_check = volume[market].reindex(series.index).fillna(0.0)
            valid_mask = (vol_check > 0.0) & vwap_check.notna()
        else:
            valid_mask = vwap_check.notna()
        upd_pct = float(valid_mask.mean() * 100.0)

    # Extract signal values at rebalance points (index from slope_df)
    signal_times = slope_df.index
    signal_vals = slope_df[market] if market in slope_df.columns else None
    
    draw_vwap_slope_panel(
        ax, slope_series,
        signal_times=signal_times, signal_values=signal_vals,
        threshold=max_vwap_slope,
        mode_label=mode_lbl, lookback_label=lb_lbl, update_pct=upd_pct,
        magnitude_threshold=magnitude_threshold,
    )
    # Overlay side-aware entry markers (long => up, short => down)
    _overlay_entry_markers(ax, market_trades, slope_series, zorder=8)
    ax.set_xlim(xlim)


def _draw_mean_reversion_row(
    ax,
    mr_df: pd.DataFrame | None,
    market: str,
    series: pd.Series,
    market_trades: pd.DataFrame,
    ind_map: dict,
    mean_reversion_threshold: float | None,
    mean_reversion_window: int,
    xlim,
    magnitude_threshold: float | None = None,
) -> None:
    """Draw mean-reversion score panel (panel 5)."""
    mr_score = (
        mr_df[market]
        if mr_df is not None and market in mr_df.columns
        else pd.Series(np.nan, index=series.index)
    )
    entry_times, entry_vals = _entry_markers(market_trades, mr_score)
    mr_ind = ind_map.get("mean_reversion")
    window_lbl = mr_ind.window if mr_ind is not None else mean_reversion_window
    
    # Extract signal values at rebalance points (index from mr_df)
    signal_times = mr_df.index if mr_df is not None else None
    signal_vals = mr_df[market] if mr_df is not None and market in mr_df.columns else None
    
    draw_mean_reversion_panel(
        ax, mr_score,
        entry_times=entry_times, entry_values=entry_vals,
        signal_times=signal_times, signal_values=signal_vals,
        threshold=mean_reversion_threshold, window_label=window_lbl,
        magnitude_threshold=magnitude_threshold,
    )
    ax.set_xlim(xlim)


# ---------------------------------------------------------------------------
# Market orchestrator
# ---------------------------------------------------------------------------

def _plot_market_panels(
    axes6: np.ndarray,
    market: str,
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    vwap: pd.DataFrame | None,
    volume: pd.DataFrame | None,
    buy_volume: pd.DataFrame | None,
    sell_volume: pd.DataFrame | None,
    slope_df: pd.DataFrame | None,
    imbalance_df: pd.DataFrame | None,
    mr_df: pd.DataFrame | None,
    slope_ind: VwapSlope | None,
    ind_map: dict,
    suffix_re,
    sdbands: dict[str, pd.DataFrame] | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    opens: pd.DataFrame | None = None,
    max_vwap_slope: float | None = None,
    mean_reversion_threshold: float | None = 0.5,
    vwap_slope_mode: str = "raw",
    vwap_slope_lookback: int = 15,
    mean_reversion_window: int = 5,
    signal_magnitude_threshold_imbalance: float | None = None,
    signal_magnitude_threshold_slope: float | None = None,
    signal_magnitude_threshold_meanrev: float | None = None,
) -> None:
    """Orchestrate all 6 panels for a single market into the pre-allocated axes."""
    result = _prepare_market_data(market, prices, vwap, volume, suffix_re, sdbands=sdbands)
    if result is None:
        for a in axes6:
            a.set_visible(False)
        return

    series, mdf, bar_width_days, bands_full, vwap_s, vol_s, display_market, price_arr, timestamps = result
    # prepare intrabar high/low series aligned to this market's index if provided
    high_s = high[market].reindex(series.index) if high is not None and market in high.columns else None
    low_s = low[market].reindex(series.index) if low is not None and market in low.columns else None
    opens_s = opens[market].reindex(series.index) if opens is not None and market in opens.columns else None
    market_trades = trades[trades["asset"] == market] if not trades.empty else pd.DataFrame()
    m_match = suffix_re.match(str(market))
    base_name = m_match.group(1) if m_match else str(market)

    _draw_price_panel(axes6[0], series, mdf, bands_full, vwap_s, vol_s, bar_width_days,
                      price_arr, timestamps, market_trades, display_market,
                      highs=high_s, lows=low_s, opens=opens_s)
    draw_volume_profile_inset(axes6[0], series, vol_s)

    xlim = axes6[0].get_xlim()

    if not _draw_vol_delta_panel(axes6[1], buy_volume, series, market_trades, base_name, str(market),
                                 suffix_re, bar_width_days, xlim, kind="buy"):
        axes6[1].set_visible(False)

    if not _draw_vol_delta_panel(axes6[2], sell_volume, series, market_trades, base_name, str(market),
                                 suffix_re, bar_width_days, xlim, kind="sell"):
        axes6[2].set_visible(False)

    _draw_vol_imbalance_row(axes6[3], imbalance_df, market, market_trades, ind_map, xlim,
                            magnitude_threshold=signal_magnitude_threshold_imbalance)
    _draw_vwap_slope_row(axes6[4], slope_df, market, series, market_trades, vwap, volume,
                         slope_ind, max_vwap_slope, vwap_slope_mode, vwap_slope_lookback, xlim,
                         magnitude_threshold=signal_magnitude_threshold_slope)
    _draw_mean_reversion_row(axes6[5], mr_df, market, series, market_trades, ind_map,
                             mean_reversion_threshold, mean_reversion_window, xlim,
                             magnitude_threshold=signal_magnitude_threshold_meanrev)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_entries_exits(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    strategy_name: str,
    event_slug: str,
    out_path: Path,
    vwap: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    buy_volume: pd.DataFrame | None = None,
    sell_volume: pd.DataFrame | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    opens: pd.DataFrame | None = None,
    indicator_series: dict[str, pd.DataFrame] | None = None,
    indicator_defs: list | None = None,
    # Legacy fallback params — used only when indicator_defs is None:
    vwap_slope_mode: str = "raw",
    vwap_slope_value_per_point: float = 1.0,
    vwap_slope_scale: float = 1.0,
    vwap_slope_lookback: int = 15,
    max_vwap_slope: float | None = None,
    mean_reversion_window: int = 5,
    mean_reversion_threshold: float | None = 0.5,
    # Signal visualization magnitude thresholds:
    signal_magnitude_threshold_imbalance: float | None = None,
    signal_magnitude_threshold_slope: float | None = None,
    signal_magnitude_threshold_meanrev: float | None = None,
) -> None:
    """Plot price/regime plus debug panels per market at dpi=100.

    For each market:
      - Panel 0: price + SD bands + regime overlays + trade markers + volume twin-axis
      - Panel 1: buy vol delta (yes − no) [weather-specific]
      - Panel 2: sell vol delta (yes − no) [weather-specific]
      - Panel 3: volume imbalance %
      - Panel 4: VWAP slope
      - Panel 5: mean-reversion score

    Indicator panels are driven by ``indicator_defs`` (typically
    ``strategy.indicator_defs``). When not provided the function builds
    ``VwapSlope``, ``VolumeImbalance``, and ``MeanReversion`` instances
    from the legacy keyword parameters for backward compatibility.
    """
    suffix_re = re.compile(r"^(.*)__(yes|no)$", re.IGNORECASE)
    markets = list(prices.columns)
    preferred_outcome = "no"
    try:
        from .data_prep import pick_plot_frame

        df_prep = pd.DataFrame()
        plot_df = pick_plot_frame(df_prep, prefer_outcome=preferred_outcome)
        preferred_set = set(plot_df["market"].astype(str).unique())
        markets = [m for m in prices.columns if str(m) in preferred_set]
    except Exception:
        orig_markets = [str(m) for m in markets]
        orig_set = set(orig_markets)
        selected_markets: list[str] = []
        handled: set[str] = set()
        pref = preferred_outcome.lower()
        for m in orig_markets:
            if m in handled:
                continue
            m_match = suffix_re.match(m)
            if m_match:
                base = m_match.group(1)
                yes_name = f"{base}__yes"
                no_name = f"{base}__no"
                if yes_name in orig_set and no_name in orig_set:
                    selected_markets.append(no_name if pref == "no" else yes_name)
                    handled.add(yes_name)
                    handled.add(no_name)
                elif yes_name in orig_set:
                    selected_markets.append(yes_name)
                    handled.add(yes_name)
                    handled.add(no_name)
                elif no_name in orig_set:
                    selected_markets.append(no_name)
                    handled.add(yes_name)
                    handled.add(no_name)
                else:
                    selected_markets.append(m)
                    handled.add(m)
            else:
                selected_markets.append(m)
                handled.add(m)
        markets = selected_markets
    if not markets:
        return

    cols = 2
    rows = (len(markets) + cols - 1) // cols
    _PANEL_RATIOS = [3, 2.0, 2.0, 0.9, 2.2, 1.2]
    _SPACER_RATIO = 0.6  # blank row between market groups; absorbs x-tick labels + next title
    # stride: number of figure rows per market group (6 panels + 1 spacer, except last group)
    stride = 7 if rows > 1 else 6
    total_fig_rows = rows * 6 + max(0, rows - 1)
    height_ratios: list[float] = []
    for r in range(rows):
        height_ratios.extend(_PANEL_RATIOS)
        if r < rows - 1:
            height_ratios.append(_SPACER_RATIO)

    fig, axes = plt.subplots(
        total_fig_rows,
        cols,
        figsize=(16, 13 * rows),
        sharex="col",
        gridspec_kw={"height_ratios": height_ratios},
    )

    if isinstance(axes, np.ndarray):
        if axes.ndim == 1:
            axes = axes.reshape(-1, 1)
    else:
        axes = np.array([[axes]])

    # Build indicator instances from legacy params when not provided
    if indicator_defs is None:
        _vwap = vwap if vwap is not None else pd.DataFrame()
        _vol = volume if volume is not None else pd.DataFrame()
        _sd_bands = SdBands()
        indicator_defs = [
            _sd_bands,
            VwapSlope(
                vwap=_vwap, volume=_vol, lookback=vwap_slope_lookback,
                mode=vwap_slope_mode, value_per_point=vwap_slope_value_per_point,
                scale=vwap_slope_scale, name="vwap_slope",
            ),
            VolumeImbalance(
                volume=_vol, sd_bands=_sd_bands,
                name="volume_imbalance",
            ),
            MeanReversion(
                window=mean_reversion_window,
                lookback_bars=max(2, vwap_slope_lookback),
                name="mean_reversion",
            ),
        ]

    # Pre-compute full indicator series (one pass over all bars, all markets)
    ind_map = {ind.name: ind for ind in indicator_defs}
    _returns = prices.pct_change()

    _sd_bands_ind = ind_map.get("sd_bands")
    
    # Compute SdBands indicator if it hasn't been computed yet
    # (e.g., when using a freshly created strategy instance for replotting)
    if _sd_bands_ind is not None:
        for i in range(len(prices)):
            _sd_bands_ind.compute(prices, _returns, i)
    
    sdbands: dict[str, pd.DataFrame] | None = (
        _sd_bands_ind.band_series if _sd_bands_ind is not None else None  # type: ignore[union-attr]
    )

    slope_ind: VwapSlope | None = ind_map.get("vwap_slope")  # type: ignore[assignment]

    if indicator_series is not None:
        slope_df     = indicator_series.get("vwap_slope")
        imbalance_df = indicator_series.get("volume_imbalance")
        mr_df        = indicator_series.get("mean_reversion")
    else:
        slope_df = slope_ind.compute_series(prices, _returns) if slope_ind is not None else None
        imbalance_df = (
            ind_map["volume_imbalance"].compute_series(prices, _returns)
            if "volume_imbalance" in ind_map else None
        )
        mr_df = (
            ind_map["mean_reversion"].compute_series(prices, _returns)
            if "mean_reversion" in ind_map else None
        )

    for idx, market in enumerate(markets):
        grid_row = idx // cols
        grid_col = idx % cols
        axes6 = axes[grid_row * stride: grid_row * stride + 6, grid_col]
        _plot_market_panels(
            axes6=axes6, market=market, prices=prices, trades=trades,
            vwap=vwap, volume=volume, buy_volume=buy_volume, sell_volume=sell_volume,
            slope_df=slope_df, imbalance_df=imbalance_df, mr_df=mr_df,
            slope_ind=slope_ind, ind_map=ind_map, suffix_re=suffix_re,
            sdbands=sdbands,
            max_vwap_slope=max_vwap_slope, mean_reversion_threshold=mean_reversion_threshold,
            vwap_slope_mode=vwap_slope_mode, vwap_slope_lookback=vwap_slope_lookback,
            mean_reversion_window=mean_reversion_window,
            signal_magnitude_threshold_imbalance=signal_magnitude_threshold_imbalance,
            signal_magnitude_threshold_slope=signal_magnitude_threshold_slope,
            signal_magnitude_threshold_meanrev=signal_magnitude_threshold_meanrev,
            high=high, low=low, opens=opens,
        )

    # Hide spacer rows between market groups
    if rows > 1:
        for r in range(rows - 1):
            spacer_flat = r * stride + 6
            for c in range(cols):
                ax_sp = axes[spacer_flat, c]
                ax_sp.set_visible(False)
                ax_sp.tick_params(axis="both", which="both",
                                  bottom=False, left=False,
                                  labelbottom=False, labelleft=False)

    total_slots = rows * cols
    for i in range(len(markets), total_slots):
        grid_row = i // cols
        grid_col = i % cols
        for panel in range(6):
            axes[grid_row * stride + panel, grid_col].set_visible(False)

    # Suppress x-tick labels on all non-bottom panels; only panel 5 of each
    # market group shows them (the spacer row provides room below for the labels).
    for r in range(rows):
        for c in range(cols):
            for p in range(6):
                axes[r * stride + p, c].tick_params(axis="x", labelbottom=(p == 5))

    handles = [
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#FF0000", markersize=7,
               label="Short Entry"),
        Line2D([0], [0], marker="x", color="#111111", lw=0, markersize=7, label="Exit"),
        Line2D([0], [0], color="#6C6C6C", lw=5, alpha=0.35, label="Volume Profile"),
        Line2D([0], [0], color="#2F4F4F", lw=1.2, label="VWAP slope"),
        Line2D([0], [0], color="#7A5C00", lw=1.2, label="Mean-Rev Score"),
    ]
    for wh in WINDOW_HOURS:
        handles.append(Line2D([0], [0], color=WINDOW_COLORS.get(wh, "gray"), lw=3,
                               label=f"T-{wh}h"))
    handles.append(Line2D([0], [0], color="#00008B", lw=1.6, linestyle="--",
                           label="Anchored mean vol"))
    handles.append(Line2D([0], [0], color="#00008B", lw=1.4, label="Cum buy Δ (yes−no)"))
    handles.append(Line2D([0], [0], color="#4B0082", lw=1.4, label="Cum sell Δ (yes−no)"))

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=9,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    compact_event = _compact_event_slug(event_slug)
    plt.suptitle(f"{compact_event}  |  {strategy_name}", fontsize=10, y=0.985)
    plt.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.subplots_adjust(hspace=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
