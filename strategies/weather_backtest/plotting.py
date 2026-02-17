from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from market_regime_analysis import (
    WINDOW_COLORS,
    WINDOW_HOURS,
    classify_regime,
    classify_regime_against_full,
    rolling_sd_bands,
)


def _compact_event_slug(event_slug: str) -> str:
    short = event_slug
    short = short.replace("highest-temperature-in-", "")
    short = short.replace("-on-", " | ")
    short = short.replace("-", " ")
    return short


def _window_regime_summary(mdf: pd.DataFrame, bands_full: pd.DataFrame) -> str:
    mdf_local = mdf.copy()
    bands_local = bands_full.copy()
    mdf_local["timestamp"] = pd.to_datetime(mdf_local["timestamp"], utc=True).dt.tz_localize(None)
    bands_local["timestamp"] = pd.to_datetime(bands_local["timestamp"], utc=True).dt.tz_localize(None)

    final_ts = mdf_local["timestamp"].max()
    parts: list[str] = []
    for wh in sorted(WINDOW_HOURS, reverse=True):
        start_time = final_ts - pd.Timedelta(hours=wh)
        sub = mdf_local[mdf_local["timestamp"] >= start_time]
        if len(sub) < 2:
            continue
        reg, _conf = classify_regime_against_full(sub, bands_local)
        parts.append(f"T-{wh}h:{reg}")
    return " | ".join(parts) if parts else "no windows"


def plot_entries_exits(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    strategy_name: str,
    event_slug: str,
    out_path: Path,
    vwap: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    vwap_slope_mode: str = "raw",
    vwap_slope_value_per_point: float = 1.0,
    vwap_slope_scale: float = 1.0,
    vwap_slope_lookback: int = 15,
    max_vwap_slope: float | None = None,
    mean_reversion_window: int = 5,
    mean_reversion_threshold: float | None = 0.5,
) -> None:
    """Plot price/regime plus three debug panels at dpi=100.

    For each market, top panel shows price + regime overlays + entries/exits.
    The next panels show total volume and the percent imbalance
    (above - below) / total.
    Middle panel shows rolling VWAP slope.
    Bottom panel shows rolling mean-reversion score (0..1).
    """
    markets = list(prices.columns)
    if not markets:
        return

    cols = 2
    rows = (len(markets) + cols - 1) // cols
    # Add panels for volume and imbalance percent.
    fig, axes = plt.subplots(
        rows * 5,
        cols,
        figsize=(14, 9.6 * rows),
        sharex="col",
        gridspec_kw={"height_ratios": [3, 0.8, 0.7, 1, 1] * rows},
    )

    if isinstance(axes, np.ndarray):
        if axes.ndim == 1:
            axes = axes.reshape(-1, 1)
    else:
        axes = np.array([[axes]])

    def _transform_slope(raw_slope: float) -> float:
        if vwap_slope_mode == "raw":
            return float(raw_slope)
        value_per_point = vwap_slope_value_per_point if vwap_slope_value_per_point != 0 else 1.0
        normalized = float(raw_slope / value_per_point)
        if vwap_slope_mode == "scaled":
            return float(normalized * vwap_slope_scale)
        if vwap_slope_mode == "angle":
            return float(np.degrees(np.arctan(normalized)) * vwap_slope_scale)
        raise ValueError(f"Unsupported vwap_slope_mode={vwap_slope_mode!r}")

    def _rolling_slope(series: pd.Series, lookback: int) -> tuple[pd.Series, pd.Series]:
        values = series.to_numpy(dtype=float)
        bar_idx = np.arange(len(values), dtype=float)
        out_raw = np.full(len(values), np.nan, dtype=float)
        out_metric = np.full(len(values), np.nan, dtype=float)
        for i in range(len(values)):
            start = max(0, i - lookback + 1)
            y = values[start: i + 1]
            x_window = bar_idx[start: i + 1]
            if np.isnan(y).all() or len(y) < 2:
                continue
            valid = ~np.isnan(y)
            y = y[valid]
            x = x_window[valid]
            if len(y) < 2:
                continue
            x = x - x[0]
            raw_slope = float(np.polyfit(x, y, 1)[0])
            out_raw[i] = raw_slope
            out_metric[i] = _transform_slope(raw_slope)
        return pd.Series(out_metric, index=series.index), pd.Series(out_raw, index=series.index)

    def _rolling_vwap_volume_imbalance_pct(
        price_series: pd.Series,
        vwap_series: pd.Series,
        vol_series: pd.Series,
        lookback: int,
    ) -> pd.Series:
        # Prediction-market bars are often sparse. Compute imbalance on actual
        # trade-update bars, then carry forward briefly for readability.
        valid = price_series.notna() & vwap_series.notna() & (vol_series.fillna(0.0) > 0.0)
        if int(valid.sum()) < 3:
            return pd.Series(np.nan, index=price_series.index, dtype=float)

        px_u = price_series[valid]
        vw_u = vwap_series[valid]
        vol_u = vol_series[valid].astype(float)

        above_u = vol_u.where(px_u > vw_u, 0.0)
        below_u = vol_u.where(px_u < vw_u, 0.0)

        lookback_updates = max(3, int(lookback))
        min_updates = max(3, int(np.ceil(lookback_updates * 0.10)))

        roll_above_u = above_u.rolling(window=lookback_updates, min_periods=min_updates).sum()
        roll_below_u = below_u.rolling(window=lookback_updates, min_periods=min_updates).sum()
        total_u = roll_above_u + roll_below_u
        imbalance_u = (roll_above_u - roll_below_u).divide(total_u.where(total_u > 0.0)) * 100.0

        # Align back to full bar timeline.
        imbalance_full = imbalance_u.reindex(price_series.index)

        # Carry forward between updates, but stop after a staleness horizon.
        max_stale_bars = max(3, int(np.ceil(lookback_updates * 0.5)))
        valid_pos = np.where(valid.to_numpy(dtype=bool), np.arange(len(valid)), -1)
        last_valid_pos = np.maximum.accumulate(valid_pos)
        bar_pos = np.arange(len(valid), dtype=int)
        bars_since_update = bar_pos - last_valid_pos
        stale_mask = (last_valid_pos >= 0) & (bars_since_update <= max_stale_bars)

        imbalance_display = imbalance_full.ffill()
        imbalance_display = imbalance_display.where(stale_mask)
        return imbalance_display

    def _rolling_mean_reversion_score(
        price_series: pd.Series,
        lookback: int,
        mr_window: int,
    ) -> pd.Series:
        values = price_series.to_numpy(dtype=float)
        out = np.full(len(values), np.nan, dtype=float)
        for i in range(len(values)):
            start = max(0, i - lookback + 1)
            y = values[start: i + 1]
            y = y[~np.isnan(y)]
            if len(y) <= 1:
                continue

            roll = pd.Series(y).rolling(window=max(1, int(mr_window)), min_periods=1).mean().to_numpy(dtype=float)
            dev = y - roll
            valid = dev[~np.isnan(dev)]
            if len(valid) <= 1:
                continue

            changes = np.sum(np.diff(np.sign(valid)) != 0)
            out[i] = float(np.clip(changes / (len(valid) - 1), 0.0, 1.0))
        return pd.Series(out, index=price_series.index)

    def _volume_profile(price_series: pd.Series, vol_series: pd.Series | None, bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
        p = price_series.to_numpy(dtype=float)
        if vol_series is None:
            w = np.ones_like(p)
        else:
            w = vol_series.to_numpy(dtype=float)

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

        hist, edges = np.histogram(p_valid, bins=edges, weights=w_valid)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return centers, hist.astype(float)

    for idx, market in enumerate(markets):
        grid_row = idx // cols
        grid_col = idx % cols
        ax = axes[grid_row * 5, grid_col]
        ax_vol = axes[grid_row * 5 + 1, grid_col]
        ax_vol_ratio = axes[grid_row * 5 + 2, grid_col]
        ax_slope = axes[grid_row * 5 + 3, grid_col]
        ax_meanrev = axes[grid_row * 5 + 4, grid_col]

        series = prices[market].dropna()
        if series.empty:
            continue

        mdf = pd.DataFrame({"timestamp": pd.to_datetime(series.index), "price": series.values}).sort_values("timestamp")

        bands_full = rolling_sd_bands(mdf)
        t_full = bands_full["timestamp"].values
        ax.fill_between(t_full, bands_full["-3sd"], bands_full["+3sd"], alpha=0.06, color="#B6B6B6")
        ax.fill_between(t_full, bands_full["-2sd"], bands_full["+2sd"], alpha=0.08, color="#BABABA")
        ax.fill_between(t_full, bands_full["-1sd"], bands_full["+1sd"], alpha=0.12, color="#909090")
        ax.plot(t_full, bands_full["mean"], color="black", linestyle="--", linewidth=1.0, alpha=0.45)
        ax.plot(mdf["timestamp"], mdf["price"], color="#A9C4FF", linewidth=1.2, alpha=0.95, zorder=3)

        final_ts = mdf["timestamp"].max()
        prev_hours = 0
        for wh in sorted(WINDOW_HOURS):
            start_time = final_ts - pd.Timedelta(hours=wh)
            end_time = final_ts - pd.Timedelta(hours=prev_hours)
            if prev_hours == 0:
                sub = mdf[(mdf["timestamp"] >= start_time) & (mdf["timestamp"] <= end_time)]
            else:
                sub = mdf[(mdf["timestamp"] >= start_time) & (mdf["timestamp"] < end_time)]
            prev_hours = wh
            if len(sub) < 2:
                continue
            ax.plot(
                sub["timestamp"],
                sub["price"],
                color=WINDOW_COLORS.get(wh, "gray"),
                linewidth=2.0,
                alpha=0.95,
                zorder=6,
            )

        full_regime, full_conf = classify_regime(bands_full)
        windows_str = _window_regime_summary(mdf, bands_full)

        market_trades = trades[trades["asset"] == market] if not trades.empty else pd.DataFrame()
        if not market_trades.empty:
            ax.scatter(pd.to_datetime(market_trades["entry_time"]), market_trades["entry_price"], marker="v", s=36, color="#8B0000", zorder=8)
            ax.scatter(pd.to_datetime(market_trades["exit_time"]), market_trades["exit_price"], marker="x", s=40, color="#111111", zorder=9)
            for _, tr in market_trades.iterrows():
                ax.plot(
                    [pd.to_datetime(tr["entry_time"]), pd.to_datetime(tr["exit_time"])],
                    [tr["entry_price"], tr["exit_price"]],
                    color="#6B6B6B",
                    linewidth=0.8,
                    alpha=0.7,
                    zorder=7,
                )
            trade_count = len(market_trades)
            mean_pnl = float(market_trades["pnl"].mean())
            trade_info = f"trades={trade_count}, avgPnL={mean_pnl:+.3f}"
        else:
            trade_info = "trades=0"

        ax.set_title(f"{market}\nFull: {full_regime} ({full_conf:.2f}) — {windows_str}\n{trade_info}", fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelbottom=True, labelsize=8)
        ax.grid(alpha=0.3)

        # Left-side volume profile synchronized with price y-axis.
        vol_for_profile = None
        if volume is not None and market in volume.columns:
            vol_for_profile = volume[market].reindex(series.index).fillna(0.0)
        vp_centers, vp_hist = _volume_profile(series, vol_for_profile)
        if len(vp_centers) > 0 and float(np.nanmax(vp_hist)) > 0.0:
            ax_profile = ax.inset_axes([-0.16, 0.0, 0.14, 1.0], transform=ax.transAxes)
            vp_norm = vp_hist / float(np.nanmax(vp_hist))
            if len(vp_centers) > 1:
                h = float(np.median(np.diff(vp_centers))) * 0.9
            else:
                h = float(max((series.max() - series.min()) * 0.02, 1e-4))
            ax_profile.barh(vp_centers, vp_norm, height=h, color="#6C6C6C", alpha=0.35)
            ax_profile.set_ylim(ax.get_ylim())
            ax_profile.set_xlim(1.05, 0.0)
            ax_profile.set_xticks([])
            ax_profile.tick_params(axis="y", left=False, labelleft=False)
            ax_profile.set_facecolor("none")
            for side in ["left", "top", "bottom"]:
                ax_profile.spines[side].set_visible(False)
            ax_profile.spines["right"].set_alpha(0.25)

        # Simple volume timeline below the price panel
        if vol_for_profile is not None and len(vol_for_profile):
            ax_vol.plot(vol_for_profile.index, vol_for_profile.values, color="#6C6C6C", linewidth=0.8)
            ax_vol.fill_between(vol_for_profile.index, 0, vol_for_profile.values, color="#B6B6B6", alpha=0.25)
            ax_vol.set_ylabel("vol", fontsize=7)
            ax_vol.tick_params(axis="x", rotation=45, labelsize=7)
            ax_vol.tick_params(axis="y", labelsize=7)
            ax_vol.grid(alpha=0.18)
            ax_vol.set_xlim(ax.get_xlim())
        else:
            ax_vol.text(0.5, 0.5, "No volume data", ha="center", va="center", fontsize=8)
            ax_vol.set_axis_off()

        # Volume imbalance percent panel
        if vwap is not None and market in vwap.columns and vol_for_profile is not None:
            vwap_series = vwap[market].reindex(series.index)
            vol_series = vol_for_profile.reindex(series.index).fillna(0.0)
            valid_mask = vwap_series.notna() & (vol_series > 0.0)

            vol_above = vol_series.where(valid_mask & (series > vwap_series), 0.0)
            vol_below = vol_series.where(valid_mask & (series < vwap_series), 0.0)

            roll_window = max(3, int(vwap_slope_lookback))
            roll_above = vol_above.rolling(window=roll_window, min_periods=1).sum()
            roll_below = vol_below.rolling(window=roll_window, min_periods=1).sum()
            total_vol = roll_above + roll_below
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_pct = (roll_above - roll_below).divide(total_vol.where(total_vol > 0.0)) * 100.0
            ax_vol_ratio.plot(ratio_pct.index, ratio_pct.values, color="#5A5A5A", linewidth=0.8)
            ax_vol_ratio.axhline(0.0, color="#888888", linestyle="--", linewidth=0.7, alpha=0.6)
            if not market_trades.empty:
                entry_times = pd.to_datetime(market_trades["entry_time"])
                ratio_at_entry = ratio_pct.reindex(entry_times).to_numpy(dtype=float)
                ax_vol_ratio.scatter(entry_times, ratio_at_entry, marker="v", s=22, color="#8B0000", zorder=6)
            ax_vol_ratio.set_ylabel("imb %", fontsize=7)
            ax_vol_ratio.tick_params(axis="x", rotation=45, labelsize=7)
            ax_vol_ratio.tick_params(axis="y", labelsize=7)
            ax_vol_ratio.grid(alpha=0.18)
            ax_vol_ratio.set_xlim(ax.get_xlim())
        else:
            ax_vol_ratio.text(0.5, 0.5, "No VWAP data", ha="center", va="center", fontsize=8)
            ax_vol_ratio.set_axis_off()

        # VWAP slope panel (optional if VWAP matrix provided)
        if vwap is not None and market in vwap.columns:
            vwap_series = vwap[market].reindex(series.index)
            valid_vwap_mask = vwap_series.notna()
            if volume is not None and market in volume.columns:
                vol_series = volume[market].reindex(series.index).fillna(0.0)
                valid_vwap_mask = (vol_series > 0.0) & vwap_series.notna()
                vwap_series = vwap_series.where(valid_vwap_mask)

            slope_series, slope_raw_series = _rolling_slope(vwap_series, max(2, int(vwap_slope_lookback)))
            # Keep update-only slope (true signal) and add carry-forward display line for visual continuity.
            slope_display = slope_series.ffill()
            ax_slope.plot(
                slope_display.index,
                slope_display.values,
                color="#708090",
                linewidth=0.9,
                alpha=0.55,
                linestyle="--",
            )
            ax_slope.plot(slope_series.index, slope_series.values, color="#2F4F4F", linewidth=1.1)
            ax_slope.axhline(0.0, color="#888888", linestyle="--", linewidth=0.9, alpha=0.8)
            if max_vwap_slope is not None:
                ax_slope.axhline(
                    float(max_vwap_slope),
                    color="#8B0000",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.9,
                )

            market_trades = trades[trades["asset"] == market] if not trades.empty else pd.DataFrame()
            if not market_trades.empty:
                entry_times = pd.to_datetime(market_trades["entry_time"])
                slope_at_entry = slope_series.reindex(entry_times).to_numpy(dtype=float)
                ax_slope.scatter(entry_times, slope_at_entry, marker="v", s=24, color="#8B0000", zorder=6)

            # Log-like visualization for signed slope values.
            slope_vals = slope_series.to_numpy(dtype=float)
            finite_vals = slope_vals[np.isfinite(slope_vals)]
            abs_nonzero = np.abs(finite_vals[np.abs(finite_vals) > 0.0]) if len(finite_vals) else np.array([])
            if len(abs_nonzero) > 0:
                linthresh = float(np.nanpercentile(abs_nonzero, 35))
                linthresh = max(linthresh, 1e-6)
                ax_slope.set_yscale("symlog", linthresh=linthresh, linscale=1.0)

            # Robust y-limits: avoid a single outlier flattening the full panel.
            if len(finite_vals) >= 8:
                lo = float(np.nanpercentile(finite_vals, 2.0))
                hi = float(np.nanpercentile(finite_vals, 98.0))
                if max_vwap_slope is not None:
                    lo = min(lo, float(max_vwap_slope))
                    hi = max(hi, float(max_vwap_slope))
                lo = min(lo, 0.0)
                hi = max(hi, 0.0)
                if hi > lo:
                    pad = (hi - lo) * 0.12
                    ax_slope.set_ylim(lo - pad, hi + pad)

            ax_slope.set_ylabel("slope", fontsize=7)
            mode_label = vwap_slope_mode
            if vwap_slope_mode != "raw":
                mode_label += f", vpp={vwap_slope_value_per_point:g}, scale={vwap_slope_scale:g}"
            update_pct = float(valid_vwap_mask.mean() * 100.0) if len(valid_vwap_mask) else 0.0
            ax_slope.set_title(
                f"VWAP slope [{mode_label}] (symlog y, lookback={vwap_slope_lookback} bars, updates={update_pct:.1f}%)",
                fontsize=7,
            )
            ax_slope.tick_params(axis="x", rotation=45, labelsize=7)
            ax_slope.tick_params(axis="y", labelsize=7)
            ax_slope.grid(alpha=0.25)
            ax_slope.set_xlim(ax.get_xlim())
        else:
            ax_slope.text(0.5, 0.5, "No VWAP data", ha="center", va="center", fontsize=8)
            ax_slope.set_axis_off()

        # Mean-reversion debug panel (same score logic as strategy, rolling over lookback bars).
        mr_score = _rolling_mean_reversion_score(
            price_series=series,
            lookback=max(2, int(vwap_slope_lookback)),
            mr_window=max(1, int(mean_reversion_window)),
        )
        ax_meanrev.plot(mr_score.index, mr_score.values, color="#7A5C00", linewidth=1.1)
        if mean_reversion_threshold is not None:
            ax_meanrev.axhline(
                float(mean_reversion_threshold),
                color="#8B0000",
                linestyle=":",
                linewidth=1.0,
                alpha=0.9,
            )
        ax_meanrev.axhline(0.0, color="#888888", linestyle="--", linewidth=0.9, alpha=0.6)

        if not market_trades.empty:
            entry_times = pd.to_datetime(market_trades["entry_time"])
            mr_at_entry = mr_score.reindex(entry_times).to_numpy(dtype=float)
            ax_meanrev.scatter(entry_times, mr_at_entry, marker="v", s=22, color="#8B0000", zorder=6)

        ax_meanrev.set_ylabel("mr", fontsize=7)
        ax_meanrev.set_ylim(-0.02, 1.02)
        ax_meanrev.set_title(
            f"Mean-reversion score (window={mean_reversion_window}, threshold={mean_reversion_threshold if mean_reversion_threshold is not None else 'n/a'})",
            fontsize=7,
        )
        ax_meanrev.tick_params(axis="x", rotation=45, labelsize=7)
        ax_meanrev.tick_params(axis="y", labelsize=7)
        ax_meanrev.grid(alpha=0.25)
        ax_meanrev.set_xlim(ax.get_xlim())

    total_slots = rows * cols
    for i in range(len(markets), total_slots):
        grid_row = i // cols
        grid_col = i % cols
        axes[grid_row * 5, grid_col].set_visible(False)
        axes[grid_row * 5 + 1, grid_col].set_visible(False)
        axes[grid_row * 5 + 2, grid_col].set_visible(False)
        axes[grid_row * 5 + 3, grid_col].set_visible(False)
        axes[grid_row * 5 + 4, grid_col].set_visible(False)

    handles = [
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#8B0000", markersize=7, label="Short Entry"),
        Line2D([0], [0], marker="x", color="#111111", lw=0, markersize=7, label="Exit"),
        Line2D([0], [0], color="#6C6C6C", lw=5, alpha=0.35, label="Volume Profile"),
        Line2D([0], [0], color="#2F4F4F", lw=1.2, label="VWAP slope"),
        Line2D([0], [0], color="#7A5C00", lw=1.2, label="Mean-Rev Score"),
    ]
    for wh in WINDOW_HOURS:
        handles.append(Line2D([0], [0], color=WINDOW_COLORS.get(wh, "gray"), lw=3, label=f"T-{wh}h"))

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=9,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    compact_event = _compact_event_slug(event_slug)
    plt.suptitle(f"{compact_event}  |  {strategy_name}", fontsize=10, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
