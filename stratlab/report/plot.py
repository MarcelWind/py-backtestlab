"""Portfolio visualization and plotting."""

import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

from stratlab.strategy.indicators import (
    analyze_band_position_vs_reference,
    detect_mean_reversion_vs_reference,
    market_regimes,
    volume_profile,
)


def plot_backtest(
    results: dict,
    title: str = "Portfolio Backtest",
    figsize: tuple[float, float] = (14, 10),
) -> Figure:
    """
    Plot portfolio equity curve and position sizes.

    Args:
        results: Dict from strategy.run() with 'returns' and 'weights'
        title: Plot title
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure
    """
    returns = results["returns"]
    weights = results["weights"]
    metrics = results.get("metrics", {})

    # Compute equity curve (starting at 1.0)
    equity = (1 + returns).cumprod()

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=figsize,
        height_ratios=[2, 1],
        sharex=True,
    )

    # --- Equity Curve ---
    ax1.plot(equity.index, equity.values, color="steelblue", linewidth=1.5)
    ax1.fill_between(equity.index, 1, equity.values, alpha=0.3, color="steelblue")

    # Drawdown shading
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    ax1.fill_between(
        equity.index,
        equity.values,
        rolling_max.values,
        alpha=0.3,
        color="red",
        label="Drawdown",
    )

    # Add metrics annotation
    stats_text = (
        f"Total Return: {metrics.get('total_return', 0):.1%}\n"
        f"Sharpe: {metrics.get('sharpe', 0):.2f}\n"
        f"Sortino: {metrics.get('sortino', 0):.2f}\n"
        f"Max DD: {metrics.get('max_drawdown', 0):.1%}"
    )
    ax1.text(
        0.02, 0.98, stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax1.set_ylabel("Portfolio Value", fontsize=11)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color="gray", linestyle="--", linewidth=0.8)

    # --- Position Sizes (Stacked Area) ---
    # Filter to only assets that were held at some point
    held_assets = weights.columns[weights.sum() > 0]
    weights_held = weights[held_assets]

    if len(held_assets) > 0:
        ax2.stackplot(
            weights_held.index,
            weights_held.T.values,
            labels=held_assets,
            alpha=0.8,
        )
        ax2.legend(
            loc="upper left",
            fontsize=8,
            ncol=min(6, len(held_assets)),
            framealpha=0.9,
        )

    # Show cash position
    cash = 1 - weights.sum(axis=1)
    if cash.max() > 0.01:
        ax2.fill_between(
            weights.index,
            weights.sum(axis=1),
            1,
            alpha=0.3,
            color="gray",
            label="Cash",
        )

    ax2.set_ylabel("Position Size", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison(
    results_dict: dict[str, dict],
    title: str = "Strategy Comparison",
    figsize: tuple[float, float] = (14, 6),
) -> Figure:
    """
    Plot multiple equity curves for comparison.

    Args:
        results_dict: Dict mapping strategy name to results dict
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, results in results_dict.items():
        returns = results["returns"]
        equity = (1 + returns).cumprod()
        sharpe = results.get("metrics", {}).get("sharpe", 0)
        ax.plot(equity.index, equity.values, label=f"{name} (Sharpe: {sharpe:.2f})", linewidth=1.5)

    ax.set_ylabel("Portfolio Value", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    return fig


def plot_rolling_correlation(
    results_a: dict,
    results_b: dict,
    window: int = 30,
    use_roc: bool = False,
    title: str = "Rolling Window Correlation",
    figsize: tuple[float, float] = (14, 6),
    label_a: str = "Strategy",
    label_b: str = "Benchmark",
) -> Figure:
    """
    Plot correlation between two strategies over non-overlapping windows.

    Computes correlation for each window of length n, showing how the
    relationship between strategies changes over time.

    Args:
        results_a: Results dict for first strategy
        results_b: Results dict for second strategy
        window: Window size in periods (e.g., 30 days)
        use_roc: If True, use return-on-capital (strip out cash)
        title: Plot title
        figsize: Figure size
        label_a: Label for first series
        label_b: Label for second series

    Returns:
        Matplotlib Figure
    """
    returns_a = results_a["returns"]
    returns_b = results_b["returns"]

    # Apply ROC if requested
    if use_roc:
        returns_a = compute_return_on_capital(returns_a, results_a["weights"])
        returns_b = compute_return_on_capital(returns_b, results_b["weights"])

    # Align the two series
    aligned = pd.concat([returns_a, returns_b], axis=1, keys=[label_a, label_b]).dropna()

    # Compute correlations for non-overlapping windows
    n_windows = len(aligned) // window
    correlations = []
    window_dates = []

    for i in range(n_windows):
        start_idx = i * window
        end_idx = start_idx + window
        window_data = aligned.iloc[start_idx:end_idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr = window_data[label_a].corr(window_data[label_b])
        # Handle NaN correlation (zero variance in one series)
        if np.isnan(corr):
            corr = 0.0
        correlations.append(corr)
        window_dates.append(window_data.index[-1])  # End date of window

    corr_series = pd.Series(correlations, index=window_dates)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1], sharex=True)

    # --- Correlation over time ---
    colors = ["green" if c > 0 else "red" for c in correlations]
    ax1.bar(window_dates, correlations, width=window * 0.8, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.axhline(y=0, color="black", linewidth=1)
    ax1.axhline(y=corr_series.mean(), color="blue", linestyle="--", linewidth=1.5, label=f"Mean: {corr_series.mean():.3f}")

    ax1.set_ylabel("Correlation", fontsize=11)
    ax1.set_title(f"{title} ({window}-day windows)", fontsize=14, fontweight="bold")
    ax1.set_ylim(-1.1, 1.1)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add stats
    stats_text = (
        f"Mean: {corr_series.mean():.3f}\n"
        f"Std: {corr_series.std():.3f}\n"
        f"Min: {corr_series.min():.3f}\n"
        f"Max: {corr_series.max():.3f}"
    )
    ax1.text(
        0.02, 0.98, stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # --- Cumulative returns comparison ---
    equity_a = (1 + aligned[label_a]).cumprod()
    equity_b = (1 + aligned[label_b]).cumprod()
    ax2.plot(equity_a.index, equity_a.values, label=label_a, linewidth=1.5)
    ax2.plot(equity_b.index, equity_b.values, label=label_b, linewidth=1.5, alpha=0.7)
    ax2.set_ylabel("Cumulative Return", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compute_return_on_capital(
    returns: pd.Series,
    weights: pd.DataFrame,
) -> pd.Series:
    """
    Compute return on capital deployed (cash-stripped returns).

    If portfolio is 60% invested, a 3% portfolio return becomes 5% ROC.

    Args:
        returns: Portfolio returns series
        weights: Weights DataFrame (same index as returns)

    Returns:
        Series of return-on-capital values
    """
    # Total exposure at each point (1 - cash)
    exposure = weights.sum(axis=1)
    # Avoid division by zero for fully cash positions
    exposure = exposure.replace(0, np.nan)
    # ROC = portfolio return / exposure
    roc = returns / exposure
    return roc.fillna(0)


def plot_return_distribution(
    results_dict: dict[str, dict],
    window: int = 30,
    use_roc: bool = True,
    title: str = "Return Distribution",
    figsize: tuple[float, float] = (14, 8),
) -> Figure:
    """
    Plot return distributions for multiple strategies.

    Computes returns over non-overlapping windows and shows histogram/KDE.

    Args:
        results_dict: Dict mapping strategy name to results dict
            Each results dict must have 'returns' and 'weights' keys
        window: Window size for aggregating returns (e.g., 30 days)
        use_roc: If True, compute return-on-capital (strip out cash)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    window_returns = {}
    stats = []

    for name, results in results_dict.items():
        returns = results["returns"]
        weights = results["weights"]

        # Compute return on capital if requested
        if use_roc:
            returns = compute_return_on_capital(returns, weights)

        # Aggregate into non-overlapping windows
        n_windows = len(returns) // window
        windowed = []
        for i in range(n_windows):
            start_idx = i * window
            end_idx = start_idx + window
            # Compound returns over window
            window_vals = (1 + returns.iloc[start_idx:end_idx]).to_numpy()
            window_ret = float(np.prod(window_vals)) - 1.0
            windowed.append(window_ret)

        window_returns[name] = np.array(windowed)

        # Compute stats
        arr = np.array(windowed)
        stats.append({
            "Strategy": name,
            "Mean": np.mean(arr),
            "Std": np.std(arr),
            "Median": np.median(arr),
            "Min": np.min(arr),
            "Max": np.max(arr),
            "Skew": pd.Series(arr).skew(),
            "Win%": np.mean(arr > 0) * 100,
        })

    # --- Grouped Histogram ---
    n_strategies = len(window_returns)
    n_bins = 20

    # Compute bin edges
    all_vals = np.concatenate(list(window_returns.values()))
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Width for each strategy's bar (with gap)
    bar_width = bin_width / (n_strategies + 1)
    cmap = plt.colormaps["tab10"]
    colors = [cmap(i) for i in range(n_strategies)]

    for i, (name, rets) in enumerate(window_returns.items()):
        # Compute histogram counts
        counts, _ = np.histogram(rets, bins=bin_edges)
        # Offset each strategy's bars
        offset = (i - n_strategies / 2 + 0.5) * bar_width
        ax1.bar(
            bin_centers + offset,
            counts,
            width=bar_width * 0.9,
            label=name,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
        )

    ax1.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel(f"{window}-Day Return", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title(f"{title} - Histogram", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # --- Box Plot ---
    data = [window_returns[name] for name in results_dict.keys()]
    labels = list(results_dict.keys())
    bp = ax2.boxplot(data, labels=labels, patch_artist=True)

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.6)

    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax2.set_ylabel(f"{window}-Day Return", fontsize=11)
    ax2.set_title(f"{title} - Box Plot", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add stats table
    stats_df = pd.DataFrame(stats)
    stats_text = f"{'Strategy':<15} {'Mean':>8} {'Std':>8} {'Win%':>6}\n"
    stats_text += "-" * 40 + "\n"
    for _, row in stats_df.iterrows():
        stats_text += f"{row['Strategy']:<15} {row['Mean']:>7.1%} {row['Std']:>7.1%} {row['Win%']:>5.0f}%\n"

    fig.text(
        0.5, 0.02, stats_text,
        ha="center",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    roc_label = " (Return on Capital)" if use_roc else ""
    fig.suptitle(f"{title}{roc_label}", fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    return fig


def plot_scatter_correlation(
    results_a: dict,
    results_b: dict,
    window: int = 1,
    use_roc: bool = False,
    title: str = "Return Correlation",
    figsize: tuple[float, float] = (10, 8),
    label_a: str = "Strategy A",
    label_b: str = "Strategy B",
) -> Figure:
    """
    Scatter plot of returns between two strategies with trendline.

    Args:
        results_a: Results dict for first strategy
        results_b: Results dict for second strategy
        window: Window size for aggregating returns (1 = daily)
        use_roc: If True, use return-on-capital (strip out cash)
        title: Plot title
        figsize: Figure size
        label_a: Label for x-axis strategy
        label_b: Label for y-axis strategy

    Returns:
        Matplotlib Figure
    """
    returns_a = results_a["returns"]
    returns_b = results_b["returns"]

    # Apply ROC if requested
    if use_roc:
        returns_a = compute_return_on_capital(returns_a, results_a["weights"])
        returns_b = compute_return_on_capital(returns_b, results_b["weights"])

    # Align series
    aligned = pd.concat([returns_a, returns_b], axis=1, keys=[label_a, label_b]).dropna()

    # Aggregate into windows if window > 1
    if window > 1:
        n_windows = len(aligned) // window
        windowed_a = []
        windowed_b = []
        for i in range(n_windows):
            start_idx = i * window
            end_idx = start_idx + window
            vals_a = (1 + aligned[label_a].iloc[start_idx:end_idx]).to_numpy()
            vals_b = (1 + aligned[label_b].iloc[start_idx:end_idx]).to_numpy()
            windowed_a.append(float(np.prod(vals_a)) - 1.0)
            windowed_b.append(float(np.prod(vals_b)) - 1.0)
        x: np.ndarray = np.array(windowed_a, dtype=np.float64)
        y: np.ndarray = np.array(windowed_b, dtype=np.float64)
    else:
        x = np.asarray(aligned[label_a], dtype=np.float64)
        y = np.asarray(aligned[label_b], dtype=np.float64)

    # Compute correlation and regression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        correlation = np.corrcoef(x, y)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0

    # Linear regression (y = mx + b)
    slope, intercept = np.polyfit(x, y, 1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(x, y, alpha=0.5, s=20, edgecolor="none")

    # Trendline
    x_min, x_max = float(np.min(x)), float(np.max(x))
    x_line = np.array([x_min, x_max])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="red", linewidth=2, label=f"Trend: y = {slope:.2f}x + {intercept:.4f}")

    # Reference lines
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # 45-degree line (perfect correlation)
    lim_min = min(x_min, float(np.min(y)))
    lim_max = max(x_max, float(np.max(y)))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="green", linestyle=":", linewidth=1, alpha=0.7, label="y = x")

    # Labels
    period_label = f"{window}-Day" if window > 1 else "Daily"
    ax.set_xlabel(f"{label_a} {period_label} Return", fontsize=11)
    ax.set_ylabel(f"{label_b} {period_label} Return", fontsize=11)
    ax.set_title(f"{title} (r = {correlation:.3f})", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Stats box
    stats_text = (
        f"Correlation: {correlation:.3f}\n"
        f"Slope (β): {slope:.3f}\n"
        f"Intercept (α): {intercept:.4f}\n"
        f"N: {len(x)} observations"
    )
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.tight_layout()
    return fig


def plot_market_indicators(
    prices: pd.DataFrame,
    indicator_defs: list,
    trades: pd.DataFrame | None = None,
    vwap: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    strategy_name: str = "",
    out_path: Path | str | None = None,
    figsize_per_market: tuple[float, float] = (10, 14),
) -> Figure:
    """Plot per-market price, VWAP, volume, and indicator subpanels.

    Renders one column of subplots per market in ``prices``. Panel 0 (price)
    is always drawn; additional rows come from each indicator's ``plot_panel``
    attribute. Indicators with ``plot_panel=None`` are silently skipped.

    Args:
        prices:             Price DataFrame (rows=timestamps, cols=markets).
        indicator_defs:     List of ``Indicator`` instances. Each must expose
                            a ``plot_panel`` attribute (int > 0 or None) and
                            implement ``compute_series(prices, returns)``.
        trades:             Optional trades DataFrame with columns ``market``,
                            ``entry_time``, and optionally ``exit_time`` and
                            ``pnl``. Entry (▼) and exit (✕) markers are drawn
                            on the price panel; entry markers are repeated on
                            indicator panels.
        vwap:               Optional VWAP DataFrame aligned to ``prices``.
                            Rendered as an orange overlay on the price panel.
        volume:             Optional volume DataFrame aligned to ``prices``.
                            Rendered as gray bars on a twin y-axis with an
                            expanding-mean dashed line.
        strategy_name:      Title prefix shown above the figure.
        out_path:           If provided, save the figure here and close it.
        figsize_per_market: ``(width, height)`` contributed by each market
                            column.

    Returns:
        Matplotlib ``Figure``.
    """
    markets = prices.columns.tolist()
    n_markets = len(markets)
    returns = prices.pct_change()

    # Determine extra panel IDs (int > 0, deduplicated, sorted)
    extra_panels: list[int] = sorted({
        ind.plot_panel  # type: ignore[union-attr]
        for ind in indicator_defs
        if getattr(ind, "plot_panel", None) is not None and ind.plot_panel > 0
    })

    n_rows = 1 + len(extra_panels)
    height_ratios = [3] + [1.2] * len(extra_panels)

    fig, axes = plt.subplots(
        n_rows,
        n_markets,
        figsize=(figsize_per_market[0] * n_markets, figsize_per_market[1]),
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
    )

    # Pre-compute full indicator series once per indicator
    # panel_id -> [(ind_name, series_df), ...]
    panel_data: dict[int, list[tuple[str, pd.DataFrame]]] = {}
    for ind in indicator_defs:
        pid = getattr(ind, "plot_panel", None)
        if pid is not None and pid > 0:
            series_df = ind.compute_series(prices, returns)
            panel_data.setdefault(pid, []).append((ind.name, series_df))

    # Bar width for volume (matplotlib datetime axes use days as unit)
    bar_width_days: float = 1.0
    if len(prices.index) > 1:
        try:
            diffs = pd.DatetimeIndex(prices.index).to_series().diff().dropna()
            bar_width_days = float(diffs.median().total_seconds() / 86400) * 0.8
        except Exception:
            bar_width_days = 1.0

    for col, market in enumerate(markets):
        # ------------------------------------------------------------------ #
        # Panel 0 — Price
        # ------------------------------------------------------------------ #
        ax_p = axes[0][col]
        price_col = prices[market]
        ax_p.plot(
            price_col.index, price_col.values,
            color="steelblue", linewidth=0.8, label="Price",
        )

        # VWAP overlay
        if vwap is not None and market in vwap.columns:
            vwap_col = vwap[market].reindex(price_col.index)
            ax_p.plot(
                vwap_col.index, vwap_col.values,
                color="darkorange", linewidth=0.8, alpha=0.85, label="VWAP",
            )

        # Volume on twin y-axis
        if volume is not None and market in volume.columns:
            ax_v = ax_p.twinx()
            vol_col = volume[market].reindex(prices.index).fillna(0.0)
            ax_v.bar(
                vol_col.index, vol_col.values,
                width=bar_width_days, color="gray", alpha=0.25,
            )
            vol_mean = vol_col.expanding().mean()
            ax_v.plot(
                vol_mean.index, vol_mean.values,
                color="gray", linestyle="--", linewidth=0.7, alpha=0.6,
            )
            vol_max = float(vol_col.max())
            ax_v.set_ylim(0, vol_max * 4 if vol_max > 0 else 1)
            ax_v.tick_params(axis="y", labelsize=7, colors="gray")
            ax_v.set_ylabel("Volume", fontsize=7, color="gray")

        # Trade entry / exit markers
        n_trades = 0
        if trades is not None and "market" in trades.columns:
            mkt_tr = trades[trades["market"] == market]
            n_trades = len(mkt_tr)
            for _, tr in mkt_tr.iterrows():
                et = tr.get("entry_time")
                xt = tr.get("exit_time")
                ep = (
                    float(prices[market][et])
                    if et is not None and et in prices.index
                    else None
                )
                xp = (
                    float(prices[market][xt])
                    if xt is not None and pd.notna(xt) and xt in prices.index
                    else None
                )
                if ep is not None:
                    ax_p.scatter(et, ep, marker="v", color="limegreen", s=40, zorder=5)
                if xp is not None:
                    ax_p.scatter(
                        xt, xp, marker="x", color="red", s=40, linewidths=1.2, zorder=5,
                    )
                if ep is not None and xp is not None:
                    raw_pnl = tr.get("pnl", None)
                    lc = "red" if (raw_pnl is not None and raw_pnl < 0) else "limegreen"
                    ax_p.plot(
                        [et, xt], [ep, xp],
                        color=lc, linewidth=0.5, alpha=0.45,
                    )

        title = f"{market}  —  {strategy_name}" if strategy_name else market
        if n_trades:
            title += f"  ({n_trades} trades)"
        ax_p.set_title(title, fontsize=9, fontweight="bold")
        ax_p.grid(True, alpha=0.25)
        ax_p.tick_params(axis="x", labelsize=7, rotation=30)
        ax_p.tick_params(axis="y", labelsize=7)
        if col == 0:
            ax_p.set_ylabel("Price", fontsize=9)
        handles, leg_labels = ax_p.get_legend_handles_labels()
        if handles:
            ax_p.legend(handles, leg_labels, loc="upper left", fontsize=7, framealpha=0.7)

        # ------------------------------------------------------------------ #
        # Panels 1..N — Indicator panels
        # ------------------------------------------------------------------ #
        for row, pid in enumerate(extra_panels, start=1):
            ax_i = axes[row][col]

            if pid in panel_data:
                for ind_name, series_df in panel_data[pid]:
                    if market in series_df.columns:
                        s = series_df[market]
                        ax_i.plot(s.index, s.values, linewidth=0.8, label=ind_name)

                ax_i.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

                # Optional symmetric threshold bands (if indicator declares one)
                for ind in indicator_defs:
                    if getattr(ind, "plot_panel", None) == pid and hasattr(ind, "threshold"):
                        thr = float(ind.threshold)
                        ax_i.axhline(
                            y=thr, color="firebrick",
                            linestyle=":", linewidth=0.8, alpha=0.8,
                        )
                        ax_i.axhline(
                            y=-thr, color="firebrick",
                            linestyle=":", linewidth=0.8, alpha=0.8,
                        )

                # Trade entry markers on indicator panels
                if (
                    trades is not None
                    and "market" in trades.columns
                    and "entry_time" in trades.columns
                ):
                    mkt_tr2 = trades[trades["market"] == market]
                    for ind_name, series_df in panel_data[pid]:
                        if market in series_df.columns:
                            for _, tr in mkt_tr2.iterrows():
                                et = tr.get("entry_time")
                                if et is not None and et in series_df.index:
                                    val = float(series_df[market][et])
                                    if not np.isnan(val):
                                        ax_i.scatter(
                                            et, val,
                                            marker="v", color="limegreen", s=25, zorder=5,
                                        )

            ax_i.grid(True, alpha=0.25)
            ax_i.tick_params(axis="x", labelsize=7, rotation=30)
            ax_i.tick_params(axis="y", labelsize=7)
            if col == 0 and pid in panel_data:
                panel_names = [n for n, _ in panel_data[pid]]
                ax_i.set_ylabel(", ".join(panel_names), fontsize=8)

    suptitle = f"{strategy_name} — Market Indicators" if strategy_name else "Market Indicators"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    if out_path is not None:
        fig.savefig(Path(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Window-based Regime Analysis Helpers
# ---------------------------------------------------------------------------

def compute_window_regime_summary(
    price_series: pd.Series | np.ndarray,
    full_bands: pd.DataFrame,
    window_hours: dict[int, str] | None = None,
    timestamps: np.ndarray | pd.Index | None = None,
) -> str:
    """
    Compute regime classifications for multiple lookback windows.

    Returns a string summary like "T-24h:Imb. Up | T-12h:Balanced | T-6h:Mean-Reverting".

    Parameters
    ----------
    price_series:
        Prices for the market (pd.Series with datetime index or numpy array).
    full_bands:
        Full-history band statistics DataFrame from sd_bands_rolling().
        Index should be timestamps.
    window_hours:
        Dict mapping hours to labels, e.g. {48: "2d", 24: "1d", 6: "6h"}.
        If None, uses default: {48: "T-48h", 24: "T-24h", 12: "T-12h", 6: "T-6h"}.
    timestamps:
        Optional explicit timestamps for price_series. If not provided, 
        uses price_series.index.

    Returns:
        String like "T-24h:Imb. Up | T-12h:Balanced" or "no windows" if insufficient data.
    """
    if window_hours is None:
        window_hours = {48: "T-48h", 24: "T-24h", 12: "T-12h", 6: "T-6h"}

    # Extract price array and timestamps
    if isinstance(price_series, pd.Series):
        if timestamps is None:
            timestamps = price_series.index
        prices_arr = price_series.to_numpy(dtype=float)
    else:
        prices_arr = np.asarray(price_series, dtype=float)

    if timestamps is None:
        return "no windows"

    if len(prices_arr) == 0 or full_bands.empty:
        return "no windows"

    # Get the final timestamp
    final_ts = pd.Timestamp(timestamps[-1])
    parts = []

    # For each window in reverse order (largest first)
    for wh in sorted(window_hours.keys(), reverse=True):
        start_time = final_ts - pd.Timedelta(hours=wh)
        # Extract window data
        window_mask = timestamps >= start_time
        window_prices = prices_arr[window_mask]
        window_ts = timestamps[window_mask]

        if len(window_prices) < 2:
            continue

        # Compute band position and mean reversion
        band_pos = analyze_band_position_vs_reference(
            window_prices, full_bands, window_ts
        )
        mean_rev = detect_mean_reversion_vs_reference(
            window_prices, full_bands, window_ts
        )

        # Classify regime
        regime, _conf = market_regimes(band_pos, mean_rev)
        label = window_hours[wh]
        parts.append(f"{label}:{regime}")

    return " | ".join(parts) if parts else "no windows"


def plot_price_with_regime_windows(
    ax,
    timestamps: np.ndarray | pd.Index,
    prices: np.ndarray | pd.Series,
    full_bands: pd.DataFrame,
    window_hours: dict[int, str] | None = None,
    window_colors: dict[int, str] | None = None,
    baseline_color: str = "#A9C4FF",
    baseline_alpha: float = 0.6,
    baseline_linewidth: float = 1.0,
    window_linewidth: float = 2.2,
    window_alpha: float = 0.95,
) -> None:
    """
    Plot prices with colored overlays for different lookback windows.

    Useful for visualizing market regimes across multiple time horizons.

    Parameters
    ----------
    ax:
        Matplotlib axis to plot on.
    timestamps:
        Array of timestamps (DatetimeIndex or np.ndarray).
    prices:
        Array of prices.
    full_bands:
        Full-history bands DataFrame (for context, not plotted).
    window_hours:
        Dict mapping hours to window labels. Default: {48, 36, 24, 12, 6}.
    window_colors:
        Dict mapping hours to colors. If None, uses grayscale.
    baseline_color:
        Color for full price line baseline.
    baseline_alpha:
        Alpha for baseline.
    baseline_linewidth:
        Line width for baseline.
    window_linewidth:
        Line width for window overlays.
    window_alpha:
        Alpha for window overlays.
    """
    if window_hours is None:
        window_hours = {48: "T-48h", 36: "T-36h", 24: "T-24h", 12: "T-12h", 6: "T-6h"}

    if window_colors is None:
        window_colors = {
            48: "purple",
            36: "red",
            24: "blue",
            12: "green",
            6: "black",
        }

    # Convert to pandas structures if needed
    if not isinstance(timestamps, (pd.DatetimeIndex, pd.Index)):
        timestamps = pd.DatetimeIndex(timestamps)
    if isinstance(prices, pd.Series):
        prices = prices.to_numpy(dtype=float)
    else:
        prices = np.asarray(prices, dtype=float)

    # Plot full price baseline
    ax.plot(
        timestamps,
        prices,
        color=baseline_color,
        linewidth=baseline_linewidth,
        alpha=baseline_alpha,
        zorder=3,
        label="Full Price",
    )

    # Plot each window with its color
    final_ts = timestamps[-1]
    prev_hours = 0
    for wh in sorted(window_hours.keys()):
        start_time = final_ts - pd.Timedelta(hours=wh)
        end_time = final_ts - pd.Timedelta(hours=prev_hours)

        # Build window mask
        if prev_hours == 0:
            mask = (timestamps >= start_time) & (timestamps <= end_time)
        else:
            mask = (timestamps >= start_time) & (timestamps < end_time)

        window_ts = timestamps[mask]
        window_prices = prices[mask]

        if len(window_prices) < 2:
            continue

        color = window_colors.get(wh, "gray")
        ax.plot(
            window_ts,
            window_prices,
            color=color,
            linewidth=window_linewidth,
            alpha=window_alpha,
            zorder=6,
            label=window_hours.get(wh, f"T-{wh}h"),
        )

        prev_hours = wh



# ---------------------------------------------------------------------------
# Per-Panel Drawing Functions
# ---------------------------------------------------------------------------

def draw_price_sd_volume_panel(
    ax,
    series: pd.Series,
    bands: pd.DataFrame,
    vwap_s: pd.Series | None = None,
    vol_s: pd.Series | None = None,
    bar_width_days: float = 15.0 / 1440.0,
) -> None:
    """Draw SD bands, mean line, price line, optional VWAP and volume twin-axis.

    Parameters
    ----------
    ax:
        Matplotlib Axes to draw on.
    series:
        Price series with datetime index.
    bands:
        DataFrame from sd_bands_rolling() augmented with "timestamp" and "price" columns.
    vwap_s:
        Optional VWAP series aligned to series.index.
    vol_s:
        Optional volume series aligned to series.index.
    bar_width_days:
        Width of volume bars in days (used for volume twin-axis bars).
    """
    t_full = bands["timestamp"].values
    ax.fill_between(t_full, bands["-3sd"], bands["+3sd"], alpha=0.06, color="#B6B6B6")
    ax.fill_between(t_full, bands["-2sd"], bands["+2sd"], alpha=0.08, color="#BABABA")
    ax.fill_between(t_full, bands["-1sd"], bands["+1sd"], alpha=0.12, color="#909090")
    ax.plot(t_full, bands["mean"], color="black", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.plot(bands["timestamp"], bands["price"], color="#A9C4FF", linewidth=1.2, alpha=0.95, zorder=3)

    if vwap_s is not None and len(vwap_s.dropna()) > 0:
        vwap_clean = vwap_s.reindex(series.index)
        ax.plot(vwap_clean.index, vwap_clean.values, color="#FFA500", linewidth=1.0,
                alpha=0.8, linestyle="-", zorder=4, label="VWAP")

    if vol_s is not None and len(vol_s) > 0:
        ax_vol_twin = ax.twinx()
        ax_vol_twin.bar(
            vol_s.index, vol_s.values,
            width=bar_width_days, color="#8B0000", alpha=0.35,
            align="center", edgecolor="none",
        )
        ax_vol_twin.set_ylabel("vol", fontsize=7)
        max_vol = float(np.nanmax(vol_s.to_numpy(dtype=float))) if len(vol_s) else 1.0
        ax_vol_twin.set_ylim(0, max_vol * 1.1 if max_vol > 0 else 1.0)
        ax_vol_twin.tick_params(axis="y", labelsize=7)

        try:
            vol_aligned = vol_s.reindex(series.index)
            valid_updates = vol_aligned.fillna(0.0) > 0.0
            if int(valid_updates.sum()) >= 1:
                updates = vol_aligned[valid_updates]
                cum_mean_updates = updates.expanding(min_periods=1).mean()
                cum_mean_display = cum_mean_updates.reindex(series.index).ffill()
                if cum_mean_display.isna().any():
                    try:
                        first_val = float(cum_mean_updates.iloc[0])
                    except Exception:
                        first_val = float("nan")
                    if np.isfinite(first_val):
                        cum_mean_display = cum_mean_display.fillna(first_val)
                finite_vals = cum_mean_display.to_numpy(dtype=float)
                if np.isfinite(finite_vals).any():
                    ax_vol_twin.plot(
                        cum_mean_display.index, cum_mean_display.values,
                        color="#00008B", linewidth=0.7, alpha=0.75, linestyle="--", zorder=12,
                    )
        except Exception:
            pass


def draw_regime_overlays(
    ax,
    mdf: pd.DataFrame,
    bands: pd.DataFrame,
    window_hours: dict,
    window_colors: dict,
) -> str:
    """Overlay colored line segments for each lookback window on the price axis.

    Parameters
    ----------
    ax:
        The same Axes used for the price panel.
    mdf:
        DataFrame with "timestamp" and "price" columns.
    bands:
        Full-history bands DataFrame (from sd_bands_rolling).
    window_hours:
        Dict mapping hours (int) to label strings, e.g. {24: "T-24h", 6: "T-6h"}.
    window_colors:
        Dict mapping hours (int) to color strings.

    Returns
    -------
    str
        Regime summary string from compute_window_regime_summary.
    """
    final_ts = mdf["timestamp"].max()
    prev_hours = 0
    for wh in sorted(window_hours.keys()):
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
            sub["timestamp"], sub["price"],
            color=window_colors.get(wh, "gray"),
            linewidth=2.0, alpha=0.95, zorder=6,
        )

    # Normalize timestamps for compute_window_regime_summary (strip tz)
    mdf_local = mdf.copy()
    bands_local = bands.copy()
    for col in ("timestamp",):
        if col in mdf_local.columns:
            mdf_local[col] = pd.to_datetime(mdf_local[col], utc=True).dt.tz_convert(None)
        if col in bands_local.columns:
            bands_local[col] = pd.to_datetime(bands_local[col], utc=True).dt.tz_convert(None)

    ts_arr = mdf_local["timestamp"].to_numpy()
    prices_arr = mdf_local["price"].to_numpy(dtype=float)
    window_labels = {wh: f"T-{wh}h" for wh in window_hours.keys()}
    return compute_window_regime_summary(prices_arr, bands_local, window_hours=window_labels, timestamps=ts_arr)


def draw_volume_profile_inset(
    ax,
    series: pd.Series,
    vol_s: pd.Series | None,
    inset_bounds: tuple[float, float, float, float] = (-0.16, 0.0, 0.14, 1.0),
) -> None:
    """Draw a volume profile as a left-side inset on an existing axes.

    Parameters
    ----------
    ax:
        Host Axes (typically the price panel).
    series:
        Price series used for y-axis extent.
    vol_s:
        Volume series aligned to series.index. If None, uniform weights are used.
    inset_bounds:
        (x0, y0, width, height) in axes-fraction coordinates for the inset.
    """
    vp_centers, vp_hist = volume_profile(series, vol_s)
    if len(vp_centers) == 0 or float(np.nanmax(vp_hist)) == 0.0:
        return
    ax_profile = ax.inset_axes(list(inset_bounds), transform=ax.transAxes)
    vp_norm = vp_hist / float(np.nanmax(vp_hist))
    h = (float(np.median(np.diff(vp_centers))) * 0.9
         if len(vp_centers) > 1
         else float(max((series.max() - series.min()) * 0.02, 1e-4)))
    ax_profile.barh(vp_centers, vp_norm, height=h, color="#6C6C6C", alpha=0.35)
    ax_profile.set_ylim(ax.get_ylim())
    ax_profile.set_xlim(1.05, 0.0)
    ax_profile.set_xticks([])
    ax_profile.tick_params(axis="y", left=False, labelleft=False)
    ax_profile.set_facecolor("none")
    for side in ["left", "top", "bottom"]:
        ax_profile.spines[side].set_visible(False)
    ax_profile.spines["right"].set_alpha(0.25)


def draw_trade_markers(ax, market_trades: pd.DataFrame) -> str:
    """Draw entry/exit scatter markers and connector lines on an axes.

    Parameters
    ----------
    ax:
        Matplotlib Axes to draw on.
    market_trades:
        DataFrame with columns: entry_time, exit_time, entry_price, exit_price, pnl.
        May be empty.

    Returns
    -------
    str
        Trade info string for use in titles, e.g. "trades=3, avgPnL=+0.123".
    """
    if market_trades.empty:
        return "trades=0"
    ax.scatter(
        pd.to_datetime(market_trades["entry_time"]), market_trades["entry_price"],
        marker="v", s=36, color="#8B0000", zorder=8,
    )
    ax.scatter(
        pd.to_datetime(market_trades["exit_time"]), market_trades["exit_price"],
        marker="x", s=40, color="#111111", zorder=9,
    )
    for _, tr in market_trades.iterrows():
        ax.plot(
            [pd.to_datetime(tr["entry_time"]), pd.to_datetime(tr["exit_time"])],
            [tr["entry_price"], tr["exit_price"]],
            color="#6B6B6B", linewidth=0.8, alpha=0.7, zorder=7,
        )
    trade_count = len(market_trades)
    mean_pnl = float(market_trades["pnl"].mean())
    return f"trades={trade_count}, avgPnL={mean_pnl:+.3f}"


def draw_volume_imbalance_panel(
    ax,
    ratio_pct: pd.Series,
    entry_times=None,
    entry_values: np.ndarray | None = None,
    lookback_label: int | str = "?",
) -> None:
    """Draw volume imbalance % panel.

    Parameters
    ----------
    ax:
        Matplotlib Axes.
    ratio_pct:
        Volume imbalance series (values between roughly −1 and 1).
    entry_times:
        Optional trade entry timestamps for scatter markers.
    entry_values:
        Indicator values at entry_times for scatter markers.
    lookback_label:
        Rolling lookback for the panel title.
    """
    ax.plot(ratio_pct.index, ratio_pct.values, color="#5A5A5A", linewidth=0.8)
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.7, alpha=0.6)
    if entry_times is not None and entry_values is not None:
        ax.scatter(entry_times, entry_values, marker="v", s=22, color="#8B0000", zorder=6)
    ax.set_title(
        f"vol imbalance % (above−below)/total · rolling {lookback_label} bars",
        fontsize=7, loc="left", pad=2,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.18)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))


def draw_vwap_slope_panel(
    ax,
    slope_series: pd.Series,
    entry_times=None,
    entry_values: np.ndarray | None = None,
    threshold: float | None = None,
    mode_label: str = "raw",
    lookback_label: int | str = "?",
    update_pct: float = 0.0,
) -> None:
    """Draw VWAP slope panel with symlog y-axis.

    Parameters
    ----------
    ax:
        Matplotlib Axes.
    slope_series:
        VWAP slope series.
    entry_times:
        Optional trade entry timestamps.
    entry_values:
        Slope values at entry_times.
    threshold:
        Optional threshold drawn as a red dashed hline.
    mode_label:
        Slope mode string for the title (e.g. "raw" or "normalized, vpp=1.0").
    lookback_label:
        Lookback bars for the title.
    update_pct:
        Fraction of bars with valid VWAP+volume data (for title).
    """
    slope_display = slope_series.ffill()
    ax.plot(slope_display.index, slope_display.values, color="#708090", linewidth=0.9,
            alpha=0.55, linestyle="--")
    ax.plot(slope_series.index, slope_series.values, color="#2F4F4F", linewidth=1.1)
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.9, alpha=0.8)
    if threshold is not None:
        ax.axhline(float(threshold), color="#8B0000", linestyle=":", linewidth=1.0, alpha=0.9)
    if entry_times is not None and entry_values is not None:
        ax.scatter(entry_times, entry_values, marker="v", s=24, color="#8B0000", zorder=6)

    slope_vals = slope_series.to_numpy(dtype=float)
    finite_vals = slope_vals[np.isfinite(slope_vals)]
    abs_nonzero = np.abs(finite_vals[np.abs(finite_vals) > 0.0]) if len(finite_vals) else np.array([])
    if len(abs_nonzero) > 0:
        linthresh = float(np.nanpercentile(abs_nonzero, 35))
        linthresh = max(linthresh, 1e-6)
        ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0)

    if len(finite_vals) >= 8:
        lo = float(np.nanpercentile(finite_vals, 2.0))
        hi = float(np.nanpercentile(finite_vals, 98.0))
        if threshold is not None:
            lo = min(lo, float(threshold))
            hi = max(hi, float(threshold))
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
        if hi > lo:
            pad = (hi - lo) * 0.12
            ax.set_ylim(lo - pad, hi + pad)

    ax.set_title(
        f"VWAP slope [{mode_label}] · symlog y · lb={lookback_label} bars · upd={update_pct:.1f}%",
        fontsize=7, loc="left", pad=2,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))


def draw_mean_reversion_panel(
    ax,
    mr_score: pd.Series,
    entry_times=None,
    entry_values: np.ndarray | None = None,
    threshold: float | None = 0.5,
    window_label: int | str = "?",
) -> None:
    """Draw mean-reversion score panel.

    Parameters
    ----------
    ax:
        Matplotlib Axes.
    mr_score:
        Mean-reversion score series (0..1).
    entry_times:
        Optional trade entry timestamps.
    entry_values:
        Score values at entry_times.
    threshold:
        Optional threshold drawn as a red dotted hline.
    window_label:
        Rolling window for the title.
    """
    ax.plot(mr_score.index, mr_score.values, color="#7A5C00", linewidth=1.1)
    if threshold is not None:
        ax.axhline(float(threshold), color="#8B0000", linestyle=":", linewidth=1.0, alpha=0.9)
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.9, alpha=0.6)
    if entry_times is not None and entry_values is not None:
        ax.scatter(entry_times, entry_values, marker="v", s=22, color="#8B0000", zorder=6)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"mean-rev score · window={window_label} · thr={threshold if threshold is not None else 'n/a'}",
        fontsize=7, loc="left", pad=2,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
