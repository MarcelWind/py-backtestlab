"""Portfolio visualization and plotting."""

import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_backtest(
    results: dict,
    title: str = "Portfolio Backtest",
    figsize: tuple = (14, 10),
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
    figsize: tuple = (14, 6),
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
    returns_a: pd.Series,
    returns_b: pd.Series,
    window: int = 30,
    title: str = "Rolling Window Correlation",
    figsize: tuple = (14, 6),
    label_a: str = "Strategy",
    label_b: str = "Benchmark",
) -> Figure:
    """
    Plot correlation between two return series over non-overlapping windows.

    Computes correlation for each window of length n, showing how the
    relationship between strategies changes over time.

    Args:
        returns_a: First return series (e.g., active strategy)
        returns_b: Second return series (e.g., buy-and-hold benchmark)
        window: Window size in periods (e.g., 30 days)
        title: Plot title
        figsize: Figure size
        label_a: Label for first series
        label_b: Label for second series

    Returns:
        Matplotlib Figure
    """
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
    figsize: tuple = (14, 8),
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
            window_ret = (1 + returns.iloc[start_idx:end_idx]).prod() - 1
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
    colors = plt.cm.tab10.colors

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

    colors = plt.cm.tab10.colors
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
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
