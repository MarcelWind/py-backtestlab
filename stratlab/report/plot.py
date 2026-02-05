"""Portfolio visualization and plotting."""

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
