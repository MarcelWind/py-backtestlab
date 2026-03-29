"""Reporting and diagnostics."""

from .metrics import compute_metrics
from .plot import (
    plot_backtest,
    plot_comparison,
    plot_cumulative_log_return,
    plot_mcpt_histogram,
    plot_mcpt_permutation_overlay,
    plot_return_distribution,
    plot_rolling_correlation,
    plot_scatter_correlation,
    compute_window_regime_summary,
    plot_price_with_regime_windows,
)