from datetime import datetime
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from stratlab.config import RESULTS_DIR
from stratlab.data import load_ohlcv
from stratlab.universe.selector import UNIVERSE
from stratlab.strategy import MomentumStrategy, SharpeStrategy, BuyAndHoldStrategy
from stratlab.backtest.backtester import Backtester
from stratlab.optimize import optimize
from stratlab.report.plot import (
    plot_backtest,
    plot_comparison,
    plot_scatter_correlation,
    plot_return_distribution,
)
from stratlab.report.ui import BacktestUI



def load_prices() -> pd.DataFrame:
    """Load close prices for all assets in the universe."""
    prices = pd.DataFrame({sym: load_ohlcv(sym)["close"] for sym in UNIVERSE})
    print(f"Loaded {len(prices.columns)} assets, {len(prices)} days\n")
    return prices


def split_data(
    prices: pd.DataFrame,
    train_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split price data into in-sample (training) and out-of-sample (test) sets.

    Args:
        prices: Full price DataFrame
        train_ratio: Fraction of data for training (default 70%)

    Returns:
        Tuple of (train_prices, test_prices)
    """
    split_idx = int(len(prices) * train_ratio)
    train = prices.iloc[:split_idx]
    test = prices.iloc[split_idx:]

    print(f"Data split: {len(train)} days IS, {len(test)} days OOS")
    print(f"  IS:  {train.index[0].date()} to {train.index[-1].date()}")
    print(f"  OOS: {test.index[0].date()} to {test.index[-1].date()}")

    return train, test


def run_backtests(prices: pd.DataFrame) -> dict[str, dict]:
    """Run all strategy backtests and return results dict."""
    strategies = {
        "Momentum": (MomentumStrategy(lookback=30, top_n=3), 7),
        "Sharpe-Opt": (SharpeStrategy(lookback=60), 30),
        "Buy&Hold": (BuyAndHoldStrategy(), 9999),
    }

    results = {}
    for name, (strategy, rebalance_freq) in strategies.items():
        bt = Backtester(strategy=strategy, rebalance_freq=rebalance_freq)
        results[name] = bt.run(prices)

    return results


def print_comparison(results: dict[str, dict]) -> None:
    """Print metrics comparison table."""
    print("=" * 50)
    print("COMPARISON")
    print("=" * 50)

    names = list(results.keys())
    header = f"  {'Metric':<20}" + "".join(f"{n:>12}" for n in names)
    print(header)
    print(f"  {'-'*20}" + " ".join(f"{'-'*12}" for _ in names))

    for metric in ["sharpe", "sortino", "calmar", "total_return", "max_drawdown"]:
        values = [results[n]["metrics"].get(metric, 0) for n in names]
        row = f"  {metric:<20}" + "".join(f"{v:>12.4f}" for v in values)
        print(row)


def create_plots(results: dict[str, dict]) -> list[tuple[str, Figure]]:
    """Generate all plots and return as list of (name, figure) tuples."""
    figures: list[tuple[str, Figure]] = []
    names = list(results.keys())

    # Get data length from first result
    data_len = len(results[names[0]]["returns"])

    # Backtest plot for first strategy
    first_name = names[0]
    fig1 = plot_backtest(results[first_name], title=first_name)
    figures.append(("Backtest", fig1))

    # Comparison plot (all strategies)
    fig2 = plot_comparison(results, title="Strategy Comparison")
    figures.append(("Comparison", fig2))

    # Scatter correlation (first vs last strategy) - need enough data for windowing
    scatter_window = min(30, data_len // 5)  # At least 5 windows
    if len(names) >= 2 and scatter_window >= 5:
        fig3 = plot_scatter_correlation(
            results_a=results[names[0]],
            results_b=results[names[-1]],
            window=scatter_window,
            label_a=names[0],
            label_b=names[-1],
            title=f"{names[0]} vs {names[-1]}",
            use_roc=True,
        )
        figures.append(("Correlation", fig3))

    # Return distribution - adjust window for short datasets
    dist_window = min(10, data_len // 10)  # At least 10 windows
    if dist_window >= 5:
        fig4 = plot_return_distribution(
            results,
            window=dist_window,
            use_roc=True,
            title=f"{dist_window}-Day Return Distribution",
        )
        figures.append(("Distribution", fig4))

    return figures


def save_plots(figures: list[tuple[str, Figure]], timestamp: str) -> None:
    """Save all plots to disk."""
    for name, fig in figures:
        filename = f"{name.lower()}_{timestamp}.png"
        fig.savefig(RESULTS_DIR / filename, dpi=150)
        print(f"Saved: results/{filename}")


def save_metrics(results: dict[str, dict], timestamp: str) -> None:
    """Save metrics to CSV."""
    metrics_df = pd.DataFrame({name: r["metrics"] for name, r in results.items()}).T
    metrics_df.to_csv(RESULTS_DIR / f"metrics_{timestamp}.csv")
    print(f"Saved: results/metrics_{timestamp}.csv")


def run_optimization(prices: pd.DataFrame) -> dict[str, Any]:
    """Run Monte Carlo parameter optimization."""
    print("\n" + "=" * 50)
    print("MONTE CARLO OPTIMIZATION")
    print("=" * 50)

    opt_result = optimize(
        strategy_class=MomentumStrategy,
        param_space={"lookback": (10, 90), "top_n": (1, 5)},
        prices=prices,
        objective="total_return",
        n_trials=50,
        rebalance_freq=(7, 30),
        seed=42,
        verbose=True,
    )

    print(f"\nBest parameters: {opt_result.best_params}")
    print(f"Best Total Return: {opt_result.best_score:.4f}")
    print(f"\nTop 5 trials:")
    print(opt_result.top_n(5)[["lookback", "top_n", "_rebalance_freq", "_objective"]])
    return opt_result.best_params


def run_optimized_backtest(prices: pd.DataFrame, best_params: dict[str, Any]) -> dict:
    """Run backtest with optimized parameters."""
    strategy_params = {k: v for k, v in best_params.items() if k != "rebalance_freq"}
    rebalance_freq = best_params.get("rebalance_freq", 30)

    strategy = MomentumStrategy(**strategy_params)
    bt = Backtester(strategy=strategy, rebalance_freq=rebalance_freq)
    return bt.run(prices)


def main() -> None:
    """Main entry point."""
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    prices = load_prices()

    # Split into in-sample and out-of-sample
    prices_is, prices_oos = split_data(prices, train_ratio=0.7)

    # Optimize on in-sample data
    best_params = run_optimization(prices_is)

    # Test on out-of-sample data
    print(f"\n{'=' * 50}")
    print("OUT-OF-SAMPLE VALIDATION")
    print("=" * 50)
    print(f"Best parameters: {best_params}")

    results_oos = {
        "Optimized": run_optimized_backtest(prices_oos, best_params),
        "Buy&Hold": Backtester(BuyAndHoldStrategy(), rebalance_freq=9999).run(prices_oos),
    }

    print("\nOOS Results:")
    print_comparison(results_oos)

    # Also show IS results for comparison
    print("\nIS Results (for reference):")
    results_is = {
        "Optimized": run_optimized_backtest(prices_is, best_params),
        "Buy&Hold": Backtester(BuyAndHoldStrategy(), rebalance_freq=9999).run(prices_is),
    }
    print_comparison(results_is)

    # Create and save plots
    figures = create_plots(results_oos)
    print()
    save_plots(figures, timestamp)
    save_metrics(results_oos, timestamp)

    # Show plots with enhanced UI
    ui = BacktestUI(figures, metrics_data=results_oos["Optimized"]["metrics"])
    ui.show()


if __name__ == "__main__":
    main()
