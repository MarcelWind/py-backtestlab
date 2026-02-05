from datetime import datetime
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt

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


def load_prices() -> pd.DataFrame:
    """Load close prices for all assets in the universe."""
    prices = pd.DataFrame({sym: load_ohlcv(sym)["close"] for sym in UNIVERSE})
    print(f"Loaded {len(prices.columns)} assets, {len(prices)} days\n")
    return prices


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


def save_plots(results: dict[str, dict], timestamp: str) -> None:
    """Generate and save all plots."""
    names = list(results.keys())

    # Backtest plot for first strategy
    first_name = names[0]
    fig1 = plot_backtest(results[first_name], title=first_name)
    fig1.savefig(RESULTS_DIR / f"backtest_{timestamp}.png", dpi=150)
    print(f"\nSaved: results/backtest_{timestamp}.png")

    # Comparison plot (all strategies)
    fig2 = plot_comparison(results, title="Strategy Comparison")
    fig2.savefig(RESULTS_DIR / f"comparison_{timestamp}.png", dpi=150)
    print(f"Saved: results/comparison_{timestamp}.png")

    # Scatter correlation (first vs last strategy)
    if len(names) >= 2:
        fig3 = plot_scatter_correlation(
            results_a=results[names[0]],
            results_b=results[names[-1]],
            window=30,
            label_a=names[0],
            label_b=names[-1],
            title=f"{names[0]} vs {names[-1]}",
            use_roc=True,
        )
        fig3.savefig(RESULTS_DIR / f"correlation_{timestamp}.png", dpi=150)
        print(f"Saved: results/correlation_{timestamp}.png")

    # Return distribution
    fig4 = plot_return_distribution(
        results,
        window=10,
        use_roc=True,
        title="10-Day Return Distribution",
    )
    fig4.savefig(RESULTS_DIR / f"distribution_{timestamp}.png", dpi=150)
    print(f"Saved: results/distribution_{timestamp}.png")


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

    # Run optimization
    best_params = run_optimization(prices)

    # Run optimized strategy vs benchmark
    print(f"\nRunning backtest with best parameters: {best_params}")
    results = {
        "Optimized": run_optimized_backtest(prices, best_params),
        "Buy&Hold": Backtester(BuyAndHoldStrategy(), rebalance_freq=9999).run(prices),
    }

    print_comparison(results)
    save_plots(results, timestamp)
    save_metrics(results, timestamp)
    plt.show()


if __name__ == "__main__":
    main()
