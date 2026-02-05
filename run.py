from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from stratlab.config import RESULTS_DIR
from stratlab.data import load_ohlcv
from stratlab.universe.selector import UNIVERSE
from stratlab.strategy import MomentumStrategy, SharpeStrategy
from stratlab.backtest.backtester import Backtester, PortfolioBacktest
from stratlab.report.plot import plot_backtest, plot_comparison

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

# Timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load close prices for all assets
prices = pd.DataFrame({sym: load_ohlcv(sym)['close'] for sym in UNIVERSE})
print(f"Loaded {len(prices.columns)} assets, {len(prices)} days\n")

# --- Define Strategies ---
mom_strategy = MomentumStrategy(lookback=30, top_n=3)
sharpe_strategy = SharpeStrategy(lookback=60)

# --- Run with Generic Backtester ---
mom_bt = Backtester(strategy=mom_strategy, rebalance_freq=7)
mom_results = mom_bt.run(prices)

sharpe_bt = Backtester(strategy=sharpe_strategy, rebalance_freq=30)
sharpe_results = sharpe_bt.run(prices)

# --- Or use backward-compatible PortfolioBacktest ---
# sharpe_results = PortfolioBacktest(lookback=60, rebalance_freq=30).run(prices)

# --- Print Comparison ---
print("=" * 50)
print("COMPARISON")
print("=" * 50)
print(f"  {'Metric':<20} {'Momentum':>12} {'Sharpe-Opt':>12}")
print(f"  {'-'*20} {'-'*12} {'-'*12}")
for k in ['sharpe', 'sortino', 'calmar', 'total_return', 'max_drawdown']:
    m = mom_results['metrics'].get(k, 0)
    s = sharpe_results['metrics'].get(k, 0)
    print(f"  {k:<20} {m:>12.4f} {s:>12.4f}")

# --- Save Plots ---
fig1 = plot_backtest(mom_results, title="Momentum Strategy (Top 3, 30-day)")
fig1.savefig(RESULTS_DIR / f"momentum_{timestamp}.png", dpi=150)
print(f"\nSaved: results/momentum_{timestamp}.png")

fig2 = plot_comparison(
    {"Momentum": mom_results, "Sharpe-Optimized": sharpe_results},
    title="Strategy Comparison"
)
fig2.savefig(RESULTS_DIR / f"comparison_{timestamp}.png", dpi=150)
print(f"Saved: results/comparison_{timestamp}.png")

# --- Save Metrics to CSV ---
metrics_df = pd.DataFrame({
    "Momentum": mom_results['metrics'],
    "Sharpe-Optimized": sharpe_results['metrics'],
}).T
metrics_df.to_csv(RESULTS_DIR / f"metrics_{timestamp}.csv")
print(f"Saved: results/metrics_{timestamp}.csv")

plt.show()
