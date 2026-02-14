from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt

from stratlab.config import RESULTS_DIR
from stratlab.data import load_ohlcv
from stratlab.universe.selector import UNIVERSE
from stratlab.backtest.backtester import Backtester
from stratlab.optimize import optimize
from stratlab.report.plot import (
    plot_backtest,
    plot_comparison,
    plot_scatter_correlation,
    plot_return_distribution,
)


from strategies.weather_market_imbalance import WeatherMarketImbalanceStrategy
from fetch_data import load_zip

def load_prices(event_slug: str = "", market: str = "") -> pd.DataFrame:
    """Load OHLCV data from the zip and pivot to a price DataFrame."""
    df = load_zip()
    if event_slug:
        df = df[df["event_slug"] == event_slug]
    if market:
        df = df[df["market"] == market]
    return df

def run_backtest(strategy: Any, prices: pd.DataFrame) -> pd.DataFrame:
    """Run a backtest for the given strategy and price data."""
    backtester = Backtester(strategy=strategy, rebalance_freq=1)
    results = backtester.run(prices)
    return pd.DataFrame(results["portfolio_returns"], index=prices.index[1:])


def main():
    prices = load_prices('highest-temperature-in-ankara-on-february-10-2026', '1-c-or-below')
    print("Loaded price data:")
    print(prices.head())

    strategy = WeatherMarketImbalanceStrategy(lookback=1)
    backtest_results = run_backtest(strategy, prices)
    print("Backtest results:")
    print(backtest_results.head())

if __name__ == "__main__":
    main()
