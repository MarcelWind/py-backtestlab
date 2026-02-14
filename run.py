from datetime import datetime
from pathlib import Path
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

from fetch_data import load_zip

def load_prices(event_slug: str = "", market: str = "") -> pd.DataFrame:
    """Load OHLCV data from the zip and pivot to a price DataFrame."""
    df = load_zip()
    if event_slug:
        df = df[df["event_slug"] == event_slug]
    if market:
        df = df[df["market"] == market]
    return df


def main():
    df = load_prices('highest-temperature-in-ankara-on-february-10-2026', '1-c-or-below')
    print("Loaded price data:")
    print(df.head())

if __name__ == "__main__":
    main()
