"""Example: plot simple candlesticks using plot_market_indicators."""
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from stratlab.report.plot import plot_candles


def make_sample_ohlc(n=60, start_price=100.0):
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="min")
    prices = np.cumsum(np.random.randn(n) * 0.2) + start_price
    opens = pd.Series(prices, index=idx).shift(1).fillna(prices[0])
    closes = pd.Series(prices, index=idx)
    highs = pd.Series(np.maximum(opens, closes) + np.abs(np.random.rand(n) * 0.3), index=idx)
    lows = pd.Series(np.minimum(opens, closes) - np.abs(np.random.rand(n) * 0.3), index=idx)

    opens = pd.DataFrame({"TEST": opens})
    highs = pd.DataFrame({"TEST": highs})
    lows = pd.DataFrame({"TEST": lows})
    closes = pd.DataFrame({"TEST": closes})
    return opens, highs, lows, closes


def main():
    opens, highs, lows, closes = make_sample_ohlc()
    fig, ax = plt.subplots(figsize=(6, 4))
    # plot_candles expects (ax, opens, highs, lows, closes, ...)
    plot_candles(
        ax,
        opens.iloc[:, 0],
        highs.iloc[:, 0],
        lows.iloc[:, 0],
        closes.iloc[:, 0],
        up_color="#7a72f0",
        down_color="#000000",
        width=None,
    )
    out = Path("test/plot_candles_example.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved candlestick example to {out}")


if __name__ == "__main__":
    main()
