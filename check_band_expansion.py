#!/usr/bin/env python
"""Check if bands expand over time."""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.weather_backtest import load_event_ohlcv_resampled
from strategies.weather_market_imbalance import WeatherMarketImbalanceStrategy


def check_band_expansion():
    """Check if bands expand over time."""
    event_slug = "highest-temperature-in-atlanta-on-february-20-2026"
    
    prices, vwap, volume, buy_volume, sell_volume, high, low, open_ = load_event_ohlcv_resampled(
        event_slug,
        prefer_outcome="no",
    )
    
    strategy = WeatherMarketImbalanceStrategy.from_profile(
        "balanced",
        vwap=vwap,
        volume=volume,
        high=high,
        low=low,
        open_=open_,
    )
    
    sdbands_ind = None
    for ind in strategy.indicator_defs:
        if ind.name == "sd_bands":
            sdbands_ind = ind
            break
    
    returns = prices.pct_change()
    
    # Compute bands
    print("Computing bands...")
    for i in range(len(prices)):
        sdbands_ind.compute(prices, returns, i)
    
    band_series = sdbands_ind.band_series
    asset = '67-f-or-below'
    df = band_series[asset]
    
    print(f"\nBand data for {asset}:")
    print(f"Showing rows at indices: 0, 100, 500, 1000, 2000, 2990")
    print()
    
    indices_to_show = [0, 100, 500, 1000, 2000, 2990]
    for idx in indices_to_show:
        row = df.iloc[idx]
        print(f"Bar {idx} ({df.index[idx]}):")
        print(f"  mean={row['mean']:.6f}")
        print(f"  +1sd={row['+1sd']:.6f}")
        print(f"  -1sd={row['-1sd']:.6f}")
        band_width = row['+1sd'] - row['-1sd']
        print(f"  band_width={band_width:.6f}")
        print()


if __name__ == "__main__":
    check_band_expansion()
