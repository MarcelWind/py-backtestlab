#!/usr/bin/env python
"""Diagnostic script to inspect band data in a recent backtest."""

import sys
from pathlib import Path
import pandas as pd
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.weather_backtest import load_event_ohlcv_resampled
from strategies.weather_market_strategy import WeatherMarketImbalanceStrategy
from stratlab.strategy.indicators import SdBands


def inspect_band_data():
    """Load event data and inspect SdBands computation."""
    event_slug = "highest-temperature-in-atlanta-on-february-20-2026"
    
    print(f"Loading event: {event_slug}")
    prices, vwap, volume, buy_volume, sell_volume, high, low, open_ = load_event_ohlcv_resampled(
        event_slug,
        prefer_outcome="no",
    )
    
    print(f"\nPrice data shape: {prices.shape}")
    print(f"Price data columns: {list(prices.columns)}")
    print(f"Price index range: {prices.index[0]} to {prices.index[-1]}")
    print(f"Price duplicates in index: {prices.index.duplicated().sum()}")
    
    # Create strategy to access SdBands
    strategy = WeatherMarketImbalanceStrategy.from_profile(
        "balanced",
        vwap=vwap,
        volume=volume,
        high=high,
        low=low,
        open_=open_,
        buy_volume=buy_volume,
        sell_volume=sell_volume,
    )
    
    # Find SdBands in indicator_defs
    sdbands_ind = None
    for ind in strategy.indicator_defs:
        if ind.name == "sd_bands":
            sdbands_ind = ind
            break
    
    if sdbands_ind is None:
        print("\nERROR: SdBands not found in indicator_defs!")
        return
    
    print(f"\nSdBands indicator found")
    print(f"  pricing_method: {sdbands_ind.pricing_method}")
    print(f"  has high data: {sdbands_ind._high is not None}")
    print(f"  has low data: {sdbands_ind._low is not None}")
    print(f"  has open data: {sdbands_ind._open is not None}")
    
    # Manually compute bands like plotting.py does
    print(f"\nComputing SdBands for all {len(prices)} bars...")
    returns = prices.pct_change()
    for i in range(len(prices)):
        if i % 100 == 0:
            print(f"  Bar {i}/{len(prices)}")
        sdbands_ind.compute(prices, returns, i)
    
    # Check band_series
    band_series = sdbands_ind.band_series
    print(f"\nBand series after computation:")
    print(f"  Keys: {list(band_series.keys())}")
    print(f"  Total assets: {len(band_series)}")
    
    if not band_series:
        print("\n  ⚠️  WARNING: band_series is empty!")
        print(f"  _history contents: {list(sdbands_ind._history.keys())}")
        return
    
    # Show stats for first 2 assets
    for i, (asset, df) in enumerate(list(band_series.items())[:2]):
        print(f"\n  {asset}:")
        print(f"    Shape: {df.shape}")
        print(f"    Columns: {list(df.columns)}")
        print(f"    Index range: {df.index[0]} to {df.index[-1]}")
        print(f"    Duplicate indices: {df.index.duplicated().sum()}")
        print(f"    Sample values (first 3 rows):")
        for idx, row in df.head(3).iterrows():
            print(f"      {idx}: mean={row['mean']:.2f}, +1sd={row['+1sd']:.2f}, -1sd={row['-1sd']:.2f}")


if __name__ == "__main__":
    inspect_band_data()
