#!/usr/bin/env python
"""Diagnostic script to inspect price data in detail."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.weather_backtest import load_event_ohlcv_resampled


def inspect_price_data():
    """Load event data and inspect price/OHLC data."""
    event_slug = "highest-temperature-in-atlanta-on-february-20-2026"
    
    print(f"Loading event: {event_slug}")
    prices, vwap, volume, buy_volume, sell_volume, high, low, open_ = load_event_ohlcv_resampled(
        event_slug,
        prefer_outcome="no",
    )
    
    print(f"\n=== PRICE DATA ===")
    print(f"Shape: {prices.shape}")
    print(f"NaN count per asset:")
    for col in prices.columns:
        nan_count = prices[col].isna().sum()
        print(f"  {col}: {nan_count} / {len(prices)}")
    
    print(f"\nSample prices for first asset (67-f-or-below):")
    asset = prices.columns[0]
    sample = prices[asset].dropna().head(10)
    for idx, val in sample.items():
        print(f"  {idx}: {val}")
    
    print(f"\n=== HIGH DATA ===")
    print(f"Shape: {high.shape}")
    print(f"NaN count per asset:")
    for col in high.columns:
        nan_count = high[col].isna().sum()
        print(f"  {col}: {nan_count} / {len(high)}")
    
    print(f"\nSample highs for first asset:")
    asset = high.columns[0]
    sample = high[asset].dropna().head(10)
    for idx, val in sample.items():
        print(f"  {idx}: {val}")
    
    print(f"\n=== LOW DATA ===")
    print(f"Shape: {low.shape}")
    print(f"NaN count per asset:")
    for col in low.columns:
        nan_count = low[col].isna().sum()
        print(f"  {col}: {nan_count} / {len(low)}")
    
    print(f"\nSample lows for first asset:")
    asset = low.columns[0]
    sample = low[asset].dropna().head(10)
    for idx, val in sample.items():
        print(f"  {idx}: {val}")
    
    print(f"\n=== OPEN DATA ===")
    print(f"Shape: {open_.shape}")
    print(f"NaN count per asset:")
    for col in open_.columns:
        nan_count = open_[col].isna().sum()
        print(f"  {col}: {nan_count} / {len(open_)}")
    
    print(f"\nSample opens for first asset:")
    asset = open_.columns[0]
    sample = open_[asset].dropna().head(10)
    for idx, val in sample.items():
        print(f"  {idx}: {val}")
    
    # Check alignment
    print(f"\n=== ALIGNMENT CHECK ===")
    print(f"Prices index: {prices.index[0]} to {prices.index[-1]}")
    print(f"High index: {high.index[0]} to {high.index[-1]}")
    print(f"Low index: {low.index[0]} to {low.index[-1]}")
    print(f"Open index: {open_.index[0]} to {open_.index[-1]}")
    print(f"Indices equal: {prices.index.equals(high.index) and high.index.equals(low.index) and low.index.equals(open_.index)}")


if __name__ == "__main__":
    inspect_price_data()
