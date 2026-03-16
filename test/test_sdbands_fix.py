#!/usr/bin/env python
"""Quick test to verify SdBands computation fix."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratlab.strategy.indicators import SdBands, VwapVolumeImbalance, _compute_price


def test_sdbands_compute():
    """Test that SdBands.compute() populates band_series correctly."""
    # Create synthetic price data
    n_bars = 50
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="h")

    
    # Create price data for 2 assets
    prices = pd.DataFrame({
        "BTC": np.linspace(40000, 42000, n_bars) + np.random.randn(n_bars) * 100,
        "ETH": np.linspace(2000, 2200, n_bars) + np.random.randn(n_bars) * 50,
    }, index=dates)
    
    # Create OHLC data
    high = prices * 1.01
    low = prices * 0.99
    open_ = prices.shift(1).fillna(prices.iloc[0])
    
    returns = prices.pct_change()
    
    # Test with pricing_method
    sdbands = SdBands(
        pricing_method="typical",
        high=high,
        low=low,
        open_=open_,
    )
    
    # Compute for all bars
    for i in range(len(prices)):
        result = sdbands.compute(prices, returns, i)
        assert isinstance(result, pd.DataFrame), f"compute() should return DataFrame, got {type(result)}"
        assert len(result) > 0, f"Result should have rows, got {len(result)}"
        # Result should have 7 rows (mean, +1sd, -1sd, +2sd, -2sd, +3sd, -3sd)
        # and 2 columns (BTC, ETH)
        assert len(result.index) == 7, f"Expected 7 band rows, got {len(result.index)}"
    
    # Check that band_series has data
    band_series = sdbands.band_series
    print(f"band_series keys: {list(band_series.keys())}")
    assert len(band_series) == 2, f"Expected 2 assets in band_series, got {len(band_series)}"
    
    for asset, df in band_series.items():
        print(f"\n{asset}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Index: {df.index[:5]}...")
        assert df.shape[0] == n_bars, f"Expected {n_bars} rows for {asset}, got {df.shape[0]}"
        expected_cols = ["mean", "+1sd", "-1sd", "+2sd", "-2sd", "+3sd", "-3sd"]
        assert all(col in df.columns for col in expected_cols), f"Missing band columns in {asset}"
    
    print("\n✓ SdBands compute test passed!")


def test_compute_price():
    """Test _compute_price utility function."""
    close, high, low, open_ = 100, 105, 95, 98
    
    # Test close pricing
    price = _compute_price(close, high, low, open_, "close")
    assert price == close, f"close: expected {close}, got {price}"
    
    # Test median
    price = _compute_price(close, high, low, open_, "median")
    expected = (high + low) / 2
    assert price == expected, f"median: expected {expected}, got {price}"
    
    # Test typical
    price = _compute_price(close, high, low, open_, "typical")
    expected = (high + low + close) / 3
    assert abs(price - expected) < 0.01, f"typical: expected {expected}, got {price}"
    
    # Test with None values
    price = _compute_price(close, None, None, None, "typical")
    assert price == close, f"typical with None OHLC: expected {close}, got {price}"
    
    print("✓ _compute_price test passed!")


def test_vwap_volume_imbalance_uses_sdbands_mean():
    """VwapVolumeImbalance should classify bars against expanding mean from SdBands."""
    dates = pd.date_range("2024-01-01", periods=3, freq="h")
    prices = pd.DataFrame({"A": [10.0, 20.0, 30.0]}, index=dates)
    volume = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
    returns = prices.pct_change()

    sdbands = SdBands(pricing_method="close")
    for i in range(len(prices)):
        sdbands.compute(prices, returns, i)

    imbalance = VwapVolumeImbalance(volume=volume, sd_bands=sdbands, lookback=3)
    out = imbalance.compute(prices, returns, index=2)
    got = float(out["A"])

    # Expanding means are [10, 15, 20] so bars are [<=mean, >mean, >mean].
    # Ratio = (2 - 1) / (2 + 1) * 100 = 33.333...
    assert np.isfinite(got), "Expected finite imbalance value"
    assert abs(got - (100.0 / 3.0)) < 1e-6, f"Expected 33.333..., got {got}"

    print("✓ VwapVolumeImbalance mean-baseline test passed!")


if __name__ == "__main__":
    print("Testing SdBands fix...")
    test_compute_price()
    test_sdbands_compute()
    test_vwap_volume_imbalance_uses_sdbands_mean()
    print("\n✅ All tests passed!")
