#!/usr/bin/env python3
"""Download META (Facebook) data and test the cumulative volume delta plot."""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception:
    raise

from stratlab.report.plot import draw_cumulative_volume_delta_panel

# Download last 30 days of 1h data
ticker = "META"
period = "30d"
interval = "1h"

OUT = Path("test") / f"{ticker.lower()}_{interval}_{period}_cvd_test.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

print(f"Downloading {ticker} {period} {interval}...")
df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
if df is None or df.empty:
    raise SystemExit("No data downloaded")

# Keep Close and Volume
df = df[['Close', 'Volume']].dropna()
if df.empty:
    raise SystemExit("No Close/Volume data")

# Approximate signed volume delta: volume * sign(price change)
price = df['Close']
vol = df['Volume']
price_diff = price.diff().fillna(0.0)
# Compute a simple signed-volume proxy: volume * sign(price change)
sig = np.sign(price_diff.to_numpy(dtype=float))
if len(sig) > 0:
    sig[0] = 1.0
delta_arr = vol.to_numpy(dtype=float).ravel() * sig.ravel()
delta_series = pd.Series(delta_arr, index=df.index)

# Compute bar width in days similar to plotting utilities
try:
    idx_dt = pd.DatetimeIndex(df.index)
    median_delta = idx_dt.to_series().diff().median().to_timedelta64()
    bar_width_days = float(pd.Timedelta(median_delta) / pd.Timedelta(days=1)) * 0.8
except Exception:
    bar_width_days = 1.0 / 24.0

# Make figure
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
ax0.plot(price.index, price.values, color='steelblue')
ax0.set_title(f"{ticker} price")
ax0.grid(True, alpha=0.3)

# draw our CVD panel
print("Drawing cumulative volume delta panel...")
draw_cumulative_volume_delta_panel(ax1, delta_series, bar_width_days=bar_width_days, title=f"{ticker} signed vol Δ")

plt.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"Saved plot to {OUT}")
