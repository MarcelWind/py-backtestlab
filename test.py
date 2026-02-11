import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from stratlab.data import load_ohlcv
from stratlab.data.fred import compute_global_m2

# -- 1. Load data -------------------------------------------------------------
print("Loading BTC prices...")
btc_daily = load_ohlcv("BTC")["close"]
btc_daily.index = btc_daily.index.tz_localize(None)  # strip UTC for alignment
btc_weekly = btc_daily.resample("W-MON").last()
btc_ret = btc_weekly.pct_change() * 100  # weekly % return

print("Loading Global M2...")
gm2 = compute_global_m2("2010-01-01")
m2_yoy = gm2["global_m2"].pct_change(12) * 100  # YoY % change (monthly)

# Upsample M2 from monthly to weekly (forward-fill)
m2_yoy_weekly = m2_yoy.resample("W-MON").ffill()

# -- 2. Align on common weekly dates ------------------------------------------
df = pd.DataFrame({
    "btc_ret": btc_ret,
    "m2_yoy": m2_yoy_weekly,
}).dropna()

# Filter to 2020 onwards
df = df["2020":]

print(f"\nOverlapping period: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} weeks)")

# -- 3. Cross-correlation at lags 0-52 weeks ----------------------------------
max_lag = 52
lags = range(0, max_lag + 1)

def lag_correlations(macro, target, lags):
    """Shift macro forward by n weeks, correlate with target."""
    results = []
    for lag in lags:
        shifted = macro.shift(lag)
        valid = pd.concat([shifted, target], axis=1).dropna()
        if len(valid) < 30:
            results.append((lag, np.nan, np.nan))
            continue
        r, p = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
        results.append((lag, r, p))
    return pd.DataFrame(results, columns=["lag", "corr", "pvalue"])

corr_m2 = lag_correlations(df["m2_yoy"], df["btc_ret"], lags)
best_m2 = corr_m2.loc[corr_m2["corr"].abs().idxmax()]

print(f"\n{'='*50}")
print("OPTIMAL LAG RESULTS (weekly)")
print(f"{'='*50}")
print(f"Global M2 YoY -> BTC:  lag={int(best_m2['lag'])}wk  r={best_m2['corr']:.3f}  p={best_m2['pvalue']:.4f}")
print(f"\nAll lags (every 4 weeks):")
for _, row in corr_m2.iterrows():
    if int(row["lag"]) % 4 == 0 or row["lag"] == best_m2["lag"]:
        marker = " <-- best" if row["lag"] == best_m2["lag"] else ""
        print(f"  {int(row['lag']):3d}wk  r={row['corr']:+.3f}  p={row['pvalue']:.4f}{marker}")

# -- 4. Plot -------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("Global M2 -> BTC Lag Correlation (Weekly)", fontsize=16, fontweight="bold")

# Panel 1: Correlation by lag
ax = axes[0]
colors = ["tab:blue" if r >= 0 else "tab:red" for r in corr_m2["corr"]]
bars = ax.bar(corr_m2["lag"], corr_m2["corr"], color=colors, alpha=0.8, edgecolor="black", linewidth=0.3)
ax.axhline(0, color="black", linewidth=0.5)
best_idx = int(best_m2["lag"])
bars[best_idx].set_edgecolor("gold")
bars[best_idx].set_linewidth(3)
ax.set_xlabel("Lag (weeks)")
ax.set_ylabel("Pearson r")
ax.set_title(f"Global M2 YoY -> BTC Weekly Returns  (best: {int(best_m2['lag'])}wk, r={best_m2['corr']:.3f})")
ax.set_xticks(range(0, max_lag + 1, 4))
ax.grid(alpha=0.3, axis="y")

# Panel 2: Time series overlay at optimal lag
ax_left = axes[1]
opt_lag = int(best_m2["lag"])
m2_shifted = df["m2_yoy"].shift(opt_lag)

ax_left.plot(df.index, m2_shifted, color="tab:blue", linewidth=1.5, label=f"M2 YoY (shifted {opt_lag}wk)")
ax_left.set_ylabel("M2 YoY Change (%)", color="tab:blue")
ax_left.tick_params(axis="y", labelcolor="tab:blue")
ax_left.grid(alpha=0.3)

ax_right = ax_left.twinx()
ax_right.plot(df.index, df["btc_ret"], color="tab:orange", linewidth=0.7, alpha=0.6, label="BTC Weekly Return")
ax_right.set_ylabel("BTC Return (%)", color="tab:orange")
ax_right.tick_params(axis="y", labelcolor="tab:orange")

lines1, labels1 = ax_left.get_legend_handles_labels()
lines2, labels2 = ax_right.get_legend_handles_labels()
ax_left.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
ax_left.set_title(f"Visual: Global M2 YoY (shifted {opt_lag}wk) vs BTC Returns")

plt.tight_layout()
plt.show()
