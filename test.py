import matplotlib.pyplot as plt
from stratlab.data.fred import compute_global_m2

# Fetch G4 global M2 (US + EU + JP + CN, FX-adjusted)
gm2 = compute_global_m2("2010-01-01")

# --- Plot ---
fig, axes = plt.subplots(4, 1, figsize=(14, 16))
fig.suptitle("G4 Global M2 (FX-Adjusted)", fontsize=16, fontweight="bold")

# 1) Global M2 total
ax = axes[0]
ax.plot(gm2.index, gm2["global_m2"], color="tab:blue", linewidth=2)
ax.set_ylabel("Trillions USD")
ax.set_title("Global M2 (US + EU + JP + CN)")
ax.grid(alpha=0.3)

# 2) Stacked area by region
ax = axes[1]
ax.stackplot(
    gm2.index,
    gm2["us"], gm2["eu"], gm2["jp"], gm2["cn"],
    labels=["US", "Eurozone", "Japan", "China"],
    alpha=0.8,
)
ax.set_ylabel("Trillions USD")
ax.set_title("Regional Breakdown")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

# 3) YoY % change
ax = axes[2]
yoy = gm2["global_m2"].pct_change(12) * 100
ax.bar(yoy.index, yoy, width=25, color=yoy.apply(lambda x: "tab:green" if x > 0 else "tab:red"), alpha=0.8)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("YoY Change (%)")
ax.set_title("Global M2 — Year-over-Year Growth")
ax.grid(alpha=0.3)

# 4) Z-score (120-month rolling)
ax = axes[3]
m2 = gm2["global_m2"]
z = (m2 - m2.rolling(120).mean()) / m2.rolling(120).std()
ax.plot(z.index, z, color="tab:purple", linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.5)
ax.axhline(1, color="gray", linewidth=0.5, linestyle="--")
ax.axhline(-1, color="gray", linewidth=0.5, linestyle="--")
ax.axhline(2, color="gray", linewidth=0.5, linestyle="--")
ax.axhline(-2, color="gray", linewidth=0.5, linestyle="--")
ax.fill_between(z.index, z, 0, where=z > 0, color="tab:green", alpha=0.2)
ax.fill_between(z.index, z, 0, where=z < 0, color="tab:red", alpha=0.2)
ax.set_ylabel("Z-Score")
ax.set_title("Global M2 — Z-Score (120-Month Rolling)")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
