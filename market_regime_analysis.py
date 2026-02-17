import argparse
from pathlib import Path
import os
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# -------------------------
# Data standardization
# -------------------------
def _standardize_df(df, inferred_market=None):
    df = df.copy()
    ts_cols = [c for c in df.columns if c.lower() in ("timestamp", "time", "date", "datetime")]
    price_cols = [c for c in df.columns if c.lower() in ("price", "last", "close", "value")]
    if not ts_cols or not price_cols:
        return None
    ts_col = ts_cols[0]
    price_col = price_cols[0]
    df = df.rename(columns={ts_col: "timestamp", price_col: "price"})
    if "market" not in df.columns:
        df["market"] = inferred_market or os.path.splitext(os.path.basename(inferred_market or "unknown"))[0]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["timestamp", "price"]).sort_values("timestamp").reset_index(drop=True)
    return df[["market", "timestamp", "price"]]


# -------------------------
# Rolling SD bands
# -------------------------
def rolling_sd_bands(df_market):
    prices = df_market["price"].values
    timestamps = df_market["timestamp"].values
    rows = []
    for i in range(len(prices)):
        hist = prices[: i + 1]
        m = hist.mean()
        s = hist.std(ddof=0)
        rows.append(
            {
                "timestamp": timestamps[i],
                "price": prices[i],
                "mean": m,
                "+1sd": m + s,
                "-1sd": m - s,
                "+2sd": m + 2 * s,
                "-2sd": m - 2 * s,
                "+3sd": m + 3 * s,
                "-3sd": m - 3 * s,
            }
        )
    return pd.DataFrame(rows)


# -------------------------
# Simple metrics & regime rules
# -------------------------
def analyze_band_position(bands_df):
    if bands_df.empty:
        return {"above_mean_pct": 0.0, "above_1sd_pct": 0.0, "below_minus_1sd_pct": 0.0, "within_1sd_pct": 0.0}
    p = np.asarray(bands_df["price"].values)
    m = np.asarray(bands_df["mean"].values)
    up1 = np.asarray(bands_df["+1sd"].values)
    down1 = np.asarray(bands_df["-1sd"].values)
    n = len(p)
    return {
        "above_mean_pct": float((p > m).sum() / n * 100.0),
        "above_1sd_pct": float((p > up1).sum() / n * 100.0),
        "below_minus_1sd_pct": float((p < down1).sum() / n * 100.0),
        "within_1sd_pct": float(((p >= down1) & (p <= up1)).sum() / n * 100.0),
    }


def analyze_band_position_against_reference(sub_df, bands_full):
    if sub_df.empty or bands_full.empty:
        return {"above_mean_pct": 0.0, "above_1sd_pct": 0.0, "below_minus_1sd_pct": 0.0, "within_1sd_pct": 0.0}

    # Align window timestamps to full-history bands (nearest previous)
    ref = bands_full[["timestamp", "mean", "+1sd", "-1sd"]].sort_values("timestamp")
    sub = sub_df[["timestamp", "price"]].sort_values("timestamp")
    aligned = pd.merge_asof(sub, ref, on="timestamp", direction="backward")

    aligned = aligned.dropna(subset=["price", "mean", "+1sd", "-1sd"])
    if aligned.empty:
        return {"above_mean_pct": 0.0, "above_1sd_pct": 0.0, "below_minus_1sd_pct": 0.0, "within_1sd_pct": 0.0}

    p = np.asarray(aligned["price"].values)
    m = np.asarray(aligned["mean"].values)
    up1 = np.asarray(aligned["+1sd"].values)
    down1 = np.asarray(aligned["-1sd"].values)
    n = len(p)

    return {
        "above_mean_pct": float((p > m).sum() / n * 100.0),
        "above_1sd_pct": float((p > up1).sum() / n * 100.0),
        "below_minus_1sd_pct": float((p < down1).sum() / n * 100.0),
        "within_1sd_pct": float(((p >= down1) & (p <= up1)).sum() / n * 100.0),
    }


def detect_mean_reversion(prices, window=5):
    if len(prices) < window:
        return 0.0
    roll = pd.Series(prices).rolling(window=window, min_periods=1).mean().values
    dev = prices - roll
    valid = dev[~np.isnan(dev)]
    if len(valid) <= 1:
        return 0.0
    changes = np.sum(np.diff(np.sign(valid)) != 0)
    return float(np.clip(changes / (len(valid) - 1), 0.0, 1.0))


def detect_mean_reversion_against_full(sub_df, bands_full, window=5):
    """
    Compute mean-reversion score for sub_df using deviations from the full-history mean.
    Returns 0-1 where higher = more oscillatory around the full mean.
    """
    if sub_df.empty or bands_full.empty:
        return 0.0

    # Align sub timestamps to full-history bands (nearest previous)
    ref = bands_full[["timestamp", "mean"]].sort_values("timestamp")
    sub = sub_df[["timestamp", "price"]].sort_values("timestamp")
    aligned = pd.merge_asof(sub, ref, on="timestamp", direction="backward").dropna(subset=["price", "mean"])
    if len(aligned) < 2:
        return 0.0

    dev = np.asarray(aligned["price"].values) - np.asarray(aligned["mean"].values)
    valid = dev[~np.isnan(dev)]
    if len(valid) <= 1:
        return 0.0
    changes = np.sum(np.diff(np.sign(valid)) != 0)
    return float(np.clip(changes / (len(valid) - 1), 0.0, 1.0))


def classify_regime(bands_df):
    pos = analyze_band_position(bands_df)
    mean_rev = detect_mean_reversion(bands_df["price"].values)
    above_mean = pos["above_mean_pct"]
    above_1sd = pos["above_1sd_pct"]
    below_1sd = pos["below_minus_1sd_pct"]
    within = pos["within_1sd_pct"]

    # Priority rules (simple and interpretable)
    if above_mean > 60 and above_1sd > 40:
        return "Imb. Up", min(1.0, (above_mean - 50) / 50 + above_1sd / 100.0)
    if below_1sd > 40 and above_mean < 40:
        return "Imb. Down", min(1.0, (40 - above_mean) / 50 + below_1sd / 100.0)
    if mean_rev >= 0.5:
        return "Mean-Reverting", mean_rev
    if within >= 70:
        return "Balanced", within / 100.0
    return "Rotational", 0.5


def classify_regime_against_full(sub_df, bands_full):
    """
    Classify sub_df regime but measure position vs full-history bands and
    mean-reversion vs the full-history mean (aligned).
    """
    pos = analyze_band_position_against_reference(sub_df, bands_full)
    mean_rev = detect_mean_reversion_against_full(sub_df, bands_full)
    above_mean = pos["above_mean_pct"]
    above_1sd = pos["above_1sd_pct"]
    below_1sd = pos["below_minus_1sd_pct"]
    within = pos["within_1sd_pct"]

    if above_mean > 60 and above_1sd > 40:
        return "Imb. Up", min(1.0, (above_mean - 50) / 50 + above_1sd / 100.0)
    if below_1sd > 40 and above_mean < 40:
        return "Imb. Down", min(1.0, (40 - above_mean) / 50 + below_1sd / 100.0)
    if mean_rev >= 0.5:
        return "Mean-Reverting", mean_rev
    if within >= 70:
        return "Balanced", within / 100.0
    return "Rotational", 0.5


# -------------------------
# IO: load CSVs and process per-file
# -------------------------
def find_csv_files(data_root: str, event: Optional[str] = None):
    root = Path(data_root)
    targets = []
    if event:
        p = root / event
        if p.exists():
            targets.append(p)
    else:
        targets = [p for p in root.iterdir() if p.is_dir()]
    records = []
    for sp in targets:
        if not sp.exists():
            continue
        for csv_path in sp.rglob("*.csv"):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                try:
                    df = pd.read_csv(csv_path, sep=";")
                except Exception:
                    continue
            std = _standardize_df(df, inferred_market=csv_path.stem)
            if std is not None and not std.empty:
                records.append((sp.name, csv_path, std))
    return records


# -------------------------
# Plotting with windowed regimes
# -------------------------
WINDOW_HOURS = [48, 36, 24, 12, 6]
WINDOW_COLORS = {48: "purple", 36: "red", 24: "blue", 12: "green", 6: "black"}


def plot_and_save(event_name, csv_path, df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markets = sorted(df["market"].unique())
    if not markets:
        return
    n = len(markets)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes] if hasattr(axes, "plot") else list(axes.flat)

    sorted_windows = sorted(WINDOW_HOURS)  # ascending: smallest → largest

    for idx, market in enumerate(markets):
        ax = axes[idx]
        mdf = df[df["market"] == market].sort_values("timestamp").reset_index(drop=True)
        if mdf.empty:
            continue

        # Full-history bands for background and full-history regime
        bands_full = rolling_sd_bands(mdf)
        t_full = bands_full["timestamp"].values
        ax.fill_between(t_full, bands_full["-3sd"], bands_full["+3sd"], alpha=0.06, color="#B6B6B6")
        ax.fill_between(t_full, bands_full["-2sd"], bands_full["+2sd"], alpha=0.08, color="#BABABA")
        ax.fill_between(t_full, bands_full["-1sd"], bands_full["+1sd"], alpha=0.12, color="#909090")
        ax.plot(t_full, bands_full["mean"], color="black", linestyle="--", linewidth=1.0, alpha=0.4)

        full_regime, full_conf = classify_regime(bands_full)

        # Plot entire price faintly as baseline
        ax.plot(mdf["timestamp"], mdf["price"], color="#A9C4FF", linewidth=1.0, alpha=0.6, zorder=3)

        final_ts = mdf["timestamp"].max()

        # Disjoint windows: iterate ascending, use previous window boundary as 'end'
        prev_hours = 0
        window_regimes = {}
        for wh in sorted_windows:
            start_time = final_ts - pd.Timedelta(hours=wh)
            end_time = final_ts - pd.Timedelta(hours=prev_hours)  # prev_hours==0 → end_time==final_ts
            # include start_time <= t < end_time (for final window prev_hours==0 include <= final_ts)
            if prev_hours == 0:
                sub = mdf[(mdf["timestamp"] >= start_time) & (mdf["timestamp"] <= end_time)]
            else:
                sub = mdf[(mdf["timestamp"] >= start_time) & (mdf["timestamp"] < end_time)]
            prev_hours = wh

            if len(sub) < 2:
                continue

            # classify the sub-window against full-history bands
            regime_w, conf_w = classify_regime_against_full(sub, bands_full)
            color = WINDOW_COLORS.get(wh, "gray")
            ax.plot(sub["timestamp"], sub["price"], color=color, linewidth=2.2, alpha=0.95, zorder=6)
            window_regimes[wh] = (regime_w, conf_w)

        # Build title: market + full-history regime + available window regimes
        window_parts = []
        for wh in sorted(window_regimes.keys(), reverse=True):
            reg, cf = window_regimes[wh]
            window_parts.append(f"T-{wh}h:{reg}")
        windows_str = " | ".join(window_parts) if window_parts else "no windows"
        ax.set_title(f"{market}\nFull: {full_regime} ({full_conf:.2f}) — {windows_str}", fontsize=8)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)

    # hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    # Global legend for windows
    handles = []
    for wh in WINDOW_HOURS:
        handles.append(Line2D([0], [0], color=WINDOW_COLORS.get(wh, "gray"), lw=3, label=f"T-{wh}h"))
    fig.legend(handles=handles, loc="upper center", ncol=len(WINDOW_HOURS), frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    event_dir = out_dir / event_name
    event_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{csv_path.stem}.png"
    out_path = event_dir / out_name
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="SD-bands regime plotting (per-CSV) with windowed regimes")
    parser.add_argument("--data-dir", default="polymarket_data")
    parser.add_argument("--event", default=None, help="optional event subfolder name")
    parser.add_argument("--output-dir", default="market-regime-analysis")
    args = parser.parse_args()

    records = find_csv_files(args.data_dir, event=args.event)
    if not records:
        print("No CSV files found.")
        return
    print(f"Found {len(records)} CSV files. Saving PNGs to {args.output_dir}")
    for event_name, csv_path, df in records:
        try:
            plot_and_save(event_name, csv_path, df, args.output_dir)
        except Exception as e:
            print(f"Failed {csv_path}: {e}")


if __name__ == "__main__":
    main()