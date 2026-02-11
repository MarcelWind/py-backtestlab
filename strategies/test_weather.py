"""End-to-end test of the VWAP-adjusted forecast pipeline on Ankara weather markets.

Uses four settled Ankara events (Feb 7-10, 2026) — each with 7 temperature
buckets — to train and evaluate the AdjustedForecast model.

Train on Feb 7, 8, 9; evaluate on Feb 10.
"""

import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is on sys.path so stratlab is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.fetch_data import load_zip
from strategies.weather_prediction import (
    FEATURE_COLS,
    AdjustedForecast,
    build_dataset,
    compute_bucket_bounds,
    evaluate,
    forecast,
    parse_bucket_name,
    prepare_snapshots,
    shuffle_test,
)

# ── Settlement metadata for each Ankara event ────────────────────────────
# realized = midpoint of the winning bucket (inferred from close ≈ 1.0)
ANKARA_EVENTS: dict[str, dict] = {
    "highest-temperature-in-ankara-on-february-7-2026": {
        "settlement": pd.Timestamp("2026-02-08T00:00:00", tz="UTC"),
        "realized": 10.0,   # 10-c won
    },
    "highest-temperature-in-ankara-on-february-8-2026": {
        "settlement": pd.Timestamp("2026-02-09T00:00:00", tz="UTC"),
        "realized": 10.5,   # 10-c-or-higher won → midpoint of (10, 11)
    },
    "highest-temperature-in-ankara-on-february-9-2026": {
        "settlement": pd.Timestamp("2026-02-10T00:00:00", tz="UTC"),
        "realized": 9.0,    # 9-c won
    },
    "highest-temperature-in-ankara-on-february-10-2026": {
        "settlement": pd.Timestamp("2026-02-11T00:00:00", tz="UTC"),
        "realized": 6.0,    # 6-c won
    },
}

TRAIN_EVENTS = [
    "highest-temperature-in-ankara-on-february-7-2026",
    "highest-temperature-in-ankara-on-february-8-2026",
    "highest-temperature-in-ankara-on-february-9-2026",
]
TEST_EVENT = "highest-temperature-in-ankara-on-february-10-2026"


def main() -> None:
    # ── 1. Load raw OHLCV data ────────────────────────────────────────────
    print("Loading data from zip...")
    all_data = load_zip()
    ankara = all_data[all_data["event_slug"].isin(ANKARA_EVENTS)]
    print(f"  {len(ankara)} candles across {ankara['event_slug'].nunique()} Ankara events")

    # ── 2. Verify bucket parsing on one event ─────────────────────────────
    sample_event = ankara[ankara["event_slug"] == TEST_EVENT]
    markets = sample_event["market"].unique().tolist()
    print(f"\nBucket parsing for {TEST_EVENT}:")
    bounds = compute_bucket_bounds(markets)
    for name, (lo, hi) in sorted(bounds.items(), key=lambda x: x[1][0]):
        parsed = parse_bucket_name(name)
        print(f"  {name:25s}  ->  low={lo:5.1f}  high={hi:5.1f}  mid={(lo+hi)/2:5.1f}  ({parsed['unit'].upper()})")

    # ── 3. Prepare aligned snapshots for every event ──────────────────────
    print("\nPreparing hourly snapshots...")
    snapshot_frames: list[pd.DataFrame] = []
    for slug, meta in ANKARA_EVENTS.items():
        ev = ankara[ankara["event_slug"] == slug]
        snaps = prepare_snapshots(
            ev,
            freq="1h",
            settlement_time=meta["settlement"],
            realized=meta["realized"],
        )
        snapshot_frames.append(snaps)
        n_ts = snaps["timestamp"].nunique()
        print(f"  {slug}: {n_ts} snapshots × {snaps['market'].nunique()} buckets")

    all_snapshots = pd.concat(snapshot_frames, ignore_index=True)

    # ── 4. Build feature dataset ──────────────────────────────────────────
    print("\nBuilding feature dataset...")
    dataset = build_dataset(all_snapshots)
    print(f"  {len(dataset)} snapshot-level rows")
    print(f"  Columns: {list(dataset.columns)}")
    print(f"\n  Feature summary:\n{dataset[FEATURE_COLS].describe().to_string()}")

    # ── 5. Train / test split by event ────────────────────────────────────
    train_snaps = all_snapshots[all_snapshots["event_slug"].isin(TRAIN_EVENTS)]
    test_snaps = all_snapshots[all_snapshots["event_slug"] == TEST_EVENT]

    train_ds = build_dataset(train_snaps)
    test_ds = build_dataset(test_snaps)
    print(f"\nTrain: {len(train_ds)} rows from {train_ds['realized'].nunique()} events")
    print(f"Test:  {len(test_ds)} rows from {TEST_EVENT}")

    # ── 6. Fit model ──────────────────────────────────────────────────────
    model = AdjustedForecast(alpha=1.0)
    model.fit(train_ds[FEATURE_COLS].values, train_ds["target"].values)
    print(f"\nModel coefficients: {dict(zip(FEATURE_COLS, model.coef_))}")
    print(f"Intercept: {model.intercept_:.4f}")

    # ── 7. Forecast & evaluate on held-out event ──────────────────────────
    test_result = forecast(model, test_ds)
    metrics = evaluate(test_result)

    realized_val = ANKARA_EVENTS[TEST_EVENT]["realized"]
    print(f"\n{'='*60}")
    print(f"EVALUATION — {TEST_EVENT}")
    print(f"{'='*60}")
    print(f"  Realized temperature:  {realized_val}°C")
    print(f"  Baseline MAE (market): {metrics['baseline_mae']:.3f}°C")
    print(f"  Adjusted MAE (model):  {metrics['adjusted_mae']:.3f}°C")
    print(f"  Improvement:           {metrics['improvement']:+.3f}°C")

    # Show a few individual predictions
    print(f"\n  Sample predictions (last 5 snapshots):")
    tail = test_result.tail(5)
    for _, row in tail.iterrows():
        print(f"    mu_m={row['mu_m']:.2f}  adjusted={row['adjusted']:.2f}  "
              f"realized={row['realized']:.1f}  hours_left={row['hours_to_settlement']:.1f}")

    # ── 8. Quick sanity: full-dataset in-sample ───────────────────────────
    full_model = AdjustedForecast(alpha=1.0)
    full_model.fit(dataset[FEATURE_COLS].values, dataset["target"].values)
    full_result = forecast(full_model, dataset)
    full_metrics = evaluate(full_result)
    print(f"\nIn-sample (all 4 events):")
    print(f"  Baseline MAE: {full_metrics['baseline_mae']:.3f}°C")
    print(f"  Adjusted MAE: {full_metrics['adjusted_mae']:.3f}°C")
    print(f"  Improvement:  {full_metrics['improvement']:+.3f}°C")

    # ── 9. Shuffle test — is the signal real? ─────────────────────────────
    print(f"\nShuffle test (50 trials on all snapshots)...")
    shuffled_impr = shuffle_test(all_snapshots, alpha=1.0, n_trials=50, seed=42)
    real_impr = full_metrics["improvement"]
    pct_above = (shuffled_impr >= real_impr).mean() * 100
    print(f"  Real improvement:      {real_impr:+.3f}°C")
    print(f"  Shuffled mean:         {shuffled_impr.mean():+.3f}°C")
    print(f"  Shuffled std:          {shuffled_impr.std():.3f}°C")
    print(f"  Shuffled range:        [{shuffled_impr.min():+.3f}, {shuffled_impr.max():+.3f}]")
    print(f"  % shuffled >= real:    {pct_above:.1f}%")
    if pct_above < 5:
        print("  => Signal looks genuine (p < 0.05)")
    else:
        print("  => Signal may be spurious — shuffled data performs comparably")


if __name__ == "__main__":
    main()
