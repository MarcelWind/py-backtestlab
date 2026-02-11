"""Full analysis pipeline for weather prediction market events.

Loads all events from data.zip, auto-detects settlement, trains per-city
leave-one-event-out models, evaluates, runs shuffle tests, and produces
forecasts for unsettled events.

Usage:
    python strategies/run_weather_pipeline.py [--zip data.zip] [--freq 1h]
        [--alpha 1.0] [--min-buckets 3] [--shuffle-trials 100]
        [--output results/weather/]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.fetch_data import load_zip
from strategies.weather_prediction import (
    FEATURE_COLS,
    AdjustedForecast,
    build_dataset,
    build_event_registry,
    evaluate,
    forecast,
    prepare_snapshots,
    shuffle_test,
)


# ---------------------------------------------------------------------------
# Snapshot preparation (all events)
# ---------------------------------------------------------------------------

def prepare_all_snapshots(
    data: pd.DataFrame,
    registry: pd.DataFrame,
    freq: str = "1h",
    min_buckets: int = 3,
) -> pd.DataFrame:
    """Prepare aligned snapshots for every qualifying event.

    Skips events with fewer than *min_buckets* buckets.
    """
    frames: list[pd.DataFrame] = []
    for _, row in registry.iterrows():
        if row["n_buckets"] < min_buckets:
            continue
        slug = row["event_slug"]
        ev = data[data["event_slug"] == slug]
        snaps = prepare_snapshots(
            ev,
            freq=freq,
            settlement_time=row["settlement_time"],
            realized=row["realized"] if row["is_settled"] else None,
        )
        frames.append(snaps)

    if not frames:
        raise RuntimeError("No qualifying events found")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Leave-one-event-out cross-validation
# ---------------------------------------------------------------------------

def loeo_cross_validate(
    snapshots: pd.DataFrame,
    slugs: list[str],
    alpha: float = 1.0,
) -> list[dict]:
    """LOEO-CV over *slugs*.  Returns one metrics dict per held-out event."""
    results: list[dict] = []
    for held_out in slugs:
        train_slugs = [s for s in slugs if s != held_out]
        train_snaps = snapshots[snapshots["event_slug"].isin(train_slugs)]
        test_snaps = snapshots[snapshots["event_slug"] == held_out]

        train_ds = build_dataset(train_snaps)
        test_ds = build_dataset(test_snaps)

        if train_ds.empty or test_ds.empty:
            continue

        model = AdjustedForecast(alpha=alpha)
        model.fit(train_ds[FEATURE_COLS].values, train_ds["target"].values)

        test_result = forecast(model, test_ds)
        metrics = evaluate(test_result)
        metrics["event_slug"] = held_out
        metrics["n_snapshots"] = len(test_ds)
        results.append(metrics)

    return results


# ---------------------------------------------------------------------------
# Forecast unsettled events
# ---------------------------------------------------------------------------

def forecast_unsettled(
    settled_snaps: pd.DataFrame,
    unsettled_snaps: pd.DataFrame,
    registry: pd.DataFrame,
    alpha: float = 1.0,
) -> list[dict]:
    """Train per-city models on settled data; forecast each unsettled event.

    Returns one dict per unsettled event with the latest-snapshot forecast.
    """
    forecasts: list[dict] = []

    for city, city_reg in registry[~registry["is_settled"]].groupby("city"):
        city_str = str(city)

        # Train on settled events for this city
        settled_city_slugs = registry[
            (registry["city"] == city_str) & registry["is_settled"]
        ]["event_slug"].tolist()

        train_snaps = settled_snaps[settled_snaps["event_slug"].isin(settled_city_slugs)]
        train_ds = build_dataset(train_snaps)
        if train_ds.empty:
            continue

        model = AdjustedForecast(alpha=alpha)
        model.fit(train_ds[FEATURE_COLS].values, train_ds["target"].values)

        # Forecast each unsettled event
        for _, row in city_reg.iterrows():
            slug = row["event_slug"]
            ev_snaps = unsettled_snaps[unsettled_snaps["event_slug"] == slug]
            if ev_snaps.empty:
                continue

            ev_ds = build_dataset(ev_snaps)
            if ev_ds.empty:
                continue

            ev_result = forecast(model, ev_ds)
            # Use the latest snapshot as the headline prediction
            latest = ev_result.iloc[-1]
            forecasts.append({
                "event_slug": slug,
                "city": city_str,
                "unit": row["unit"],
                "mu_m": float(latest["mu_m"]),
                "adjusted": float(latest["adjusted"]),
                "delta_hat": float(latest["delta_hat"]),
                "hours_to_settlement": float(latest["hours_to_settlement"]),
            })

    return forecasts


# ---------------------------------------------------------------------------
# Per-city shuffle test
# ---------------------------------------------------------------------------

def run_shuffle_tests(
    snapshots: pd.DataFrame,
    registry: pd.DataFrame,
    alpha: float = 1.0,
    n_trials: int = 100,
    seed: int = 42,
) -> list[dict]:
    """Run shuffle tests per city. Returns one dict per city."""
    results: list[dict] = []
    settled = registry[registry["is_settled"]]

    for city, city_reg in settled.groupby("city"):
        city_str = str(city)
        city_slugs = city_reg["event_slug"].tolist()
        city_snaps = snapshots[snapshots["event_slug"].isin(city_slugs)]

        if city_snaps.empty:
            continue

        # Real in-sample improvement
        ds = build_dataset(city_snaps)
        if len(ds) < 4:
            continue
        model = AdjustedForecast(alpha=alpha)
        split = len(ds) // 2
        model.fit(ds.iloc[:split][FEATURE_COLS].values, ds.iloc[:split]["target"].values)
        result = forecast(model, ds.iloc[split:])
        real_impr = evaluate(result)["improvement"]

        shuffled = shuffle_test(city_snaps, alpha=alpha, n_trials=n_trials, seed=seed)
        p_value = float((shuffled >= real_impr).mean())

        results.append({
            "city": city_str,
            "real_improvement": real_impr,
            "shuffled_mean": float(shuffled.mean()),
            "shuffled_std": float(shuffled.std()),
            "p_value": p_value,
            "n_events": len(city_slugs),
        })

    return results


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_metrics(per_event: pd.DataFrame) -> dict:
    """Aggregate per-event LOEO metrics into summary stats."""
    return {
        "n_events": len(per_event),
        "mean_baseline_mae": float(per_event["baseline_mae"].mean()),
        "mean_adjusted_mae": float(per_event["adjusted_mae"].mean()),
        "mean_improvement": float(per_event["improvement"].mean()),
        "median_improvement": float(per_event["improvement"].median()),
        "pct_improved": float((per_event["improvement"] > 0).mean() * 100),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(
    registry: pd.DataFrame,
    per_event: pd.DataFrame,
    summary: dict,
    shuffle_results: list[dict],
    unsettled_fc: pd.DataFrame,
    min_buckets: int,
) -> None:
    """Print a comprehensive summary to stdout."""
    sep = "=" * 70

    # -- Registry overview --
    settled = registry[registry["is_settled"]]
    unsettled = registry[~registry["is_settled"]]
    excluded = registry[registry["n_buckets"] < min_buckets]
    print(sep)
    print("EVENT REGISTRY")
    print(sep)
    print(f"  Total events:    {len(registry)}")
    print(f"  Settled:         {len(settled)}")
    print(f"  Unsettled:       {len(unsettled)}")
    print(f"  Excluded (<{min_buckets} buckets): {len(excluded)}")
    print(f"  Cities:          {sorted(registry['city'].unique())}")
    print()

    # -- Per-city LOEO table --
    print(sep)
    print("LEAVE-ONE-EVENT-OUT CROSS-VALIDATION (per city)")
    print(sep)
    city_groups = per_event.groupby("city")
    header = f"  {'City':<15s} {'Events':>6s} {'Base MAE':>9s} {'Adj MAE':>9s} {'Improv':>9s} {'% Better':>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for city, grp in sorted(city_groups):
        n = len(grp)
        base = grp["baseline_mae"].mean()
        adj = grp["adjusted_mae"].mean()
        imp = grp["improvement"].mean()
        pct = (grp["improvement"] > 0).mean() * 100
        print(f"  {str(city):<15s} {n:>6d} {base:>9.3f} {adj:>9.3f} {imp:>+9.3f} {pct:>8.0f}%")
    print()

    # -- Global summary --
    print(sep)
    print("GLOBAL SUMMARY")
    print(sep)
    print(f"  Events evaluated:      {summary['n_events']}")
    print(f"  Mean baseline MAE:     {summary['mean_baseline_mae']:.3f}")
    print(f"  Mean adjusted MAE:     {summary['mean_adjusted_mae']:.3f}")
    print(f"  Mean improvement:      {summary['mean_improvement']:+.3f}")
    print(f"  Median improvement:    {summary['median_improvement']:+.3f}")
    print(f"  % events improved:     {summary['pct_improved']:.0f}%")
    print()

    # -- Shuffle tests --
    if shuffle_results:
        print(sep)
        print("SHUFFLE TESTS (per city)")
        print(sep)
        header = f"  {'City':<15s} {'Real Improv':>12s} {'Shuf Mean':>10s} {'Shuf Std':>9s} {'p-value':>8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in sorted(shuffle_results, key=lambda x: x["city"]):
            sig = " *" if r["p_value"] < 0.05 else ""
            print(f"  {r['city']:<15s} {r['real_improvement']:>+12.3f} "
                  f"{r['shuffled_mean']:>+10.3f} {r['shuffled_std']:>9.3f} "
                  f"{r['p_value']:>8.3f}{sig}")
        print("  (* p < 0.05)")
        print()

    # -- Unsettled forecasts --
    if not unsettled_fc.empty:
        print(sep)
        print("UNSETTLED EVENT FORECASTS (latest snapshot)")
        print(sep)
        header = f"  {'Event':<55s} {'Market':>7s} {'Adjusted':>9s} {'Delta':>7s} {'Hrs Left':>9s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, row in unsettled_fc.sort_values("event_slug").iterrows():
            unit_label = "F" if row["unit"] == "f" else "C"
            print(f"  {row['event_slug']:<55s} "
                  f"{row['mu_m']:>6.1f}{unit_label} "
                  f"{row['adjusted']:>8.1f}{unit_label} "
                  f"{row['delta_hat']:>+7.2f} "
                  f"{row['hours_to_settlement']:>9.1f}")
        print()


def save_csvs(
    registry: pd.DataFrame,
    per_event: pd.DataFrame,
    unsettled_fc: pd.DataFrame,
    summary: dict,
    shuffle_results: list[dict],
    output_dir: Path,
) -> None:
    """Write result CSVs and a summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    registry.to_csv(output_dir / "event_registry.csv", index=False)
    per_event.to_csv(output_dir / "per_event_metrics.csv", index=False)

    if not unsettled_fc.empty:
        unsettled_fc.to_csv(output_dir / "unsettled_forecasts.csv", index=False)

    summary_out = {**summary, "shuffle_tests": shuffle_results}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_out, f, indent=2, default=str)

    print(f"Results saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", default="data.zip", help="Path to data.zip")
    p.add_argument("--freq", default="1h", help="Resample frequency (default: 1h)")
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha")
    p.add_argument("--min-buckets", type=int, default=3, help="Minimum buckets per event")
    p.add_argument("--shuffle-trials", type=int, default=100, help="Shuffle test trials per city")
    p.add_argument("--output", default="results/weather", help="Output directory for CSVs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load data
    print("Loading data...")
    data = load_zip(args.zip)
    print(f"  {len(data)} candles, {data['event_slug'].nunique()} events")

    # 2. Build event registry
    print("Building event registry...")
    registry = build_event_registry(data)
    settled_reg = registry[registry["is_settled"] & (registry["n_buckets"] >= args.min_buckets)]
    unsettled_reg = registry[~registry["is_settled"] & (registry["n_buckets"] >= args.min_buckets)]
    n_excluded = len(registry) - len(settled_reg) - len(unsettled_reg)
    print(f"  {len(settled_reg)} settled, {len(unsettled_reg)} unsettled, {n_excluded} excluded")

    # 3. Prepare snapshots
    print("Preparing snapshots (settled)...")
    settled_snaps = prepare_all_snapshots(data, settled_reg, freq=args.freq, min_buckets=args.min_buckets)
    n_settled_ts = settled_snaps.groupby("event_slug")["timestamp"].nunique().sum()
    print(f"  {n_settled_ts} total hourly snapshots across {settled_reg['event_slug'].nunique()} events")

    print("Preparing snapshots (unsettled)...")
    unsettled_snaps = prepare_all_snapshots(data, unsettled_reg, freq=args.freq, min_buckets=args.min_buckets)
    n_unsettled_ts = unsettled_snaps.groupby("event_slug")["timestamp"].nunique().sum()
    print(f"  {n_unsettled_ts} total hourly snapshots across {unsettled_reg['event_slug'].nunique()} events")

    # 4. Per-city LOEO cross-validation
    print("Running per-city leave-one-event-out CV...")
    all_loeo: list[dict] = []
    for city, city_reg in settled_reg.groupby("city"):
        city_str = str(city)
        city_slugs = city_reg["event_slug"].tolist()
        city_snaps = settled_snaps[settled_snaps["event_slug"].isin(city_slugs)]

        city_results = loeo_cross_validate(city_snaps, city_slugs, alpha=args.alpha)
        for r in city_results:
            r["city"] = city_str
            r["unit"] = city_reg["unit"].iloc[0]
        all_loeo.extend(city_results)
        print(f"  {city_str}: {len(city_results)} events evaluated")

    per_event = pd.DataFrame(all_loeo)
    summary = aggregate_metrics(per_event)

    # 5. Shuffle tests
    print(f"Running shuffle tests ({args.shuffle_trials} trials per city)...")
    shuffle_results = run_shuffle_tests(
        settled_snaps, settled_reg,
        alpha=args.alpha, n_trials=args.shuffle_trials, seed=42,
    )
    for r in shuffle_results:
        print(f"  {r['city']}: p={r['p_value']:.3f}")

    # 6. Forecast unsettled events
    print("Forecasting unsettled events...")
    unsettled_forecasts = forecast_unsettled(
        settled_snaps, unsettled_snaps, registry, alpha=args.alpha,
    )
    unsettled_fc = pd.DataFrame(unsettled_forecasts)
    print(f"  {len(unsettled_fc)} forecasts produced")

    # 7. Output
    print()
    print_results(registry, per_event, summary, shuffle_results, unsettled_fc, args.min_buckets)
    save_csvs(registry, per_event, unsettled_fc, summary, shuffle_results, Path(args.output))


if __name__ == "__main__":
    main()
