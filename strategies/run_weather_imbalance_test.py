"""Smoke test runner for WeatherMarketImbalanceStrategy.

Runs a single event with default strategy parameters to quickly validate
entry/exit behavior before running wider parameter sweeps.
"""

from pathlib import Path
import sys
import argparse
import json
import time
import pandas as pd

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratlab.backtest.backtester import Backtester
from stratlab.config import RESULTS_DIR
from strategies.weather_backtest import load_event_ohlcv_resampled, plot_entries_exits
from strategies.weather_market_imbalance import WeatherMarketImbalanceStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-event weather imbalance smoke test")
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=WeatherMarketImbalanceStrategy.available_profiles(),
        help="Preset profile controlling imbalance thresholds and filters",
    )
    parser.add_argument(
        "--event-slug",
        default="highest-temperature-in-nyc-on-february-16-2026",
        help="Event slug to test",
    )
    parser.add_argument(
        "--resample-minutes",
        type=int,
        default=5,
        help="Optional bar downsampling for speed (e.g. 5, 10). 0 keeps native bars.",
    )
    args = parser.parse_args()

    event_slug = args.event_slug
    resample_rule = f"{args.resample_minutes}min" if args.resample_minutes and args.resample_minutes > 0 else None

    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / "weather_imbalance_test" / timestamp / event_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    active_profile_params = WeatherMarketImbalanceStrategy.profile_params(args.profile)
    all_summary_rows: list[dict[str, object]] = []

    print(f"Loaded event target: {event_slug}")
    if resample_rule:
        print(f"Resample rule: {resample_rule}")
    print("VWAP source: data.zip vwap column")

    prices, vwap, volume = load_event_ohlcv_resampled(
        event_slug,
        resample_rule=resample_rule,
    )

    strategy = WeatherMarketImbalanceStrategy.from_profile(args.profile, vwap=vwap, volume=volume)
    backtester = Backtester(strategy=strategy, rebalance_freq=1)
    started = time.perf_counter()
    result = backtester.run(prices)
    runtime_sec = time.perf_counter() - started
    strategy.finalize(prices)

    returns = result["returns"]
    metrics = result["metrics"]
    trades_df = pd.DataFrame(strategy.trade_log)

    trades_path = out_dir / "trades.csv"
    summary_path = out_dir / "summary.csv"
    plot_path = out_dir / "entries_exits.png"

    if trades_df.empty:
        pd.DataFrame(
            columns=[
                "asset",
                "entry_index",
                "entry_time",
                "entry_price",
                "exit_index",
                "exit_time",
                "exit_price",
                "exit_reason",
                "pnl",
                "regime",
                "confidence",
                "vwap_slope",
                "vwap_slope_raw",
                "vwap_volume_imbalance_pct",
            ]
        ).to_csv(trades_path, index=False)
    else:
        trades_df.to_csv(trades_path, index=False)

    plot_entries_exits(
        prices=prices,
        trades=trades_df,
        strategy_name="weather_imbalance_default",
        event_slug=event_slug,
        out_path=plot_path,
        vwap=vwap,
        volume=volume,
        vwap_slope_mode=strategy.vwap_slope_mode,
        vwap_slope_value_per_point=strategy.vwap_slope_value_per_point,
        vwap_slope_scale=strategy.vwap_slope_scale,
        vwap_slope_lookback=strategy.vwap_slope_lookback,
        max_vwap_slope=strategy.max_vwap_slope,
        mean_reversion_window=strategy.mean_reversion_window,
        mean_reversion_threshold=strategy.mean_reversion_threshold,
    )

    summary_row = {
        "profile": args.profile,
        "profile_params_json": json.dumps(active_profile_params, sort_keys=True),
        "event_slug": event_slug,
        "resample_rule": resample_rule or "native",
        "n_price_rows": int(len(prices)),
        "runtime_seconds": float(runtime_sec),
        "lookback_hours": strategy.lookback_hours,
        "trade_count": int(len(trades_df)),
        "win_rate": float((trades_df["pnl"] > 0).mean()) if len(trades_df) else 0.0,
        "avg_trade_pnl": float(trades_df["pnl"].mean()) if len(trades_df) else 0.0,
        "total_return": metrics.get("total_return", 0.0),
        "sharpe": metrics.get("sharpe", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "volatility": metrics.get("volatility", 0.0),
        "mean_bar_return": float(returns.mean()) if len(returns) else 0.0,
        "trades_file": "trades.csv",
        "plot_file": "entries_exits.png",
    }
    all_summary_rows.append(summary_row)
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)

    print(
        f"markets={len(prices.columns)} | rows={len(prices)} | "
        f"runtime={runtime_sec:.2f}s | trades={len(trades_df)}"
    )
    print(f"Saved summary: {summary_path}")
    print(f"Saved trades: {trades_path}")
    print(f"Saved plot: {plot_path}")

    combined_summary_path = out_dir / "summary_all_modes.csv"
    combined_summary = pd.DataFrame(all_summary_rows)
    combined_summary.to_csv(combined_summary_path, index=False)

    print("\nSmoke test complete")
    print(f"Active profile: {args.profile}")
    print(f"Profile params: {active_profile_params}")
    print(combined_summary.to_string(index=False))
    print(f"Saved combined summary: {combined_summary_path}")


if __name__ == "__main__":
    main()
