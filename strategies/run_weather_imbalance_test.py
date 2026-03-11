"""Smoke test runner for WeatherMarketImbalanceStrategy.

Runs a single event with default strategy parameters to quickly validate
entry/exit behavior before running wider parameter sweeps.
"""

from pathlib import Path
import sys
import argparse
import json
import time
import pickle
import pandas as pd
# import logging

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratlab.backtest.backtester import Backtester
from stratlab.config import RESULTS_DIR
from strategies.weather_backtest import load_event_ohlcv_resampled, plot_entries_exits
from strategies.weather_market_imbalance import WeatherMarketImbalanceStrategy


def find_latest_backtest_dir(event_slug: str) -> Path | None:
    """Find the latest backtest results directory for an event.
    
    Returns the most recent timestamp directory, or None if none found.
    """
    base_dir = RESULTS_DIR / "weather_imbalance_test" / event_slug
    if not base_dir.exists():
        return None
    
    dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    return dirs[-1] if dirs else None


def replot_backtest(
    event_slug: str,
    timestamp: str | None = None,
    resample_minutes: int = 5,
    prefer_outcome: str = "yes",
    signal_magnitude_threshold_imbalance: float | None = None,
    signal_magnitude_threshold_slope: float | None = None,
    signal_magnitude_threshold_meanrev: float | None = None,
    verbose: bool = False,
) -> None:
    """Replot a previous backtest without re-running it.
    
    Parameters
    ----------
    event_slug:
        Event slug (e.g. "highest-temperature-in-atlanta-on-february-22-2026")
    timestamp:
        Result timestamp (e.g. "20260223_155833"). If None, uses latest.
    resample_minutes:
        Original resample rate (must match how backtest was run).
    prefer_outcome:
        "yes" or "no" for outcome preference.
    signal_magnitude_threshold_*:
        Optional signal visualization thresholds.
    verbose:
        Print signal statistics and timing info.
    """
    t0 = time.perf_counter()
    
    # Find the backtest directory
    if timestamp is None:
        result_dir = find_latest_backtest_dir(event_slug)
        if result_dir is None:
            print(f"No backtest results found for event: {event_slug}")
            return
        timestamp = result_dir.name
    else:
        result_dir = RESULTS_DIR / "weather_imbalance_test" / event_slug / timestamp
        if not result_dir.exists():
            print(f"Backtest directory not found: {result_dir}")
            return

    # Load trades from saved CSV
    trades_path = result_dir / "trades.csv"
    if not trades_path.exists():
        print(f"No trades file found: {trades_path}")
        return
    
    trades_df = pd.read_csv(trades_path)
    print(f"Loaded {len(trades_df)} trades from {trades_path}")

    # Load indicator signals from saved pickle
    indicator_signals_path = result_dir / "indicator_signals.pkl"
    indicator_signals = None
    if indicator_signals_path.exists():
        with open(indicator_signals_path, "rb") as f:
            indicator_signals = pickle.load(f)
        print(f"Loaded indicator signals from {indicator_signals_path}")
        
        # Print signal statistics if verbose
        if verbose and indicator_signals:
            print("\n--- Signal Statistics ---")
            for sig_name, sig_df in indicator_signals.items():
                if isinstance(sig_df, pd.DataFrame):
                    for col in sig_df.columns:
                        vals = sig_df[col].dropna()
                        if len(vals) > 0:
                            nonzero = (vals.abs() > 1e-10).sum()
                            print(f"  {sig_name}[{col}]: {len(vals)} values, "
                                  f"{nonzero} non-zero, min={vals.min():.6f}, "
                                  f"mean={vals.mean():.6f}, max={vals.max():.6f}")
    else:
        print(f"Warning: No indicator signals file found at {indicator_signals_path}")
        indicator_signals = None

    # Load event data (same as backtest)
    resample_rule = f"{resample_minutes}min" if resample_minutes and resample_minutes > 0 else None
    
    t_load_start = time.perf_counter()

    # For strategy, single preferred outcome to operate on
    prices, vwap, volume, buy_volume, sell_volume, high, low, open_ = load_event_ohlcv_resampled(
        event_slug,
        resample_rule=resample_rule,
        prefer_outcome=prefer_outcome,
    )

    # For plotting, load unfiltered matrices 
    try:
        _p, _v, _vol, buy_volume_full, sell_volume_full, _, _, _ = load_event_ohlcv_resampled(
            event_slug,
            resample_rule=resample_rule,
            prefer_outcome=None,
        )
    except Exception:
        buy_volume_full = buy_volume
        sell_volume_full = sell_volume
    
    t_load_end = time.perf_counter()
    if verbose:
        print(f"Data loading: {t_load_end - t_load_start:.2f}s")

    # Create strategy instance to get indicator defs
    strategy = WeatherMarketImbalanceStrategy.from_profile(
        "balanced",
        vwap=vwap,
        volume=volume,
        high=high,
        low=low,
        open_=open_,
        buy_volume=buy_volume_full,
        sell_volume=sell_volume_full,
    )
    
    # Generate new plot with same data but possibly different visualization params
    plot_path = result_dir / f"entries_exits_replot.png"
    
    t_plot_start = time.perf_counter()
    plot_entries_exits(
        prices=prices,
        trades=trades_df,
        strategy_name="weather_imbalance_replot",
        event_slug=event_slug,
        out_path=plot_path,
        vwap=vwap,
        volume=volume,
        high=high,
        low=low,
        opens=open_,
        buy_volume=buy_volume_full,
        sell_volume=sell_volume_full,
        indicator_series=indicator_signals,
        indicator_defs=strategy.indicator_defs,
        max_vwap_slope=strategy.max_vwap_slope,
        mean_reversion_threshold=strategy.mean_reversion_threshold,
        signal_magnitude_threshold_imbalance=signal_magnitude_threshold_imbalance,
        signal_magnitude_threshold_slope=signal_magnitude_threshold_slope,
        signal_magnitude_threshold_meanrev=signal_magnitude_threshold_meanrev,
    )
    t_plot_end = time.perf_counter()
    if verbose:
        print(f"Plotting: {t_plot_end - t_plot_start:.2f}s")
    
    print(f"Saved replotted figure: {plot_path}")
    
    t_total = time.perf_counter() - t0
    if verbose:
        print(f"Total replot time: {t_total:.2f}s")


def main() -> None:
    # configure basic logging.  INFO level is usually sufficient; set
    # the strategy logger itself to DEBUG if deeper inspection is required.
    # logging.basicConfig(level=logging.INFO)
    # logging.getLogger("strategies.weather_market_imbalance").setLevel(logging.DEBUG)
    # suppress verbose debug from third-party libraries (matplotlib fonts, etc.)
    # logging.getLogger("matplotlib").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Run single-event weather imbalance smoke test")
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=WeatherMarketImbalanceStrategy.available_profiles(),
        help="Preset profile controlling imbalance thresholds and filters",
    )
    parser.add_argument(
        "--event-slug",
        default="highest-temperature-in-atlanta-on-february-22-2026",
        help="Event slug to test",
    )
    parser.add_argument(
        "--resample-minutes",
        type=int,
        default=5,
        help="Optional bar downsampling for speed (e.g. 5, 10). 0 keeps native bars.",
    )
    parser.add_argument(
        "--prefer-outcome",
        choices=["yes", "no"],
        default="yes",
        help="Prefer this outcome token when markets are suffixed with __yes/__no",
    )
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Skip backtest and replot a previous result (uses latest if --timestamp not specified)",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Specific backtest timestamp to replot (e.g. 20260223_155833). Only used with --replot-only.",
    )
    parser.add_argument(
        "--signal-magnitude-imbalance",
        type=float,
        default=None,
        help="Magnitude threshold for volume imbalance signal markers",
    )
    parser.add_argument(
        "--signal-magnitude-slope",
        type=float,
        default=None,
        help="Magnitude threshold for VWAP slope signal markers",
    )
    parser.add_argument(
        "--signal-magnitude-meanrev",
        type=float,
        default=None,
        help="Magnitude threshold for mean-reversion signal markers",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print signal statistics and timing breakdown",
    )
    args = parser.parse_args()

    event_slug = args.event_slug
    resample_rule = f"{args.resample_minutes}min" if args.resample_minutes and args.resample_minutes > 0 else None

    # Handle replot-only mode
    if args.replot_only:
        replot_backtest(
            event_slug=event_slug,
            timestamp=args.timestamp,
            resample_minutes=args.resample_minutes,
            prefer_outcome=args.prefer_outcome,
            signal_magnitude_threshold_imbalance=args.signal_magnitude_imbalance,
            signal_magnitude_threshold_slope=args.signal_magnitude_slope,
            signal_magnitude_threshold_meanrev=args.signal_magnitude_meanrev,
            verbose=args.verbose,
        )
        return

    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / "weather_imbalance_test" / event_slug / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    active_profile_params = WeatherMarketImbalanceStrategy.profile_params(args.profile)
    all_summary_rows: list[dict[str, object]] = []

    print(f"Loaded event target: {event_slug}")
    if resample_rule:
        print(f"Resample rule: {resample_rule}")
    print("VWAP source: data.zip vwap column")

    prices, vwap, volume, buy_volume, sell_volume, high, low, open_ = load_event_ohlcv_resampled(
        event_slug,
        resample_rule=resample_rule,
        prefer_outcome=args.prefer_outcome,
    )

    # Also load unfiltered matrices (no prefer_outcome) so plotting can
    # access both `__yes` and `__no` outcome columns when available.
    try:
        _p, _v, _vol, buy_volume_full, sell_volume_full, _, _, _ = load_event_ohlcv_resampled(
            event_slug,
            resample_rule=resample_rule,
            prefer_outcome=None,  # this may filter out some columns, but try to respect the user's preference if possible
        )
    except Exception:
        buy_volume_full = buy_volume
        sell_volume_full = sell_volume

    strategy = WeatherMarketImbalanceStrategy.from_profile(
        args.profile,
        vwap=vwap,
        volume=volume,
        high=high,
        low=low,
        open_=open_,
        buy_volume=buy_volume_full,
        sell_volume=sell_volume_full,
    )
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
    indicator_signals_path = out_dir / "indicator_signals.pkl"

    # Save indicator signals for later replots
    indicator_signals = result.get("indicator_signals", {})
    with open(indicator_signals_path, "wb") as f:
        pickle.dump(indicator_signals, f)

    # If intrabar high/low series are available, prefer those as the recorded
    # exit fill price for stop/take exits so recorded `exit_price` matches
    # the plotted intrabar marker.
    def _apply_intrabar_fills(df, high_df, low_df):
        if df is None or df.empty:
            return df
        if high_df is None and low_df is None:
            return df
        for idx, row in df.iterrows():
            exit_time_raw = row.get("exit_time", None)
            if pd.isna(exit_time_raw) or exit_time_raw is None:
                continue
            try:
                et = pd.to_datetime(exit_time_raw)
            except Exception:
                continue
            asset = row.get("asset")
            reason = str(row.get("exit_reason", ""))
            chosen = None
            if reason == "stop_loss" and high_df is not None and asset in high_df.columns:
                try:
                    val = high_df[asset].reindex([et]).iloc[0]
                    if pd.notna(val):
                        chosen = float(val)
                except Exception:
                    pass
            if reason == "take_profit" and low_df is not None and asset in low_df.columns:
                try:
                    val = low_df[asset].reindex([et]).iloc[0]
                    if pd.notna(val):
                        chosen = float(val)
                except Exception:
                    pass
            if chosen is not None:
                df.at[idx, "exit_price"] = chosen
        return df

    trades_df = _apply_intrabar_fills(trades_df, high, low)
    trades_df.to_csv(trades_path, index=False)

    plot_entries_exits(
        prices=prices,
        trades=trades_df,
        strategy_name="weather_imbalance_default",
        event_slug=event_slug,
        out_path=plot_path,
        vwap=vwap,
        volume=volume,
        high=high,
        low=low,
        # prefer passing the unfiltered buy/sell matrices so both
        # `__yes` and `__no` outcome columns (if present) can be plotted
        # regardless of the strategy's `prefer_outcome` used for backtesting.
        buy_volume=buy_volume_full,
        sell_volume=sell_volume_full,
        indicator_series=strategy.indicator_series,
        indicator_defs=strategy.indicator_defs,
        max_vwap_slope=strategy.max_vwap_slope,
        mean_reversion_threshold=strategy.mean_reversion_threshold,
        signal_magnitude_threshold_imbalance=args.signal_magnitude_imbalance,
        signal_magnitude_threshold_slope=args.signal_magnitude_slope,
        signal_magnitude_threshold_meanrev=args.signal_magnitude_meanrev,
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
