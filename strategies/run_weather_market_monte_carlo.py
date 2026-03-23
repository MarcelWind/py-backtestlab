"""Dedicated Monte Carlo runner for WeatherMarketImbalanceStrategy.

This script samples weather-strategy parameters around a chosen preset profile,
optimizes on backtest metrics, and saves full trial history.

Usage example:
    python strategies/run_weather_market_monte_carlo.py \
        --events-file strategies/weather_market_monte_carlo_events.json \
        --param-config strategies/weather_market_monte_carlo_params.json \
        --profile balanced \
        --n-trials 300 \
        --objective sharpe \
        --resample-minutes 1
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratlab.backtest.backtester import Backtester, compute_returns
from stratlab.config import RESULTS_DIR
from stratlab.data.dtype_utils import to_float32_frame
from stratlab.io.csv_writer import TrialCsvWriter
from stratlab.io.events import load_event_slugs_from_file
from stratlab.io.indicator_cache import (
    build_cache_key_material,
    clear_cache,
    compute_file_fingerprint,
    hash_payload,
    load_indicator_snapshots,
    save_indicator_snapshots,
)
from stratlab.optimize.rule_search import estimate_max_unique_trials, load_param_rules, params_key, sample_trial_params
from stratlab.validation.partition import EventPartition, split_events_in_sample_out_of_sample
from stratlab.validation.significance import empirical_one_tailed_pvalue, walk_forward_pvalue_from_event_order
from strategies.weather_backtest import load_event_ohlcv_resampled_with_unfiltered_cvd
from strategies.weather_market_strategy import WeatherMarketImbalanceStrategy

_PARAM_CONFIG_PATH = Path(__file__).with_name("weather_market_monte_carlo_params.json")
_EMPTY_TRADES_DF = pd.DataFrame()
_DEFAULT_PRECOMPUTE_CACHE_DIR = RESULTS_DIR / "indicator_cache" / "weather_imbalance_mc"


class EventBundle(TypedDict):
    prices: pd.DataFrame
    returns: pd.DataFrame
    vwap: pd.DataFrame
    volume: pd.DataFrame
    high: pd.DataFrame | None
    low: pd.DataFrame | None
    open_: pd.DataFrame | None
    buy_volume: pd.DataFrame | None
    sell_volume: pd.DataFrame | None
    # Pre-computed indicator state snapshots (name → snapshot dict).
    # Populated after load by _precompute_indicator_snapshots(); None until then.
    indicator_snapshots: "dict[str, dict] | None"


_MC_WORKER_EVENT_DATA: dict[str, EventBundle] | None = None
_MC_WORKER_BASE_PARAMS: dict[str, object] | None = None
_MC_WORKER_REBALANCE_FREQ: int = 1
_MC_WORKER_OBJECTIVE: str = "sharpe"


def _mc_worker_init(
    event_data: dict[str, EventBundle],
    base_params: dict[str, object],
    rebalance_freq: int,
    objective: str,
) -> None:
    """Initialize per-process globals for event-parallel trial evaluation."""
    global _MC_WORKER_EVENT_DATA, _MC_WORKER_BASE_PARAMS, _MC_WORKER_REBALANCE_FREQ, _MC_WORKER_OBJECTIVE
    _MC_WORKER_EVENT_DATA = event_data
    _MC_WORKER_BASE_PARAMS = base_params
    _MC_WORKER_REBALANCE_FREQ = int(rebalance_freq)
    _MC_WORKER_OBJECTIVE = objective


def _mc_evaluate_event(payload: tuple[str, dict[str, Any]]) -> tuple[str, float, float]:
    """Evaluate one event for one sampled parameter set inside a worker process."""
    slug, chosen_params = payload
    if _MC_WORKER_EVENT_DATA is None or _MC_WORKER_BASE_PARAMS is None:
        raise RuntimeError("Monte Carlo worker is not initialized")

    bundle = _MC_WORKER_EVENT_DATA[slug]
    strategy_kwargs = dict(_MC_WORKER_BASE_PARAMS)
    strategy_kwargs.update(chosen_params)
    strategy_kwargs["vwap"] = bundle["vwap"]
    strategy_kwargs["volume"] = bundle["volume"]
    strategy_kwargs["high"] = bundle["high"]
    strategy_kwargs["low"] = bundle["low"]
    strategy_kwargs["open_"] = bundle["open_"]
    strategy_kwargs["buy_volume"] = bundle["buy_volume"]
    strategy_kwargs["sell_volume"] = bundle["sell_volume"]

    metrics, _trades = _run_single_backtest(
        prices=bundle["prices"],
        returns=bundle["returns"],
        strategy_kwargs=strategy_kwargs,
        rebalance_freq=_MC_WORKER_REBALANCE_FREQ,
        collect_trades=False,
        indicator_snapshots=bundle.get("indicator_snapshots"),
    )
    raw_obj = float(metrics.get(_MC_WORKER_OBJECTIVE, 0.0))
    signed_obj = _metric_direction(_MC_WORKER_OBJECTIVE) * raw_obj
    return slug, raw_obj, signed_obj


def _vprint(verbose: bool, message: str) -> None:
    """Print progress messages only when verbose mode is enabled."""
    if verbose:
        print(message, flush=True)


def _to_float32_frame(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Downcast float64 columns to float32 to reduce resident memory."""
    if df is None or df.empty:
        return df
    float_cols = list(df.select_dtypes(include=["float64"]).columns)
    if not float_cols:
        return df
    out = df.copy()
    out[float_cols] = out[float_cols].astype(np.float32)
    return out


def _metric_direction(metric: str) -> float:
    """Return +1 when higher-is-better, -1 when lower-is-better."""
    lower_is_better = {"max_drawdown", "volatility"}
    return -1.0 if metric in lower_is_better else 1.0


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS."""
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _resolve_indicator_names_for_bundle(
    bundle: EventBundle,
    base_params: dict[str, object],
) -> list[str]:
    """Build strategy once to capture enabled indicator set for cache keying."""
    strategy_kwargs: dict[str, Any] = dict(base_params)
    strategy_kwargs["vwap"] = bundle["vwap"]
    strategy_kwargs["volume"] = bundle["volume"]
    strategy_kwargs["high"] = bundle["high"]
    strategy_kwargs["low"] = bundle["low"]
    strategy_kwargs["open_"] = bundle["open_"]
    strategy_kwargs["buy_volume"] = bundle["buy_volume"]
    strategy_kwargs["sell_volume"] = bundle["sell_volume"]
    strategy_kwargs.setdefault("track_delta_history", False)
    strategy = WeatherMarketImbalanceStrategy(**strategy_kwargs)
    return sorted(str(ind.name) for ind in strategy.indicator_defs)


def _event_source_fingerprint(bundle: EventBundle) -> str:
    """Create a lightweight deterministic fingerprint of loaded event matrices."""

    def _frame_sig(name: str, df: pd.DataFrame | None) -> dict[str, Any]:
        if df is None:
            return {"name": name, "missing": True}
        n_rows = int(len(df.index))
        n_cols = int(len(df.columns))
        first_ts = str(df.index[0]) if n_rows > 0 else ""
        last_ts = str(df.index[-1]) if n_rows > 0 else ""
        first_col = str(df.columns[0]) if n_cols > 0 else ""
        last_col = str(df.columns[-1]) if n_cols > 0 else ""
        return {
            "name": name,
            "rows": n_rows,
            "cols": n_cols,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "first_col": first_col,
            "last_col": last_col,
        }

    payload = [
        _frame_sig("prices", bundle.get("prices")),
        _frame_sig("returns", bundle.get("returns")),
        _frame_sig("vwap", bundle.get("vwap")),
        _frame_sig("volume", bundle.get("volume")),
        _frame_sig("high", bundle.get("high")),
        _frame_sig("low", bundle.get("low")),
        _frame_sig("open_", bundle.get("open_")),
        _frame_sig("buy_volume", bundle.get("buy_volume")),
        _frame_sig("sell_volume", bundle.get("sell_volume")),
    ]
    return hash_payload(payload)


def _run_trials_with_interrupt_handling(
    target_trials: int,
    rules: list[dict[str, Any]],
    base_params: dict[str, object],
    event_data: dict[str, EventBundle],
    in_sample_event_slugs: list[str],
    out_of_sample_event_slugs: list[str],
    rng: np.random.Generator,
    objective: str,
    rebalance_freq: int,
    workers: int,
    verbose: bool,
    trial_writer: "TrialCsvWriter",
    top_n: int,
    is_pvalue_threshold: float,
    oos_pvalue_threshold: float,
) -> tuple[int, float, dict[str, Any] | None, bool, list[dict[str, Any]], dict[str, int]]:
    """Run Monte Carlo trials and return partial results if interrupted."""
    seen: set[tuple[tuple[str, Any], ...]] = set()
    rows_written = 0
    best_score = float("-inf")
    best_params: dict[str, Any] | None = None
    interrupted = False
    top_rows: list[dict[str, Any]] = []

    max_retries_per_trial = 30
    objective_sign = _metric_direction(objective)
    worker_count = max(1, int(workers))
    use_process_pool = worker_count > 1 and len(event_data) > 1
    in_sample_event_set = set(in_sample_event_slugs)
    out_of_sample_event_set = set(out_of_sample_event_slugs)
    historical_in_sample_scores: list[float] = []
    historical_oos_prefix_scores: dict[int, list[float]] = {}
    accepted_trials = 0
    rejected_trials_is = 0
    rejected_trials_oos = 0

    # Constraints to avoid sampling settings that exceed available history.
    def _constraints(params: dict[str, Any], prices_df: pd.DataFrame) -> bool:
        bars = len(prices_df)
        if bars < 20:
            return False
        slope_lb = int(params.get("vwap_slope_lookback", base_params.get("vwap_slope_lookback", 15)))
        imb_lb = int(
            params.get(
                "vwap_volume_imbalance_lookback",
                base_params.get("vwap_volume_imbalance_lookback", slope_lb),
            )
        )
        mean_rev_window = int(params.get("mean_reversion_window", base_params.get("mean_reversion_window", 5)))
        return max(slope_lb, imb_lb, mean_rev_window) < bars - 2

    executor: concurrent.futures.ProcessPoolExecutor | None = None
    try:
        if use_process_pool:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=min(worker_count, len(event_data)),
                initializer=_mc_worker_init,
                initargs=(event_data, base_params, int(rebalance_freq), objective),
            )

        for trial_idx in range(target_trials):
            _vprint(verbose, f"[trial {trial_idx + 1}/{target_trials}] Sampling candidate parameters")
            chosen_params: dict[str, Any] | None = None
            for _ in range(max_retries_per_trial):
                sampled = sample_trial_params(rules, base_params, rng)
                pkey = params_key(sampled)
                if pkey in seen:
                    continue
                merged_candidate = dict(base_params)
                merged_candidate.update(sampled)
                if not all(
                    _constraints(merged_candidate, bundle["prices"])
                    for bundle in event_data.values()
                ):
                    continue
                seen.add(pkey)
                chosen_params = sampled
                break

            if chosen_params is None:
                if verbose:
                    print(f"Stopping early at trial {trial_idx}: no new valid parameter sets found.")
                break

            per_event_scores: dict[str, float] = {}
            per_event_scores_in_sample: dict[str, float] = {}
            per_event_scores_out_of_sample: dict[str, float] = {}
            row: dict[str, Any] = dict(chosen_params)

            if use_process_pool and executor is not None:
                futures = {
                    executor.submit(_mc_evaluate_event, (slug, chosen_params)): slug
                    for slug in event_data.keys()
                }
                for future in concurrent.futures.as_completed(futures):
                    slug, raw_obj, signed_obj = future.result()
                    per_event_scores[slug] = signed_obj
                    if slug in in_sample_event_set:
                        per_event_scores_in_sample[slug] = signed_obj
                    elif slug in out_of_sample_event_set:
                        per_event_scores_out_of_sample[slug] = signed_obj
                    row[f"m_{slug}_{objective}"] = raw_obj
                    _vprint(
                        verbose,
                        f"[trial {trial_idx + 1}/{target_trials}] Event '{slug}' {objective}={raw_obj:.6f}",
                    )
            else:
                for slug, bundle in event_data.items():
                    _vprint(
                        verbose,
                        f"[trial {trial_idx + 1}/{target_trials}] Running backtest for event '{slug}'",
                    )
                    strategy_kwargs = dict(base_params)
                    strategy_kwargs.update(chosen_params)
                    strategy_kwargs["vwap"] = bundle["vwap"]
                    strategy_kwargs["volume"] = bundle["volume"]
                    strategy_kwargs["high"] = bundle["high"]
                    strategy_kwargs["low"] = bundle["low"]
                    strategy_kwargs["open_"] = bundle["open_"]
                    strategy_kwargs["buy_volume"] = bundle["buy_volume"]
                    strategy_kwargs["sell_volume"] = bundle["sell_volume"]

                    metrics, _trades = _run_single_backtest(
                        prices=bundle["prices"],
                        returns=bundle["returns"],
                        strategy_kwargs=strategy_kwargs,
                        rebalance_freq=rebalance_freq,
                        collect_trades=False,
                        indicator_snapshots=bundle.get("indicator_snapshots"),
                    )
                    raw_obj = float(metrics.get(objective, 0.0))
                    signed_obj = objective_sign * raw_obj
                    per_event_scores[slug] = signed_obj
                    if slug in in_sample_event_set:
                        per_event_scores_in_sample[slug] = signed_obj
                    elif slug in out_of_sample_event_set:
                        per_event_scores_out_of_sample[slug] = signed_obj
                    row[f"m_{slug}_{objective}"] = raw_obj
                    _vprint(
                        verbose,
                        f"[trial {trial_idx + 1}/{target_trials}] Event '{slug}' {objective}={raw_obj:.6f}",
                    )

            agg_score = (
                float(np.mean(list(per_event_scores_in_sample.values())))
                if per_event_scores_in_sample
                else float("-inf")
            )
            oos_score = (
                float(np.mean(list(per_event_scores_out_of_sample.values())))
                if per_event_scores_out_of_sample
                else float("nan")
            )
            row["_objective"] = agg_score
            row["_objective_in_sample"] = agg_score
            row["_objective_out_of_sample"] = oos_score
            row["_event_count"] = len(per_event_scores)
            row["_event_count_in_sample"] = len(per_event_scores_in_sample)
            row["_event_count_out_of_sample"] = len(per_event_scores_out_of_sample)

            is_pvalue = empirical_one_tailed_pvalue(agg_score, historical_in_sample_scores)
            oos_pvalue = walk_forward_pvalue_from_event_order(
                event_scores=per_event_scores_out_of_sample,
                ordered_oos_slugs=out_of_sample_event_slugs,
                historical_prefix_scores=historical_oos_prefix_scores,
            )
            is_gate_pass = bool(is_pvalue < float(is_pvalue_threshold))
            oos_gate_pass = bool((not out_of_sample_event_slugs) or (oos_pvalue < float(oos_pvalue_threshold)))
            accepted = bool(is_gate_pass and oos_gate_pass)
            row["_pvalue_in_sample"] = is_pvalue
            row["_pvalue_out_of_sample_walk_forward"] = oos_pvalue
            row["_gate_in_sample_pass"] = is_gate_pass
            row["_gate_out_of_sample_pass"] = oos_gate_pass
            row["_accepted"] = accepted

            # Stream each trial row to disk immediately to avoid holding the full
            # trial history in memory for large runs.
            trial_writer.write_row(row)
            rows_written += 1

            if accepted:
                accepted_trials += 1
            else:
                if not is_gate_pass:
                    rejected_trials_is += 1
                if not oos_gate_pass:
                    rejected_trials_oos += 1

            if np.isfinite(agg_score):
                historical_in_sample_scores.append(float(agg_score))
            if out_of_sample_event_slugs:
                prefix_scores: list[float] = []
                for idx, slug in enumerate(out_of_sample_event_slugs, start=1):
                    score = per_event_scores_out_of_sample.get(slug)
                    if score is None or not np.isfinite(score):
                        break
                    prefix_scores.append(float(score))
                    prefix_mean = float(np.mean(prefix_scores))
                    historical_oos_prefix_scores.setdefault(idx, []).append(prefix_mean)

            if accepted:
                top_rows.append(row)
                if len(top_rows) > top_n:
                    top_rows = sorted(top_rows, key=lambda r: float(r.get("_objective", float("-inf"))), reverse=True)[:top_n]

            if accepted and agg_score > best_score:
                best_score = agg_score
                best_params = dict(chosen_params)
                _vprint(
                    verbose,
                    f"[trial {trial_idx + 1}/{target_trials}] New best objective={best_score:.6f}",
                )

            if (trial_idx + 1) % 10 == 0:
                gc.collect()
            if verbose and (trial_idx + 1) % 10 == 0:
                print(f"Trial {trial_idx + 1}/{target_trials}: best objective={best_score:.6f}")
    except KeyboardInterrupt:
        interrupted = True
        print("KeyboardInterrupt received. Saving partial Monte Carlo results...")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        trial_writer.flush()

    acceptance_stats = {
        "accepted_trials": int(accepted_trials),
        "rejected_trials_in_sample_gate": int(rejected_trials_is),
        "rejected_trials_out_of_sample_gate": int(rejected_trials_oos),
    }
    return rows_written, best_score, best_params, interrupted, top_rows, acceptance_stats


def _run_single_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame | None,
    strategy_kwargs: dict[str, Any],
    rebalance_freq: int,
    collect_trades: bool = True,
    indicator_snapshots: "dict[str, dict] | None" = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    strategy_kwargs = dict(strategy_kwargs)
    strategy_kwargs.setdefault("track_delta_history", False)
    strategy = WeatherMarketImbalanceStrategy(**strategy_kwargs)
    setattr(strategy, "record_indicator_history", False)
    if indicator_snapshots:
        for ind in strategy.indicator_defs:
            snap = indicator_snapshots.get(ind.name)
            if snap is not None:
                cast(Any, ind).restore(snap)
    result = Backtester(strategy=strategy, rebalance_freq=rebalance_freq).run(
        prices,
        precomputed_returns=returns,
        include_returns=False,
        include_weights=False,
        include_indicator_signals=False,
    )
    strategy.finalize(prices)
    metrics = {k: float(v) for k, v in result["metrics"].items()}
    trades_df = pd.DataFrame(strategy.trade_log) if collect_trades else _EMPTY_TRADES_DF
    strategy.trade_log.clear()
    return metrics, trades_df


def _precompute_indicator_snapshots(
    bundle: "EventBundle",
    base_params: dict[str, object],
    rebalance_freq: int,
    verbose: bool = False,
    event_slug: str | None = None,
) -> dict[str, dict]:
    """Warm snapshot-capable indicators and return their state snapshots.

    Because SdBands and Vwap depend only on market data (not sampled parameters),
    the same snapshot can be restored into every Monte Carlo trial for this event,
    eliminating the per-trial catch-up loop over all bars.
    """
    strategy_kwargs: dict[str, Any] = dict(base_params)
    strategy_kwargs["vwap"] = bundle["vwap"]
    strategy_kwargs["volume"] = bundle["volume"]
    strategy_kwargs["high"] = bundle["high"]
    strategy_kwargs["low"] = bundle["low"]
    strategy_kwargs["open_"] = bundle["open_"]
    strategy_kwargs["buy_volume"] = bundle["buy_volume"]
    strategy_kwargs["sell_volume"] = bundle["sell_volume"]
    strategy_kwargs.setdefault("track_delta_history", False)

    strategy = WeatherMarketImbalanceStrategy(**strategy_kwargs)
    setattr(strategy, "record_indicator_history", False)

    # Avoid full strategy/backtester execution here. We only need indicator
    # internal state for indicators that support snapshot/restore.
    snapshot_inds = [ind for ind in strategy.indicator_defs if hasattr(ind, "snapshot")]
    if not snapshot_inds:
        return {}

    prices = bundle["prices"]
    returns = bundle["returns"]
    n_bars = len(prices)
    if n_bars <= 0:
        return {}

    step = max(1, int(rebalance_freq))
    last_idx = n_bars - 1
    processed = 0
    for i in range(0, n_bars, step):
        for ind in snapshot_inds:
            cast(Any, ind).compute(prices, returns, i)
        processed += 1
        if verbose and processed % 1000 == 0:
            slug = event_slug or "event"
            print(
                f"[precompute] '{slug}' warmed {processed} rebalance bars",
                flush=True,
            )

    # Ensure final bar is included even when rebalance_freq > 1.
    if last_idx % step != 0:
        for ind in snapshot_inds:
            cast(Any, ind).compute(prices, returns, last_idx)

    return {
        ind.name: cast(Any, ind).snapshot()
        for ind in snapshot_inds
    }


def _load_event_bundle(
    event_slug: str,
    resample_rule: str | None,
    prefer_outcome: str,
) -> EventBundle:
    prices, vwap, volume, buy_volume_full, sell_volume_full, high, low, open_ = load_event_ohlcv_resampled_with_unfiltered_cvd(
        event_slug,
        resample_rule=resample_rule,
        prefer_outcome=prefer_outcome,
    )

    prices = cast(pd.DataFrame, to_float32_frame(prices))
    returns = cast(pd.DataFrame, to_float32_frame(compute_returns(prices)))
    vwap = cast(pd.DataFrame, to_float32_frame(vwap))
    volume = cast(pd.DataFrame, to_float32_frame(volume))
    high = to_float32_frame(high)
    low = to_float32_frame(low)
    open_ = to_float32_frame(open_)
    buy_volume_full = to_float32_frame(buy_volume_full)
    sell_volume_full = to_float32_frame(sell_volume_full)

    return {
        "prices": prices,
        "returns": returns,
        "vwap": vwap,
        "volume": volume,
        "high": high,
        "low": low,
        "open_": open_,
        "buy_volume": buy_volume_full,
        "sell_volume": sell_volume_full,
        "indicator_snapshots": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo optimization for WeatherMarketImbalanceStrategy"
    )
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=WeatherMarketImbalanceStrategy.available_profiles(),
        help="Base preset profile used for defaults and non-tuned fields.",
    )
    parser.add_argument(
        "--param-config",
        type=str,
        default=str(_PARAM_CONFIG_PATH),
        help="JSON file describing parameter search rules and enabled_if dependencies.",
    )
    parser.add_argument(
        "--event-slug",
        action="append",
        default=[],
        help="Event slug to include. Pass multiple times for multi-event optimization.",
    )
    parser.add_argument(
        "--events-file",
        type=str,
        default=None,
        help=(
            "JSON file with events to include. Supports a top-level list, "
            "or an object with 'event_slugs'/'events'."
        ),
    )
    parser.add_argument(
        "--resample-minutes",
        type=int,
        default=5,
        help="Optional bar downsampling (e.g. 1, 5, 10). 0 keeps native bars.",
    )
    parser.add_argument(
        "--prefer-outcome",
        choices=["yes", "no"],
        default="yes",
        help=(
            "Preferred outcome token for price matrix. "
            "CVD buy/sell volumes are always loaded unfiltered with yes/no pairs."
        ),
    )
    parser.add_argument(
        "--objective",
        default="sharpe",
        choices=[
            "sharpe",
            "sortino",
            "calmar",
            "total_return",
            "annual_return",
            "max_drawdown",
            "volatility",
            "profit_factor",
            "winning_percentage",
        ],
        help="Metric to maximize during Monte Carlo search.",
    )
    parser.add_argument("--n-trials", type=int, default=200, help="Number of random trials.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--partition-seed",
        type=int,
        default=None,
        help="Seed for deterministic IS/OOS event split. Defaults to --seed.",
    )
    parser.add_argument(
        "--out-of-sample-ratio",
        type=float,
        default=0.4,
        help="Fraction of loaded events assigned to out-of-sample validation.",
    )
    parser.add_argument(
        "--disable-is-oos-split",
        action="store_true",
        help="Disable IS/OOS partitioning and optimize using all loaded events.",
    )
    parser.add_argument(
        "--rebalance-freq",
        type=int,
        default=1,
        help="Backtester rebalance frequency for all trials.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes for parallel event evaluation per trial. "
            "Use 1 to disable multiprocessing."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many best trials to save as top_trials.csv.",
    )
    parser.add_argument(
        "--best-rerun",
        action="store_true",
        help=(
            "Re-run best params at the end to regenerate trades and full best metrics. "
            "By default this rerun is skipped for faster completion."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print optimization progress.",
    )
    parser.add_argument(
        "--allow-longs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override preset allow_longs flag for this run.",
    )
    parser.add_argument(
        "--allow-shorts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override preset allow_shorts flag for this run.",
    )
    parser.add_argument(
        "--is-pvalue-threshold",
        type=float,
        default=0.01,
        help="Accept trial only if in-sample empirical p-value is below this threshold.",
    )
    parser.add_argument(
        "--oos-pvalue-threshold",
        type=float,
        default=0.05,
        help="Accept trial only if OOS walk-forward empirical p-value is below this threshold.",
    )
    parser.add_argument(
        "--precompute-indicators",
        action="store_true",
        help=(
            "Precompute and snapshot indicator state before Monte Carlo trials. "
            "This can be memory intensive for many events and is disabled by default."
        ),
    )
    parser.add_argument(
        "--precompute-cache-mode",
        choices=["off", "read", "write", "read-write"],
        default="read-write",
        help=(
            "Disk cache mode for indicator snapshots used by --precompute-indicators: "
            "off/read/write/read-write."
        ),
    )
    parser.add_argument(
        "--precompute-cache-dir",
        type=str,
        default=str(_DEFAULT_PRECOMPUTE_CACHE_DIR),
        help="Directory for persistent indicator snapshot cache files.",
    )
    parser.add_argument(
        "--clear-precompute-cache",
        action="store_true",
        help="Clear indicator snapshot cache directory before this run.",
    )
    args = parser.parse_args()

    if args.n_trials <= 0:
        raise ValueError("--n-trials must be > 0")
    if args.rebalance_freq <= 0:
        raise ValueError("--rebalance-freq must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if not (0.0 < float(args.out_of_sample_ratio) < 1.0):
        raise ValueError("--out-of-sample-ratio must be between 0 and 1")
    if not (0.0 < float(args.is_pvalue_threshold) < 1.0):
        raise ValueError("--is-pvalue-threshold must be between 0 and 1")
    if not (0.0 < float(args.oos_pvalue_threshold) < 1.0):
        raise ValueError("--oos-pvalue-threshold must be between 0 and 1")

    _vprint(
        bool(args.verbose),
        (
            "Starting weather Monte Carlo run "
            f"(profile={args.profile}, objective={args.objective}, requested_trials={args.n_trials})"
        ),
    )
    run_start = time.perf_counter()

    resample_rule = f"{args.resample_minutes}min" if args.resample_minutes > 0 else None

    event_slugs_cli = [str(s).strip() for s in args.event_slug if str(s).strip()]
    event_slugs_file: list[str] = []
    events_file_path: Path | None = None
    if args.events_file:
        events_file_path = Path(args.events_file)
        if not events_file_path.is_absolute():
            events_file_path = PROJECT_ROOT / events_file_path
        event_slugs_file = load_event_slugs_from_file(events_file_path)
        _vprint(
            bool(args.verbose),
            f"Loaded {len(event_slugs_file)} event slugs from '{events_file_path}'",
        )

    event_slugs = list(dict.fromkeys(event_slugs_file + event_slugs_cli))
    if not event_slugs:
        raise ValueError("Provide at least one event via --event-slug or --events-file")

    partition_seed = int(args.seed if args.partition_seed is None else args.partition_seed)
    partition: EventPartition | None = None
    if args.disable_is_oos_split:
        in_sample_event_slugs = list(event_slugs)
        out_of_sample_event_slugs: list[str] = []
    else:
        partition = split_events_in_sample_out_of_sample(
            event_slugs,
            out_of_sample_ratio=float(args.out_of_sample_ratio),
            seed=partition_seed,
        )
        in_sample_event_slugs = partition.in_sample_event_slugs
        out_of_sample_event_slugs = partition.out_of_sample_event_slugs

    event_data: dict[str, EventBundle] = {}
    for idx, slug in enumerate(event_slugs, start=1):
        # _vprint(
        #     bool(args.verbose),
        #     f"[{idx}/{len(event_slugs)}] Loading event data for '{slug}' (resample={resample_rule or 'native'})",
        # )
        event_data[slug] = _load_event_bundle(
            event_slug=slug,
            resample_rule=resample_rule,
            prefer_outcome=args.prefer_outcome,
        )
        _vprint(
            bool(args.verbose),
            (
                f"[{idx}/{len(event_slugs)}] Loaded '{slug}' "
                f"with {len(event_data[slug]['prices'])} bars "
                f"(resample={resample_rule or 'native'})"
            ),
        )

    _vprint(
        bool(args.verbose),
        (
            "Event split prepared "
            f"(in_sample={len(in_sample_event_slugs)}, out_of_sample={len(out_of_sample_event_slugs)}, "
            f"partition_seed={partition_seed}, split_enabled={not args.disable_is_oos_split})"
        ),
    )

    base_params = WeatherMarketImbalanceStrategy.profile_params(args.profile)
    if args.allow_longs is not None:
        base_params["allow_longs"] = bool(args.allow_longs)
    if args.allow_shorts is not None:
        base_params["allow_shorts"] = bool(args.allow_shorts)
    param_config_path = Path(args.param_config)
    if not param_config_path.is_absolute():
        param_config_path = PROJECT_ROOT / param_config_path
    rules = load_param_rules(param_config_path)
    max_possible_trials = estimate_max_unique_trials(rules, base_params)

    if max_possible_trials is not None and max_possible_trials > 0:
        target_trials = min(args.n_trials, max_possible_trials)
    else:
        target_trials = args.n_trials

    _vprint(
        bool(args.verbose),
        (
            "Prepared parameter search "
            f"(target_trials={target_trials}, max_possible={max_possible_trials}, workers={args.workers})"
        ),
    )

    cache_dir = Path(args.precompute_cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    cache_mode = str(args.precompute_cache_mode)
    cache_read_enabled = cache_mode in {"read", "read-write"}
    cache_write_enabled = cache_mode in {"write", "read-write"}
    if bool(args.clear_precompute_cache):
        clear_cache(cache_dir)
        _vprint(bool(args.verbose), f"Cleared precompute cache at '{cache_dir}'")

    # Snapshot precompute is optional because storing full indicator history for
    # many events can consume large amounts of memory (and can be multiplied by
    # worker processes when multiprocessing is enabled).
    precompute_enabled = bool(args.precompute_indicators)
    if precompute_enabled and int(args.workers) > 1:
        print(
            "Disabling --precompute-indicators because --workers > 1 would "
            "replicate large snapshots across worker processes."
        )
        precompute_enabled = False
    if precompute_enabled and len(event_data) > 24:
        print(
            "Disabling --precompute-indicators for this run because the event "
            f"count is high ({len(event_data)}). Use fewer events to precompute safely."
        )
        precompute_enabled = False

    if precompute_enabled:
        indicator_names: list[str] = []
        indicator_names_hash = ""
        base_params_hash = hash_payload(base_params)
        indicators_code_fingerprint = compute_file_fingerprint(
            PROJECT_ROOT / "stratlab" / "strategy" / "indicators.py"
        )
        if event_data:
            indicator_names = _resolve_indicator_names_for_bundle(next(iter(event_data.values())), base_params)
            indicator_names_hash = hash_payload(indicator_names)

        cache_hits = 0
        cache_writes = 0
        cache_misses = 0
        _vprint(bool(args.verbose), "Pre-computing indicator snapshots for each event")
        for idx, (slug, bundle) in enumerate(event_data.items(), start=1):
            key_material = build_cache_key_material(
                event_slug=slug,
                resample_rule=resample_rule or "native",
                prefer_outcome=str(args.prefer_outcome),
                rebalance_freq=int(args.rebalance_freq),
                profile=str(args.profile),
                base_params_hash=base_params_hash,
                indicator_names_hash=indicator_names_hash,
                indicator_code_fingerprint=indicators_code_fingerprint,
                source_data_fingerprint=_event_source_fingerprint(bundle),
            )
            cache_key = hash_payload(key_material)

            if cache_read_enabled:
                cached_snapshots = load_indicator_snapshots(
                    cache_dir=cache_dir,
                    event_slug=slug,
                    cache_key=cache_key,
                )
                if cached_snapshots is not None:
                    bundle["indicator_snapshots"] = cached_snapshots
                    cache_hits += 1
                    _vprint(
                        bool(args.verbose),
                        f"[precompute {idx}/{len(event_data)}] Cache hit for '{slug}'",
                    )
                    continue

            cache_misses += 1
            _vprint(
                bool(args.verbose),
                f"[precompute {idx}/{len(event_data)}] Building snapshot for '{slug}'",
            )
            bundle["indicator_snapshots"] = _precompute_indicator_snapshots(
                bundle,
                base_params,
                int(args.rebalance_freq),
                verbose=bool(args.verbose),
                event_slug=slug,
            )
            if cache_write_enabled and bundle["indicator_snapshots"] is not None:
                save_indicator_snapshots(
                    cache_dir=cache_dir,
                    event_slug=slug,
                    cache_key=cache_key,
                    meta={
                        **key_material,
                        "python_version": platform.python_version(),
                        "indicator_names": indicator_names,
                    },
                    snapshots=bundle["indicator_snapshots"],
                )
                cache_writes += 1
        _vprint(bool(args.verbose), f"Indicator snapshots ready for {len(event_data)} event(s)")
        _vprint(
            bool(args.verbose),
            (
                "Indicator cache summary: "
                f"hits={cache_hits}, misses={cache_misses}, writes={cache_writes}, "
                f"mode={cache_mode}, dir={cache_dir}"
            ),
        )
    else:
        _vprint(bool(args.verbose), "Skipping indicator snapshot precompute")

    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    event_group = event_slugs[0] if len(event_slugs) == 1 else f"{len(event_slugs)}_events"
    out_dir = RESULTS_DIR / "weather_imbalance_mc" / event_group / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    _vprint(bool(args.verbose), f"Writing outputs to {out_dir}")
    per_event_dir = out_dir / "best_trades_by_event"
    per_event_dir.mkdir(parents=True, exist_ok=True)

    trials_path = out_dir / "trials.csv"
    top_path = out_dir / "top_trials.csv"
    best_params_path = out_dir / "best_params.json"
    summary_path = out_dir / "best_run_summary.json"
    best_trades_path = out_dir / "best_trades.csv"

    top_n = max(1, int(args.top_n))
    writer = TrialCsvWriter(trials_path)
    rng = np.random.default_rng(args.seed)

    _vprint(bool(args.verbose), "Beginning Monte Carlo trial execution")
    try:
        n_trials_effective, best_score, best_params, interrupted, top_rows, acceptance_stats = _run_trials_with_interrupt_handling(
            target_trials=target_trials,
            rules=rules,
            base_params=base_params,
            event_data=event_data,
            in_sample_event_slugs=list(in_sample_event_slugs),
            out_of_sample_event_slugs=list(out_of_sample_event_slugs),
            rng=rng,
            objective=args.objective,
            rebalance_freq=int(args.rebalance_freq),
            workers=int(args.workers),
            verbose=bool(args.verbose),
            trial_writer=writer,
            top_n=top_n,
            is_pvalue_threshold=float(args.is_pvalue_threshold),
            oos_pvalue_threshold=float(args.oos_pvalue_threshold),
        )
    finally:
        writer.close()

    if top_rows:
        top_df = pd.DataFrame(top_rows).nlargest(top_n, "_objective")
    else:
        top_df = pd.DataFrame()
    top_df.to_csv(top_path, index=False)

    best_params = dict(best_params or {})
    best_params["rebalance_freq"] = int(args.rebalance_freq)
    best_params_no_rebalance = {k: v for k, v in best_params.items() if k != "rebalance_freq"}

    # Merge optimized values on top of the selected profile and run once more
    # to capture finalized trades (including end-of-data closes).
    best_metrics_by_event: dict[str, dict[str, float]] = {}
    best_trades_rows = 0
    best_trades_header_written = False
    if not args.best_rerun:
        # Recover objective-only per-event values from the top trial row.
        best_trial_row: dict[str, Any] = {}
        if not top_df.empty:
            best_trial_row = cast(dict[str, Any], top_df.iloc[0].to_dict())
        for slug in event_slugs:
            key = f"m_{slug}_{args.objective}"
            if key in best_trial_row and pd.notna(best_trial_row[key]):
                best_metrics_by_event[slug] = {args.objective: float(best_trial_row[key])}
        # Preserve output contract by creating empty trade files.
        for slug in event_slugs:
            pd.DataFrame().to_csv(per_event_dir / f"{slug}.csv", index=False)
        pd.DataFrame().to_csv(best_trades_path, index=False)
        _vprint(
            bool(args.verbose),
            "Skipped final best-params rerun; wrote parameter outputs only.",
        )
    else:
        for slug, bundle in event_data.items():
            _vprint(bool(args.verbose), f"Re-running best parameters for '{slug}'")
            best_strategy_kwargs = dict(base_params)
            best_strategy_kwargs.update(best_params_no_rebalance)
            best_strategy_kwargs["vwap"] = bundle["vwap"]
            best_strategy_kwargs["volume"] = bundle["volume"]
            best_strategy_kwargs["high"] = bundle["high"]
            best_strategy_kwargs["low"] = bundle["low"]
            best_strategy_kwargs["open_"] = bundle["open_"]
            best_strategy_kwargs["buy_volume"] = bundle["buy_volume"]
            best_strategy_kwargs["sell_volume"] = bundle["sell_volume"]

            event_metrics, event_trades = _run_single_backtest(
                prices=bundle["prices"],
                returns=bundle["returns"],
                strategy_kwargs=best_strategy_kwargs,
                rebalance_freq=int(best_params.get("rebalance_freq", args.rebalance_freq)),
                collect_trades=True,
            )
            best_metrics_by_event[slug] = event_metrics
            if not event_trades.empty:
                event_trades = event_trades.copy()
                event_trades["event_slug"] = slug
                event_trades.to_csv(
                    best_trades_path,
                    mode="a",
                    index=False,
                    header=(not best_trades_header_written),
                )
                best_trades_header_written = True
                best_trades_rows += int(len(event_trades))
            event_trades.to_csv(per_event_dir / f"{slug}.csv", index=False)
            _vprint(
                bool(args.verbose),
                f"Saved per-event best trades for '{slug}' ({len(event_trades)} rows)",
            )

    if not best_trades_header_written and args.best_rerun:
        pd.DataFrame().to_csv(best_trades_path, index=False)

    aggregate_best_metrics: dict[str, float] = {}

    def _aggregate_metrics_for_events(slugs: list[str]) -> dict[str, float]:
        if not slugs:
            return {}
        metrics_subset = {slug: best_metrics_by_event[slug] for slug in slugs if slug in best_metrics_by_event}
        if not metrics_subset:
            return {}
        metric_names = sorted(next(iter(metrics_subset.values())).keys())
        out: dict[str, float] = {}
        for metric_name in metric_names:
            vals = [v.get(metric_name, 0.0) for v in metrics_subset.values()]
            out[metric_name] = float(np.mean(vals))
        return out

    if best_metrics_by_event:
        metric_names = sorted(next(iter(best_metrics_by_event.values())).keys())
        for metric_name in metric_names:
            vals = [v.get(metric_name, 0.0) for v in best_metrics_by_event.values()]
            aggregate_best_metrics[metric_name] = float(np.mean(vals))

    aggregate_metrics_in_sample = _aggregate_metrics_for_events(in_sample_event_slugs)
    aggregate_metrics_out_of_sample = _aggregate_metrics_for_events(out_of_sample_event_slugs)

    if partition is None:
        partition_payload: dict[str, Any] = {
            "method": "disabled",
            "seed": partition_seed,
            "n_events_total": len(event_slugs),
            "n_events_in_sample": len(in_sample_event_slugs),
            "n_events_out_of_sample": 0,
            "in_sample_ratio": 1.0,
            "out_of_sample_ratio": 0.0,
            "in_sample_event_slugs": in_sample_event_slugs,
            "out_of_sample_event_slugs": out_of_sample_event_slugs,
        }
    else:
        partition_payload = partition.to_dict()

    best_payload = {
        "event_slugs": event_slugs,
        "in_sample_event_slugs": in_sample_event_slugs,
        "out_of_sample_event_slugs": out_of_sample_event_slugs,
        "events_file": str(events_file_path) if events_file_path else None,
        "profile": args.profile,
        "param_config": str(param_config_path),
        "objective": args.objective,
        "n_trials": int(args.n_trials),
        "n_trials_effective": int(n_trials_effective),
        "max_possible_trials_by_step": max_possible_trials,
        "seed": int(args.seed),
        "resample_rule": resample_rule or "native",
        "best_score": float(best_score),
        "best_params": best_params,
        "selection_gates": {
            "in_sample_pvalue_threshold": float(args.is_pvalue_threshold),
            "out_of_sample_pvalue_threshold": float(args.oos_pvalue_threshold),
        },
        "acceptance_stats": acceptance_stats,
        "partition": partition_payload,
        "search_rules": rules,
    }
    best_params_path.write_text(json.dumps(best_payload, indent=2, sort_keys=True))

    summary_payload = {
        "best_metrics_mean": aggregate_best_metrics,
        "best_metrics_mean_in_sample": aggregate_metrics_in_sample,
        "best_metrics_mean_out_of_sample": aggregate_metrics_out_of_sample,
        "best_metrics_by_event": best_metrics_by_event,
        "partition": partition_payload,
        "n_best_trades": int(best_trades_rows),
        "best_rerun": bool(args.best_rerun),
        "skip_best_rerun": bool(not args.best_rerun),
        "selection_gates": {
            "in_sample_pvalue_threshold": float(args.is_pvalue_threshold),
            "out_of_sample_pvalue_threshold": float(args.oos_pvalue_threshold),
        },
        "acceptance_stats": acceptance_stats,
        "output_dir": str(out_dir),
        "trials_file": trials_path.name,
        "top_trials_file": top_path.name,
        "best_params_file": best_params_path.name,
        "best_trades_file": best_trades_path.name,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))

    elapsed_seconds = time.perf_counter() - run_start

    print(f"Events: {event_slugs}")
    print(f"In-sample events: {in_sample_event_slugs}")
    print(f"Out-of-sample events: {out_of_sample_event_slugs}")
    print(f"Profile: {args.profile}")
    print(f"Objective: {args.objective}")
    print(f"Trials run: {n_trials_effective} / requested {args.n_trials}")
    if interrupted:
        print("Run interrupted by user. Partial results were saved.")
    if max_possible_trials is not None:
        print(f"Max unique trials from step grid: {max_possible_trials}")
    if not args.best_rerun:
        print("Skipped final best-params rerun (faster completion; no regenerated best trades).")
    print(
        "Acceptance gates: "
        f"IS p<{args.is_pvalue_threshold:.4f}, OOS walk-forward p<{args.oos_pvalue_threshold:.4f}"
    )
    print(f"Accepted trials: {acceptance_stats['accepted_trials']} / {n_trials_effective}")
    print(f"Best score: {best_score:.6f}")
    print(f"Best params: {best_params}")
    print(f"Total runtime: {_format_elapsed(elapsed_seconds)}")
    print(f"Saved: {trials_path}")
    print(f"Saved: {top_path}")
    print(f"Saved: {best_params_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {best_trades_path}")


if __name__ == "__main__":
    main()
