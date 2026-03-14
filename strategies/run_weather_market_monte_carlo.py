"""Dedicated Monte Carlo runner for WeatherMarketImbalanceStrategy.

This script samples weather-strategy parameters around a chosen preset profile,
optimizes on backtest metrics, and saves full trial history.

Usage example:
    python strategies/run_weather_market_monte_carlo.py \
        --event-slug highest-temperature-in-nyc-on-february-22-2026 \
        --profile balanced \
        --n-trials 300 \
        --objective sharpe \
        --resample-minutes 1
"""

from __future__ import annotations

import argparse
import json
import itertools
import math
import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratlab.backtest.backtester import Backtester
from stratlab.config import RESULTS_DIR
from stratlab.optimize.search import ParamSpec, ParamType
from strategies.weather_backtest import load_event_ohlcv_resampled
from strategies.weather_market_strategy import WeatherMarketImbalanceStrategy

_PARAM_CONFIG_PATH = Path(__file__).with_name("weather_market_monte_carlo_params.json")


class EventBundle(TypedDict):
    prices: pd.DataFrame
    vwap: pd.DataFrame
    volume: pd.DataFrame
    high: pd.DataFrame | None
    low: pd.DataFrame | None
    open_: pd.DataFrame | None
    buy_volume: pd.DataFrame | None
    sell_volume: pd.DataFrame | None


def _vprint(verbose: bool, message: str) -> None:
    """Print progress messages only when verbose mode is enabled."""
    if verbose:
        print(message, flush=True)


def _metric_direction(metric: str) -> float:
    """Return +1 when higher-is-better, -1 when lower-is-better."""
    lower_is_better = {"max_drawdown", "volatility"}
    return -1.0 if metric in lower_is_better else 1.0


def _load_param_rules(config_path: Path) -> list[dict[str, Any]]:
    """Load Monte Carlo parameter rules from JSON file."""
    raw = json.loads(config_path.read_text())
    rules = raw.get("parameters", [])
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"Invalid parameter config at {config_path}: expected non-empty 'parameters' list")

    seen: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("Each parameter rule must be an object")
        name = str(rule.get("name", "")).strip()
        if not name:
            raise ValueError("Each parameter rule must include a non-empty 'name'")
        if name in seen:
            raise ValueError(f"Duplicate parameter rule for {name!r}")
        seen.add(name)
        rule_type = str(rule.get("type", "")).strip()
        if rule_type not in {"bool", "int", "float", "log_float"}:
            raise ValueError(f"Unsupported rule type {rule_type!r} for parameter {name!r}")
        if rule_type == "bool":
            values = rule.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError(f"Boolean rule {name!r} requires non-empty 'values'")
        else:
            if "low" not in rule or "high" not in rule:
                raise ValueError(f"Numeric rule {name!r} requires 'low' and 'high'")
    return rules


def _rule_enabled(rule: dict[str, Any], chosen: dict[str, Any], base_params: dict[str, object]) -> bool:
    cond = rule.get("enabled_if", {})
    if not cond:
        return True
    if not isinstance(cond, dict):
        raise ValueError(f"enabled_if for {rule.get('name')} must be a mapping")
    for dep_name, dep_val in cond.items():
        actual = chosen.get(dep_name, base_params.get(dep_name))
        if actual != dep_val:
            return False
    return True


def _rule_to_spec(rule: dict[str, Any]) -> ParamSpec:
    low = float(rule["low"])
    high = float(rule["high"])
    rtype = str(rule["type"])
    if rtype == "int":
        return ParamSpec(low=int(round(low)), high=int(round(high)), param_type=ParamType.INT)
    if rtype == "log_float":
        return ParamSpec(low=low, high=high, param_type=ParamType.LOG_FLOAT)
    return ParamSpec(low=low, high=high, param_type=ParamType.FLOAT)


def _sample_trial_params(
    rules: list[dict[str, Any]],
    base_params: dict[str, object],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Sample one candidate respecting enabled_if conditions."""
    sampled: dict[str, Any] = {}
    for rule in rules:
        name = str(rule["name"])
        if not _rule_enabled(rule, sampled, base_params):
            continue

        rtype = str(rule["type"])
        if rtype == "bool":
            values = rule.get("values", [True, False])
            sampled[name] = values[int(rng.integers(0, len(values)))]
            continue

        spec = _rule_to_spec(rule)
        raw_val = spec.sample(rng)
        step = float(rule.get("step", 0.0))
        if spec.param_type in (ParamType.INT, ParamType.LOG_INT):
            sampled[name] = int(round(_round_to_step(float(raw_val), max(1.0, step), spec.low, spec.high)))
        else:
            sampled[name] = float(_round_to_step(float(raw_val), step, spec.low, spec.high))
    return sampled


def _estimate_max_unique_trials(rules: list[dict[str, Any]], base_params: dict[str, object]) -> int | None:
    """Estimate unique parameter combinations implied by rule steps and booleans."""
    bool_rules = [r for r in rules if str(r.get("type")) == "bool"]
    num_rules = [r for r in rules if str(r.get("type")) != "bool"]

    # Build all boolean assignments.
    bool_value_lists: list[list[Any]] = [list(r.get("values", [True, False])) for r in bool_rules]
    total = 0
    for combo in itertools.product(*bool_value_lists) if bool_value_lists else [()]:
        chosen_bool = {str(r["name"]): combo[i] for i, r in enumerate(bool_rules)}
        branch_product = 1
        for rule in num_rules:
            if not _rule_enabled(rule, chosen_bool, base_params):
                continue
            low = float(rule["low"])
            high = float(rule["high"])
            if low == high:
                levels = 1
            else:
                step = float(rule.get("step", 0.0))
                if step <= 0:
                    return None
                levels = int(max(0, math.floor((high - low) / step) + 1))
                levels = max(1, levels)
            branch_product *= levels
        total += branch_product
    return int(total)


def _round_to_step(value: float, step: float, low: float, high: float) -> float:
    if step <= 0:
        return float(min(max(value, low), high))
    snapped = low + round((value - low) / step) * step
    return float(min(max(snapped, low), high))


def _params_key(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(params.items(), key=lambda x: x[0]))


def _run_trials_with_interrupt_handling(
    target_trials: int,
    rules: list[dict[str, Any]],
    base_params: dict[str, object],
    event_data: dict[str, EventBundle],
    rng: np.random.Generator,
    objective: str,
    rebalance_freq: int,
    verbose: bool,
) -> tuple[list[dict[str, Any]], float, dict[str, Any] | None, bool]:
    """Run Monte Carlo trials and return partial results if interrupted."""
    seen: set[tuple[tuple[str, Any], ...]] = set()
    trial_rows: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_params: dict[str, Any] | None = None
    interrupted = False

    max_retries_per_trial = 30
    objective_sign = _metric_direction(objective)

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

    try:
        for trial_idx in range(target_trials):
            _vprint(verbose, f"[trial {trial_idx + 1}/{target_trials}] Sampling candidate parameters")
            chosen_params: dict[str, Any] | None = None
            for _ in range(max_retries_per_trial):
                sampled = _sample_trial_params(rules, base_params, rng)
                pkey = _params_key(sampled)
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
            row: dict[str, Any] = dict(chosen_params)

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
                    strategy_kwargs=strategy_kwargs,
                    rebalance_freq=rebalance_freq,
                )
                raw_obj = float(metrics.get(objective, 0.0))
                signed_obj = objective_sign * raw_obj
                per_event_scores[slug] = signed_obj
                row[f"m_{slug}_{objective}"] = raw_obj
                _vprint(
                    verbose,
                    f"[trial {trial_idx + 1}/{target_trials}] Event '{slug}' {objective}={raw_obj:.6f}",
                )

            agg_score = float(np.mean(list(per_event_scores.values()))) if per_event_scores else float("-inf")
            row["_objective"] = agg_score
            row["_event_count"] = len(per_event_scores)
            trial_rows.append(row)

            if agg_score > best_score:
                best_score = agg_score
                best_params = dict(chosen_params)
                _vprint(
                    verbose,
                    f"[trial {trial_idx + 1}/{target_trials}] New best objective={best_score:.6f}",
                )

            if verbose and (trial_idx + 1) % 10 == 0:
                print(f"Trial {trial_idx + 1}/{target_trials}: best objective={best_score:.6f}")
    except KeyboardInterrupt:
        interrupted = True
        print("KeyboardInterrupt received. Saving partial Monte Carlo results...")

    return trial_rows, best_score, best_params, interrupted


def _run_single_backtest(
    prices: pd.DataFrame,
    strategy_kwargs: dict[str, Any],
    rebalance_freq: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    strategy = WeatherMarketImbalanceStrategy(**strategy_kwargs)
    result = Backtester(strategy=strategy, rebalance_freq=rebalance_freq).run(prices)
    strategy.finalize(prices)
    metrics = {k: float(v) for k, v in result["metrics"].items()}
    trades_df = pd.DataFrame(strategy.trade_log)
    return metrics, trades_df


def _load_event_bundle(
    event_slug: str,
    resample_rule: str | None,
    prefer_outcome: str,
) -> EventBundle:
    prices, vwap, volume, buy_volume, sell_volume, high, low, open_ = load_event_ohlcv_resampled(
        event_slug,
        resample_rule=resample_rule,
        prefer_outcome=prefer_outcome,
    )
    try:
        (
            _p,
            _v,
            _vol,
            buy_volume_full,
            sell_volume_full,
            _h,
            _l,
            _o,
        ) = load_event_ohlcv_resampled(
            event_slug,
            resample_rule=resample_rule,
            prefer_outcome=None,
        )
    except Exception:
        buy_volume_full = buy_volume
        sell_volume_full = sell_volume

    return {
        "prices": prices,
        "vwap": vwap,
        "volume": volume,
        "high": high,
        "low": low,
        "open_": open_,
        "buy_volume": buy_volume_full,
        "sell_volume": sell_volume_full,
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
        required=True,
        help="Event slug to include. Pass multiple times for multi-event optimization.",
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
        help="Preferred outcome token for price matrix.",
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
        "--rebalance-freq",
        type=int,
        default=1,
        help="Backtester rebalance frequency for all trials.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many best trials to save as top_trials.csv.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print optimization progress.",
    )
    args = parser.parse_args()

    if args.n_trials <= 0:
        raise ValueError("--n-trials must be > 0")
    if args.rebalance_freq <= 0:
        raise ValueError("--rebalance-freq must be > 0")

    _vprint(
        bool(args.verbose),
        (
            "Starting weather Monte Carlo run "
            f"(profile={args.profile}, objective={args.objective}, requested_trials={args.n_trials})"
        ),
    )

    resample_rule = f"{args.resample_minutes}min" if args.resample_minutes > 0 else None

    event_slugs = list(dict.fromkeys(args.event_slug))
    event_data: dict[str, EventBundle] = {}
    for idx, slug in enumerate(event_slugs, start=1):
        _vprint(
            bool(args.verbose),
            f"[{idx}/{len(event_slugs)}] Loading event data for '{slug}' (resample={resample_rule or 'native'})",
        )
        event_data[slug] = _load_event_bundle(
            event_slug=slug,
            resample_rule=resample_rule,
            prefer_outcome=args.prefer_outcome,
        )
        _vprint(
            bool(args.verbose),
            (
                f"[{idx}/{len(event_slugs)}] Loaded '{slug}' "
                f"with {len(event_data[slug]['prices'])} bars"
            ),
        )

    base_params = WeatherMarketImbalanceStrategy.profile_params(args.profile)
    param_config_path = Path(args.param_config)
    if not param_config_path.is_absolute():
        param_config_path = PROJECT_ROOT / param_config_path
    rules = _load_param_rules(param_config_path)
    max_possible_trials = _estimate_max_unique_trials(rules, base_params)

    if max_possible_trials is not None and max_possible_trials > 0:
        target_trials = min(args.n_trials, max_possible_trials)
    else:
        target_trials = args.n_trials

    _vprint(
        bool(args.verbose),
        (
            "Prepared parameter search "
            f"(target_trials={target_trials}, max_possible={max_possible_trials})"
        ),
    )

    rng = np.random.default_rng(args.seed)
    _vprint(bool(args.verbose), "Beginning Monte Carlo trial execution")
    trial_rows, best_score, best_params, interrupted = _run_trials_with_interrupt_handling(
        target_trials=target_trials,
        rules=rules,
        base_params=base_params,
        event_data=event_data,
        rng=rng,
        objective=args.objective,
        rebalance_freq=int(args.rebalance_freq),
        verbose=bool(args.verbose),
    )

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

    trials_df = pd.DataFrame(trial_rows)
    trials_df.to_csv(trials_path, index=False)

    top_n = max(1, int(args.top_n))
    top_df = trials_df.nlargest(top_n, "_objective") if not trials_df.empty else trials_df
    top_df.to_csv(top_path, index=False)

    best_params = dict(best_params or {})
    best_params["rebalance_freq"] = int(args.rebalance_freq)
    best_params_no_rebalance = {k: v for k, v in best_params.items() if k != "rebalance_freq"}

    # Merge optimized values on top of the selected profile and run once more
    # to capture finalized trades (including end-of-data closes).
    best_metrics_by_event: dict[str, dict[str, float]] = {}
    all_best_trades: list[pd.DataFrame] = []
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
            strategy_kwargs=best_strategy_kwargs,
            rebalance_freq=int(best_params.get("rebalance_freq", args.rebalance_freq)),
        )
        best_metrics_by_event[slug] = event_metrics
        if not event_trades.empty:
            event_trades = event_trades.copy()
            event_trades["event_slug"] = slug
            all_best_trades.append(event_trades)
        event_trades.to_csv(per_event_dir / f"{slug}.csv", index=False)
        _vprint(
            bool(args.verbose),
            f"Saved per-event best trades for '{slug}' ({len(event_trades)} rows)",
        )

    if all_best_trades:
        best_trades = pd.concat(all_best_trades, ignore_index=True)
    else:
        best_trades = pd.DataFrame()
    best_trades.to_csv(best_trades_path, index=False)

    aggregate_best_metrics: dict[str, float] = {}
    if best_metrics_by_event:
        metric_names = sorted(next(iter(best_metrics_by_event.values())).keys())
        for metric_name in metric_names:
            vals = [v.get(metric_name, 0.0) for v in best_metrics_by_event.values()]
            aggregate_best_metrics[metric_name] = float(np.mean(vals))

    best_payload = {
        "event_slugs": event_slugs,
        "profile": args.profile,
        "param_config": str(param_config_path),
        "objective": args.objective,
        "n_trials": int(args.n_trials),
        "n_trials_effective": int(len(trials_df)),
        "max_possible_trials_by_step": max_possible_trials,
        "seed": int(args.seed),
        "resample_rule": resample_rule or "native",
        "best_score": float(best_score),
        "best_params": best_params,
        "search_rules": rules,
    }
    best_params_path.write_text(json.dumps(best_payload, indent=2, sort_keys=True))

    summary_payload = {
        "best_metrics_mean": aggregate_best_metrics,
        "best_metrics_by_event": best_metrics_by_event,
        "n_best_trades": int(len(best_trades)),
        "output_dir": str(out_dir),
        "trials_file": trials_path.name,
        "top_trials_file": top_path.name,
        "best_params_file": best_params_path.name,
        "best_trades_file": best_trades_path.name,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))

    print(f"Events: {event_slugs}")
    print(f"Profile: {args.profile}")
    print(f"Objective: {args.objective}")
    print(f"Trials run: {len(trials_df)} / requested {args.n_trials}")
    if interrupted:
        print("Run interrupted by user. Partial results were saved.")
    if max_possible_trials is not None:
        print(f"Max unique trials from step grid: {max_possible_trials}")
    print(f"Best score: {best_score:.6f}")
    print(f"Best params: {best_params}")
    print(f"Saved: {trials_path}")
    print(f"Saved: {top_path}")
    print(f"Saved: {best_params_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {best_trades_path}")


if __name__ == "__main__":
    main()
