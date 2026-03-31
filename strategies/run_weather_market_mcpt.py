"""MCPT-based Monte Carlo runner for WeatherMarketImbalanceStrategy.

This script performs parameter optimization via random search followed by
insample and outsample Monte Carlo Permutation Tests (MCPT) to validate
strategy significance.

Usage example:
    python strategies/run_weather_market_monte_carlo.py \\
        --events-file strategies/weather_market_monte_carlo_events.json \\
        --param-config strategies/weather_market_monte_carlo_params.json \\
        --profile balanced \\
        --n-trials 300 \\
        --n-permutations-insample 1000 \\
        --n-permutations-outsample 1000 \\
        --objective sharpe \\
        --resample-minutes 1
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratlab.backtest.backtester import compute_returns
from stratlab.config import RESULTS_DIR
from stratlab.data.dtype_utils import to_float32_frame
from stratlab.io.events import load_event_slugs_from_file
from stratlab.optimize.rule_search import load_param_rules
from stratlab.report.plot import plot_cumulative_log_return, plot_mcpt_histogram
from stratlab.validation.mcpt import (
    MCPTResult,
    concat_returns_in_order,
    make_scoring_fn,
    run_insample_mcpt,
    run_oos_mcpt,
)
from stratlab.validation.batch_mcpt import run_batch_mcpt
from stratlab.validation.partition import split_events_in_sample_out_of_sample
from strategies.weather_backtest import load_event_ohlcv_resampled_with_unfiltered_cvd
from strategies.weather_mcpt_adapter import (
    EventBundle,
    WeatherMarketMCPTAdapter,
    bundles_to_mcpt_events,
)
from strategies.weather_market_strategy import WeatherMarketImbalanceStrategy

_PARAM_CONFIG_PATH = Path(__file__).with_name("weather_market_monte_carlo_params.json")


def _vprint(verbose: bool, message: str) -> None:
    """Print progress messages only when verbose mode is enabled."""
    if verbose:
        print(message, flush=True)


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS."""
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
        description="MCPT-based Monte Carlo runner for WeatherMarketImbalanceStrategy"
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
        default="total_return",
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
        help="Metric to maximize during optimization and MCPT scoring.",
    )
    parser.add_argument("--n-trials", type=int, default=200, help="Number of random search trials for optimization.")
    parser.add_argument(
        "--n-permutations-insample",
        type=int,
        default=1000,
        help="Number of bar-permutation iterations for insample MCPT.",
    )
    parser.add_argument(
        "--n-permutations-outsample",
        type=int,
        default=1000,
        help="Number of bar-permutation iterations for outsample MCPT.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--partition-seed",
        type=int,
        default=None,
        help="Seed for deterministic insample/outsample event split. Defaults to --seed.",
    )
    parser.add_argument(
        "--out-of-sample-ratio",
        type=float,
        default=0.4,
        help="Fraction of loaded events assigned to out-of-sample validation.",
    )
    parser.add_argument(
        "--rebalance-freq",
        type=int,
        default=1,
        help="Backtester rebalance frequency for all trials.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print optimization and MCPT progress.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for the permutation loop (default: 1 = sequential).",
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
        "--batch",
        action="store_true",
        default=False,
        help=(
            "Use the batch-vectorized permutation engine instead of the "
            "per-permutation loop. Faster but requires hold-to-end exit mode."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=["random", "bayesian"],
        default="random",
        help=(
            "Parameter search algorithm. 'random' uses uniform random sampling; "
            "'bayesian' uses Optuna TPE (Tree-structured Parzen Estimator)."
        ),
    )
    args = parser.parse_args()

    if args.n_trials <= 0:
        raise ValueError("--n-trials must be > 0")
    if args.rebalance_freq <= 0:
        raise ValueError("--rebalance-freq must be > 0")
    if args.n_permutations_insample <= 0:
        raise ValueError("--n-permutations-insample must be > 0")
    if args.n_permutations_outsample <= 0:
        raise ValueError("--n-permutations-outsample must be > 0")
    if not (0.0 < float(args.out_of_sample_ratio) < 1.0):
        raise ValueError("--out-of-sample-ratio must be between 0 and 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    verbose = bool(args.verbose)
    _vprint(
        verbose,
        (
            "Starting weather MCPT run "
            f"(profile={args.profile}, objective={args.objective}, "
            f"optimizer={args.optimizer}, "
            f"n_trials={args.n_trials}, "
            f"insample_perms={args.n_permutations_insample}, "
            f"outsample_perms={args.n_permutations_outsample}, "
            f"workers={args.workers})"
        ),
    )
    run_start = time.perf_counter()

    # --- Load events --------------------------------------------------------
    resample_rule = f"{args.resample_minutes}min" if args.resample_minutes > 0 else None

    event_slugs_cli = [str(s).strip() for s in args.event_slug if str(s).strip()]
    event_slugs_file: list[str] = []
    events_file_path: Path | None = None
    if args.events_file:
        events_file_path = Path(args.events_file)
        if not events_file_path.is_absolute():
            events_file_path = PROJECT_ROOT / events_file_path
        event_slugs_file = load_event_slugs_from_file(events_file_path)
        _vprint(verbose, f"Loaded {len(event_slugs_file)} event slugs from '{events_file_path}'")

    event_slugs = list(dict.fromkeys(event_slugs_file + event_slugs_cli))
    if not event_slugs:
        raise ValueError("Provide at least one event via --event-slug or --events-file")

    event_data: dict[str, EventBundle] = {}
    for idx, slug in enumerate(event_slugs, start=1):
        event_data[slug] = _load_event_bundle(
            event_slug=slug,
            resample_rule=resample_rule,
            prefer_outcome=args.prefer_outcome,
        )
        _vprint(
            verbose,
            (
                f"[{idx}/{len(event_slugs)}] Loaded '{slug}' "
                f"with {len(event_data[slug]['prices'])} bars "
                f"(resample={resample_rule or 'native'})"
            ),
        )

    # --- Partition IS / OOS -------------------------------------------------
    partition_seed = int(args.seed if args.partition_seed is None else args.partition_seed)
    partition = split_events_in_sample_out_of_sample(
        event_slugs,
        out_of_sample_ratio=float(args.out_of_sample_ratio),
        seed=partition_seed,
    )
    in_sample_event_slugs = partition.in_sample_event_slugs
    out_of_sample_event_slugs = partition.out_of_sample_event_slugs

    _vprint(
        verbose,
        (
            f"Event partition: insample={len(in_sample_event_slugs)}, "
            f"outsample={len(out_of_sample_event_slugs)}, seed={partition_seed}"
        ),
    )

    # --- Convert to MCPT format ---------------------------------------------
    all_mcpt_events = bundles_to_mcpt_events(event_data)
    is_events = {s: all_mcpt_events[s] for s in in_sample_event_slugs}
    oos_events = {s: all_mcpt_events[s] for s in out_of_sample_event_slugs}

    # --- Prepare adapter ----------------------------------------------------
    base_params = WeatherMarketImbalanceStrategy.profile_params(args.profile)
    if args.allow_longs is not None:
        base_params["allow_longs"] = bool(args.allow_longs)
    if args.allow_shorts is not None:
        base_params["allow_shorts"] = bool(args.allow_shorts)

    param_config_path = Path(args.param_config)
    if not param_config_path.is_absolute():
        param_config_path = PROJECT_ROOT / param_config_path
    rules = load_param_rules(param_config_path)

    scoring_fn = make_scoring_fn(args.objective)
    rng = np.random.default_rng(args.seed)

    adapter = WeatherMarketMCPTAdapter(
        base_params=base_params,
        rules=rules,
        rng=rng,
        objective=args.objective,
        n_trials=args.n_trials,
        rebalance_freq=args.rebalance_freq,
        scoring_fn=scoring_fn,
        verbose=verbose,
        log_fn=lambda msg: _vprint(verbose, msg),
        optimizer=args.optimizer,
    )

    # --- Output directory ---------------------------------------------------
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    event_group = event_slugs[0] if len(event_slugs) == 1 else f"{len(event_slugs)}_events"
    out_dir = RESULTS_DIR / "weather_imbalance_mc" / event_group / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    _vprint(verbose, f"Writing outputs to {out_dir}")

    # --- Insample MCPT ------------------------------------------------------
    _vprint(verbose, "Running insample MCPT")
    if args.batch:
        # Optimization still runs via the adapter (captures real returns).
        # The permutation loop is replaced by the batch-vectorized engine.
        _vprint(verbose, "[batch] optimizing parameters via adapter")
        is_params = adapter.optimize(is_events, in_sample_event_slugs)
        is_result: MCPTResult = run_batch_mcpt(
            adapter,
            is_events,
            in_sample_event_slugs,
            params=is_params,
            n_permutations=args.n_permutations_insample,
            scoring_fn=scoring_fn,
            verbose=verbose,
            log_fn=lambda msg: _vprint(verbose, msg),
            label="insample",
        )
        is_result.params = is_params
    else:
        is_result = run_insample_mcpt(
            adapter,
            is_events,
            in_sample_event_slugs,
            n_permutations=args.n_permutations_insample,
            scoring_fn=scoring_fn,
            verbose=verbose,
            log_fn=lambda msg: _vprint(verbose, msg),
            workers=args.workers,
        )
    gc.collect()

    _vprint(
        verbose,
        f"Insample MCPT done: real_pf={is_result.real_pf:.6f}, p_value={is_result.p_value:.6f}",
    )

    # --- Outsample MCPT -----------------------------------------------------
    oos_result: MCPTResult | None = None
    if out_of_sample_event_slugs:
        _vprint(verbose, "Running outsample MCPT")
        if args.batch:
            oos_result = run_batch_mcpt(
                adapter,
                oos_events,
                out_of_sample_event_slugs,
                params=is_result.params,
                n_permutations=args.n_permutations_outsample,
                scoring_fn=scoring_fn,
                verbose=verbose,
                log_fn=lambda msg: _vprint(verbose, msg),
                label="outsample",
            )
        else:
            oos_result = run_oos_mcpt(
                adapter,
                oos_events,
                out_of_sample_event_slugs,
                params=is_result.params,
                n_permutations=args.n_permutations_outsample,
                scoring_fn=scoring_fn,
                verbose=verbose,
                log_fn=lambda msg: _vprint(verbose, msg),
                workers=args.workers,
            )
        gc.collect()

        _vprint(
            verbose,
            f"Outsample MCPT done: real_pf={oos_result.real_pf:.6f}, p_value={oos_result.p_value:.6f}",
        )
    else:
        _vprint(verbose, "No outsample events — skipping outsample MCPT")

    # --- Plots --------------------------------------------------------------
    is_cohort_rets = concat_returns_in_order(is_result.per_event_returns, in_sample_event_slugs)
    objective_label = args.objective.replace("_", " ").title()
    fig_is_hist = plot_mcpt_histogram(
        is_result.permuted_pfs,
        is_result.real_pf,
        is_result.p_value,
        title="Insample MCPT",
        xlabel=objective_label,
    )
    fig_is_hist.savefig(out_dir / "insample_mcpt_histogram.png", dpi=150)

    fig_is_cum = plot_cumulative_log_return(
        is_cohort_rets, title="Insample Cumulative Log Return"
    )
    fig_is_cum.savefig(out_dir / "insample_cumulative_log_return.png", dpi=150)

    if oos_result is not None:
        oos_cohort_rets = concat_returns_in_order(oos_result.per_event_returns, out_of_sample_event_slugs)
        fig_oos_hist = plot_mcpt_histogram(
            oos_result.permuted_pfs,
            oos_result.real_pf,
            oos_result.p_value,
            title="Outsample MCPT",
            xlabel=objective_label,
        )
        fig_oos_hist.savefig(out_dir / "outsample_mcpt_histogram.png", dpi=150)

        fig_oos_cum = plot_cumulative_log_return(
            oos_cohort_rets, title="Outsample Cumulative Log Return"
        )
        fig_oos_cum.savefig(out_dir / "outsample_cumulative_log_return.png", dpi=150)

    import matplotlib.pyplot as plt
    plt.close("all")

    # --- JSON summary -------------------------------------------------------
    partition_payload = partition.to_dict()

    summary_payload: dict[str, Any] = {
        "profile": args.profile,
        "objective": args.objective,
        "optimizer": args.optimizer,
        "n_trials": args.n_trials,
        "n_permutations_insample": args.n_permutations_insample,
        "n_permutations_outsample": args.n_permutations_outsample,
        "seed": args.seed,
        "partition_seed": partition_seed,
        "resample_rule": resample_rule or "native",
        "rebalance_freq": args.rebalance_freq,
        "event_slugs": event_slugs,
        "events_file": str(events_file_path) if events_file_path else None,
        "partition": partition_payload,
        "insample": {
            "real_pf": is_result.real_pf,
            "p_value": is_result.p_value,
            "per_event_pf": is_result.per_event_pf,
            "n_permutations": args.n_permutations_insample,
        },
        "best_params": is_result.params,
        "param_config": str(param_config_path),
        "search_rules": rules,
    }
    if oos_result is not None:
        summary_payload["outsample"] = {
            "real_pf": oos_result.real_pf,
            "p_value": oos_result.p_value,
            "per_event_pf": oos_result.per_event_pf,
            "n_permutations": args.n_permutations_outsample,
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True, default=str))

    # --- Console output -----------------------------------------------------
    elapsed_seconds = time.perf_counter() - run_start

    print(f"Events: {event_slugs}")
    print(f"Insample events: {in_sample_event_slugs}")
    print(f"Outsample events: {out_of_sample_event_slugs}")
    print(f"Profile: {args.profile}")
    print(f"Objective: {args.objective}")
    print(f"Optimize trials: {args.n_trials}")
    print(f"Best params: {is_result.params}")
    print(f"Insample — real score: {is_result.real_pf:.6f}, p-value: {is_result.p_value:.6f}")
    if oos_result is not None:
        print(f"Outsample — real score: {oos_result.real_pf:.6f}, p-value: {oos_result.p_value:.6f}")
    print(f"Total runtime: {_format_elapsed(elapsed_seconds)}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
