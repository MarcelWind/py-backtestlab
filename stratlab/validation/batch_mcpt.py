"""Batch-vectorized Monte Carlo Permutation Test runner.

Provides :func:`run_batch_mcpt` which replaces the per-permutation Python
loop with a fully vectorized pipeline:

1. Pre-generate *N* permuted datasets per event (3D numpy arrays).
2. Compute all indicators for all permutations in one pass.
3. Compute entry signals, positions, and weights in batch.
4. Dot-product weights × returns → portfolio returns → score.

Falls back to the existing per-permutation loop for features that the
vectorized engine does not yet support (e.g., TP/SL exits, trailing stop).
"""

from __future__ import annotations

import gc
import platform
import time
from typing import Any, Callable

import numpy as np
import pandas as pd


def _rss_mb() -> float:
    """Return current process RSS in MB (Linux/macOS)."""
    try:
        import resource
        # macOS reports in bytes, Linux in KB
        rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return rusage / (1024 * 1024)
        return rusage / 1024
    except Exception:
        return 0.0

from .batch_permute import BatchPermutedEvent, batch_permute_event
from .mcpt import (
    MCPTResult,
    SubmarketMCPT,
    EventLevelMCPTStrategy,
    ScoringFn,
    concat_returns_in_order,
    profit_factor,
    _apply_event_strategy_to_events,
)
from ..strategy.vectorized_indicators import compute_all_indicators
from ..strategy.vectorized_weights import batch_strategy_returns


def _reconstruct_matrices(event_data: dict[str, pd.DataFrame]):
    """Light wrapper to unpack event data into aligned matrices."""
    from ..backtest.backtester import compute_returns

    cols = list(event_data.keys())
    if not cols:
        empty = pd.DataFrame()
        return empty, empty, None, None, None, None, None, None, None

    first = event_data[cols[0]]
    idx = first.index

    prices = pd.DataFrame({c: event_data[c]["close"] for c in cols}, index=idx)
    returns = compute_returns(prices)

    def _maybe(field):
        data = {c: event_data[c][field] for c in cols if field in event_data[c].columns}
        return pd.DataFrame(data, index=idx) if data else None

    vwap = _maybe("vwap")
    volume = _maybe("volume")
    high = _maybe("high")
    low = _maybe("low")
    open_ = _maybe("open")

    buy_data, sell_data = {}, {}
    for c in cols:
        df = event_data[c]
        if "buy_volume__yes" in df.columns:
            buy_data[f"{c}__yes"] = df["buy_volume__yes"]
        if "buy_volume__no" in df.columns:
            buy_data[f"{c}__no"] = df["buy_volume__no"]
        if "sell_volume__yes" in df.columns:
            sell_data[f"{c}__yes"] = df["sell_volume__yes"]
        if "sell_volume__no" in df.columns:
            sell_data[f"{c}__no"] = df["sell_volume__no"]

    buy_volume = pd.DataFrame(buy_data, index=idx) if buy_data else None
    sell_volume = pd.DataFrame(sell_data, index=idx) if sell_data else None

    return prices, returns, vwap, volume, high, low, open_, buy_volume, sell_volume


def _build_indicator_params(params: dict) -> dict:
    """Extract indicator-relevant params from strategy params dict."""
    lh = params.get("lookback_hours")
    # Convert lookback_hours to bars (5-min data → 12 bars/hour)
    # The caller may already provide lookback_bars explicitly.
    lookback_bars = params.get("lookback_bars")
    if lookback_bars is None and lh is not None:
        lookback_bars = max(1, int(float(lh) * 12))
    if lookback_bars is None:
        lookback_bars = 72  # default 6 hours at 5-min

    return {
        "lookback_bars": lookback_bars,
        "mean_reversion_window": params.get("mean_reversion_window", 5),
        "vwap_slope_lookback": params.get("vwap_slope_lookback", 30),
        "vwap_slope_mode": params.get("vwap_slope_mode", "scaled"),
        "vwap_slope_value_per_point": params.get("vwap_slope_value_per_point", 1e-4),
        "vwap_slope_scale": params.get("vwap_slope_scale", 1.0),
    }


def run_batch_mcpt(
    strategy: EventLevelMCPTStrategy,
    events: dict[str, dict[str, pd.DataFrame]],
    event_order: list[str],
    params: Any,
    n_permutations: int = 1000,
    *,
    scoring_fn: Callable[[pd.Series], float] = profit_factor,
    verbose: bool = False,
    log_fn: Callable[[str], None] | None = None,
    label: str = "batch",
) -> MCPTResult:
    """Run MCPT using the batch-vectorized pipeline.

    This function computes the real strategy returns via the existing
    per-bar adapter, then uses the vectorized engine for all permutations.

    Parameters
    ----------
    strategy : EventLevelMCPTStrategy
        Adapter with ``apply_event`` method.
    events : dict
    event_order : list[str]
    params : Any
        Optimised strategy parameters.
    n_permutations : int
    scoring_fn : callable
    verbose : bool
    log_fn : callable
    label : str
        Tag for log messages (``"insample"`` or ``"outsample"``).

    Returns
    -------
    MCPTResult
    """
    if log_fn is None:
        log_fn = print

    # --- Real returns (via existing adapter) --------------------------------
    real_event_rets, real_event_pf, real_mkt_rets, real_mkt_pf = _apply_event_strategy_to_events(
        strategy, events, event_order, params, scoring_fn,
    )
    real_cohort_rets = concat_returns_in_order(real_event_rets, event_order)
    real_pf = scoring_fn(real_cohort_rets)

    total_perms = n_permutations - 1
    if verbose:
        log_fn(f"[{label}] real score={real_pf:.6f}, starting {total_perms} batch permutations")

    # --- Pre-compute total bar count for compact accumulator ----------------
    event_bar_counts: dict[str, int] = {}
    for slug in event_order:
        event_data = events[slug]
        if not event_data:
            event_bar_counts[slug] = 0
            continue
        first_key = next(iter(event_data))
        n_bars = len(event_data[first_key])
        event_bar_counts[slug] = n_bars
    total_bars = sum(event_bar_counts.values())

    # Pre-allocated compact accumulator: (total_perms, total_bars) float32.
    # Each event fills its slice in-place; NaN marks unused slots.
    perm_cohort_rets = np.full(
        (total_perms, total_bars), np.nan, dtype=np.float32,
    )
    bar_offset = 0

    # Per-submarket accumulators — only float scores (not full Series)
    def _submarket_keys(slug: str) -> list[str]:
        return list(real_mkt_rets[slug].keys())

    perm_mkt_pfs: dict[str, dict[str, list[float]]] = {
        slug: {col: [] for col in _submarket_keys(slug)} for slug in event_order
    }

    # Strategy params dict for the vectorized engine
    strat_params = dict(params) if isinstance(params, dict) else {}
    ind_params = _build_indicator_params(strat_params)

    t0 = time.monotonic()

    for slug in event_order:
        event_data = events[slug]
        prices, returns, vwap, volume, high, low, open_, buy_vol, sell_vol = _reconstruct_matrices(event_data)
        if prices.empty or len(prices) < 3:
            bar_offset += event_bar_counts.get(slug, 0)
            continue

        assets = prices.columns.tolist()
        n_bars = len(prices)

        # Batch permute the event
        bp = batch_permute_event(
            prices,
            total_perms,
            high=high,
            low=low,
            open_=open_,
            vwap=vwap,
            volume=volume,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            seed_offset=1,
        )

        # Free reconstructed matrices — data now lives in bp arrays
        del prices, returns, vwap, volume, high, low, open_, buy_vol, sell_vol

        # Compute all indicators on permuted data
        indicators = compute_all_indicators(
            bp.close,
            high=bp.high,
            low=bp.low,
            open_=bp.open_,
            volume=bp.volume,
            buy_volume_yes=bp.buy_volume_yes,
            buy_volume_no=bp.buy_volume_no,
            sell_volume_yes=bp.sell_volume_yes,
            sell_volume_no=bp.sell_volume_no,
            **ind_params,
        )

        # Compute batch strategy returns
        port_rets_2d = batch_strategy_returns(indicators, bp.returns, strat_params)
        # port_rets_2d: (total_perms, n_bars) float32

        # Free heavy temporaries before accumulating results
        del bp, indicators

        # Fill compact accumulator in-place (no pd.Series objects created)
        perm_cohort_rets[:, bar_offset:bar_offset + n_bars] = port_rets_2d

        # Per-submarket scores (floats only — no Series stored)
        for pi in range(total_perms):
            finite_mask = np.isfinite(port_rets_2d[pi])
            rets_arr = port_rets_2d[pi][finite_mask]
            mpf = scoring_fn(pd.Series(rets_arr, dtype=np.float64))
            if np.isfinite(mpf):
                for col in _submarket_keys(slug):
                    perm_mkt_pfs[slug][col].append(float(mpf))

        del port_rets_2d
        bar_offset += n_bars

        if verbose:
            elapsed = time.monotonic() - t0
            rss = _rss_mb()
            log_fn(
                f"[{label}] event {slug}: batch done "
                f"({elapsed:.1f}s elapsed, RSS={rss:.0f}MB)"
            )

        gc.collect()

    # --- Assemble cohort-level scores ---------------------------------------
    permuted_pfs: list[float] = []
    perm_better_count = 1

    for pi in range(total_perms):
        row = perm_cohort_rets[pi]
        finite_mask = np.isfinite(row)
        cohort = pd.Series(row[finite_mask].astype(np.float64))
        pf = scoring_fn(cohort)
        if np.isfinite(pf):
            permuted_pfs.append(float(pf))
        if pf >= real_pf:
            perm_better_count += 1

        if verbose and (pi + 1 == 1 or (pi + 1) % 10 == 0 or pi + 1 == total_perms):
            running_p = perm_better_count / (pi + 2)  # +2: 1-indexed + real
            log_fn(f"[{label}] scored {pi + 1}/{total_perms}: p={running_p:.4f}")

    # Free the large accumulator now that scoring is done
    del perm_cohort_rets
    gc.collect()

    p_value = float(perm_better_count / n_permutations)

    # --- Assemble per-submarket results -------------------------------------
    per_submarket: dict[str, dict[str, SubmarketMCPT]] = {}
    for slug in event_order:
        per_submarket[slug] = {}
        for col in _submarket_keys(slug):
            real_mpf = real_mkt_pf[slug][col]
            perm_list = perm_mkt_pfs[slug][col]
            sm_better = 1 + sum(1 for x in perm_list if x >= real_mpf)
            sm_p = float(sm_better / n_permutations)
            per_submarket[slug][col] = SubmarketMCPT(
                real_profit_factor=real_mpf,
                p_value=sm_p,
                permuted_pfs=perm_list,
                real_returns=real_mkt_rets[slug][col],
                permuted_returns=[],  # omitted in batch mode to save memory
            )

    if verbose:
        elapsed = time.monotonic() - t0
        log_fn(f"[{label}] batch MCPT done: score={real_pf:.6f}, p={p_value:.6f} ({elapsed:.1f}s)")

    return MCPTResult(
        real_pf=real_pf,
        p_value=p_value,
        permuted_pfs=permuted_pfs,
        params=params,
        per_event_pf=real_event_pf,
        per_event_returns=real_event_rets,
        per_submarket=per_submarket,
    )
