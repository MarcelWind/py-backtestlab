"""Strategy-agnostic Monte Carlo Permutation Test (MCPT) engine.

Provides a pluggable framework for running insample and outsample permutation tests on
any event-submarket strategy that conforms to :class:`MCPTStrategy`.

Public API
----------
- :class:`MCPTStrategy` — protocol that any strategy must satisfy.
- :class:`MCPTResult` — dataclass returned by each MCPT run.
- :class:`SubmarketMCPT` — per-submarket breakdown.
- :func:`run_insample_mcpt` — in-sample MCPT with strategy-driven optimisation.
- :func:`run_oos_mcpt` — out-of-sample MCPT with pre-trained params.
- :func:`next_log_returns` — bar-to-bar log returns shifted for signal alignment.
- :func:`profit_factor` — lightweight PF suitable for hot loops.
- :func:`aggregate_market_returns` — average across submarkets per bar.
- :func:`concat_returns_in_order` — concatenate event return streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .bar_permute import get_permutation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def next_log_returns(close: pd.Series) -> pd.Series:
    """Compute next-bar log returns: ``log(close).diff().shift(-1)``."""
    arr = np.log(close.to_numpy(dtype=float))
    return pd.Series(arr, index=close.index).diff().shift(-1)


def profit_factor(returns: pd.Series) -> float:
    """Ratio of total positive returns to total negative returns.

    Returns ``inf`` when all returns are non-negative and positive sum exists,
    ``nan`` when there are no returns.
    """
    positive = float(returns[returns > 0].sum())
    negative = float(returns[returns < 0].abs().sum())
    if negative <= 0.0:
        return float("inf") if positive > 0.0 else float("nan")
    return positive / negative


def aggregate_market_returns(returns_by_market: dict[str, pd.Series]) -> pd.Series:
    """Average per-bar returns across submarkets (equal-weight)."""
    if not returns_by_market:
        return pd.Series(dtype=float)
    stacked = pd.concat(returns_by_market, axis=1, sort=False)
    return stacked.mean(axis=1, skipna=True).dropna()


def concat_returns_in_order(
    returns_by_event: dict[str, pd.Series],
    event_order: list[str],
) -> pd.Series:
    """Concatenate event return streams in a deterministic order.

    Timestamps are dropped (integer-indexed) so that non-overlapping event
    windows can be joined without datetime conflicts.
    """
    chunks: list[pd.Series] = []
    for slug in event_order:
        series = returns_by_event.get(slug)
        if series is None or len(series) == 0:
            continue
        chunks.append(pd.Series(series.to_numpy(dtype=float)))
    if not chunks:
        return pd.Series(dtype=float)
    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# Protocol & result dataclasses
# ---------------------------------------------------------------------------

@runtime_checkable
class MCPTStrategy(Protocol):
    """Structural protocol for strategies usable with the MCPT engine.

    *Any* class exposing ``optimize`` and ``apply`` with compatible signatures
    satisfies this protocol — no inheritance required.
    """

    def optimize(
        self,
        events: dict[str, dict[str, pd.DataFrame]],
        event_order: list[str],
    ) -> Any:
        """Derive strategy parameters from a cohort of in-sample events.

        Parameters
        ----------
        events:
            ``{event_slug: {submarket_col: ohlc_df, ...}, ...}``
        event_order:
            Slugs in the order they should be evaluated.

        Returns
        -------
        Opaque params object passed to :meth:`apply`.
        """
        ...

    def apply(self, ohlc: pd.DataFrame, params: Any) -> pd.Series:
        """Return a signal series for one submarket given pre-trained params.

        The signal is multiplied element-wise with next-bar log returns to
        produce strategy returns.
        """
        ...


@dataclass
class SubmarketMCPT:
    """Per-submarket MCPT breakdown."""

    real_profit_factor: float
    p_value: float
    permuted_pfs: list[float]
    real_returns: pd.Series
    permuted_returns: list[pd.Series]


@dataclass
class MCPTResult:
    """Aggregated result of a single MCPT run (insample or outsample)."""

    # Cohort-level
    real_pf: float
    p_value: float
    permuted_pfs: list[float]
    params: Any = None

    # Per-event
    per_event_pf: dict[str, float] = field(default_factory=dict)
    per_event_returns: dict[str, pd.Series] = field(default_factory=dict)

    # Per-submarket (nested: event_slug → submarket_col → SubmarketMCPT)
    per_submarket: dict[str, dict[str, SubmarketMCPT]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MCPT runners
# ---------------------------------------------------------------------------

def _apply_strategy_to_events(
    strategy: MCPTStrategy,
    events: dict[str, dict[str, pd.DataFrame]],
    event_order: list[str],
    params: Any,
    scoring_fn: Callable[[pd.Series], float],
) -> tuple[dict[str, pd.Series], dict[str, float], dict[str, dict[str, pd.Series]], dict[str, dict[str, float]]]:
    """Apply strategy to every submarket, return aggregated + per-market results."""
    event_returns: dict[str, pd.Series] = {}
    event_pf: dict[str, float] = {}
    market_returns: dict[str, dict[str, pd.Series]] = {}
    market_pf: dict[str, dict[str, float]] = {}

    for slug in event_order:
        event_markets = events[slug]
        rets_by_market: dict[str, pd.Series] = {}
        pf_by_market: dict[str, float] = {}
        for col, df in event_markets.items():
            signal = strategy.apply(df, params)
            r = next_log_returns(df["close"])
            mr = (signal * r).dropna()
            rets_by_market[col] = mr
            pf_by_market[col] = scoring_fn(mr)
        event_returns[slug] = aggregate_market_returns(rets_by_market)
        event_pf[slug] = scoring_fn(event_returns[slug])
        market_returns[slug] = rets_by_market
        market_pf[slug] = pf_by_market

    return event_returns, event_pf, market_returns, market_pf


def run_insample_mcpt(
    strategy: MCPTStrategy,
    events: dict[str, dict[str, pd.DataFrame]],
    event_order: list[str],
    n_permutations: int = 1000,
    *,
    scoring_fn: Callable[[pd.Series], float] = profit_factor,
    verbose: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> MCPTResult:
    """Run in-sample MCPT.

    1. Call ``strategy.optimize`` on *events* to derive parameters.
    2. Compute the real strategy returns under those parameters.
    3. Permute bars ``n_permutations - 1`` times, recompute strategy returns
       each time (with the **same** optimised params), and estimate p-value.

    Parameters
    ----------
    strategy:
        Any object satisfying :class:`MCPTStrategy`.
    events:
        ``{event_slug: {submarket_col: ohlc_df, ...}, ...}``
    event_order:
        Slug ordering for concatenation (must match keys in *events*).
    n_permutations:
        Total permutation count (real + synthetic).
    scoring_fn:
        Metric applied to a return series.  Defaults to :func:`profit_factor`.
    verbose:
        Emit progress messages via *log_fn*.
    log_fn:
        Callback for verbose output.  Defaults to ``print``.
    """
    if log_fn is None:
        log_fn = print

    params = strategy.optimize(events, event_order)

    # --- Real returns -------------------------------------------------------
    real_event_rets, real_event_pf, real_mkt_rets, real_mkt_pf = _apply_strategy_to_events(
        strategy, events, event_order, params, scoring_fn,
    )
    real_cohort_rets = concat_returns_in_order(real_event_rets, event_order)
    real_pf = scoring_fn(real_cohort_rets)

    # --- Permutation loop ---------------------------------------------------
    perm_better_count = 1
    permuted_pfs: list[float] = []

    # Per-submarket accumulators
    perm_mkt_pfs: dict[str, dict[str, list[float]]] = {
        slug: {col: [] for col in events[slug]} for slug in event_order
    }
    perm_mkt_rets: dict[str, dict[str, list[pd.Series]]] = {
        slug: {col: [] for col in events[slug]} for slug in event_order
    }

    for perm_i in range(1, n_permutations):
        perm_event_rets: dict[str, pd.Series] = {}
        for slug in event_order:
            event_markets = events[slug]
            rets_by_market: dict[str, pd.Series] = {}
            for col, df in event_markets.items():
                perm_df = get_permutation(df)
                signal = strategy.apply(perm_df, params)
                r = next_log_returns(perm_df["close"])
                mr = (signal * r).dropna()
                rets_by_market[col] = mr
                perm_mkt_rets[slug][col].append(mr)
                mpf = scoring_fn(mr)
                if np.isfinite(mpf):
                    perm_mkt_pfs[slug][col].append(float(mpf))
            perm_event_rets[slug] = aggregate_market_returns(rets_by_market)

        perm_pf = scoring_fn(concat_returns_in_order(perm_event_rets, event_order))
        if np.isfinite(perm_pf):
            permuted_pfs.append(float(perm_pf))
        if perm_pf >= real_pf:
            perm_better_count += 1
        if verbose and (perm_i % 50 == 0 or perm_i == n_permutations - 1):
            log_fn(f"insample perm {perm_i}/{n_permutations - 1}: cohort_score={perm_pf:.6f}")

    p_value = float(perm_better_count / n_permutations)

    # --- Assemble per-submarket results -------------------------------------
    per_submarket: dict[str, dict[str, SubmarketMCPT]] = {}
    for slug in event_order:
        per_submarket[slug] = {}
        for col in events[slug]:
            real_mpf = real_mkt_pf[slug][col]
            perm_list = perm_mkt_pfs[slug][col]
            sm_better = 1 + sum(1 for x in perm_list if x >= real_mpf)
            sm_p = float(sm_better / n_permutations)
            per_submarket[slug][col] = SubmarketMCPT(
                real_profit_factor=real_mpf,
                p_value=sm_p,
                permuted_pfs=perm_list,
                real_returns=real_mkt_rets[slug][col],
                permuted_returns=perm_mkt_rets[slug][col],
            )

    if verbose:
        log_fn(f"insample cohort score={real_pf:.6f}, p-value={p_value:.6f}")

    return MCPTResult(
        real_pf=real_pf,
        p_value=p_value,
        permuted_pfs=permuted_pfs,
        params=params,
        per_event_pf=real_event_pf,
        per_event_returns=real_event_rets,
        per_submarket=per_submarket,
    )


def run_oos_mcpt(
    strategy: MCPTStrategy,
    events: dict[str, dict[str, pd.DataFrame]],
    event_order: list[str],
    params: Any,
    n_permutations: int = 1000,
    *,
    scoring_fn: Callable[[pd.Series], float] = profit_factor,
    verbose: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> MCPTResult:
    """Run out-of-sample MCPT with pre-trained parameters.

    Identical to :func:`run_insample_mcpt` except that ``strategy.optimize``
    is **not** called — the caller supplies *params* directly (typically
    from a preceding insample run).
    """
    if log_fn is None:
        log_fn = print

    # --- Real returns -------------------------------------------------------
    real_event_rets, real_event_pf, real_mkt_rets, real_mkt_pf = _apply_strategy_to_events(
        strategy, events, event_order, params, scoring_fn,
    )
    real_cohort_rets = concat_returns_in_order(real_event_rets, event_order)
    real_pf = scoring_fn(real_cohort_rets)

    # --- Permutation loop ---------------------------------------------------
    perm_better_count = 1
    permuted_pfs: list[float] = []

    perm_mkt_pfs: dict[str, dict[str, list[float]]] = {
        slug: {col: [] for col in events[slug]} for slug in event_order
    }
    perm_mkt_rets: dict[str, dict[str, list[pd.Series]]] = {
        slug: {col: [] for col in events[slug]} for slug in event_order
    }

    for perm_i in range(1, n_permutations):
        perm_event_rets: dict[str, pd.Series] = {}
        for slug in event_order:
            event_markets = events[slug]
            rets_by_market: dict[str, pd.Series] = {}
            for col, df in event_markets.items():
                perm_df = get_permutation(df)
                signal = strategy.apply(perm_df, params)
                r = next_log_returns(perm_df["close"])
                mr = (signal * r).dropna()
                rets_by_market[col] = mr
                perm_mkt_rets[slug][col].append(mr)
                mpf = scoring_fn(mr)
                if np.isfinite(mpf):
                    perm_mkt_pfs[slug][col].append(float(mpf))
            perm_event_rets[slug] = aggregate_market_returns(rets_by_market)

        perm_pf = scoring_fn(concat_returns_in_order(perm_event_rets, event_order))
        if np.isfinite(perm_pf):
            permuted_pfs.append(float(perm_pf))
        if perm_pf >= real_pf:
            perm_better_count += 1
        if verbose and (perm_i % 50 == 0 or perm_i == n_permutations - 1):
            log_fn(f"outsample perm {perm_i}/{n_permutations - 1}: cohort_score={perm_pf:.6f}")

    p_value = float(perm_better_count / n_permutations)

    # --- Assemble per-submarket results -------------------------------------
    per_submarket: dict[str, dict[str, SubmarketMCPT]] = {}
    for slug in event_order:
        per_submarket[slug] = {}
        for col in events[slug]:
            real_mpf = real_mkt_pf[slug][col]
            perm_list = perm_mkt_pfs[slug][col]
            sm_better = 1 + sum(1 for x in perm_list if x >= real_mpf)
            sm_p = float(sm_better / n_permutations)
            per_submarket[slug][col] = SubmarketMCPT(
                real_profit_factor=real_mpf,
                p_value=sm_p,
                permuted_pfs=perm_list,
                real_returns=real_mkt_rets[slug][col],
                permuted_returns=perm_mkt_rets[slug][col],
            )

    if verbose:
        log_fn(f"outsample cohort score={real_pf:.6f}, p-value={p_value:.6f}")

    return MCPTResult(
        real_pf=real_pf,
        p_value=p_value,
        permuted_pfs=permuted_pfs,
        params=params,
        per_event_pf=real_event_pf,
        per_event_returns=real_event_rets,
        per_submarket=per_submarket,
    )
