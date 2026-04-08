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

import logging
import multiprocessing as mp

import numpy as np
import pandas as pd

from .bar_permute import get_permutation, permute_event_bars

logger = logging.getLogger(__name__)


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


class ScoringFn:
    """Picklable scoring callable wrapping :func:`compute_metrics`.

    Higher returned values always mean *better*.  For metrics where
    lower-is-better (``max_drawdown``, ``volatility``) the sign is negated.
    """

    _LOWER_IS_BETTER = frozenset({"max_drawdown", "volatility"})

    def __init__(self, metric: str, annualization_factor: int = 252) -> None:
        self.metric = metric
        self.annualization_factor = annualization_factor

    def __call__(self, returns: pd.Series) -> float:
        if len(returns) == 0:
            return float("nan")
        from ..report.metrics import compute_metrics

        m = compute_metrics(returns, annualization_factor=self.annualization_factor)
        val = float(m.get(self.metric, 0.0))
        if self.metric in self._LOWER_IS_BETTER:
            val = -val
        return val


def make_scoring_fn(
    metric: str,
    annualization_factor: int = 252,
) -> ScoringFn:
    """Build a picklable scoring function from a :func:`compute_metrics` key."""
    return ScoringFn(metric, annualization_factor)


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


@runtime_checkable
class EventLevelMCPTStrategy(Protocol):
    """Protocol for event-level strategies (e.g. multi-column Backtester-based).

    Unlike :class:`MCPTStrategy` which operates on individual submarket OHLC
    DataFrames, this protocol operates on whole events — each event is a dict
    of submarket DataFrames that may contain arbitrary columns (OHLC + volume,
    vwap, buy/sell volume, etc.).

    ``apply_event`` returns per-submarket *return series* directly (not
    signals), because the strategy internally computes portfolio returns via
    a Backtester or equivalent.
    """

    def optimize(
        self,
        events: dict[str, dict[str, pd.DataFrame]],
        event_order: list[str],
    ) -> Any:
        """Derive strategy parameters from in-sample events.

        Parameters
        ----------
        events:
            ``{event_slug: {submarket_col: df, ...}, ...}``
        event_order:
            Slugs in evaluation order.

        Returns
        -------
        Opaque params object passed to :meth:`apply_event`.
        """
        ...

    def apply_event(
        self,
        event_data: dict[str, pd.DataFrame],
        params: Any,
    ) -> dict[str, pd.Series]:
        """Apply strategy to one event's submarkets and return per-submarket returns.

        Parameters
        ----------
        event_data:
            ``{submarket_col: df, ...}`` with all columns available.
        params:
            Opaque params from :meth:`optimize`.

        Returns
        -------
        ``{submarket_col: returns_series, ...}``
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


def _apply_event_strategy_to_events(
    strategy: EventLevelMCPTStrategy,
    events: dict[str, dict[str, pd.DataFrame]],
    event_order: list[str],
    params: Any,
    scoring_fn: Callable[[pd.Series], float],
) -> tuple[dict[str, pd.Series], dict[str, float], dict[str, dict[str, pd.Series]], dict[str, dict[str, float]]]:
    """Apply an event-level strategy to every event, return aggregated + per-market results."""
    event_returns: dict[str, pd.Series] = {}
    event_pf: dict[str, float] = {}
    market_returns: dict[str, dict[str, pd.Series]] = {}
    market_pf: dict[str, dict[str, float]] = {}

    for slug in event_order:
        rets_by_market = strategy.apply_event(events[slug], params)
        pf_by_market: dict[str, float] = {}
        for col, mr in rets_by_market.items():
            pf_by_market[col] = scoring_fn(mr)
        event_returns[slug] = aggregate_market_returns(rets_by_market)
        event_pf[slug] = scoring_fn(event_returns[slug])
        market_returns[slug] = rets_by_market
        market_pf[slug] = pf_by_market

    return event_returns, event_pf, market_returns, market_pf


# ---------------------------------------------------------------------------
# Parallel permutation helpers
# ---------------------------------------------------------------------------

_PERM_CTX: dict[str, Any] = {}


def _perm_worker_init(
    strategy: Any,
    events: dict[str, dict[str, pd.DataFrame]],
    event_order: list[str],
    params: Any,
    scoring_fn: Callable[[pd.Series], float],
    event_level: bool,
) -> None:
    """Initializer for permutation worker processes (fork context)."""
    _PERM_CTX["strategy"] = strategy
    _PERM_CTX["events"] = events
    _PERM_CTX["event_order"] = event_order
    _PERM_CTX["params"] = params
    _PERM_CTX["scoring_fn"] = scoring_fn
    _PERM_CTX["event_level"] = event_level


def _run_single_permutation(
    perm_i: int,
) -> tuple[int, float, dict[str, dict[str, tuple[pd.Series, "float | None"]]]]:
    """Execute one bar permutation.

    Returns ``(perm_i, cohort_pf, per_market_results)`` where
    *per_market_results* maps ``slug -> col -> (returns, score_or_None)``.
    """
    strategy = _PERM_CTX["strategy"]
    events = _PERM_CTX["events"]
    event_order = _PERM_CTX["event_order"]
    params = _PERM_CTX["params"]
    scoring_fn = _PERM_CTX["scoring_fn"]
    event_level = _PERM_CTX["event_level"]

    perm_event_rets: dict[str, pd.Series] = {}
    mkt_results: dict[str, dict[str, tuple[pd.Series, float | None]]] = {}

    if event_level:
        for slug in event_order:
            perm_data = permute_event_bars(events[slug], seed=perm_i)
            rets_by_market = strategy.apply_event(perm_data, params)
            slug_res: dict[str, tuple[pd.Series, float | None]] = {}
            for col, mr in rets_by_market.items():
                mpf = scoring_fn(mr)
                slug_res[col] = (mr, float(mpf) if np.isfinite(mpf) else None)
            mkt_results[slug] = slug_res
            perm_event_rets[slug] = aggregate_market_returns(rets_by_market)
    else:
        for slug in event_order:
            event_markets = events[slug]
            rets_by_market_sm: dict[str, pd.Series] = {}
            slug_res_sm: dict[str, tuple[pd.Series, float | None]] = {}
            for col, df in event_markets.items():
                perm_df = get_permutation(df)
                signal = strategy.apply(perm_df, params)
                r = next_log_returns(perm_df["close"])
                mr = (signal * r).dropna()
                rets_by_market_sm[col] = mr
                mpf = scoring_fn(mr)
                slug_res_sm[col] = (mr, float(mpf) if np.isfinite(mpf) else None)
            mkt_results[slug] = slug_res_sm
            perm_event_rets[slug] = aggregate_market_returns(rets_by_market_sm)

    perm_pf = scoring_fn(concat_returns_in_order(perm_event_rets, event_order))
    return perm_i, float(perm_pf), mkt_results


def _collect_perm_result(
    perm_pf: float,
    mkt_results: dict[str, dict[str, tuple[pd.Series, "float | None"]]],
    real_pf: float,
    permuted_pfs: list[float],
    perm_mkt_pfs: dict[str, dict[str, list[float]]],
    perm_mkt_rets: dict[str, dict[str, list[pd.Series]]],
    event_order: list[str],
) -> bool:
    """Accumulate one permutation result. Returns True if perm >= real."""
    if np.isfinite(perm_pf):
        permuted_pfs.append(perm_pf)
    for slug in event_order:
        for col, (mr, mpf) in mkt_results[slug].items():
            perm_mkt_rets[slug][col].append(mr)
            if mpf is not None:
                perm_mkt_pfs[slug][col].append(mpf)
    return bool(perm_pf >= real_pf)


# ---------------------------------------------------------------------------
# MCPT runners
# ---------------------------------------------------------------------------

def run_insample_mcpt(
    strategy: "MCPTStrategy | EventLevelMCPTStrategy",
    events: dict[str, dict[str, pd.DataFrame]],
    event_order: list[str],
    n_permutations: int = 1000,
    *,
    scoring_fn: Callable[[pd.Series], float] = profit_factor,
    workers: int = 1,
) -> MCPTResult:
    """Run in-sample MCPT.

    1. Call ``strategy.optimize`` on *events* to derive parameters.
    2. Compute the real strategy returns under those parameters.
    3. Permute bars ``n_permutations - 1`` times, recompute strategy returns
       each time (with the **same** optimised params), and estimate p-value.

    Supports both :class:`MCPTStrategy` (per-submarket signal) and
    :class:`EventLevelMCPTStrategy` (whole-event Backtester-based) strategies.

    Parameters
    ----------
    strategy:
        Any object satisfying :class:`MCPTStrategy` or :class:`EventLevelMCPTStrategy`.
    events:
        ``{event_slug: {submarket_col: df, ...}, ...}``
    event_order:
        Slug ordering for concatenation (must match keys in *events*).
    n_permutations:
        Total permutation count (real + synthetic).
    scoring_fn:
        Metric applied to a return series.  Defaults to :func:`profit_factor`.
    workers:
        Number of parallel worker processes for the permutation loop.
        ``1`` (default) runs sequentially.
    """
    event_level = isinstance(strategy, EventLevelMCPTStrategy)

    params = strategy.optimize(events, event_order)

    # --- Real returns -------------------------------------------------------
    if event_level:
        real_event_rets, real_event_pf, real_mkt_rets, real_mkt_pf = _apply_event_strategy_to_events(
            strategy, events, event_order, params, scoring_fn,
        )
    else:
        real_event_rets, real_event_pf, real_mkt_rets, real_mkt_pf = _apply_strategy_to_events(
            strategy, events, event_order, params, scoring_fn,  # type: ignore[arg-type]
        )
    real_cohort_rets = concat_returns_in_order(real_event_rets, event_order)
    real_pf = scoring_fn(real_cohort_rets)

    logger.info(
        "[insample] real score=%.6f, starting %d permutations%s",
        real_pf,
        n_permutations - 1,
        f" ({workers} workers)" if workers > 1 else "",
    )

    # --- Permutation loop ---------------------------------------------------
    perm_better_count = 1
    permuted_pfs: list[float] = []

    # Per-submarket accumulators — for event-level strategies the submarket
    # keys come from apply_event() output (may differ from event data keys).
    def _submarket_keys(slug: str) -> list[str]:
        if event_level:
            return list(real_mkt_rets[slug].keys())
        return list(events[slug].keys())

    perm_mkt_pfs: dict[str, dict[str, list[float]]] = {
        slug: {col: [] for col in _submarket_keys(slug)} for slug in event_order
    }
    perm_mkt_rets: dict[str, dict[str, list[pd.Series]]] = {
        slug: {col: [] for col in _submarket_keys(slug)} for slug in event_order
    }

    total_perms = n_permutations - 1

    if workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=workers,
            initializer=_perm_worker_init,
            initargs=(strategy, events, event_order, params, scoring_fn, event_level),
        ) as pool:
            completed = 0
            for pi, ppf, mres in pool.imap_unordered(
                _run_single_permutation, range(1, n_permutations)
            ):
                if _collect_perm_result(
                    ppf, mres, real_pf, permuted_pfs, perm_mkt_pfs, perm_mkt_rets, event_order,
                ):
                    perm_better_count += 1
                completed += 1
                if completed == 1 or completed % 10 == 0 or completed == total_perms:
                    running_p = perm_better_count / (completed + 1)
                    logger.debug("[insample] perm %d/%d: p=%.4f", completed, total_perms, running_p)
    else:
        for perm_i in range(1, n_permutations):
            perm_event_rets: dict[str, pd.Series] = {}
            if event_level:
                for slug in event_order:
                    perm_data = permute_event_bars(events[slug], seed=perm_i)
                    rets_by_market = strategy.apply_event(perm_data, params)  # type: ignore[union-attr]
                    for col, mr in rets_by_market.items():
                        perm_mkt_rets[slug][col].append(mr)
                        mpf = scoring_fn(mr)
                        if np.isfinite(mpf):
                            perm_mkt_pfs[slug][col].append(float(mpf))
                    perm_event_rets[slug] = aggregate_market_returns(rets_by_market)
            else:
                for slug in event_order:
                    event_markets = events[slug]
                    rets_by_market_sm: dict[str, pd.Series] = {}
                    for col, df in event_markets.items():
                        perm_df = get_permutation(df)
                        signal = strategy.apply(perm_df, params)  # type: ignore[union-attr]
                        r = next_log_returns(perm_df["close"])
                        mr = (signal * r).dropna()
                        rets_by_market_sm[col] = mr
                        perm_mkt_rets[slug][col].append(mr)
                        mpf = scoring_fn(mr)
                        if np.isfinite(mpf):
                            perm_mkt_pfs[slug][col].append(float(mpf))
                    perm_event_rets[slug] = aggregate_market_returns(rets_by_market_sm)

            perm_pf = scoring_fn(concat_returns_in_order(perm_event_rets, event_order))
            if np.isfinite(perm_pf):
                permuted_pfs.append(float(perm_pf))
            if perm_pf >= real_pf:
                perm_better_count += 1
            if perm_i == 1 or perm_i % 10 == 0 or perm_i == total_perms:
                running_p = perm_better_count / (perm_i + 1)
                logger.debug("[insample] perm %d/%d: p=%.4f", perm_i, total_perms, running_p)

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
                permuted_returns=perm_mkt_rets[slug][col],
            )

    logger.info("insample cohort score=%.6f, p-value=%.6f", real_pf, p_value)

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
    strategy: "MCPTStrategy | EventLevelMCPTStrategy",
    events: dict[str, dict[str, pd.DataFrame]],
    event_order: list[str],
    params: Any,
    n_permutations: int = 1000,
    *,
    scoring_fn: Callable[[pd.Series], float] = profit_factor,
    workers: int = 1,
) -> MCPTResult:
    """Run out-of-sample MCPT with pre-trained parameters.

    Identical to :func:`run_insample_mcpt` except that ``strategy.optimize``
    is **not** called — the caller supplies *params* directly (typically
    from a preceding insample run).

    Supports both :class:`MCPTStrategy` and :class:`EventLevelMCPTStrategy`.
    """
    event_level = isinstance(strategy, EventLevelMCPTStrategy)

    # --- Real returns -------------------------------------------------------
    if event_level:
        real_event_rets, real_event_pf, real_mkt_rets, real_mkt_pf = _apply_event_strategy_to_events(
            strategy, events, event_order, params, scoring_fn,
        )
    else:
        real_event_rets, real_event_pf, real_mkt_rets, real_mkt_pf = _apply_strategy_to_events(
            strategy, events, event_order, params, scoring_fn,  # type: ignore[arg-type]
        )
    real_cohort_rets = concat_returns_in_order(real_event_rets, event_order)
    real_pf = scoring_fn(real_cohort_rets)

    logger.info(
        "[outsample] real score=%.6f, starting %d permutations%s",
        real_pf,
        n_permutations - 1,
        f" ({workers} workers)" if workers > 1 else "",
    )

    # --- Permutation loop ---------------------------------------------------
    perm_better_count = 1
    permuted_pfs: list[float] = []

    def _submarket_keys(slug: str) -> list[str]:
        if event_level:
            return list(real_mkt_rets[slug].keys())
        return list(events[slug].keys())

    perm_mkt_pfs: dict[str, dict[str, list[float]]] = {
        slug: {col: [] for col in _submarket_keys(slug)} for slug in event_order
    }
    perm_mkt_rets: dict[str, dict[str, list[pd.Series]]] = {
        slug: {col: [] for col in _submarket_keys(slug)} for slug in event_order
    }

    total_perms = n_permutations - 1

    if workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=workers,
            initializer=_perm_worker_init,
            initargs=(strategy, events, event_order, params, scoring_fn, event_level),
        ) as pool:
            completed = 0
            for pi, ppf, mres in pool.imap_unordered(
                _run_single_permutation, range(1, n_permutations)
            ):
                if _collect_perm_result(
                    ppf, mres, real_pf, permuted_pfs, perm_mkt_pfs, perm_mkt_rets, event_order,
                ):
                    perm_better_count += 1
                completed += 1
                if completed == 1 or completed % 10 == 0 or completed == total_perms:
                    running_p = perm_better_count / (completed + 1)
                    logger.debug("[outsample] perm %d/%d: p=%.4f", completed, total_perms, running_p)
    else:
        for perm_i in range(1, n_permutations):
            perm_event_rets: dict[str, pd.Series] = {}
            if event_level:
                for slug in event_order:
                    perm_data = permute_event_bars(events[slug], seed=perm_i)
                    rets_by_market = strategy.apply_event(perm_data, params)  # type: ignore[union-attr]
                    for col, mr in rets_by_market.items():
                        perm_mkt_rets[slug][col].append(mr)
                        mpf = scoring_fn(mr)
                        if np.isfinite(mpf):
                            perm_mkt_pfs[slug][col].append(float(mpf))
                    perm_event_rets[slug] = aggregate_market_returns(rets_by_market)
            else:
                for slug in event_order:
                    event_markets = events[slug]
                    rets_by_market_sm: dict[str, pd.Series] = {}
                    for col, df in event_markets.items():
                        perm_df = get_permutation(df)
                        signal = strategy.apply(perm_df, params)  # type: ignore[union-attr]
                        r = next_log_returns(perm_df["close"])
                        mr = (signal * r).dropna()
                        rets_by_market_sm[col] = mr
                        perm_mkt_rets[slug][col].append(mr)
                        mpf = scoring_fn(mr)
                        if np.isfinite(mpf):
                            perm_mkt_pfs[slug][col].append(float(mpf))
                    perm_event_rets[slug] = aggregate_market_returns(rets_by_market_sm)

            perm_pf = scoring_fn(concat_returns_in_order(perm_event_rets, event_order))
            if np.isfinite(perm_pf):
                permuted_pfs.append(float(perm_pf))
            if perm_pf >= real_pf:
                perm_better_count += 1
            if perm_i == 1 or perm_i % 10 == 0 or perm_i == total_perms:
                running_p = perm_better_count / (perm_i + 1)
                logger.debug("[outsample] perm %d/%d: p=%.4f", perm_i, total_perms, running_p)

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
                permuted_returns=perm_mkt_rets[slug][col],
            )

    logger.info("outsample cohort score=%.6f, p-value=%.6f", real_pf, p_value)

    return MCPTResult(
        real_pf=real_pf,
        p_value=p_value,
        permuted_pfs=permuted_pfs,
        params=params,
        per_event_pf=real_event_pf,
        per_event_returns=real_event_rets,
        per_submarket=per_submarket,
    )
