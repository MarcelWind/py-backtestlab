"""Adapter bridging WeatherMarketImbalanceStrategy to the MCPT engine.

Provides :class:`WeatherMarketMCPTAdapter` which satisfies the
:class:`~stratlab.validation.mcpt.EventLevelMCPTStrategy` protocol, and
helper functions for converting between the production ``EventBundle`` format
and the MCPT ``{slug: {submarket: DataFrame}}`` event layout.
"""

from __future__ import annotations

import gc
import logging
import re
import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratlab.backtest.backtester import Backtester, compute_returns
from stratlab.optimize.rule_search import load_param_rules, params_key, sample_trial_params
from stratlab.strategy.indicators import select_yes_no_columns
from stratlab.validation.mcpt import aggregate_market_returns, concat_returns_in_order
from strategies.weather_market_strategy import WeatherMarketImbalanceStrategy

_VALID_OPTIMIZERS = ("random", "bayesian")

logger = logging.getLogger(__name__)


# Re-use the suffix regex from indicators for column matching.
_SUFFIX_RE = re.compile(r"^(.*)__(yes|no)$", re.IGNORECASE)


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
    indicator_snapshots: "dict[str, dict] | None"


# ---------------------------------------------------------------------------
# Data conversion helpers
# ---------------------------------------------------------------------------

def bundles_to_mcpt_events(
    bundle_map: dict[str, EventBundle],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Convert ``{slug: EventBundle}`` to MCPT event format.

    Each submarket becomes a DataFrame with columns:
    ``close, open, high, low, vwap, volume`` and optionally
    ``buy_volume__yes, buy_volume__no, sell_volume__yes, sell_volume__no``.

    Returns ``{slug: {submarket_col: enriched_df, ...}, ...}``.
    """
    mcpt_events: dict[str, dict[str, pd.DataFrame]] = {}

    for slug, bundle in bundle_map.items():
        prices = bundle["prices"]
        open_ = bundle.get("open_")
        high = bundle.get("high")
        low = bundle.get("low")
        vwap = bundle.get("vwap")
        volume = bundle.get("volume")
        buy_vol = bundle.get("buy_volume")
        sell_vol = bundle.get("sell_volume")

        event_markets: dict[str, pd.DataFrame] = {}
        for col in prices.columns:
            data: dict[str, pd.Series] = {"close": prices[col]}
            if open_ is not None and col in open_.columns:
                data["open"] = open_[col]
            if high is not None and col in high.columns:
                data["high"] = high[col]
            if low is not None and col in low.columns:
                data["low"] = low[col]
            if vwap is not None and col in vwap.columns:
                data["vwap"] = vwap[col]
            if volume is not None and col in volume.columns:
                data["volume"] = volume[col]

            # Resolve paired yes/no volume columns.
            m = _SUFFIX_RE.match(str(col))
            base = m.group(1) if m else str(col)
            if buy_vol is not None and not buy_vol.empty:
                no_col, yes_col = select_yes_no_columns(buy_vol, base, str(col))
                if yes_col is not None:
                    data["buy_volume__yes"] = buy_vol[yes_col]
                if no_col is not None:
                    data["buy_volume__no"] = buy_vol[no_col]
            if sell_vol is not None and not sell_vol.empty:
                no_col, yes_col = select_yes_no_columns(sell_vol, base, str(col))
                if yes_col is not None:
                    data["sell_volume__yes"] = sell_vol[yes_col]
                if no_col is not None:
                    data["sell_volume__no"] = sell_vol[no_col]

            event_markets[str(col)] = pd.DataFrame(data, index=prices.index)

        mcpt_events[slug] = event_markets

    return mcpt_events


def _reconstruct_matrices(
    event_data: dict[str, pd.DataFrame],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    """Reconstruct multi-column matrices from per-submarket DataFrames.

    Returns (prices, returns, vwap, volume, high, low, open_, buy_volume, sell_volume).
    """
    cols = list(event_data.keys())
    if not cols:
        empty = pd.DataFrame()
        return empty, empty, None, None, None, None, None, None, None

    first = event_data[cols[0]]
    idx = first.index

    prices = pd.DataFrame({c: event_data[c]["close"] for c in cols}, index=idx)
    returns = compute_returns(prices)

    def _maybe_frame(field: str) -> pd.DataFrame | None:
        data = {c: event_data[c][field] for c in cols if field in event_data[c].columns}
        return pd.DataFrame(data, index=idx) if data else None

    vwap = _maybe_frame("vwap")
    volume = _maybe_frame("volume")
    high = _maybe_frame("high")
    low = _maybe_frame("low")
    open_ = _maybe_frame("open")

    # Reconstruct buy/sell volume with ``col__yes`` / ``col__no`` column names.
    buy_data: dict[str, pd.Series] = {}
    sell_data: dict[str, pd.Series] = {}
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


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class WeatherMarketMCPTAdapter:
    """EventLevelMCPTStrategy adapter for WeatherMarketImbalanceStrategy.

    ``optimize`` performs random parameter search over in-sample events.
    ``apply_event`` runs a full Backtester pass on one event and returns
    the portfolio return series (single-key dict ``{"_portfolio_": ...}``).
    """

    def __init__(
        self,
        base_params: dict[str, object],
        rules: list[dict[str, Any]],
        rng: np.random.Generator,
        objective: str,
        n_trials: int,
        rebalance_freq: int = 1,
        scoring_fn: "Any | None" = None,
        optimizer: str = "random",
        cache_snapshots: bool = False,
    ) -> None:
        if optimizer not in _VALID_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {_VALID_OPTIMIZERS}, got {optimizer!r}")
        self.base_params = dict(base_params)
        self.rules = rules
        self.rng = rng
        self.objective = objective
        self.n_trials = n_trials
        self.rebalance_freq = rebalance_freq
        self.scoring_fn = scoring_fn
        self.optimizer = optimizer
        self.cache_snapshots = cache_snapshots

    # -- EventLevelMCPTStrategy interface -----------------------------------

    def optimize(
        self,
        events: dict[str, dict[str, pd.DataFrame]],
        event_order: list[str],
    ) -> dict[str, Any]:
        """Search for best parameters on in-sample events.

        Uses random search or Bayesian optimization (Optuna TPE) depending
        on ``self.optimizer``.
        """
        if self.optimizer == "bayesian":
            return self._optimize_bayesian(events, event_order)

        logger.info(
            "[optimize] Starting random search: %d trials, objective=%s",
            self.n_trials, self.objective,
        )

        seen: set[tuple[tuple[str, Any], ...]] = set()
        best_score = float("-inf")
        best_params: dict[str, Any] = {}
        objective_sign = _metric_direction(self.objective)
        max_retries = 30

        # Per-event indicator snapshots populated on the first trial's
        # backtests and reused for all subsequent trials so data-only
        # indicators (SdBands, Vwap, CumulativeYesNoDelta) skip the O(n)
        # catch-up loop.  Disabled by default to save memory on large
        # event sets (~200MB+ for 150+ events).
        event_snapshots: dict[str, dict[str, dict]] = {}
        use_snapshots = self.cache_snapshots

        for trial_idx in range(self.n_trials):
            # Sample unique params
            chosen: dict[str, Any] | None = None
            for _ in range(max_retries):
                sampled = sample_trial_params(self.rules, self.base_params, self.rng)
                pkey = params_key(sampled)
                if pkey not in seen:
                    seen.add(pkey)
                    chosen = sampled
                    break
            if chosen is None:
                logger.warning("[optimize] Exhausted unique samples at trial %d", trial_idx)
                break

            # Evaluate on all IS events — collect returns, then score the
            # concatenated cohort to match the MCPT engine's real_pf calc.
            all_event_rets: dict[str, pd.Series] = {}
            capture = use_snapshots and trial_idx == 0
            for slug in event_order:
                if capture:
                    rets, snaps = self._backtest_event_with_capture(
                        events[slug], chosen,
                    )
                    if snaps:
                        event_snapshots[slug] = snaps
                else:
                    rets = self._backtest_event(
                        events[slug], chosen,
                        indicator_snapshots=event_snapshots.get(slug) if use_snapshots else None,
                    )
                all_event_rets[slug] = rets

            cohort_rets = concat_returns_in_order(all_event_rets, event_order)
            if cohort_rets.empty:
                logger.debug(
                    "[optimize] trial %d/%d: no valid returns (%d events)",
                    trial_idx + 1, self.n_trials, len(event_order),
                )
                continue
            agg_score = self._score_returns(cohort_rets)
            if not np.isfinite(agg_score):
                continue

            is_new_best = agg_score > best_score
            if is_new_best:
                best_score = agg_score
                best_params = dict(chosen)

            marker = " ** new best" if is_new_best else ""
            logger.debug(
                "[optimize] trial %d/%d: %s=%.6f (%d events)%s",
                trial_idx + 1, self.n_trials,
                self.objective, agg_score / objective_sign,
                len(event_order), marker,
            )

            if (trial_idx + 1) % 10 == 0:
                gc.collect()

        logger.info("[optimize] Done. Best %s=%.6f", self.objective, best_score / objective_sign)

        return best_params

    def _optimize_bayesian(
        self,
        events: dict[str, dict[str, pd.DataFrame]],
        event_order: list[str],
    ) -> dict[str, Any]:
        """Bayesian (Optuna TPE) parameter search over in-sample events."""
        import optuna

        from stratlab.optimize.bayesian_search import create_study, rules_to_optuna_params

        logger.info(
            "[optimize] Starting Bayesian (TPE) search: %d trials, objective=%s",
            self.n_trials, self.objective,
        )
        if logger.isEnabledFor(logging.DEBUG):
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        objective_sign = _metric_direction(self.objective)
        event_snapshots: dict[str, dict[str, dict]] = {}
        use_snapshots = self.cache_snapshots
        first_trial_done = False

        def _objective(trial: optuna.Trial) -> float:
            nonlocal first_trial_done, event_snapshots
            chosen = rules_to_optuna_params(trial, self.rules, self.base_params)

            all_event_rets: dict[str, pd.Series] = {}
            capture = use_snapshots and not first_trial_done

            for slug in event_order:
                if capture:
                    rets, snaps = self._backtest_event_with_capture(
                        events[slug], chosen,
                    )
                    if snaps:
                        event_snapshots[slug] = snaps
                else:
                    rets = self._backtest_event(
                        events[slug], chosen,
                        indicator_snapshots=event_snapshots.get(slug) if use_snapshots else None,
                    )
                all_event_rets[slug] = rets

            first_trial_done = True

            cohort_rets = concat_returns_in_order(all_event_rets, event_order)
            if cohort_rets.empty:
                return float("-inf")

            agg_score = self._score_returns(cohort_rets)
            if not np.isfinite(agg_score):
                return float("-inf")

            logger.debug(
                "[optimize] trial %d/%d: %s=%.6f (%d events)",
                trial.number + 1, self.n_trials,
                self.objective, agg_score / objective_sign,
                len(event_order),
            )

            if (trial.number + 1) % 10 == 0:
                gc.collect()

            return agg_score

        # Derive int seed from the RNG for Optuna sampler reproducibility.
        optuna_seed = int(self.rng.integers(0, 2**31))
        study = create_study(optuna_seed, direction="maximize")
        study.optimize(_objective, n_trials=self.n_trials, show_progress_bar=False)

        # Extract best params from the study.  ``study.best_params`` only
        # contains the suggested values; fixed-value and disabled params
        # are not tracked by Optuna.  Re-run the conversion on a frozen
        # trial to get the full dict.
        best_trial = study.best_trial
        best_params = rules_to_optuna_params(best_trial, self.rules, self.base_params)
        best_value = study.best_value

        logger.info(
            "[optimize] Done. Best %s=%.6f (trial %d)",
            self.objective, best_value / objective_sign, best_trial.number + 1,
        )

        # Free intermediate state before proceeding to MCPT.
        del study, event_snapshots
        gc.collect()

        return best_params

    def apply_event(
        self,
        event_data: dict[str, pd.DataFrame],
        params: Any,
    ) -> dict[str, pd.Series]:
        """Run Backtester on one event and return portfolio returns."""
        rets = self._backtest_event(event_data, params)
        return {"_portfolio_": rets}

    # -- Internal helpers ---------------------------------------------------

    # Indicator names whose state depends only on input data (not strategy
    # params), so their snapshots are valid across optimization trials.
    _DATA_ONLY_INDICATORS = frozenset({
        "sd_bands", "vwap", "cum_buy_delta", "cum_sell_delta", "vwap_bands",
    })

    @staticmethod
    def _capture_indicator_snapshots(strategy: "WeatherMarketImbalanceStrategy") -> dict[str, dict]:
        """Extract snapshots from data-only indicators on *strategy*."""
        snaps: dict[str, dict] = {}
        for ind in getattr(strategy, "indicator_defs", []):
            if ind.name in WeatherMarketMCPTAdapter._DATA_ONLY_INDICATORS and hasattr(ind, "snapshot"):
                snaps[ind.name] = ind.snapshot()
        return snaps

    @staticmethod
    def _restore_indicator_snapshots(
        strategy: "WeatherMarketImbalanceStrategy",
        snapshots: dict[str, dict],
    ) -> None:
        """Restore data-only indicator snapshots into *strategy*."""
        for ind in getattr(strategy, "indicator_defs", []):
            snap = snapshots.get(ind.name)
            if snap is not None and hasattr(ind, "restore"):
                ind.restore(snap)

    @staticmethod
    def _get_or_reconstruct(
        event_data: dict[str, pd.DataFrame],
    ) -> tuple:
        """Reconstruct matrices from per-submarket DataFrames."""
        return _reconstruct_matrices(event_data)

    def _backtest_event(
        self,
        event_data: dict[str, pd.DataFrame],
        chosen_params: dict[str, Any],
        indicator_snapshots: dict[str, dict] | None = None,
    ) -> pd.Series:
        """Run a single backtest on event data, return portfolio returns.

        When *indicator_snapshots* is provided, data-only indicators are
        restored from snapshots before the backtest, eliminating redundant
        catch-up computation.
        """
        prices, returns, vwap, volume, high, low, open_, buy_vol, sell_vol = (
            _reconstruct_matrices(event_data)
        )
        if prices.empty or len(prices) < 3:
            return pd.Series(dtype=float)

        strategy_kwargs: dict[str, Any] = dict(self.base_params)
        strategy_kwargs.update(chosen_params)
        strategy_kwargs["vwap"] = vwap
        strategy_kwargs["volume"] = volume
        strategy_kwargs["high"] = high
        strategy_kwargs["low"] = low
        strategy_kwargs["open_"] = open_
        strategy_kwargs["buy_volume"] = buy_vol
        strategy_kwargs["sell_volume"] = sell_vol
        strategy_kwargs.setdefault("track_delta_history", False)

        strategy = WeatherMarketImbalanceStrategy(**strategy_kwargs)
        setattr(strategy, "record_indicator_history", False)

        if indicator_snapshots:
            self._restore_indicator_snapshots(strategy, indicator_snapshots)

        result = Backtester(strategy=strategy, rebalance_freq=self.rebalance_freq).run(
            prices,
            precomputed_returns=returns,
            include_returns=True,
            include_weights=False,
            include_indicator_signals=False,
        )
        strategy.finalize(prices)
        strategy.trade_log.clear()

        portfolio_returns = result.get("returns")
        del strategy, result
        if portfolio_returns is None or not isinstance(portfolio_returns, pd.Series):
            return pd.Series(dtype=float)
        return portfolio_returns

    def _backtest_event_with_capture(
        self,
        event_data: dict[str, pd.DataFrame],
        chosen_params: dict[str, Any],
    ) -> tuple[pd.Series, dict[str, dict]]:
        """Like _backtest_event but also returns indicator snapshots."""
        prices, returns, vwap, volume, high, low, open_, buy_vol, sell_vol = (
            _reconstruct_matrices(event_data)
        )
        if prices.empty or len(prices) < 3:
            return pd.Series(dtype=float), {}

        strategy_kwargs: dict[str, Any] = dict(self.base_params)
        strategy_kwargs.update(chosen_params)
        strategy_kwargs["vwap"] = vwap
        strategy_kwargs["volume"] = volume
        strategy_kwargs["high"] = high
        strategy_kwargs["low"] = low
        strategy_kwargs["open_"] = open_
        strategy_kwargs["buy_volume"] = buy_vol
        strategy_kwargs["sell_volume"] = sell_vol
        strategy_kwargs.setdefault("track_delta_history", False)

        strategy = WeatherMarketImbalanceStrategy(**strategy_kwargs)
        setattr(strategy, "record_indicator_history", False)

        result = Backtester(strategy=strategy, rebalance_freq=self.rebalance_freq).run(
            prices,
            precomputed_returns=returns,
            include_returns=True,
            include_weights=False,
            include_indicator_signals=False,
        )
        strategy.finalize(prices)
        snaps = self._capture_indicator_snapshots(strategy)
        strategy.trade_log.clear()

        portfolio_returns = result.get("returns")
        del strategy, result
        if portfolio_returns is None or not isinstance(portfolio_returns, pd.Series):
            return pd.Series(dtype=float), snaps
        return portfolio_returns, snaps

    def _score_returns(self, returns: pd.Series) -> float:
        """Score a return series using the configured objective."""
        if self.scoring_fn is not None:
            return float(self.scoring_fn(returns))
        from stratlab.report.metrics import compute_metrics
        metrics = compute_metrics(returns)
        raw = float(metrics.get(self.objective, 0.0))
        return _metric_direction(self.objective) * raw


def _metric_direction(metric: str) -> float:
    """Return +1 when higher-is-better, -1 when lower-is-better."""
    lower_is_better = {"max_drawdown", "volatility"}
    return -1.0 if metric in lower_is_better else 1.0
