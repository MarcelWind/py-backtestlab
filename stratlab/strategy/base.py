"""Base strategy class and protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .indicators import Indicator


class Strategy(ABC):
    """
    Abstract base class for portfolio strategies.

    Strategies define how to allocate weights across assets.
    The backtester calls generate_weights() at each rebalance point.

    Indicator system
    ----------------
    Subclasses may declare a list of Indicator instances as ``indicator_defs``.
    The backtester calls ``_compute_indicators()`` before each ``generate_weights``
    call, storing results in ``self.indicators`` keyed by indicator name.

    Example::

        class MyStrategy(Strategy):
            indicator_defs = [
                VwapSlope(vwap=vwap_df, lookback=30),
                BandPosition(lookback_hours=6.0),
            ]

            def generate_weights(self, prices, returns, index):
                slope = self.indicators["vwap_slope"]  # pd.Series per asset
                bands = self.indicators["band_position"]  # pd.DataFrame (stats x assets)
    """

    lookback: int  # Required lookback period for the strategy

    # Subclasses may override with a list of Indicator instances.
    indicator_defs: list[Indicator] = []

    # Populated before each generate_weights call by _compute_indicators().
    indicators: dict[str, Any]

    def _compute_indicators(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        index: int,
    ) -> None:
        """Pre-compute all declared indicators and store results in ``self.indicators``."""
        if not hasattr(self, "_ind_acc"):
            self._ind_acc: dict[str, list[tuple]] = {}
        result: dict[str, Any] = {}
        for ind in self.indicator_defs:
            val = ind.compute(prices, returns, index)
            result[ind.name] = val
            if isinstance(val, pd.Series):
                self._ind_acc.setdefault(ind.name, []).append((prices.index[index], val))
        self.indicators = result

    @property
    def indicator_series(self) -> dict[str, pd.DataFrame]:
        """Per-bar indicator history accumulated during the backtest run.

        Returns a dict mapping indicator name to a DataFrame of shape
        (n_rebalance_bars, n_assets).  Only indicators whose ``compute()``
        returns a ``pd.Series`` are included (e.g. VwapSlope, VwapVolumeImbalance,
        MeanReversion). DataFrame-returning indicators (e.g. BandPosition) are
        excluded.
        """
        acc = getattr(self, "_ind_acc", {})
        return {
            name: pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])
            for name, rows in acc.items()
            if rows
        }

    @abstractmethod
    def generate_weights(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        index: int,
    ) -> np.ndarray:
        """
        Generate portfolio weights at a given point in time.

        Args:
            prices: Full price DataFrame (all history up to current point)
            returns: Full returns DataFrame (all history up to current point)
            index: Current index position in the DataFrames

        Returns:
            Array of weights for each asset (same order as DataFrame columns)
        """
        pass


class BuyAndHoldStrategy(Strategy):
    """
    Static buy-and-hold benchmark strategy.

    Holds a fixed weight distribution that never changes.
    Useful as a baseline to compare active strategies against.
    """

    lookback = 0

    def __init__(self, weights: np.ndarray | list[float] | None = None):
        """
        Args:
            weights: Fixed weight vector. If None, will equal-weight all assets.
        """
        self._weights = np.array(weights) if weights is not None else None

    def generate_weights(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        index: int,
    ) -> np.ndarray:
        """Return the fixed weight vector."""
        if self._weights is not None:
            return self._weights
        # Default: equal weight across all assets
        n_assets = len(prices.columns)
        return np.ones(n_assets) / n_assets


@dataclass
class StrategySpec:
    """Specification for a strategy with its parameter space."""

    name: str
    param_space: dict[str, tuple[Any, Any]]  # param_name -> (min, max)

    def get_default_params(self) -> dict[str, Any]:
        """Return midpoint of param ranges as defaults."""
        defaults = {}
        for name, (low, high) in self.param_space.items():
            if isinstance(low, int) and isinstance(high, int):
                defaults[name] = (low + high) // 2
            else:
                defaults[name] = (low + high) / 2
        return defaults
