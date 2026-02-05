"""Base strategy class and protocol."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class Strategy(ABC):
    """
    Abstract base class for portfolio strategies.

    Strategies define how to allocate weights across assets.
    The backtester calls generate_weights() at each rebalance point.
    """

    lookback: int  # Required lookback period for the strategy

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
