"""Sharpe ratio optimization strategy."""

import numpy as np
import pandas as pd

from .allocation import optimize_sharpe_weights
from .base import Strategy


class SharpeStrategy(Strategy):
    """
    Mean-variance optimization strategy.

    Allocates to maximize portfolio Sharpe ratio using historical returns.
    Can hold cash if no positive Sharpe allocation exists.
    """

    def __init__(
        self,
        lookback: int = 60,
        risk_free_rate: float = 0.0,
        allow_cash: bool = True,
    ):
        """
        Args:
            lookback: Days of history for optimization
            risk_free_rate: Annual risk-free rate
            allow_cash: If True, can hold partial cash position
        """
        self.lookback = lookback
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 252
        self.allow_cash = allow_cash

    def generate_weights(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        index: int,
    ) -> np.ndarray:
        """Generate Sharpe-optimized weights."""
        lookback_returns = returns.iloc[index - self.lookback:index]

        return optimize_sharpe_weights(
            lookback_returns,
            risk_free_rate=self.daily_rf,
            allow_cash=self.allow_cash,
        )
