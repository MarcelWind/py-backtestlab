"""Relative momentum portfolio strategy."""

import numpy as np
import pandas as pd

from .base import Strategy
from .signals import momentum_score, top_n_mask, equal_weight_allocation


class MomentumStrategy(Strategy):
    """
    Relative momentum portfolio strategy.

    Ranks assets by momentum and allocates to top performers.
    Can go to cash if no asset has positive momentum.
    """

    def __init__(
        self,
        lookback: int = 30,
        top_n: int = 3,
        min_momentum: float | None = 0.0,
    ):
        """
        Args:
            lookback: Days for momentum calculation
            top_n: Number of top assets to hold
            min_momentum: Minimum momentum to be selected (None = no filter, 0 = positive only)
        """
        self.lookback = lookback
        self.top_n = top_n
        self.min_momentum = min_momentum

    def generate_weights(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        index: int,
    ) -> np.ndarray:
        """Generate momentum-based weights."""
        # Compute momentum at this point
        mom_scores = prices.iloc[index] / prices.iloc[index - self.lookback] - 1

        # Select top N with minimum momentum filter
        mask = top_n_mask(
            mom_scores.to_frame().T,
            n=self.top_n,
            min_score=self.min_momentum,
        ).iloc[0]

        # Equal weight allocation
        weights = equal_weight_allocation(mask.to_frame().T).iloc[0]
        return weights.values
