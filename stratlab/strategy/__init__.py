"""Strategy layer - signals, allocation methods, and strategies."""

from .allocation import optimize_sharpe_weights
from .base import BuyAndHoldStrategy, Strategy, StrategySpec
from .momentum import MomentumStrategy
from .sharpe import SharpeStrategy
from .signals import (
    apply_signals,
    equal_weight_allocation,
    momentum_score,
    rank_assets,
    scores_to_weights,
    positions_to_weights,
    top_n_mask,
)
