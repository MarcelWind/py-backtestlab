"""Portfolio backtesting engine."""

import numpy as np
import pandas as pd

from ..report.metrics import compute_metrics
from ..strategy.base import Strategy


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns from price data."""
    return prices.pct_change().dropna()


class Backtester:
    """
    Generic portfolio backtesting engine.

    Runs any Strategy through historical data and tracks performance.
    The strategy defines the allocation logic; the backtester handles execution.
    """

    def __init__(self, strategy: Strategy, rebalance_freq: int = 30):
        """
        Args:
            strategy: Strategy instance that generates weights
            rebalance_freq: Days between rebalances
        """
        self.strategy = strategy
        self.rebalance_freq = rebalance_freq

    def run(
        self,
        prices: pd.DataFrame,
        *,
        include_returns: bool = True,
        include_weights: bool = True,
        include_indicator_signals: bool = True,
    ) -> dict:
        """
        Run backtest with the configured strategy.

        Args:
            prices: DataFrame of close prices (rows=dates, cols=assets)

        Returns:
            Dict with keys:
            - 'returns': Portfolio returns series aligned to rebalance dates
            - 'weights': Weights history as DataFrame (n_rebalances, n_assets)
            - 'metrics': Performance metrics dict (sharpe, total_return, etc)
            - 'indicator_signals': Dict mapping indicator name to DataFrame of
              per-rebalance, per-asset signals (n_rebalances, n_assets)

            The `include_*` flags can be set to False to avoid materializing
            heavy outputs when only metrics are needed.
        """
        returns = compute_returns(prices)
        n_days = len(returns)
        assets = returns.columns.tolist()
        lookback = self.strategy.lookback

        portfolio_returns = []
        weights_history = [] if include_weights else None
        dates = []

        current_weights = np.zeros(len(assets))

        for i in range(lookback, n_days):
            # Rebalance check
            days_since_start = i - lookback
            if days_since_start % self.rebalance_freq == 0:
                self.strategy._compute_indicators(prices, returns, i)
                current_weights = self.strategy.generate_weights(prices, returns, i)

            # Calculate portfolio return for this day
            day_returns = returns.iloc[i].to_numpy()
            port_return = float(np.dot(current_weights, day_returns))

            portfolio_returns.append(port_return)
            if include_weights and weights_history is not None:
                weights_history.append(current_weights.copy())
            dates.append(returns.index[i])

        portfolio_returns = pd.Series(portfolio_returns, index=dates, name="portfolio")
        result: dict[str, object] = {
            "metrics": compute_metrics(portfolio_returns),
        }

        if include_returns:
            result["returns"] = portfolio_returns

        if include_weights and weights_history is not None:
            result["weights"] = pd.DataFrame(weights_history, index=dates, columns=assets)

        if include_indicator_signals:
            result["indicator_signals"] = self.strategy.indicator_series

        return result


# Backward compatibility alias
class PortfolioBacktest(Backtester):
    """
    Convenience class that uses SharpeStrategy by default.

    For backward compatibility with existing code.
    """

    def __init__(
        self,
        lookback: int = 60,
        rebalance_freq: int = 30,
        risk_free_rate: float = 0.0,
    ):
        from ..strategy.sharpe import SharpeStrategy

        strategy = SharpeStrategy(
            lookback=lookback,
            risk_free_rate=risk_free_rate,
        )
        super().__init__(strategy=strategy, rebalance_freq=rebalance_freq)
