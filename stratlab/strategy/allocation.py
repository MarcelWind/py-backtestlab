"""Portfolio allocation methods."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_sharpe_weights(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    allow_cash: bool = True,
) -> np.ndarray:
    """
    Find portfolio weights that maximize Sharpe ratio.

    Args:
        returns: DataFrame of asset returns (rows=dates, cols=assets)
        risk_free_rate: Daily risk-free rate (default 0)
        allow_cash: If True, weights can sum to less than 1 (hold cash)

    Returns:
        Array of weights (same order as returns columns)
    """
    n_assets = len(returns.columns)

    if n_assets == 0 or len(returns) < 2:
        return np.array([])

    mean_returns = returns.mean().values.copy()
    cov_matrix = returns.cov().values.copy()

    # Handle near-singular covariance (add small regularization)
    cov_matrix += np.eye(n_assets) * 1e-8

    def neg_sharpe(weights):
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        if port_vol < 1e-10:
            return 0  # Avoid division by zero
        return -(port_return - risk_free_rate) / port_vol

    # Constraints: long-only, weights sum <= 1
    constraints = []
    if allow_cash:
        constraints.append({"type": "ineq", "fun": lambda w: 1 - np.sum(w)})
    else:
        constraints.append({"type": "eq", "fun": lambda w: np.sum(w) - 1})

    bounds = [(0, 1) for _ in range(n_assets)]

    # Initial guess: equal weight
    x0 = np.ones(n_assets) / n_assets

    result = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    weights = result.x

    # Clean up small weights (noise)
    weights[weights < 0.01] = 0

    # If all weights are zero or negative Sharpe, go to cash
    if weights.sum() < 0.01 or neg_sharpe(weights) >= 0:
        return np.zeros(n_assets)

    return weights
