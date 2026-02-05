"""Performance metrics calculation and summaries."""

from typing import Any

import numpy as np
import pandas as pd


def _to_float(x: Any) -> float:
    """Safely convert to float, handling edge cases."""
    if isinstance(x, complex):
        x = x.real
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def compute_metrics(returns: pd.Series, annualization_factor: int = 252) -> dict:
    """
    Compute comprehensive portfolio performance metrics.

    Args:
        returns: Series of daily returns
        annualization_factor: Trading days per year (default 252)

    Returns:
        Dict with performance metrics
    """
    if len(returns) == 0:
        return {}

    # Basic returns
    cumulative = (1 + returns).prod()
    total_return = _to_float(cumulative) - 1
    annual_return = _to_float((1 + total_return) ** (annualization_factor / len(returns)) - 1)
    average_return = _to_float(returns.mean())

    # Volatility
    volatility = _to_float(returns.std() * np.sqrt(annualization_factor))

    # Win/loss metrics
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    non_zero = returns[returns != 0]
    winning_percentage = len(wins) / len(non_zero) if len(non_zero) > 0 else 0.0

    # Profit factor
    gross_profit = _to_float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = abs(_to_float(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Drawdown metrics
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = _to_float(drawdown.min())
    avg_drawdown = _to_float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0

    # Risk-adjusted returns
    sharpe = annual_return / volatility if volatility > 0 else 0.0

    # Sortino (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = _to_float(downside_returns.std() * np.sqrt(annualization_factor)) if len(downside_returns) > 0 else 0.0
    sortino = annual_return / downside_std if downside_std > 0 else 0.0

    # Calmar
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # Kelly criterion
    if len(wins) > 0 and len(losses) > 0:
        win_prob = len(wins) / len(non_zero)
        loss_prob = 1 - win_prob
        avg_win = _to_float(wins.mean())
        avg_loss = abs(_to_float(losses.mean()))
        kelly = win_prob - (loss_prob * avg_loss / avg_win) if avg_win > 0 else 0.0
    else:
        kelly = 0.0

    # Distribution metrics
    skewness = _to_float(returns.skew()) if len(returns) > 2 else 0.0
    kurtosis = _to_float(returns.kurtosis()) if len(returns) > 3 else 0.0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "average_return": average_return,
        "volatility": volatility,
        "winning_percentage": winning_percentage,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "avg_drawdown": avg_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "kelly": kelly,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "n_days": len(returns),
    }
