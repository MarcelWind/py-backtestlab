"""Objective/scoring functions for optimization."""

from typing import Callable, Literal

# Type alias for objective functions
ObjectiveFunc = Callable[[dict[str, float]], float]

# Built-in metric names from compute_metrics
MetricName = Literal[
    "sharpe",
    "sortino",
    "calmar",
    "total_return",
    "annual_return",
    "max_drawdown",
    "volatility",
    "profit_factor",
    "winning_percentage",
]


def make_objective(
    metric: str | MetricName,
    maximize: bool = True,
) -> ObjectiveFunc:
    """
    Create an objective function from a metric name.

    Args:
        metric: Name of metric from compute_metrics output
        maximize: If True, higher is better; if False, lower is better

    Returns:
        Function that extracts score from metrics dict
    """
    sign = 1.0 if maximize else -1.0

    def objective(metrics: dict[str, float]) -> float:
        value = metrics.get(metric, 0.0)
        return sign * float(value)

    return objective


def composite_objective(
    weights: dict[str, float],
    maximize: dict[str, bool] | None = None,
) -> ObjectiveFunc:
    """
    Create a weighted combination of metrics.

    Args:
        weights: Dict of {metric_name: weight}
        maximize: Dict of {metric_name: True if higher is better}
                  Defaults to True for all metrics

    Returns:
        Objective function computing weighted sum

    Example:
        obj = composite_objective(
            weights={'sharpe': 0.6, 'max_drawdown': 0.4},
            maximize={'sharpe': True, 'max_drawdown': False}
        )
    """
    maximize = maximize or {}

    def objective(metrics: dict[str, float]) -> float:
        total = 0.0
        for metric, weight in weights.items():
            value = metrics.get(metric, 0.0)
            sign = 1.0 if maximize.get(metric, True) else -1.0
            total += weight * sign * float(value)
        return total

    return objective
