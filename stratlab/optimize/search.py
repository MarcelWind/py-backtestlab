"""Monte Carlo parameter search implementation."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Type

import numpy as np
import pandas as pd

from ..backtest.backtester import Backtester
from ..strategy.base import Strategy
from .objective import ObjectiveFunc, make_objective


class ParamType(Enum):
    """Parameter sampling distribution types."""

    INT = auto()  # Uniform integer
    FLOAT = auto()  # Uniform float
    LOG_INT = auto()  # Log-uniform integer (good for lookback)
    LOG_FLOAT = auto()  # Log-uniform float


@dataclass
class ParamSpec:
    """Specification for a single parameter."""

    low: float
    high: float
    param_type: ParamType | None = None  # None = auto-infer from bounds

    def __post_init__(self):
        if self.param_type is None:
            # Auto-infer from bounds
            if isinstance(self.low, int) and isinstance(self.high, int):
                self.param_type = ParamType.INT
            else:
                self.param_type = ParamType.FLOAT

    def sample(self, rng: np.random.Generator) -> Any:
        """Sample a value from this parameter's distribution."""
        if self.param_type in (ParamType.LOG_INT, ParamType.LOG_FLOAT):
            # Log-uniform: sample in log space, transform back
            log_low = np.log(max(self.low, 1e-10))
            log_high = np.log(max(self.high, 1e-10))
            val = np.exp(rng.uniform(log_low, log_high))
        else:
            val = rng.uniform(self.low, self.high)

        if self.param_type in (ParamType.INT, ParamType.LOG_INT):
            return int(round(val))
        return val


@dataclass
class OptimizationResult:
    """Results from Monte Carlo optimization."""

    best_params: dict[str, Any]
    best_score: float
    best_metrics: dict[str, float]
    trials: pd.DataFrame
    objective_name: str

    def top_n(self, n: int = 10) -> pd.DataFrame:
        """Return top N trials sorted by objective score."""
        return self.trials.nlargest(n, "_objective")

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(\n"
            f"  objective='{self.objective_name}',\n"
            f"  best_score={self.best_score:.4f},\n"
            f"  best_params={self.best_params},\n"
            f"  n_trials={len(self.trials)}\n"
            f")"
        )


def _normalize_param_space(
    param_space: dict[str, tuple | ParamSpec],
) -> dict[str, ParamSpec]:
    """Convert tuple bounds to ParamSpec objects."""
    normalized = {}
    for name, spec in param_space.items():
        if isinstance(spec, ParamSpec):
            normalized[name] = spec
        elif isinstance(spec, tuple) and len(spec) == 2:
            normalized[name] = ParamSpec(low=spec[0], high=spec[1])
        else:
            raise ValueError(f"Invalid param spec for '{name}': {spec}")
    return normalized


def _sample_params(
    param_space: dict[str, ParamSpec],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Sample a single parameter configuration."""
    return {name: spec.sample(rng) for name, spec in param_space.items()}


def monte_carlo_optimize(
    strategy_class: Type[Strategy],
    param_space: dict[str, tuple | ParamSpec],
    prices: pd.DataFrame,
    objective: str | ObjectiveFunc = "sharpe",
    n_trials: int = 100,
    rebalance_freq: int | tuple[int, int] = 30,
    constraints: Callable[[dict, pd.DataFrame], bool] | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> OptimizationResult:
    """
    Monte Carlo random search for strategy parameters.

    Args:
        strategy_class: Strategy class to optimize (e.g., MomentumStrategy)
        param_space: Parameter bounds as {name: (low, high)} or {name: ParamSpec}
        prices: Price DataFrame for backtesting
        objective: Metric name ('sharpe', 'sortino', etc.) or custom ObjectiveFunc
        n_trials: Number of random parameter samples
        rebalance_freq: Fixed value or (low, high) tuple to also optimize
        constraints: Optional function(params, prices) -> bool for validity
        seed: Random seed for reproducibility
        verbose: Print progress updates

    Returns:
        OptimizationResult with best params and full trial history

    Example:
        result = monte_carlo_optimize(
            MomentumStrategy,
            param_space={'lookback': (10, 90), 'top_n': (1, 5)},
            prices=prices,
            objective='sharpe',
            n_trials=100,
        )
        print(result.best_params)
        print(result.top_n(5))
    """
    rng = np.random.default_rng(seed)

    # Normalize param space
    strategy_space = _normalize_param_space(param_space)

    # Handle rebalance_freq optimization
    optimize_rebalance = isinstance(rebalance_freq, tuple)
    rebalance_space: ParamSpec | None = None
    if optimize_rebalance:
        rebalance_space = ParamSpec(
            low=rebalance_freq[0],
            high=rebalance_freq[1],
            param_type=ParamType.INT,
        )

    # Create objective function
    if isinstance(objective, str):
        obj_func = make_objective(objective, maximize=True)
        obj_name = objective
    else:
        obj_func = objective
        obj_name = "custom"

    # Default constraint: lookback must fit in data
    def default_constraint(params: dict, data: pd.DataFrame) -> bool:
        lookback = params.get("lookback", 0)
        return lookback < len(data) - 10  # Need at least 10 days after lookback

    constraint_func = constraints or default_constraint

    # Run trials
    trial_records = []
    best_score = float("-inf")
    best_params: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None

    max_retries = 10

    for trial_idx in range(n_trials):
        # Sample parameters with retry for constraints
        for _ in range(max_retries):
            params = _sample_params(strategy_space, rng)
            rb_freq: int
            if optimize_rebalance and rebalance_space is not None:
                rb_freq = rebalance_space.sample(rng)
            else:
                rb_freq = int(rebalance_freq)  # type: ignore[arg-type]

            if constraint_func(params, prices):
                break
        else:
            # Skip this trial if we can't find valid params
            continue

        try:
            # Create strategy and run backtest
            strategy = strategy_class(**params)
            backtester = Backtester(strategy, rebalance_freq=rb_freq)
            result = backtester.run(prices)

            # Compute objective score
            metrics = result["metrics"]
            score = obj_func(metrics)

            # Record trial
            record = {**params, "_rebalance_freq": rb_freq, "_objective": score}
            record.update({f"m_{k}": v for k, v in metrics.items()})
            trial_records.append(record)

            # Track best
            if score > best_score:
                best_score = score
                best_params = {**params, "rebalance_freq": rb_freq}
                best_metrics = metrics

            if verbose and (trial_idx + 1) % 10 == 0:
                print(
                    f"Trial {trial_idx + 1}/{n_trials}: best {obj_name}={best_score:.4f}"
                )

        except Exception as e:
            if verbose:
                print(f"Trial {trial_idx} failed: {e}")
            continue

    # Build results DataFrame
    trials_df = pd.DataFrame(trial_records)

    return OptimizationResult(
        best_params=best_params or {},
        best_score=best_score if best_score > float("-inf") else 0.0,
        best_metrics=best_metrics or {},
        trials=trials_df,
        objective_name=obj_name,
    )


# Convenience alias
optimize = monte_carlo_optimize
