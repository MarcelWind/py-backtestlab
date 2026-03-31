"""Parameter optimization - search and scoring."""

from .objective import (
    ObjectiveFunc,
    composite_objective,
    make_objective,
)
from .search import (
    OptimizationResult,
    ParamSpec,
    ParamType,
    monte_carlo_optimize,
    optimize,
)
from .rule_search import (
    estimate_max_unique_trials,
    load_param_rules,
    params_key,
    round_to_step,
    rule_enabled,
    rule_to_spec,
    sample_trial_params,
)
from .bayesian_search import (
    create_study,
    rules_to_optuna_params,
)

__all__ = [
    "ObjectiveFunc",
    "composite_objective",
    "make_objective",
    "OptimizationResult",
    "ParamSpec",
    "ParamType",
    "monte_carlo_optimize",
    "optimize",
    "estimate_max_unique_trials",
    "load_param_rules",
    "params_key",
    "round_to_step",
    "rule_enabled",
    "rule_to_spec",
    "sample_trial_params",
    "create_study",
    "rules_to_optuna_params",
]
