"""Bayesian (Optuna TPE) parameter search for rule-driven configs."""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna

from .rule_search import round_to_step, rule_enabled


def rules_to_optuna_params(
    trial: optuna.Trial,
    rules: list[dict[str, Any]],
    base_params: dict[str, object],
) -> dict[str, Any]:
    """Convert JSON parameter rules into Optuna trial suggestions.

    Handles ``enabled_if`` conditionals using Optuna's define-by-run API:
    a param is only suggested when all its dependencies are already satisfied
    by previously suggested values.

    Fixed-value params (``low == high`` for numerics, single-element
    ``values`` for bool/choice) are set directly without calling
    ``trial.suggest_*`` to reduce search dimensionality.
    """
    sampled: dict[str, Any] = {}

    for rule in rules:
        name: str = str(rule["name"])

        if not rule_enabled(rule, sampled, base_params):
            continue

        rtype: str = str(rule["type"])

        # -- bool / choice ---------------------------------------------------
        if rtype in {"bool", "choice"}:
            values = rule.get("values", [True, False])
            if len(values) == 1:
                sampled[name] = values[0]
            else:
                sampled[name] = trial.suggest_categorical(name, values)
            continue

        # -- numeric (int, float, log_float) ---------------------------------
        low = rule["low"]
        high = rule["high"]
        step = rule.get("step", 0)

        # Fixed-value shortcut
        if low == high:
            sampled[name] = int(round(low)) if rtype == "int" else float(low)
            continue

        if rtype == "int":
            int_step = max(1, int(round(step))) if step else 1
            sampled[name] = trial.suggest_int(name, int(round(low)), int(round(high)), step=int_step)

        elif rtype == "log_float":
            # Optuna suggest_float with log=True requires low > 0
            raw = trial.suggest_float(name, float(low), float(high), log=True)
            if step > 0:
                raw = round_to_step(raw, float(step), float(low), float(high))
            sampled[name] = raw

        else:  # float
            if step > 0:
                raw = trial.suggest_float(name, float(low), float(high), step=float(step))
            else:
                raw = trial.suggest_float(name, float(low), float(high))
            sampled[name] = raw

    return sampled


def create_study(
    seed: int,
    *,
    direction: str = "maximize",
) -> optuna.Study:
    """Create an Optuna study with TPE sampler seeded for reproducibility."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    return study
