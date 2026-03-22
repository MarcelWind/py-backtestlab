"""Rule-driven parameter search helpers for Monte Carlo runners."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from stratlab.io.json_utils import json_error_context
from .search import ParamSpec, ParamType


def load_param_rules(config_path: Path) -> list[dict[str, Any]]:
    """Load Monte Carlo parameter rules from JSON file."""
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"Parameter config file not found: {config_path}") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        context = json_error_context(text, exc.lineno)
        raise ValueError(
            f"Parameter config is not valid JSON: {config_path} "
            f"(line {exc.lineno}, column {exc.colno}): {exc.msg}\n{context}"
        ) from exc

    rules = raw.get("parameters", [])
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"Invalid parameter config at {config_path}: expected non-empty 'parameters' list")

    seen: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("Each parameter rule must be an object")
        name = str(rule.get("name", "")).strip()
        if not name:
            raise ValueError("Each parameter rule must include a non-empty 'name'")
        if name in seen:
            raise ValueError(f"Duplicate parameter rule for {name!r}")
        seen.add(name)
        rule_type = str(rule.get("type", "")).strip()
        if rule_type not in {"bool", "choice", "int", "float", "log_float"}:
            raise ValueError(f"Unsupported rule type {rule_type!r} for parameter {name!r}")
        if rule_type in {"bool", "choice"}:
            values = rule.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError(f"Rule {name!r} of type {rule_type!r} requires non-empty 'values'")
            if rule_type == "bool" and not all(isinstance(v, bool) for v in values):
                raise ValueError(f"Boolean rule {name!r} must contain only true/false values")
        else:
            if "low" not in rule or "high" not in rule:
                raise ValueError(f"Numeric rule {name!r} requires 'low' and 'high'")
    return rules


def rule_enabled(rule: dict[str, Any], chosen: dict[str, Any], base_params: dict[str, object]) -> bool:
    """Return whether a rule should be enabled under enabled_if conditions."""
    cond = rule.get("enabled_if", {})
    if not cond:
        return True
    if not isinstance(cond, dict):
        raise ValueError(f"enabled_if for {rule.get('name')} must be a mapping")
    for dep_name, dep_val in cond.items():
        actual = chosen.get(dep_name, base_params.get(dep_name))
        if actual != dep_val:
            return False
    return True


def rule_to_spec(rule: dict[str, Any]) -> ParamSpec:
    """Convert JSON rule representation into ParamSpec."""
    low = float(rule["low"])
    high = float(rule["high"])
    rtype = str(rule["type"])
    if rtype == "int":
        return ParamSpec(low=int(round(low)), high=int(round(high)), param_type=ParamType.INT)
    if rtype == "log_float":
        return ParamSpec(low=low, high=high, param_type=ParamType.LOG_FLOAT)
    return ParamSpec(low=low, high=high, param_type=ParamType.FLOAT)


def round_to_step(value: float, step: float, low: float, high: float) -> float:
    """Snap sampled value to a grid step while keeping [low, high] bounds."""
    if step <= 0:
        return float(min(max(value, low), high))
    snapped = low + round((value - low) / step) * step
    return float(min(max(snapped, low), high))


def sample_trial_params(
    rules: list[dict[str, Any]],
    base_params: dict[str, object],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Sample one candidate respecting enabled_if conditions."""
    sampled: dict[str, Any] = {}
    for rule in rules:
        name = str(rule["name"])
        if not rule_enabled(rule, sampled, base_params):
            continue

        rtype = str(rule["type"])
        if rtype in {"bool", "choice"}:
            values = rule.get("values", [True, False])
            sampled[name] = values[int(rng.integers(0, len(values)))]
            continue

        spec = rule_to_spec(rule)
        raw_val = spec.sample(rng)
        step = float(rule.get("step", 0.0))
        if spec.param_type in (ParamType.INT, ParamType.LOG_INT):
            sampled[name] = int(round(round_to_step(float(raw_val), max(1.0, step), spec.low, spec.high)))
        else:
            sampled[name] = float(round_to_step(float(raw_val), step, spec.low, spec.high))
    return sampled


def estimate_max_unique_trials(rules: list[dict[str, Any]], base_params: dict[str, object]) -> int | None:
    """Estimate unique parameter combinations implied by rule steps and booleans."""
    discrete_rules = [r for r in rules if str(r.get("type")) in {"bool", "choice"}]
    num_rules = [r for r in rules if str(r.get("type")) not in {"bool", "choice"}]

    discrete_value_lists: list[list[Any]] = [list(r.get("values", [True, False])) for r in discrete_rules]
    total = 0
    for combo in itertools.product(*discrete_value_lists) if discrete_value_lists else [()]:
        chosen_discrete = {str(r["name"]): combo[i] for i, r in enumerate(discrete_rules)}
        branch_product = 1
        for rule in num_rules:
            if not rule_enabled(rule, chosen_discrete, base_params):
                continue
            low = float(rule["low"])
            high = float(rule["high"])
            if low == high:
                levels = 1
            else:
                step = float(rule.get("step", 0.0))
                if step <= 0:
                    return None
                levels = int(max(0, math.floor((high - low) / step) + 1))
                levels = max(1, levels)
            branch_product *= levels
        total += branch_product
    return int(total)


def params_key(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Stable hashable key for deduplicating sampled parameter dictionaries."""
    return tuple(sorted(params.items(), key=lambda x: x[0]))
