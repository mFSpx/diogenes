# DARWIN HAMMER — match 1100, survivor 0
# gen: 3
# parent_a: fold_change_detection.py (gen0)
# parent_b: hybrid_regret_engine_hybrid_doomsday_cale_m19_s7.py (gen2)
# born: 2026-05-29T23:32:44Z

"""
Module for the hybrid algorithm combining fold-change detection and hybrid regret engine.
The mathematical bridge between the two algorithms is established by applying the fold-change detection
update equations to the expected values of the MathAction objects. This allows the algorithm to adaptively
update the expected values based on the input signals, while also considering the Doomsday algorithm and
Gini coefficient for distribution and evaluation.

Parent algorithms:
- fold_change_detection.py
- hybrid_regret_engine_hybrid_doomsday_cale_m19_s7.py
"""

import math
import random
import sys
import numpy as np
from dataclasses import dataclass
from collections.abc import Iterable
from datetime import date

@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index (0=Monday … 6=Sunday) using Doomsday algorithm."""
    return (date(year, month, day).weekday() + 1) % 7

def gini_coefficient(values: Iterable[float]) -> float:
    """Standard Gini coefficient for a non‑negative distribution."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

def step(u: float, x: float, y: float, dt: float = 1.0, gain: float = 1.0, decay_x: float = 1.0, decay_y: float = 1.0, eps: float = 1e-12) -> tuple[float, float]:
    """Advance the feed-forward state using Euler integration."""
    if dt < 0:
        raise ValueError('dt must be non-negative')
    ratio = u / max(abs(x), eps)
    dy = gain * ratio - decay_y * y
    dx = u - decay_x * x
    return x + dt * dx, y + dt * dy

def _weekday_sequence(year: int, month: int, num_days: int) -> list[int]:
    """Generate a list of weekday indices for the first *num_days* of a month."""
    return [doomsday(year, month, day) for day in range(1, num_days + 1)]

def _map_actions_to_weekdays(
    actions: list[MathAction],
    year: int,
    month: int,
    num_days: int,
) -> dict[str, int]:
    """
    Assign each action to a weekday by cycling through the month’s weekday sequence.
    Returns a mapping action_id → weekday_index.
    """
    if not actions:
        return {}
    weekdays = _weekday_sequence(year, month, num_days)
    mapping = {}
    for idx, act in enumerate(actions):
        mapping[act.id] = weekdays[idx % len(weekdays)]
    return mapping

def update_action_expected_values(actions: list[MathAction], inputs: list[float], **kwargs) -> list[MathAction]:
    """
    Update the expected values of the MathAction objects using the fold-change detection update equations.
    """
    updated_actions = []
    for action in actions:
        x, y = 0.0, 0.0
        for u in inputs:
            x, y = step(u, x, y, **kwargs)
        updated_actions.append(MathAction(action.id, y, action.cost, action.risk))
    return updated_actions

def compute_hybrid_regret_weighted_strategy(
    actions: list[MathAction],
    counterfactuals: list[MathCounterfactual],
    year: int,
    month: int,
    num_days: int,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """
    Compute the classic regret‑weighted probabilities and distribute them over weekdays.
    """
    updated_actions = update_action_expected_values(actions, [action.expected_value for action in actions])
    mapping = _map_actions_to_weekdays(updated_actions, year, month, num_days)
    probabilities = {}
    for action in updated_actions:
        probabilities[action.id] = action.expected_value / sum(a.expected_value for a in updated_actions)
    return {action_id: probabilities[action_id] for action_id in mapping}

def evaluate_inequality(actions: list[MathAction], counterfactuals: list[MathCounterfactual]) -> float:
    """
    Evaluate the inequality based on the updated expected values and counterfactuals.
    """
    gini = gini_coefficient([action.expected_value for action in actions])
    regret = sum([cf.outcome_value * cf.probability for cf in counterfactuals])
    return gini * regret

if __name__ == "__main__":
    actions = [MathAction("action1", 10.0), MathAction("action2", 20.0)]
    counterfactuals = [MathCounterfactual("action1", 5.0), MathCounterfactual("action2", 10.0)]
    year, month, num_days = 2024, 9, 30
    epsilon = 1e-6
    print(compute_hybrid_regret_weighted_strategy(actions, counterfactuals, year, month, num_days, epsilon))
    print(evaluate_inequality(actions, counterfactuals))