# DARWIN HAMMER — match 136, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_model__m176_s1.py (gen3)
# parent_b: hybrid_regret_engine_hybrid_doomsday_cale_m19_s7.py (gen2)
# born: 2026-05-29T23:27:07Z

import sys
import math
import random
import pathlib
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Dict

# ----------
# Parent A structures
# ----------
def _blade_sign(indices: List[int]) -> (List[int], int):
    """Return (sorted_blade, sign) after bubble-sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate index cancels (e_i ^ e_i = 0)
                lst.pop(j)
                lst.pop(j)  # second element shifts to j after first pop
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a: List[int], blade_b: List[int]) -> (List[int], int):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def geometric_product(a: Dict[frozenset, float], b: Dict[frozenset, float]) -> Dict[frozenset, float]:
    """
    Full Clifford product `ab`.
    `a` and `b` are dicts mapping frozenset blades -> scalar coefficient.
    Returns a new dict representing the multivector product.
    """
    result = {}
    for blade_a, coef_a in a.items():
        for blade_b, coef_b in b.items():
            blade, sign = _multiply_blades(list(blade_a), list(blade_b))
            result[blade] = result.get(blade, 0) + sign * coef_a * coef_b
    return result

# ----------
# Parent B structures
# ----------
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
    return (datetime(year, month, day).weekday() + 1) % 7

def gini_coefficient(values: Iterable[float]) -> float:
    """Standard Gini coefficient for a non-negative distribution."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

def _weekday_sequence(year: int, month: int, num_days: int) -> List[int]:
    """Generate a list of weekday indices for the first *num_days* of a month."""
    return [doomsday(year, month, day) for day in range(1, num_days + 1)]

def _map_actions_to_weekdays(
    actions: List[MathAction],
    year: int,
    month: int,
    num_days: int,
) -> Dict[str, int]:
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

def compute_hybrid_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    year: int,
    month: int,
    num_days: int,
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """
    1. Compute the classic regret-weighted probabilities (parent A).
    2. Distribute those probabilities over weekdays using the Doomsday mapping (parent B).
    3. Evaluate the inequality 
    """
    # Compute regret-weighted probabilities
    regret_probabilities = {}
    for action in actions:
        regret = 0
        for counterfactual in counterfactuals:
            if counterfactual.action_id == action.id:
                regret += counterfactual.outcome_value * counterfactual.probability
        regret_probabilities[action.id] = regret / sum(abs(cf.outcome_value * cf.probability) for cf in counterfactuals if cf.action_id == action.id)

    # Distribute probabilities over weekdays
    weekday_mapping = _map_actions_to_weekdays(actions, year, month, num_days)
    weekday_probabilities = {}
    for action_id, weekday in weekday_mapping.items():
        weekday_probabilities[weekday] = weekday_probabilities.get(weekday, 0) + regret_probabilities.get(action_id, 0)

    # Apply Clifford product to modulate probabilities
    base_blade = {frozenset([0]): 1}
    modulation_blade = {frozenset([1]): 1}
    clifford_product = geometric_product(base_blade, modulation_blade)
    modulated_probabilities = {}
    for weekday, probability in weekday_probabilities.items():
        modulated_probabilities[weekday] = probability * clifford_product.get(frozenset([0]), 1)

    # Normalize modulated probabilities
    total_probability = sum(modulated_probabilities.values())
    if total_probability > 0:
        modulated_probabilities = {k: v / total_probability for k, v in modulated_probabilities.items()}

    return modulated_probabilities

def hybrid_operation(actions: List[MathAction], counterfactuals: List[MathCounterfactual], year: int, month: int, num_days: int) -> Dict[str, float]:
    """
    Perform the hybrid operation by combining the regret-weighted probabilities 
    with the Clifford product and the Doomsday mapping.
    """
    return compute_hybrid_regret_weighted_strategy(actions, counterfactuals, year, month, num_days)

def hybrid_gini_coefficient(actions: List[MathAction], counterfactuals: List[MathCounterfactual], year: int, month: int, num_days: int) -> float:
    """
    Compute the Gini coefficient of the hybrid operation.
    """
    probabilities = compute_hybrid_regret_weighted_strategy(actions, counterfactuals, year, month, num_days)
    values = list(probabilities.values())
    return gini_coefficient(values)

if __name__ == "__main__":
    actions = [MathAction("action1", 0.5), MathAction("action2", 0.3)]
    counterfactuals = [MathCounterfactual("action1", 0.2), MathCounterfactual("action2", 0.1)]
    year = 2024
    month = 12
    num_days = 31
    print(hybrid_operation(actions, counterfactuals, year, month, num_days))
    print(hybrid_gini_coefficient(actions, counterfactuals, year, month, num_days))