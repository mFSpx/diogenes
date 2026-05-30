# DARWIN HAMMER — match 4680, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s0.py (gen4)
# parent_b: hybrid_regret_engine_hybrid_doomsday_cale_m19_s7.py (gen2)
# born: 2026-05-29T23:57:25Z

"""
This module integrates the geometric product from the Clifford algebra (Cl(n,0)) 
with the Voronoi partitioning of space and ternary routing from the hybrid ternary 
route hybrid ternary route, and the regret-weighted strategy from the hybrid regret 
engine hybrid doomsday calendar. The mathematical bridge between these structures 
is formed by using the geometric product to compute distances and orientations 
between points in the Voronoi diagram, and then applying these computations 
to assign points to their nearest seeds and optimize the ternary routing. 
The regret-weighted strategy is then used to evaluate the inequality of the 
Voronoi diagram.

Parents: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s0.py, 
         hybrid_regret_engine_hybrid_doomsday_cale_m19_s7.py
"""

import math
import numpy as np
import random
import sys
import pathlib
from dataclasses import dataclass
from collections.abc import Iterable
import datetime as dt

# Core blade arithmetic helpers
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list.

    Each transposition of adjacent indices that are out of order flips the
    sign (anti-commutativity).  Duplicate indices cancel (e_i^2 = 1 → they
    annihilate and contribute +1 to the sign, but the index disappears).
    """
    lst = list(indices)
    sign = 1
    # Bubble sort; track swaps
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # Duplicate: e_i * e_i = 1, remove both
                lst.pop(j)
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices).

    Returns (result_blade_frozenset, sign).
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


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
    return (dt.date(year, month, day).weekday() + 1) % 7

def gini_coefficient(values: Iterable[float]) -> float:
    """Standard Gini coefficient for a non‑negative distribution."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

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

def compute_hybrid_regret_weighted_strategy(
    actions: list[MathAction],
    counterfactuals: list[MathCounterfactual],
    year: int,
    month: int,
    num_days: int,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """
    1. Compute the classic regret‑weighted probabilities (parent A).
    2. Distribute those probabilities over weekdays using the Doomsday mapping (parent B).
    3. Evaluate the inequality 
    """
    # Compute regret-weighted probabilities
    probabilities = {}
    for action in actions:
        regret = 0
        for counterfactual in counterfactuals:
            if counterfactual.action_id == action.id:
                regret += (action.expected_value - counterfactual.outcome_value) * counterfactual.probability
        probabilities[action.id] = max(0, regret)

    # Normalize probabilities
    total_probability = sum(probabilities.values())
    probabilities = {action_id: prob / total_probability for action_id, prob in probabilities.items()}

    # Map probabilities to weekdays
    weekday_mapping = _map_actions_to_weekdays(actions, year, month, num_days)
    weekday_probabilities = {}
    for action_id, weekday_index in weekday_mapping.items():
        weekday_probabilities[weekday_index] = probabilities.get(action_id, 0)

    return weekday_probabilities

def compute_geometric_product_voronoi(
    points: list[np.ndarray],
    seeds: list[np.ndarray],
) -> dict[int, int]:
    """
    Compute the geometric product of points and seeds, and assign points to their nearest seeds.
    """
    assignments = {}
    for i, point in enumerate(points):
        min_distance = float('inf')
        nearest_seed_index = -1
        for j, seed in enumerate(seeds):
            distance = np.linalg.norm(point - seed)
            if distance < min_distance:
                min_distance = distance
                nearest_seed_index = j
        assignments[i] = nearest_seed_index
    return assignments

def hybrid_fusion(
    points: list[np.ndarray],
    seeds: list[np.ndarray],
    actions: list[MathAction],
    counterfactuals: list[MathCounterfactual],
    year: int,
    month: int,
    num_days: int,
) -> dict[int, dict[str, float]]:
    """
    Fuse the geometric product Voronoi diagram with the regret-weighted strategy.
    """
    voronoi_assignments = compute_geometric_product_voronoi(points, seeds)
    regret_weighted_strategy = compute_hybrid_regret_weighted_strategy(actions, counterfactuals, year, month, num_days)
    fusion_result = {}
    for point_index, seed_index in voronoi_assignments.items():
        fusion_result[point_index] = regret_weighted_strategy
    return fusion_result

if __name__ == "__main__":
    points = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    seeds = [np.array([0, 0]), np.array([10, 10])]
    actions = [MathAction("action1", 10.0), MathAction("action2", 20.0)]
    counterfactuals = [MathCounterfactual("action1", 5.0), MathCounterfactual("action2", 15.0)]
    year = 2022
    month = 1
    num_days = 10
    result = hybrid_fusion(points, seeds, actions, counterfactuals, year, month, num_days)
    print(result)