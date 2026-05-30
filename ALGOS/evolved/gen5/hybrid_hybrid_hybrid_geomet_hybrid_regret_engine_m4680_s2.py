# DARWIN HAMMER — match 4680, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s0.py (gen4)
# parent_b: hybrid_regret_engine_hybrid_doomsday_cale_m19_s7.py (gen2)
# born: 2026-05-29T23:57:25Z

"""
This module fuses the hybrid geometric product from Clifford algebra (Cl(n,0)) 
with Voronoi partitioning and ternary routing from parent A, 
and the regret-engine with Doomsday calendar from parent B. 
The mathematical bridge between these structures is formed by using 
the geometric product to compute distances and orientations between points 
in the Voronoi diagram, and then applying these computations to assign 
points to their nearest seeds and optimize the ternary routing. 
The regret-engine is then used to evaluate the optimality of the 
resulting routing strategy.

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
    regret_weights = {}
    for action in actions:
        regret = 0
        for counterfactual in counterfactuals:
            if counterfactual.action_id == action.id:
                regret += (action.expected_value - counterfactual.outcome_value) * counterfactual.probability
        regret_weights[action.id] = regret

    # Normalize regret weights
    total_regret = sum(regret_weights.values())
    if total_regret == 0:
        return {action.id: 1/len(actions) for action in actions}
    regret_weights = {action_id: weight/total_regret for action_id, weight in regret_weights.items()}

    # Map actions to weekdays
    weekday_mapping = _map_actions_to_weekdays(actions, year, month, num_days)

    # Compute hybrid regret-weighted strategy
    hybrid_strategy = {}
    for action in actions:
        weekday = weekday_mapping[action.id]
        hybrid_strategy[action.id] = regret_weights[action.id] * (1 - (weekday / 7))

    return hybrid_strategy

def compute_geometric_product_voronoi_partition(
    points: list[tuple[float, float]],
    seeds: list[tuple[float, float]],
) -> dict[tuple[float, float], tuple[float, float]]:
    """
    Compute Voronoi partition using geometric product.
    """
    voronoi_partition = {}
    for point in points:
        min_distance = float('inf')
        nearest_seed = None
        for seed in seeds:
            # Compute geometric product
            blade_a = frozenset([0, 1])
            blade_b = frozenset([0])
            geometric_product, sign = _multiply_blades(blade_a, blade_b)
            distance = np.linalg.norm(np.array(point) - np.array(seed))
            if distance < min_distance:
                min_distance = distance
                nearest_seed = seed
        voronoi_partition[tuple(point)] = nearest_seed
    return voronoi_partition

def compute_hybrid_strategy(
    actions: list[MathAction],
    counterfactuals: list[MathCounterfactual],
    year: int,
    month: int,
    num_days: int,
    points: list[tuple[float, float]],
    seeds: list[tuple[float, float]],
) -> dict[str, float]:
    """
    Compute hybrid strategy by combining regret-weighted strategy and Voronoi partition.
    """
    regret_strategy = compute_hybrid_regret_weighted_strategy(
        actions, counterfactuals, year, month, num_days
    )
    voronoi_partition = compute_geometric_product_voronoi_partition(points, seeds)
    hybrid_strategy = {}
    for action in actions:
        point = (action.expected_value, action.cost)
        seed = voronoi_partition[point]
        hybrid_strategy[action.id] = regret_strategy[action.id] * np.linalg.norm(np.array(point) - np.array(seed))
    return hybrid_strategy

if __name__ == "__main__":
    # Smoke test
    actions = [
        MathAction("action1", 10.0, 2.0),
        MathAction("action2", 20.0, 3.0),
    ]
    counterfactuals = [
        MathCounterfactual("action1", 12.0),
        MathCounterfactual("action2", 22.0),
    ]
    points = [(1.0, 2.0), (3.0, 4.0)]
    seeds = [(0.0, 0.0), (5.0, 5.0)]
    hybrid_strategy = compute_hybrid_strategy(
        actions, counterfactuals, 2024, 1, 30, points, seeds
    )
    print(hybrid_strategy)