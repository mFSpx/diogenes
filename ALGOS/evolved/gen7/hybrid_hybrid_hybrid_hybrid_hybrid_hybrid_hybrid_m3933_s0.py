# DARWIN HAMMER — match 3933, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1088_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1203_s4.py (gen6)
# born: 2026-05-29T23:52:35Z

"""
Hybrid module combining the sinusoidal rotation and matrix operations from 
'hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1088_s0.py' and the regret-weighted Hoeffding tree and bandit 
with Gini-modulated confidence from 'hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1203_s4.py'. 
The mathematical bridge lies in applying the sinusoidal rotation to the probability distribution of 
Voronoi partitions obtained from the geometric product, and then using these probabilities to 
weight the regret-weighted Hoeffding tree and bandit selection.

The mathematical interface is established by representing the Voronoi partitions as a probability 
distribution, and then applying the sinusoidal rotation to this distribution. 
The resulting probabilities are then used to weight the regret-weighted Hoeffding tree and bandit selection.
"""

import numpy as np
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

Point = tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def weekday_weight_vector(groups: list[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    Sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow / 7.0)
    weight_vec = 1.0 + 0.5 * np.sin(base_angles + phase)
    return weight_vec / np.sum(weight_vec)

def geometric_product(seeds: list[Point], points: list[Point]) -> dict[int, list[Point]]:
    regions = assign(points, seeds)
    return regions

def gini_coefficient(values: List[float]) -> float:
    """
    Compute the Gini coefficient for a list of values.

    Args:
    values (List[float]): A list of values.

    Returns:
    float: The Gini coefficient.
    """
    values = np.array(values)
    if values.size == 0:
        return 0.0
    idx = values.argsort()
    n = len(values)
    index = np.arange(1, n+1)
    n = np.float64(n)
    return ((np.sum((2 * index - n  - 1) * values[idx])) / (n * np.sum(values)))

def hybrid_operation(seeds: list[Point], points: list[Point], dow: int) -> dict[int, list[Point]]:
    regions = geometric_product(seeds, points)
    groups = [f"seed_{i}" for i in range(len(seeds))]
    weight_vec = weekday_weight_vector(groups, dow)
    weighted_regions = {}
    for i, region in regions.items():
        weighted_regions[i] = region * weight_vec[i]
    return weighted_regions

@dataclass(frozen=True)
class MathAction:
    """Action used in the regret‑weighted Hoeffding tree."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class BanditAction:
    """Bandit arm with propensity‑adjusted confidence bound."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"

def modulate_confidence_bound(confidence_bound: float, gini_coefficient: float, lambda_g: float) -> float:
    """
    Modulate the confidence bound using the Gini coefficient.

    Args:
    confidence_bound (float): The original confidence bound.
    gini_coefficient (float): The Gini coefficient.
    lambda_g (float): The modulation parameter.

    Returns:
    float: The modulated confidence bound.
    """
    return confidence_bound * (1 + lambda_g * gini_coefficient)

def hybrid_bandit_selection(actions: List[MathAction], lambda_g: float, T: float) -> BanditAction:
    """
    Perform hybrid bandit selection using the modulated confidence bound.

    Args:
    actions (List[MathAction]): A list of actions.
    lambda_g (float): The modulation parameter.
    T (float): The temperature.

    Returns:
    BanditAction: The selected bandit action.
    """
    # Compute Gini coefficient for action values
    action_values = [action.expected_value for action in actions]
    gini_coef = gini_coefficient(action_values)

    # Modulate confidence bounds
    modulated_actions = []
    for action in actions:
        confidence_bound = 1 / math.sqrt(len(actions))
        modulated_confidence_bound = modulate_confidence_bound(confidence_bound, gini_coef, lambda_g)
        modulated_actions.append(BanditAction(action.id, 1.0, action.expected_value, modulated_confidence_bound))

    # Select action with highest expected reward and modulated confidence bound
    selected_action = max(modulated_actions, key=lambda action: action.expected_reward / action.confidence_bound)
    return selected_action

if __name__ == "__main__":
    seeds = [(0, 0), (1, 1), (2, 2)]
    points = [(0.5, 0.5), (1.5, 1.5), (2.5, 2.5)]
    dow = 3
    hybrid_regions = hybrid_operation(seeds, points, dow)
    print(hybrid_regions)

    actions = [MathAction("action1", 10.0), MathAction("action2", 20.0), MathAction("action3", 30.0)]
    lambda_g = 0.5
    T = 1.0
    selected_action = hybrid_bandit_selection(actions, lambda_g, T)
    print(selected_action)