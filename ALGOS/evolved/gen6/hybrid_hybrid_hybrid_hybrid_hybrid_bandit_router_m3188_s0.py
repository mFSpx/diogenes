# DARWIN HAMMER — match 3188, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_gini_coeffici_m1406_s1.py (gen5)
# parent_b: hybrid_bandit_router_poikilotherm_schoolf_m20_s5.py (gen1)
# born: 2026-05-29T23:48:15Z

"""
This module integrates the governing equations of 'hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s2.py' 
and 'hybrid_bandit_router_poikilotherm_schoolf_m20_s5.py'. The mathematical bridge lies in the use of 
the Gini coefficient to inform the Hoeffding bound in the decision to incorporate a higher-cost edge 
into the growing minimum-cost tree, while also leveraging the Schoolfield temperature model to adjust 
the propensity scores in the bandit router. By evaluating the Gini coefficient of the edge costs at 
each step, we can leverage the Hoeffding bound to guide the decision-making process in a way that balances 
exploration (via Hoeffding-UCB) and exploitation (via annealed acceptance), while also considering the 
temperature-dependent developmental rates.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Mapping, Hashable, Set
from functools import lru_cache

R_CAL = 1.987  # cal mol^-1 K^-1
K25 = 298.15  # reference temperature (25 °C) in Kelvin

@dataclass(frozen=True)
class Point:
    """A point in 2D space."""
    x: float
    y: float

@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""

    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""

    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL


_POLICY: Dict[str, List[float]] = {}

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a.x - b.x, a.y - b.y)

def tree_cost(nodes: Dict[str, Point],
              edges: List[Tuple[str, str]],
              root: str,
              path_weight: float = 0.2) -> float:
    """
    Compute the total cost of a tree:
      material = sum of edge lengths
      path_cost = weighted sum of distances from root to every node
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    material = 0
    for u, v in edges:
        material += length(nodes[u], nodes[v])

    path_cost = 0
    visited = set()
    stack = [(root, 0)]
    while stack:
        node, dist = stack.pop()
        if node not in visited:
            visited.add(node)
            path_cost += dist
            for neighbor in adj[node]:
                if neighbor not in visited:
                    stack.append((neighbor, dist + length(nodes[node], nodes[neighbor])))

    return material + path_weight * path_cost

def gini_coefficient(values: List[float]) -> float:
    """Gini coefficient of a list of values."""
    values = np.array(values)
    index = np.argsort(values)
    n = len(values)
    index = np.arange(1, n+1)
    return ((np.sum((2 * index - n - 1) * values[index-1])) / (n * np.sum(values)))

def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

@lru_cache(maxsize=None)
def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield temperature model."""
    t_low = params.t_low
    t_high = params.t_high
    delta_h_activation = params.delta_h_activation
    delta_h_low = params.delta_h_low
    delta_h_high = params.delta_h_high
    r_cal = params.r_cal

    return (params.rho_25 * np.exp(-delta_h_activation / (r_cal * temp_k)))

def reset_policy() -> None:
    """Clear all stored reward statistics."""
    _POLICY.clear()

def _reward(a: str) -> float:
    """Empirical mean reward for action *a* (0 if never tried)."""
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def _count(a: str) -> float:
    """Number of times action *a* has been observed."""
    return _POLICY.get(a, [0.0, 0.0])[1]

def update_policy(updates: List[BanditUpdate]) -> None:
    """In-place update of the global policy with a batch of observations."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def calculate_propensity(action_id: str, temp_k: float) -> float:
    """Calculate the propensity score based on the developmental rate."""
    rate = developmental_rate(temp_k)
    reward = _reward(action_id)
    count = _count(action_id)
    return reward * rate / count if count else 0.0

def decide_action(actions: List[str], temp_k: float) -> str:
    """Decide the action based on the propensity scores."""
    propensities = [calculate_propensity(a, temp_k) for a in actions]
    return actions[np.argmax(propensities)]

def evaluate_tree(nodes: Dict[str, Point],
                  edges: List[Tuple[str, str]],
                  root: str,
                  path_weight: float = 0.2,
                  temp_k: float = K25) -> float:
    """
    Evaluate the total cost of a tree, considering the temperature-dependent developmental rates.
    """
    material = tree_cost(nodes, edges, root, path_weight)
    propensity = calculate_propensity(root, temp_k)
    return material * propensity

if __name__ == "__main__":
    nodes = {"A": Point(0, 0), "B": Point(1, 0), "C": Point(0, 1)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    temp_k = c_to_k(25)
    
    print("Tree cost:", tree_cost(nodes, edges, root))
    print("Gini coefficient:", gini_coefficient([1, 2, 3, 4, 5]))
    print("Developmental rate:", developmental_rate(temp_k))
    print("Propensity score:", calculate_propensity("A", temp_k))
    print("Decided action:", decide_action(["A", "B", "C"], temp_k))
    print("Evaluated tree cost:", evaluate_tree(nodes, edges, root, temp_k=temp_k))