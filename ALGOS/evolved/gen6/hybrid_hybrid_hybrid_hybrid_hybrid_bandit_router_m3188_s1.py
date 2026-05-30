# DARWIN HAMMER — match 3188, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_gini_coeffici_m1406_s1.py (gen5)
# parent_b: hybrid_bandit_router_poikilotherm_schoolf_m20_s5.py (gen1)
# born: 2026-05-29T23:48:20Z

"""
This module integrates the governing equations of 
'hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s2.py' 
and 'hybrid_bandit_router_poikilotherm_schoolf_m20_s5.py'. 
The mathematical bridge lies in the use of the Gini coefficient 
to inform the bandit decision-making process in the context 
of a growing minimum-cost tree. By evaluating the Gini coefficient 
of the edge costs at each step, we can leverage the bandit 
algorithm to guide the decision-making process in a way that 
balances exploration and exploitation.

The Gini coefficient is used to calculate the inequality 
of the edge costs. This inequality measure is then used 
to adjust the bandit algorithm's confidence bound, which 
in turn guides the decision to incorporate or not incorporate 
a higher-cost edge into the tree. The Schoolfield temperature 
model is used to simulate the effect of temperature on the 
developmental rate of the system.
"""

import numpy as np
import math
import random
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass(frozen=True)
class Point:
    """A point in 2D space."""
    x: float
    y: float

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a.x - b.x, a.y - b.y)

def gini_coefficient(values: List[float]) -> float:
    """Compute the Gini coefficient."""
    values = np.array(values)
    values = values.flatten()
    if np.isscalar(values):
        return 0.0
    values = np.sort(values)
    index = np.arange(1, values.shape[0]+1)
    n = values.shape[0]
    return ((np.sum((2 * index - n  - 1) * values)) / (n * np.sum(values)))

@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float

@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: Dict[str, List[float]] = {}

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
    """In‑place update of the global policy with a batch of observations."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

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

def hybrid_decision(nodes: Dict[str, Point],
                    edges: List[Tuple[str, str]],
                    root: str,
                    temperature: float) -> BanditAction:
    """
    Make a hybrid decision by combining the Gini coefficient 
    of edge costs and the bandit algorithm.
    """
    edge_costs = [length(nodes[u], nodes[v]) for u, v in edges]
    gini = gini_coefficient(edge_costs)
    confidence_bound = 1 / (1 + gini)
    action_id = "incorporate_edge"
    propensity = 1.0
    expected_reward = tree_cost(nodes, edges, root) * temperature
    return BanditAction(action_id, propensity, expected_reward, confidence_bound)

def developmental_rate(temp_k: float) -> float:
    """
    Compute the developmental rate using the Schoolfield temperature model.
    """
    R_CAL = 1.987  # cal mol^-1 K^-1
    K25 = 298.15  # reference temperature (25 °C) in Kelvin
    rho_25 = 1.0
    delta_h_activation = 12_000.0
    t_low = 283.15
    t_high = 307.15
    delta_h_low = -45_000.0
    delta_h_high = 65_000.0
    r_cal = R_CAL
    return rho_25 * math.exp((delta_h_activation / r_cal) * (1 / K25 - 1 / temp_k))

if __name__ == "__main__":
    nodes = {"A": Point(0, 0), "B": Point(1, 0), "C": Point(0, 1)}
    edges = [("A", "B"), ("A", "C")]
    root = "A"
    temperature = developmental_rate(298.15)
    decision = hybrid_decision(nodes, edges, root, temperature)
    print(decision)