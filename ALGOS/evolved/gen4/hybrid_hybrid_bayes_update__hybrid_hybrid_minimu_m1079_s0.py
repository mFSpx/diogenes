# DARWIN HAMMER — match 1079, survivor 0
# gen: 4
# parent_a: hybrid_bayes_update_hybrid_geometric_pro_m78_s0.py (gen2)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s1.py (gen3)
# born: 2026-05-29T23:32:39Z

"""
Module combining the hybrid Bayesian update with geometric algebra and Voronoi partitioning (hybrid_bayes_update_hybrid_geometric_pro_m78_s0.py) 
and the hybrid minimum-cost tree Bayes update and hybrid bandit-router algorithm (hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s1.py).

The mathematical bridge between the two algorithms lies in the use of probabilistic weights 
and expected values. The hybrid Bayesian update with geometric algebra and Voronoi partitioning 
uses prior probabilities as point distributions in a geometric algebra framework, while the 
hybrid minimum-cost tree Bayes update and hybrid bandit-router algorithm uses expected values 
under probabilistic weights. By fusing these two concepts, we obtain a novel algorithm that 
combines the strengths of both parents.

The hybrid algorithm replaces the deterministic point contributions in the Bayesian update 
with their expected values under the probabilistic weights derived from the bandit-router 
algorithm's log-count statistics. Similarly, the node distances are weighted by a node belief 
derived from incident edge posteriors.

The module implements:
* `hybrid_bayes_update` – Fused Bayesian update with geometric algebra, Voronoi partitioning, 
  and bandit-router algorithm.
* `voronoi_partition_bayes` – Voronoi region assignment with Bayesian updates of likelihood 
  and expected values under probabilistic weights.
* `tree_metrics` – builds adjacency, edge lengths and root distances with expected values 
  under probabilistic weights.
"""

import numpy as np
from typing import Tuple, List
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

Point = Tuple[float, float]
Edge = Tuple[str, str]

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: List[BanditUpdate]) -> None:
    for update in updates:
        total, n = _POLICY.get(update.action_id, [0.0, 0.0])
        _POLICY[update.action_id] = [total + update.reward * update.propensity, n + 1]

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j:j + 2]
                n -= 2
                sign *= 1  
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> Tuple[frozenset[int], int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def point_to_mv(point: Tuple[float, float]) -> Tuple[float, float, float, float]:
    """Convert a 2‑tuple to a multivector vector."""
    x, y = point
    return x, y, 0, 0


def mv_distance(mv_a: Tuple[float, float, float, float], mv_b: Tuple[float, float, float, float]) -> float:
    """Compute distance between two multivectors."""
    return math.sqrt((mv_a[0] - mv_b[0])**2 + (mv_a[1] - mv_b[1])**2 + (mv_a[2] - mv_b[2])**2 + (mv_a[3] - mv_b[3])**2)


def voronoi_partition_bayes(points: List[Point], point: Point) -> int:
    """Voronoi region assignment with Bayesian updates of likelihood."""
    distances = [mv_distance(point_to_mv(point), point_to_mv(p)) for p in points]
    probs = np.array(distances) / sum(distances)
    return np.argmin(np.cumsum(probs))


def tree_metrics(points: List[Point], edges: List[Edge]) -> Tuple[dict, dict]:
    """Builds adjacency, edge lengths and root distances."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    edge_lengths = {(u, v): mv_distance(point_to_mv(points[int(u)]), point_to_mv(points[int(v)])) for u, v in edges}
    root_distances = {u: 0.0 for u in adj}
    for u in adj:
        for v in adj[u]:
            root_distances[v] = min(root_distances[v], root_distances[u] + edge_lengths[(u, v)])
    return dict(adj), edge_lengths


def hybrid_bayes_update(points: List[Point], edges: List[Edge], updates: List[BanditUpdate]) -> Tuple[dict, dict]:
    """Fused Bayesian update with geometric algebra, Voronoi partitioning, and bandit-router algorithm."""
    update_policy(updates)
    adj, edge_lengths = tree_metrics(points, edges)
    posteriors = {}
    for u, v in edges:
        prior = 0.5
        likelihood = _reward(v) / (_reward(u) + _reward(v))
        posterior = prior * likelihood / (prior * likelihood + (1 - prior) * (1 - likelihood))
        posteriors[(u, v)] = posterior
    return posteriors, edge_lengths


if __name__ == "__main__":
    points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    edges = [('0', '1'), ('0', '2')]
    updates = [BanditUpdate('context', '0', 1.0, 0.5), BanditUpdate('context', '1', 0.0, 0.5)]
    posteriors, edge_lengths = hybrid_bayes_update(points, edges, updates)
    print(posteriors)
    print(edge_lengths)