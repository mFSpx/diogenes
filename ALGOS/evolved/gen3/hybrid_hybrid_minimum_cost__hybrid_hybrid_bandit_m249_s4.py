# DARWIN HAMMER — match 249, survivor 4
# gen: 3
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# parent_b: hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s1.py (gen2)
# born: 2026-05-29T23:27:56Z

"""
Module combining hybrid_minimum_cost_tree_bayes_update_m6_s2.py and 
hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s1.py through a log-posterior 
interface. The Minimum-Cost Tree scoring and Bayesian evidence update are fused 
with the bandit-router and sketch-RLCT algorithm by replacing the deterministic 
edge contribution with its expected value under the posterior edge belief.

The mathematical bridge between the two algorithms is the use of log-posterior 
statistics. The Minimum-Cost Tree scoring and Bayesian evidence update use 
the log-posterior statistics to compute the expected cost, while the 
bandit-router and sketch-RLCT algorithm use the log-posterior statistics 
to estimate the expected reward and the effective number of activation patterns.

The hybrid replaces the deterministic edge contribution ℓ(e) in (1) by its 
**expected** value under the posterior edge belief *p_e* obtained from (2).  
Similarly, node distances are weighted by a node belief *q_v* derived from incident 
edge posteriors and the log-count statistics from the bandit-router algorithm.  
The resulting hybrid cost is

    C_h = Σ_e p_e·ℓ(e) + λ Σ_v q_v·d(v)                (3)

and the hybrid reward is

    R_h = Σ_a q_a·r(a)                                (4)
"""

import math
import random
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
from collections import defaultdict

Point = Tuple[float, float]
Edge = Tuple[str, str]
BanditAction = Tuple[str, float, float, float]

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a
    """
    adj = defaultdict(list)
    edge_len = {}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_len[(u, v)] = length(nodes[u], nodes[v])
        edge_len[(v, u)] = edge_len[(u, v)]
    distances = {root: 0.0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for neighbour in adj[node]:
            if neighbour not in distances:
                distances[neighbour] = distances[node] + 1.0
                queue.append(neighbour)
    return dict(adj), edge_len, distances

def bayes_edge_posteriors(
    edges: List[Edge], 
    prior: float, 
    likelihood: float, 
    false_positive_rate: float
) -> Dict[Edge, float]:
    """
    Compute posterior edge beliefs.

    Returns
    -------
    posteriors : dict mapping edge → posterior belief
    """
    posteriors = {}
    for u, v in edges:
        p_post = (prior * likelihood) / (likelihood * prior + false_positive_rate * (1 - prior))
        posteriors[(u, v)] = p_post
        posteriors[(v, u)] = p_post
    return posteriors

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
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def count_min_sketch(
    items: List[str], width: int = 64, depth: int = 4
) -> List[List[int]]:
    """Count‑Min sketch of item frequencies."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = hash(item) % width
            table[d][idx] += 1
    return table

def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    prior: float, 
    likelihood: float, 
    false_positive_rate: float,
    updates: List[BanditUpdate]
) -> Tuple[float, Dict[Edge, float], Dict[str, float]]:
    adj, edge_len, distances = tree_metrics(nodes, edges, root)
    posteriors = bayes_edge_posteriors(edges, prior, likelihood, false_positive_rate)
    update_policy(updates)
    log_count_stats = count_min_sketch([u.action_id for u in updates])
    hybrid_cost = 0.0
    weighted_distances = {}
    for node in distances:
        weighted_distances[node] = distances[node] * np.mean(log_count_stats)
    for u, v in edges:
        hybrid_cost += posteriors[(u, v)] * edge_len[(u, v)]
    for node in weighted_distances:
        hybrid_cost += weighted_distances[node]
    return hybrid_cost, posteriors, weighted_distances

def hybrid_reward(updates: List[BanditUpdate]) -> float:
    rewards = []
    for u in updates:
        rewards.append(u.reward * _reward(u.action_id))
    return np.mean(rewards)

if __name__ == "__main__":
    nodes = {
        'A': (0.0, 0.0),
        'B': (1.0, 0.0),
        'C': (1.0, 1.0),
        'D': (0.0, 1.0)
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    root = 'A'
    prior = 0.5
    likelihood = 0.8
    false_positive_rate = 0.2
    updates = [BanditUpdate('context1', 'action1', 1.0, 0.5), 
               BanditUpdate('context1', 'action2', 0.0, 0.5)]
    hybrid_cost, posteriors, weighted_distances = hybrid_tree_cost(nodes, edges, root, prior, likelihood, false_positive_rate, updates)
    hybrid_reward_val = hybrid_reward(updates)
    print(f"Hybrid cost: {hybrid_cost}")
    print(f"Hybrid reward: {hybrid_reward_val}")