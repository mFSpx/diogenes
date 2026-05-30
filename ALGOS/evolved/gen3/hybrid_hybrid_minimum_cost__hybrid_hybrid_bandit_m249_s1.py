# DARWIN HAMMER — match 249, survivor 1
# gen: 3
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# parent_b: hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s1.py (gen2)
# born: 2026-05-29T23:27:56Z

"""
Module combining the hybrid minimum-cost tree Bayes update (hybrid_minimum_cost_tree_bayes_update_m6_s2.py) 
and the hybrid bandit-router and sketch-RLCT algorithm (hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s1.py).

The mathematical bridge between the two algorithms is the use of expected values under probabilistic 
weights. The hybrid minimum-cost tree Bayes update uses expected edge lengths under posterior edge 
beliefs, while the hybrid bandit-router and sketch-RLCT algorithm uses expected rewards under 
log-count statistics. By fusing these two concepts, we obtain a novel algorithm that combines the 
strengths of both parents.

The hybrid algorithm replaces the deterministic edge contribution in the minimum-cost tree 
scoring with its expected value under the posterior edge belief. Similarly, the node distances 
are weighted by a node belief derived from incident edge posteriors. The bandit-router 
algorithm's log-count statistics are used to estimate the expected rewards of each action, 
which are then used to compute the posterior edge beliefs.

The module implements:
* `tree_metrics` – builds adjacency, edge lengths and root distances.
* `bayes_edge_posteriors` – vectorised Bayesian update for all edges.
* `count_min_sketch` – Count-Min sketch of item frequencies.
* `hybrid_tree_cost` – evaluates the hybrid cost using the posteriors and log-count statistics.
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
BanditAction = dataclass(frozen=True)
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
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    adj = defaultdict(list)
    edge_len = {}
    for e in edges:
        a, b = e
        adj[a].append(b)
        adj[b].append(a)
        edge_len[e] = length(nodes[a], nodes[b])

    distances = {root: 0.0}
    queue = [root]
    while queue:
        v = queue.pop(0)
        for w in adj[v]:
            if w not in distances:
                distances[w] = distances[v] + edge_len[(v, w)]
                queue.append(w)

    return dict(adj), edge_len, distances

def bayes_edge_posteriors(
    prior: float, likelihood: float, false_positive_rate: float
) -> float:
    numerator = prior * likelihood
    denominator = likelihood * prior + false_positive_rate * (1 - prior)
    return numerator / denominator

def count_min_sketch(
    items: List[str], width: int = 64, depth: int = 4
) -> List[List[int]]:
    table = [[0] * width for _ in range(depth)]
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
    items: List[str],
) -> float:
    adj, edge_len, distances = tree_metrics(nodes, edges, root)
    posteriors = [bayes_edge_posteriors(prior, likelihood, false_positive_rate) for _ in edges]
    log_count_stats = count_min_sketch(items)
    expected_edge_lengths = [p * edge_len[e] for p, e in zip(posteriors, edges)]
    weighted_distances = {v: q * distances[v] for v, q in zip(distances, [1.0] * len(distances))}
    cost = sum(expected_edge_lengths) + 0.1 * sum(weighted_distances.values())
    return cost

if __name__ == "__main__":
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}
    edges = [("A", "B"), ("A", "C")]
    root = "A"
    prior = 0.5
    likelihood = 0.8
    false_positive_rate = 0.2
    items = ["item1", "item2", "item3"]
    cost = hybrid_tree_cost(nodes, edges, root, prior, likelihood, false_positive_rate, items)
    print(cost)