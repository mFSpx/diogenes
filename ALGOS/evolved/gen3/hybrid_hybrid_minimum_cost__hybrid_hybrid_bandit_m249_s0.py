# DARWIN HAMMER — match 249, survivor 0
# gen: 3
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# parent_b: hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s1.py (gen2)
# born: 2026-05-29T23:27:56Z

"""
Module for the hybrid minimum-cost tree Bayesian bandit-router algorithm.
This module combines the minimum-cost tree Bayesian update algorithm from 'hybrid_minimum_cost_tree_bayes_update_m6_s2.py'
and the hybrid bandit-router and sketch-RLCT algorithm from 'hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s1.py'
by finding a mathematical interface between their structures. The minimum-cost tree Bayesian update algorithm uses
a deterministic cost function with probabilistic weights, while the hybrid bandit-router and sketch-RLCT algorithm
uses a Count-Min sketch to estimate the empirical log-likelihood sum and the effective number of activation patterns.
The mathematical bridge between the two algorithms is the use of log-count statistics to estimate the expected reward
of each action, and the probabilistic weights to modify the cost function.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

# Types
Point = Tuple[float, float]
Edge = Tuple[str, str]

# Algorithm A – deterministic tree utilities
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
    edge_len : dict mapping edge (ordered a, b) → length
    node_dist : dict mapping node → root‑to‑node distance
    """
    adj = defaultdict(list)
    edge_len = {}
    node_dist = {}

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        edge_len[(b, a)] = length(nodes[b], nodes[a])

    # Compute root-to-node distances using BFS
    queue = [(root, 0)]
    visited = set()
    while queue:
        node, dist = queue.pop(0)
        if node not in visited:
            visited.add(node)
            node_dist[node] = dist
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))

    return dict(adj), edge_len, node_dist

# Algorithm B – bandit-router utilities
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
            idx = int(hash(item) % width)
            table[d][idx] += 1
    return table

# Hybrid algorithm
def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    actions: List[BanditAction],
    updates: List[BanditUpdate],
    lambda_: float,
) -> float:
    """
    Evaluate the hybrid cost function using the posteriors.

    Returns
    -------
    cost : float
    """
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    update_policy(updates)
    expected_rewards = {action.action_id: _reward(action.action_id) for action in actions}

    # Compute the expected edge lengths
    expected_edge_len = {}
    for a, b in edge_len:
        expected_edge_len[(a, b)] = expected_rewards[a] * edge_len[(a, b)]

    # Compute the hybrid cost
    cost = 0
    for a, b in edge_len:
        cost += expected_edge_len[(a, b)]
    cost += lambda_ * sum(node_dist.values())

    return cost

def hybrid_bandit_update(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    actions: List[BanditAction],
    updates: List[BanditUpdate],
) -> List[BanditAction]:
    """
    Update the bandit actions using the hybrid algorithm.

    Returns
    -------
    updated_actions : List[BanditAction]
    """
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    update_policy(updates)
    expected_rewards = {action.action_id: _reward(action.action_id) for action in actions}

    # Update the bandit actions
    updated_actions = []
    for action in actions:
        updated_reward = expected_rewards[action.action_id]
        updated_actions.append(
            BanditAction(
                action_id=action.action_id,
                propensity=action.propensity,
                expected_reward=updated_reward,
                confidence_bound=action.confidence_bound,
                algorithm=action.algorithm,
            )
        )

    return updated_actions

if __name__ == "__main__":
    nodes = {
        "A": (0, 0),
        "B": (1, 0),
        "C": (1, 1),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    actions = [
        BanditAction("A", 0.5, 1.0, 0.1, "algorithm1"),
        BanditAction("B", 0.5, 1.0, 0.1, "algorithm1"),
        BanditAction("C", 0.5, 1.0, 0.1, "algorithm1"),
    ]
    updates = [
        BanditUpdate("context1", "A", 1.0, 0.5),
        BanditUpdate("context2", "B", 1.0, 0.5),
        BanditUpdate("context3", "C", 1.0, 0.5),
    ]

    cost = hybrid_tree_cost(nodes, edges, root, actions, updates, 0.1)
    updated_actions = hybrid_bandit_update(nodes, edges, root, actions, updates)

    print("Hybrid cost:", cost)
    print("Updated actions:")
    for action in updated_actions:
        print(action)