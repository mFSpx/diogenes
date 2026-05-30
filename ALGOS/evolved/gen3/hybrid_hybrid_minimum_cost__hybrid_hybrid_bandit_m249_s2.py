# DARWIN HAMMER — match 249, survivor 2
# gen: 3
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# parent_b: hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s1.py (gen2)
# born: 2026-05-29T23:27:56Z

"""
Module for the hybrid minimum-cost tree Bayes update and bandit-router sketch-RLCT algorithm.

This module combines the minimum-cost tree Bayes update algorithm from 'hybrid_minimum_cost_tree_bayes_update_m6_s2.py'
and the bandit-router sketch-RLCT algorithm from 'hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s1.py' by finding a 
mathematical interface between their structures. The minimum-cost tree Bayes update algorithm uses a deterministic cost 
function with probabilistic weights, while the bandit-router sketch-RLCT algorithm uses log-count statistics to 
estimate the expected reward of each action. The combined quantities feed the free-energy asymptotic and the RLCT 
regression.

The mathematical bridge between the two algorithms is the use of probabilistic weights and log-count statistics. The 
minimum-cost tree Bayes update algorithm uses probabilistic weights to compute the expected cost, while the 
bandit-router sketch-RLCT algorithm uses log-count statistics to estimate the expected reward. By combining these two 
approaches, we can create a hybrid algorithm that uses probabilistic weights and log-count statistics to compute the 
expected cost and the expected reward.
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
    edge_len : dict mapping edge (ordered a, b) → Euclidean length
    root_dist : dict mapping node → root‑to‑node distance
    """
    adj = defaultdict(list)
    edge_len = {}
    root_dist = {}

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        edge_len[(b, a)] = length(nodes[b], nodes[a])

    root_dist[root] = 0.0
    stack = [root]
    while stack:
        node = stack.pop()
        for neighbour in adj[node]:
            if neighbour not in root_dist:
                root_dist[neighbour] = root_dist[node] + edge_len[(node, neighbour)]
                stack.append(neighbour)

    return dict(adj), edge_len, root_dist

# Algorithm B – bandit-router sketch-RLCT utilities
_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: List[Dict]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u['action_id'], [0.0, 0.0])
        stats[0] += float(u['reward'])
        stats[1] += 1.0

def count_min_sketch(
    items: List[str], width: int = 64, depth: int = 4
) -> List[List[int]]:
    """Count‑Min sketch of item frequencies."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hash(item) % (width * depth)) % width
            table[d][idx] += 1
    return table

# Hybrid minimum-cost tree Bayes update and bandit-router sketch-RLCT algorithm
def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    updates: List[Dict]
) -> float:
    """
    Evaluates the hybrid cost using the posteriors and the expected reward.

    Returns
    -------
    C_h : float
        The hybrid cost
    """
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    update_policy(updates)

    C_h = 0.0
    for a, b in edges:
        p_e = _reward(a) / (_count(a) + _count(b))
        C_h += p_e * edge_len[(a, b)]

    for node in nodes:
        q_v = _reward(node) / (_count(node) + 1.0)
        C_h += q_v * root_dist[node]

    return C_h

def hybrid_log_count_sketch(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    updates: List[Dict]
) -> List[List[int]]:
    """
    Evaluates the log-count sketch using the expected reward.

    Returns
    -------
    table : List[List[int]]
        The log-count sketch
    """
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    update_policy(updates)

    items = [node for node in nodes]
    table = count_min_sketch(items)

    return table

def hybrid_policy_update(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    updates: List[Dict]
) -> Dict[str, List[float]]:
    """
    Updates the policy using the hybrid tree cost and the log-count sketch.

    Returns
    -------
    policy : Dict[str, List[float]]
        The updated policy
    """
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    C_h = hybrid_tree_cost(nodes, edges, root, updates)
    table = hybrid_log_count_sketch(nodes, edges, root, updates)

    for node in nodes:
        stats = _POLICY.setdefault(node, [0.0, 0.0])
        stats[0] += C_h
        stats[1] += 1.0

    return _POLICY

if __name__ == "__main__":
    nodes = {
        'A': (0.0, 0.0),
        'B': (1.0, 0.0),
        'C': (1.0, 1.0),
        'D': (0.0, 1.0),
    }
    edges = [
        ('A', 'B'),
        ('B', 'C'),
        ('C', 'D'),
        ('D', 'A'),
    ]
    root = 'A'
    updates = [
        {'action_id': 'A', 'reward': 1.0},
        {'action_id': 'B', 'reward': 2.0},
        {'action_id': 'C', 'reward': 3.0},
        {'action_id': 'D', 'reward': 4.0},
    ]

    C_h = hybrid_tree_cost(nodes, edges, root, updates)
    table = hybrid_log_count_sketch(nodes, edges, root, updates)
    policy = hybrid_policy_update(nodes, edges, root, updates)

    print(C_h)
    print(table)
    print(policy)