# DARWIN HAMMER — match 249, survivor 3
# gen: 3
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# parent_b: hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s1.py (gen2)
# born: 2026-05-29T23:27:56Z

"""
Module for the hybrid minimum-cost tree and bandit-router-sketch-RLCT algorithm.

This module combines the minimum-cost tree algorithm from 'hybrid_minimum_cost_tree_bayes_update_m6_s2.py'
and the bandit-router-sketch-RLCT algorithm from 'hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s1.py'
by finding a mathematical interface between their structures. The minimum-cost tree algorithm uses a 
deterministic cost function to evaluate the cost of a tree, while the bandit-router-sketch-RLCT algorithm 
uses a probabilistic approach to estimate the expected reward of each action. The mathematical bridge 
between the two algorithms is the use of probabilistic weights to modify the deterministic cost function.

The hybrid algorithm uses the probabilistic weights from the bandit-router-sketch-RLCT algorithm to modify 
the deterministic cost function from the minimum-cost tree algorithm. This is achieved by using the 
expected reward of each action as a weight in the cost function.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import random

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Algorithm A – deterministic tree utilities
# ----------------------------------------------------------------------
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
    node_distances : dict mapping node → root‑to‑node distance
    """
    adj = {node: [] for node in nodes}
    edge_len = {}
    node_distances = {root: 0.0}
    
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        edge_len[(b, a)] = length(nodes[b], nodes[a])
    
    for node in nodes:
        if node != root:
            node_distances[node] = float('inf')
            for a, b in edges:
                if a == root and b == node:
                    node_distances[node] = edge_len[(a, b)]
                elif a == node and b == root:
                    node_distances[node] = edge_len[(a, b)]
                elif a == node and b in node_distances:
                    node_distances[node] = min(node_distances[node], node_distances[b] + edge_len[(a, b)])
                elif a in node_distances and b == node:
                    node_distances[node] = min(node_distances[node], node_distances[a] + edge_len[(a, b)])
    
    return adj, edge_len, node_distances


# ----------------------------------------------------------------------
# Algorithm B – bandit-router-sketch-RLCT utilities
# ----------------------------------------------------------------------
class BanditAction:
    def __init__(self, action_id: str, propensity: float, expected_reward: float, confidence_bound: float, algorithm: str):
        self.action_id = action_id
        self.propensity = propensity
        self.expected_reward = expected_reward
        self.confidence_bound = confidence_bound
        self.algorithm = algorithm

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: List['BanditUpdate']) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

class BanditUpdate:
    def __init__(self, context_id: str, action_id: str, reward: float, propensity: float):
        self.context_id = context_id
        self.action_id = action_id
        self.reward = reward
        self.propensity = propensity

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


# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    bandit_actions: List[BanditAction],
) -> float:
    """
    Evaluate the hybrid cost function.

    Returns
    -------
    cost : float
    """
    adj, edge_len, node_distances = tree_metrics(nodes, edges, root)
    cost = 0.0
    
    for a, b in edges:
        action_id = f"{a}->{b}"
        expected_reward = next((ba.expected_reward for ba in bandit_actions if ba.action_id == action_id), 0.0)
        cost += expected_reward * edge_len[(a, b)]
    
    for node in nodes:
        if node != root:
            action_id = f"{root}->{node}"
            expected_reward = next((ba.expected_reward for ba in bandit_actions if ba.action_id == action_id), 0.0)
            cost += expected_reward * node_distances[node]
    
    return cost


def hybrid_bandit_update(
    updates: List[BanditUpdate],
    bandit_actions: List[BanditAction],
) -> None:
    """
    Update the bandit actions using the hybrid update rule.
    """
    update_policy(updates)
    for ba in bandit_actions:
        ba.expected_reward = _reward(ba.action_id)


def hybrid_count_min_sketch(
    items: List[str],
    width: int = 64,
    depth: int = 4,
) -> List[List[int]]:
    """
    Count‑Min sketch of item frequencies.
    """
    return count_min_sketch(items, width, depth)


if __name__ == "__main__":
    nodes = {
        'A': (0.0, 0.0),
        'B': (1.0, 0.0),
        'C': (1.0, 1.0),
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    root = 'A'
    bandit_actions = [
        BanditAction('A->B', 0.5, 1.0, 0.1, 'algorithm'),
        BanditAction('B->C', 0.5, 1.0, 0.1, 'algorithm'),
        BanditAction('C->A', 0.5, 1.0, 0.1, 'algorithm'),
    ]
    print(hybrid_tree_cost(nodes, edges, root, bandit_actions))
    updates = [
        BanditUpdate('context', 'A->B', 1.0, 0.5),
        BanditUpdate('context', 'B->C', 1.0, 0.5),
        BanditUpdate('context', 'C->A', 1.0, 0.5),
    ]
    hybrid_bandit_update(updates, bandit_actions)
    items = ['item1', 'item2', 'item3']
    print(hybrid_count_min_sketch(items))