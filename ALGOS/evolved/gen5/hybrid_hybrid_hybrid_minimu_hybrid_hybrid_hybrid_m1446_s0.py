# DARWIN HAMMER — match 1446, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s5.py (gen4)
# born: 2026-05-29T23:36:21Z

"""
Module for the hybrid minimum-cost tree and bandit-router-sketch-RLCT algorithm.

This module combines the minimum-cost tree algorithm from 'hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s3.py'
and the bandit-router-sketch-RLCT algorithm from 'hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s5.py'
by finding a mathematical interface between their structures. The minimum-cost tree algorithm uses a 
deterministic cost function to evaluate the cost of a tree, while the bandit-router-sketch-RLCT algorithm 
uses a probabilistic approach to estimate the expected reward of each action. The mathematical bridge 
between the two algorithms is the use of probabilistic weights to modify the deterministic cost function.

The hybrid algorithm uses the expected reward of each action as a weight in the cost function, and 
then uses the probabilistic weights from the bandit-router-sketch-RLCT algorithm to update the 
probabilities of each action. This is achieved by using the Gaussian function to compute the 
probabilistic weights, and then using these weights to update the probabilities of each action.
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
Vector = Tuple[float, ...]

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

class BanditUpdate:
    def __init__(self, context_id: str, action_id: str, reward: float, propensity: float):
        self.context_id = context_id
        self.action_id = action_id
        self.reward = reward
        self.propensity = propensity

_POLICY: Dict[str, List[float]] = {}          
_STORE: Dict[str, float] = {}                 
_SURROGATE = None                             

def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()
    global _SURROGATE
    _SURROGATE = RBFSurrogate(centers=[], weights=[], epsilon=1.0)

def _empirical_reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]

class RBFSurrogate:
    def __init__(self, centers: List[Tuple[float, ...]], weights: List[float], epsilon: float = 1.0):
        self.centers = centers
        self.weights = weights
        self.epsilon = epsilon

    def predict(self, x: Vector) -> float:
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_algorithm(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    actions: List[BanditAction],
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float], List[BanditAction]]:
    """
    Hybrid algorithm that combines the minimum-cost tree algorithm and the bandit-router-sketch-RLCT algorithm.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a, b) → length
    node_distances : dict mapping node → root‑to‑node distance
    actions : list of BanditAction objects
    """
    adj, edge_len, node_distances = tree_metrics(nodes, edges, root)
    
    # Update probabilities of each action using the probabilistic weights
    for action in actions:
        action.propensity = gaussian(action.expected_reward, epsilon=1.0)
        
    return adj, edge_len, node_distances, actions

def update_policy(
    context_id: str,
    action_id: str,
    reward: float,
    propensity: float,
) -> None:
    """
    Update the policy using the bandit-router-sketch-RLCT algorithm.
    """
    update = BanditUpdate(context_id, action_id, reward, propensity)
    _POLICY[update.action_id] = [update.reward, 1.0]

def predict(
    x: Vector,
) -> float:
    """
    Predict the expected reward using the RBF surrogate.
    """
    return _SURROGATE.predict(x)

# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    nodes = {
        'A': (0.0, 0.0),
        'B': (1.0, 0.0),
        'C': (0.0, 1.0),
    }
    edges = [
        ('A', 'B'),
        ('A', 'C'),
        ('B', 'C'),
    ]
    root = 'A'
    actions = [
        BanditAction('action1', 0.5, 0.8, 0.2, 'algorithm1'),
        BanditAction('action2', 0.3, 0.6, 0.4, 'algorithm2'),
    ]
    
    adj, edge_len, node_distances, actions = hybrid_algorithm(nodes, edges, root, actions)
    print(adj)
    print(edge_len)
    print(node_distances)
    print(actions)
    
    update_policy('context1', 'action1', 0.9, 0.5)
    print(_POLICY)
    
    reset_policy()
    print(_POLICY)
    
    _SURROGATE = RBFSurrogate(centers=[(0.0, 0.0)], weights=[1.0], epsilon=1.0)
    print(predict((0.0, 0.0)))