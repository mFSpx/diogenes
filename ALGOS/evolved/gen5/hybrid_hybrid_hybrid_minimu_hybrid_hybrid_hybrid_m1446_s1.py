# DARWIN HAMMER — match 1446, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s5.py (gen4)
# born: 2026-05-29T23:36:21Z

"""
Module for the hybrid algorithm that fuses the minimum-cost tree and bandit-router-sketch-RLCT 
structures from 'hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s3.py' and 
'hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s5.py'. 

The mathematical bridge between the two algorithms is found by modifying the 
deterministic cost function from the minimum-cost tree algorithm using the 
probabilistic weights from the bandit-router-sketch-RLCT algorithm. Specifically, 
the expected reward of each action from the bandit-router-sketch-RLCT algorithm 
is used as a weight in the cost function. This allows the hybrid algorithm to 
leverage the strengths of both approaches: the deterministic optimization of the 
minimum-cost tree algorithm and the probabilistic exploration of the 
bandit-router-sketch-RLCT algorithm.

The governing equations of both parents are integrated through the use of 
probabilistic weights in the cost function. The matrix operations from the 
bandit-router-sketch-RLCT algorithm are used to compute the expected rewards, 
which are then used to modify the cost function from the minimum-cost tree 
algorithm.

"""

import math
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Callable, Sequence
from pathlib import Path
import sys

Vector = Sequence[float]
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

@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

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

def hybrid_cost_function(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    bandit_actions: List[BanditAction],
) -> Tuple[Dict[Edge, float], Dict[str, float]]:
    """
    Compute the hybrid cost function.

    Returns
    -------
    edge_cost : dict mapping edge (ordered a, b) → hybrid cost
    node_cost : dict mapping node → hybrid cost
    """
    adj, edge_len, node_distances = tree_metrics(nodes, edges, root)
    edge_cost = {}
    node_cost = {}

    for a, b in edges:
        # Compute the expected reward for the edge
        expected_reward = sum(
            action.expected_reward 
            for action in bandit_actions 
            if action.action_id in [a, b]
        ) / len(bandit_actions)

        # Compute the hybrid cost for the edge
        edge_cost[(a, b)] = edge_len[(a, b)] * (1 - expected_reward)
        edge_cost[(b, a)] = edge_len[(b, a)] * (1 - expected_reward)

    for node in nodes:
        # Compute the expected reward for the node
        expected_reward = sum(
            action.expected_reward 
            for action in bandit_actions 
            if action.action_id == node
        ) / len(bandit_actions)

        # Compute the hybrid cost for the node
        node_cost[node] = node_distances[node] * (1 - expected_reward)

    return edge_cost, node_cost

def hybrid_policy(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    bandit_actions: List[BanditAction],
) -> Dict[str, List[float]]:
    """
    Compute the hybrid policy.

    Returns
    -------
    policy : dict mapping node → list of probabilities
    """
    edge_cost, node_cost = hybrid_cost_function(nodes, edges, root, bandit_actions)
    policy = {}

    for node in nodes:
        # Compute the probabilities for the node
        probabilities = [
            math.exp(-edge_cost[(node, neighbor)]) 
            for neighbor in edge_cost 
            if node in edge_cost[(node, neighbor)]
        ]
        policy[node] = probabilities

    return policy

if __name__ == "__main__":
    nodes = {
        'A': (0, 0),
        'B': (1, 0),
        'C': (1, 1),
        'D': (0, 1)
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    root = 'A'

    bandit_actions = [
        BanditAction('A', 0.5, 0.8, 0.1, 'algorithm1'),
        BanditAction('B', 0.3, 0.6, 0.2, 'algorithm2'),
        BanditAction('C', 0.2, 0.4, 0.3, 'algorithm3'),
        BanditAction('D', 0.1, 0.2, 0.4, 'algorithm4')
    ]

    edge_cost, node_cost = hybrid_cost_function(nodes, edges, root, bandit_actions)
    policy = hybrid_policy(nodes, edges, root, bandit_actions)

    print("Edge Cost:", edge_cost)
    print("Node Cost:", node_cost)
    print("Policy:", policy)