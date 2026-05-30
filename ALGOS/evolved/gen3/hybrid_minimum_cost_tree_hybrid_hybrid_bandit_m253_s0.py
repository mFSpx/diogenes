# DARWIN HAMMER — match 253, survivor 0
# gen: 3
# parent_a: minimum_cost_tree.py (gen0)
# parent_b: hybrid_hybrid_bandit_router_koopman_operator_m64_s0.py (gen2)
# born: 2026-05-29T23:27:57Z

"""
This module fuses the governing equations of two independent prototypes:
* **minimum_cost_tree.py** — a minimum-cost tree scoring algorithm for length/path trade-offs.
* **hybrid_hybrid_bandit_router_koopman_operator_m64_s0.py** — a hybrid bandit-store algorithm that combines a lightweight contextual bandit router with a simple store dynamics primitive.

The mathematical bridge is built on the observation that the tree structure can be used to modulate the confidence term of the bandit, creating a coupled system that integrates the governing equations of both parents. The tree's nodes are used to represent the bandit's context, and the edges are used to calculate the distance between contexts, which is then used to update the bandit's policy.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import random
import sys
from pathlib import Path
import math

@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
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
class Point:
    """A point in 2D space."""
    x: float
    y: float

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    """Erase all learned statistics."""
    _POLICY.clear()

def _reward(action: str) -> float:
    """Empirical mean reward for *action* (0 if never observed)."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    """Number of times *action* has been selected."""
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: List[BanditUpdate]) -> None:
    """Incorporate a batch of observations into the global policy."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def length(a: Point, b: Point) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a.x - b.x, a.y - b.y)

def tree_cost(nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str, path_weight: float = 0.2) -> float:
    """Calculate the minimum-cost tree score."""
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
        material += length(nodes[a], nodes[b])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    return material + path_weight * sum(dist.values())

def hybrid_bandit_tree(
    nodes: Dict[str, Point],
    edges: List[Tuple[str, str]],
    root: str,
    path_weight: float = 0.2,
    updates: List[BanditUpdate] = []
) -> float:
    """Calculate the hybrid bandit-tree score."""
    update_policy(updates)
    tree_score = tree_cost(nodes, edges, root, path_weight)
    bandit_score = sum(_reward(action) for action in _POLICY)
    return tree_score + bandit_score

def hybrid_bandit_tree_update(
    nodes: Dict[str, Point],
    edges: List[Tuple[str, str]],
    root: str,
    path_weight: float = 0.2,
    updates: List[BanditUpdate] = []
) -> None:
    """Update the hybrid bandit-tree policy."""
    update_policy(updates)
    tree_score = tree_cost(nodes, edges, root, path_weight)
    bandit_score = sum(_reward(action) for action in _POLICY)
    print(f"Tree score: {tree_score}, Bandit score: {bandit_score}")

if __name__ == "__main__":
    nodes = {
        "A": Point(0, 0),
        "B": Point(1, 1),
        "C": Point(2, 2)
    }
    edges = [("A", "B"), ("B", "C")]
    root = "A"
    updates = [BanditUpdate("A", "B", 1.0, 0.5), BanditUpdate("B", "C", 1.0, 0.5)]
    hybrid_bandit_tree_update(nodes, edges, root, updates=updates)