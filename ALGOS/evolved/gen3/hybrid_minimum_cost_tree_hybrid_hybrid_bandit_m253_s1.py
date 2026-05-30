# DARWIN HAMMER — match 253, survivor 1
# gen: 3
# parent_a: minimum_cost_tree.py (gen0)
# parent_b: hybrid_hybrid_bandit_router_koopman_operator_m64_s0.py (gen2)
# born: 2026-05-29T23:27:57Z

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

def calculate_bandit_confidence(updates: List[BanditUpdate]) -> Dict[str, float]:
    """Calculate the confidence term for the bandit."""
    confidence_terms = {}
    for u in updates:
        count = _count(u.action_id)
        if count > 0:
            confidence_terms[u.action_id] = 1 / math.sqrt(count)
    return confidence_terms

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
    confidence_terms = calculate_bandit_confidence(updates)
    bandit_score = sum(_reward(action) + confidence_terms.get(action, 0) for action in _POLICY)
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
    confidence_terms = calculate_bandit_confidence(updates)
    bandit_score = sum(_reward(action) + confidence_terms.get(action, 0) for action in _POLICY)
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