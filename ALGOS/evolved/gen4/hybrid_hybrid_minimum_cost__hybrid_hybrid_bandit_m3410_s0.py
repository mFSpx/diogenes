# DARWIN HAMMER — match 3410, survivor 0
# gen: 4
# parent_a: hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s1.py (gen3)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py (gen2)
# born: 2026-05-29T23:49:51Z

"""
This module combines the topologies of the minimum-cost tree and hybrid bandit models
(PARENT ALGORITHM A: hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s1.py and
PARENT ALGORITHM B: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py) into a
single unified system. The bridge between these two structures is formed by the
Euclidean distance metric from the minimum-cost tree, which is used to modulate the
inflow rate (propensity) of the bandit actions.

The resulting hybrid system, called HybridCostBandit, uses the minimum-cost tree score
as a reward signal to update the bandit policy and the VRAM store. The VRAM store is
then used to modulate the learning rate of the bandit, creating a deeper feedback loop.
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
    """Result of a bandit decision."""
    action_id: str
    propensity: float            # interpreted as inflow rate
    expected_reward: float
    confidence_bound: float      # interpreted as outflow rate
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    """Observed reward for a given action."""
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
_STORE: Dict[str, float] = {}

def reset_policy() -> None:
    """Erase all learned statistics."""
    global _POLICY
    _POLICY.clear()

def reset_store() -> None:
    """Erase all VRAM store statistics."""
    global _STORE
    _STORE.clear()

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

def _store_update(action: str, reward: float) -> None:
    """Update the VRAM store based on the reward signal."""
    global _STORE
    if action in _STORE:
        _STORE[action] = _STORE[action] * 0.99 + reward
    else:
        _STORE[action] = reward

def _modulate_learning_rate(propensity: float, store: float) -> float:
    """Modulate the learning rate based on the VRAM store."""
    return propensity * store

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
        confidence_terms[u.action_id] = 1.0 / np.sqrt(_count(u.action_id))
    return confidence_terms

def hybrid_cost_bandit(updates: List[BanditUpdate], nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str) -> None:
    """Update the bandit policy and VRAM store based on the minimum-cost tree score."""
    score = tree_cost(nodes, edges, root)
    for u in updates:
        _store_update(u.action_id, score)
        _modulate_learning_rate(u.propensity, _STORE[u.action_id])
        update_policy([u])

def calculate_bandit_reward(updates: List[BanditUpdate]) -> Dict[str, float]:
    """Calculate the bandit reward based on the VRAM store."""
    rewards = {}
    for u in updates:
        rewards[u.action_id] = _STORE[u.action_id]
    return rewards

if __name__ == "__main__":
    # Smoke test
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0)}
    edges = [("A", "B"), ("B", "C"), ("A", "C")]
    root = "A"
    updates = [BanditUpdate("A", "A", 1.0, 1.0), BanditUpdate("B", "B", 2.0, 2.0)]
    reset_policy()
    reset_store()
    hybrid_cost_bandit(updates, nodes, edges, root)
    rewards = calculate_bandit_reward(updates)
    print(rewards)