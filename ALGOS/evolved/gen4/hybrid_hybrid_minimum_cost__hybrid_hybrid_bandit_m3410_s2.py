# DARWIN HAMMER — match 3410, survivor 2
# gen: 4
# parent_a: hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s1.py (gen3)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py (gen2)
# born: 2026-05-29T23:49:51Z

"""
This module fuses the governing equations of hybrid_minimum_cost_tree_hybrid_hybrid_bandit_router_koopman_operator_m253_s1.py
and hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py. The mathematical bridge between the two parents lies in 
the use of a contextual bandit and a tree structure. The hybrid algorithm integrates the bandit's propensity and expected 
reward into the calculation of the minimum-cost tree score.

The key insight is that the bandit's propensity and expected reward can be used to weight the edges of the tree, 
allowing the algorithm to prefer certain paths based on their potential reward.

The hybrid algorithm uses a data-driven approach to learn the optimal weights for the tree structure, 
and then uses these weights to calculate the minimum-cost tree score.

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

def tree_cost(nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str, path_weight: float = 0.2, 
               bandit_actions: List[BanditAction] = []) -> float:
    """Calculate the minimum-cost tree score with bandit weights."""
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
        # Find the bandit action corresponding to this edge
        edge_action = next((ba for ba in bandit_actions if ba.action_id == f"{a},{b}"), None)
        if edge_action:
            # Use the bandit's propensity and expected reward to weight the edge
            edge_weight = edge_action.propensity * edge_action.expected_reward
        else:
            edge_weight = 1.0
        material += length(nodes[a], nodes[b]) * edge_weight
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
        stats = _POLICY.get(u.action_id, [0.0, 0.0])
        confidence_terms[u.action_id] = stats[0] / stats[1] if stats[1] else 0.0
    return confidence_terms

def hybrid_algorithm(nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str, 
                     bandit_updates: List[BanditUpdate], path_weight: float = 0.2) -> float:
    """Run the hybrid algorithm."""
    update_policy(bandit_updates)
    bandit_actions = [BanditAction(action_id=f"{a},{b}", 
                                    propensity=_reward(f"{a},{b}"), 
                                    expected_reward=_reward(f"{a},{b}"), 
                                    confidence_bound=calculate_bandit_confidence(bandit_updates).get(f"{a},{b}", 0.0), 
                                    algorithm="Hybrid") 
                      for a, b in edges]
    return tree_cost(nodes, edges, root, path_weight, bandit_actions)

if __name__ == "__main__":
    nodes = {"A": Point(0, 0), "B": Point(1, 0), "C": Point(0, 1)}
    edges = [("A", "B"), ("A", "C")]
    root = "A"
    bandit_updates = [BanditUpdate(context_id="context1", action_id="A,B", reward=1.0, propensity=0.5), 
                      BanditUpdate(context_id="context1", action_id="A,C", reward=2.0, propensity=0.3)]
    print(hybrid_algorithm(nodes, edges, root, bandit_updates))