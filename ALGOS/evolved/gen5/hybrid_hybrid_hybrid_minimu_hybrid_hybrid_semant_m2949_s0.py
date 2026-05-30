# DARWIN HAMMER — match 2949, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_minimum_cost__hybrid_jepa_energy_h_m1737_s1.py (gen4)
# parent_b: hybrid_hybrid_semantic_neig_hybrid_perceptual_de_m806_s0.py (gen3)
# born: 2026-05-29T23:46:58Z

"""
Hybrid Bandit Tree with Semantic-Morphology & Perceptual-Privacy System

Parents:
- hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m1737_s1.py (bandit tree with minimum cost)
- hybrid_hybrid_semantic_neig_hybrid_perceptual_de_m806_s0.py (semantic-morphology & perceptual-privacy system)

Mathematical Bridge:
The reconstruction-risk score *r* (0-1) derived from the Count-Min sketch is used to modulate the morphology-derived recovery priority *p* before it is combined with the semantic cosine similarity *c*. 
The unified hybrid neighbor score is integrated with the bandit tree's cost calculation, resulting in a new cost function that balances pure semantic closeness against a privacy-aware physical robustness.

    h(i, j) = α·c(v_i,v_j) + (1-α)·p(m_j)·(1-r)
    tree_cost = material + self.path_weight * sum(dist.values()) + β * h(i, j)
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

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

@dataclass(frozen=True)
class Point:
    x: float
    y: float

class Morphology:
    __slots__ = ("length", "width", "height", "mass")

    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalised priority in [0,1] derived from morphology."""
    return min(1.0, max(0.0, righting_time_index(m) / max_index))

class HybridBanditTree:
    def __init__(self, dp_epsilon: float = 0.1, path_weight: float = 0.2, alpha: float = 0.5, beta: float = 0.5):
        self._policy: Dict[str, List[float]] = {}
        self.dp_epsilon = dp_epsilon
        self.path_weight = path_weight
        self.alpha = alpha
        self.beta = beta

    def reset_policy(self) -> None:
        self._policy.clear()

    def _reward(self, action: str) -> float:
        total, n = self._policy.get(action, [0.0, 0.0])
        return total / n if n else 0.0

    def _count(self, action: str) -> float:
        return self._policy.get(action, [0.0, 0.0])[1]

    def update_policy(self, updates: List[BanditUpdate]) -> None:
        for u in updates:
            stats = self._policy.setdefault(u.action_id, [0.0, 0.0])
            stats[0] += float(u.reward)
            stats[1] += 1.0

    def length(self, a: Point, b: Point) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def tree_cost(self, nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str, morphology: Dict[str, Morphology], updates: List[BanditUpdate] = []) -> float:
        self.update_policy(updates)
        adj: Dict[str, List[str]] = {n: [] for n in nodes}
        material = 0.0
        for a, b in edges:
            adj[a].append(b); adj[b].append(a)
            material += self.length(nodes[a], nodes[b])
        dist = {root: 0.0}
        stack = [root]
        while stack:
            a = stack.pop()
            for b in adj[a]:
                if b not in dist:
                    dist[b] = dist[a] + self.length(nodes[a], nodes[b])
                    stack.append(b)
        
        semantic_cost = 0.0
        for a, b in edges:
            semantic_similarity = self.semantic_similarity(nodes[a], nodes[b])
            recovery_priority_b = recovery_priority(morphology[b])
            reconstruction_risk = self.reconstruction_risk(nodes[a], nodes[b])
            semantic_cost += self.beta * (self.alpha * semantic_similarity + (1 - self.alpha) * recovery_priority_b * (1 - reconstruction_risk))
        
        return material + self.path_weight * sum(dist.values()) + semantic_cost

    def semantic_similarity(self, a: Point, b: Point) -> float:
        # Simple cosine similarity for demonstration purposes
        return np.dot([a.x, a.y], [b.x, b.y]) / (np.linalg.norm([a.x, a.y]) * np.linalg.norm([b.x, b.y]))

    def reconstruction_risk(self, a: Point, b: Point) -> float:
        # Simple reconstruction risk for demonstration purposes
        return 0.5

def main():
    nodes = {"A": Point(0, 0), "B": Point(3, 4), "C": Point(6, 8)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    morphology = {"A": Morphology(1, 2, 3, 4), "B": Morphology(5, 6, 7, 8), "C": Morphology(9, 10, 11, 12)}
    updates = [BanditUpdate("context1", "A", 1.0, 0.5), BanditUpdate("context2", "B", 2.0, 0.6)]

    hybrid_bandit_tree = HybridBanditTree()
    cost = hybrid_bandit_tree.tree_cost(nodes, edges, root, morphology, updates)
    print("Tree cost:", cost)

if __name__ == "__main__":
    main()