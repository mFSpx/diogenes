# DARWIN HAMMER — match 253, survivor 2
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

class HybridBanditTree:
    def __init__(self):
        self._policy: Dict[str, List[float]] = {}

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

    def tree_cost(self, nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str, path_weight: float = 0.2) -> float:
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
        return material + path_weight * sum(dist.values())

    def hybrid_bandit_tree(self, nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str, path_weight: float = 0.2, updates: List[BanditUpdate] = []) -> float:
        self.update_policy(updates)
        tree_score = self.tree_cost(nodes, edges, root, path_weight)
        bandit_score = sum(self._reward(action) for action in self._policy)
        return tree_score + bandit_score

    def hybrid_bandit_tree_update(self, nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str, path_weight: float = 0.2, updates: List[BanditUpdate] = []) -> None:
        self.update_policy(updates)
        tree_score = self.tree_cost(nodes, edges, root, path_weight)
        bandit_score = sum(self._reward(action) for action in self._policy)
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
    hbt = HybridBanditTree()
    hbt.hybrid_bandit_tree_update(nodes, edges, root, updates=updates)