# DARWIN HAMMER — match 2139, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s5.py (gen4)
# parent_b: hybrid_fold_change_detectio_hybrid_regret_engine_m1100_s0.py (gen3)
# born: 2026-05-29T23:41:00Z

"""
Module for the hybrid algorithm combining the tree cost optimization of 
hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s5.py and the 
adaptive update mechanism of hybrid_fold_change_detectio_hybrid_regret_engine_m1100_s0.py.
The mathematical bridge between the two algorithms is established by applying 
the fold-change detection update equations to the expected values of the 
MathAction objects, which are used to evaluate the regret of different 
configurations in the tree cost optimization problem.

Parent algorithms:
- hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s5.py
- hybrid_fold_change_detectio_hybrid_regret_engine_m1100_s0.py
"""

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

@dataclass(frozen=True)
class Point:
    x: float
    y: float

@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

def tree_cost(
    nodes: Dict[str, Point],
    edges: List[Tuple[str, str]],
    root: str,
    path_weight: float = 0.2,
) -> float:
    """Total cost = material + weighted path‑to‑root cost."""
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        material += euclidean(nodes[u], nodes[v])

    # BFS from root to compute distances
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + euclidean(nodes[cur], nodes[nxt])
                stack.append(nxt)

    path_cost = sum(dist.values())
    return material + path_weight * path_cost

def step(u: float, x: float, y: float, dt: float = 1.0, gain: float = 1.0, decay_x: float = 1.0, decay_y: float = 1.0, eps: float = 1e-12) -> tuple[float, float]:
    """Advance the feed-forward state using Euler integration."""
    if dt < 0:
        raise ValueError('dt must be non-negative')
    ratio = u / max(abs(x), eps)
    dy = gain * ratio - decay_y * y
    dx = u - decay_x * x
    return x + dt * dx, y + dt * dy

def update_action(action: MathAction, u: float, dt: float = 1.0, gain: float = 1.0, decay_x: float = 1.0, decay_y: float = 1.0, eps: float = 1e-12) -> MathAction:
    new_expected_value, _ = step(u, action.expected_value, 0.0, dt, gain, decay_x, decay_y, eps)
    return MathAction(action.id, new_expected_value, action.cost, action.risk)

def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Tuple[str, str]],
    root: str,
    actions: List[MathAction],
    path_weight: float = 0.2,
    dt: float = 1.0,
    gain: float = 1.0,
    decay_x: float = 1.0,
    decay_y: float = 1.0,
) -> Tuple[float, List[MathAction]]:
    tree_cost_value = tree_cost(nodes, edges, root, path_weight)
    updated_actions = [update_action(action, tree_cost_value, dt, gain, decay_x, decay_y) for action in actions]
    return tree_cost_value, updated_actions

if __name__ == "__main__":
    nodes = {"A": Point(0.0, 0.0), "B": Point(1.0, 0.0), "C": Point(0.5, 1.0)}
    edges = [("A", "B"), ("A", "C"), ("B", "C")]
    root = "A"
    actions = [MathAction("action1", 1.0), MathAction("action2", 2.0)]
    tree_cost_value, updated_actions = hybrid_tree_cost(nodes, edges, root, actions)
    print("Tree cost:", tree_cost_value)
    print("Updated actions:")
    for action in updated_actions:
        print(action)