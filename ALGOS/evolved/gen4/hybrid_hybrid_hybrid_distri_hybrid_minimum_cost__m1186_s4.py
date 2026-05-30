# DARWIN HAMMER — match 1186, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s0.py (gen2)
# parent_b: hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s1.py (gen3)
# born: 2026-05-29T23:33:31Z

import math
import random
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Mapping, Hashable, Set

import numpy as np

@dataclass(frozen=True)
class Point:
    x: float
    y: float

def length(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

def tree_cost(nodes: Dict[str, Point],
              edges: List[Tuple[str, str]],
              root: str,
              path_weight: float = 0.2) -> float:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + length(nodes[cur], nodes[nxt])
                stack.append(nxt)

    path_cost = sum(dist.values())
    return material + path_weight * path_cost

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)

def acceptance_probability(delta_e: float, temperature: float) -> float:
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)

_POLICY: Dict[str, List[float]] = {}   
def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> int:
    return int(_POLICY.get(action, [0.0, 0.0])[1])

def update_policy(updates: List[Tuple[str, float]]) -> None:
    for action_id, reward in updates:
        stats = _POLICY.setdefault(action_id, [0.0, 0.0])
        stats[0] += float(reward)
        stats[1] += 1.0

def hoeffding_ucb(action_id: str, r: float, delta: float) -> float:
    n = _count(action_id)
    if n == 0:
        return float('inf')
    mean = _reward(action_id)
    eps = hoeffding_bound(r, delta, n)
    return mean + eps

def select_edge_ucb(nodes: Dict[str, Point],
                    candidate_edges: List[Tuple[str, str]],
                    r: float,
                    delta: float) -> Tuple[str, str]:
    best_edge = None
    best_score = -float('inf')
    for a, b in candidate_edges:
        aid = f"{a}-{b}"
        score = hoeffding_ucb(aid, r, delta)
        if score > best_score:
            best_score = score
            best_edge = (a, b)
    if best_edge is None:
        raise RuntimeError("No candidate edges available")
    return best_edge

def annealed_edge_accept(current_cost: float,
                         new_cost: float,
                         temperature: float) -> bool:
    delta = new_cost - current_cost
    prob = acceptance_probability(delta, temperature)
    return random.random() < prob

def hybrid_grow_tree(nodes: Dict[str, Point],
                     root: str,
                     r: float = 1.0,
                     delta: float = 0.05,
                     t0: float = 1.0,
                     alpha: float = 0.95,
                     max_iters: int = 50) -> List[Tuple[str, str]]:
    tree_edges: List[Tuple[str, str]] = []
    all_nodes = set(nodes.keys())
    full_edges = [(a, b) for a in all_nodes for b in all_nodes if a < b]

    for k in range(max_iters):
        temperature = cooling_temperature(k, t0, alpha)

        parent: Dict[str, str] = {}
        def find(x: str) -> str:
            while parent.get(x, x) != x:
                x = parent[x]
            return x

        def union(x: str, y: str) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[ry] = rx
            return True

        candidate_edges = [(a, b) for a, b in full_edges if (a, b) not in tree_edges and (b, a) not in tree_edges and union(a, b)]

        if candidate_edges:
            new_edge = select_edge_ucb(nodes, candidate_edges, r, delta)
            new_cost = tree_cost(nodes, tree_edges + [new_edge], root)
            if annealed_edge_accept(tree_cost(nodes, tree_edges, root), new_cost, temperature):
                tree_edges.append(new_edge)
                update_policy([(f"{new_edge[0]}-{new_edge[1]}", -new_cost)])

    return tree_edges

def main():
    nodes = {
        'A': Point(0, 0),
        'B': Point(1, 0),
        'C': Point(1, 1),
        'D': Point(0, 1)
    }
    root = 'A'
    result = hybrid_grow_tree(nodes, root)
    print(result)

if __name__ == "__main__":
    main()