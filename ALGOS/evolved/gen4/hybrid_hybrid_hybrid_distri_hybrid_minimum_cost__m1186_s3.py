# DARWIN HAMMER — match 1186, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s0.py (gen2)
# parent_b: hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s1.py (gen3)
# born: 2026-05-29T23:33:31Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Mapping, Hashable, Set

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric and graph utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Point:
    """A point in 2D space."""
    x: float
    y: float

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a.x - b.x, a.y - b.y)

def tree_cost(nodes: Dict[str, Point],
              edges: List[Tuple[str, str]],
              root: str,
              path_weight: float = 0.2) -> float:
    """
    Compute the total cost of a tree:
      material = sum of edge lengths
      path_cost = weighted sum of distances from root to every node
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])

    # BFS/DFS to compute root‑to‑node distances
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

# ----------------------------------------------------------------------
# Hoeffding bound & cooling schedule
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound ε = sqrt( (r² * ln(1/δ)) / (2n) )."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Geometric cooling schedule T_k = t0 * α^k."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)

def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance: 1 if ΔE≤0 else exp(-ΔE/T)."""
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)

# ----------------------------------------------------------------------
# Bandit policy
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}   # action_id -> [total_reward, count]

def reset_policy() -> None:
    """Erase all learned statistics."""
    _POLICY.clear()

def _reward(action: str) -> float:
    """Empirical mean reward for *action* (0 if never observed)."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> int:
    """Number of times *action* has been selected."""
    return int(_POLICY.get(action, [0.0, 0.0])[1])

def update_policy(updates: List[Tuple[str, float]]) -> None:
    """
    Batch update of the policy.
    Each tuple is (action_id, reward).
    """
    for action_id, reward in updates:
        stats = _POLICY.setdefault(action_id, [0.0, 0.0])
        stats[0] += float(reward)
        stats[1] += 1.0

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hoeffding_ucb(action_id: str, r: float, delta: float) -> float:
    """
    Upper‑confidence bound for a bandit arm using Hoeffding confidence.
    μ̂ + ε, where ε = hoeffding_bound(r, δ, n).
    """
    n = _count(action_id)
    if n == 0:
        # Encourage exploration of unseen edges
        return float('inf')
    mean = _reward(action_id)
    eps = hoeffding_bound(r, delta, n)
    return mean + eps

def select_edge_ucb(nodes: Dict[str, Point],
                    candidate_edges: List[Tuple[str, str]],
                    r: float,
                    delta: float) -> Tuple[str, str]:
    """
    Choose the edge with the highest Hoeffding‑UCB.
    Edge identifier is the string "a-b".
    """
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
    """
    Decide whether to accept a higher‑cost edge using simulated‑annealing.
    """
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
    """
    Incrementally construct a minimum‑cost spanning tree.
    At each iteration:
      1. Generate the set of admissible edges (those not yet in the tree).
      2. Select an edge via Hoeffding‑UCB.
      3. Compute the hypothetical tree cost if the edge were added.
      4. Accept the edge with annealed probability.
      5. Update the bandit policy with reward = -new_cost (lower cost ⇒ higher reward).
    Returns the final edge list.
    """
    # Initialise empty tree
    tree_edges: List[Tuple[str, str]] = []
    all_nodes = set(nodes.keys())
    # Pre‑compute full undirected edge set (complete graph)
    full_edges = [(a, b) for a in all_nodes for b in all_nodes if a < b]

    for k in range(max_iters):
        temperature = cooling_temperature(k, t0, alpha)

        # Candidate edges are those not yet in the tree and that keep the graph acyclic
        # Simple cycle check via union‑find
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

        candidate_edges = [(a, b) for a, b in full_edges if (a, b) not in tree_edges and (b, a) not in tree_edges]
        for a, b in tree_edges:
            parent[a] = a
            parent[b] = b

        acyclic_edges = []
        for edge in candidate_edges:
            a, b = edge
            if union(a, b):
                acyclic_edges.append(edge)

        if not acyclic_edges:
            break

        # Select edge with UCB
        best_edge = select_edge_ucb(nodes, acyclic_edges, r, delta)

        # Compute hypothetical tree cost
        hypothetical_tree = tree_edges + [best_edge]
        hypothetical_cost = tree_cost(nodes, hypothetical_tree, root)

        # Accept edge with annealed probability
        current_cost = tree_cost(nodes, tree_edges, root)
        if annealed_edge_accept(current_cost, hypothetical_cost, temperature):
            tree_edges.append(best_edge)
            update_policy([(f"{best_edge[0]}-{best_edge[1]}", -hypothetical_cost)])

    return tree_edges