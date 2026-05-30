# DARWIN HAMMER — match 1468, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m522_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s4.py (gen4)
# born: 2026-05-29T23:36:39Z

"""Hybrid algorithm merging:
- Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m522_s0.py): VRAM‑aware TTT‑Linear weight update using Ollivier‑Ricci curvature.
- Parent B (hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s4.py): Minimum‑cost tree evaluation, Hoeffding‑UCB bandit policy, and simulated‑annealing acceptance.

Mathematical bridge:
Both parents employ an *update rule* that scales a gradient (or reward) by a *curvature / confidence* term.
In Parent A the curvature is the squared residual of a linear model;
in Parent B the confidence term is  (1+S/(S+1))/√(1+Nₐ)  derived from a bandit store S and action count Nₐ.
The hybrid therefore:
1. Uses a linear model W to predict geometric quantities (edge lengths) on a graph.
2. Computes Ollivier‑Ricci curvature as the mean‑squared residual of those predictions.
3. Modulates learning‑rate‑like updates of W by the curvature **and** by the bandit confidence term.
4. Couples the curvature‑aware learning rate with the simulated‑annealing temperature that drives tree‑structure optimisation.

The resulting system jointly refines the tree topology and the predictive model in a unified optimisation loop.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Mapping, Hashable, Set

import numpy as np

# ----------------------------------------------------------------------
# Structures and utilities from Parent B
# ----------------------------------------------------------------------
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
    """Material + weighted sum of distances from root to every node."""
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

# ----------------------------------------------------------------------
# Bandit policy (Parent B)
# ----------------------------------------------------------------------
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
    avg = _reward(action_id)
    return avg + hoeffding_bound(r, delta, n)

# ----------------------------------------------------------------------
# Functions from Parent A (curvature‑aware linear update)
# ----------------------------------------------------------------------
def confidence_term(S: float, N_a: int) -> float:
    """Confidence term used by the bandit, identical to Parent A."""
    return (1 + S / (S + 1)) / math.sqrt(1 + N_a)

def edge_features(edge: Tuple[str, str], nodes: Dict[str, Point]) -> np.ndarray:
    """Feature vector [x₁, y₁, x₂, y₂] for an edge."""
    a, b = edge
    p1, p2 = nodes[a], nodes[b]
    return np.array([p1.x, p1.y, p2.x, p2.y], dtype=float)

def predict_edge_lengths(W: np.ndarray, edges: List[Tuple[str, str]],
                         nodes: Dict[str, Point]) -> np.ndarray:
    """Linear model predictions for all edges."""
    X = np.vstack([edge_features(e, nodes) for e in edges])  # shape (m,4)
    return X @ W  # shape (m,)

def curvature_residual(W: np.ndarray, edges: List[Tuple[str, str]],
                       nodes: Dict[str, Point]) -> float:
    """Ollivier‑Ricci curvature proxy: mean squared residual of edge‑length predictions."""
    preds = predict_edge_lengths(W, edges, nodes)
    actual = np.array([length(nodes[a], nodes[b]) for a, b in edges], dtype=float)
    residual = preds - actual
    return float(np.mean(residual ** 2) + 1e-12)  # avoid division by zero

def gradient_wrt_W(W: np.ndarray, edges: List[Tuple[str, str]],
                   nodes: Dict[str, Point]) -> np.ndarray:
    """Exact gradient of MSE loss for the linear edge‑length model."""
    X = np.vstack([edge_features(e, nodes) for e in edges])          # (m,4)
    preds = X @ W                                                   # (m,)
    actual = np.array([length(nodes[a], nodes[b]) for a, b in edges])
    residual = preds - actual                                       # (m,)
    # dMSE/dW = (2/m) * X^T * residual
    m = len(edges)
    return (2.0 / m) * X.T @ residual

def krampus_update(W: np.ndarray, edges: List[Tuple[str, str]],
                   nodes: Dict[str, Point], S: float, N_a: int,
                   lr: float = 0.01) -> np.ndarray:
    """Hybrid weight update: gradient scaled by curvature and bandit confidence."""
    grad = gradient_wrt_W(W, edges, nodes)
    curv = curvature_residual(W, edges, nodes)
    conf = confidence_term(S, N_a)
    # The denominator mixes curvature (larger curvature → smaller step) with confidence (larger confidence → larger step)
    denom = curv / conf
    W_new = W - lr * grad / denom
    return W_new

# ----------------------------------------------------------------------
# Hybrid optimisation step (combines tree SA and weight learning)
# ----------------------------------------------------------------------
def propose_tree_swap(edges: List[Tuple[str, str]],
                     nodes: Dict[str, Point]) -> List[Tuple[str, str]]:
    """
    Randomly remove one edge and reconnect the two resulting components
    with a new random edge that keeps the graph a tree.
    """
    if len(edges) < 2:
        return edges.copy()

    # Remove a random edge
    remove_idx = random.randrange(len(edges))
    removed = edges[remove_idx]
    remaining = edges[:remove_idx] + edges[remove_idx + 1:]

    # Build adjacency for the remaining edges
    adj: Dict[str, Set[str]] = {k: set() for k in nodes}
    for a, b in remaining:
        adj[a].add(b)
        adj[b].add(a)

    # Find the two components via BFS
    def bfs(start: str) -> Set[str]:
        seen = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    comp1 = bfs(removed[0])
    comp2 = set(nodes.keys()) - comp1

    # Choose a random pair across the cut
    a = random.choice(list(comp1))
    b = random.choice(list(comp2))
    new_edge = (a, b)

    return remaining + [new_edge]

def hybrid_annealing_step(edges: List[Tuple[str, str]],
                          nodes: Dict[str, Point],
                          root: str,
                          W: np.ndarray,
                          S: float,
                          N_a: int,
                          step: int) -> Tuple[List[Tuple[str, str]], np.ndarray, float]:
    """
    Perform one hybrid iteration:
    1. Propose a new tree topology.
    2. Evaluate cost difference.
    3. Accept/reject using a temperature that incorporates the bandit confidence term.
    4. Update the linear model W with curvature‑aware gradient.
    Returns the possibly updated edge list, the new weight matrix, and the acceptance probability.
    """
    old_cost = tree_cost(nodes, edges, root)
    new_edges = propose_tree_swap(edges, nodes)
    new_cost = tree_cost(nodes, new_edges, root)
    delta_e = new_cost - old_cost

    # Temperature modulated by confidence term (higher confidence → hotter system)
    base_temp = cooling_temperature(step)
    temperature = base_temp * (1.0 + confidence_term(S, N_a))

    accept_prob = acceptance_probability(delta_e, temperature)
    if random.random() < accept_prob:
        # Accept new topology
        edges = new_edges
        # Reward for the action (negative delta_e encourages lower cost)
        update_policy([("swap", -delta_e)])
        # Update model weights using the new tree
        W = krampus_update(W, edges, nodes, S, N_a)
    else:
        # Penalise the rejected action slightly
        update_policy([("swap", -0.1 * abs(delta_e))])

    return edges, W, accept_prob

# ----------------------------------------------------------------------
# Helper to generate a simple initial tree (spanning tree via nearest neighbour)
# ----------------------------------------------------------------------
def nearest_neighbour_tree(nodes: Dict[str, Point]) -> List[Tuple[str, str]]:
    """Greedy construction of a spanning tree by repeatedly connecting the nearest unconnected node."""
    if not nodes:
        return []
    unvisited = set(nodes.keys())
    visited = set()
    edges: List[Tuple[str, str]] = []

    current = unvisited.pop()
    visited.add(current)

    while unvisited:
        best_pair = None
        best_dist = float('inf')
        for v in visited:
            for u in unvisited:
                d = length(nodes[v], nodes[u])
                if d < best_dist:
                    best_dist = d
                    best_pair = (v, u)
        a, b = best_pair
        edges.append((a, b))
        visited.add(b)
        unvisited.remove(b)

    return edges

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Create a random set of points
    num_points = 10
    nodes = {f"P{i}": Point(random.uniform(0, 10), random.uniform(0, 10))
             for i in range(num_points)}
    root_id = "P0"

    # Initialise a spanning tree
    edges = nearest_neighbour_tree(nodes)

    # Initialise linear model weights (4 features -> scalar)
    W = np.random.randn(4) * 0.1

    # Bandit store S and action count N_a (start with zero)
    S = 0.0
    N_a = 0

    # Run a few hybrid iterations
    for step in range(20):
        edges, W, prob = hybrid_annealing_step(edges, nodes, root_id, W, S, N_a, step)
        # Update S and N_a from policy statistics for the next iteration
        S = _reward("swap")
        N_a = _count("swap")
        if step % 5 == 0:
            curv = curvature_residual(W, edges, nodes)
            cost = tree_cost(nodes, edges, root_id)
            print(f"Step {step:02d} | Cost={cost:.3f} | Curvature={curv:.5f} | AcceptProb={prob:.3f}")

    print("Final weight vector:", W)
    print("Final tree edges:", edges)