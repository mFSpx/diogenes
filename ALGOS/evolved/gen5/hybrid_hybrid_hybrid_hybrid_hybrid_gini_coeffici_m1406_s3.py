# DARWIN HAMMER — match 1406, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s2.py (gen4)
# parent_b: hybrid_gini_coefficient_hybrid_hybrid_rbf_su_m344_s1.py (gen4)
# born: 2026-05-29T23:36:05Z

"""Hybrid Minimum‑Cost Tree with Gini‑Adjusted Hoeffding Confidence and RBF Similarity.

Parents:
- hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s2.py
- hybrid_gini_coefficient_hybrid_hybrid_rbf_su_m344_s1.py

Mathematical bridge:
Each candidate edge is treated as a bandit arm.  Its Upper‑Confidence‑Bound (UCB) is

    UCB(e) = \hat{μ}_e + ε_e ,

where \hat{μ}_e is the empirical reward (negative edge length) and ε_e is a Hoeffding
confidence term.  The Hoeffding term is modulated by the Gini coefficient of the
feature distribution at the target node, i.e.

    ε'_e = ε_e · (1 + Gini(v)) .

Thus nodes with highly unequal feature values obtain a wider confidence interval,
encouraging exploration where the data is heterogeneous.  Edge weights for the
deterministic cost component are obtained from an RBF similarity matrix built
from node feature vectors.  The simulated‑annealing schedule from the first parent
governs acceptance of higher‑cost edges.

The implementation below provides the core equations, a simple hybrid UCB‑driven
tree construction, and a smoke test."""
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Mapping, Hashable, Set, Iterable, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Geometry utilities (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Point:
    """2‑D point."""
    x: float
    y: float

def euclidean_distance(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a.x - b.x, a.y - b.y)

def tree_cost(
    nodes: Dict[str, Point],
    edges: List[Tuple[str, str]],
    root: str,
    path_weight: float = 0.2,
) -> float:
    """
    Deterministic cost of a rooted tree:
      material = sum of Euclidean edge lengths
      path_cost = weighted sum of distances from root to every node
    """
    # adjacency list
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # material cost
    material = sum(euclidean_distance(nodes[u], nodes[v]) for u, v in edges)

    # BFS to compute root‑to‑node distances
    dist: Dict[str, float] = {root: 0.0}
    queue: List[str] = [root]
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + euclidean_distance(nodes[cur], nodes[nb])
                queue.append(nb)

    path_cost = sum(dist.values())
    return material + path_weight * path_cost

# ----------------------------------------------------------------------
# Statistical utilities (Hoeffding + Gini)
# ----------------------------------------------------------------------
def hoeffding_bound(range_: float, delta: float, n: int) -> float:
    """
    Classic Hoeffding bound for the mean of bounded random variables.
    ε = sqrt( (R^2 * ln(1/δ)) / (2 n) )
    """
    if n <= 0:
        return float('inf')
    return math.sqrt((range_ ** 2 * math.log(1.0 / delta)) / (2.0 * n))

def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient of a non‑negative value collection."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    numerator = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return numerator / (n * sum(xs))

def gini_adjusted_hoeffding(
    values: Iterable[float],
    range_: float,
    delta: float,
    n: int,
) -> float:
    """
    Hoeffding bound scaled by (1 + Gini).  Nodes with higher inequality get a
    larger confidence interval, encouraging exploration.
    """
    base = hoeffding_bound(range_, delta, n)
    gini = gini_coefficient(values)
    return base * (1.0 + gini)

# ----------------------------------------------------------------------
# Simulated‑annealing acceptance
# ----------------------------------------------------------------------
def acceptance_probability(delta_cost: float, temperature: float) -> float:
    """
    Standard Metropolis acceptance:
        P = 1               if ΔE ≤ 0
        P = exp(-ΔE / T)    otherwise
    """
    if delta_cost <= 0.0:
        return 1.0
    return math.exp(-delta_cost / max(temperature, 1e-12))

def cooling_schedule(initial_T: float, iteration: int, alpha: float = 0.99) -> float:
    """Exponential cooling: T_k = T_0 * α^k."""
    return initial_T * (alpha ** iteration)

# ----------------------------------------------------------------------
# Feature similarity (RBF + Hamming) – from Parent B
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash based on median threshold."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:  # limit to 64 bits
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Bitwise Hamming distance."""
    return (a ^ b).bit_count()

def rbf_similarity(vec_a: Sequence[float], vec_b: Sequence[float], epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    if len(vec_a) != len(vec_b):
        raise ValueError("vectors must have same dimension")
    sq = sum((x - y) ** 2 for x, y in zip(vec_a, vec_b))
    return math.exp(-epsilon * sq)

def similarity_matrix(
    features: Dict[Hashable, Sequence[float]],
    epsilon: float = 1.0,
) -> Tuple[np.ndarray, List[Hashable]]:
    """
    Returns an (n, n) similarity matrix S where S_ij = RBF similarity of feature
    vectors i and j, together with the ordered list of node identifiers.
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        vi = features[ni]
        for j, nj in enumerate(nodes):
            if i <= j:
                sim = rbf_similarity(vi, features[nj], epsilon)
                S[i, j] = sim
                S[j, i] = sim
    return S, nodes

# ----------------------------------------------------------------------
# Hybrid UCB‑driven tree construction
# ----------------------------------------------------------------------
def hybrid_ucb(
    reward_sum: float,
    count: int,
    feature_vals: Iterable[float],
    range_: float,
    delta: float,
) -> float:
    """
    Upper‑Confidence‑Bound for an edge treated as a bandit arm.
    Reward is negative length (higher is better).  Confidence term is
    Gini‑adjusted Hoeffding.
    """
    mean = reward_sum / max(count, 1)
    conf = gini_adjusted_hoeffding(feature_vals, range_, delta, count)
    return mean + conf

def build_hybrid_tree(
    nodes: Dict[str, Point],
    features: Dict[str, List[float]],
    root: str,
    delta: float = 0.1,
    range_: float = 1.0,
    initial_T: float = 10.0,
    max_iter: int = 10_000,
) -> List[Tuple[str, str]]:
    """
    Constructs a spanning tree using a hybrid UCB policy.
    - Each candidate edge (u, v) where u is already in the tree and v is not,
      receives a reward r = -length(u, v).
    - UCB combines empirical mean reward with a Gini‑adjusted Hoeffding bound.
    - If the selected edge would increase the deterministic material cost,
      it is accepted with simulated‑annealing probability.
    Returns the list of edges (undirected) of the resulting tree.
    """
    if root not in nodes:
        raise ValueError("root must be a key of nodes")

    # Pre‑compute similarity matrix for optional weighting (not used directly in cost)
    _sim_mat, _order = similarity_matrix(features)

    visited: Set[str] = {root}
    edges: List[Tuple[str, str]] = []
    # Statistics per directed edge (u->v) stored as (reward_sum, count)
    stats: Dict[Tuple[str, str], Tuple[float, int]] = {}

    iteration = 0
    while len(visited) < len(nodes) and iteration < max_iter:
        candidates: List[Tuple[str, str]] = []
        for u in visited:
            for v in nodes:
                if v in visited:
                    continue
                candidates.append((u, v))

        # Compute UCB for each candidate
        ucb_values: List[float] = []
        for (u, v) in candidates:
            reward_sum, cnt = stats.get((u, v), (0.0, 0))
            # feature values of the target node drive the Gini term
            feat_vals = features[v]
            ucb = hybrid_ucb(reward_sum, cnt, feat_vals, range_, delta)
            ucb_values.append(ucb)

        # Choose edge with highest UCB (exploration/exploitation trade‑off)
        best_idx = int(np.argmax(ucb_values))
        u_best, v_best = candidates[best_idx]

        # Deterministic material cost of the edge
        edge_len = euclidean_distance(nodes[u_best], nodes[v_best])
        reward = -edge_len  # higher reward for shorter edges

        # Update statistics
        prev_sum, prev_cnt = stats.get((u_best, v_best), (0.0, 0))
        stats[(u_best, v_best)] = (prev_sum + reward, prev_cnt + 1)

        # Simulated‑annealing acceptance based on *increase* in total material cost
        current_material = sum(euclidean_distance(nodes[a], nodes[b]) for a, b in edges)
        delta_cost = edge_len if (current_material + edge_len) > current_material else 0.0
        T = cooling_schedule(initial_T, iteration)
        if random.random() < acceptance_probability(delta_cost, T):
            edges.append((u_best, v_best))
            visited.add(v_best)

        iteration += 1

    return edges

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic instance
    random.seed(42)
    np.random.seed(42)

    # 6 nodes placed on a unit square
    node_ids = [f"N{i}" for i in range(6)]
    nodes: Dict[str, Point] = {
        nid: Point(x=random.random(), y=random.random()) for nid in node_ids
    }

    # Random 5‑dimensional feature vectors (non‑negative)
    features: Dict[str, List[float]] = {
        nid: list(np.abs(np.random.randn(5))) for nid in node_ids
    }

    root_node = node_ids[0]

    tree_edges = build_hybrid_tree(
        nodes=nodes,
        features=features,
        root=root_node,
        delta=0.05,
        range_=1.0,
        initial_T=5.0,
        max_iter=5000,
    )

    print("Constructed edges:")
    for e in tree_edges:
        print(f"  {e[0]} -- {e[1]}  (len={euclidean_distance(nodes[e[0]], nodes[e[1]]):.3f})")

    total = tree_cost(nodes, tree_edges, root_node)
    print(f"Total hybrid tree cost: {total:.4f}")