# DARWIN HAMMER — match 2471, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_ternar_m1199_s0.py (gen4)
# born: 2026-05-29T23:42:27Z

"""Hybrid Resource Allocation via Tree Metrics and SSIM Similarity

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s1.py (Algorithm A)
- hybrid_hybrid_hybrid_gliner_hybrid_hybrid_ternar_m1199_s0.py (Algorithm B)

Mathematical Bridge:
Algorithm A provides Euclidean edge lengths and root‑to‑node distances on a tree.
Algorithm B supplies a structural similarity index (SSIM) between two prototype
vectors.  The hybrid algorithm treats the SSIM score as a *global similarity*
factor and modulates the per‑node allocation by the node's distance from the root.
Edge weights are therefore defined as

    w_{e} = ℓ_{e} · (1 + ρ·d_{v}/D),

where ℓ_{e} is the Euclidean length of edge *e*, d_{v} is the distance of the
incident child node *v* from the root, D = max_{u} d_{u} normalises distances,
and ρ = SSIM(x, y) ∈ [0, 1] scales the influence of similarity.  The resulting
weighted tree drives a proportional distribution of work units across a set of
model groups, preserving the probabilistic flavour of the Bayesian update from
Algorithm A while respecting the allocation semantics of Algorithm B.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core building blocks from the parents
# ----------------------------------------------------------------------


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    visited = {root}
    frontier = [root]
    while frontier:
        current = frontier.pop()
        for neigh in adj[current]:
            if neigh not in visited:
                # identify the directed edge used for the path
                edge = (current, neigh) if (current, neigh) in edge_len else (neigh, current)
                dist[neigh] = dist[current] + edge_len[edge]
                visited.add(neigh)
                frontier.append(neigh)

    return adj, edge_len, dist


def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two vectors.
    Result is clipped to [0, 1] for stability.
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    )
    return float(np.clip(ssim, 0.0, 1.0))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


GROUPS = ("codex", "groq", "cohere", "local_models")


def hybrid_edge_weights(
    edge_len: Dict[Tuple[str, str], float],
    node_dist: Dict[str, float],
    ssim: float,
) -> Dict[Tuple[str, str], float]:
    """
    Combine Euclidean edge length with SSIM‑scaled distance factor.

    w_e = ℓ_e * (1 + ssim * d_child / D)

    where d_child is the distance of the child node (the node farther from the root)
    and D = max(node_dist) normalises the distance term.
    """
    if not node_dist:
        return {e: ℓ for e, ℓ in edge_len.items()}

    D = max(node_dist.values())
    if D == 0:
        D = 1.0

    weighted: Dict[Tuple[str, str], float] = {}
    for (a, b), ℓ in edge_len.items():
        # Determine which endpoint is farther from the root
        if node_dist[a] > node_dist[b]:
            d_child = node_dist[a]
        else:
            d_child = node_dist[b]
        weight = ℓ * (1.0 + ssim * (d_child / D))
        weighted[(a, b)] = weight
    return weighted


def hybrid_allocate_workshare(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    prototype: np.ndarray,
    target: np.ndarray,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
) -> Dict[str, float]:
    """
    Allocate work units to *groups* using a hybrid of tree geometry and SSIM.

    Steps
    -----
    1. Compute tree metrics (adjacency, edge lengths, root distances).
    2. Compute SSIM between ``prototype`` and ``target`` → ρ.
    3. Derive edge weights with ``hybrid_edge_weights``.
    4. Collapse edge weights to a node‑wise importance score by summing incident
       weighted edges.
    5. Normalise node importance to obtain a probability distribution.
    6. Distribute the *LLM* portion of ``total_units`` proportionally to node
       importance, then split each node’s share evenly across the requested groups.
    7. Return a flat dictionary mapping each group to its final allocated units.
    """
    # 1. Tree metrics
    _, edge_len, node_dist = tree_metrics(nodes, edges, root)

    # 2. SSIM similarity
    rho = compute_ssim(prototype, target)

    # 3. Hybrid edge weights
    weighted_edges = hybrid_edge_weights(edge_len, node_dist, rho)

    # 4. Node importance (sum of incident weighted edges)
    node_importance: Dict[str, float] = {n: 0.0 for n in nodes}
    for (a, b), w in weighted_edges.items():
        node_importance[a] += w
        node_importance[b] += w

    # 5. Normalise to probabilities
    total_importance = sum(node_importance.values())
    if total_importance == 0:
        prob = {n: 1.0 / len(nodes) for n in nodes}
    else:
        prob = {n: v / total_importance for n, v in node_importance.items()}

    # 6. Split deterministic vs LLM units
    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units

    # Allocate LLM units to nodes according to probability, then to groups
    per_group_units = {g: 0.0 for g in groups}
    for node, p in prob.items():
        node_share = llm_units * p
        share_per_group = node_share / len(groups)
        for g in groups:
            per_group_units[g] += share_per_group

    # Add deterministic portion equally to all groups (policy choice)
    det_per_group = deterministic_units / len(groups)
    for g in groups:
        per_group_units[g] += det_per_group

    # Round to 6 decimal places for readability
    rounded = {g: round(v, 6) for g, v in per_group_units.items()}
    rounded["total_units"] = round(total_units, 6)
    rounded["deterministic_units"] = round(deterministic_units, 6)
    rounded["llm_units"] = round(llm_units, 6)
    rounded["ssim"] = round(rho, 6)

    return rounded


def hybrid_minimum_cost_tree(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    prototype: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, List[Tuple[str, str]]]:
    """
    Compute a *minimum* cost spanning tree where the cost of each edge is the
    hybrid weight defined in ``hybrid_edge_weights``.  Because the input graph
    is already a tree (as required by the parent algorithm), the minimum‑cost
    tree is simply the set of edges with the smallest hybrid weights that
    keep connectivity.  We therefore sort edges by hybrid weight and perform
    a Kruskal‑style union‑find to select the cheapest feasible subset.

    Returns
    -------
    total_cost : float
        Sum of hybrid weights of the selected edges.
    selected_edges : list of edges (as supplied) forming the minimum‑cost tree.
    """
    # Compute metrics and hybrid weights
    _, edge_len, node_dist = tree_metrics(nodes, edges, root)
    rho = compute_ssim(prototype, target)
    weighted = hybrid_edge_weights(edge_len, node_dist, rho)

    # Union‑find data structure
    parent: Dict[str, str] = {}

    def find(u: str) -> str:
        while parent.get(u, u) != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u: str, v: str) -> bool:
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        parent[rv] = ru
        return True

    # Sort edges by hybrid weight (ascending)
    sorted_edges = sorted(edges, key=lambda e: weighted[e])

    total_cost = 0.0
    selected: List[Tuple[str, str]] = []
    for a, b in sorted_edges:
        if union(a, b):
            selected.append((a, b))
            total_cost += weighted[(a, b)]
        if len(selected) == len(nodes) - 1:
            break

    return total_cost, selected


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic tree
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
        "E": (0.5, 1.5),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("C", "E")]
    root = "A"

    # Random prototype / target vectors
    rng = np.random.default_rng(42)
    prototype = rng.random(128)
    target = rng.random(128)

    total_units = 1000.0

    allocation = hybrid_allocate_workshare(
        nodes,
        edges,
        root,
        prototype,
        target,
        total_units,
        deterministic_target_pct=85.0,
    )
    print("Hybrid Allocation:")
    for k, v in allocation.items():
        print(f"  {k}: {v}")

    cost, sel_edges = hybrid_minimum_cost_tree(nodes, edges, root, prototype, target)
    print("\nHybrid Minimum‑Cost Tree:")
    print(f"  Total cost: {cost:.6f}")
    print(f"  Selected edges: {sel_edges}")