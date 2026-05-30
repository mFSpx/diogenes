# DARWIN HAMMER — match 5531, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_hybrid_m2405_s0.py (gen6)
# born: 2026-05-30T00:02:42Z

"""
Hybrid Minimum‑Cost Tree with Workshare‑Perceptual Allocation
==========================================================

Parents
-------
* **hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s4.py** – defines a
  Bayesian‑augmented minimum‑cost tree.  Edge contributions are replaced by their
  expected value under a posterior belief *pₑ* and node distances are weighted by
  a node belief *qᵥ* derived from incident edge posteriors.

* **hybrid_hybrid_hybrid_worksh_hybrid_hybrid_hybrid_m2405_s0.py** – provides a
  deterministic work‑share allocator that uses a 64‑bit perceptual hash of a
  numeric feature vector to drive exploration/exploitation across a fixed set
  of groups.

Mathematical Bridge
-------------------
The bridge is the **perceptual hash**.  For every node *v* we compute a hash
`h(v)` from its geometric feature vector (its coordinates).  The hash is mapped
to one of the predefined groups `G = {g₀,…,g_{K‑1}}`.  Each group carries a
scalar work‑share weight `w_g` (e.g. proportional to the number of nodes that
belong to the group).  The node belief *qᵥ* is then defined as


qᵥ = ( Σ_{e∈Inc(v)} pₑ ) · w_{group(v)} / Z


where `Inc(v)` are edges incident to *v* and `Z` is a normalising constant that
makes Σ_v qᵥ = 1.  This fuses the Bayesian edge posteriors of Parent A with the
hash‑driven allocation of Parent B.

The hybrid cost and reward are finally


C_h = Σₑ pₑ·ℓ(e) + λ Σᵥ qᵥ·d(v)
R_h = Σₐ qₐ·r(a)                (a = action = node)


The implementation below provides three core functions that realise this
fusion:
1. `compute_phash` – perceptual hash of a numeric vector (Parent B).
2. `posterior_edge_beliefs` – simple Bayesian update yielding *pₑ* (Parent A).
3. `hybrid_cost_reward` – computes `C_h` and `R_h` using the bridge described
   above.

A small smoke test demonstrates the end‑to‑end workflow.
"""

import math
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Parent B – perceptual hash utilities
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
K = len(GROUPS)


def _pct(value: float) -> float:
    """Round a float to six decimal places (utility from Parent B)."""
    return round(float(value), 6)


def compute_phash(values: List[float]) -> int:
    """
    64‑bit perceptual hash of a numeric sequence.

    A bit *i* is set to 1 iff `values[i] >= mean(first_64_values)`.
    If fewer than 64 values are supplied the sequence is padded with its mean.
    """
    if not values:
        raise ValueError("cannot hash empty sequence")
    # Ensure length 64
    arr = np.array(values, dtype=float)
    if arr.size < 64:
        mean_val = arr.mean()
        pad = np.full(64 - arr.size, mean_val, dtype=float)
        arr = np.concatenate([arr, pad])
    else:
        arr = arr[:64]

    mean_val = arr.mean()
    bits = (arr >= mean_val).astype(np.uint64)
    # Pack bits into a single integer (least‑significant bit = first element)
    hash_int = np.packbits(bits[::-1])[0].astype(np.uint64)
    # The above gives only 8 bits; we need 64 – we therefore build manually
    hash_int = np.uint64(0)
    for i, b in enumerate(bits):
        if b:
            hash_int |= np.uint64(1) << np.uint64(i)
    return int(hash_int)


def group_from_hash(hash_int: int) -> str:
    """Map a 64‑bit hash to one of the predefined groups."""
    idx = hash_int % K
    return GROUPS[idx]


# ----------------------------------------------------------------------
# Parent A – tree utilities
# ----------------------------------------------------------------------
def euclidean_length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_adjacency(edges: List[Edge]) -> Dict[str, List[str]]:
    """Return adjacency list for an undirected edge set."""
    adj: Dict[str, List[str]] = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def root_to_node_distances(
    nodes: Dict[str, Point],
    adj: Dict[str, List[str]],
    root: str,
) -> Dict[str, float]:
    """
    Breadth‑first traversal from *root* computing cumulative Euclidean distance
    to every reachable node.
    """
    distances: Dict[str, float] = {root: 0.0}
    visited = {root}
    queue = [root]

    while queue:
        cur = queue.pop(0)
        for nxt in adj[cur]:
            if nxt in visited:
                continue
            edge_len = euclidean_length(nodes[cur], nodes[nxt])
            distances[nxt] = distances[cur] + edge_len
            visited.add(nxt)
            queue.append(nxt)
    return distances


def posterior_edge_beliefs(
    edges: List[Edge],
    nodes: Dict[str, Point],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> Dict[Edge, float]:
    """
    Very simple Bayesian update for each edge length ℓ(e).

    Assume a Gamma prior on the (positive) cost and observe a single noisy
    measurement equal to the Euclidean length.  The posterior mean of the cost
    (which we treat as the belief pₑ) is:

        pₑ = (α + ℓ) / (α + β + 1)

    This mirrors the Bayesian evidence update of Parent A while staying
    lightweight.
    """
    beliefs: Dict[Edge, float] = {}
    for u, v in edges:
        length = euclidean_length(nodes[u], nodes[v])
        posterior_mean = (prior_alpha + length) / (prior_alpha + prior_beta + 1.0)
        # Store both orientations for convenience
        beliefs[(u, v)] = posterior_mean
        beliefs[(v, u)] = posterior_mean
    return beliefs


# ----------------------------------------------------------------------
# Hybrid core – integrating both parents
# ----------------------------------------------------------------------
def node_group_weights(
    nodes: Dict[str, Point],
) -> Dict[str, float]:
    """
    Compute a work‑share weight for each node based on its perceptual hash.

    The weight is the relative frequency of the node's group among all nodes.
    """
    group_counts: Dict[str, int] = defaultdict(int)
    node_groups: Dict[str, str] = {}

    for nid, coord in nodes.items():
        # Feature vector = (x, y)
        ph = compute_phash([coord[0], coord[1]])
        grp = group_from_hash(ph)
        node_groups[nid] = grp
        group_counts[grp] += 1

    total = sum(group_counts.values())
    group_weights = {g: cnt / total for g, cnt in group_counts.items()}

    # Assign each node the weight of its group
    node_weights = {nid: group_weights[node_groups[nid]] for nid in nodes}
    return node_weights


def hybrid_cost_reward(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    lam: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute the hybrid cost C_h and reward R_h.

    Steps
    -----
    1. Posterior edge beliefs pₑ (Parent A).
    2. Root‑to‑node distances d(v) (Parent A).
    3. Node group work‑share weights w_v (Parent B).
    4. Node belief qᵥ = ( Σ_{e∈Inc(v)} pₑ ) · w_v  then normalised.
    5. Hybrid cost   C_h = Σₑ pₑ·ℓ(e) + λ Σᵥ qᵥ·d(v)
    6. Hybrid reward R_h = Σᵥ qᵥ·r(v)  where r(v)=1 (unit reward per activation).

    Returns
    -------
    (C_h, R_h) as floats.
    """
    # 1. Edge posteriors
    p_edge = posterior_edge_beliefs(edges, nodes)

    # 2. Distances from root
    adj = build_adjacency(edges)
    dists = root_to_node_distances(nodes, adj, root)

    # 3. Work‑share weights from perceptual hash
    w_node = node_group_weights(nodes)

    # 4. Compute raw node beliefs
    raw_q: Dict[str, float] = {}
    for v in nodes:
        incident_sum = sum(p_edge[(v, nb)] for nb in adj[v])
        raw_q[v] = incident_sum * w_node[v]

    # Normalise to obtain qᵥ
    total_q = sum(raw_q.values())
    if total_q == 0:
        # Avoid division by zero – fall back to uniform belief
        q_node = {v: 1.0 / len(nodes) for v in nodes}
    else:
        q_node = {v: val / total_q for v, val in raw_q.items()}

    # 5. Hybrid cost
    edge_cost = sum(p_edge[(u, v)] * euclidean_length(nodes[u], nodes[v]) for u, v in edges) / 2.0
    # (divide by 2 because each undirected edge appears twice in p_edge)
    node_term = lam * sum(q_node[v] * dists[v] for v in nodes)
    C_h = edge_cost + node_term

    # 6. Hybrid reward (unit reward per node)
    R_h = sum(q_node[v] * 1.0 for v in nodes)  # simplifies to 1.0 after normalisation
    return C_h, R_h


# ----------------------------------------------------------------------
# Additional demonstration functions
# ----------------------------------------------------------------------
def random_tree(num_nodes: int, seed: int = 42) -> Tuple[Dict[str, Point], List[Edge], str]:
    """
    Generate a random geometric tree with *num_nodes* vertices.
    Returns (nodes, edges, root_id).
    """
    random.seed(seed)
    np.random.seed(seed)

    nodes: Dict[str, Point] = {}
    for i in range(num_nodes):
        nid = f"n{i}"
        # Random point in unit square
        nodes[nid] = (random.random(), random.random())

    # Build a random spanning tree using a simple Kruskal‑like approach
    all_edges = [(u, v) for u in nodes for v in nodes if u < v]
    random.shuffle(all_edges)

    parent = {nid: nid for nid in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            return True
        return False

    edges: List[Edge] = []
    for u, v in all_edges:
        if union(u, v):
            edges.append((u, v))
        if len(edges) == num_nodes - 1:
            break

    root_id = list(nodes.keys())[0]
    return nodes, edges, root_id


def demo_hybrid():
    """Run a quick demo on a small random tree."""
    nodes, edges, root = random_tree(num_nodes=8, seed=7)
    cost, reward = hybrid_cost_reward(nodes, edges, root, lam=0.3)
    print("Hybrid cost (C_h):", _pct(cost))
    print("Hybrid reward (R_h):", _pct(reward))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_hybrid()