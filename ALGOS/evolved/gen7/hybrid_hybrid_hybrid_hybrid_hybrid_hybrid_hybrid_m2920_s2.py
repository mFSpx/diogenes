# DARWIN HAMMER — match 2920, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1977_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_physar_m1316_s4.py (gen6)
# born: 2026-05-29T23:46:40Z

"""Hybrid Algorithm combining Ollivier‑Ricci curvature‑based regret weighting (Parent A)
and MinHash/Count‑Min sketch with Physarum‑Sheaf dynamics (Parent B).

Mathematical bridge:
- The regret‑weighted strategy of Parent A uses edge weights derived from the
  Ollivier‑Ricci curvature κ(e).  We reinterpret these weights as a “transport
  capacity” in a Physarum‑Sheaf network.
- Parent B supplies MinHash signatures for node feature sets; the Jaccard similarity
  J(i,j) between signatures modulates the effective curvature:
        κ̂(e_{ij}) = κ(e_{ij}) · (1 + α·J(i,j))
  where α∈[0,1] controls the influence of similarity.
- The modified curvatures κ̂ are fed to a Physarum‑Sheaf update that redistributes
  flow f on edges proportionally to κ̂ and to the current Count‑Min sketch counts
  C(e).  The sketch provides a compact estimate of cumulative regret on each edge.

The resulting system simultaneously captures topological curvature, similarity‑driven
information transport, and lightweight streaming statistics.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Edge:
    i: int
    j: int

@dataclass
class Graph:
    adjacency: np.ndarray  # shape (n, n), 0/1 entries
    weights: np.ndarray = field(init=False)  # curvature weights, same shape

    def __post_init__(self):
        n = self.adjacency.shape[0]
        self.weights = np.zeros((n, n), dtype=float)

# ----------------------------------------------------------------------
# Parent A – Ollivier‑Ricci curvature (simplified)
# ----------------------------------------------------------------------
def compute_ollivier_ricci(graph: Graph, steps: int = 3) -> None:
    """Populate graph.weights with a simple Ollivier‑Ricci curvature estimate.

    For each edge (i,j) we compare the 1‑step random‑walk distributions
    from i and j.  The curvature κ is approximated as:
        κ = 1 - W1(p_i, p_j) / d(i,j)
    where W1 is the Earth‑Mover distance (here reduced to total variation)
    and d(i,j)=1 for adjacent nodes.
    """
    n = graph.adjacency.shape[0]
    degree = graph.adjacency.sum(axis=1)
    for i in range(n):
        for j in range(i + 1, n):
            if graph.adjacency[i, j] == 0:
                continue
            # 1‑step walk distributions (normalized neighbor sets)
            pi = np.zeros(n)
            pj = np.zeros(n)
            if degree[i] > 0:
                pi[graph.adjacency[i] == 1] = 1.0 / degree[i]
            if degree[j] > 0:
                pj[graph.adjacency[j] == 1] = 1.0 / degree[j]
            # total variation distance as proxy for Wasserstein‑1
            tv = np.abs(pi - pj).sum() / 2.0
            kappa = 1.0 - tv  # d(i,j)=1
            graph.weights[i, j] = graph.weights[j, i] = kappa

# ----------------------------------------------------------------------
# Parent B – MinHash signature
# ----------------------------------------------------------------------
def minhash_signature(elements: Iterable[int], num_perm: int = 64) -> np.ndarray:
    """Return a MinHash signature vector of length num_perm for a set of integers."""
    max_hash = (1 << 32) - 1
    sig = np.full(num_perm, max_hash, dtype=np.uint32)
    for idx, perm in enumerate(range(num_perm)):
        for e in elements:
            # simple mixed hash: (a*e + b) mod prime
            a = 0x5bd1e995 + perm * 0x12345
            b = 0x27d4eb2d + perm * 0x54321
            h = (a * e + b) & max_hash
            if h < sig[idx]:
                sig[idx] = h
    return sig

def jaccard_estimate(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    return np.mean(sig1 == sig2)

# ----------------------------------------------------------------------
# Parent B – Count‑Min sketch
# ----------------------------------------------------------------------
def init_count_min_sketch(width: int = 256, depth: int = 4) -> List[np.ndarray]:
    return [np.zeros(width, dtype=np.int64) for _ in range(depth)]

def _hash(item: int, seed: int, width: int) -> int:
    h = hashlib.sha256(f"{seed}:{item}".encode()).hexdigest()
    return int(h, 16) % width

def update_count_min_sketch(sketch: List[np.ndarray], items: Iterable[int]) -> None:
    depth = len(sketch)
    width = sketch[0].size
    for item in items:
        for d in range(depth):
            idx = _hash(item, d, width)
            sketch[d][idx] += 1

def query_count_min_sketch(sketch: List[np.ndarray], item: int) -> int:
    depth = len(sketch)
    width = sketch[0].size
    mins = []
    for d in range(depth):
        idx = _hash(item, d, width)
        mins.append(sketch[d][idx])
    return min(mins)

# ----------------------------------------------------------------------
# Physarum‑Sheaf style flow update using curvature and sketch
# ----------------------------------------------------------------------
def physarum_sheaf_flow(
    graph: Graph,
    sketch: List[np.ndarray],
    alpha: float = 0.5,
    epsilon: float = 1e-6,
    iterations: int = 20,
) -> np.ndarray:
    """Compute a flow matrix F where F[i,j]≈transport on edge (i,j).

    The update rule mimics Physarum dynamics:
        f_{ij}^{new} = f_{ij} * (κ̂_{ij} / (c_{ij} + ε))
    where κ̂ incorporates MinHash similarity and c_{ij} is the sketch count.
    The flow is renormalized each iteration to keep total flow constant.
    """
    n = graph.adjacency.shape[0]
    # initialise uniform flow on existing edges
    F = np.where(graph.adjacency == 1, 1.0, 0.0)

    # pre‑compute MinHash signatures for each node (using node id as dummy feature set)
    node_signatures = [minhash_signature({i}, num_perm=64) for i in range(n)]

    for it in range(iterations):
        # compute similarity‑modulated curvature κ̂
        kappa_hat = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                if graph.adjacency[i, j] == 0:
                    continue
                J = jaccard_estimate(node_signatures[i], node_signatures[j])
                kappa = graph.weights[i, j]
                kappa_hat[i, j] = kappa_hat[j, i] = kappa * (1.0 + alpha * J)

        # sketch‑based resistance c_{ij}
        c = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                if graph.adjacency[i, j] == 0:
                    continue
                # use edge id i*n + j as deterministic item for sketch query
                edge_id = i * n + j
                c[i, j] = c[j, i] = query_count_min_sketch(sketch, edge_id) + epsilon

        # Physarum update
        F = F * (kappa_hat / (c + epsilon))

        # renormalize total flow to 1.0
        total = F.sum()
        if total > 0:
            F /= total

    return F

# ----------------------------------------------------------------------
# Hybrid feature extraction using regret‑weighted curvature and sketch
# ----------------------------------------------------------------------
def hybrid_feature_vector(
    node_idx: int,
    graph: Graph,
    sketch: List[np.ndarray],
    alpha: float = 0.5,
) -> np.ndarray:
    """Produce a feature vector for a node that blends curvature‑based regret
    and streaming statistics.

    Components:
        - weighted sum of neighboring curvatures (regret signal)
        - normalized sketch counts for incident edges
        - MinHash signature of the node (capturing set‑like context)
    """
    n = graph.adjacency.shape[0]

    # regret signal: sum_{j} κ̂_{ij}
    sig = np.zeros(64, dtype=np.uint32)  # placeholder for MinHash part
    node_sig = minhash_signature({node_idx}, num_perm=64)
    sig = node_sig.astype(float)

    # curvature part
    curvature_sum = 0.0
    for j in range(n):
        if graph.adjacency[node_idx, j] == 0:
            continue
        # similarity J between node and neighbor
        neighbor_sig = minhash_signature({j}, num_perm=64)
        J = jaccard_estimate(node_sig, neighbor_sig)
        kappa = graph.weights[node_idx, j]
        curvature_sum += kappa * (1.0 + alpha * J)

    # sketch part
    sketch_vals = []
    for j in range(n):
        if graph.adjacency[node_idx, j] == 0:
            continue
        edge_id = node_idx * n + j
        sketch_vals.append(query_count_min_sketch(sketch, edge_id))
    if sketch_vals:
        sketch_arr = np.array(sketch_vals, dtype=float)
        sketch_norm = sketch_arr / (sketch_arr.sum() + 1e-9)
    else:
        sketch_norm = np.zeros(1)

    # concatenate
    feature = np.concatenate(
        [
            np.array([curvature_sum]),
            sketch_norm,
            sig,
        ]
    )
    return feature

# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
def _build_random_graph(num_nodes: int = 10, edge_prob: float = 0.3) -> Graph:
    rng = np.random.default_rng(seed=42)
    adj = rng.random((num_nodes, num_nodes))
    adj = (adj + adj.T) / 2.0  # symmetrize
    adj = (adj < edge_prob).astype(int)
    np.fill_diagonal(adj, 0)
    return Graph(adjacency=adj)

if __name__ == "__main__":
    # construct graph
    g = _build_random_graph(num_nodes=12, edge_prob=0.25)
    # compute curvature (Parent A)
    compute_ollivier_ricci(g)

    # initialise sketch (Parent B)
    cms = init_count_min_sketch(width=256, depth=4)

    # simulate streaming of edge IDs as "regret events"
    edge_ids = [i * g.adjacency.shape[0] + j for i in range(g.adjacency.shape[0])
                for j in range(i + 1, g.adjacency.shape[0]) if g.adjacency[i, j]]
    random.shuffle(edge_ids)
    update_count_min_sketch(cms, edge_ids[:50])  # feed first 50 events

    # run Physarum‑Sheaf flow hybrid (core hybrid operation)
    flow = physarum_sheaf_flow(g, cms, alpha=0.3, iterations=10)
    print("Flow matrix (non‑zero entries):")
    nz = np.where(flow > 1e-8)
    for i, j in zip(*nz):
        print(f"  ({i}, {j}) -> {flow[i, j]:.4f}")

    # extract a hybrid feature vector for node 0
    fv = hybrid_feature_vector(0, g, cms, alpha=0.3)
    print("\nHybrid feature vector for node 0 (length {}):".format(len(fv)))
    print(fv[:10], "...")  # show first few entries