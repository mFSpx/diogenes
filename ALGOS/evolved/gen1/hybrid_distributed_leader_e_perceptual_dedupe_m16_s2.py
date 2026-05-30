# DARWIN HAMMER — match 16, survivor 2
# gen: 1
# parent_a: distributed_leader_election.py (gen0)
# parent_b: perceptual_dedupe.py (gen0)
# born: 2026-05-29T23:20:19Z

"""Hybrid algorithm combining distributed leader election (maximal independent set)
and perceptual hashing clustering.

Mathematical bridge:
- Each node n in a graph G=(V,E) carries a numeric feature vector f_n.
- From f_n we compute a perceptual hash h_n (64‑bit integer) using the
  phash routine of *perceptual_dedupe.py*.
- The Hamming distance d(h_i,h_j) defines a similarity weight
  w_ij = 1 - d(h_i,h_j)/64 ∈ [0,1].
- In the broadcast phase of the MIS algorithm (*distributed_leader_election.py*)
  the raw broadcast probability p_raw = 1/2^{phase-step} is modulated by the
  average similarity of a candidate node to its undecided neighbours:
        p_mod = p_raw * avg_{j∈N(i)∩U} w_ij .
  Thus nodes that are perceptually similar to many neighbours broadcast
  less aggressively, encouraging diversity among elected leaders.

The hybrid functions below implement:
1. hashing of node attributes,
2. construction of a similarity matrix from Hamming distances,
3. a MIS procedure that uses the similarity‑modulated broadcast probability.

The result is a set of leaders that are both graph‑independent and
perceptually diverse.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Mapping, Hashable, Sequence, List, Dict, Set, Tuple

import numpy as np
import math

Node = Hashable
Graph = Mapping[Node, Set[Node]]
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Perceptual hashing utilities (from perceptual_dedupe.py)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Return a 64‑bit perceptual hash of a list of floats."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    # limit to first 64 values for deterministic size
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers."""
    return (a ^ b).bit_count()

def cluster_by_phash(hashes: Dict[Node, int], max_distance: int = 4) -> List[List[Node]]:
    """Simple greedy clustering based on Hamming distance."""
    clusters: List[List[Node]] = []
    for node, h in hashes.items():
        for c in clusters:
            if hamming_distance(h, hashes[c[0]]) <= max_distance:
                c.append(node)
                break
        else:
            clusters.append([node])
    return clusters

# ----------------------------------------------------------------------
# Similarity matrix construction
# ----------------------------------------------------------------------
def similarity_matrix(hashes: Dict[Node, int]) -> Tuple[np.ndarray, List[Node]]:
    """
    Build a symmetric similarity matrix S where
        S[i, j] = 1 - d(h_i, h_j) / 64
    Returns the matrix and the ordered list of nodes.
    """
    nodes = list(hashes.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = hashes[ni]
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
            else:
                hj = hashes[nj]
                d = hamming_distance(hi, hj)
                S[i, j] = 1.0 - d / 64.0
    return S, nodes

# ----------------------------------------------------------------------
# Broadcast probability with similarity modulation
# ----------------------------------------------------------------------
def broadcast_probability(phase: int, step: int) -> float:
    """Raw broadcast probability p = 1 / 2^{phase-step}, clamped to [0,1]."""
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


def modulated_probability(
    raw_p: float,
    node_idx: int,
    undecided_mask: np.ndarray,
    adjacency: np.ndarray,
    similarity: np.ndarray,
) -> float:
    """
    Scale raw_p by the average similarity of node `node_idx` to its undecided neighbours.
    If the node has no undecided neighbours, the factor is 1.
    """
    neigh_mask = adjacency[node_idx] & undecided_mask
    if not np.any(neigh_mask):
        return raw_p
    avg_sim = similarity[node_idx, neigh_mask].mean()
    return raw_p * avg_sim

# ----------------------------------------------------------------------
# Hybrid maximal independent set
# ----------------------------------------------------------------------
def hybrid_maximal_independent_set(
    graph: Graph,
    features: Mapping[Node, FeatureVec],
    phases: int = 8,
    seed: int | str | None = None,
) -> Set[Node]:
    """
    Compute a maximal independent set where broadcast probabilities are
    modulated by perceptual similarity of node feature vectors.
    """
    # 1. Compute hashes and similarity matrix
    hashes = {n: compute_phash(list(features[n])) for n in graph}
    sim_mat, ordered_nodes = similarity_matrix(hashes)

    # 2. Build adjacency matrix aligned with ordered_nodes
    idx_of = {n: i for i, n in enumerate(ordered_nodes)}
    n = len(ordered_nodes)
    adjacency = np.zeros((n, n), dtype=bool)
    for u, nbrs in graph.items():
        iu = idx_of[u]
        for v in nbrs:
            if v in idx_of:
                iv = idx_of[v]
                adjacency[iu, iv] = True
                adjacency[iv, iu] = True  # undirected

    rng = random.Random(seed)
    undecided = np.ones(n, dtype=bool)   # boolean mask of nodes still in play
    leaders_mask = np.zeros(n, dtype=bool)

    for phase in range(1, phases + 1):
        if not undecided.any():
            break
        raw_p = broadcast_probability(phases, phase)

        # decide which undecided nodes broadcast
        broadcast_flags = np.zeros(n, dtype=bool)
        for i in np.where(undecided)[0]:
            p_i = modulated_probability(raw_p, i, undecided, adjacency, sim_mat)
            if rng.random() < p_i:
                broadcast_flags[i] = True

        # new leaders are broadcast nodes with no broadcasting neighbour
        conflict = adjacency @ broadcast_flags.astype(int)
        new_leaders = broadcast_flags & (conflict == 0)

        # update masks
        leaders_mask |= new_leaders
        # block leaders and their neighbours
        blocked = new_leaders[:, None] | adjacency[new_leaders]
        undecided &= ~blocked.any(axis=0)

    # deterministic cleanup for any remaining undecided nodes
    remaining = np.where(undecided)[0]
    for i in remaining:
        if not (adjacency[i] & leaders_mask).any():
            leaders_mask[i] = True
            undecided[i] = False
            # block its neighbours as well
            undecided &= ~adjacency[i]

    return {ordered_nodes[i] for i, flag in enumerate(leaders_mask) if flag}

# ----------------------------------------------------------------------
# Helper: generate a synthetic graph with feature vectors for demo
# ----------------------------------------------------------------------
def random_geometric_graph(
    num_nodes: int,
    radius: float,
    seed: int | None = None,
) -> Tuple[Graph, Dict[Node, FeatureVec]]:
    """
    Generate a random geometric graph in the unit square.
    Each node gets a 2‑dimensional coordinate as its feature vector.
    """
    rng = random.Random(seed)
    positions = {i: (rng.random(), rng.random()) for i in range(num_nodes)}
    graph: Dict[Node, Set[Node]] = {i: set() for i in range(num_nodes)}
    for i in range(num_nodes):
        xi, yi = positions[i]
        for j in range(i + 1, num_nodes):
            xj, yj = positions[j]
            if (xi - xj) ** 2 + (yi - yj) ** 2 <= radius ** 2:
                graph[i].add(j)
                graph[j].add(i)
    # use the coordinates themselves as the feature vectors
    features = {i: list(pos) for i, pos in positions.items()}
    return graph, features

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a modest graph
    G, feats = random_geometric_graph(num_nodes=50, radius=0.25, seed=42)

    # Run hybrid MIS
    leaders = hybrid_maximal_independent_set(G, feats, phases=6, seed=123)

    print(f"Number of nodes: {len(G)}")
    print(f"Leaders selected: {len(leaders)}")
    # Verify independence
    for u in leaders:
        if any(v in leaders for v in G[u]):
            print("Error: leaders are not independent!", file=sys.stderr)
            sys.exit(1)
    print("Hybrid maximal independent set computed successfully.")