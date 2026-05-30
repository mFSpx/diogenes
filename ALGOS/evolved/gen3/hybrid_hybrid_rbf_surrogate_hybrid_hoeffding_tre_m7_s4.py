# DARWIN HAMMER — match 7, survivor 4
# gen: 3
# parent_a: hybrid_rbf_surrogate_hybrid_distributed_l_m58_s0.py (gen2)
# parent_b: hybrid_hoeffding_tree_tropical_maxplus_m18_s1.py (gen1)
# born: 2026-05-29T23:25:20Z

"""Hybrid RBF–Tropical Hoeffding Algorithm
Combines:
- Parent A (rbf_surrogate + perceptual similarity) – provides radial basis
  function kernels and a perceptual‑hash similarity matrix S.
- Parent B (Hoeffding bound + tropical max‑plus algebra) – provides statistical
  split decisions and tropical operations (t_add, t_mul, t_matmul).

Mathematical bridge:
The RBF kernel K(i,j)=exp(-ε²‖f_i‑f_j‖²) is a dense similarity measure.
The perceptual similarity matrix S(i,j)∈[0,1] is another similarity.
Both are square matrices of size n (number of nodes).  In tropical algebra the
“multiplication’’ is ordinary addition and the “addition’’ is maximum.
Thus we can fuse the two similarities by a tropical matrix product

    C = K ⊗ S   where   (C)_{ij}= max_k ( K_{ik} + S_{kj} ).

The resulting matrix C encodes a combined similarity that respects both
geometric (RBF) and perceptual (phash) notions.  The Hoeffding bound is then
used on the gains derived from C to decide whether a node should be
“split’’ (i.e., promoted to the maximal independent set) in a streaming
setting.  The hybrid algorithm therefore intertwines the continuous kernel
world with the discrete statistical decision world.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Hashable, Sequence, List, Dict, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Dict[Node, Set[Node]]
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Parent A utilities (RBF & perceptual similarity)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    """Very simple perceptual hash: 1‑bit per value relative to the mean."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers interpreted as bit strings."""
    return (a ^ b).bit_count()


def similarity_matrix(features: Dict[Node, FeatureVec]) -> Tuple[np.ndarray, List[Node]]:
    """
    Perceptual similarity matrix S where S[i,j] = 1 - d/64,
    d = Hamming distance between perceptual hashes of node i and j.
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    hashes = [compute_phash(list(features[n])) for n in nodes]

    for i in range(n):
        for j in range(i, n):
            d = hamming_distance(hashes[i], hashes[j])
            sim = 1.0 - d / 64.0
            S[i, j] = sim
            S[j, i] = sim
    return S, nodes


def rbf_kernel_matrix(features: Dict[Node, FeatureVec], epsilon: float = 1.0) -> Tuple[np.ndarray, List[Node]]:
    """
    Dense RBF kernel K where K[i,j] = exp(-ε² * ||f_i - f_j||²).
    """
    nodes = list(features.keys())
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K, nodes


# ----------------------------------------------------------------------
# Parent B utilities (tropical algebra & Hoeffding bound)
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a range r, confidence 1‑δ, after n samples."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def should_split(best_gain: float, second_best_gain: float,
                 r: float, delta: float, n: int,
                 tie_threshold: float = 0.05) -> SplitDecision:
    """Hoeffding‑based decision whether a candidate split is statistically justified."""
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)


def t_add(x, y):
    """Tropical addition (max)."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication (ordinary addition)."""
    return np.add(x, y)


def t_matmul(A, B):
    """
    Tropical matrix multiplication:
        (A ⊗ B)_{ij} = max_k ( A_{ik} + B_{kj} )
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # Broadcasting to compute all pairwise sums A[i,k] + B[k,j]
    # Result shape (i, j, k) -> take max over k
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)


# ----------------------------------------------------------------------
# Hybrid core functions (mathematical fusion)
# ----------------------------------------------------------------------
def combined_similarity(features: Dict[Node, FeatureVec],
                        epsilon: float = 1.0) -> Tuple[np.ndarray, List[Node]]:
    """
    Compute the tropical combination C = K ⊗ S of the RBF kernel K and the
    perceptual similarity S.  Returns C and the ordered list of nodes.
    """
    K, nodes_k = rbf_kernel_matrix(features, epsilon)
    S, nodes_s = similarity_matrix(features)
    if nodes_k != nodes_s:
        raise ValueError("Node ordering mismatch between K and S")
    C = t_matmul(K, S)          # tropical product
    return C, nodes_k


def node_gain_matrix(combined: np.ndarray) -> np.ndarray:
    """
    Derive a gain for each node from the combined similarity matrix.
    Gain_i = max_j C_{ij}  (the strongest combined similarity of node i).
    """
    return np.max(combined, axis=1)


def hybrid_split_decision(gains: np.ndarray,
                          r: float,
                          delta: float,
                          n: int,
                          tie_threshold: float = 0.05) -> SplitDecision:
    """
    Apply Hoeffding bound on the ordered gains to decide whether the best
    candidate should be split (i.e., added to the independent set).
    """
    if gains.size < 2:
        # Trivial case – only one candidate, always split
        return SplitDecision(True, 0.0, 0.0, "single_candidate")
    sorted_idx = np.argsort(gains)[::-1]  # descending
    best = gains[sorted_idx[0]]
    second = gains[sorted_idx[1]]
    return should_split(best, second, r, delta, n, tie_threshold)


def modulated_probability(raw_p: float,
                          node_idx: int,
                          undecided_mask: np.ndarray,
                          combined: np.ndarray) -> float:
    """
    Modulate a raw broadcast probability by the relative strength of the
    combined similarity of the node to the currently undecided set.
    """
    if not (0.0 <= raw_p <= 1.0):
        raise ValueError("raw_p must be in [0,1]")
    # Consider only undecided neighbours
    row = combined[node_idx]
    relevant = row[undecided_mask]
    if relevant.size == 0:
        return raw_p  # no modulation possible
    # Normalise by max possible value (which is 1 after Gaussian+S scaling)
    modulation = np.mean(relevant)  # average similarity to undecided nodes
    return min(1.0, max(0.0, raw_p * modulation))


def hybrid_max_independent_set(features: Dict[Node, FeatureVec],
                               graph: Graph,
                               raw_p: float = 0.5,
                               epsilon: float = 1.0,
                               r: float = 1.0,
                               delta: float = 0.05,
                               n_samples: int = 100) -> Set[Node]:
    """
    A lightweight demonstration of the hybrid algorithm:
    1. Build the combined similarity matrix C = K ⊗ S.
    2. Compute per‑node gains.
    3. Use Hoeffding‑based split decision to pick a seed node.
    4. Grow an independent set by repeatedly adding the highest‑gain
       undecided node, modulating its broadcast probability with C.
    Returns the set of selected nodes.
    """
    C, nodes = combined_similarity(features, epsilon)
    gains = node_gain_matrix(C)

    # Map node index ↔ node object
    idx_to_node = {i: n for i, n in enumerate(nodes)}
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Initial decision which node becomes the first leader
    decision = hybrid_split_decision(gains, r, delta, n_samples)
    if not decision.should_split:
        # No statistically justified leader; return empty set
        return set()

    # Start with the best node
    leader_idx = int(np.argmax(gains))
    independent_set: Set[Node] = {idx_to_node[leader_idx]}

    # Track which nodes are still undecided
    undecided = np.ones(len(nodes), dtype=bool)
    undecided[leader_idx] = False

    # Iteratively consider remaining nodes
    while undecided.any():
        # Compute modulated probabilities for all undecided nodes
        probs = np.array([
            modulated_probability(raw_p, i, undecided, C) if undecided[i] else 0.0
            for i in range(len(nodes))
        ])

        # Pick candidate with highest (gain * prob) product
        score = gains * probs
        candidate_idx = int(np.argmax(score * undecided))

        # Verify independence: candidate must not be adjacent to any node already in the set
        candidate_node = idx_to_node[candidate_idx]
        if all(candidate_node not in graph[member] for member in independent_set):
            independent_set.add(candidate_node)

        # Mark as decided
        undecided[candidate_idx] = False

    return independent_set


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny random feature set for 6 nodes
    random.seed(42)
    np.random.seed(42)
    nodes = [f"v{i}" for i in range(6)]
    features: Dict[Node, FeatureVec] = {
        n: np.random.rand(5).tolist() for n in nodes
    }

    # Simple undirected graph (complete graph for stress test)
    graph: Graph = {n: set(nodes) - {n} for n in nodes}

    # Run the hybrid MIS algorithm
    mis = hybrid_max_independent_set(features, graph,
                                     raw_p=0.6,
                                     epsilon=0.8,
                                     r=0.5,
                                     delta=0.01,
                                     n_samples=200)

    print("Selected independent set:", mis)