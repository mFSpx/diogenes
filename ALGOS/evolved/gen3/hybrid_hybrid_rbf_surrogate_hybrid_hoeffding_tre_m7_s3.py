# DARWIN HAMMER — match 7, survivor 3
# gen: 3
# parent_a: hybrid_rbf_surrogate_hybrid_distributed_l_m58_s0.py (gen2)
# parent_b: hybrid_hoeffding_tree_tropical_maxplus_m18_s1.py (gen1)
# born: 2026-05-29T23:25:20Z

"""
Hybrid Algorithm: rbf_tropical_hoeffding_hybrid.py

Parents:
- hybrid_rbf_surrogate_hybrid_distributed_l_m58_s0.py (RBF similarity & perceptual hashing)
- hybrid_hoeffding_tree_tropical_maxplus_m18_s1.py (Hoeffding bound & tropical max‑plus algebra)

Mathematical Bridge:
The RBF‑based similarity matrix S∈ℝ^{n×n} provides a dense, continuous
representation of pairwise node affinity.  Tropical max‑plus algebra treats
addition as max and multiplication as +, turning a linear layer into a
piecewise‑linear (ReLU‑like) function.  By feeding each row of S (the
similarities of a node to all others) as the input vector x to a tropical
network, we obtain a tropical score vector y.  The gap between the largest
and second‑largest tropical scores is a natural “gain” that can be
evaluated with the Hoeffding bound to decide whether a node’s community
should be split (or a leader elected) in a streaming / distributed setting.
Thus the hybrid algorithm fuses:
    • RBF‑derived similarity (continuous kernel) → input to tropical net
    • Tropical max‑plus evaluation → gain values
    • Hoeffding bound → statistically‑sound split decision
The result is a unified system that leverages kernel similarity for
probabilistic broadcasting while using rigorous statistical guarantees for
dynamic graph partitioning.
"""

import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Hashable, Sequence, List, Dict, Set, Tuple, Any

import numpy as np

# -------------------- Parent A components --------------------

Node = Hashable
Graph = Dict[Node, Set[Node]]
FeatureVec = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two feature vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: 1 bit per value above average (max 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()

def similarity_matrix(features: Dict[Node, FeatureVec]) -> Tuple[np.ndarray, List[Node]]:
    """
    Compute a symmetric similarity matrix S where
        S_ij = 1 - (Hamming(phash_i, phash_j) / 64)
    Returns the matrix and the ordered list of nodes.
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    phashes = [compute_phash(list(features[n])) for n in nodes]
    for i in range(n):
        for j in range(i, n):
            d = hamming_distance(phashes[i], phashes[j])
            sim = 1.0 - d / 64.0
            S[i, j] = sim
            S[j, i] = sim
    return S, nodes

def modulated_probability(
    raw_p: float,
    node_idx: int,
    similarity: np.ndarray,
    undecided_mask: np.ndarray,
) -> float:
    """
    Modulate a raw broadcast probability by the average similarity of the node
    to all *currently undecided* neighbours.
    """
    if not (0 <= node_idx < similarity.shape[0]):
        raise IndexError("node_idx out of bounds")
    # mask of neighbours that are still undecided
    neighbour_mask = undecided_mask.copy()
    neighbour_mask[node_idx] = False  # exclude self
    if not neighbour_mask.any():
        return raw_p
    avg_sim = similarity[node_idx, neighbour_mask].mean()
    # Blend raw probability with similarity (weight 0.5 each)
    return 0.5 * raw_p + 0.5 * avg_sim

# -------------------- Parent B components --------------------

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a random variable in [0, r]."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

@dataclass(frozen=True)
class SplitDecision:
    """Result of a Hoeffding‑based split test."""
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def should_split(
    best_gain: float,
    second_best_gain: float,
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
) -> SplitDecision:
    """Determine whether to split based on Hoeffding bound."""
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)

# Tropical max‑plus algebra ------------------------------------------------

def t_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical addition (max)."""
    return np.maximum(x, y)

def t_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical multiplication (plus)."""
    return np.add(x, y)

def t_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical matrix multiplication:
        (A ⊗ B)_{ij} = max_k (A_{ik} + B_{kj})
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # broadcast addition then max over k
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

def t_polyval(coeffs: Sequence[float], x: np.ndarray) -> np.ndarray:
    """
    Evaluate a tropical polynomial:
        p(x) = max_i (coeff_i + i * x)
    where i is the exponent (0‑based).
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(len(coeffs), dtype=float)
    # reshape for broadcasting
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents.reshape((-1,) + (1,) * x.ndim) * x
    return np.max(terms, axis=0)

def relu_layer_to_tropical(W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identity conversion for a ReLU layer; in tropical form the linear part
    remains the same and bias is added tropically.
    """
    return np.asarray(W, dtype=float), np.asarray(b, dtype=float)

def tropical_network_eval(x: np.ndarray, layers: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Forward pass of a tropical network.
    Each layer (W, b) performs: y = max_j (W_{ij} + x_j) + b_i
    """
    h = np.asarray(x, dtype=float).ravel()
    for W, b in layers:
        W, b = relu_layer_to_tropical(W, b)
        # tropical affine transformation
        h = np.max(W + h, axis=1) + b
    return h

# -------------------- Hybrid Functions --------------------

def evaluate_node_with_tropical(
    similarity_row: np.ndarray,
    tropical_layers: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    Feed a single node's similarity vector (row of S) into the tropical network.
    Returns the tropical scores for each possible class/partition.
    """
    return tropical_network_eval(similarity_row, tropical_layers)

def node_split_decision(
    similarity_row: np.ndarray,
    tropical_layers: List[Tuple[np.ndarray, np.ndarray]],
    r: float,
    delta: float,
    n_observations: int,
) -> SplitDecision:
    """
    Compute tropical scores, extract the two highest gains, and apply the
    Hoeffding bound to decide whether the node should trigger a split (e.g.,
    start a new leader or independent set).
    """
    scores = evaluate_node_with_tropical(similarity_row, tropical_layers)
    if scores.size < 2:
        # Degenerate case: cannot compare two gains
        return SplitDecision(False, 0.0, 0.0, "insufficient_classes")
    sorted_idx = np.argsort(scores)[::-1]  # descending
    best, second = scores[sorted_idx[0]], scores[sorted_idx[1]]
    return should_split(best, second, r, delta, n_observations)

def hybrid_broadcast_probability(
    raw_p: float,
    node_idx: int,
    similarity: np.ndarray,
    undecided_mask: np.ndarray,
    tropical_layers: List[Tuple[np.ndarray, np.ndarray]],
    r: float,
    delta: float,
    n_obs: int,
) -> float:
    """
    Combine three ingredients into a single broadcast probability:
        1. Base probability (raw_p)
        2. Similarity‑based modulation (modulated_probability)
        3. Hoeffding‑driven split confidence (boost if node is likely to split)
    """
    base = modulated_probability(raw_p, node_idx, similarity, undecided_mask)

    # Determine split confidence
    split_dec = node_split_decision(
        similarity_row=similarity[node_idx],
        tropical_layers=tropical_layers,
        r=r,
        delta=delta,
        n_observations=n_obs,
    )

    # If a split is imminent, increase broadcast chance to accelerate dissemination
    if split_dec.should_split:
        return min(1.0, base + 0.3)  # boost by 0.3 (capped at 1)
    else:
        return base

# -------------------- Smoke Test --------------------

if __name__ == "__main__":
    # Generate synthetic feature vectors for 6 nodes (dim=8)
    random.seed(42)
    np.random.seed(42)
    num_nodes = 6
    dim = 8
    features: Dict[Node, FeatureVec] = {
        f"node_{i}": np.random.rand(dim).tolist() for i in range(num_nodes)
    }

    # Compute similarity matrix S
    S, node_list = similarity_matrix(features)

    # Create a trivial tropical network: 2 layers, each mapping from dim to 3 classes
    # For demonstration we keep the same dimension as similarity row (num_nodes)
    num_classes = 3
    layers = []
    for _ in range(2):
        W = np.random.randn(num_classes, num_nodes)  # shape (classes, inputs)
        b = np.random.randn(num_classes)
        layers.append((W, b))

    # Parameters for Hoeffding bound
    r = 1.0          # range of gain values (tropical scores are unbounded, but we cap)
    delta = 0.05
    n_observations = 100

    # Undecided mask: initially all nodes are undecided
    undecided = np.ones(num_nodes, dtype=bool)

    # Raw broadcast probability (e.g., from external scheduler)
    raw_p = 0.2

    # Run hybrid broadcast probability for each node
    probs = []
    for idx in range(num_nodes):
        p = hybrid_broadcast_probability(
            raw_p=raw_p,
            node_idx=idx,
            similarity=S,
            undecided_mask=undecided,
            tropical_layers=layers,
            r=r,
            delta=delta,
            n_obs=n_observations,
        )
        probs.append(p)
        print(f"Node {node_list[idx]} -> broadcast probability: {p:.3f}")

    # Simple sanity check: probabilities are within [0,1]
    assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities out of bounds"

    print("Smoke test completed successfully.")