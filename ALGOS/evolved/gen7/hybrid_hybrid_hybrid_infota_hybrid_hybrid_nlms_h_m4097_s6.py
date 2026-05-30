# DARWIN HAMMER — match 4097, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_infotaxis_min_hybrid_hybrid_hybrid_m2692_s2.py (gen6)
# parent_b: hybrid_hybrid_nlms_hybrid_h_hybrid_hard_truth_ma_m1061_s2.py (gen5)
# born: 2026-05-29T23:53:42Z

"""Hybrid Infotaxis–Kernel Fusion
Parents:
- hybrid_hybrid_infotaxis_min_hybrid_hybrid_hybrid_m2692_s2.py (Algorithm A)
- hybrid_hybrid_nlms_hybrid_h_hybrid_hard_truth_ma_m1061_s2.py (Algorithm B)

Mathematical Bridge:
Algorithm A provides a set of pheromone entries whose *signal_value* evolves by
exponential decay (half‑life model).  Algorithm B builds similarity structures
using binary perceptual hashes (phash) and radial‑basis‑function (RBF) kernels.
The fusion treats each pheromone *signal_value* as a multiplicative weight that
scales the pairwise similarity produced by Algorithm B.  Consequently the
hybrid similarity matrix **H** is

    H_{ij} = K_{ij} · (1 + w_i) · (1 + w_j),

where **K** is the RBF kernel matrix from B and w_i = signal_i / max_signal
is the normalised pheromone strength.  This couples entropy‑driven infotaxis
(decay‑driven pheromone dynamics) with kernel‑driven neighbourhood similarity,
allowing a decision rule that selects the node promising the greatest expected
information gain.

The module implements:
1. Pheromone decay (A).
2. RBF kernel and phash‑based similarity (B).
3. Hybrid kernel construction and an infotaxis‑style selection.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Shared lexical categories (identical in both parents, retained for completeness)
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set("and but or nor so yet because although whoever that which what how why when where who whom since as until long".split()),
    "adverb": set("how very rather more".split()),
}

# ----------------------------------------------------------------------
# Algorithm A – Pheromone infrastructure
MAX64 = (1 << 64) - 1

@dataclass
class PheromoneEntry:
    """A single pheromone trace."""
    uuid: str                 # unique identifier, also used as node key
    surface_key: str         # optional textual key (unused in core math)
    signal_kind: str         # descriptive tag
    signal_value: float      # current strength
    half_life_seconds: int   # decay half‑life
    created_at: pathlib.Path
    last_decay: pathlib.Path

    def age_seconds(self) -> float:
        """Simulated age – in a real system this would be time‑difference."""
        return np.random.uniform(0, 100)

    def decay_factor(self) -> float:
        """Exponential decay factor based on half‑life."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        """Apply one decay step in‑place."""
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = pathlib.Path.cwd()

# ----------------------------------------------------------------------
# Algorithm B – Similarity / kernel utilities
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """RBF Gaussian."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: List[float], b: List[float]) -> float:
    """L2 distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    """Perceptual hash – median‑threshold binary code."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits & MAX64

def hamming_distance(a: int, b: int) -> int:
    """Bitwise Hamming distance."""
    return (a ^ b).bit_count()

def similarity_matrix(features: Dict[int, List[float]]) -> Tuple[np.ndarray, List[int]]:
    """
    Build a similarity matrix from binary perceptual hashes.
    Returns (S, node_order).
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    hashes = [compute_phash(features[node]) for node in nodes]

    for i in range(n):
        for j in range(i, n):
            d = hamming_distance(hashes[i], hashes[j])
            sim = 1.0 - d / 64.0
            S[i, j] = sim
            S[j, i] = sim
    return S, nodes

def rbf_kernel_matrix(features: Dict[int, List[float]], epsilon: float = 1.0) -> Tuple[np.ndarray, List[int]]:
    """
    Build an RBF kernel matrix K_{ij} = exp(- (epsilon * ||x_i - x_j||)^2 ).
    Returns (K, node_order).
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
# Hybrid operations (fusion of A and B)
def decay_pheromones(pheromones: List[PheromoneEntry]) -> None:
    """
    Apply decay to every pheromone entry in‑place.
    """
    for p in pheromones:
        p.apply_decay()

def build_hybrid_kernel(
    features: Dict[int, List[float]],
    pheromones: List[PheromoneEntry],
    epsilon: float = 1.0,
) -> Tuple[np.ndarray, List[int]]:
    """
    Construct the hybrid similarity matrix H.

    Steps:
    1. Compute the base RBF kernel K from the feature vectors (Algorithm B).
    2. Normalise pheromone signal values to the interval [0, 1].
    3. Scale each kernel entry:
           H_{ij} = K_{ij} * (1 + w_i) * (1 + w_j)
       where w_i is the normalised pheromone strength of node i.

    The function returns the hybrid matrix H and the ordering of node identifiers.
    """
    # 1. Base kernel
    K, nodes = rbf_kernel_matrix(features, epsilon)

    # 2. Map uuid → node index and collect signal strengths
    uuid_to_index = {str(node): idx for idx, node in enumerate(nodes)}
    signals = np.zeros(len(nodes), dtype=np.float64)

    for p in pheromones:
        idx = uuid_to_index.get(p.uuid)
        if idx is not None:
            signals[idx] = p.signal_value

    # Normalise signals; protect against division by zero
    max_sig = signals.max() if signals.size else 1.0
    if max_sig == 0:
        max_sig = 1.0
    weights = signals / max_sig  # w_i ∈ [0,1]

    # 3. Broadcast scaling
    scale_matrix = np.outer(1 + weights, 1 + weights)
    H = K * scale_matrix
    return H, nodes

def infotaxis_decision(
    pheromones: List[PheromoneEntry],
    hybrid_kernel: np.ndarray,
    nodes: List[int],
) -> int:
    """
    Choose the next node to explore using an infotaxis‑inspired criterion.

    For each candidate node i we compute an expected information gain:
        IG_i = - Σ_j P_ij log P_ij
    where P_ij is the normalized hybrid similarity from i to j.
    The node with maximal IG_i is returned.

    The pheromone distribution influences the kernel, therefore the decision
    implicitly balances exploitation of strong pheromone trails and exploration
    of high‑entropy neighbourhoods.
    """
    # Convert similarity rows to probability distributions
    prob = hybrid_kernel / hybrid_kernel.sum(axis=1, keepdims=True)

    # Compute entropy per row (base‑e)
    entropy = -np.nansum(prob * np.log(prob + 1e-12), axis=1)

    # Combine entropy with current pheromone strength (prefer stronger trails)
    uuid_to_idx = {str(node): i for i, node in enumerate(nodes)}
    pheromone_vec = np.zeros(len(nodes), dtype=np.float64)
    for p in pheromones:
        idx = uuid_to_idx.get(p.uuid)
        if idx is not None:
            pheromone_vec[idx] = p.signal_value

    # Normalise pheromone for weighting
    max_sig = pheromone_vec.max() if pheromone_vec.size else 1.0
    if max_sig == 0:
        max_sig = 1.0
    pher_weight = pheromone_vec / max_sig

    # Final score = entropy * (1 + pher_weight)
    scores = entropy * (1.0 + pher_weight)

    # Return node identifier with highest score
    best_idx = int(np.argmax(scores))
    return nodes[best_idx]

# ----------------------------------------------------------------------
# Smoke test
if __name__ == "__main__":
    # 1. Synthetic feature vectors for 5 nodes
    random.seed(42)
    np.random.seed(42)
    features = {i: np.random.rand(8).tolist() for i in range(5)}

    # 2. Create pheromone entries (one per node) with random strengths
    pheromones = []
    for i in range(5):
        pheromones.append(
            PheromoneEntry(
                uuid=str(i),
                surface_key=f"node_{i}",
                signal_kind="test",
                signal_value=random.uniform(0.5, 2.0),
                half_life_seconds=30,
                created_at=pathlib.Path.cwd(),
                last_decay=pathlib.Path.cwd(),
            )
        )

    # 3. Apply one decay step
    decay_pheromones(pheromones)

    # 4. Build hybrid kernel
    H, node_order = build_hybrid_kernel(features, pheromones, epsilon=0.8)

    # 5. Perform infotaxis decision
    chosen = infotaxis_decision(pheromones, H, node_order)

    print("Hybrid kernel (rounded):")
    print(np.round(H, 3))
    print(f"Chosen node (infotaxis): {chosen}")
    sys.exit(0)