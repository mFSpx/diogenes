# DARWIN HAMMER — match 4097, survivor 7
# gen: 7
# parent_a: hybrid_hybrid_infotaxis_min_hybrid_hybrid_hybrid_m2692_s2.py (gen6)
# parent_b: hybrid_hybrid_nlms_hybrid_h_hybrid_hard_truth_ma_m1061_s2.py (gen5)
# born: 2026-05-29T23:53:42Z

import numpy as np
import math
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
    created_at: float
    last_decay: float

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
        self.last_decay = self.created_at + self.age_seconds()

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
    max_sig = np.max(signals) if signals.size else 1.0
    if max_sig == 0:
        max_sig = 1.0
    weights = np.clip(signals / max_sig, 0, 1)  # w_i ∈ [0,1]

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
    ig_with_pheromone = []
    for p in pheromones:
        idx = uuid_to_idx.get(p.uuid)
        if idx is not None:
            ig_with_pheromone.append((idx, entropy[idx] * p.signal_value))

    # Select node with maximum IG
    if ig_with_pheromone:
        _, max_idx = max(ig_with_pheromone, key=lambda x: x[1])
    else:
        max_idx = np.argmax(entropy)

    return nodes[max_idx]

def main():
    features = {0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]}
    pheromones = [
        PheromoneEntry("0", "", "", 1.0, 10, 0.0, 0.0),
        PheromoneEntry("1", "", "", 0.5, 10, 0.0, 0.0),
    ]
    H, nodes = build_hybrid_kernel(features, pheromones)
    next_node = infotaxis_decision(pheromones, H, nodes)
    print(next_node)

if __name__ == "__main__":
    main()