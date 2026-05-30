# DARWIN HAMMER — match 4542, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_fisher_m2240_s0.py (gen6)
# parent_b: hybrid_hybrid_infotaxis_min_hybrid_hybrid_distri_m20_s1.py (gen3)
# born: 2026-05-29T23:56:31Z

"""Hybrid RBF‑Fisher‑MinHash Fusion

Parents:
- hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_fisher_m2240_s0.py (RBF kernel + Fisher‑information weighting)
- hybrid_hybrid_infotaxis_min_hybrid_hybrid_distri_m20_s1.py (MinHash entropy + burst admission scoring)

Mathematical bridge:
Algorithm A supplies a dense RBF kernel matrix **K** whose statistical moments (mean μ and
standard deviation σ) define a Gaussian‑style Fisher‑information scalar  
`I = 1/(2·σ²)`.  
Algorithm B supplies an entropy measure **H** derived from a MinHash signature of a
probability distribution (here approximated by Shannon entropy).  

The hybrid algorithm treats **I** as an *uncertainty weight* that modulates the
entropy‑driven burst‑admission score of Algorithm B, and simultaneously uses **I**
to weight the perceptual‑hash similarity derived from **K**.  The result is a
single hybrid similarity matrix that blends geometric (RBF), perceptual (phash)
and information‑theoretic (entropy, Fisher) perspectives.
"""

import math
import random
import sys
import pathlib
from typing import Dict, List, Sequence, Tuple

import numpy as np

Node = object
FeatureVec = Sequence[float]


def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def compute_rbf_kernel(
    features: Dict[Node, FeatureVec], epsilon: float = 1.0
) -> Tuple[np.ndarray, List[Node]]:
    """
    Build a dense RBF kernel matrix K where
        K[i, j] = exp(-ε² * ||f_i - f_j||²)

    Returns:
        K (n×n ndarray) and the ordered list of nodes.
    """
    nodes = list(features.keys())
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        K[i, i] = 1.0
        for j in range(i + 1, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            val = gaussian_rbf(dist, epsilon)
            K[i, j] = K[j, i] = val
    return K, nodes


def fisher_information_from_kernel(K: np.ndarray) -> float:
    """
    Derive a scalar Fisher‑information value from the kernel matrix.
    Uses the kernel entries as a proxy for distances: μ = mean(K), σ = std(K),
    and I = 1 / (2·σ²).  If σ is zero (degenerate case) a small epsilon is used.
    """
    mu = np.mean(K)
    sigma = np.std(K)
    eps = 1e-12
    sigma = max(sigma, eps)
    I = 1.0 / (2.0 * sigma ** 2)
    return I


def compute_phash(values: List[float]) -> int:
    """
    Simple perceptual hash: 1‑bit per value relative to the median.
    Up to 64 bits are used; excess values are ignored.
    """
    if not values:
        return 0
    median = np.median(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= median)
    return bits


def phash_similarity_matrix(K: np.ndarray) -> np.ndarray:
    """
    Convert each row of K into a perceptual hash and compute the
    pairwise Hamming similarity (normalized to [0,1]).
    """
    n = K.shape[0]
    hashes = [compute_phash(K[i, :].tolist()) for i in range(n)]
    sim = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            ham = (hashes[i] ^ hashes[j]).bit_count()
            # similarity = 1 - normalized Hamming distance
            s = 1.0 - ham / 64.0
            sim[i, j] = sim[j, i] = s
    return sim


def shingle_probabilities(probs: List[float], k: int = 128) -> List[int]:
    """
    Approximate a MinHash signature for a probability distribution.
    For reproducibility a deterministic pseudo‑random permutation is built
    from a fixed seed derived from the index.
    """
    signature = []
    for i in range(k):
        # deterministic hash based on index i
        rng = random.Random(i)
        permuted = [rng.random() for _ in probs]
        min_idx = min(range(len(permuted)), key=lambda idx: permuted[idx] / (probs[idx] + 1e-12))
        signature.append(min_idx)
    return signature


def entropy_of_distribution(probs: List[float]) -> float:
    """
    Shannon entropy of a discrete probability distribution.
    Probabilities are renormalised to sum to 1 if necessary.
    """
    total = sum(probs)
    if total == 0:
        return 0.0
    norm = [p / total for p in probs]
    ent = -sum(p * math.log(p + 1e-12) for p in norm)
    return ent


def burst_admission_score(cluster: List[float], entropy_val: float, fisher_weight: float,
                          m_head: float = 1.0, drag_cd: float = 0.3,
                          fluid_density: float = 1000.0, area: float = 0.001,
                          v0: float = 0.0) -> float:
    """
    Hybrid burst admission score.
    Core idea from the Chelydrid ambush model, but the uncertainty term
    (entropy) is weighted by the Fisher information I.
    """
    # Simplified drag force term
    drag = 0.5 * fluid_density * drag_cd * area * (v0 ** 2 + 1e-12)
    # Base kinetic term proportional to cluster mass (sum of probabilities)
    kinetic = m_head * sum(cluster)
    # Entropy‑Fisher coupling
    uncertainty = entropy_val * fisher_weight
    # Final score (higher = more likely to admit the burst)
    score = (kinetic / (drag + 1e-12)) * uncertainty
    return score


def hybrid_similarity(
    features: Dict[Node, FeatureVec],
    cluster_probs: List[float],
    epsilon: float = 1.0,
    k_minhash: int = 128,
) -> np.ndarray:
    """
    Produce a hybrid similarity matrix that merges:
      • RBF‑based perceptual similarity (phash)
      • Fisher‑information weighting derived from the kernel statistics
      • Entropy of an external probability distribution (MinHash bridge)

    The matrix is given by:
        S_hybrid = S_phash * I * (1 + H)
    where:
        S_phash – normalized Hamming similarity from perceptual hashes,
        I – Fisher information scalar from the kernel,
        H – Shannon entropy of the supplied probability distribution.
    """
    K, _ = compute_rbf_kernel(features, epsilon)
    I = fisher_information_from_kernel(K)
    S_phash = phash_similarity_matrix(K)

    H = entropy_of_distribution(cluster_probs)

    hybrid = S_phash * I * (1.0 + H)
    return hybrid


if __name__ == "__main__":
    # Simple smoke test
    random.seed(42)
    np.random.seed(42)

    # Create dummy feature vectors for 4 nodes
    dummy_features = {
        f"node_{i}": np.random.rand(5).tolist() for i in range(4)
    }

    # Dummy probability distribution (e.g., cluster membership scores)
    dummy_probs = [0.1, 0.3, 0.4, 0.2]

    # Compute hybrid similarity matrix
    H = hybrid_similarity(dummy_features, dummy_probs, epsilon=0.8)

    print("Hybrid similarity matrix:")
    print(H)

    # Demonstrate burst admission score using the same entropy & Fisher weight
    K_mat, _ = compute_rbf_kernel(dummy_features, epsilon=0.8)
    I_val = fisher_information_from_kernel(K_mat)
    ent = entropy_of_distribution(dummy_probs)
    score = burst_admission_score(dummy_probs, ent, I_val)
    print("\nBurst admission score (hybrid):", score)