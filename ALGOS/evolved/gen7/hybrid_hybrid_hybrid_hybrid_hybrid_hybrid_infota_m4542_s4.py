# DARWIN HAMMER — match 4542, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_fisher_m2240_s0.py (gen6)
# parent_b: hybrid_hybrid_infotaxis_min_hybrid_hybrid_distri_m20_s1.py (gen3)
# born: 2026-05-29T23:56:31Z

"""
Hybrid RBF‑Fisher‑MinHash Algorithm
Parents:
- hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_fisher_m2240_s0.py
- hybrid_hybrid_infotaxis_min_hybrid_hybrid_distri_m20_s1.py

Mathematical bridge:
The RBF kernel provides a dense similarity matrix K whose entries are interpreted as
pairwise distances d_ij = 1‑K[i,j].  These distances are normalised to a probability‑like
vector and fed to a MinHash pipeline that yields a signature σ.  The Shannon entropy
H(σ) of the signature quantifies the uncertainty of the underlying distance
distribution.  H(σ) is then used as the *width* w of a Gaussian‑type Fisher‑information
kernel I(d; μ, w) that weights the original RBF similarities.  The resulting hybrid
matrix H = K ⊙ I (element‑wise product) captures both geometric similarity (RBF) and
statistical uncertainty (MinHash‑derived Fisher information).  The same entropy H(σ)
also modulates a burst‑admission score for a physics‑based “strike” model, linking the
information‑theoretic component of Algorithm B with the perceptual‑geometric component
of Algorithm A.
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


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def rbf_kernel_matrix(
    features: Dict[Node, FeatureVec], epsilon: float = 1.0
) -> Tuple[np.ndarray, List[Node]]:
    """
    Compute dense RBF kernel matrix K where
        K[i, j] = exp(-ε² * ||f_i - f_j||²)

    Returns:
        K   – (n, n) ndarray of float64
        order – list of nodes matching rows/cols of K
    """
    nodes = list(features.keys())
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        K[i, i] = 1.0
        for j in range(i + 1, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            val = gaussian(dist, epsilon)
            K[i, j] = K[j, i] = val
    return K, nodes


def _hash_family(k: int, seed: int = 0) -> List[Tuple[int, int]]:
    """Create k simple universal hash functions (a, b) modulo a large prime."""
    random.seed(seed)
    prime = 2 ** 61 - 1
    return [(random.randrange(1, prime), random.randrange(0, prime)) for _ in range(k)]


def minhash_signature(values: List[float], k: int = 64, seed: int = 0) -> List[int]:
    """
    Compute a MinHash signature for a list of real numbers.
    The list is first converted to strings, each hash function selects the
    minimum hashed value over all elements.
    """
    if not values:
        return [0] * k
    hash_funcs = _hash_family(k, seed)
    sig = [sys.maxsize] * k
    for v in values:
        token = str(v).encode("utf-8")
        h = int.from_bytes(token, "little", signed=False)
        for idx, (a, b) in enumerate(hash_funcs):
            candidate = (a * h + b) % (2 ** 61 - 1)
            if candidate < sig[idx]:
                sig[idx] = candidate
    return sig


def signature_entropy(sig: List[int]) -> float:
    """
    Shannon entropy of a MinHash signature interpreted as a discrete distribution.
    """
    if not sig:
        return 0.0
    # Count occurrences of each hash value (they are mostly unique)
    counts = {}
    for v in sig:
        counts[v] = counts.get(v, 0) + 1
    total = len(sig)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent


def fisher_weight_matrix(
    distances: np.ndarray, centre: float, width: float
) -> np.ndarray:
    """
    Gaussian‑type Fisher‑information weighting:
        I(d; μ, w) = exp(- (d-μ)² / (2 w²))

    distances – symmetric matrix of non‑negative scalars
    centre    – mean distance (μ)
    width     – standard deviation (w) – here derived from entropy
    """
    if width <= 0:
        width = 1e-6
    diff = distances - centre
    I = np.exp(- (diff ** 2) / (2 * width ** 2))
    return I


def hybrid_similarity_matrix(
    features: Dict[Node, FeatureVec],
    epsilon: float = 1.0,
    k_minhash: int = 64,
) -> Tuple[np.ndarray, List[Node]]:
    """
    Full hybrid pipeline:
    1. Build RBF kernel K.
    2. Convert K to a distance matrix D = 1 - K.
    3. Flatten D (excluding diagonal) to a probability‑like vector and compute a MinHash signature.
    4. Obtain entropy H from the signature.
    5. Use H as the width parameter for a Fisher‑information matrix I.
    6. Return the element‑wise product H = K ⊙ I.
    """
    K, order = rbf_kernel_matrix(features, epsilon)
    D = 1.0 - K
    # Extract upper‑triangular (excluding diagonal) as a flat list
    triu_vals = D[np.triu_indices_from(D, k=1)].tolist()
    sig = minhash_signature(triu_vals, k=k_minhash)
    H = signature_entropy(sig)

    # Use entropy as a proxy for variance; map to a positive scale
    width = max(H, 1e-6)

    centre = float(np.mean(D))
    I = fisher_weight_matrix(D, centre, width)

    hybrid = K * I  # element‑wise product
    return hybrid, order


def burst_admission_score(
    entropy_val: float,
    velocity: float,
    distance: float,
    m_head: float = 1.0,
    drag_cd: float = 0.3,
    fluid_density: float = 1000.0,
    area: float = 0.001,
) -> float:
    """
    Physics‑driven admission score modulated by entropy.
    The base kinetic term is (1/2) m v².
    Drag deceleration is approximated by (½ ρ C_d A v²) / m.
    Entropy acts as an uncertainty penalty: higher entropy → lower score.
    """
    kinetic = 0.5 * m_head * velocity ** 2
    drag_force = 0.5 * fluid_density * drag_cd * area * velocity ** 2
    drag_term = drag_force / m_head
    base_score = kinetic - drag_term * distance
    penalty = math.exp(-entropy_val)  # 0 < penalty ≤ 1
    return max(base_score * penalty, 0.0)


def strike_on_cluster(
    cluster: List[float],
    initial_velocity: float = 5.0,
    steps: int = 10,
) -> List[float]:
    """
    Simulate a simple strike over a cluster of probability values.
    At each step the entropy of the current cluster (via MinHash) determines the
    admissible velocity reduction; the burst admission score is recorded.
    Returns the list of scores per step.
    """
    scores = []
    current_cluster = cluster[:]
    velocity = initial_velocity
    for step in range(steps):
        sig = minhash_signature(current_cluster, k=64, seed=step)
        ent = signature_entropy(sig)
        # distance is approximated by the mean absolute deviation of the cluster
        distance = float(np.mean([abs(x - np.mean(current_cluster)) for x in current_cluster]))
        score = burst_admission_score(ent, velocity, distance)
        scores.append(score)

        # simple physics update: reduce velocity proportionally to entropy
        velocity *= max(0.5, 1.0 - ent * 0.1)
        # mutate cluster slightly to emulate dispersion
        current_cluster = [x * random.uniform(0.95, 1.05) for x in current_cluster]
    return scores


if __name__ == "__main__":
    # Smoke test for the hybrid pipeline
    # Create a tiny synthetic feature set
    features = {
        "A": [0.1, 0.2, 0.3],
        "B": [0.2, 0.1, 0.4],
        "C": [0.9, 0.8, 0.7],
        "D": [0.5, 0.5, 0.5],
    }

    hybrid_mat, order = hybrid_similarity_matrix(features, epsilon=1.5, k_minhash=32)
    print("Hybrid similarity matrix:")
    print(hybrid_mat)

    # Create a dummy probability cluster and run the strike simulation
    dummy_cluster = [random.random() for _ in range(20)]
    scores = strike_on_cluster(dummy_cluster, initial_velocity=8.0, steps=5)
    print("\nBurst admission scores per step:")
    for i, s in enumerate(scores, 1):
        print(f"Step {i}: {s:.4f}")