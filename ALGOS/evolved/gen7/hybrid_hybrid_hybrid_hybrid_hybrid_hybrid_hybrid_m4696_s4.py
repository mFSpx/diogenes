# DARWIN HAMMER — match 4696, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_gini_c_m2063_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_hybrid_m1138_s2.py (gen5)
# born: 2026-05-29T23:57:30Z

"""Hybrid Algorithm Fusion of:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_gini_c_m2063_s0.py (Algorithm A)
- hybrid_hybrid_hybrid_gliner_hybrid_hybrid_hybrid_m1138_s2.py (Algorithm B)

Mathematical Bridge:
Algorithm A projects high‑dimensional text features onto a model space using a bilinear form
and then processes the projected vector with tropical max‑plus algebra (⊕ = max, ⊗ = +)
to obtain a Gini‑based inequality score.  
Algorithm B extracts a set of span scores, converts them into pheromone signals
(−log score) and evaluates their information content with Shannon entropy.
The fusion treats the tropical‑max‑plus output of A as a feature distribution,
normalises it to a probability vector, and feeds it to the entropy calculation of B.
Thus the Gini coefficient, tropical aggregation, and entropy are mathematically
combined into a single hybrid decision metric.

The module provides:
* tropical algebra primitives,
* Gini coefficient computation,
* entropy evaluation,
* a bilinear projection that links the two topologies,
* a pheromone‑aware decision routine that merges all three measures.
"""

import numpy as np
import math
import random
import sys
import pathlib

# ----------------------------------------------------------------------
# Tropical algebra (max‑plus) primitives
# ----------------------------------------------------------------------
def tropical_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical addition (⊕): element‑wise maximum."""
    return np.maximum(x, y)


def tropical_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical multiplication (⊗): element‑wise addition."""
    return x + y


def tropical_sum(vec: np.ndarray) -> float:
    """Tropical sum of a vector (⊕ over all elements) → max element."""
    return float(np.max(vec))


def tropical_product(vec: np.ndarray) -> float:
    """Tropical product of a vector (⊗ over all elements) → sum of elements."""
    return float(np.sum(vec))


# ----------------------------------------------------------------------
# Gini coefficient (inequality) for a numeric vector
# ----------------------------------------------------------------------
def gini_coefficient(x: np.ndarray) -> float:
    """Compute the Gini coefficient of a 1‑D array."""
    if x.ndim != 1:
        raise ValueError("Gini coefficient requires a 1‑D array.")
    if len(x) == 0:
        return 0.0
    sorted_x = np.sort(np.abs(x))  # ensure non‑negative for the formula
    n = len(x)
    cumulative = np.cumsum(sorted_x)
    sum_x = cumulative[-1]
    if sum_x == 0:
        return 0.0
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * sorted_x)) / (n * sum_x) - (n + 1) / n
    return float(gini)


# ----------------------------------------------------------------------
# Shannon entropy for a probability distribution
# ----------------------------------------------------------------------
def shannon_entropy(p: np.ndarray) -> float:
    """Compute Shannon entropy of a probability vector p (must sum to 1)."""
    if p.ndim != 1:
        raise ValueError("Entropy requires a 1‑D probability vector.")
    eps = np.finfo(float).eps
    p = np.clip(p, eps, 1.0)  # avoid log(0)
    return float(-np.sum(p * np.log(p)))


# ----------------------------------------------------------------------
# Bilinear projection (high‑dimensional → low‑dimensional)
# ----------------------------------------------------------------------
def bilinear_projection(features: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Project `features` (shape (batch, d_in)) onto a lower‑dimensional space
    using a bilinear form defined by `weight` (shape (d_in, d_out)).
    Returns a matrix of shape (batch, d_out).
    """
    if features.shape[1] != weight.shape[0]:
        raise ValueError("Incompatible dimensions for bilinear projection.")
    return features @ weight


# ----------------------------------------------------------------------
# Pheromone handling (Algorithm B)
# ----------------------------------------------------------------------
class PheromoneEntry:
    """Simple container for a pheromone signal attached to a span key."""
    def __init__(self, key: str, signal: float, half_life: float = 60.0):
        self.key = key
        self.signal = signal
        self.half_life = half_life  # seconds


class PheromoneStore:
    """In‑memory store for pheromone entries."""
    def __init__(self):
        self._store = {}

    def add(self, entry: PheromoneEntry):
        self._store[entry.key] = entry

    def get(self, key: str) -> PheromoneEntry:
        return self._store.get(key)


def compute_pheromone_signal(score: float) -> float:
    """
    Convert a span score into a pheromone signal.
    Uses the negative natural logarithm as a proxy (higher score → weaker signal).
    """
    if score <= 0.0:
        score = sys.float_info.min
    return -math.log(score)


# ----------------------------------------------------------------------
# Hybrid core routine – fuses A and B
# ----------------------------------------------------------------------
def hybrid_decision(
    raw_features: np.ndarray,
    span_scores: np.ndarray,
    proj_weight: np.ndarray,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
) -> float:
    """
    Compute a unified decision metric from raw features and span scores.

    Steps:
    1. Bilinear projection of raw_features (Algorithm A).
    2. Tropical aggregation of the projected vectors.
    3. Gini coefficient of the tropical‑aggregated vector.
    4. Normalise the tropical vector → probability distribution.
    5. Shannon entropy of that distribution (Algorithm B).
    6. Pheromone‑based adjustment derived from span_scores.
    7. Weighted sum of (Gini, Entropy, Pheromone‑mean) → final score.
    """
    # 1. Projection
    proj = bilinear_projection(raw_features, proj_weight)          # (batch, d_out)

    # 2. Tropical aggregation per sample
    trop_max = np.apply_along_axis(tropical_sum, 1, proj)          # (batch,)
    trop_prod = np.apply_along_axis(tropical_product, 1, proj)    # (batch,)

    # 3. Gini on the tropical‑max vector
    gini_vals = np.array([gini_coefficient(np.array([m, p])) for m, p in zip(trop_max, trop_prod)])

    # 4. Normalise tropical‑max vector to probabilities
    prob_vec = trop_max / (np.sum(trop_max) + np.finfo(float).eps)

    # 5. Entropy of the probability vector
    entropy_val = shannon_entropy(prob_vec)

    # 6. Pheromone signal from span scores
    pheromone_signals = np.vectorize(compute_pheromone_signal)(span_scores)
    pheromone_mean = float(np.mean(pheromone_signals))

    # 7. Weighted combination
    final_score = (
        alpha * float(np.mean(gini_vals)) +
        beta * entropy_val +
        gamma * pheromone_mean
    )
    return final_score


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
def _smoke_test():
    rng = np.random.default_rng(42)

    # Simulated high‑dimensional text features (e.g., 128‑dim, batch 10)
    raw_feat = rng.random((10, 128))

    # Random projection matrix to 16‑dimensional model space
    proj_w = rng.random((128, 16))

    # Simulated span scores (e.g., confidence scores from a span detector)
    span_scores = rng.uniform(0.1, 1.0, size=10)

    # Compute hybrid decision metric
    score = hybrid_decision(raw_feat, span_scores, proj_w)

    print(f"Hybrid decision score: {score:.6f}")

    # Demonstrate pheromone store usage
    store = PheromoneStore()
    for i, s in enumerate(span_scores):
        key = f"span_{i}"
        signal = compute_pheromone_signal(s)
        store.add(PheromoneEntry(key, signal))

    # Retrieve one entry
    entry = store.get("span_3")
    if entry:
        print(f"Pheromone entry for span_3 → signal={entry.signal:.4f}")

if __name__ == "__main__":
    _smoke_test()