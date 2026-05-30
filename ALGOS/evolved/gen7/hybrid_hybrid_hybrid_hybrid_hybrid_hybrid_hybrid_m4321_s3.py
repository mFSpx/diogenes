# DARWIN HAMMER — match 4321, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2138_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m2622_s0.py (gen5)
# born: 2026-05-29T23:54:52Z

"""
Hybrid Darwin Hammer – Distributed Sparse Fusion

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2138_s0.py (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m2622_s0.py (Algorithm B)

Mathematical Bridge
-------------------
Algorithm A provides a *log‑count ratio* based scalar
    h(action) = log_count_ratio * count
which is used for pheromone infotaxis and decision‑hygiene entropy.

Algorithm B expands a dense input vector into a *hash‑based sparse* representation
    S = expand(values, m, salt)

The fusion treats the sparse vector as a *state‑space* whose occupied slots are
weighted by the A‑side scalar h(action).  A *confidence scalar* derived from an
exponential decay (B‑side) rescales both the expansion and the weighting,
producing a unified operation:

    confidence = exp(‑γ·t)

    V_i = confidence · S_i · h(i)                (i indexes original actions)

The resulting vector V is then used in the same entropy calculation that
Algorithm A defines, but now the probability distribution lives in the
sparse, confidence‑scaled space.

The module implements three core hybrid functions:
1. hybrid_store_factor – A‑side scalar.
2. sparse_hybrid_vector – B‑side expansion combined with A‑side weighting and
   exponential confidence scaling.
3. decision_hygiene_shannon_entropy – entropy of the resulting distribution.
"""

import sys
import pathlib
import random
import math
import hashlib
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core A‑side utilities
# ----------------------------------------------------------------------
def hybrid_store_factor(count: float, log_count_ratio: float) -> float:
    """
    Compute the hybrid store factor used in pheromone infotaxis and entropy.

    Parameters
    ----------
    count : float
        Raw occurrence count for an action.
    log_count_ratio : float
        Log‑ratio of current vs. reference count.

    Returns
    -------
    float
        The product log_count_ratio * count.
    """
    return log_count_ratio * count


# ----------------------------------------------------------------------
# Core B‑side utilities (hash‑based sparse expansion)
# ----------------------------------------------------------------------
def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """
    Hash‑based sparse expansion of `values` into a vector of length `m`.

    Each value contributes to three pseudo‑random positions with a random sign.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if (h[8] & 1) else -1.0
            out[j] += sign * v
    return out


def top_k_mask(values: List[float], k: int) -> List[int]:
    """
    Binary mask with 1 at the indices of the top‑k values.
    """
    k = max(0, min(k, len(values)))
    winners = {
        i
        for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]
    }
    return [1 if i in winners else 0 for i in range(len(values))]


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def confidence_scalar(decay_rate: float, step: int) -> float:
    """
    Exponential decay confidence scalar.

    confidence = exp(-decay_rate * step)
    """
    if decay_rate < 0:
        raise ValueError("decay_rate must be non‑negative")
    return math.exp(-decay_rate * step)


def sparse_hybrid_vector(
    values: List[float],
    counts: List[float],
    log_ratios: List[float],
    m: int,
    salt: str,
    decay_rate: float,
    step: int,
) -> np.ndarray:
    """
    Produce the hybrid sparse vector V_i = confidence * S_i * h(i).

    Parameters
    ----------
    values : List[float]
        Original dense feature values (e.g., pheromone levels per action).
    counts : List[float]
        Occurrence counts per action (same length as values).
    log_ratios : List[float]
        Log‑count ratios per action (same length as values).
    m : int
        Length of the sparse target space.
    salt : str
        Salt for deterministic hashing.
    decay_rate : float
        Exponential decay rate for confidence scaling.
    step : int
        Current discrete time step.

    Returns
    -------
    np.ndarray
        The confidence‑scaled, weighted sparse vector of shape (m,).
    """
    if not (len(values) == len(counts) == len(log_ratios)):
        raise ValueError("values, counts, and log_ratios must have the same length")

    # 1. Sparse expansion of the raw values.
    sparse = np.array(expand(values, m, salt), dtype=np.float64)

    # 2. Compute per‑action hybrid store factors.
    h_factors = np.array(
        [hybrid_store_factor(c, lr) for c, lr in zip(counts, log_ratios)],
        dtype=np.float64,
    )

    # 3. Map each original action to the three positions it touched during
    #    expansion and accumulate the weighted contribution.
    weighted_sparse = np.zeros(m, dtype=np.float64)
    for i, v in enumerate(values):
        factor = h_factors[i]
        for r in range(3):
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if (h[8] & 1) else -1.0
            weighted_sparse[j] += sign * v * factor

    # 4. Apply confidence scaling.
    confidence = confidence_scalar(decay_rate, step)
    return confidence * weighted_sparse


def decision_hygiene_shannon_entropy(
    vector: np.ndarray, epsilon: float = 1e-12
) -> float:
    """
    Compute Shannon entropy of a (possibly sparse) non‑negative vector.

    The vector is first normalized to a probability distribution.  A small
    epsilon prevents log(0).

    Parameters
    ----------
    vector : np.ndarray
        Non‑negative values.
    epsilon : float
        Numerical floor for probabilities.

    Returns
    -------
    float
        Shannon entropy in nats.
    """
    if np.any(vector < 0):
        raise ValueError("Entropy input must be non‑negative")
    total = vector.sum()
    if total == 0:
        return 0.0
    probs = vector / total
    probs = np.clip(probs, epsilon, 1.0)
    return -np.sum(probs * np.log(probs))


# ----------------------------------------------------------------------
# Example ModelTier (lightweight placeholder from Parent A)
# ----------------------------------------------------------------------
class ModelTier:
    """Lightweight descriptor of a model with RAM footprint."""

    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

    def priority(self, morphology_index: float) -> float:
        """
        Compute recovery priority from a morphology‑derived index in [0,1].
        """
        return max(0.0, min(1.0, morphology_index))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data for 5 actions
    values = [0.8, 0.1, 0.5, 0.3, 0.9]          # e.g., pheromone levels
    counts = [10, 4, 7, 2, 15]                 # occurrence counts
    log_ratios = [math.log(v / (c + 1e-6)) for v, c in zip(values, counts)]

    m = 32                                     # size of sparse space
    salt = "hybrid_test"
    decay_rate = 0.05
    step = 12

    # Build hybrid vector
    hybrid_vec = sparse_hybrid_vector(
        values,
        counts,
        log_ratios,
        m,
        salt,
        decay_rate,
        step,
    )

    # Compute entropy
    entropy = decision_hygiene_shannon_entropy(hybrid_vec)

    print("Hybrid sparse vector (first 10 entries):", hybrid_vec[:10])
    print("Shannon entropy of hybrid distribution:", entropy)

    # Simple top‑k mask on the original values for comparison
    mask = top_k_mask(values, k=3)
    print("Top‑3 mask on original values:", mask)