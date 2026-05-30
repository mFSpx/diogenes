# DARWIN HAMMER — match 1210, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_path_signatur_hybrid_krampus_brain_m25_s3.py (gen2)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hard_truth_ma_m93_s3.py (gen3)
# born: 2026-05-29T23:34:26Z

"""Hybrid Algorithm: Path‑Signature + RBF Surrogate (Parents A & B)

Parent A – hybrid_path_signature_kan_m30_s1.py
    • Provides a lead‑lag transform of a multivariate discrete path.
    • Computes level‑1 (total increment) and level‑2 (iterated‑integral) signatures.

Parent B – hybrid_rbf_surrogate_hybrid_hard_truth_ma_m93_s3.py
    • Supplies a deterministic master‑vector extractor for free‑form text.
    • Uses a radial‑basis‑function (Gaussian) surrogate to evaluate similarity
      between high‑dimensional feature vectors.

Mathematical Bridge
-------------------
A sequence of texts 𝚃 = (t₁,…,tₙ) is first mapped to a sequence of master vectors
𝑿 = (𝒙₁,…,𝒙ₙ) ⊂ ℝᴰ via the deterministic extractor (Parent B).  
The lead‑lag transform lifts 𝑿 to an augmented path 𝑿̃ ∈ ℝ²ᴰ (Parent A).  
From 𝑿̃ we compute the level‑1 signature **s₁** ∈ ℝ²ᴰ and the flattened level‑2
signature **s₂** ∈ ℝ^{(2D)²}.  Concatenating **s₁** and **s₂** yields a single
signature vector **σ**.

The RBF surrogate (Parent B) treats **σ** as a point in a high‑dimensional space.
Given two text sequences we obtain signatures σᵃ, σᵇ and evaluate the Gaussian
kernel  

    K(σᵃ,σᵇ) = exp( -ε² ‖σᵃ‑σᵇ‖² )

which quantifies their stylometric‑path similarity.  This similarity can modulate
graph‑broadcast probabilities or any downstream decision rule.

The code below implements this fused pipeline with three core functions:
    1. `master_vector` – deterministic text → vector.
    2. `path_signature` – lead‑lag + level‑1 & level‑2 signatures.
    3. `rbf_similarity` – Gaussian RBF on signature vectors.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Mapping, Hashable, Set

import numpy as np

# ----------------------------------------------------------------------
# Deterministic master‑vector extractor (Parent B component)
# ----------------------------------------------------------------------
def master_vector(text: str, dim: int = 16) -> np.ndarray:
    """
    Produce a deterministic pseudo‑random vector of length `dim` from `text`.
    The same text always yields the same vector, independent of global RNG state.
    """
    rnd = random.Random(hash(text))
    return np.array([rnd.random() for _ in range(dim)], dtype=float)


# ----------------------------------------------------------------------
# Lead‑lag transform and signature computation (Parent A component)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Given a path X of shape (n, d), return the lead‑lag augmented path
    of shape (2n‑2, 2d).  For each consecutive pair (x_i, x_{i+1}) we emit:
        [x_i, x_i]   (lead)
        [x_i, x_{i+1}] (lag)
    This construction preserves causality and is standard for signatures.
    """
    n, d = path.shape
    if n < 2:
        raise ValueError("Path must contain at least two points for lead‑lag.")
    augmented = []
    for i in range(n - 1):
        lead = np.concatenate([path[i], path[i]])          # (2d,)
        lag = np.concatenate([path[i], path[i + 1]])       # (2d,)
        augmented.append(lead)
        augmented.append(lag)
    return np.vstack(augmented)  # shape (2n-2, 2d)


def compute_signature(aug_path: np.ndarray) -> np.ndarray:
    """
    Compute level‑1 and level‑2 signatures for an augmented path.
    Level‑1: total increment (last - first).
    Level‑2: sum_{i<j} Δ_i ⊗ Δ_j, flattened to a 1‑D array.
    Returns a concatenated vector [s1, s2].
    """
    increments = np.diff(aug_path, axis=0)               # shape (m-1, 2d)
    s1 = increments.sum(axis=0)                         # (2d,)

    # Level‑2 via double loop (acceptable for modest dimensions)
    m = increments.shape[0]
    d2 = increments.shape[1]
    s2 = np.zeros(d2 * d2, dtype=float)
    idx = 0
    for i in range(m):
        for j in range(i + 1, m):
            outer = np.outer(increments[i], increments[j])  # (2d,2d)
            s2[idx: idx + d2 * d2] += outer.ravel()
            idx += d2 * d2
    return np.concatenate([s1, s2])


def path_signature(texts: List[str], dim: int = 16) -> np.ndarray:
    """
    Full pipeline: texts → master vectors → lead‑lag → signature.
    """
    vectors = np.vstack([master_vector(t, dim) for t in texts])  # (n, dim)
    aug = lead_lag_transform(vectors)                           # (2n-2, 2dim)
    return compute_signature(aug)                               # (signature length,)


# ----------------------------------------------------------------------
# Radial‑basis‑function surrogate (Parent B component)
# ----------------------------------------------------------------------
def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 1‑D arrays."""
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same shape.")
    return float(np.sqrt(((a - b) ** 2).sum()))


def gaussian_kernel(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF with bandwidth ε."""
    return math.exp(-((epsilon * r) ** 2))


def rbf_similarity(sig_a: np.ndarray, sig_b: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Compute the RBF surrogate similarity between two signature vectors.
    """
    dist = euclidean(sig_a, sig_b)
    return gaussian_kernel(dist, epsilon)


# ----------------------------------------------------------------------
# Example hybrid operation combining both topologies
# ----------------------------------------------------------------------
def hybrid_similarity(texts_a: List[str], texts_b: List[str],
                      dim: int = 16, epsilon: float = 1.0) -> float:
    """
    Given two sequences of texts, compute their fused similarity:
        1. Convert each sequence to a signature vector.
        2. Evaluate the Gaussian RBF on the two signatures.
    """
    sig_a = path_signature(texts_a, dim)
    sig_b = path_signature(texts_b, dim)
    return rbf_similarity(sig_a, sig_b, epsilon)


# ----------------------------------------------------------------------
# Simple broadcast probability derived from similarity (illustrative)
# ----------------------------------------------------------------------
def broadcast_probability(similarity: float, base_rate: float = 0.1) -> float:
    """
    Modulate a base broadcast probability by the similarity score.
    The function maps similarity ∈ (0,1] to a probability ∈ (base_rate, 1].
    """
    if not (0.0 < similarity <= 1.0):
        raise ValueError("Similarity must be in (0, 1].")
    return min(1.0, base_rate + (1.0 - base_rate) * similarity)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Two tiny corpora
    corpus_a = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence transforms scientific research."
    ]
    corpus_b = [
        "A swift auburn animal leaps above a sleepy canine.",
        "Machine learning accelerates discovery in physics."
    ]

    sim = hybrid_similarity(corpus_a, corpus_b, dim=12, epsilon=0.5)
    prob = broadcast_probability(sim, base_rate=0.05)

    print(f"Hybrid RBF similarity: {sim:.6f}")
    print(f"Derived broadcast probability: {prob:.4f}")