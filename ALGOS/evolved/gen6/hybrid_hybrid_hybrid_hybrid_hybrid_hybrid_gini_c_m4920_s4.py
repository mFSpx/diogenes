# DARWIN HAMMER — match 4920, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m413_s1.py (gen4)
# parent_b: hybrid_hybrid_gini_coeffici_hybrid_tropical_maxp_m1173_s4.py (gen5)
# born: 2026-05-29T23:58:52Z

"""Hybrid Bayesian‑Gini‑Tropical RBF Algorithm
==========================================

This module fuses the two parent algorithms:

* **Parent A** – Bayesian evidence update with semantic‑neighbor likelihoods.
* **Parent B** – Gini‑coefficient‑augmented decision scores propagated through a
  tropical (max‑plus) matrix derived from an RBF similarity kernel.

**Mathematical bridge**

1. The Bayesian posterior `P(H|E)` produced by *Parent A* is a probability.
   Converting it to log‑space yields `ℓ = log P(H|E)`.  
2. *Parent B* works entirely with log‑probabilities: tropical multiplication is
   ordinary addition, tropical addition is `max`, and the tropical weight
   matrix is `W_ij = -log S_ij` where `S_ij` is an RBF similarity.
3. By feeding the log‑posterior vector `ℓ` into the tropical matrix
   multiplication `ℓ' = W ⊗ ℓ` (i.e. `ℓ'_i = max_j (ℓ_j - W_ij)`), we propagate the
   Bayesian belief through the semantic‑neighbour graph.
4. Finally the Gini coefficient computed from the class‑distribution at a node
   (Parent B) is added to the propagated log‑belief, giving a hybrid split score
   that simultaneously accounts for class‑imbalance, probabilistic evidence,
   and semantic proximity.

The three public functions below demonstrate this fusion:
`bayesian_update_vector`, `tropical_propagate`, and `hybrid_split_score`. """

import math
import random
import sys
import pathlib
from typing import List, Sequence, Tuple, Iterable, Hashable

import numpy as np

# ---------------------------------------------------------------------------
# Core utilities – shared by both parents
# ---------------------------------------------------------------------------

Point = Tuple[float, float]


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability P(E) = L·π + FP·(1‑π)."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Bayesian posterior P(H|E) = π·L / P(E)."""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal


# ---------------------------------------------------------------------------
# Parent A – Bayesian update (vectorised)
# ---------------------------------------------------------------------------


def bayesian_update_vector(
    priors: np.ndarray,
    likelihoods: np.ndarray,
    false_positives: np.ndarray,
) -> np.ndarray:
    """
    Vectorised Bayesian update.

    Parameters
    ----------
    priors : np.ndarray
        Prior probabilities (π) for each hypothesis.
    likelihoods : np.ndarray
        Likelihoods P(E|H) for each hypothesis.
    false_positives : np.ndarray
        False‑positive rates P(E|¬H) for each hypothesis.

    Returns
    -------
    np.ndarray
        Posterior probabilities P(H|E) for each hypothesis.
    """
    if not (priors.shape == likelihoods.shape == false_positives.shape):
        raise ValueError("All input arrays must have the same shape")
    marginals = bayes_marginal(priors, likelihoods, false_positives)
    post = bayes_update(priors, likelihoods, marginals)
    # Numerical safety – clip to [0,1]
    return np.clip(post, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Parent B – Gini coefficient and tropical (max‑plus) propagation
# ---------------------------------------------------------------------------


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient of a non‑negative value list."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def _hash_vector(seed: int, token: str, dim: int = 8) -> np.ndarray:
    """
    Deterministic pseudo‑embedding derived from a blake2b hash.
    The vector is normalised to unit length.
    """
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    raw = hashlib.blake2b(data, digest_size=dim * 8).digest()
    ints = np.frombuffer(raw, dtype=np.uint64)
    vec = ints.astype(np.float64)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def rbf_similarity_matrix(tokens: Sequence[str], seed: int = 0, gamma: float = 0.5) -> np.ndarray:
    """
    Build an RBF similarity matrix S where
        S_ij = exp(-γ * ||v_i - v_j||²)
    with v_i a deterministic pseudo‑embedding of token i.
    """
    dim = 8
    embeddings = np.stack([_hash_vector(seed, t, dim) for t in tokens])
    # Pairwise squared Euclidean distances
    diff = embeddings[:, None, :] - embeddings[None, :, :]
    sq_dist = np.einsum("ijk,ijk->ij", diff, diff)
    S = np.exp(-gamma * sq_dist)
    # Numerical stability – enforce symmetry and unit diagonal
    np.fill_diagonal(S, 1.0)
    return S


def tropical_weight_matrix(similarity: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convert a similarity matrix S to tropical weights W:
        W_ij = -log(S_ij + ε)
    The addition of ε prevents log(0).
    """
    return -np.log(np.clip(similarity, eps, None))


def tropical_propagate(weights: np.ndarray, log_beliefs: np.ndarray) -> np.ndarray:
    """
    Tropical (max‑plus) matrix‑vector multiplication:
        (W ⊗ ℓ)_i = max_j (ℓ_j - W_ij)

    Parameters
    ----------
    weights : np.ndarray, shape (n, n)
        Tropical weight matrix (non‑negative).
    log_beliefs : np.ndarray, shape (n,)
        Log‑probabilities (ℓ).

    Returns
    -------
    np.ndarray, shape (n,)
        Updated log‑beliefs after propagation.
    """
    if weights.shape[0] != weights.shape[1]:
        raise ValueError("Weight matrix must be square")
    if weights.shape[0] != log_beliefs.shape[0]:
        raise ValueError("Dimension mismatch between weights and beliefs")
    # Broadcasting: subtract weight column‑wise, then take max across rows
    return np.max(log_beliefs[None, :] - weights, axis=1)


# ---------------------------------------------------------------------------
# Hybrid operation
# ---------------------------------------------------------------------------


def hybrid_split_score(
    class_counts: Sequence[int],
    priors: np.ndarray,
    likelihoods: np.ndarray,
    false_positives: np.ndarray,
    tokens: Sequence[str],
    seed: int = 0,
) -> float:
    """
    Compute a hybrid decision score for a node.

    Steps
    -----
    1. Bayesian posterior for each hypothesis (vectorised).
    2. Convert posterior to log‑space.
    3. Build an RBF similarity matrix from the supplied ``tokens`` and turn it
       into a tropical weight matrix.
    4. Propagate the log‑posteriors through the tropical matrix.
    5. Compute the Gini coefficient of the class distribution and add it to
       the *maximum* propagated log‑belief (the tropical “sum”).
    6. Return the resulting scalar score.

    The resulting score blends class‑imbalance (Gini), probabilistic evidence
    (Bayes) and semantic proximity (tropical RBF propagation).
    """
    # 1. Bayesian update
    post = bayesian_update_vector(priors, likelihoods, false_positives)

    # 2. Log‑space
    log_post = np.log(np.clip(post, 1e-12, 1.0))

    # 3. Similarity → tropical weights
    sim = rbf_similarity_matrix(tokens, seed=seed, gamma=0.5)
    W = tropical_weight_matrix(sim)

    # 4. Tropical propagation
    propagated = tropical_propagate(W, log_post)

    # 5. Gini term
    gini = gini_coefficient(class_counts)

    # Combine: max propagated log‑belief plus Gini
    score = np.max(propagated) + gini
    return float(score)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Dummy data for a tiny node
    class_counts = [30, 10, 5]                     # three classes
    priors = np.array([0.6, 0.3, 0.1])             # prior belief per hypothesis
    likelihoods = np.array([0.8, 0.4, 0.2])        # P(E|H)
    false_positives = np.array([0.1, 0.2, 0.3])    # P(E|¬H)
    tokens = ["alpha", "beta", "gamma"]           # semantic tokens attached to the node

    score = hybrid_split_score(
        class_counts,
        priors,
        likelihoods,
        false_positives,
        tokens,
        seed=42,
    )
    print(f"Hybrid split score: {score:.6f}")