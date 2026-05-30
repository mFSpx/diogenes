# DARWIN HAMMER — match 2884, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_semant_hybrid_hybrid_hybrid_m2659_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m947_s0.py (gen5)
# born: 2026-05-29T23:46:25Z

"""Hybrid Algorithm: Fusion of Semantic‑Neighbor RBF Filtering (Parent A) and Fisher‑Bandit Optimization (Parent B)

Mathematical Bridge
-------------------
Parent A provides a radial‑basis‑function (RBF) similarity matrix **S** between
temporal motifs, together with a spatial‑diversity filter.  
Parent B supplies a Fisher information scalar **I** (and its associated
information **F**) for each motif together with a bandit‑style propensity
**p** and confidence bound **c**.

The hybrid algorithm multiplies the RBF similarity by the Fisher intensity
**I** to obtain a *Fisher‑weighted similarity* **W = I·S**.  The bandit
propensity is then used as a confidence scalar that modulates the selection
score of each motif:

    score_i = (Σ_j W_ij) * p_i / (1 + c_i)

Thus the semantic‑neighbor structure is guided by statistical information
(Fisher) and exploration‑exploitation dynamics (bandit), yielding a unified
motif‑ranking and filtering pipeline.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int
    # feature vector for spatial/semantic embedding (e.g., morphology, location)
    features: Tuple[float, ...]


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    confidence_bound: float
    algorithm: str = "HybridFisherBandit"


# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------


def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_rbf_similarity(motifs: List[TemporalMotif], epsilon: float = 1.0) -> np.ndarray:
    """Return the symmetric RBF similarity matrix S_ij for a list of motifs."""
    n = len(motifs)
    S = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            dist = euclidean(list(motifs[i].features), list(motifs[j].features))
            sim = gaussian_rbf(dist, epsilon)
            S[i, j] = S[j, i] = sim
    return S


def compute_fisher_information(
    theta: np.ndarray, mu: float, sigma: float, v: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Fisher intensity I and Fisher information F for each scalar theta.

    I = v * exp(-((theta - mu)/sigma)^2)
    F = v * ((2*(theta - mu)/sigma^2)^2) / I
    """
    I = np.exp(-((theta - mu) / sigma) ** 2)
    # Guard against division by zero
    I = np.clip(I, 1e-12, None)
    F = (2 * (theta - mu) / sigma ** 2) ** 2 / I
    return v * I, v * F


def bandit_propensity(scores: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Convert raw scores into propensities using a softmax with temperature alpha.
    """
    max_score = np.max(scores)
    exp_vals = np.exp((scores - max_score) / alpha)
    probs = exp_vals / np.sum(exp_vals)
    return probs


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def fisher_weighted_similarity(
    motifs: List[TemporalMotif],
    mu: float,
    sigma: float,
    v: float = 1.0,
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Compute the Fisher‑weighted RBF similarity matrix W = I * S,
    where I is the Fisher intensity per motif.
    """
    # 1. Base RBF similarity
    S = compute_rbf_similarity(motifs, epsilon)

    # 2. Fisher intensity per motif (using support as theta)
    supports = np.array([m.support for m in motifs], dtype=float)
    I, _ = compute_fisher_information(supports, mu, sigma, v)

    # 3. Weight each row/column of S by the corresponding I
    W = S * I[:, None]  # broadcasting I over rows
    return W


def hybrid_motif_scores(
    motifs: List[TemporalMotif],
    mu: float,
    sigma: float,
    v: float = 1.0,
    epsilon: float = 1.0,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Compute a hybrid selection score for each motif:
        score_i = (Σ_j W_ij) * p_i / (1 + c_i)

    where:
        W_ij = I_i * S_ij   (Fisher‑weighted similarity)
        p_i  = propensity derived from raw scores (softmax)
        c_i  = confidence bound proportional to Fisher information F_i
    """
    # Fisher‑weighted similarity matrix
    W = fisher_weighted_similarity(motifs, mu, sigma, v, epsilon)

    # Raw aggregation (sum over neighbors)
    raw_scores = W.sum(axis=1)

    # Propensity from raw scores (softmax)
    propensities = bandit_propensity(raw_scores, alpha)

    # Confidence bound from Fisher information (higher F → larger bound)
    supports = np.array([m.support for m in motifs], dtype=float)
    _, F = compute_fisher_information(supports, mu, sigma, v)
    confidence_bounds = np.sqrt(F)  # simple monotone transform

    # Final hybrid score
    scores = raw_scores * propensities / (1.0 + confidence_bounds)
    return scores


def select_top_k_motifs(
    motifs: List[TemporalMotif],
    k: int,
    mu: float,
    sigma: float,
    v: float = 1.0,
    epsilon: float = 1.0,
    alpha: float = 0.5,
) -> List[TemporalMotif]:
    """
    Rank motifs by hybrid scores and return the top‑k diverse motifs.
    Diversity is enforced by a greedy farthest‑point heuristic on the
    Fisher‑weighted similarity matrix.
    """
    if k <= 0:
        return []

    scores = hybrid_motif_scores(motifs, mu, sigma, v, epsilon, alpha)
    sorted_indices = np.argsort(-scores)  # descending

    # Greedy diversity selection
    selected: List[int] = []
    remaining = set(sorted_indices.tolist())

    # Start with the highest‑scoring motif
    first = sorted_indices[0]
    selected.append(first)
    remaining.remove(first)

    # Pre‑compute the weighted similarity matrix for distance checks
    W = fisher_weighted_similarity(motifs, mu, sigma, v, epsilon)

    while len(selected) < min(k, len(motifs)):
        # For each candidate, compute its minimum similarity to already selected
        min_similarities = {}
        for idx in remaining:
            sims = [W[idx, s_idx] for s_idx in selected]
            min_similarities[idx] = min(sims) if sims else 0.0

        # Choose the candidate with the smallest similarity (most diverse)
        next_idx = min(min_similarities, key=min_similarities.get)
        selected.append(next_idx)
        remaining.remove(next_idx)

    return [motifs[i] for i in selected]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a synthetic motif pool
    random.seed(42)
    np.random.seed(42)

    def random_pattern(length: int = 4) -> Tuple[str, ...]:
        alphabet = ["A", "B", "C", "D", "E"]
        return tuple(random.choice(alphabet) for _ in range(length))

    pool: List[TemporalMotif] = []
    for i in range(20):
        pattern = random_pattern()
        support = random.randint(5, 100)
        # Feature vector: [support, random spatial coords]
        features = (float(support), random.random() * 10.0, random.random() * 10.0)
        pool.append(TemporalMotif(pattern=pattern, support=support, features=features))

    # Hyper‑parameters for the hybrid model
    MU = np.mean([m.support for m in pool])
    SIGMA = np.std([m.support for m in pool]) + 1e-6  # avoid zero
    V = 1.0
    EPS = 0.8
    ALPHA = 0.3
    K = 5

    top_motifs = select_top_k_motifs(
        motifs=pool,
        k=K,
        mu=MU,
        sigma=SIGMA,
        v=V,
        epsilon=EPS,
        alpha=ALPHA,
    )

    print(f"Selected top {K} hybrid motifs:")
    for idx, m in enumerate(top_motifs, 1):
        print(f"{idx}: pattern={m.pattern}, support={m.support}, features={m.features}")

    sys.exit(0)