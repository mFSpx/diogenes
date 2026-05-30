# DARWIN HAMMER — match 5545, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_semant_hybrid_hybrid_hybrid_m2659_s4.py (gen6)
# parent_b: hybrid_xgboost_objective_hybrid_hybrid_hybrid_m201_s0.py (gen4)
# born: 2026-05-30T00:02:46Z

"""Hybrid Algorithm Fusion of:
- hybrid_hybrid_semantic_neig_hybrid_temporal_moti_m47_s0 (semantic neighbor search, temporal motifs, morphology indices)
- hybrid_xgboost_objective_hybrid_hybrid_hybrid_m201_s0 (XGBoost leaf weight optimization with endpoint health regularization)

Mathematical Bridge:
Each TemporalMotif is represented both as a semantic embedding (used for neighbor
search) and as a point in Euclidean space for which an RBF similarity matrix can
be built.  The RBF affinities are turned into an inequality measure via the
Gini coefficient, and together with a morphology‑driven recovery priority they
produce a *hybrid health score* `h_i`.  This score is then injected as the
`endpoint_health` term in the XGBoost‑style gradient/hessian calculations and in
the leaf‑weight formula.  Consequently the tree‑learning process is regularized
by the physical‑morphology information of the motifs while still exploiting the
statistical power of gradient boosting.

The module provides three core operations:
1. `hybrid_health_score` – computes the unified health/priority score.
2. `binary_logistic_grad_hess` – binary logistic gradients/Hessians scaled by the health scores.
3. `optimal_leaf_weight` – XGBoost leaf weight formula that incorporates the health scores.

A minimal “one‑tree” trainer (`train_one_stump`) demonstrates the end‑to‑end
fusion without requiring the full XGBoost library."""


import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Sequence
import numpy as np


# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int
    embedding: Tuple[float, ...]          # semantic vector
    morphology: Morphology                # physical attributes


# ----------------------------------------------------------------------
# Parent‑A style helpers
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """A simple proxy for sphericity: ratio of volume to surface area."""
    volume = length * width * height
    surface = 2 * (length * width + width * height + height * length)
    return volume / surface if surface != 0 else 0.0


def recovery_priority(morph: Morphology) -> float:
    """Higher priority for compact (high sphericity) and low mass objects."""
    sph = sphericity_index(morph.length, morph.width, morph.height)
    # Normalise mass to avoid overflow; assume mass > 0
    mass_factor = 1.0 / (1.0 + math.log1p(morph.mass))
    return sph * mass_factor


def rbf_similarity(x: np.ndarray, y: np.ndarray, gamma: float = 0.5) -> float:
    """Radial‑Basis‑Function similarity."""
    diff = x - y
    return math.exp(-gamma * np.dot(diff, diff))


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a 1‑D array."""
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals, dtype=float)
    sum_vals = cumulative[-1]
    if sum_vals == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_vals) / n
    return gini


def hybrid_health_score(motifs: Sequence[TemporalMotif]) -> np.ndarray:
    """
    Compute the unified health/priority score for each motif.

    hybrid_score_i = recovery_priority(morph_i) *
                     mean_j RBF(emb_i, emb_j) *
                     (1 - Gini_i)

    where Gini_i is the Gini coefficient of the affinity vector of motif i.
    """
    embeddings = np.array([np.array(m.embedding, dtype=float) for m in motifs])
    n = embeddings.shape[0]

    # Pairwise RBF matrix (dense)
    rbf_mat = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            rbf_mat[i, j] = rbf_similarity(embeddings[i], embeddings[j])

    # Mean similarity to all other motifs (including self for stability)
    mean_sim = rbf_mat.mean(axis=1)

    # Gini per row (inequality of affinities)
    gini_vec = np.array([gini_coefficient(rbf_mat[i, :]) for i in range(n)])

    # Recovery priority from morphology
    priority_vec = np.array([recovery_priority(m.morphology) for m in motifs])

    hybrid = priority_vec * mean_sim * (1.0 - gini_vec)
    # Normalise to [0, 1] for stability
    min_h, max_h = hybrid.min(), hybrid.max()
    if max_h > min_h:
        hybrid = (hybrid - min_h) / (max_h - min_h)
    else:
        hybrid = np.zeros_like(hybrid)
    return hybrid


# ----------------------------------------------------------------------
# Parent‑B style helpers (XGBoost‑ish)
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(x, dtype=float)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)
    return out


def binary_logistic_grad_hess(
    y_true: np.ndarray,
    margin: np.ndarray,
    endpoint_health: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    First‑ and second‑order gradients for binary logistic loss, scaled by
    endpoint health (here the hybrid health scores).
    """
    p = sigmoid(margin)
    g = (p - y_true) * endpoint_health
    h = (p * (1.0 - p)) * endpoint_health
    return g, h


def optimal_leaf_weight(
    gradient_sum: float,
    hessian_sum: float,
    reg_lambda: float = 1.0,
    endpoint_health: float = 1.0,
) -> float:
    """
    XGBoost leaf weight formula with health‑scaled regularisation.
    """
    return -gradient_sum / (hessian_sum + reg_lambda) * endpoint_health


def split_gain(
    left_grad: float,
    left_hess: float,
    right_grad: float,
    right_hess: float,
    *,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    endpoint_health: float = 1.0,
) -> float:
    """
    Gain of a binary split, scaled by the health factor.
    """
    gain_left = (left_grad ** 2) / (left_hess + reg_lambda)
    gain_right = (right_grad ** 2) / (right_hess + reg_lambda)
    gain_parent = ((left_grad + right_grad) ** 2) / (left_hess + right_hess + reg_lambda)
    raw_gain = 0.5 * (gain_left + gain_right - gain_parent) - gamma
    return raw_gain * endpoint_health


# ----------------------------------------------------------------------
# Demonstration: a single‑stump trainer that uses the hybrid health scores
# ----------------------------------------------------------------------
def train_one_stump(
    X: np.ndarray,
    y: np.ndarray,
    motifs: Sequence[TemporalMotif],
    feature_index: int = 0,
    threshold: float = None,
) -> Tuple[float, float, float]:
    """
    Train a depth‑1 decision tree (stump) on a single feature.
    The leaf weights are computed with `optimal_leaf_weight` where the
    endpoint health is the hybrid health score of the corresponding motif.
    Returns (threshold, left_weight, right_weight).
    """
    n_samples = X.shape[0]
    if threshold is None:
        # simple heuristic: median of the chosen feature
        threshold = np.median(X[:, feature_index])

    # Partition indices
    left_idx = X[:, feature_index] <= threshold
    right_idx = ~left_idx

    # Compute hybrid health scores for the *samples* (assume one‑to‑one mapping)
    health = hybrid_health_score(motifs)

    # Gradients/Hessians from logistic loss (margin = 0 initial)
    margin = np.zeros(n_samples, dtype=float)
    g, h = binary_logistic_grad_hess(y, margin, health)

    # Sums per leaf
    left_grad = float(g[left_idx].sum())
    left_hess = float(h[left_idx].sum())
    right_grad = float(g[right_idx].sum())
    right_hess = float(h[right_idx].sum())

    # Leaf weights using health‑scaled formula (average health per leaf)
    left_health = float(health[left_idx].mean()) if left_idx.any() else 1.0
    right_health = float(health[right_idx].mean()) if right_idx.any() else 1.0

    left_weight = optimal_leaf_weight(left_grad, left_hess, endpoint_health=left_health)
    right_weight = optimal_leaf_weight(right_grad, right_hess, endpoint_health=right_health)

    # Optionally compute split gain (just for illustration)
    gain = split_gain(
        left_grad, left_hess, right_grad, right_hess,
        endpoint_health=(left_health + right_health) / 2.0,
    )
    # The gain is not used further; it's printed in the smoke test.

    return threshold, left_weight, right_weight, gain


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Create synthetic motifs
    n = 20
    motifs: List[TemporalMotif] = []
    for i in range(n):
        embedding = tuple(np.random.randn(8))
        morph = Morphology(
            length=random.uniform(0.5, 5.0),
            width=random.uniform(0.5, 5.0),
            height=random.uniform(0.5, 5.0),
            mass=random.uniform(1.0, 20.0),
        )
        motifs.append(
            TemporalMotif(
                pattern=("A", "B", "C"),
                support=random.randint(1, 10),
                embedding=embedding,
                morphology=morph,
            )
        )

    # Feature matrix (use first 3 dimensions of embedding as toy features)
    X = np.array([m.embedding[:3] for m in motifs], dtype=float)
    # Binary labels
    y = np.random.randint(0, 2, size=n).astype(float)

    # Train a single stump
    thresh, w_left, w_right, gain = train_one_stump(X, y, motifs, feature_index=0)

    print(f"Stump threshold on feature 0: {thresh:.4f}")
    print(f"Left leaf weight: {w_left:.4f}")
    print(f"Right leaf weight: {w_right:.4f}")
    print(f"Split gain (health‑scaled): {gain:.6f}")

    # Verify that hybrid health scores lie in [0,1]
    health_vec = hybrid_health_score(motifs)
    assert np.all(health_vec >= 0.0) and np.all(health_vec <= 1.0), "Health scores out of bounds"

    print("Smoke test completed successfully.")