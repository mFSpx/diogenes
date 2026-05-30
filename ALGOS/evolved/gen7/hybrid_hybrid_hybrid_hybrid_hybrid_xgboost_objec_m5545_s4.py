# DARWIN HAMMER — match 5545, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_semant_hybrid_hybrid_hybrid_m2659_s4.py (gen6)
# parent_b: hybrid_xgboost_objective_hybrid_hybrid_hybrid_m201_s0.py (gen4)
# born: 2026-05-30T00:02:46Z

"""Hybrid Algorithm Fusion of:
- hybrid_hybrid_semantic_neig_hybrid_temporal_moti_m47_s0 (semantic neighbor & RBF similarity with Gini weighting)
- hybrid_xgboost_objective_hybrid_hybrid_hybrid_m201_s0 (XGBoost leaf optimization with endpoint health scaling)

Mathematical Bridge:
The bridge is the *morphology‑derived health factor* `recovery_priority`.  
In Parent A this factor weights the physical right‑hand‑time index; in Parent B it
scales the gradient, Hessian and leaf‑weight formulas of XGBoost.  By using the
hybrid score (semantic‑RBF‑Gini) as the supervised target `y` and the same
`recovery_priority` as the endpoint‑health regularizer, the two topologies are
fused into a single learning‑decision metric:

    y_true      = hybrid_score(morphology, motif, RBF, Gini)
    health      = recovery_priority(morphology)                # scaling factor
    grad, hess  = binary_logistic_grad_hess(y_true, margin, health)
    leaf_weight = optimal_leaf_weight(∑grad, ∑hess, λ, health)

The code below implements the core mathematics, provides three hybrid
operations and a smoke‑test that runs end‑to‑end without external data."""
import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Set, Hashable, Sequence
import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Physical dimensions used to compute a health/priority factor."""
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class TemporalMotif:
    """Semantic representation of a temporal pattern."""
    pattern: Tuple[str, ...]
    support: int
    embedding: Tuple[float, ...]  # semantic vector

@dataclass(frozen=True)
class HybridMotif:
    """Resulting motif after hybrid scoring."""
    pattern: Tuple[str, ...]
    support: int
    centroid_lat: float
    centroid_lon: float
    score: float


# ----------------------------------------------------------------------
# Parent‑A style components
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Simple sphericity proxy: volume / (surface area)⁽³ᐟ²⁾ (normalized)."""
    if min(length, width, height) <= 0:
        return 0.0
    volume = length * width * height
    surface = 2 * (length * width + width * height + height * length)
    return volume / (surface ** 1.5)


def recovery_priority(morph: Morphology) -> float:
    """
    Morphology‑driven health factor.
    Higher mass and more spherical shapes receive higher priority.
    The result is normalised to [0, 1].
    """
    sphere = sphericity_index(morph.length, morph.width, morph.height)
    # mass is scaled by a heuristic max mass of 1000 for normalisation
    mass_factor = min(morph.mass / 1000.0, 1.0)
    priority = 0.6 * sphere + 0.4 * mass_factor
    return max(0.0, min(priority, 1.0))


def rbf_similarity_matrix(
    embeddings: Sequence[Tuple[float, ...]],
    gamma: float = 0.5,
) -> np.ndarray:
    """
    Gaussian RBF affinity between all embeddings.
    Returns a symmetric (N, N) matrix where N = len(embeddings).
    """
    arr = np.asarray(embeddings, dtype=float)
    sq_norms = np.sum(arr ** 2, axis=1, keepdims=True)
    dists = sq_norms + sq_norms.T - 2.0 * arr @ arr.T
    np.maximum(dists, 0, out=dists)  # numerical safety
    return np.exp(-gamma * dists)


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
    return float(gini)


def hybrid_score(
    morph: Morphology,
    motif: TemporalMotif,
    rbf_mat: np.ndarray,
    neighbor_indices: Sequence[int],
) -> float:
    """
    Core metric from Parent A.
    - health = recovery_priority(morph)
    - mean_rbf = mean RBF similarity to semantic neighbours
    - inequality = 1 - Gini(affinity vector of the motif)
    """
    health = recovery_priority(morph)

    # affinity vector for this motif (row of the matrix)
    affinities = rbf_mat[motif.support]  # use support as index placeholder
    # restrict to neighbour indices (semantic neighbours)
    neigh_aff = affinities[list(neighbor_indices)]
    mean_rbf = float(np.mean(neigh_aff)) if neigh_aff.size > 0 else 0.0

    # Gini over the full affinity row
    gini = gini_coefficient(affinities)
    inequality = 1.0 - gini

    return health * mean_rbf * inequality


# ----------------------------------------------------------------------
# Parent‑B style components (XGBoost‑like)
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def binary_logistic_grad_hess(
    y_true: np.ndarray,
    margin: np.ndarray,
    endpoint_health: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    First and second order gradients for binary logistic loss,
    scaled element‑wise by endpoint health.
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
    XGBoost optimal leaf weight, further scaled by endpoint health.
    """
    denom = hessian_sum + reg_lambda
    if denom == 0:
        return 0.0
    return -gradient_sum / denom * endpoint_health


def split_gain(
    left_gradient: float,
    left_hessian: float,
    right_gradient: float,
    right_hessian: float,
    *,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    endpoint_health: float = 1.0,
) -> float:
    """
    Split gain formula, multiplied by endpoint health to embed the
    morphology‑derived regularisation.
    """
    def _gain(g: float, h: float) -> float:
        return (g ** 2) / (h + reg_lambda)

    gain_left = _gain(left_gradient, left_hessian)
    gain_right = _gain(right_gradient, right_hessian)
    gain_parent = _gain(left_gradient + right_gradient, left_hessian + right_hessian)

    raw_gain = 0.5 * (gain_left + gain_right - gain_parent) - gamma
    return max(0.0, raw_gain) * endpoint_health


# ----------------------------------------------------------------------
# Hybrid operations that combine both parent logics
# ----------------------------------------------------------------------
def compute_hybrid_dataset(
    morphologies: Sequence[Morphology],
    motifs: Sequence[TemporalMotif],
    gamma_rbf: float = 0.5,
    k_neighbors: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns three arrays aligned with `motifs`:
    - y: hybrid scores (target for XGBoost)
    - health: recovery priority per motif (endpoint health)
    - margin_init: random initial margin predictions for the first boosting round
    """
    embeddings = [m.embedding for m in motifs]
    rbf_mat = rbf_similarity_matrix(embeddings, gamma=gamma_rbf)

    y = np.empty(len(motifs), dtype=float)
    health = np.empty(len(motifs), dtype=float)

    for idx, (morph, motif) in enumerate(zip(morphologies, motifs)):
        # find k nearest neighbours in embedding space (excluding self)
        distances = np.linalg.norm(
            np.asarray(embeddings) - np.asarray(motif.embedding), axis=1
        )
        neighbour_idx = np.argsort(distances)[1 : k_neighbors + 1]
        y[idx] = hybrid_score(morph, motif, rbf_mat, neighbour_idx)
        health[idx] = recovery_priority(morph)

    # initialise margins randomly around 0 (logit space)
    margin_init = np.random.uniform(-0.1, 0.1, size=len(motifs))
    return y, health, margin_init


def one_boosting_step(
    y_true: np.ndarray,
    margin: np.ndarray,
    health: np.ndarray,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    Performs a single gradient‑boosting update using the hybrid leaf weight
    and split‑gain formulas.
    Returns the updated margin vector and the leaf weight applied globally.
    (In a full tree implementation each leaf would have its own weight; here we
    demonstrate the fused mathematics with a single leaf for simplicity.)
    """
    g, h = binary_logistic_grad_hess(y_true, margin, health)

    grad_sum = float(np.sum(g))
    hess_sum = float(np.sum(h))
    # Use the average health as the scaling factor for the leaf weight
    avg_health = float(np.mean(health))
    leaf_w = optimal_leaf_weight(grad_sum, hess_sum, reg_lambda, avg_health)

    # Update margin (Newton‑Raphson step)
    margin_new = margin + leaf_w

    # Demonstrate split gain on a dummy split (left/right halves)
    split_idx = len(margin) // 2
    left_g, left_h = float(np.sum(g[:split_idx])), float(np.sum(h[:split_idx]))
    right_g, right_h = float(np.sum(g[split_idx:])), float(np.sum(h[split_idx:]))

    gain = split_gain(
        left_g,
        left_h,
        right_g,
        right_h,
        reg_lambda=reg_lambda,
        gamma=gamma,
        endpoint_health=avg_health,
    )
    return margin_new, gain


def evaluate_hybrid_model(
    morphologies: Sequence[Morphology],
    motifs: Sequence[TemporalMotif],
    steps: int = 3,
) -> Tuple[np.ndarray, List[float]]:
    """
    Runs a few boosting iterations and returns the final margin predictions
    together with the list of split gains observed at each step.
    """
    y, health, margin = compute_hybrid_dataset(morphologies, motifs)
    gains = []
    for _ in range(steps):
        margin, gain = one_boosting_step(y, margin, health)
        gains.append(gain)
    return margin, gains


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic problem (5 motifs)
    random.seed(42)
    np.random.seed(42)

    morphs = [
        Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass=random.uniform(10, 500),
        )
        for _ in range(5)
    ]

    motifs = [
        TemporalMotif(
            pattern=("A", "B", "C"),
            support=i,
            embedding=tuple(np.random.randn(8)),
        )
        for i in range(5)
    ]

    final_margin, split_gains = evaluate_hybrid_model(morphs, motifs, steps=4)

    print("Final margin predictions:", final_margin)
    print("Split gains per iteration:", split_gains)