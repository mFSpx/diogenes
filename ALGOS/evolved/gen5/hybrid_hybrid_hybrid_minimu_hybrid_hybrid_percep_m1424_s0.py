# DARWIN HAMMER — match 1424, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s1.py (gen2)
# parent_b: hybrid_hybrid_perceptual_de_hybrid_semantic_neig_m968_s1.py (gen4)
# born: 2026-05-29T23:36:13Z

"""Hybrid Bayesian‑RBF‑Morphology Module

Parents:
- **Parent A**: Bayesian tree cost computation with Euclidean edge lengths,
  label scoring and probability updates.
- **Parent B**: Radial‑basis‑function (Gaussian) surrogate model, perceptual‑hash
  clustering and morphology‑driven recovery priority (ρ).

Mathematical Bridge:
Both parents express *similarity* between two entities:

* Parent A produces a posterior probability  
  `p̂ = prior·likelihood / (likelihood·prior + false_positive·(1‑prior))`.
* Parent B produces a Gaussian kernel  
  `K = exp(-(ε·‖x‑c‖)²)`.

We treat the Bayesian posterior `p̂` as a *data‑driven similarity* and feed it
as the amplitude of the Gaussian RBF kernel.  The combined similarity for an
edge (a,b) is therefore  


S_ab = p̂_ab · exp(-(ε·d_ab)²)


where `d_ab` is the Euclidean distance between the node coordinates.  This
similarity is finally modulated by the morphology‑derived recovery priority  


ρ = righting_time / max_index   ∈ [0,1]


and the hybrid edge cost is defined as  


cost_ab = (1‑ρ) · (1‑S_ab) · d_ab .


The three public functions below demonstrate fitting a trivial surrogate
model (estimating a global scaling factor), predicting a hybrid cost for a
new edge and computing the dynamic failure threshold used by the circuit‑
breaker logic of Parent B.
"""

import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]
Morphology = Dict[str, float]  # expects at least 'righting_time' and 'max_index'


# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = likelihood·prior + false_positive·(1‑prior)."""
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_posterior(prior: float, likelihood: float, false_positive: float) -> float:
    """Posterior probability for an edge given prior, likelihood and false‑positive."""
    marginal = bayes_marginal(prior, likelihood, false_positive)
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial‑basis‑function kernel."""
    return math.exp(-((epsilon * r) ** 2))


def morphology_priority(attrs: Morphology) -> float:
    """Recovery priority ρ = righting_time / max_index, clipped to [0,1]."""
    rt = attrs.get("righting_time", 0.0)
    mx = attrs.get("max_index", 1.0)  # avoid division by zero
    rho = rt / mx if mx != 0 else 0.0
    return max(0.0, min(1.0, rho))


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def hybrid_similarity(
    distance: float,
    posterior: float,
    epsilon: float = 1.0,
) -> float:
    """
    Combined similarity S = posterior * Gaussian_RBF(distance).
    """
    return posterior * gaussian_rbf(distance, epsilon)


def hybrid_edge_cost(
    a: str,
    b: str,
    nodes: Dict[str, Point],
    prior_prob: Dict[str, float],
    likelihoods: Dict[Edge, float],
    false_positives: Dict[Edge, float],
    epsilon: float,
    morph_attrs: Morphology,
) -> float:
    """
    Compute the hybrid cost for edge (a,b) using the formula:

        cost = (1‑ρ) * (1‑S) * d

    where
        ρ  – morphology priority,
        S  – hybrid similarity,
        d  – Euclidean distance.
    """
    d = euclidean_distance(nodes[a], nodes[b])
    post = bayes_posterior(prior_prob[a], likelihoods[(a, b)], false_positives[(a, b)])
    S = hybrid_similarity(d, post, epsilon)
    rho = morphology_priority(morph_attrs)
    return (1.0 - rho) * (1.0 - S) * d


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def fit_hybrid_model(
    nodes: Dict[str, Point],
    edges: List[Edge],
    prior_prob: Dict[str, float],
    likelihoods: Dict[Edge, float],
    false_positives: Dict[Edge, float],
    epsilon: float = 1.0,
) -> float:
    """
    Fit a trivial surrogate model that learns a global scaling factor α
    minimizing squared error between α·posterior and observed edge
    distances.  The returned α is used as a multiplier for future
    posterior values inside `hybrid_similarity`.

    This is deliberately lightweight – the focus is on demonstrating the
    mathematical fusion, not on building a full RBF trainer.
    """
    distances = []
    posteriors = []
    for a, b in edges:
        distances.append(euclidean_distance(nodes[a], nodes[b]))
        posteriors.append(
            bayes_posterior(prior_prob[a], likelihoods[(a, b)], false_positives[(a, b)])
        )
    D = np.array(distances, dtype=float)
    P = np.array(posteriors, dtype=float)

    # Solve α in least‑squares sense: minimize ||α·P - D||²
    # α = (P·D) / (P·P)  (provided P·P ≠ 0)
    numerator = np.dot(P, D)
    denominator = np.dot(P, P) + 1e-12  # guard against division by zero
    alpha = numerator / denominator
    return float(alpha)


def predict_hybrid_cost(
    a: str,
    b: str,
    nodes: Dict[str, Point],
    prior_prob: Dict[str, float],
    likelihoods: Dict[Edge, float],
    false_positives: Dict[Edge, float],
    epsilon: float,
    morph_attrs: Morphology,
    model_alpha: float,
) -> float:
    """
    Predict hybrid cost for a *new* edge (a,b) using a pre‑trained α.
    The posterior is scaled by α before entering the similarity term.
    """
    d = euclidean_distance(nodes[a], nodes[b])
    raw_post = bayes_posterior(prior_prob[a], likelihoods[(a, b)], false_positives[(a, b)])
    scaled_post = model_alpha * raw_post
    # Clip scaled posterior to [0,1] to keep it a valid probability.
    scaled_post = max(0.0, min(1.0, scaled_post))
    S = hybrid_similarity(d, scaled_post, epsilon)
    rho = morphology_priority(morph_attrs)
    return (1.0 - rho) * (1.0 - S) * d


def compute_dynamic_threshold(base_threshold: float, morph_attrs: Morphology) -> float:
    """
    Dynamic failure threshold used by the endpoint circuit‑breaker:

        threshold = base_threshold * (1 - ρ)

    where ρ is the morphology‑derived priority.
    """
    rho = morphology_priority(morph_attrs)
    return base_threshold * (1.0 - rho)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple graph with three nodes forming a line
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (2.0, 0.0),
    }
    edges = [("A", "B"), ("B", "C")]

    # Probabilistic parameters (synthetic)
    prior_prob = {"A": 0.6, "B": 0.7, "C": 0.5}
    likelihoods = {("A", "B"): 0.9, ("B", "C"): 0.8}
    false_positives = {("A", "B"): 0.1, ("B", "C"): 0.2}

    # Morphology attributes (synthetic)
    morph_attrs = {"righting_time": 3.0, "max_index": 5.0}

    epsilon = 0.8
    base_threshold = 10.0

    # Fit the hybrid surrogate (learn α)
    alpha = fit_hybrid_model(
        nodes,
        edges,
        prior_prob,
        likelihoods,
        false_positives,
        epsilon,
    )
    print(f"Learned alpha: {alpha:.4f}")

    # Predict cost for a new edge A‑C using the fitted model
    cost_ac = predict_hybrid_cost(
        "A",
        "C",
        nodes,
        prior_prob,
        likelihoods,
        false_positives,
        epsilon,
        morph_attrs,
        alpha,
    )
    print(f"Hybrid cost for edge A‑C: {cost_ac:.4f}")

    # Compute dynamic threshold
    dyn_thr = compute_dynamic_threshold(base_threshold, morph_attrs)
    print(f"Dynamic failure threshold: {dyn_thr:.4f}")

    # Verify that direct edge cost (using the full formula) is consistent
    direct_cost_ac = hybrid_edge_cost(
        "A",
        "C",
        nodes,
        prior_prob,
        likelihoods,
        false_positives,
        epsilon,
        morph_attrs,
    )
    print(f"Direct hybrid edge cost A‑C: {direct_cost_ac:.4f}")

    # Simple sanity check: cost should be non‑negative and less than raw distance
    raw_dist_ac = euclidean_distance(nodes["A"], nodes["C"])
    assert 0.0 <= cost_ac <= raw_dist_ac, "Predicted cost out of expected bounds"
    assert 0.0 <= direct_cost_ac <= raw_dist_ac, "Direct cost out of expected bounds"

    print("Smoke test completed successfully.")