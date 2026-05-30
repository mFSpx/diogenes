# DARWIN HAMMER — match 4814, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_infotaxis_hyb_m2474_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_percep_m1424_s0.py (gen5)
# born: 2026-05-29T23:58:20Z

"""
Hybrid Algorithm: Hard Truth Math Model + Infotaxis-Semantic Morphology + Bayesian-RBF-Morphology Module

Parents:
- hard_truth_math_model_pool_m8_s3.py (stylometry / LSM utilities)
- hybrid_infotaxis_hybrid_semantic_neig_m739_s4.py (Hybrid Infotaxis–Semantic Morphology System)
- hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s1.py (Bayesian tree cost computation with Euclidean edge lengths)
- hybrid_hybrid_perceptual_de_hybrid_semantic_neig_m968_s1.py (Radial‑basis‑function (Gaussian) surrogate model, perceptual‑hash clustering and morphology‑driven recovery priority)

Mathematical Bridge:
Both parents express *similarity* between two entities, which we treat as a *data‑driven similarity*. We found a mathematical bridge by combining the LSM vectors from the first parent with the Gaussian RBF kernel from the second parent. Specifically, we scale the LSM vectors with the recovery priority derived from the document's morphology and then use them as the amplitude of the Gaussian RBF kernel. The combined similarity for an edge (a,b) is therefore:


S_ab = p̂_ab · exp(-(ε·d_ab)²)

where d_ab is the Euclidean distance between the node coordinates. This similarity is finally modulated by the morphology‑derived recovery priority and the hybrid edge cost is defined as:


cost_ab = (1‑ρ) · (1‑S_ab) · d_ab.


The three public functions below demonstrate fitting a trivial surrogate model (estimating a global scaling factor), predicting a hybrid cost for a new edge and computing the dynamic failure threshold used by the circuit‑breaker logic of Parent B.
"""

import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path
from typing import Dict, List

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


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
def gaussian_rbf_kernel(a: Point, b: Point, epsilon: float) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-(epsilon * euclidean_distance(a, b)) ** 2)


# ----------------------------------------------------------------------
# Hybrid Algorithm building blocks
# ----------------------------------------------------------------------
def scale_lsm_vectors(text: str, morphology: Morphology, epsilon: float) -> Dict[str, float]:
    """Scale LSM vectors with recovery priority."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    recovery_priority = morphology['righting_time'] / morphology['max_index']
    scaled_vectors = {k: v * recovery_priority for k, v in cnt.items()}
    return scaled_vectors


def hybrid_similarity(a: Point, b: Point, scaled_vectors_a: Dict[str, float], scaled_vectors_b: Dict[str, float], epsilon: float) -> float:
    """Hybrid similarity."""
    p_hat = bayes_marginal(0.5, 0.5, 0.1)
    scaled_lsm_a = sum(scaled_vectors_a.values())
    scaled_lsm_b = sum(scaled_vectors_b.values())
    s_ab = p_hat * gaussian_rbf_kernel(a, b, epsilon) * scaled_lsm_a * scaled_lsm_b
    return s_ab


def hybrid_edge_cost(a: Point, b: Point, morphology: Morphology, epsilon: float) -> float:
    """Hybrid edge cost."""
    recovery_priority = morphology['righting_time'] / morphology['max_index']
    s_ab = hybrid_similarity(a, b, scale_lsm_vectors('', morphology, epsilon), scale_lsm_vectors('', morphology, epsilon), epsilon)
    cost_ab = (1 - recovery_priority) * (1 - s_ab) * euclidean_distance(a, b)
    return cost_ab


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def words(text: str) -> List[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> Dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return cnt


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def fit_surrogate_model(text: str, epsilon: float) -> float:
    """Fit a trivial surrogate model (estimating a global scaling factor)."""
    morphology = {
        'righting_time': random.random(),
        'max_index': random.random()
    }
    scaled_vectors = scale_lsm_vectors(text, morphology, epsilon)
    return sum(scaled_vectors.values())


def predict_edge_cost(a: Point, b: Point, text: str, epsilon: float) -> float:
    """Predict a hybrid cost for a new edge."""
    morphology = {
        'righting_time': random.random(),
        'max_index': random.random()
    }
    scaled_vectors_a = scale_lsm_vectors(text, morphology, epsilon)
    scaled_vectors_b = scale_lsm_vectors(text, morphology, epsilon)
    return hybrid_edge_cost(a, b, morphology, epsilon)


def compute_dynamic_failure_threshold(morphology: Morphology, epsilon: float) -> float:
    """Compute the dynamic failure threshold used by the circuit‑breaker logic."""
    recovery_priority = morphology['righting_time'] / morphology['max_index']
    return 1 - recovery_priority


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    a = (0.0, 0.0)
    b = (1.0, 1.0)
    epsilon = 0.1
    print(fit_surrogate_model(text, epsilon))
    print(predict_edge_cost(a, b, text, epsilon))
    print(compute_dynamic_failure_threshold({'righting_time': 0.5, 'max_index': 1.0}, epsilon))