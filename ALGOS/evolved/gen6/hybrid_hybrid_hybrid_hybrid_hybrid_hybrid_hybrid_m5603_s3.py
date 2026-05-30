# DARWIN HAMMER — match 5603, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m1120_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m1217_s2.py (gen5)
# born: 2026-05-30T00:03:16Z

"""Hybrid Algorithm: Curvature‑Fisher Fusion

Parents:
- hybrid_hybrid_bandit_hybrid_hybrid_krampu_m9_s4.py (node‑wise curvature proxy, linear test‑time training)
- hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m1217_s2.py (Gaussian beam, Fisher information score)

Mathematical Bridge:
The Ollivier‑Ricci curvature vector 𝜅 obtained from the adjacency matrix is interpreted as a
temperature‑like factor 𝜏 = exp(‑β·𝜅).  In the Fisher‑score formulation the Gaussian‑beam width σ
controls the locality of the information density.  By substituting σ←σ·𝜏 the graph geometry
directly modulates the Fisher information, coupling structural curvature with statistical
sensitivity.  The resulting weighted Fisher score is then used as a confidence weight in a
Normalized Least‑Mean‑Squares (NLMS) test‑time update, completing the fusion of the two
parental topologies into a single unified system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ----------------------------------------------------------------------
# Core mathematical components
# ----------------------------------------------------------------------
def compute_curvature(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Ollivier‑Ricci curvature proxy for an undirected weighted graph.
    Returns a 1‑D array κ where κ[i] is the summed curvature contribution of node i.
    """
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("adj_matrix must be a square matrix")
    n = adj_matrix.shape[0]
    degree = np.sum(adj_matrix, axis=1, keepdims=True)  # column vector of node strengths
    # Avoid division by zero
    degree[degree == 0] = 1.0
    prob_matrix = adj_matrix / degree  # transition probabilities
    curvature = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] > 0:
                pij = prob_matrix[i, j]
                pj = prob_matrix[j, :]
                pi = prob_matrix[i, :]
                # Earth‑Mover distance proxy: log ratio of edge weight to product of node strengths
                log_term = math.log(adj_matrix[i, j] / (np.sum(adj_matrix[i]) * np.sum(adj_matrix[j]) + 1e-12) + 1e-12)
                curvature[i] += adj_matrix[i, j] * log_term
    return curvature


def temperature_from_curvature(curvature: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """
    Maps curvature values to a temperature‑like factor τ = exp(‑β·κ).
    The factor is bounded in (0, 1] and will be used to modulate Fisher‑score width.
    """
    return np.exp(-beta * curvature)


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Standard Gaussian kernel."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def weighted_fisher_score(theta: float, center: float, base_width: float,
                          temperature: float) -> float:
    """
    Fisher score where the Gaussian width is scaled by a temperature factor derived
    from graph curvature: σ = base_width * τ.
    """
    width = base_width * max(temperature, 1e-6)  # keep width positive
    return fisher_score(theta, center, width)


# ----------------------------------------------------------------------
# NLMS (Normalized Least‑Mean‑Squares) test‑time training
# ----------------------------------------------------------------------
def nlms_update(weights: np.ndarray, x: np.ndarray, d: float, mu: float = 0.1) -> np.ndarray:
    """
    Performs one NLMS weight update:
        e = d - wᵀx
        w ← w + (mu / (||x||² + ε)) * e * x
    Returns the updated weight vector.
    """
    eps = 1e-12
    x_norm_sq = np.dot(x, x) + eps
    error = d - np.dot(weights, x)
    adaptation = (mu / x_norm_sq) * error * x
    return weights + adaptation


# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_step(adj_matrix: np.ndarray,
                theta: float,
                center: float,
                base_width: float,
                feature_vec: np.ndarray,
                target: float,
                weights: np.ndarray,
                beta: float = 0.5,
                mu: float = 0.1) -> Tuple[np.ndarray, float]:
    """
    Executes one hybrid iteration:
      1. Compute node‑wise curvature κ from the adjacency matrix.
      2. Derive temperature τ = exp(‑β·κ) and take its mean as a global scaling factor.
      3. Compute a weighted Fisher score for the supplied theta using τ.
      4. Form a scalar confidence weight w_conf = sigmoid(FisherScore).
      5. Scale the feature vector by w_conf and perform an NLMS update.
    Returns the updated weights and the confidence weight.
    """
    # 1. Curvature
    curvature = compute_curvature(adj_matrix)

    # 2. Temperature scaling (global)
    temperature_vec = temperature_from_curvature(curvature, beta=beta)
    global_temperature = float(np.mean(temperature_vec))

    # 3. Weighted Fisher score
    f_score = weighted_fisher_score(theta, center, base_width, global_temperature)

    # 4. Confidence via sigmoid to keep in (0,1)
    w_conf = 1.0 / (1.0 + math.exp(-f_score))

    # 5. NLMS update with confidence‑scaled features
    scaled_features = feature_vec * w_conf
    new_weights = nlms_update(weights, scaled_features, target, mu=mu)

    return new_weights, w_conf


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small synthetic graph (5 nodes)
    rng = np.random.default_rng(42)
    adj = rng.random((5, 5))
    adj = (adj + adj.T) / 2  # make symmetric
    np.fill_diagonal(adj, 0)  # no self‑loops

    # Parameters for Fisher component
    theta_val = 0.7
    center_val = 0.5
    base_width_val = 0.2

    # Feature vector and target for NLMS
    feat = rng.random(5)
    target_val = 1.3
    init_weights = rng.random(5)

    # Run hybrid step
    updated_w, confidence = hybrid_step(
        adj_matrix=adj,
        theta=theta_val,
        center=center_val,
        base_width=base_width_val,
        feature_vec=feat,
        target=target_val,
        weights=init_weights,
        beta=0.3,
        mu=0.05,
    )

    print("Updated weights:", updated_w)
    print("Confidence weight (sigmoid of Fisher score):", confidence)