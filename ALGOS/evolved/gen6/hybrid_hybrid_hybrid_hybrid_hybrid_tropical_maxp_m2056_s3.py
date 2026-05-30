# DARWIN HAMMER — match 2056, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_percep_m598_s0.py (gen5)
# parent_b: hybrid_tropical_maxplus_hybrid_hybrid_hard_t_m852_s0.py (gen3)
# born: 2026-05-29T23:40:36Z

"""Hybrid NLMS‑RBF with Tropical Max‑Plus Model Loading

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – adaptive NLMS weight update combined with Radial Basis Function
  (RBF) similarity graphs.
* **Parent B** – tropical max‑plus algebra used to evaluate a cost (here a
  “tropical load cost”) from model RAM requirements and stylometry scores.

**Mathematical bridge**

The bridge is the *tropical cost* λ that is derived from model characteristics
using tropical max‑plus evaluation.  λ is injected into the NLMS update as a
scale factor for the learning‑rate μ:


μ' = μ / (1 + λ)
w_{t+1} = w_t + μ'·e·x / (‖x‖² + ε)


Thus the adaptive filter adapts slower for expensive models (high λ) and
faster for cheap ones, while the RBF kernel provides a non‑linear feature
mapping that is weighted by the same adaptive coefficients.  The resulting
system can be used for tasks such as adaptive similarity‑based model selection
or online regression with resource‑aware learning.

The module supplies three primary hybrid operations:
1. `tropical_cost` – evaluates a tropical polynomial from RAM and stylometry.
2. `rbf_features` – builds an RBF feature vector from input samples.
3. `hybrid_nlms_update` – performs an NLMS weight update modulated by the
   tropical cost.
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
class ModelInfo:
    """Light‑weight descriptor for a model used in tropical evaluation."""
    name: str
    ram_mb: int
    stylometry_score: float  # e.g. similarity to a target stylometric fingerprint


@dataclass(frozen=True)
class Node:
    """Graph node used by the RBF similarity graph."""
    id: int
    weight: float


# ----------------------------------------------------------------------
# Tropical max‑plus utilities (Parent B)
# ----------------------------------------------------------------------


def tropical_add(a: float, b: float) -> float:
    """Tropical addition (max)."""
    return max(a, b)


def tropical_mul(a: float, b: float) -> float:
    """Tropical multiplication (ordinary addition)."""
    return a + b


def tropical_polynomial(coeffs: List[float], x: float) -> float:
    """
    Evaluate a tropical polynomial  P(x) = max_i (coeff_i + i·x).

    Args:
        coeffs: List of tropical coefficients (one per monomial degree).
        x: The tropical variable (here a stylometry score).

    Returns:
        The tropical polynomial value.
    """
    values = [tropical_mul(coeff, i * x) for i, coeff in enumerate(coeffs)]
    result = values[0]
    for v in values[1:]:
        result = tropical_add(result, v)
    return result


def tropical_cost(model: ModelInfo, base_coeffs: List[float]) -> float:
    """
    Compute a resource‑aware cost for a model using tropical evaluation.

    The RAM requirement is injected as an offset to the polynomial coefficients
    so that larger RAM yields higher cost.

    Args:
        model: ModelInfo instance.
        base_coeffs: Base tropical coefficients (same length for all models).

    Returns:
        Non‑negative tropical cost λ.
    """
    # Offset the constant term by RAM (scaled down to keep numbers moderate)
    coeffs = base_coeffs.copy()
    coeffs[0] += model.ram_mb / 1024.0  # treat 1 GiB ≈ 1 unit
    return tropical_polynomial(coeffs, model.stylometry_score)


# ----------------------------------------------------------------------
# RBF utilities (Parent A)
# ----------------------------------------------------------------------


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(np.sum((a - b) ** 2))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))


def rbf_features(x: np.ndarray, centers: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Compute RBF feature vector φ(x) = [exp(-ε²‖x‑c_i‖²)]_i.

    Args:
        x: Input vector.
        centers: Array of shape (M, D) containing M RBF centers.
        epsilon: Width parameter of the Gaussian.

    Returns:
        Feature vector of length M.
    """
    diffs = centers - x  # broadcasting
    dists = np.linalg.norm(diffs, axis=1)
    return np.exp(- (epsilon * dists) ** 2)


# ----------------------------------------------------------------------
# Hybrid NLMS update (core fusion)
# ----------------------------------------------------------------------


def predict(weights: np.ndarray, phi: np.ndarray) -> float:
    """Linear prediction using weight vector and RBF features."""
    return float(np.dot(weights, phi))


def hybrid_nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    model: ModelInfo,
    centers: np.ndarray,
    base_mu: float = 0.5,
    eps: float = 1e-9,
    tropical_coeffs: List[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Perform a single NLMS update where the learning rate is scaled by a tropical cost.

    Steps:
    1. Compute RBF feature vector φ(x).
    2. Predict y = w·φ.
    3. Compute error e = target – y.
    4. Evaluate tropical cost λ for the supplied model.
    5. Scale the base learning rate: μ' = μ / (1 + λ).
    6. NLMS weight update using φ as the regressor.

    Returns:
        (new_weights, error)
    """
    if tropical_coeffs is None:
        tropical_coeffs = [0.0, 0.5, 0.2]  # default small coefficients

    phi = rbf_features(x, centers)
    y = predict(weights, phi)
    error = target - y

    # Tropical cost λ
    lam = tropical_cost(model, tropical_coeffs)

    mu_scaled = base_mu / (1.0 + lam)

    power = float(np.dot(phi, phi) + eps)
    new_weights = weights + mu_scaled * error * phi / power
    return new_weights, error


# ----------------------------------------------------------------------
# Graph construction using RBF similarities (Parent A)
# ----------------------------------------------------------------------


def construct_rbf_graph(weights: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build a fully connected similarity graph where edge weights are derived
    from the RBF‑based similarity of node weights.

    Similarity between node i and j:
        s_ij = exp(-|w_i - w_j|²)
    """
    n = len(weights)
    graph: Dict[int, List[Tuple[int, float]]] = {}
    for i in range(n):
        graph[i] = []
        for j in range(n):
            if i == j:
                continue
            diff = weights[i] - weights[j]
            sim = math.exp(-diff * diff)
            graph[i].append((j, sim))
    return graph


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Dummy data
    D = 5  # input dimension
    M = 8  # number of RBF centers / weight dimension

    # Random centers for the RBFs
    centers = np.random.randn(M, D)

    # Initial NLMS weights
    w = np.zeros(M)

    # Synthetic input / target pair
    x = np.random.randn(D)
    target = 1.23

    # Model information (used for tropical cost)
    model = ModelInfo(name="demo_model", ram_mb=2048, stylometry_score=0.37)

    # Perform a hybrid update
    w_new, err = hybrid_nlms_update(
        weights=w,
        x=x,
        target=target,
        model=model,
        centers=centers,
        base_mu=0.7,
        tropical_coeffs=[0.0, 0.3, 0.1, 0.05],
    )

    # Build the similarity graph from the updated weights
    graph = construct_rbf_graph(w_new)

    # Simple sanity checks (will raise if something is wrong)
    assert isinstance(w_new, np.ndarray) and w_new.shape == (M,)
    assert isinstance(err, float)
    assert isinstance(graph, dict) and len(graph) == M

    print("Hybrid NLMS update completed successfully.")
    print(f"Error after update: {err:.6f}")
    print(f"First three updated weights: {w_new[:3]}")
    print(f"Graph node 0 connections (first three): {graph[0][:3]}")