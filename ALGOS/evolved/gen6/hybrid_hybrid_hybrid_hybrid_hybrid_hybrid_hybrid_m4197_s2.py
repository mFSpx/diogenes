# DARWIN HAMMER — match 4197, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_endpoi_m247_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_percyp_m1554_s1.py (gen5)
# born: 2026-05-29T23:54:11Z

"""
Hybrid Fusion of Hybrid_Ternary_Router Variational Free‑Energy (Parent A) and
Hybrid NLMS‑LTC Fisher Information Fusion (Parent B).

Mathematical Bridge
------------------
Both parents expose *morphology‑driven geometric indices* (sphericity, flatness).
Parent B uses these indices to weight a Fisher‑information regularizer inside an
NLMS weight update.  Parent A introduces a variational free‑energy term that
modulates the health score of an endpoint.

The fusion therefore treats the *free‑energy* as a scalar multiplier that
scales the Fisher‑information regularization inside the NLMS update and also
appears in the endpoint health score.  Concretely:

* Compute morphology indices `s = sphericity`, `f = flatness`.
* Compute a free‑energy `E` from a data vector.
* Define a *morphology factor* `M = s * f`.
* Fisher score `I = fisher_information_score(p, a)`.
* NLMS update with combined regularization:

w ← w + η * (e * x) / (‖x‖² + ε)  –  η * I * E * M * w

where `e = a – p` is the prediction error.
* Endpoint health score:

H = reliability * M * E

The endpoint with maximal `H` is selected.

The module implements three core functions that demonstrate this hybrid
operation and a smoke‑test under `if __name__ == "__main__":`.
"""

import math
import random
import sys
import pathlib
import hashlib
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Geometry utilities (shared by both parents)
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Parent A uses a volume‑based definition, Parent B a length‑based one.
    We adopt the volume‑based version for richer morphology information."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Common definition from both parents."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


# ----------------------------------------------------------------------
# Parent‑B style MinHash utilities (kept for completeness)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: List[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Parent‑B Fisher information utilities
# ----------------------------------------------------------------------
def fisher_information_score(predicted_output: float, actual_output: float) -> float:
    """Simple Fisher information approximation used as a regularizer."""
    diff = predicted_output - actual_output
    if diff == 0.0:
        # Avoid division by zero; use a large finite value
        return 1e12
    return (1.0 / diff) ** 2


# ----------------------------------------------------------------------
# Parent‑A variational free‑energy (simplified)
# ----------------------------------------------------------------------
def variational_free_energy(data: np.ndarray) -> float:
    """
    A toy variational free‑energy: negative log‑likelihood of a zero‑mean
    Gaussian with unit variance, summed over the data vector.
    """
    if data.ndim != 1:
        raise ValueError("data must be a 1‑D array")
    # -log p(x) = 0.5 * x^2 + const ; const omitted because it does not affect
    # relative scores.
    return -0.5 * np.sum(data ** 2)


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def compute_morphology_factor(length: float, width: float, height: float) -> float:
    """
    Returns the product of sphericity and flatness indices.
    This factor mediates between the free‑energy term and the Fisher regularizer.
    """
    s = sphericity_index(length, width, height)
    f = flatness_index(length, width, height)
    return s * f


def nlms_update_with_regularization(
    weight_vector: np.ndarray,
    feature_vector: np.ndarray,
    learning_rate: float,
    predicted_output: float,
    actual_output: float,
    free_energy: float,
    morphology_factor: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Hybrid NLMS update that incorporates:
      * standard NLMS correction,
      * Fisher‑information regularization,
      * scaling by free‑energy and morphology factor.

    Parameters
    ----------
    weight_vector : np.ndarray
        Current weight vector (shape: (n,)).
    feature_vector : np.ndarray
        Input feature vector (shape: (n,)).
    learning_rate : float
        Base learning rate η.
    predicted_output : float
        Model prediction for the current feature vector.
    actual_output : float
        Ground‑truth target.
    free_energy : float
        Variational free‑energy computed from auxiliary data.
    morphology_factor : float
        Product sphericity × flatness for the associated endpoint.
    epsilon : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Updated weight vector.
    """
    if weight_vector.shape != feature_vector.shape:
        raise ValueError("weight_vector and feature_vector must have the same shape")
    error = actual_output - predicted_output
    norm_sq = np.dot(feature_vector, feature_vector) + epsilon

    # Standard NLMS term
    nlms_correction = learning_rate * error * feature_vector / norm_sq

    # Fisher information regularizer
    fisher_score = fisher_information_score(predicted_output, actual_output)

    # Combined regularization term (scaled by free‑energy and morphology)
    regularization = learning_rate * fisher_score * free_energy * morphology_factor * weight_vector

    # Updated weights
    new_weights = weight_vector + nlms_correction - regularization
    return new_weights


def health_score(
    reliability: float,
    length: float,
    width: float,
    height: float,
    free_energy: float,
) -> float:
    """
    Endpoint health score = reliability × morphology_factor × free_energy.
    """
    if not (0.0 <= reliability <= 1.0):
        raise ValueError("reliability must be in [0, 1]")
    morph_factor = compute_morphology_factor(length, width, height)
    return reliability * morph_factor * free_energy


def select_best_endpoint(
    endpoints: List[dict],
    free_energy: float,
) -> Tuple[int, float]:
    """
    Given a list of endpoint dictionaries, each containing:
        - 'reliability' (float in [0,1])
        - 'length', 'width', 'height' (float > 0)

    Returns the index of the endpoint with the maximal health score and the
    corresponding score.
    """
    best_idx = -1
    best_score = -math.inf
    for idx, ep in enumerate(endpoints):
        score = health_score(
            reliability=ep["reliability"],
            length=ep["length"],
            width=ep["width"],
            height=ep["height"],
            free_energy=free_energy,
        )
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx, best_score


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic morphology for three endpoints
    endpoints = [
        {"reliability": 0.9, "length": 2.0, "width": 1.5, "height": 1.0},
        {"reliability": 0.7, "length": 1.0, "width": 1.0, "height": 2.0},
        {"reliability": 0.85, "length": 1.8, "width": 1.2, "height": 1.2},
    ]

    # Auxiliary data for free‑energy (random Gaussian vector)
    aux_data = np.random.randn(50)
    E = variational_free_energy(aux_data)

    # Choose best endpoint based on health score
    best_idx, best_score = select_best_endpoint(endpoints, free_energy=E)
    print(f"Best endpoint index: {best_idx}, health score: {best_score:.4e}")

    # Prepare NLMS structures
    n_features = 8
    w = np.zeros(n_features)
    x = np.random.randn(n_features)
    lr = 0.05

    # Simulated prediction/target
    pred = float(np.dot(w, x))
    target = random.uniform(-1.0, 1.0)

    # Morphology factor from the selected endpoint
    sel_ep = endpoints[best_idx]
    morph_factor = compute_morphology_factor(
        sel_ep["length"], sel_ep["width"], sel_ep["height"]
    )

    # Perform hybrid NLMS update
    w_new = nlms_update_with_regularization(
        weight_vector=w,
        feature_vector=x,
        learning_rate=lr,
        predicted_output=pred,
        actual_output=target,
        free_energy=E,
        morphology_factor=morph_factor,
    )
    print("Weight vector before update:", w)
    print("Weight vector after update :", w_new)
    # Verify that the update produced a different vector
    assert not np.allclose(w, w_new), "NLMS update failed to change weights"
    print("Smoke test completed successfully.")