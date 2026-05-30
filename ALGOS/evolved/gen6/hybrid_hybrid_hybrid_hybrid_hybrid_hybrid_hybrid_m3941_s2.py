# DARWIN HAMMER — match 3941, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hoeffd_hybrid_hybrid_rbf_su_m732_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2587_s0.py (gen5)
# born: 2026-05-29T23:52:41Z

"""Hybrid algorithm integrating tropical algebra, Fisher information, and RBF surrogates.

Parents:
- **Parent A** (`hybrid_hybrid_hybrid_hoeffd_hybrid_hybrid_rbf_su_m732_s1.py`) provides tropical (max‑plus) algebra utilities and a Gaussian RBF kernel.
- **Parent B** (`hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2587_s0.py`) supplies regex‑based feature extraction, a Gaussian‑beam based Fisher information estimator, and RBF surrogate logic.

Mathematical bridge:
The Fisher information scalar derived from the Gaussian‑beam model is used as a *global scaling factor* for the coefficients of a tropical polynomial.  The feature vector extracted via regex is turned into a tropical matrix **A** (features placed on the diagonal, –∞ elsewhere).  An RBF kernel matrix **B** (distances between a query point and a set of centres) is built with the same Gaussian kernel as Parent A.  The core hybrid operation is a tropical matrix product


C = A ⊗ B   (max‑plus multiplication)


followed by a tropical polynomial evaluation whose coefficients are multiplied by the Fisher‑information scale.  The resulting scalar is the hybrid model output.

The module below implements this pipeline with three public functions demonstrating the fusion.
"""

import math
import random
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Tropical (max‑plus) algebra utilities (from Parent A)
# ----------------------------------------------------------------------
def t_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical addition (max)."""
    return np.maximum(x, y)


def t_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical multiplication (ordinary addition)."""
    return np.add(x, y)


def t_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical matrix multiplication:
        (A ⊗ B)[i, j] = max_k (A[i, k] + B[k, j])
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape[1] != B.shape[0]:
        raise ValueError("inner dimensions must agree for tropical matmul")
    # Broadcast to compute all pairwise sums, then max over k
    sums = A[:, :, np.newaxis] + B[np.newaxis, :, :]
    return np.max(sums, axis=1)


def t_polyval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Evaluate a tropical (max‑plus) polynomial
        p(x) = max_i (c_i + i * x)
    where i is the exponent (0‑based).
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)

    exponents = np.arange(coeffs.size, dtype=float).reshape((-1,) + (1,) * x.ndim)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents * x
    return np.max(terms, axis=0)


# ----------------------------------------------------------------------
# RBF surrogate utilities (shared)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Euclidean distance."""
    if a.shape != b.shape:
        raise ValueError("vectors must have the same shape")
    return float(np.linalg.norm(a - b))


def rbf_kernel(x: np.ndarray, c: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian RBF kernel evaluate for a set of centres."""
    d = np.linalg.norm(x - c, axis=1)
    return np.exp(- (d ** 2) / (2 * sigma ** 2))


# ----------------------------------------------------------------------
# Regex‑based feature extraction (from Parent B)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)


def extract_features(text: str) -> np.ndarray:
    """
    Extract a 2‑dimensional feature vector from *text*.
    Feature 0 = count of evidence‑type tokens,
    Feature 1 = count of planning‑type tokens.
    """
    evidence = len(EVIDENCE_RE.findall(text))
    planning = len(PLANNING_RE.findall(text))
    return np.array([evidence, planning], dtype=float)


# ----------------------------------------------------------------------
# Fisher information via Gaussian‑beam model (from Parent B)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_information(theta: float, center: float, width: float) -> float:
    """
    Fisher information for the Gaussian‑beam model.
    I(θ) = (∂/∂θ log I(θ))² = ((θ‑center) / width²)²
    """
    if width <= 0.0:
        raise ValueError("width must be positive")
    diff = theta - center
    return (diff / (width ** 2)) ** 2


# ----------------------------------------------------------------------
# Hybrid core operations
# ----------------------------------------------------------------------
def tropical_rbf_predict(
    x: np.ndarray,
    tropical_coeffs: np.ndarray,
    rbf_centers: np.ndarray,
    sigma: float,
    fisher_scale: float,
) -> float:
    """
    Hybrid prediction:

    1. Scale tropical coefficients by the Fisher information factor.
    2. Evaluate the tropical polynomial at the scalar ``x_scalar = ||x||``.
    3. Compute an RBF surrogate value using ``x`` and ``rbf_centers``.
    4. Combine the two results multiplicatively (the tropical part shapes the
       amplitude, the RBF part provides smooth interpolation).

    Parameters
    ----------
    x : np.ndarray
        Input vector.
    tropical_coeffs : np.ndarray
        Coefficients of the tropical polynomial.
    rbf_centers : np.ndarray
        Shape (m, d) – centre points of the RBF surrogate.
    sigma : float
        Width of the Gaussian RBF.
    fisher_scale : float
        Global scaling factor from Fisher information.

    Returns
    -------
    float
        Hybrid model output.
    """
    # 1. Fisher‑scaled coefficients
    scaled_coeffs = tropical_coeffs * fisher_scale

    # 2. Tropical polynomial evaluated at the norm of x
    x_norm = np.linalg.norm(x)
    tropical_val = float(t_polyval(scaled_coeffs, x_norm))

    # 3. RBF surrogate (average over centres)
    rbf_vals = rbf_kernel(x.reshape(1, -1), rbf_centers, sigma)  # shape (1, m)
    rbf_val = float(np.mean(rbf_vals))

    # 4. Combine
    return tropical_val * rbf_val


def tropical_feature_matrix(features: np.ndarray) -> np.ndarray:
    """
    Build a tropical matrix ``A`` from a feature vector.
    The diagonal holds the feature values, off‑diagonal entries are –∞
    (represented by a very large negative number for numerical stability).

    Parameters
    ----------
    features : np.ndarray
        Shape (n,)

    Returns
    -------
    np.ndarray
        Shape (n, n) tropical matrix.
    """
    n = features.size
    neg_inf = -1e12
    A = np.full((n, n), neg_inf, dtype=float)
    np.fill_diagonal(A, features)
    return A


def tropical_rbf_update(
    features: np.ndarray,
    query: np.ndarray,
    rbf_centers: np.ndarray,
    sigma: float,
    theta: float,
    center: float,
    width: float,
) -> np.ndarray:
    """
    Perform a full hybrid update:

    * Construct tropical matrix ``A`` from *features*.
    * Build an RBF kernel matrix ``B`` where each entry B[i, j] = exp(-||c_i - q_j||²/(2σ²)).
      Here ``c_i`` are the RBF centres and ``q_j`` is the query (replicated).
    * Compute tropical product C = A ⊗ B.
    * Scale C by the Fisher information factor.

    Parameters
    ----------
    features : np.ndarray
        Feature vector (size n).
    query : np.ndarray
        Query vector (dimension d).
    rbf_centers : np.ndarray
        Shape (m, d).
    sigma : float
        RBF width.
    theta, center, width : float
        Parameters for Fisher information.

    Returns
    -------
    np.ndarray
        Scaled tropical product matrix C (shape n × m).
    """
    A = tropical_feature_matrix(features)               # (n, n)
    # Build B of shape (n, m) – we replicate the query n times
    queries = np.tile(query, (features.size, 1))        # (n, d)
    B = rbf_kernel(queries, rbf_centers, sigma)         # (n, m)

    C = t_matmul(A, B)                                  # (n, m)

    fisher_scale = fisher_information(theta, center, width)
    return C * fisher_scale


# ----------------------------------------------------------------------
# Simple dataclass to hold model hyper‑parameters
# ----------------------------------------------------------------------
@dataclass
class HybridModel:
    tropical_coeffs: np.ndarray
    rbf_centers: np.ndarray
    sigma: float
    fisher_center: float
    fisher_width: float

    def predict(self, x: np.ndarray, theta: float) -> float:
        """
        Wrapper around :func:`tropical_rbf_predict` using the model's stored
        parameters.
        """
        fisher_scale = fisher_information(theta, self.fisher_center, self.fisher_width)
        return tropical_rbf_predict(
            x,
            self.tropical_coeffs,
            self.rbf_centers,
            self.sigma,
            fisher_scale,
        )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample text for feature extraction
    sample_text = (
        "The plan was verified with source logs and a final checklist. "
        "Evidence was captured in the screenshot and audit record."
    )
    feats = extract_features(sample_text)  # e.g., [evidence, planning]

    # Random seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Define RBF centres (5 centres in 3‑D space)
    centres = np.random.randn(5, 3)

    # Build a hybrid model
    model = HybridModel(
        tropical_coeffs=np.array([0.5, -0.2, 0.3, 0.0]),  # 4‑degree tropical poly
        rbf_centers=centres,
        sigma=1.0,
        fisher_center=0.0,
        fisher_width=1.5,
    )

    # Query vector
    x_query = np.array([0.7, -1.2, 0.3])

    # Predict with a chosen theta
    theta_val = 0.4
    pred = model.predict(x_query, theta_val)
    print(f"Hybrid prediction: {pred:.6f}")

    # Demonstrate the full matrix update
    C = tropical_rbf_update(
        features=feats,
        query=x_query,
        rbf_centers=centres,
        sigma=1.0,
        theta=theta_val,
        center=model.fisher_center,
        width=model.fisher_width,
    )
    print("Scaled tropical‑RBF matrix C:")
    print(C)