# DARWIN HAMMER — match 3941, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hoeffd_hybrid_hybrid_rbf_su_m732_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2587_s0.py (gen5)
# born: 2026-05-29T23:52:41Z

"""Hybrid Algorithm Fusion of:
- Parent A: tropical algebra utilities + Gaussian RBF surrogate (hybrid_hybrid_hybrid_hoeffd_hybrid_hybrid_rbf_su_m732_s1.py)
- Parent B: regex feature extraction, Gaussian beam, Fisher‑information weighting (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2587_s0.py)

Mathematical Bridge
-------------------
The bridge is built on two observations:

1. **Fisher information as a scalar weight** – the Fisher score computed from
   a Gaussian‑beam likelihood (Parent B) is used to *scale* the coefficients of
   a tropical polynomial (Parent A).  This lets the information‑theoretic
   landscape modulate the max‑plus dynamics.

2. **Regex‑derived feature vectors as RBF centres** – binary feature vectors
   obtained from the evidence‑/planning regexes (Parent B) serve as centre
   points for a Gaussian RBF surrogate (Parent A).  The RBF kernel value is
   combined with the tropical matrix product to produce the next system state.

The resulting hybrid step therefore consists of:

features  = extract_features(text)                     # B
f_weight   = fisher_weight(features, theta)            # B
t_coeffs   = t_weighted_coeffs(base_coeffs, f_weight)  # A × bridge
poly_val   = t_polyval(t_coeffs, state)                # A
rbf_val    = rbf_kernel(state, feature_centers, sigma) # A
new_state  = t_matmul(poly_val[:, None], rbf_val[None, :])

All operations stay within NumPy and the standard library, satisfying the
fusion requirements."""


import math
import random
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Tropical (max‑plus) algebra utilities (Parent A)
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
# Gaussian RBF utilities (Parent A)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Euclidean distance."""
    if a.shape != b.shape:
        raise ValueError("vectors must have the same shape")
    return float(np.linalg.norm(a - b))


def rbf_kernel(x: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian RBF kernel evaluated between a single vector ``x`` and an
    array of centre vectors ``centers`` (shape: (n_centers, dim)).
    Returns a vector of length ``n_centers``.
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1‑D feature vector")
    if centers.ndim != 2:
        raise ValueError("centers must be a 2‑D array")
    dists = np.linalg.norm(centers - x, axis=1)
    return np.exp(- (dists ** 2) / (2 * sigma ** 2))


# ----------------------------------------------------------------------
# Regex feature extraction (Parent B)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)


def extract_features(text: str) -> np.ndarray:
    """
    Convert a piece of text into a binary feature vector of length 2:
    [evidence_match, planning_match].
    """
    ev = bool(EVIDENCE_RE.search(text))
    pl = bool(PLANNING_RE.search(text))
    return np.array([float(ev), float(pl)], dtype=float)


# ----------------------------------------------------------------------
# Fisher information utilities (Parent B)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-8) -> float:
    """
    Approximate the Fisher information for a Gaussian beam likelihood.
    For a Gaussian N(center, width^2) the Fisher information w.r.t. the mean
    is 1/width^2.  We return a scaled version that also depends on the
    distance between theta and centre to keep the value data‑dependent.
    """
    if width <= 0.0:
        raise ValueError("width must be positive")
    # Base Fisher information for the mean parameter
    base = 1.0 / (width ** 2)
    # Modulate by the likelihood value to embed data dependence
    likelihood = gaussian_beam(theta, center, width)
    return base * likelihood + eps


def compute_fisher_weight(features: np.ndarray, theta: float) -> float:
    """
    Map a binary feature vector to a Fisher weight.
    Each active feature contributes a separate Gaussian beam; the total
    weight is the sum of their Fisher scores.
    """
    weight = 0.0
    # Assign arbitrary centres and widths for the two features
    centres = [0.0, 1.0]          # evidence → centre 0, planning → centre 1
    widths = [0.5, 0.3]           # spread of each beam
    for i, active in enumerate(features):
        if active:
            weight += fisher_score(theta, centres[i], widths[i])
    # Guard against zero weight
    return max(weight, 1e-6)


# ----------------------------------------------------------------------
# Hybrid core functions (fusion)
# ----------------------------------------------------------------------
def t_weighted_coeffs(base_coeffs: np.ndarray, fisher_weight: float) -> np.ndarray:
    """
    Scale tropical polynomial coefficients by a Fisher weight.
    The scaling is performed in the tropical sense:
        c_i' = c_i + log(fisher_weight)
    (adding in the ordinary sense corresponds to tropical multiplication).
    """
    if fisher_weight <= 0.0:
        raise ValueError("fisher_weight must be positive")
    log_w = math.log(fisher_weight)
    return base_coeffs + log_w


def hybrid_state_update(
    state: np.ndarray,
    text: str,
    base_coeffs: np.ndarray,
    rbf_centers: np.ndarray,
    sigma: float,
    theta: float,
) -> np.ndarray:
    """
    Perform one hybrid update step.

    1. Extract regex‑based binary features from ``text``.
    2. Compute a Fisher information weight from the features and ``theta``.
    3. Modulate tropical polynomial coefficients with the Fisher weight.
    4. Evaluate the tropical polynomial at the current ``state``.
    5. Evaluate the Gaussian RBF kernel between ``state`` and the feature centres.
    6. Combine the two results with tropical matrix multiplication to obtain the
       next state vector.

    Parameters
    ----------
    state : np.ndarray
        1‑D vector representing the current system state.
    text : str
        Input text used for feature extraction.
    base_coeffs : np.ndarray
        Coefficients of the underlying tropical polynomial (length = degree+1).
    rbf_centers : np.ndarray
        2‑D array of shape (n_centers, state_dim) used as RBF centres.
    sigma : float
        Width parameter of the Gaussian RBF.
    theta : float
        Parameter governing the Gaussian beam for Fisher information.

    Returns
    -------
    np.ndarray
        Updated state vector (1‑D).
    """
    # 1. Feature extraction
    feats = extract_features(text)

    # 2. Fisher weight
    f_weight = compute_fisher_weight(feats, theta)

    # 3. Tropical coefficient scaling
    coeffs = t_weighted_coeffs(base_coeffs, f_weight)

    # 4. Tropical polynomial evaluation (result is a scalar)
    poly_val = t_polyval(coeffs, state)          # shape: (state_dim,)

    # 5. RBF kernel evaluation (vector of length n_centers)
    rbf_vals = rbf_kernel(state, rbf_centers, sigma)   # shape: (n_centers,)

    # 6. Fuse via tropical matrix multiplication
    #    We treat ``poly_val`` as a column vector and ``rbf_vals`` as a row vector.
    A = poly_val[:, np.newaxis]   # shape (d,1)
    B = rbf_vals[np.newaxis, :]   # shape (1,n_centers)
    fused = t_matmul(A, B)        # shape (d, n_centers)

    # Collapse the centre dimension by tropical addition (max) to obtain a
    # single updated state vector of length ``d``.
    new_state = np.max(fused, axis=1)
    return new_state


def hybrid_process(
    initial_state: np.ndarray,
    texts: List[str],
    base_coeffs: np.ndarray,
    rbf_centers: np.ndarray,
    sigma: float = 1.0,
    theta_start: float = 0.0,
    theta_step: float = 0.1,
) -> Tuple[np.ndarray, List[float]]:
    """
    Run a sequence of hybrid updates over a list of textual inputs.

    Returns the final state and the list of Fisher weights encountered.
    """
    state = initial_state.copy()
    fisher_weights: List[float] = []
    theta = theta_start

    for txt in texts:
        feats = extract_features(txt)
        w = compute_fisher_weight(feats, theta)
        fisher_weights.append(w)

        state = hybrid_state_update(
            state,
            txt,
            base_coeffs,
            rbf_centers,
            sigma,
            theta,
        )
        theta += theta_step  # slowly move the beam centre

    return state, fisher_weights


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic example
    dim = 3
    init_state = np.zeros(dim)

    # Base tropical polynomial coefficients (degree = 2)
    base_coeffs = np.array([0.0, 1.0, -0.5])

    # Create a few RBF centres (here we reuse the binary feature space expanded)
    rbf_centers = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.2, 0.8],
    ])

    texts = [
        "The evidence was verified and the source is documented.",
        "We need a checklist and a timeline for the upcoming test.",
        "No relevant keywords here.",
        "Please confirm the audit log and plan the next phase.",
    ]

    final_state, weights = hybrid_process(
        initial_state=init_state,
        texts=texts,
        base_coeffs=base_coeffs,
        rbf_centers=rbf_centers,
        sigma=0.7,
        theta_start=0.2,
        theta_step=0.05,
    )

    print("Final state:", final_state)
    print("Fisher weights per step:", weights)