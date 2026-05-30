# DARWIN HAMMER — match 1310, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_nlms_hybrid_h_hybrid_endpoint_circ_m715_s3.py (gen4)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s2.py (gen3)
# born: 2026-05-29T23:35:15Z

"""Hybrid algorithm merging adaptive NLMS filtering with lead‑lag path signatures and B‑spline basis expansion.

Parents:
- hybrid_hybrid_nlms_hybrid_h_hybrid_endpoint_circ_m715_s3.py (adaptive filter with circuit‑breaker)
- hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s2.py (lead‑lag transform, B‑spline basis, entropy)

Mathematical bridge:
The lead‑lag transformed path provides a richer feature vector x(t).  Each feature column is projected onto a set of B‑spline basis functions defined on a common knot grid, yielding a design matrix B(t).  The NLMS weight‑update of the original filter is applied to the spline coefficients (the “weights”) using the same matrix‑form update rule.  Entropy of the raw path modulates the learning‑rate, while a circuit‑breaker monitors the prediction error, preserving the safety mechanism of the first parent.
"""

import numpy as np
import math
import random
import sys
import pathlib
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Tuple

# ----------------------------------------------------------------------
# Simple data holder – retained from parent A for possible future use
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

    def __post_init__(self) -> None:
        for name, value in (
            ("length", self.length),
            ("width", self.width),
            ("height", self.height),
            ("mass", self.mass),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

# ----------------------------------------------------------------------
# Circuit breaker – identical semantics to parent A
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def allow(self) -> bool:
        return not self.open

# ----------------------------------------------------------------------
# Parent B – lead‑lag transform and B‑spline basis
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Convert a sequence of d‑dimensional points into its lead‑lag representation."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (T, d)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Cox–de Boor recursion for B‑spline basis functions of order k.
    Returns a matrix B of shape (len(x), n_basis) where each column
    corresponds to a basis function evaluated at all points in x.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Knot vector with clamped ends
    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1])
    ])

    n_basis = len(grid) + k - 2          # number of basis functions
    N = len(x)

    # Zeroth‑order (piecewise constant) basis
    B = np.zeros((N, n_basis), dtype=np.float64)
    for i in range(n_basis):
        left = t[i]
        right = t[i + 1]
        B[:, i] = np.where((x >= left) & (x < right), 1.0, 0.0)
    # Include the rightmost knot
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    # Recursion for higher orders
    for order in range(2, k + 1):
        n_new = n_basis - order + 1
        B_new = np.zeros((N, n_new), dtype=np.float64)
        for i in range(n_new):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]

            term_l = ((x - t[i]) / denom_l) * B[:, i] if denom_l > 0 else np.zeros(N)
            term_r = ((t[i + order] - x) / denom_r) * B[:, i + 1] if denom_r > 0 else np.zeros(N)

            B_new[:, i] = term_l + term_r
        B = B_new
        n_basis = n_new
    return B

# ----------------------------------------------------------------------
# Entropy – simple Shannon entropy of the raw path (parent B concept)
# ----------------------------------------------------------------------
def compute_entropy(path: np.ndarray, bins: int = 10) -> float:
    """Shannon entropy of the flattened path histogram."""
    flat = path.ravel()
    hist, _ = np.histogram(flat, bins=bins, density=True)
    hist = hist[hist > 0]                     # avoid log(0)
    return -np.sum(hist * np.log(hist))

# ----------------------------------------------------------------------
# Hybrid predictor – NLMS on spline coefficients
# ----------------------------------------------------------------------
def hybrid_predict(coeffs: np.ndarray,
                   path: np.ndarray,
                   grid: np.ndarray,
                   k: int = 3) -> np.ndarray:
    """
    Predict output for each time step using spline‑expanded lead‑lag features.
    coeffs shape: (total_basis_dim,)
    Returns a vector y of length N (number of lead‑lag points).
    """
    ll = lead_lag_transform(path)                     # (N, 2d)
    N, dim = ll.shape
    # Build design matrix by concatenating basis expansions of each column
    B_parts = [bspline_basis(ll[:, j], grid, k) for j in range(dim)]
    B = np.hstack(B_parts)                            # (N, dim * n_basis)
    if coeffs.shape[0] != B.shape[1]:
        raise ValueError("Coefficient size does not match design matrix columns")
    return B @ coeffs                                 # (N,)

def hybrid_update_weights(coeffs: np.ndarray,
                          path: np.ndarray,
                          target: np.ndarray,
                          grid: np.ndarray,
                          mu: float = 0.5,
                          eps: float = 1e-9,
                          k: int = 3,
                          entropy_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    NLMS‑style update of spline coefficients.
    - coeffs : current coefficient vector
    - path   : raw path (T, d)
    - target : desired output vector of length N (same as lead‑lag length)
    - grid   : knot locations for B‑splines
    Returns (new_coeffs, error_vector).
    """
    ll = lead_lag_transform(path)
    N, dim = ll.shape
    B_parts = [bspline_basis(ll[:, j], grid, k) for j in range(dim)]
    B = np.hstack(B_parts)                     # (N, P)

    y = B @ coeffs                             # prediction
    error = target - y                         # (N,)

    # Power term per sample (scalar for each row)
    power = np.sum(B ** 2, axis=1, keepdims=True) + eps   # (N,1)

    # NLMS update: Δw = μ * (Bᵀ (error / power)) / N
    scaled_error = (error[:, np.newaxis] / power)        # (N,1)
    delta = mu * (B.T @ scaled_error).ravel() / N         # (P,)

    # Entropy modulates the step size
    new_coeffs = coeffs + entropy_factor * delta
    return new_coeffs, error

def circuit_breaker_update(cb: EndpointCircuitBreaker,
                           coeffs: np.ndarray,
                           path: np.ndarray,
                           target: np.ndarray,
                           grid: np.ndarray,
                           mu: float = 0.5,
                           eps: float = 1e-9,
                           k: int = 3,
                           error_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a hybrid weight update only if the circuit breaker permits it.
    Records a failure when the maximum absolute error exceeds error_threshold.
    Returns updated coefficients and the error vector.
    """
    if not cb.allow():
        return coeffs, np.zeros_like(target)

    # Entropy of the raw path influences learning rate
    ent = compute_entropy(path)
    entropy_factor = 1.0 + 0.5 * math.tanh(ent)   # bounded scaling

    new_coeffs, error = hybrid_update_weights(
        coeffs, path, target, grid,
        mu=mu, eps=eps, k=k, entropy_factor=entropy_factor
    )

    if np.max(np.abs(error)) > error_threshold:
        cb.record_failure()
    else:
        cb.record_success()

    return new_coeffs, error

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic 2‑D path
    T, d = 20, 2
    np.random.seed(42)
    path = np.cumsum(np.random.randn(T, d), axis=0)   # random walk

    # Lead‑lag length
    N = 2 * T - 1

    # Target: simple linear function of time for testing
    target = np.linspace(0, 1, N)

    # B‑spline grid (uniform)
    grid = np.linspace(np.min(path), np.max(path), num=5)

    # Initialise coefficients (zero)
    ll = lead_lag_transform(path)
    dim = ll.shape[1]
    n_basis = len(grid) + 3 - 2          # k=3 → cubic
    coeffs = np.zeros(dim * n_basis)

    # Circuit breaker
    cb = EndpointCircuitBreaker(failure_threshold=3)

    # Single update step
    coeffs, err = circuit_breaker_update(
        cb, coeffs, path, target, grid,
        mu=0.8, eps=1e-9, k=3, error_threshold=0.4
    )

    # Print a brief summary
    print(f"Error norm after update: {np.linalg.norm(err):.4f}")
    print(f"Circuit breaker state – open: {cb.open}, failures: {cb.failures}")
    print(f"Coefficients sample: {coeffs[:5]}")