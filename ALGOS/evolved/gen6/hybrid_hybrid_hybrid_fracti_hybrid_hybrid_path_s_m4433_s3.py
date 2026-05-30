# DARWIN HAMMER — match 4433, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_fractional_hd_pheromone_m2184_s0.py (gen3)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_hybrid_m572_s2.py (gen5)
# born: 2026-05-29T23:55:39Z

"""Hybrid Path-Pheromone Fractional Binding

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – Hybrid Power Binding with Pheromone‑Guided Endpoint Morphology Fusion.
  It provides complex‑valued hypervectors and a pheromone‑guided health score that
  is essentially a dot product between a fractional‑power‑bound vector and a
  pheromone‑derived geometric‑indices vector.

* **Parent B** – Hybrid Path Signature and Regret‑Weighted Probabilities.
  It supplies lead‑lag transformation, level‑1/2 path signatures, B‑spline basis
  generation and a regret‑weighted probability computation for actions.

The mathematical bridge is the *signature‑to‑hypervector binding*:
the (real‑valued) path signature is used as a fractional exponent that binds a
random complex hypervector (from Parent A).  The resulting bound hypervector is
then evaluated against a pheromone‑derived geometric vector (also from Parent A)
using a dot product.  The scalar health score modulates the regret‑weighted
probabilities defined in Parent B, yielding a single unified hybrid system.

The implementation provides three high‑level functions that demonstrate the
hybrid operation:

1. `encode_path_signature` – converts a path into a complex hypervector by
   fractional power binding of a random hypervector with the path’s level‑1 and
   level‑2 signatures.

2. `pheromone_geometric_vector` – builds a pheromone‑guided geometric indices
   vector using B‑spline basis functions.

3. `hybrid_health_score` – computes the dot product between the bound hypervector
   and the geometric vector, producing a health score that is subsequently used
   to weight regret‑adjusted action probabilities via
   `regret_weighted_probabilities`.

All code relies only on the Python standard library and NumPy."""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Core primitives from Parent A
# ---------------------------------------------------------------------------

def random_hv(d: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    """Generate a random hypervector of dimension ``d``.

    Parameters
    ----------
    d : int
        Dimension of the hypervector.
    kind : {"complex", "bipolar", "real"}
        Type of hypervector.
    seed : int | None
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Random hypervector.
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice([-1, 1], size=d).astype(np.float64)
    # real Gaussian, normalized to unit L2 norm
    vec = rng.normal(size=d)
    return vec / np.linalg.norm(vec)

def pheromone_geometric_vector(pheromone_signal: float,
                               length: int,
                               grid_size: int = 10,
                               spline_order: int = 3) -> np.ndarray:
    """Create a pheromone‑guided geometric indices vector using B‑splines.

    The vector is the weighted sum of B‑spline basis functions evaluated on a
    uniform grid and scaled by ``pheromone_signal``.

    Parameters
    ----------
    pheromone_signal : float
        Scalar pheromone intensity.
    length : int
        Desired length of the output vector.
    grid_size : int
        Number of interior knots for the spline basis.
    spline_order : int
        Order of the B‑spline (default cubic).

    Returns
    -------
    np.ndarray
        Real‑valued vector of shape ``(length,)``.
    """
    x = np.linspace(0.0, 1.0, length)
    grid = np.linspace(0.0, 1.0, grid_size)
    B = bspline_basis(x, grid, k=spline_order)          # (length, n_basis)
    coeffs = np.ones(B.shape[1])                        # uniform coefficients
    geom = B @ coeffs                                    # (length,)
    return pheromone_signal * geom

# ---------------------------------------------------------------------------
# Core primitives from Parent B
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Lead‑lag transform used in signature calculations."""
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path: np.ndarray) -> np.ndarray:
    """First‑order signature (endpoint difference)."""
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path: np.ndarray) -> np.ndarray:
    """Second‑order signature (area‑like term)."""
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running    = path[:-1] - path[0]            # (T‑1, d)
    return running.T @ increments               # (d, d)

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Cox–de Boor recursion for B‑spline basis functions."""
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    N = len(x)
    B = np.zeros((N, len(t) - 1), dtype=np.float64)

    # order 1 (piecewise constant)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    # higher orders
    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]

            term_l = ((x - t[i]) / denom_l) * B[:, i] if denom_l > 0 else np.zeros(N)
            term_r = ((t[i + order] - x) / denom_r) * B[:, i + 1] if denom_r > 0 else np.zeros(N)

            B_new[:, i] = term_l + term_r
        B = B_new

    return B

def regret_weighted_probabilities(actions: list[MathAction],
                                  health_score: float) -> np.ndarray:
    """Compute probabilities weighted by expected value, risk and health score.

    The base probability for each action is its ``expected_value``.
    Risk is penalised proportionally to the health score (higher health → less
    tolerance for risk).  The resulting vector is normalised to sum to 1.

    Parameters
    ----------
    actions : list[MathAction]
        List of candidate actions.
    health_score : float
        Scalar health score from the hybrid binding.

    Returns
    -------
    np.ndarray
        Normalised probability distribution over actions.
    """
    if not actions:
        raise ValueError("Action list must not be empty.")

    ev = np.array([a.expected_value for a in actions], dtype=np.float64)
    risk = np.array([a.risk for a in actions], dtype=np.float64)

    # Prevent division by zero; add a tiny epsilon.
    eps = 1e-12
    risk_factor = np.exp(-health_score * risk) + eps
    raw = ev * risk_factor
    prob = raw / raw.sum()
    return prob

# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------

def encode_path_signature(path: np.ndarray,
                          hv_dim: int = 4096,
                          seed: int | None = None) -> np.ndarray:
    """Bind a random complex hypervector with the path's signatures.

    The binding uses fractional exponentiation: for a unit‑magnitude complex
    hypervector ``h`` and a real scalar ``s`` the bound component is
    ``h**s = exp(i * theta * s)``.  The signature vector is normalised before
    binding to keep the bound hypervector on the unit circle.

    Parameters
    ----------
    path : np.ndarray
        Input trajectory of shape (T, d).
    hv_dim : int
        Dimension of the random hypervector (must be >= signature length).
    seed : int | None
        Seed for reproducibility of the random hypervector.

    Returns
    -------
    np.ndarray
        Complex hypervector of shape ``(hv_dim,)`` representing the bound state.
    """
    # Compute signatures
    lvl1 = signature_level1(path)                     # (d,)
    lvl2 = signature_level2(path).ravel()             # (d*d,)

    # Concatenate and normalise to unit L2 norm
    sig_vec = np.concatenate([lvl1, lvl2])
    if sig_vec.size == 0:
        raise ValueError("Path must have at least one dimension.")
    sig_norm = np.linalg.norm(sig_vec)
    sig_vec = sig_vec / (sig_norm + 1e-12)

    # Pad or truncate to match hv_dim
    if sig_vec.size < hv_dim:
        pad = np.zeros(hv_dim - sig_vec.size, dtype=np.float64)
        sig_vec = np.concatenate([sig_vec, pad])
    else:
        sig_vec = sig_vec[:hv_dim]

    # Random complex hypervector
    hv = random_hv(d=hv_dim, kind="complex", seed=seed)   # unit magnitude

    # Fractional power binding (element‑wise)
    bound = hv ** sig_vec                                   # still unit magnitude
    return bound

def hybrid_health_score(path: np.ndarray,
                        pheromone_signal: float,
                        hv_dim: int = 4096,
                        seed: int | None = None) -> float:
    """Compute the hybrid health score for a given path and pheromone level.

    The score is the real part of the dot product between the bound hypervector
    (fractional power binding of the path signature) and a pheromone‑guided
    geometric vector.

    Parameters
    ----------
    path : np.ndarray
        Trajectory to be encoded.
    pheromone_signal : float
        Scalar pheromone intensity.
    hv_dim : int
        Dimension of the hypervectors.
    seed : int | None
        Seed for reproducibility.

    Returns
    -------
    float
        Scalar health score.
    """
    bound_hv = encode_path_signature(path, hv_dim=hv_dim, seed=seed)   # (hv_dim,)
    geom_vec = pheromone_geometric_vector(pheromone_signal,
                                          length=hv_dim)          # (hv_dim,)

    # Ensure both vectors are complex for a consistent dot product
    geom_complex = geom_vec.astype(np.complex128)

    # Dot product and take real part (health is a real scalar)
    score = np.real(np.vdot(bound_hv, geom_complex))
    return float(score)

def hybrid_regret_distribution(actions: list[MathAction],
                               path: np.ndarray,
                               pheromone_signal: float,
                               hv_dim: int = 4096,
                               seed: int | None = None) -> np.ndarray:
    """Full pipeline: compute health score and return regret‑weighted probabilities.

    Parameters
    ----------
    actions : list[MathAction]
        Candidate actions.
    path : np.ndarray
        Trajectory to be analysed.
    pheromone_signal : float
        Pheromone intensity.
    hv_dim : int
        Hypervector dimensionality.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Probability distribution over ``actions``.
    """
    health = hybrid_health_score(path, pheromone_signal, hv_dim=hv_dim, seed=seed)
    probs = regret_weighted_probabilities(actions, health_score=health)
    return probs

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple synthetic path (5 timesteps, 3 dimensions)
    rng = np.random.default_rng(42)
    path = rng.normal(size=(5, 3))

    # Pheromone intensity (arbitrary positive scalar)
    pheromone = 0.73

    # Define a few actions
    actions = [
        MathAction(id="A", expected_value=10.0, risk=0.2),
        MathAction(id="B", expected_value=7.5,  risk=0.1),
        MathAction(id="C", expected_value=5.0,  risk=0.4),
    ]

    # Run the hybrid pipeline
    probs = hybrid_regret_distribution(actions, path, pheromone_signal=pheromone,
                                      hv_dim=2048, seed=123)

    # Display results
    for act, p in zip(actions, probs):
        print(f"Action {act.id}: probability = {p:.4f}")

    # Verify that probabilities sum to 1 (within tolerance)
    assert np.isclose(probs.sum(), 1.0), "Probabilities do not sum to 1."

    print("Hybrid health score and regret distribution computed successfully.")