# DARWIN HAMMER — match 4396, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_semantic_neig_hybrid_caputo_fracti_m258_s2.py (gen3)
# parent_b: hybrid_tropical_maxplus_hybrid_hybrid_minimu_m190_s2.py (gen3)
# born: 2026-05-29T23:55:28Z

"""Hybrid algorithm combining semantic recovery priority (Parent A) with tropical max‑plus algebra (Parent B).

Mathematical bridge:
- Each node’s *recovery priority* (a scalar in [0,1]) is treated as a tropical coefficient.
- As edges are inserted chronologically, the priorities are combined with the tropical
  max‑plus semiring, yielding a tropical polynomial whose value at a depth *x* encodes
  the best (maximum) recovery potential along any path of that length.
- The sequence of polynomial maxima is then processed with a discrete Caputo fractional
  integral, providing a memory‑aware “fractional‑memory recovery surface” that respects
  the entire construction history with algebraic decay.

The module therefore fuses:
1. Morphology‑based recovery priority (Parent A).
2. Tropical max‑plus operations and polynomial evaluation (Parent B).
3. Caputo fractional integration to remember the temporal evolution of the system.
"""

import numpy as np
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, sqrt, gamma
from random import random
import sys
from pathlib import Path
from collections import Counter

# ----------------------------------------------------------------------
# Parent A – morphology & recovery priority
# ----------------------------------------------------------------------
def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

# ----------------------------------------------------------------------
# Parent B – tropical max‑plus algebra
# ----------------------------------------------------------------------
def t_add(x, y):
    """Tropical addition (max). Works with numpy broadcasting."""
    return np.maximum(x, y)

def t_mul(x, y):
    """Tropical multiplication (ordinary addition). Works with numpy broadcasting."""
    return np.add(x, y)

def t_matmul(A, B):
    """
    Tropical matrix multiplication.
    C[i, j] = max_k ( A[i, k] + B[k, j] )
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # Expand dimensions to broadcast addition over k
    # Result shape: (A.shape[0], B.shape[1])
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

def t_polyval(coeffs, x):
    """
    Evaluate a tropical polynomial:
        p(x) = max_i ( coeffs[i] + i * x )
    coeffs – 1‑D array of tropical coefficients (use -np.inf for missing terms).
    x      – scalar or array broadcastable with coeffs.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(coeffs.shape[0], dtype=float).reshape((-1,) + (1,) * x.ndim)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents * x
    return np.max(terms, axis=0)

# ----------------------------------------------------------------------
# New component – discrete Caputo fractional integral
# ----------------------------------------------------------------------
def caputo_fractional_integral(values: np.ndarray, alpha: float, dt: float = 1.0) -> np.ndarray:
    """
    Simple Grünwald‑Letnikov‑type approximation of the Caputo fractional integral
    of order `alpha` (0 < alpha < 1) for a uniformly spaced sequence `values`.

    I^α f(t_n) ≈ (dt^α / Γ(α+1)) * Σ_{k=0}^{n} w_{n-k} f(t_k)
    with w_j = (j+1)^{α} - j^{α}
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    values = np.asarray(values, dtype=float)
    n = values.size
    coeff = (dt ** alpha) / gamma(alpha + 1.0)
    # pre‑compute weights w_j
    j = np.arange(n, dtype=float)
    w = (j + 1) ** alpha - j ** alpha
    # convolution (causal)
    integral = np.empty_like(values)
    for idx in range(n):
        integral[idx] = coeff * np.dot(w[:idx + 1][::-1], values[:idx + 1])
    return integral

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def compute_priority_sequence(tree_edges, morphologies):
    """
    Given a list of edges ordered by insertion time and a mapping node→Morphology,
    return the sequence of recovery priorities for the child nodes.
    Each edge is a tuple (parent, child, timestamp). The timestamp is ignored after
    sorting; only the order matters.
    """
    # Sort chronologically
    edges = sorted(tree_edges, key=lambda e: e[2])
    priorities = []
    for _, child, _ in edges:
        if child not in morphologies:
            raise KeyError(f"Missing morphology for node {child}")
        priorities.append(recovery_priority(morphologies[child]))
    return np.array(priorities, dtype=float)

def tropical_recovery_polynomial(priorities):
    """
    Build tropical coefficients from a list of priorities.
    The i‑th coefficient (degree i) is the maximum priority observed up to step i.
    This encodes the best recovery achievable with a path of length i.
    """
    coeffs = np.full(len(priorities), -np.inf, dtype=float)
    current_max = -np.inf
    for i, p in enumerate(priorities):
        current_max = max(current_max, p)
        coeffs[i] = current_max
    return coeffs

def hybrid_fractional_tropical_recovery(tree_edges, morphologies, alpha=0.5, dt=1.0):
    """
    Core hybrid routine:
    1. Derive the chronological priority sequence (Parent A).
    2. At each step, construct a tropical polynomial from the cumulative priorities
       (Parent B) and record its value at x = current depth.
    3. Apply a Caputo fractional integral to the recorded maxima, yielding a
       fractional‑memory recovery surface.
    Returns the final integral array.
    """
    priorities = compute_priority_sequence(tree_edges, morphologies)
    maxima = []
    for depth in range(1, len(priorities) + 1):
        coeffs = tropical_recovery_polynomial(priorities[:depth])
        # Evaluate polynomial at x = depth (any positive x works; depth reflects path length)
        val = t_polyval(coeffs, depth)
        maxima.append(val)
    maxima = np.array(maxima, dtype=float)
    frac_integral = caputo_fractional_integral(maxima, alpha, dt)
    return frac_integral

def decision_hygiene_score(poly_coeffs, x_grid):
    """
    Compute a decision‑hygiene score based on the entropy of tropical polynomial
    values over a grid of x values.
    Steps:
    1. Evaluate the tropical polynomial on the grid.
    2. Shift to non‑negative values and normalize to a probability distribution.
    3. Return Shannon entropy (higher entropy → better hygiene).
    """
    vals = t_polyval(poly_coeffs, x_grid)
    # Shift to make all entries non‑negative
    shift = -np.min(vals)
    probs = (vals + shift) + 1e-12  # avoid zeros
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log(probs))
    return entropy

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic tree with 5 nodes
    # Edge format: (parent, child, timestamp)
    edges = [
        (0, 1, 0.0),
        (1, 2, 1.0),
        (1, 3, 2.0),
        (3, 4, 3.0)
    ]

    # Random morphologies for each node (except root 0)
    morphologies = {
        1: Morphology(length=1.2, width=0.8, height=0.5, mass=2.0),
        2: Morphology(length=0.9, width=0.7, height=0.4, mass=1.5),
        3: Morphology(length=1.1, width=0.9, height=0.6, mass=2.2),
        4: Morphology(length=0.8, width=0.6, height=0.3, mass=1.0)
    }

    # Run hybrid fractional tropical recovery
    result = hybrid_fractional_tropical_recovery(edges, morphologies, alpha=0.6, dt=1.0)
    print("Fractional‑memory recovery surface:", result)

    # Build final tropical coefficients from full priority list
    full_priorities = compute_priority_sequence(edges, morphologies)
    final_coeffs = tropical_recovery_polynomial(full_priorities)

    # Decision hygiene over a range of depths
    x_vals = np.linspace(1, len(final_coeffs), 100)
    hygiene = decision_hygiene_score(final_coeffs, x_vals)
    print("Decision hygiene entropy:", hygiene)