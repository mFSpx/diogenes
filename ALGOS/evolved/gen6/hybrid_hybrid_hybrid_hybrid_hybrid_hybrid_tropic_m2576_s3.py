# DARWIN HAMMER — match 2576, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_cockpi_hybrid_hybrid_fisher_m583_s0.py (gen3)
# parent_b: hybrid_hybrid_tropical_maxp_hybrid_hybrid_geomet_m927_s0.py (gen5)
# born: 2026-05-29T23:43:05Z

"""
Hybrid Algorithm: hybrid_tropical_fisher_trust_flow.py

This module fuses the core concepts of two parent algorithms:

- **Parent A** (hybrid_hybrid_hybrid_cockpi_hybrid_hybrid_fisher_m583_s0):
  Provides *trust‑weighted velocity* derived from cockpit honesty metrics,
  a *Fisher score* computed from a Gaussian beam, and a JEPA‑style energy
  term based on a predictor‑encoder residual.

- **Parent B** (hybrid_hybrid_tropical_maxp_hybrid_hybrid_geomet_m927_s0):
  Supplies the tropical (max‑plus) algebraic operations, tropical polynomial
  evaluation, and a Clifford‑geometric distance metric with Voronoi partitioning.

**Mathematical Bridge**

Both parents operate on notions of *distance* and *scalar weighting*:

1. **Distance Fusion** – Euclidean distance (‖x‑y‖) from Parent A and
   Clifford‑geometric distance from Parent B are combined via the tropical
   addition (`max`). This yields a *tropical‑metric distance*  
   `d_trop(x, y) = max(‖x‑y‖, d_clifford(x, y))`.

2. **Trust as Tropical Scale** – The trust scalar ∈[0,1] from Parent A is
   interpreted as a tropical multiplier. In the max‑plus semiring,
   multiplication corresponds to ordinary addition, so scaling a value `v`
   by trust becomes `v + trust`. This enables a *trust‑weighted tropical
   velocity*.

3. **Fisher‑Driven Tropical Polynomial** – The Fisher score (a scalar
   curvature of the Gaussian beam) is injected as a coefficient into a
   tropical polynomial. The polynomial’s value at a candidate timestamp
   acts as a latent “energy potential” that feeds the JEPA residual.

The three functions below demonstrate this unified system:
`hybrid_trust_tropical_velocity`, `hybrid_tropical_metric_distance`,
and `hybrid_fisher_tropical_flow_energy`. They intertwine the governing
equations of both parents rather than merely concatenating code.

Author: Computational Physicist & AI Architect
Date: 2026‑05‑29
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

# ---------- Parent A utilities ----------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def trust_weighted_velocity(x0: float, x1: float, trust: float) -> float:
    """Linear trust‑weighted velocity (Parent A)."""
    return trust * (x1 - x0)

def jeap_energy(candidate: float, prev_candidate: float, fisher: float) -> float:
    """Simple JEPA‑style energy using a predictor that adds the Fisher score."""
    predictor = np.array([prev_candidate + fisher])
    encoder = np.array([candidate])
    return np.sum((encoder - predictor) ** 2)

# ---------- Parent B utilities ----------
def t_add(x, y):
    """Tropical addition (max). Works with scalars or NumPy arrays."""
    return np.maximum(x, y)

def t_mul(x, y):
    """Tropical multiplication (ordinary addition). Works with scalars or arrays."""
    return np.add(x, y)

def t_matmul(A, B):
    """Tropical matrix multiplication: C[i,j] = max_k (A[i,k] + B[k,j])."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # broadcasting to compute A[i,k] + B[k,j] for all i,k,j
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

def t_polyval(coeffs, x):
    """
    Evaluate a tropical polynomial:
        p(x) = max_i ( coeffs[i] + i * x )
    coeffs: 1‑D array of length d+1 (coefficients may be -inf).
    x: scalar or array broadcastable to coeffs.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(len(coeffs), dtype=float)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents.reshape((-1,) + (1,) * x.ndim) * x
    return np.max(terms, axis=0)

def clifford_geometric_distance(a, b):
    """Euclidean distance interpreted as a Clifford‑geometric metric."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return math.sqrt(np.sum((a - b) ** 2))

# ---------- Hybrid Functions ----------
def hybrid_trust_tropical_velocity(x0, x1, trust):
    """
    Compute a velocity that is both trust‑scaled (linear) and tropical‑scaled.
    Steps:
        1. Linear velocity v = x1 - x0.
        2. Apply trust weighting (Parent A): v_trust = trust * v.
        3. Tropical scaling: in the max‑plus semiring, multiplication = addition,
           so the tropical‑scaled velocity is v_trop = v_trust + trust.
        4. Return the tropical addition of the two components to keep the max‑plus
           semantics: max(v_trust, v_trop).
    Works element‑wise for NumPy arrays.
    """
    v = np.asarray(x1, dtype=float) - np.asarray(x0, dtype=float)
    v_trust = trust * v
    v_trop = t_mul(v, trust)          # tropical multiplication = ordinary addition
    return t_add(v_trust, v_trop)     # tropical addition = max

def hybrid_tropical_metric_distance(p, q, trust=1.0):
    """
    Fuse Euclidean distance (Parent A) and Clifford‑geometric distance (Parent B)
    using tropical addition, then modulate by trust via tropical multiplication.
    Formula:
        d_euc   = ‖p - q‖_2
        d_cliff = clifford_geometric_distance(p, q)
        d_trop  = max(d_euc, d_cliff)          # tropical addition
        d_final = d_trop + trust               # tropical multiplication (add trust)
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    d_euc = np.linalg.norm(p - q)
    d_cliff = clifford_geometric_distance(p, q)
    d_trop = t_add(d_euc, d_cliff)
    d_final = t_mul(d_trop, trust)
    return d_final

def hybrid_fisher_tropical_flow_energy(candidate, prev_candidate, theta, trust=1.0):
    """
    Combine the Fisher score (Parent A) with a tropical polynomial (Parent B)
    to produce an energy term that feeds the JEPA residual.

    Steps:
        1. Compute Fisher score F(theta).
        2. Build a tropical polynomial where the Fisher score is the leading
           coefficient and the remaining coefficients are zero.
           coeffs = [0, F(theta), -inf, -inf, ...]  (length 3 is enough)
        3. Evaluate the tropical polynomial at the candidate timestamp.
        4. Apply trust as a tropical multiplier to the polynomial value.
        5. Use the result as the Fisher‑derived term in the JEPA energy.
    """
    F = fisher_score(theta)
    # Coefficients: constant term 0, linear term = F, higher terms = -inf (ignored)
    coeffs = np.array([0.0, F, -np.inf])
    poly_val = t_polyval(coeffs, candidate)          # tropical polynomial value
    poly_trust = t_mul(poly_val, trust)               # tropical scaling by trust
    # Use the tropical‑scaled polynomial value as the Fisher term in JEPA energy
    energy = jeap_energy(candidate, prev_candidate, float(poly_trust))
    return energy

def hybrid_voronoi_tropical_flow(points, num_partitions, trust=1.0):
    """
    Demonstrates a higher‑level hybrid operation:
    1. Compute the tropical‑metric distance matrix between all points.
    2. Apply tropical scaling with trust to the distance matrix.
    3. Perform a simple Voronoi‑like partition by assigning each point to the
       nearest (tropically weighted) centroid selected via random sampling.
    Returns a list of partitions, each a list of point indices.
    """
    n = len(points)
    if n == 0:
        return []

    # Randomly pick initial centroids
    centroids_idx = random.sample(range(n), min(num_partitions, n))
    centroids = [points[i] for i in centroids_idx]

    # Compute tropical‑scaled distance matrix
    dist_matrix = np.zeros((n, len(centroids)), dtype=float)
    for i, pt in enumerate(points):
        for j, ctr in enumerate(centroids):
            d = hybrid_tropical_metric_distance(pt, ctr, trust)
            dist_matrix[i, j] = d

    # Assign points to the centroid with minimal (tropical) distance
    assignments = np.argmin(dist_matrix, axis=1)

    partitions = [[] for _ in range(len(centroids))]
    for idx, part in enumerate(assignments):
        partitions[part].append(idx)

    return partitions

# ---------- Smoke Test ----------
if __name__ == "__main__":
    # Simple scalar test for hybrid_trust_tropical_velocity
    x0_s, x1_s, trust_s = 2.0, 7.0, 0.6
    v_hybrid = hybrid_trust_tropical_velocity(x0_s, x1_s, trust_s)
    print(f"Hybrid trust‑tropical velocity (scalar): {v_hybrid}")

    # Vector test for hybrid_tropical_metric_distance
    p_vec = np.array([1.0, 2.0, 3.0])
    q_vec = np.array([4.0, 0.0, -1.0])
    d_hybrid = hybrid_tropical_metric_distance(p_vec, q_vec, trust=0.8)
    print(f"Hybrid tropical metric distance: {d_hybrid}")

    # Energy test combining Fisher and tropical polynomial
    candidate_ts = 1.5
    prev_ts = 1.0
    theta_val = 0.3
    energy = hybrid_fisher_tropical_flow_energy(candidate_ts, prev_ts, theta_val, trust=0.9)
    print(f"Hybrid Fisher‑tropical flow energy: {energy}")

    # Voronoi‑like partition test
    pts = [np.random.randn(3) for _ in range(10)]
    partitions = hybrid_voronoi_tropical_flow(pts, num_partitions=3, trust=0.7)
    for i, part in enumerate(partitions):
        print(f"Partition {i}: point indices {part}")