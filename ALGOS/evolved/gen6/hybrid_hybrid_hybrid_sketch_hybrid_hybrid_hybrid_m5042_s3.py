# DARWIN HAMMER — match 5042, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_sketches_hybr_hybrid_hybrid_hybrid_m1904_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s3.py (gen3)
# born: 2026-05-29T23:59:28Z

"""Hybrid Path‑Morphology Signature – Bridging Lead‑Lag Transform with Morphological Interpolation

Parent A:
    hybrid_hybrid_sketches_hybr_hybrid_hybrid_hybrid_m1904_s0.py
    – Provides a lead‑lag transform for discrete paths and a B‑spline basis
      (Cox‑de Boor recursion).

Parent B:
    hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s3.py
    – Defines morphological descriptors for engine endpoints and a straight‑line
      interpolant to compute loading vectors between morphologies.

Mathematical Bridge:
    The lead‑lag transform lifts a time‑ordered path into a doubled‑dimension
    space where each segment encodes both current and previous coordinates.
    Morphological descriptors (length, width, height, mass) are interpreted as
    a 4‑D vector.  By linearly interpolating between two morphologies we obtain
    a loading vector that can be projected onto the B‑spline basis of the
    transformed path.  The final hybrid signature is constructed via a simple
    geometric product (outer product + scalar part) between the projected
    path coefficients and the loading vector, yielding a unified representation
    that captures both temporal dynamics and structural characteristics.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# Parent A – Lead‑lag transform and B‑spline basis
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Embed a discrete path into a higher‑dimensional lead‑lag space.
    Input:
        path – (T, d) array of points.
    Output:
        (2*T-1, 2*d) array where each even row contains the current point
        and zeros, each odd row contains the next point and the current point.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (T, d)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], np.zeros(d)])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], np.zeros(d)])
    return out

def _cox_de_boor(x: float, i: int, k: int, t: np.ndarray) -> float:
    """
    Recursive Cox‑de Boor evaluation of a single B‑spline basis function.
    """
    if k == 1:
        return 1.0 if (t[i] <= x < t[i + 1]) or (x == t[-1] and t[i + 1] == t[-1]) else 0.0
    denom_left = t[i + k - 1] - t[i]
    denom_right = t[i + k] - t[i + 1]
    term_left = 0.0
    term_right = 0.0
    if denom_left > 1e-12:
        term_left = ((x - t[i]) / denom_left) * _cox_de_boor(x, i, k - 1, t)
    if denom_right > 1e-12:
        term_right = ((t[i + k] - x) / denom_right) * _cox_de_boor(x, i + 1, k - 1, t)
    return term_left + term_right

def bspline_basis(x: np.ndarray, knots: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Compute the B‑spline basis matrix B(x) of order k for a set of knots.
    Parameters
    ----------
    x : (N,) array_like
        Evaluation points.
    knots : (M,) array_like
        Non‑decreasing knot vector.
    k : int, default 3
        Order of the spline (degree = k‑1).
    Returns
    -------
    B : (N, M + k - 2) ndarray
        Basis matrix where B[n, i] = N_{i,k}(x_n).
    """
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if knots.ndim != 1:
        raise ValueError("knots must be a 1‑D array")
    N = x.shape[0]
    n_basis = len(knots) + k - 2
    B = np.zeros((N, n_basis), dtype=float)

    # Extend knot vector with k‑1 repeated end knots (clamped spline)
    t = np.concatenate((
        np.full(k - 1, knots[0]),
        knots,
        np.full(k - 1, knots[-1])
    ))

    for i in range(n_basis):
        for n in range(N):
            B[n, i] = _cox_de_boor(x[n], i, k, t)
    return B

# ----------------------------------------------------------------------
# Parent B – Morphology, EngineEndpoint and straight‑line interpolant
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

    def as_vector(self) -> np.ndarray:
        return np.array([self.length, self.width, self.height, self.mass], dtype=float)

@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"

def straight_line_interpolant(m1: Morphology, m2: Morphology, t: float) -> Morphology:
    """
    Linear interpolation between two morphologies.
    t ∈ [0, 1] where 0 returns m1 and 1 returns m2.
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError("Interpolation parameter t must be in [0, 1]")
    v1 = m1.as_vector()
    v2 = m2.as_vector()
    vi = (1 - t) * v1 + t * v2
    return Morphology(*vi.tolist())

# ----------------------------------------------------------------------
# Hybrid Operations – Combining the two parent structures
# ----------------------------------------------------------------------
def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Simple geometric product for vectors a, b ∈ ℝⁿ:
        a ⋅ b (scalar part) + a ∧ b (bivector part approximated by outer product).
    Returns a matrix of shape (n, n) where the diagonal contains the scalar part
    and the off‑diagonal entries contain the outer product.
    """
    a = a.reshape(-1, 1)
    b = b.reshape(1, -1)
    scalar = float(np.dot(a.ravel(), b.ravel()))
    outer = a @ b
    # Embed scalar on the diagonal
    gp = outer.copy()
    np.fill_diagonal(gp, gp.diagonal() + scalar)
    return gp

def hybrid_signature(endpoint: EngineEndpoint,
                     path: np.ndarray,
                     knots: np.ndarray,
                     interp_t: float = 0.5) -> np.ndarray:
    """
    Compute a hybrid signature that fuses:
        1. Lead‑lag transformed path projected onto a B‑spline basis.
        2. Morphological loading vector obtained by interpolating the endpoint's
           morphology with a reference morphology (here we use a unit cube).
        3. Geometric product of the projected coefficients with the loading vector.
    Parameters
    ----------
    endpoint : EngineEndpoint
        The engine endpoint providing morphological data.
    path : (T, d) ndarray
        Original discrete path.
    knots : (M,) ndarray
        Knot vector for the B‑spline basis.
    interp_t : float, default 0.5
        Interpolation factor between the endpoint morphology and a reference.
    Returns
    -------
    signature : ndarray
        Matrix representing the hybrid geometric product (size = d*2 × 4).
    """
    # 1. Lead‑lag transform
    ll_path = lead_lag_transform(path)                     # shape (2T‑1, 2d)
    # 2. Project each coordinate dimension onto B‑spline basis
    #    We treat each column of ll_path as a separate scalar function of time.
    times = np.linspace(0, 1, ll_path.shape[0])
    B = bspline_basis(times, knots, k=3)                  # (2T‑1, n_basis)
    coeffs = B.T @ ll_path                                 # (n_basis, 2d)

    # 3. Morphology loading vector (interpolated)
    reference = Morphology(1.0, 1.0, 1.0, 1.0)
    loaded_morph = straight_line_interpolant(endpoint.morphology,
                                             reference,
                                             interp_t)
    load_vec = loaded_morph.as_vector()                    # (4,)

    # 4. Combine: for each basis coefficient vector (length 2d) compute geometric product
    #    with the loading vector (length 4). Resulting shape: (n_basis, 2d, 4)
    n_basis, dim = coeffs.shape
    signature = np.empty((n_basis, dim, load_vec.size), dtype=float)
    for i in range(n_basis):
        signature[i] = geometric_product(coeffs[i], load_vec)
    return signature  # shape (n_basis, 2d, 4)

def count_min_sketch(items: List[str], width: int = 1000, depth: int = 5) -> np.ndarray:
    """
    Simple count‑min sketch for estimating frequencies of string items.
    Returns a (depth, width) integer matrix.
    """
    sketch = np.zeros((depth, width), dtype=int)

    def _hash(item: str, seed: int) -> int:
        h = hashlib.blake2b(item.encode('utf-8'), digest_size=8, person=seed.to_bytes(4, 'little'))
        return int.from_bytes(h.digest(), 'little') % width

    for item in items:
        for d in range(depth):
            col = _hash(item, d)
            sketch[d, col] += 1
    return sketch

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic path (3 points in 2‑D)
    path = np.array([[0.0, 0.0],
                     [1.0, 1.0],
                     [2.0, 0.0]])

    # Knot vector for B‑spline (uniform)
    knots = np.linspace(0, 1, 5)  # 5 interior knots

    # Define two morphologies and an endpoint
    morph_a = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)
    morph_b = Morphology(length=1.5, width=0.8, height=0.6, mass=2.5)
    endpoint = EngineEndpoint(
        engine_id="E42",
        channel="alpha",
        residency="local",
        runtime="prod",
        resource_class="high",
        always_on=True,
        endpoint="http://example.com/api",
        capabilities=["read", "write"],
        morphology=morph_a
    )

    # Compute hybrid signature
    sig = hybrid_signature(endpoint, path, knots, interp_t=0.3)
    print("Hybrid signature shape:", sig.shape)

    # Simple count‑min sketch of capabilities
    sketch = count_min_sketch(endpoint.capabilities, width=50, depth=3)
    print("Count‑Min sketch matrix:\n", sketch)