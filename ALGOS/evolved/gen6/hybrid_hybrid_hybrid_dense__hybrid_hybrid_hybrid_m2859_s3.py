# DARWIN HAMMER — match 2859, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_dense_associa_hybrid_hybrid_hybrid_m1726_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_model__m1308_s0.py (gen4)
# born: 2026-05-29T23:46:28Z

import numpy as np
import math
from typing import Optional, Tuple


# ----------------------------------------------------------------------
# Dense Associative Memory utilities (Parent A)
# ----------------------------------------------------------------------
def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for a 1‑D array."""
    z = np.asarray(z, dtype=float)
    z_max = np.max(z)
    e = np.exp(z - z_max)
    return e / e.sum()


def _logsumexp(z: np.ndarray) -> float:
    """Stable log‑sum‑exp for a 1‑D array."""
    z = np.asarray(z, dtype=float)
    m = np.max(z)
    return float(m + np.log(np.exp(z - m).sum()))


def energy(xi: np.ndarray, M: np.ndarray, beta: float = 1.0) -> float:
    """
    Dense Associative Memory energy.

    The original implementation added a redundant LSE term.
    Here we keep the canonical definition:
        E(x) = - (1/β) * log Σ_i exp(β * M_i·x) + ½‖x‖²
    """
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    logits = beta * (M @ xi)                     # (N,)
    return -_logsumexp(logits) / beta + 0.5 * (xi @ xi)


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_weights(
    theta: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Vectorised Fisher‑score weights for a Gaussian beam.

    Returns d/dθ log p(θ) = -(θ‑c)/σ² * exp(-½((θ‑c)/σ)²)
    The intensity is lower‑bounded by ``eps`` to avoid exact zeros.
    """
    theta = np.asarray(theta, dtype=float)
    centers = np.asarray(centers, dtype=float)
    widths = np.asarray(widths, dtype=float)

    if np.any(widths <= 0):
        raise ValueError("All widths must be positive")

    diff = theta - centers
    z = diff / widths
    intensity = np.exp(-0.5 * z ** 2)
    intensity = np.maximum(intensity, eps)

    return -(diff / (widths ** 2)) * intensity


def fisher_weighted_softmax(
    vec: np.ndarray,
    centers: Optional[np.ndarray] = None,
    widths: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Apply Fisher‑derived weighting to ``vec`` and return a softmax‑normalised
    vector.  If ``centers`` or ``widths`` are omitted they default to 0 and 1,
    respectively, which reduces the operation to an identity weighting.
    """
    vec = np.asarray(vec, dtype=float)

    if centers is None:
        centers = np.zeros_like(vec)
    if widths is None:
        widths = np.ones_like(vec)

    w = fisher_weights(vec, centers, widths, eps=eps)
    weighted = vec * w
    return _softmax(weighted)


def dense_associative_retrieve(
    query: np.ndarray,
    memory: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Retrieve a pattern from ``memory`` using a softmax over inner products.
    Returns the weighted sum of stored patterns.
    """
    query = np.asarray(query, dtype=float)
    memory = np.asarray(memory, dtype=float)

    scores = beta * (memory @ query)          # (N,)
    probs = _softmax(scores)                 # (N,)
    return probs @ memory                    # (d,)


# ----------------------------------------------------------------------
# Path‑signature utilities (Parent B)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform: interleaves (lead, lag) channels to encode causality.
    Input shape: (T, d). Output shape: (2·T‑1, 2·d).
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (T, d)")

    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)

    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])

    out[-1] = np.concatenate([path[-1], path[-1]])
    return out


def bspline_basis(
    x: np.ndarray,
    knots: np.ndarray,
    order: int = 3,
) -> np.ndarray:
    """
    Evaluate B‑spline basis functions of given order at positions ``x``.
    Returns a matrix of shape (len(x), n_basis).
    """
    x = np.asarray(x, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64)

    if order < 1:
        raise ValueError("order must be >= 1")

    # Augment knot vector with clamped ends
    t = np.concatenate([
        np.full(order, knots[0]),
        knots,
        np.full(order, knots[-1]),
    ])

    N = len(x)
    n_basis = len(t) - 1
    B = np.zeros((N, n_basis), dtype=np.float64)

    # Order‑1 (piecewise constant) basis
    for i in range(n_basis):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    # Include the right‑most knot
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    # Recurrence for higher orders
    for k in range(2, order + 1):
        n_basis_k = n_basis - (k - 1)
        B_new = np.zeros((N, n_basis_k), dtype=np.float64)

        for i in range(n_basis_k):
            denom_l = t[i + k - 1] - t[i]
            denom_r = t[i + k] - t[i + 1]

            left = ((x - t[i]) / denom_l) * B[:, i] if denom_l > 0 else 0.0
            right = ((t[i + k] - x) / denom_r) * B[:, i + 1] if denom_r > 0 else 0.0

            B_new[:, i] = left + right

        B = B_new
        n_basis = n_basis_k

    return B


def path_signature_approx(
    path: np.ndarray,
    knots: np.ndarray,
    spline_order: int = 3,
) -> np.ndarray:
    """
    Approximate a path signature by projecting the lead‑lag transformed path
    onto a B‑spline basis.

    Returns a flattened vector of shape (2·d·n_basis,).
    """
    ll = lead_lag_transform(path)                     # (2T‑1, 2d)
    T_ll, dim_ll = ll.shape

    # Parameterise by a uniform “time” coordinate in [0, 1].
    t_param = np.linspace(0.0, 1.0, T_ll)

    B = bspline_basis(t_param, knots, order=spline_order)   # (T_ll, n_basis)
    coeffs = B.T @ ll                                        # (n_basis, 2d)

    return coeffs.ravel()                                    # (2d·n_basis,)


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def hybrid_path_memory_query(
    path: np.ndarray,
    memory: np.ndarray,
    beta: float = 1.0,
    knots: Optional[np.ndarray] = None,
    centers: Optional[np.ndarray] = None,
    widths: Optional[np.ndarray] = None,
    spline_order: int = 3,
) -> np.ndarray:
    """
    Full hybrid operation:

    1. Convert ``path`` to a high‑dimensional signature.
    2. Modulate the signature with Fisher‑derived weights and softmax‑normalise.
    3. Use the resulting vector as a query for Dense Associative Memory retrieval.

    Parameters
    ----------
    path
        Input time‑series of shape (T, d).
    memory
        Stored patterns of shape (N, d_sig) where ``d_sig`` matches the
        signature dimension.
    beta
        Inverse temperature for the associative retrieval.
    knots
        Knot positions for the B‑spline basis. If ``None`` a uniform grid of
        20 points in [0, 1] is used.
    centers, widths
        Parameters for the Fisher weighting. If omitted they default to 0 and 1.
    spline_order
        Order of the B‑spline (default cubic).

    Returns
    -------
    np.ndarray
        Retrieved pattern of shape (d_sig,).
    """
    path = np.asarray(path, dtype=float)
    memory = np.asarray(memory, dtype=float)

    if knots is None:
        # A modest default grid; can be overridden by the caller.
        knots = np.linspace(0.0, 1.0, 20)

    # 1️⃣ Signature extraction
    sig = path_signature_approx(path, knots, spline_order=spline_order)

    # 2️⃣ Fisher modulation + softmax
    query = fisher_weighted_softmax(sig, centers=centers, widths=widths)

    # 3️⃣ Dense associative retrieval
    return dense_associative_retrieve(query, memory, beta=beta)