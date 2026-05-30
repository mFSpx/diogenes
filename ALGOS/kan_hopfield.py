# DARWIN HAMMER — match 382, survivor 0
# gen: 1
# parent_a: dense_associative_memory.py (gen0)
# parent_b: kan.py (gen0)
# born: 2026-05-29T23:28:30Z
#
# DISTILLED USE: Algorithm capability router. B-splines on each column of
# the Hopfield memory matrix — the attractor landscape is shaped by learned
# spline coefficients, not fixed weights. The similarity metric itself is
# learned per-dimension. Use for routing task descriptions to the right
# ALGOS/ primitive. Better than flat cosine, cheaper than a full attn head.

"""Hybrid Dense Associative Memory – Kolmogorov‑Arnold Network (HAM‑KAN)

This module fuses the two parent algorithms:

* **Dense Associative Memory (modern Hopfield network)** – an energy based
  retrieval rule `xi_new = Mᵀ softmax(β·M·xi)`.  The memory matrix `M` stores
  patterns and the softmax implements a Boltzmann distribution over them.

* **Kolmogorov‑Arnold Network (KAN)** – every edge carries a learnable
  univariate function, implemented as a B‑spline.  For a matrix `X` the KAN
  transformation is `X̂_{i,p} = ϕ_{p}(X_{i,p})` where each column uses its own
  spline.

**Mathematical bridge**

The Hopfield update only needs a *linear* memory matrix.  We replace that
matrix by a *KAN‑transformed* matrix `M̂ = Φ(M)`, i.e. each element of `M` is
first passed through a learned univariate spline.  The resulting `M̂` is then
used in the Hopfield energy and update rule.  Consequently the hybrid system
inherits:

* the expressive, per‑edge non‑linearity of KAN,
* the associative retrieval dynamics of modern Hopfield networks.

The core hybrid operations are:


M̂ = kan_transform(M, grids, coeffs)          # KAN edgewise transform
E(xi) = -β⁻¹·log Σ_i exp(β·M̂_i·xi) + ½·||xi||²
xi_{new} = M̂ᵀ·softmax(β·M̂·xi)                # Hopfield‑style retrieval


All functions are pure NumPy and require no external deep‑learning libraries.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

__all__ = [
    "bspline_basis",
    "spline_evaluate",
    "kan_transform",
    "softmax",
    "lse",
    "hybrid_energy",
    "hybrid_update",
    "hybrid_retrieve",
]

# ---------------------------------------------------------------------------
# 1. B‑spline utilities (parent algorithm B)
# ---------------------------------------------------------------------------


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Cox‑de Boor recursion for uniform clamped B‑splines.

    Parameters
    ----------
    x : ndarray shape (N,)
        Evaluation points (must lie inside the grid range).
    grid : ndarray shape (G,)
        Interior breakpoints (uniform spacing assumed).
    k : int, default 3
        Spline order (degree = k‑1).  k=3 yields cubic splines.

    Returns
    -------
    B : ndarray shape (N, G‑1)
        Basis matrix, one column per basis function.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Build clamped knot vector: repeat first/last knot k times.
    knots = np.concatenate(
        ([grid[0]] * k, grid[1:-1], [grid[-1]] * k)
    )
    n_basis = len(grid) - 1  # number of basis functions

    # Initialise zeroth‑order basis (piecewise constant)
    B = np.zeros((x.size, n_basis, k), dtype=np.float64)  # extra dim for recursion
    for i in range(n_basis):
        left = knots[i]
        right = knots[i + 1]
        B[:, i, 0] = (x >= left) & (x < right)
    # Special case for the rightmost knot (include the endpoint)
    B[:, -1, 0] |= x == knots[-1]

    # Recursion for higher orders
    for d in range(1, k):
        for i in range(n_basis):
            left_den = knots[i + d] - knots[i]
            right_den = knots[i + d + 1] - knots[i + 1]

            left_num = x - knots[i]
            right_num = knots[i + d + 1] - x

            left = np.where(
                left_den > 0,
                left_num / left_den * B[:, i, d - 1],
                0.0,
            )
            right = np.where(
                right_den > 0,
                right_num / right_den * B[:, i + 1, d - 1],
                0.0,
            )
            B[:, i, d] = left + right

    # Return only the highest order basis functions
    return B[:, :, k - 1]


def spline_evaluate(
    x: np.ndarray,
    grid: np.ndarray,
    coeffs: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """Evaluate a univariate spline defined by B‑spline coefficients.

    Parameters
    ----------
    x : ndarray shape (N,)
        Points at which to evaluate.
    grid : ndarray shape (G,)
        Grid defining the interior knots.
    coeffs : ndarray shape (G‑1,)
        Coefficients for each basis function.
    k : int, default 3
        Spline order.

    Returns
    -------
    y : ndarray shape (N,)
        Spline values.
    """
    B = bspline_basis(x, grid, k)          # (N, G‑1)
    return B @ coeffs                       # (N,)


# ---------------------------------------------------------------------------
# 2. KAN transformation of a matrix (edgewise univariate splines)
# ---------------------------------------------------------------------------


def kan_transform(
    M: np.ndarray,
    grids: List[np.ndarray],
    coeffs: List[np.ndarray],
    k: int = 3,
) -> np.ndarray:
    """Apply a per‑column B‑spline (KAN) transform to matrix `M`.

    Parameters
    ----------
    M : ndarray shape (N, d)
        Input matrix (e.g., memory patterns).
    grids : list of length d, each ndarray (G_i,)
        Grid for the spline of column i.
    coeffs : list of length d, each ndarray (G_i‑1,)
        Coefficients for column i.
    k : int, default 3
        Spline order.

    Returns
    -------
    M̂ : ndarray shape (N, d)
        Transformed matrix.
    """
    M = np.asarray(M, dtype=np.float64)
    N, d = M.shape
    if len(grids) != d or len(coeffs) != d:
        raise ValueError("Length of grids/coeffs must equal number of columns.")
    M_hat = np.empty_like(M)
    for col in range(d):
        col_vals = M[:, col]
        M_hat[:, col] = spline_evaluate(col_vals, grids[col], coeffs[col], k)
    return M_hat


# ---------------------------------------------------------------------------
# 3. Softmax / log‑sum‑exp utilities (parent algorithm A)
# ---------------------------------------------------------------------------


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1‑D array."""
    z = np.asarray(z, dtype=np.float64)
    z_max = np.max(z)
    e = np.exp(z - z_max)
    return e / e.sum()


def lse(z: np.ndarray) -> float:
    """Log‑sum‑exp of a 1‑D array (stable)."""
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z)
    return m + np.log(np.exp(z - m).sum())


# ---------------------------------------------------------------------------
# 4. Hybrid Hopfield‑KAN energy and dynamics
# ---------------------------------------------------------------------------


def hybrid_energy(
    xi: np.ndarray,
    M: np.ndarray,
    grids: List[np.ndarray],
    coeffs: List[np.ndarray],
    beta: float = 1.0,
    k: int = 3,
) -> float:
    """Energy of the hybrid system.

    The memory matrix is first transformed by a KAN, then fed into the modern
    Hopfield energy.

    Parameters
    ----------
    xi : ndarray shape (d,)
        Current query/state vector.
    M : ndarray shape (N, d)
        Raw memory patterns.
    grids, coeffs : per‑column spline specifications.
    beta : float
        Inverse temperature.
    k : int
        Spline order.

    Returns
    -------
    float
        Scalar energy.
    """
    xi = np.asarray(xi, dtype=np.float64)
    M_hat = kan_transform(M, grids, coeffs, k)          # (N, d)
    scores = beta * (M_hat @ xi)                       # (N,)
    energy = -lse(scores) / beta + 0.5 * np.dot(xi, xi)
    return float(energy)


def hybrid_update(
    xi: np.ndarray,
    M: np.ndarray,
    grids: List[np.ndarray],
    coeffs: List[np.ndarray],
    beta: float = 1.0,
    k: int = 3,
) -> np.ndarray:
    """One Hopfield‑style update step using KAN‑transformed memory.

    Parameters
    ----------
    xi : ndarray shape (d,)
        Current state.
    M, grids, coeffs, beta, k : as in `hybrid_energy`.

    Returns
    -------
    xi_new : ndarray shape (d,)
        Updated state vector.
    """
    xi = np.asarray(xi, dtype=np.float64)
    M_hat = kan_transform(M, grids, coeffs, k)          # (N, d)
    scores = beta * (M_hat @ xi)                       # (N,)
    w = softmax(scores)                                 # (N,)
    xi_new = M_hat.T @ w                                 # (d,)
    return xi_new


def hybrid_retrieve(
    xi: np.ndarray,
    M: np.ndarray,
    grids: List[np.ndarray],
    coeffs: List[np.ndarray],
    beta: float = 1.0,
    steps: int = 5,
    k: int = 3,
) -> np.ndarray:
    """Iteratively apply `hybrid_update` to retrieve a stored pattern.

    Parameters
    ----------
    xi : ndarray shape (d,)
        Initial query (can be noisy).
    M, grids, coeffs, beta, k : as above.
    steps : int
        Number of update iterations.

    Returns
    -------
    xi_ret : ndarray shape (d,)
        Retrieved (converged) state.
    """
    state = np.asarray(xi, dtype=np.float64)
    for _ in range(steps):
        state = hybrid_update(state, M, grids, coeffs, beta, k)
    return state


# ---------------------------------------------------------------------------
# 5. Simple utilities to generate random spline parameters (for testing)
# ---------------------------------------------------------------------------


def _random_grid(num_knots: int = 5, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    """Create a uniformly spaced grid within [low, high]."""
    return np.linspace(low, high, num_knots)


def _random_coeffs(num_basis: int) -> np.ndarray:
    """Random coefficients for B‑spline basis functions."""
    return np.random.randn(num_basis)


# ---------------------------------------------------------------------------
# 6. Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Seed for reproducibility
    np.random.seed(42)

    # Dimensions
    N = 12          # number of stored patterns
    d = 6           # dimensionality

    # Random memory matrix
    M = np.random.randn(N, d)

    # Build per‑column spline specifications
    grids: List[np.ndarray] = []
    coeffs: List[np.ndarray] = []
    for _ in range(d):
        grid = _random_grid(num_knots=6)            # 6 interior knots → 5 basis fns
        grids.append(grid)
        coeffs.append(_random_coeffs(len(grid) - 1))

    # Query vector (noisy version of a stored pattern)
    true_pattern = M[3]                         # pick one pattern as ground truth
    noise = 0.2 * np.random.randn(d)
    xi0 = true_pattern + noise

    # Hybrid retrieval
    beta = 5.0
    retrieved = hybrid_retrieve(xi0, M, grids, coeffs, beta=beta, steps=10)

    # Print diagnostics
    print("Original pattern :", true_pattern)
    print("Noisy query      :", xi0)
    print("Retrieved state  :", retrieved)
    print("Energy after retrieval :", hybrid_energy(retrieved, M, grids, coeffs, beta))
    # Simple distance check
    dist = np.linalg.norm(retrieved - true_pattern)
    print(f"Euclidean distance to true pattern: {dist:.4f}")

    # Ensure the module can be imported without side‑effects
    sys.exit(0)