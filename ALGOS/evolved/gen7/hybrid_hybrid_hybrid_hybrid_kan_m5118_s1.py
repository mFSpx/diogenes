# DARWIN HAMMER — match 5118, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_tropic_m2576_s2.py (gen6)
# parent_b: kan.py (gen0)
# born: 2026-05-29T23:59:51Z

"""
Kolmogorov-Arnold Networks (KAN) — hybrid reference implementation with DARWIN HAMMER.

This module integrates the Kolmogorov-Arnold representation theorem with the 
DARWIN HAMMER framework. The mathematical bridge between the two structures is 
the concept of Clifford-geometric distance, where the Fisher score from the DARWIN 
HAMMER is generalized to a B-spline basis evaluation in the KAN. This allows for 
the fusion of trust-weighted velocity fields with tropical polynomial evaluation 
and Clifford-algebraic operations.

The governing equations of both parents are integrated through the hybrid_flow_loss 
function, which combines the trust-weighted velocity field with the tropical 
polynomial evaluation and Clifford-geometric distance.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path


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
    return trust * (x1 - x0)


def jeap_energy(candidate: float, prev_candidate: float, fisher_score: float) -> float:
    predictor = np.array([prev_candidate + fisher_score])
    encoder = np.array([candidate])
    return np.sum((encoder - predictor) ** 2)


def t_add(x, y):
    """Tropical addition: max(x, y). Broadcasts."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication: x + y. Broadcasts."""
    return np.add(x, y)


def t_matmul(A, B):
    """Tropical matrix multiplication: max(sum(A_i * B_i)). Broadcasts."""
    return np.maximum.reduce(np.matmul(A, B))


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Evaluate B-spline basis functions of order k at positions x.

    Uses the Cox-de Boor recursion on a clamped knot vector derived from
    *grid* by repeating the first and last knot k times.

    Parameters
    ----------
    x:    shape (N,) — evaluation points (should lie within grid range).
    grid: shape (G,) — uniformly spaced interior breakpoints; the knot
          vector is constructed as k copies of grid[0], then grid[1:-1],
          then k copies of grid[-1], giving G + 2*(k-1) total knots.
    k:    spline order (polynomial degree = k - 1).  Default 3 (cubic).

    Returns
    -------
    B: shape (N, G - 1) — one column per basis function.
       Number of basis functions = len(grid) - 1.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Build clamped knot vector: repeat boundary knots (k-1) times so that
    # the polynomial spans cleanly to the boundary.
    # Knot vector length: (k-1) + G + (k-1) = G + 2(k-1)
    # After Cox-de Boor recursion, we have n_basis = G + 2(k-1) basis
    # functions.

    # Evaluate the Cox-de Boor recursion.
    n_basis = grid.size - 1 + 2 * (k - 1)
    B = np.zeros((x.size, n_basis))
    for i in range(n_basis):
        for j in range(x.size):
            if 0 <= i < k:
                if j == 0:
                    B[j, i] = 1  # left boundary knot
                elif j == x.size - 1:
                    B[j, i] = 1  # right boundary knot
                else:
                    B[j, i] = 0
            else:
                alpha = grid[i - k + 1] / (grid[i - k + 1] - grid[i - k])
                B[j, i] = alpha * B[j, i - 1] + (1 - alpha) * B[j, i - 2]
    return B


def kan_layer(x, Phi, k=3):
    """Evaluate KAN layer.

    Parameters
    ----------
    x:    shape (N,) — input points.
    Phi:  shape (H,) — univariate functions (B-splines).

    Returns
    -------
    y: shape (N, H) — output points.
    """
    B = bspline_basis(x, Phi, k=k)
    y = np.dot(B, Phi)
    return y


def hybrid_flow_loss(x, Phi, k=3, alpha=0.5):
    """Hybrid flow loss function.

    Parameters
    ----------
    x:    shape (N,) — input points.
    Phi:  shape (H,) — univariate functions (B-splines).
    k:    spline order (polynomial degree = k - 1).  Default 3 (cubic).
    alpha: weighting factor for trust-weighted velocity field.  Default 0.5.

    Returns
    -------
    loss: shape (N,) — hybrid flow loss.
    """
    # Evaluate the trust-weighted velocity field.
    v = trust_weighted_velocity(x[0], x[1], alpha)
    # Evaluate the B-spline basis functions.
    B = bspline_basis(x, Phi, k=k)
    # Evaluate the KAN layer.
    y = kan_layer(x, Phi, k=k)
    # Compute the tropical polynomial evaluation.
    t = t_add(np.dot(B, Phi), y)
    # Compute the Clifford-geometric distance.
    d = fisher_score(t, center=0, width=1, eps=1e-12)
    # Compute the hybrid flow loss.
    loss = np.sum((d - v) ** 2)
    return loss


def test_hybrid_flow_loss():
    # Define the input points.
    x = np.array([1.0, 2.0])
    # Define the univariate functions (B-splines).
    Phi = np.array([1.0, 2.0, 3.0])
    # Define the spline order (polynomial degree = k - 1).
    k = 3
    # Define the weighting factor for the trust-weighted velocity field.
    alpha = 0.5
    # Compute the hybrid flow loss.
    loss = hybrid_flow_loss(x, Phi, k=k, alpha=alpha)
    print(loss)


if __name__ == "__main__":
    test_hybrid_flow_loss()