# DARWIN HAMMER — match 2893, survivor 3
# gen: 7
# parent_a: hybrid_path_signature_kan_m30_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m1248_s0.py (gen6)
# born: 2026-05-29T23:46:26Z

"""
HYBRID ALGORITHM: Path-Signature-Physarum-Flow
=============================================

This module fuses the Path-Signature algorithm (parent A) with the Physarum-Flow algorithm (parent B). 
The mathematical bridge between the two structures is found by using the Rectified Flow's straight-line interpolant 
to generate input features for the NLMS predictor, which attempts to model the wavefront velocity of the graph-propagation engine.
This hybrid algorithm leverages the interplay between path signatures and physarum network dynamics to generate a novel flow-based signature.

"""

import numpy as np
import math
import random
import sys
import pathlib

class Multivector:
    """Sparse multivector in a Clifford algebra with n basis vectors."""
    def __init__(self, components: Dict[frozenset[int], float], n: int):
        self.n = int(n)
        # discard near-zero entries for sparsity
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    def grade(self, k: int) -> "Multivector":
        """Return the grade-k part."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the grade-0 (scalar) component."""
        return self.components.get(frozenset(), 0.0)


def lead_lag_transform(path):
    """Lead-lag transform: interleave (lead, lag) channels for causality encoding.

    path: (T, d). Returns (2T-1, 2d) interleaved lead-lag path.

    At even indices 2t   : (X_t,   X_t)    (lead and lag both at t)
    At odd  indices 2t+1 : (X_{t+1}, X_t)  (lead advances, lag holds)
    This matches the standard Chevyrev-Kormilitzin convention.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def signature_level1(path):
    """Level-1 signature: total increment vector.

    path: (T, d). Returns (d,). Equal to path[-1] - path[0].
    """
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def signature_level2(path):
    """Level-2 iterated integral tensor.

    path: (T, d). Returns (d, d).
    S2[i,j] = sum_{s<t} (X_s[i] - X_0[i]) * dX_t[j]
             = sum_{t=1..T-1} (X_{t-1}[i] - X_0[i]) * (X_t[j] - X_{t-1}[j])

    Uses the standard left-point Riemann sum on the increment path.
    """
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    # S2[i,j] = sum_t running[t,i] * increments[t,j]
    return running.T @ increments               # (d, d)


def interpolant(x0, x1, t):
    """Straight-line interpolant: Z_t = t * x1 + (1 - t) * x0."""
    return t * x1 + (1 - t) * x0


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the dot-product prediction w·x."""
    return float(weights @ x)


def hybrid_signature(path, physarum_state, weights):
    """Compute the hybrid signature of a path and a physarum network state.

    Uses the Rectified Flow's straight-line interpolant to generate input features for the NLMS predictor.

    path: (T, d). Returns (d,).
    """
    path = np.asarray(path, dtype=float)
    lead_lag_path = lead_lag_transform(path)
    signature = signature_level1(path)
    interpolant_path = interpolant(path[-1], path[0], 0.5)
    input_features = np.concatenate([interpolant_path, path])
    prediction = nlms_predict(weights, input_features)
    gradient = physarum_state.grade(1).scalar_part()
    return signature + gradient * prediction


def physarum_update(physarum_state, gradient, weights):
    """Update the physarum network state using the hybrid rule.

    physarum_state: Multivector. Returns updated Multivector.
    """
    return physarum_state + gradient * weights


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
    B: shape (N, G - 1) — one column per basis function
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


if __name__ == "__main__":
    # smoke test
    np.random.seed(0)
    path = np.random.rand(10, 3)
    physarum_state = Multivector({frozenset([1]): 2.0}, 3)
    weights = np.random.rand(3, 6)
    hybrid_signature(path, physarum_state, weights)
    physarum_update(physarum_state, 0.5, weights)