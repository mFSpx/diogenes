# DARWIN HAMMER — match 3390, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_model__m1308_s2.py (gen4)
# parent_b: hybrid_hybrid_sketches_hybr_hybrid_hybrid_hybrid_m1904_s2.py (gen5)
# born: 2026-05-29T23:49:51Z

# hybrid_hybrid_path_signatur_hybrid_hybrid_model__m1308_s2_m1904_s2.py
"""HybridPathKANFoldChange

This module fuses the core mathematics of the two parent algorithms:

* **Parent A** – HybridPathKANFoldChange (hybrid_hybrid_hybrid_path_s_hybrid_hybrid_model__m1308_s2.py)
  It provides a lead-lag transform that encodes causality and a B-spline basis
  that approximates the iterated-integral signature using the same spline
  functions that KANs employ as activation maps.
* **Parent B** – Hybrid Path Signature – Clifford Geometric Product (hybrid_hybrid_sketches_hybr_hybrid_hybrid_m1904_s2.py)
  It supplies a Clifford geometric product that combines or manipulates path
  signatures approximated by B-spline basis functions.

The mathematical bridge is the Clifford geometric product applied to the
B-spline basis functions from Parent A, which constructs a multivector that
approximates the path signature. The geometric product from Parent B provides
a natural way to concatenate or manipulate these multivectors.

The module therefore contains:

1. `lead_lag_transform` – unchanged from Parent A.
2. `bspline_basis` – unchanged from Parent A.
3. `clifford_geometric_product` – computes the Clifford geometric product of two multivectors.
4. `hybrid_step` – a single training step that ties the three pieces together:
   transform a path, expand it with B-splines, compute a simple squared-error
   loss, obtain its gradient w.r.t. **W**, and update **W** with the fold-change
   aware Euler rule.

The implementation uses only the allowed standard-library modules and NumPy."""
import numpy as np
import math
import random
import sys
import pathlib

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead-lag transform: interleave (lead, lag) pairs of consecutive points in
    the path.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     np.zeros(d)])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], np.zeros(d)])
    return out

def bspline_basis(x, grid, k=3):
    """
    B-spline basis function.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    return B

# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
def multivector(x, d):
    """
    Create a multivector.
    """
    return np.zeros((d, len(x)))

def clifford_geometric_product(u, v):
    """
    Compute the Clifford geometric product of two multivectors.
    """
    d = u.shape[0]
    result = np.zeros((d, len(u[0])))
    for i in range(d):
        for j in range(d):
            result[i] += u[i] * v[j]
    return result

# ----------------------------------------------------------------------
# Hybrid building blocks
# ----------------------------------------------------------------------
def euler_weight_update(W, dw, gamma):
    """
    Update the weight matrix W using Euler integration.
    """
    W += gamma * dw
    return W

def fold_change_factor(delta):
    """
    Compute the fold-change factor.
    """
    return 1 + 0.1 * delta

def hybrid_step(path, W, grid, kappa, gamma):
    """
    Perform a single training step.
    """
    # Transform the path
    transformed_path = lead_lag_transform(path)

    # Expand the path with B-spline basis functions
    expanded_path = bspline_basis(transformed_path, grid)

    # Compute the loss
    loss = np.mean((expanded_path.dot(W)) ** 2)

    # Compute the gradient of the loss w.r.t. W
    dw = 2 * expanded_path.T.dot(expanded_path.dot(W))

    # Compute the fold-change factor
    delta = (loss - loss_old) / loss_old
    gamma = fold_change_factor(delta)

    # Update the weight matrix
    W = euler_weight_update(W, dw, gamma)

    return W

if __name__ == "__main__":
    # Smoke test
    import numpy as np

    path = np.random.rand(10, 2)
    W = np.random.rand(2, 10)
    grid = np.linspace(0, 1, 10)
    kappa = 0.1
    gamma = 0.1
    loss_old = 0

    W = hybrid_step(path, W, grid, kappa, gamma)
    print(W)