# DARWIN HAMMER — match 3390, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_model__m1308_s2.py (gen4)
# parent_b: hybrid_hybrid_sketches_hybr_hybrid_hybrid_hybrid_m1904_s2.py (gen5)
# born: 2026-05-29T23:49:51Z

"""
Hybrid algorithm combining the core mathematics of two parent algorithms:
- Parent A: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_model__m1308_s2.py
- Parent B: hybrid_hybrid_sketches_hybr_hybrid_hybrid_hybrid_m1904_s2.py

The mathematical bridge between the two parent algorithms lies in the integration of
the B-spline basis function from Parent A with the Clifford geometric product
from Parent B. Specifically, we can use the B-spline basis to construct a
multivector that approximates the path signature, and then apply the geometric
product to combine or manipulate these signatures.

This hybrid module therefore:
- extracts deterministic scalar-vector features from a path (Parent A),
- builds a multivector containing scalar, vector and bivector parts that
  approximate the level-0/1/2 signature using B-spline basis functions,
- uses the Clifford geometric product to combine, rotate or otherwise
  manipulate these signatures (Parent B).
"""

import numpy as np
import math
import random
import sys
import pathlib

def lead_lag_transform(path):
    """
    Lead-lag transform: interleave (lead, lag) values in a path.
    
    Parameters:
    path (np.ndarray): Input path.
    
    Returns:
    np.ndarray: Transformed path.
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
    Compute the B-spline basis for a given set of points and grid.
    
    Parameters:
    x (np.ndarray): Input points.
    grid (np.ndarray): Grid points.
    k (int): B-spline degree (default: 3).
    
    Returns:
    np.ndarray: B-spline basis values.
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
    B[:, -1] = np.where(x >= t[-1], 1.0, 0.0)
    return B

def geometric_product(a, b):
    """
    Compute the Clifford geometric product of two vectors.
    
    Parameters:
    a (np.ndarray): First vector.
    b (np.ndarray): Second vector.
    
    Returns:
    np.ndarray: Geometric product.
    """
    return a * b + np.cross(a, b)

def hybrid_step(path, grid, k=3):
    """
    Perform a single hybrid step, combining the lead-lag transform, B-spline basis,
    and geometric product.
    
    Parameters:
    path (np.ndarray): Input path.
    grid (np.ndarray): Grid points.
    k (int): B-spline degree (default: 3).
    
    Returns:
    np.ndarray: Result of the hybrid step.
    """
    transformed_path = lead_lag_transform(path)
    basis = bspline_basis(np.arange(len(transformed_path)), grid, k)
    result = np.zeros_like(transformed_path)
    for i in range(len(transformed_path)):
        result[i] = geometric_product(transformed_path[i], basis[i])
    return result

if __name__ == "__main__":
    # Smoke test
    path = np.random.rand(10, 3)
    grid = np.linspace(0, 1, 5)
    result = hybrid_step(path, grid)
    print(result)