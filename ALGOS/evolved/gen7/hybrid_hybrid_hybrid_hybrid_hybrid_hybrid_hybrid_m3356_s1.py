# DARWIN HAMMER — match 3356, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_label__m1580_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_tri_al_m1845_s2.py (gen5)
# born: 2026-05-29T23:49:24Z

"""
Hybrid Algorithm Fusion of Clifford Algebra and B-spline Lead-Lag Transform

Parents:
- Parent A: `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_label__m1580_s3.py` 
  – provides Clifford (geometric) algebra operations on multivectors.
- Parent B: `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_tri_al_m1845_s2.py` 
  – provides B-spline basis functions and lead-lag transform.

Mathematical Bridge:
The scalar (grade-0) part of the geometric product of two vectors in a Clifford
algebra equals their Euclidean dot product. We use this property to turn the
Clifford product into a *context-sensitive similarity* between feature vectors.
This similarity then modulates the B-spline basis functions, yielding a hybrid
representation that is aware of geometric context while retaining the
flexibility of the B-spline framework.

The lead-lag transform from Parent B is used to create a time-series
representation of the feature vectors, which are then fed into the Clifford
algebra operations from Parent A. The resulting context-sensitive similarities
are used to modulate the B-spline basis functions, creating a hybrid system
that combines the strengths of both parents.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
from pathlib import Path
import math
import random
import sys

# ----------------------------------------------------------------------
# Clifford Algebra Core
# ----------------------------------------------------------------------
def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)  
                n -= 2
                i = -1  
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def geometric_product(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    wedge_product = np.cross(vector_a, vector_b)
    return dot_product, wedge_product

# ----------------------------------------------------------------------
# B-spline Basis and Lead-Lag Transform
# ----------------------------------------------------------------------
def _augmented_knots(grid: np.ndarray, k: int) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if k < 1:
        raise ValueError("Spline order k must be >= 1")
    left = np.full(k, grid[0])
    right = np.full(k, grid[-1])
    return np.concatenate([left, grid, right])

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    augmented_knots = _augmented_knots(grid, k)
    n_basis = len(grid) + k - 1
    basis = np.zeros((len(x), n_basis))
    for i, xi in enumerate(x):
        for j in range(n_basis):
            basis[i, j] = _bspline_basis(xi, augmented_knots, k, j)
    return basis

def _bspline_basis(xi, knots, k, j):
    if k == 0:
        return 1.0 if knots[j] <= xi < knots[j + 1] else 0.0
    else:
        a = (xi - knots[j]) / (knots[j + k] - knots[j])
        b = (knots[j + k + 1] - xi) / (knots[j + k + 1] - knots[j + 1])
        return a * _bspline_basis(xi, knots, k - 1, j) + b * _bspline_basis(xi, knots, k - 1, j + 1)

def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2-D array (time × dimension)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

# ----------------------------------------------------------------------
# Hybrid System
# ----------------------------------------------------------------------
def hybrid_representation(path: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    lead_lag_path = lead_lag_transform(path)
    T, d = lead_lag_path.shape
    similarities = np.zeros(T)
    for t in range(T):
        vector_a = lead_lag_path[t, :d // 2]
        vector_b = lead_lag_path[t, d // 2:]
        dot_product, _ = geometric_product(vector_a, vector_b)
        similarities[t] = dot_product
    modulated_basis = bspline_basis(np.arange(T), grid, k)
    hybrid_representation = modulated_basis * similarities[:, np.newaxis]
    return hybrid_representation

# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    path = np.random.rand(10, 4)
    grid = np.linspace(0, 10, 10)
    hybrid_repr = hybrid_representation(path, grid)
    print(hybrid_repr.shape)