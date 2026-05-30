# DARWIN HAMMER — match 3356, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_label__m1580_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_tri_al_m1845_s2.py (gen5)
# born: 2026-05-29T23:49:24Z

"""
Hybrid Algorithm Fusion of Clifford Algebra and B-Spline Lead-Lag Transform

Parents:
- Parent A: `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_label__m1580_s3.py` – provides
  Clifford (geometric) algebra operations on multivectors.
- Parent B: `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_tri_al_m1845_s2.py` – provides
  B-spline basis functions and lead-lag transforms.

Mathematical Bridge:
The scalar (grade-0) part of the geometric product of two vectors in a Clifford
algebra equals their Euclidean dot product. We exploit this property to turn the
Clifford product into a *context-sensitive similarity* between feature vectors.
This similarity is then used to modulate the B-spline basis functions, yielding
a hybrid confidence that is aware of geometric context while retaining the
probabilistic rigor of the original labeling framework.
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
    """Return (sorted_blade, sign) after bubble-sorting index list."""
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
                # cancel the pair and restart scanning from the beginning
                lst.pop(j)
                lst.pop(j)  # second element shifts to j after first pop
                n -= 2
                i = -1  # will become 0 after i += 1 at loop bottom
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def geometric_product(vector_a, vector_b):
    """Compute the geometric product of two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    wedge_product = np.cross(vector_a, vector_b)
    return dot_product, wedge_product

# ----------------------------------------------------------------------
# B-Spline and Lead-Lag Transform Core
# ----------------------------------------------------------------------
def _augmented_knots(grid: np.ndarray, k: int) -> np.ndarray:
    """Create a clamped knot vector for a given grid and spline order."""
    grid = np.asarray(grid, dtype=float)
    if k < 1:
        raise ValueError("Spline order k must be >= 1")
    # Clamp ends with multiplicity k
    left = np.full(k, grid[0])
    right = np.full(k, grid[-1])
    return np.concatenate([left, grid, right])

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Evaluate B-spline basis functions of order k at positions x.
    Returns B where B[i, j] = B_j(x_i).
    The number of basis functions equals len(grid) + k - 1.
    """
    augmented_knots = _augmented_knots(grid, k)
    num_basis = len(grid) + k - 1
    num_points = len(x)
    basis = np.zeros((num_points, num_basis))
    for i, xi in enumerate(x):
        for j in range(num_basis):
            basis[i, j] = _bspline_basis(xi, augmented_knots, k, j)
    return basis

def _bspline_basis(xi: float, knots: np.ndarray, k: int, j: int) -> float:
    if k == 0:
        return 1.0 if knots[j] <= xi < knots[j + 1] else 0.0
    else:
        d1 = xi - knots[j]
        d2 = knots[j + k + 1] - xi
        if d1 == 0 and d2 == 0:
            return 0.0
        elif d1 == 0:
            return (d2 / (k * (knots[j + k + 1] - knots[j + 1]))) * _bspline_basis(xi, knots, k - 1, j + 1)
        elif d2 == 0:
            return (d1 / (k * (knots[j + k] - knots[j]))) * _bspline_basis(xi, knots, k - 1, j)
        else:
            return (d1 / (k * (knots[j + k] - knots[j]))) * _bspline_basis(xi, knots, k - 1, j) + \
                   (d2 / (k * (knots[j + k + 1] - knots[j + 1]))) * _bspline_basis(xi, knots, k - 1, j + 1)

def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Interleave lead-lag channels.
    Input: (T, d) array.
    Output: (2T-1, 2d) array.
    """
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
# Hybrid Operation
# ----------------------------------------------------------------------
def hybrid_operation(vector_a, vector_b, grid, k):
    dot_product, _ = geometric_product(vector_a, vector_b)
    similarity = 1 / (1 + math.exp(-dot_product))
    basis = bspline_basis(np.array([similarity]), grid, k)
    return basis

def modulate_bspline_basis(vector_a, vector_b, grid, k):
    dot_product, _ = geometric_product(vector_a, vector_b)
    similarity = 1 / (1 + math.exp(-dot_product))
    basis = bspline_basis(np.array([similarity]), grid, k)
    modulated_basis = basis * similarity
    return modulated_basis

# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    vector_a = np.random.rand(3)
    vector_b = np.random.rand(3)
    grid = np.linspace(0, 1, 10)
    k = 3
    basis = hybrid_operation(vector_a, vector_b, grid, k)
    modulated_basis = modulate_bspline_basis(vector_a, vector_b, grid, k)
    print(basis)
    print(modulated_basis)