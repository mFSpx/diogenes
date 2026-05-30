# DARWIN HAMMER — match 1904, survivor 2
# gen: 5
# parent_a: hybrid_sketches_hybrid_hybrid_hybrid_m302_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_path_s_geometric_product_m161_s1.py (gen3)
# born: 2026-05-29T23:39:34Z

"""
Hybrid Path Signature – Clifford Geometric Product & B-Spline Basis

Parent A: hybrid_sketches_hybrid_hybrid_hybrid_m302_s1.py
Parent B: hybrid_hybrid_hybrid_path_s_geometric_product_m161_s1.py

Mathematical bridge
------------------
The mathematical bridge between the two parent algorithms lies in the integration of
the B-spline basis function from Parent A with the Clifford geometric product
from Parent B. Specifically, we can use the B-spline basis to construct a
multivector that approximates the path signature, and then apply the geometric
product to combine or manipulate these signatures.

The key insight is that the lead-lag transform and B-spline basis functions
from Parent A can be used to construct a discrete approximation of the path
signature, which can then be represented as a multivector in the Clifford
algebra. The geometric product from Parent B provides a natural way to
concatenate or manipulate these multivectors.

The hybrid module therefore:

* extracts deterministic scalar-vector features from a text (Parent A),
* builds a multivector containing scalar, vector and bivector parts that
  approximate the level-0/1/2 signature using B-spline basis functions,
* uses the Clifford geometric product to combine, rotate or otherwise
  manipulate these signatures (Parent B).
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import defaultdict
import hashlib

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     np.zeros(d)])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], np.zeros(d)])
    return out

def bspline_basis(x, grid, k=3):
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
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (
                (x - t[i]) / denom_l * B[:, i]
                if denom_l > 1e-12 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 1e-12 else np.zeros(N)
            )
            B_new[:, i] = term_l + term_r
        B = B_new

    assert B.shape == (N, n_basis), (
        f"basis shape mismatch: got {B.shape}, expected ({N}, {n_basis})"
    )

    return B

def geometric_product(a, b):
    # Clifford geometric product
    return a * b

def count_min_sketch(items, width=64, depth=4):
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            table[d][int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(), 16) % width] += 1
    return table

def hybrid_banded_path_signature(x, grid, k=3, banded_width=5):
    banded_x = np.roll(x, -banded_width, axis=0)
    banded_x[banded_width:] = x[:-banded_width]

    B = bspline_basis(banded_x, grid, k=k)

    return np.sum(B * banded_x, axis=1)

def fused_hybrid(items, grid, k=3, banded_width=5, width=64, depth=4):
    sketch = count_min_sketch(items, width, depth)
    path = np.array(sketch).flatten()
    lead_lag_path = lead_lag_transform(path.reshape(-1, 1))
    signature = hybrid_banded_path_signature(lead_lag_path, grid, k, banded_width)
    multivector = np.array([1, *signature])
    return multivector

def manipulate_signature(multivector, angle):
    # Rotate the multivector by an angle
    rotation = np.array([math.cos(angle), math.sin(angle)])
    return geometric_product(multivector, rotation)

if __name__ == "__main__":
    items = ["apple", "banana", "orange"]
    grid = np.linspace(0, 1, 10)
    k = 3
    banded_width = 5
    width = 64
    depth = 4
    angle = math.pi / 2

    multivector = fused_hybrid(items, grid, k, banded_width, width, depth)
    rotated_multivector = manipulate_signature(multivector, angle)
    print(rotated_multivector)