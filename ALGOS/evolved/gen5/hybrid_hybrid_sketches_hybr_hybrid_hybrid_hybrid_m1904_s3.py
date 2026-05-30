# DARWIN HAMMER — match 1904, survivor 3
# gen: 5
# parent_a: hybrid_sketches_hybrid_hybrid_hybrid_m302_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_path_s_geometric_product_m161_s1.py (gen3)
# born: 2026-05-29T23:39:34Z

"""
Hybrid Path Signature – Clifford Geometric Product

Parent A: hybrid_sketches_hybrid_hybrid_hybrid_m302_s1.py
Parent B: hybrid_hybrid_hybrid_path_s_geometric_product_m161_s1.py

Mathematical bridge
------------------
A path signature lives in the (truncated) tensor algebra of a path and its
level‑1 and level‑2 components can be identified with a vector (grade‑1) and a
bivector (grade‑2) of a Euclidean Clifford algebra Cl(n,0).  By mapping the
approximated iterated integrals obtained from the feature extraction of the
parent‑A code onto the corresponding multivector grades, we obtain a single
`Multivector` that encodes the signature.  The geometric product from the
parent‑B code then provides a natural way to concatenate signatures (the
signature of a concatenated path equals the geometric product of the individual
signatures).  The hybrid module therefore:

* extracts deterministic scalar‑vector features from a text (parent A),
* builds a multivector containing scalar, vector and bivector parts that
  approximate the level‑0/1/2 signature,
* uses the Clifford geometric product to combine, rotate or otherwise
  manipulate these signatures (parent B).

The three core functions below showcase this fusion.
"""

import numpy as np
import math
import random
import sys
import pathlib

class Multivector:
    def __init__(self, scalar, vector, bivector):
        self.scalar = scalar
        self.vector = vector
        self.bivector = bivector

def lead_lag_transform(path):
    """Lead‑lag embedding of a discrete path.

    Input: (T, d) ndarray.
    Output: (2T‑1, 2d) ndarray where successive points are duplicated and
    interleaved as in the classic lead‑lag transform.
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

def count_min_sketch(items: Iterable[str], width: int=64, depth: int=4) -> list[list[int]]:
    table=[[0]*width for _ in range(depth)]
    for item in items:
        for d in range(depth): 
            table[d][int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(),16)%width]+=1
    return table

def geometric_product(m1: Multivector, m2: Multivector):
    result = Multivector(
        m1.scalar*m2.scalar,
        m1.vector*m2.scalar + m1.scalar*m2.vector + np.cross(m1.vector, m2.vector),
        m1.bivector*m2.scalar + m1.scalar*m2.bivector + m1.vector*np.dot(m1.vector, m2.bivector) - m1.bivector*np.dot(m1.vector, m2.vector)
    )
    return result

def hybrid_banded_path_signature(x, grid, k=3, banded_width=5):
    banded_x = np.roll(x, -banded_width, axis=0)
    banded_x[banded_width:] = x[:-banded_width]

    B = bspline_basis(banded_x, grid, k=k)

    return np.sum(B * banded_x, axis=1)

def fused_hybrid(items: Iterable[str], grid, k=3, banded_width=5, width: int=64, depth: int=4):
    mvs = []
    for item in items:
        mv = Multivector(1.0, hybrid_banded_path_signature([random.random() for _ in range(10)], grid, k, banded_width), count_min_sketch([item], width, depth))
        mvs.append(mv)
    
    product = mvs[0]
    for mv in mvs[1:]:
        product = geometric_product(product, mv)

    return product.scalar, product.vector, product.bivector

if __name__ == "__main__":
    grid = np.linspace(0, 10, 100)
    items = ["apple", "banana", "orange"]
    fused_hybrid(items, grid)