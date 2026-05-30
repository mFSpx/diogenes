# DARWIN HAMMER — match 302, survivor 0
# gen: 4
# parent_a: sketches.py (gen0)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_path_signatur_m56_s0.py (gen3)
# born: 2026-05-29T23:28:15Z

"""
Hybrid algorithm fusing count-min sketch (sketches.py) and hybrid banded path signature (hybrid_hybrid_hybrid_bandit_hybrid_path_signatur_m56_s0.py).
The mathematical interface is established through the use of hashing and B-spline basis functions to encode and process high-dimensional data.
"""

import numpy as np
import hashlib
from collections import defaultdict
from typing import Iterable

def lead_lag_transform(path):
    """
    Lead-lag transform: interleave (lead, lag) channels for causality encoding.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(x, grid, k=3):
    """
    Evaluate B-spline basis functions of order k at positions x.
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
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (
                (x - t[i]) / denom_l * B[:, i]
                if denom_l > 0 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 0 else np.zeros(N)
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

def hybrid_banded_path_signature(x, grid, k=3, banded_width=5):
    """
    Hybrid banded path signature computation using B-spline basis functions.
    """
    banded_x = np.roll(x, -banded_width, axis=0)
    banded_x[banded_x == 0.0] = np.nan

    B = bspline_basis(banded_x, grid, k=k)

    # Weighted sum of path signature terms with B-spline coefficients
    return np.sum(B * banded_x, axis=1)

def fused_hybrid(items: Iterable[str], grid, k=3, banded_width=5, width: int=64, depth: int=4) -> np.ndarray:
    """
    Fused hybrid algorithm combining count-min sketch and hybrid banded path signature.
    """
    sketch = count_min_sketch(items, width, depth)
    sketch_values = np.array(sketch).flatten()
    return hybrid_banded_path_signature(sketch_values.reshape(-1, 1), grid, k, banded_width)

def lead_lag_fusion(items: Iterable[str], grid, k=3, banded_width=5, width: int=64, depth: int=4) -> np.ndarray:
    """
    Lead-lag fusion of count-min sketch and hybrid banded path signature.
    """
    sketch = count_min_sketch(items, width, depth)
    sketch_values = np.array(sketch).flatten()
    lead_lag_sketch = lead_lag_transform(sketch_values.reshape(-1, 1))
    return hybrid_banded_path_signature(lead_lag_sketch, grid, k, banded_width)

def minhash_lsh_fusion(docs: dict[str,set[str]], grid, k=3, banded_width=5) -> dict[str,np.ndarray]:
    """
    MinHash LSH fusion with hybrid banded path signature.
    """
    buckets=defaultdict(list)
    for doc_id, shingles in docs.items():
        key=min((hashlib.sha1(s.encode()).hexdigest()[:6] for s in shingles), default='empty')
        buckets[key].append(doc_id)
    fusion_buckets = {}
    for key, doc_ids in buckets.items():
        doc_sketch = [int(key, 16) for key in key]
        fusion_buckets[key] = hybrid_banded_path_signature(np.array(doc_sketch).reshape(-1, 1), grid, k, banded_width)
    return fusion_buckets

if __name__ == "__main__":
    items = ["item1", "item2", "item3"]
    grid = np.linspace(0, 1, 10)
    print(fused_hybrid(items, grid))
    print(lead_lag_fusion(items, grid))
    docs = {"doc1": {"shingle1", "shingle2"}, "doc2": {"shingle3", "shingle4"}}
    print(minhash_lsh_fusion(docs, grid))