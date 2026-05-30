# DARWIN HAMMER — match 2998, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_sketches_hybr_hybrid_hybrid_hybrid_m1904_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m613_s1.py (gen5)
# born: 2026-05-29T23:47:03Z

"""
Hybrid Sketches and Path Signatures – Geometric Product Fusion

Parent A: hybrid_sketches_hybrid_hybrid_hybrid_m1904_s1.py
Parent B: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m613_s1.py

Mathematical bridge:
The mathematical interface between the two parents is based on the idea of 
encoding the sketch data into a multivector format, similar to the path 
signatures in the second parent. The first parent uses a hash-based count 
min sketch to summarize the input data. This count min sketch can be 
interpreted as a vector. By using the same vector-bivector encoding as 
the second parent, we can apply the geometric product to combine and 
manipulate these sketches.

This module provides three core functions to demonstrate the hybrid operation:
- hybrid_sketches_geometric_product: combines two sketches using the geometric product
- sketch_to_multivector: converts a count min sketch to a multivector
- multivector_to_sketch: converts a multivector back to a count min sketch
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import defaultdict
import hashlib

def count_min_sketch(items: list[str], width: int=64, depth: int=4) -> list[list[int]]:
    """
    Count min sketch implementation from the first parent.

    Args:
    items (list[str]): Input data to be summarized
    width (int): Width of the count min sketch
    depth (int): Depth of the count min sketch

    Returns:
    list[list[int]]: Count min sketch of the input data
    """
    table=[[0]*width for _ in range(depth)]
    for item in items:
        for d in range(depth): 
            table[d][int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(),16)%width]+=1
    return table

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     np.zeros(d)])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], np.zeros(d)])
    return out

def signature_level1(path):
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    return running.T @ increments               # (d, d)

def sketch_to_multivector(sketch: list[list[int]]) -> np.ndarray:
    """
    Convert a count min sketch to a multivector.

    Args:
    sketch (list[list[int]]): Count min sketch to be converted

    Returns:
    np.ndarray: Multivector representation of the input sketch
    """
    depth = len(sketch)
    width = len(sketch[0])
    multivector = np.zeros((depth, width), dtype=float)
    for d in range(depth):
        for w in range(width):
            multivector[d, w] = sketch[d][w]
    return multivector

def multivector_to_sketch(multivector: np.ndarray) -> list[list[int]]:
    """
    Convert a multivector back to a count min sketch.

    Args:
    multivector (np.ndarray): Multivector to be converted

    Returns:
    list[list[int]]: Count min sketch representation of the input multivector
    """
    depth, width = multivector.shape
    sketch = [[0]*width for _ in range(depth)]
    for d in range(depth):
        for w in range(width):
            sketch[d][w] = int(multivector[d, w])
    return sketch

def hybrid_sketches_geometric_product(sketch1: list[list[int]], sketch2: list[list[int]]) -> np.ndarray:
    """
    Combine two sketches using the geometric product.

    Args:
    sketch1 (list[list[int]]): First count min sketch
    sketch2 (list[list[int]]): Second count min sketch

    Returns:
    np.ndarray: Multivector representation of the combined sketches
    """
    multivector1 = sketch_to_multivector(sketch1)
    multivector2 = sketch_to_multivector(sketch2)
    combined_multivector = np.dot(multivector1, multivector2)
    return combined_multivector

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
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
    B[:, -1] = np.where(x == t[-1], 1.0, 0.0)
    return B

if __name__ == "__main__":
    items = ["item1", "item2", "item3"]
    sketch1 = count_min_sketch(items)
    sketch2 = count_min_sketch(items)
    combined_multivector = hybrid_sketches_geometric_product(sketch1, sketch2)
    print(combined_multivector)
    path = np.random.rand(10, 2)
    transformed_path = lead_lag_transform(path)
    print(transformed_path)
    signature = signature_level1(path)
    print(signature)