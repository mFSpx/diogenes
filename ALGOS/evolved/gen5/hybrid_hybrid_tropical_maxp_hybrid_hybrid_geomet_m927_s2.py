# DARWIN HAMMER — match 927, survivor 2
# gen: 5
# parent_a: hybrid_tropical_maxplus_hybrid_hybrid_minimu_m190_s1.py (gen3)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s4.py (gen4)
# born: 2026-05-29T23:31:43Z

import numpy as np
import math
from typing import Dict, List, Tuple

def t_add(x, y):
    """Tropical addition: max(x, y). Broadcasts."""
    return np.maximum(x, y)

def t_mul(x, y):
    """Tropical multiplication: x + y. Broadcasts."""
    return np.add(x, y)

def t_matmul(A, B):
    """Tropical matrix multiply.

    C[i, j] = max_k( A[i, k] + B[k, j] )

    A: (m, p), B: (p, n) → C: (m, n).
    Handles -inf entries correctly via numpy broadcasting.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # broadcast: A[i, k, newaxis] + B[newaxis, k, j] then max over k
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

def t_polyval(coeffs, x):
    """Evaluate a tropical polynomial at x.

    Tropical polynomial: p(x) = coeffs[0] ⊕ (coeffs[1] ⊗ x) ⊕ ... ⊕ (coeffs[d] ⊗ d*x)
                               = max_i( coeffs[i] + i*x )

    coeffs: 1-D array of length d+1 (tropical coefficients, may include -inf).
    x     : scalar or array broadcastable against (d+1,).
    Returns same shape as x.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    # exponents [0, 1, ..., d] — tropical exponentiation = ordinary multiplication
    exponents = np.arange(len(coeffs), dtype=float)
    # shape: (d+1,) + x.shape
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents.reshape((-1,) + (1,) * x.ndim) * x
    return np.max(terms, axis=0)

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list.

    Duplicate indices cancel because e_i·e_i = 1.
    """
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
                # cancel pair
                del lst[j : j + 2]
                n -= 2
                i = -1  # restart outer loop
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(
    blade_a: frozenset, blade_b: frozenset
) -> Tuple[frozenset, int]:
    """Multiply two basis blades and return (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def clifford_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Clifford-geometric distance between two multivectors."""
    if a.shape != b.shape:
        raise ValueError("Multivectors must have the same shape")
    return math.sqrt(np.sum((a-b)**2))

def hybrid_voronoi(nodes: List[Tuple[float, float]], edges: List[Tuple[int, int]], root: Tuple[float, float]) -> Tuple[Dict[int, List[int]], List[Tuple[int, int, float]]]:
    """
    Build adjacency, compute Clifford-geometric edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list
    """
    adj = {}
    node_array = np.array(nodes)
    for i, node in enumerate(node_array):
        adj[i] = []
        for j, edge in enumerate(edges):
            if edge[0] == i:
                adj[i].append(edge[1])
                dist = clifford_distance(node_array[i], node_array[edge[1]])
                edges[j] = (edge[0], edge[1], dist)
    return adj, edges

def hybrid_tropical_voronoi(coeffs: np.ndarray, x: float, nodes: List[Tuple[float, float]], edges: List[Tuple[int, int]], root: Tuple[float, float]) -> Dict[int, List[Tuple[int, float]]]:
    """
    Evaluate a tropical polynomial at x and construct a Voronoi partition.

    Returns
    -------
    voronoi_partition : dict mapping node → list
    """
    tropical_polyval = t_polyval(coeffs, x)
    adj, edges = hybrid_voronoi(nodes, edges, root)
    voronoi_partition = {}
    node_array = np.array(nodes)
    for node in adj:
        voronoi_partition[node] = []
        for edge in edges:
            if edge[0] == node:
                dist = edge[2]
                voronoi_partition[node].append((edge[1], dist + tropical_polyval))
    return voronoi_partition

if __name__ == "__main__":
    coeffs = np.array([1, 2, 3])
    x = 2
    nodes = [(0, 0), (1, 0), (0, 1)]
    edges = [(0, 1), (0, 2), (1, 2)]
    root = (0, 0)
    voronoi_partition = hybrid_tropical_voronoi(coeffs, x, nodes, edges, root)
    print(voronoi_partition)