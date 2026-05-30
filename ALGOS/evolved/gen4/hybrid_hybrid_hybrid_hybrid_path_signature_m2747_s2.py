# DARWIN HAMMER — match 2747, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s1.py (gen3)
# parent_b: path_signature.py (gen0)
# born: 2026-05-29T23:45:50Z

"""
This module represents a mathematical fusion of 'hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s1.py' 
and 'path_signature.py'. The bridge between the two is the integration of the sheaf structure 
from the former with the path signature framework from the latter. Specifically, 
the path signature is used to compute a vector representation of the paths in the sheaf, 
which are then used as query vectors in the Dense Associative Memory (DAM) framework. 
The energy function from the DAM is used to compute the similarity between the query vectors 
and the stored patterns, influencing the restriction maps in the sheaf.
"""

import numpy as np
import random
import math
import sys
import pathlib

class Sheaf:
    def __init__(self, node_dims, edges):
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions = {}
        self._sections = {}

    def set_restriction(self, edge, src_map, dst_map):
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    def set_section(self, node, value):
        self._sections[node] = np.asarray(value, dtype=float)

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path):
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    # S2[i,j] = sum_t running[t,i] * increments[t,j]
    return running.T @ increments               # (d, d)

def signature(path, depth=3):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    increments = np.diff(path, axis=0)          # (T-1, d)

    # Initialize accumulators: level k holds a flat array of size d^k
    # We keep them as flat numpy arrays and rebuild shape on output
    accum = [np.zeros((d**k,)) for k in range(1, depth+1)]
    accum[0] = path[-1] - path[0]

    for t in range(1, T):
        for k in range(1, depth):
            accum[k] += accum[k-1] * increments[t-1]

    return accum

def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def _lse(z):
    m = z.max()
    return m + np.log(np.exp(z - m).sum())

def energy(xi, M, beta=1.0):
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    scores = beta * (M @ xi)
    lse_term = _lse(scores) / beta
    quadratic_term = 0.5 * xi @ xi
    return -lse_term + quadratic_term

def hybrid_energy(sheaf, query_node, M, beta=1.0):
    query_vector = sheaf._sections[query_node]
    return energy(query_vector, M, beta)

def update_restriction(sheaf, edge, M, beta=1.0):
    u, v = edge
    src_map, dst_map = sheaf._restrictions[(u, v)]
    query_vector = sheaf._sections[u]
    scores = beta * (M @ query_vector)
    lse_term = _lse(scores) / beta
    quadratic_term = 0.5 * query_vector @ query_vector
    return -lse_term + quadratic_term

def path_signature_energy(sheaf, query_node, path, M, beta=1.0):
    query_vector = signature(path)
    return energy(query_vector, M, beta)

def update_path_signature_restriction(sheaf, edge, path, M, beta=1.0):
    u, v = edge
    src_map, dst_map = sheaf._restrictions[(u, v)]
    query_vector = signature(path)
    scores = beta * (M @ query_vector)
    lse_term = _lse(scores) / beta
    quadratic_term = 0.5 * query_vector @ query_vector
    return -lse_term + quadratic_term

if __name__ == "__main__":
    node_dims = [10, 10]
    edges = [(0, 1)]
    sheaf = Sheaf(node_dims, edges)
    sheaf.set_restriction(edges[0], np.random.rand(10, 10), np.random.rand(10, 10))
    sheaf.set_section(0, np.random.rand(10))
    path = np.random.rand(10, 10)
    M = np.random.rand(10, 10)
    print(hybrid_energy(sheaf, 0, M))
    print(path_signature_energy(sheaf, 0, path, M))
    print(update_restriction(sheaf, edges[0], M))
    print(update_path_signature_restriction(sheaf, edges[0], path, M))