# DARWIN HAMMER — match 2747, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s1.py (gen3)
# parent_b: path_signature.py (gen0)
# born: 2026-05-29T23:45:50Z

import numpy as np
import random
import math
import sys
import pathlib

"""
This module represents a mathematical fusion of 'hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s4.py' 
and 'path_signature.py'. The bridge between the two is the integration of the sheaf structure from the former 
with the path signature framework of the latter. Specifically, the lead-lag transform from 'path_signature.py' 
is used to encode causality in the query vectors used in the sheaf, influencing the restriction maps in the sheaf.
The path signature tensor from 'path_signature.py' is represented as sections in the sheaf, capturing path geometry.
"""

class HybridSheaf:
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

    def lead_lag_transform(self, query_vector):
        """Lead-lag transform: interleave (lead, lag) channels for causality encoding.

        query_vector: (T, d). Returns (2T-1, 2d) interleaved lead-lag path.
        """
        T, d = query_vector.shape
        out = np.empty((2 * T - 1, 2 * d), dtype=float)
        for t in range(T - 1):
            out[2 * t]     = np.concatenate([query_vector[t],     query_vector[t]])
            out[2 * t + 1] = np.concatenate([query_vector[t + 1], query_vector[t]])
        out[2 * (T - 1)] = np.concatenate([query_vector[T - 1], query_vector[T - 1]])
        return out

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
    query_vector = sheaf.lead_lag_transform(sheaf._sections[query_node])
    return energy(query_vector, M, beta)

def update_restriction(sheaf, edge, M, beta=1.0):
    u, v = edge
    src_map, dst_map = sheaf._restrictions[(u, v)]
    query_vector = sheaf.lead_lag_transform(sheaf._sections[u])
    scores = beta * (M @ query_vector)
    lse_term = _lse(scores) / beta
    quadratic_term = 0.5 * query_vector @ query_vector
    return -lse_term + quadratic_term

def path_signature_tensor(sheaf, path):
    """Path signature tensor: captures path geometry.

    sheaf: HybridSheaf object.
    path: (T, d). Returns (d, d) path signature tensor.
    """
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    # S2[i,j] = sum_t running[t,i] * increments[t,j]
    return running.T @ increments               # (d, d)

def set_path_signature_section(sheaf, node, path):
    """Set path signature section in sheaf.

    sheaf: HybridSheaf object.
    node: node in sheaf where section will be set.
    path: (T, d) input path.
    """
    sheaf.set_section(node, path_signature_tensor(sheaf, path))

def smoke_test():
    # Create a HybridSheaf object
    node_dims = {0: 3, 1: 3, 2: 3}
    edges = [(0, 1), (1, 2)]
    sheaf = HybridSheaf(node_dims, edges)

    # Set some sections in the sheaf
    sheaf.set_section(0, np.random.rand(10, 3))
    sheaf.set_section(1, np.random.rand(10, 3))
    sheaf.set_section(2, np.random.rand(10, 3))

    # Set a path signature section in the sheaf
    path = np.random.rand(10, 3)
    set_path_signature_section(sheaf, 0, path)

    # Run the hybrid energy function
    M = np.random.rand(20, 6)
    beta = 0.5
    energy_value = hybrid_energy(sheaf, 0, M, beta)
    print("Hybrid energy:", energy_value)

if __name__ == "__main__":
    smoke_test()