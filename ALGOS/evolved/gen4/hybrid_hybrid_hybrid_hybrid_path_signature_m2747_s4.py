# DARWIN HAMMER — match 2747, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s1.py (gen3)
# parent_b: path_signature.py (gen0)
# born: 2026-05-29T23:45:50Z

"""
This module represents a mathematical fusion of 'hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s1.py' 
and 'path_signature.py'. The bridge between the two is the integration of the lead-lag transform 
from the latter with the sheaf structure and energy-based memory retrieval of the former. 
Specifically, the lead-lag transformed path is used to generate query vectors for the Dense 
Associative Memory (DAM) framework, while the sheaf's node values are used as stored patterns. 
The energy function from the DAM is used to compute the similarity between the query vectors 
and the stored patterns, influencing the restriction maps in the sheaf.

The mathematical interface is found by interpreting the lead-lag transformed path as a sequence 
of query vectors, and using the sheaf's node values as a set of stored patterns. The energy 
function is then used to compute the similarity between the query vectors and the stored patterns.

The resulting hybrid system combines the ability to capture path geometry and causality 
encoding from the lead-lag transform with the ability to perform energy-based memory retrieval 
and sheaf-based reasoning.
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


def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


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


def path_to_sheaf(path, sheaf):
    lead_lag_path = lead_lag_transform(path)
    for i in range(lead_lag_path.shape[0]):
        sheaf.set_section(i, lead_lag_path[i])


def compute_similarity(sheaf, M):
    similarities = []
    for node in sheaf._sections:
        similarity = hybrid_energy(sheaf, node, M)
        similarities.append(similarity)
    return np.array(similarities)


def main():
    # Create a sheaf with 2 nodes and 1 edge
    sheaf = Sheaf(node_dims=[2, 2], edges=[(0, 1)])

    # Set restrictions and sections
    sheaf.set_restriction((0, 1), np.random.rand(10, 2), np.random.rand(10, 2))
    sheaf.set_section(0, np.random.rand(2))
    sheaf.set_section(1, np.random.rand(2))

    # Create a random path
    path = np.random.rand(10, 2)

    # Convert path to sheaf
    path_to_sheaf(path, sheaf)

    # Create a random matrix M
    M = np.random.rand(2, 2)

    # Compute similarities
    similarities = compute_similarity(sheaf, M)
    print(similarities)


if __name__ == "__main__":
    main()