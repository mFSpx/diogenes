# DARWIN HAMMER — match 2747, survivor 9
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s1.py (gen3)
# parent_b: path_signature.py (gen0)
# born: 2026-05-29T23:45:50Z

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
        src_map = np.asarray(src_map, dtype=float)
        dst_map = np.asarray(dst_map, dtype=float)
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (src_map, dst_map)

    def set_section(self, node, value):
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node):
        return self._sections[node]

    def get_restriction(self, edge):
        return self._restrictions[edge]

def _softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()

def _lse(z):
    z = np.asarray(z, dtype=float)
    m = np.max(z)
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

def signature_level1(path):
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          
    running    = path[:-1] - path[0]            
    S2 = running.T @ increments                 
    return S2.reshape(-1)                       

def signature(path, depth=2, lead_lag=False):
    path = np.asarray(path, dtype=float)
    if lead_lag:
        path = lead_lag_transform(path)

    if depth < 1:
        raise ValueError("depth must be >= 1")
    signatures = []

    S1 = signature_level1(path)
    signatures.append(S1.reshape(-1))

    if depth == 1:
        return signatures

    S2 = signature_level2(path)
    signatures.append(S2)

    prev = signatures[-1]          
    d = path.shape[1]
    for k in range(3, depth + 1):
        incr = np.diff(path, axis=0)          
        accum = np.zeros(d ** k, dtype=float)
        for t in range(incr.shape[0]):
            prod = np.tensordot(prev.reshape(d ** (k - 1)), incr[t], axes=0)
            accum += prod.reshape(-1)
        signatures.append(accum)
        prev = accum
    return signatures

def sheaf_walk_path(sheaf, start_node, edge_sequence):
    node = start_node
    dims = sheaf.node_dims[node]
    path = [sheaf.get_section(node)]
    for edge in edge_sequence:
        u, v = edge
        if u != node:
            raise ValueError(f"Edge {edge} does not continue from current node {node}")
        node = v
        if sheaf.node_dims[node] != dims:
            raise ValueError("All nodes in the walk must have the same dimension for signature")
        path.append(sheaf.get_section(node))
    return np.stack(path, axis=0)   

def sheaf_walk_signature(sheaf, start_node, edge_sequence, depth=2, lead_lag=False):
    path = sheaf_walk_path(sheaf, start_node, edge_sequence)
    sig_levels = signature(path, depth=depth, lead_lag=lead_lag)
    return np.concatenate(sig_levels)

def signature_energy(sheaf, start_node, edge_sequence, M, beta=1.0,
                    depth=2, lead_lag=False):
    sig_vec = sheaf_walk_signature(sheaf, start_node, edge_sequence,
                                   depth=depth, lead_lag=lead_lag)
    return energy(sig_vec, M, beta=beta)

def update_restriction_via_signature(sheaf, edge, M, beta=1.0,
                                    depth=2, lead_lag=False, temperature=0.1):
    u, v = edge
    src_map, dst_map = sheaf.get_restriction(edge)
    scores = []
    for i in range(src_map.shape[0]):
        for j in range(dst_map.shape[0]):
            new_src_map = src_map.copy()
            new_dst_map = dst_map.copy()
            new_src_map[i] = new_src_map[i] + np.random.normal(0, 0.1, size=new_src_map.shape[1])
            new_dst_map[j] = new_dst_map[j] + np.random.normal(0, 0.1, size=new_dst_map.shape[1])
            sheaf.set_restriction(edge, new_src_map, new_dst_map)
            scores.append(-signature_energy(sheaf, u, [(u, v)], M, beta, depth, lead_lag))
            sheaf.set_restriction(edge, src_map, dst_map)
    scores = np.array(scores)
    prob = np.exp(scores / temperature) / np.sum(np.exp(scores / temperature))
    idx = np.random.choice(len(scores), p=prob)
    i, j = divmod(idx, src_map.shape[0])
    new_src_map = src_map.copy()
    new_dst_map = dst_map.copy()
    new_src_map[i] = new_src_map[i] + np.random.normal(0, 0.1, size=new_src_map.shape[1])
    new_dst_map[j] = new_dst_map[j] + np.random.normal(0, 0.1, size=new_dst_map.shape[1])
    sheaf.set_restriction(edge, new_src_map, new_dst_map)

def improved_update_restriction_via_signature(sheaf, edge, M, beta=1.0,
                                    depth=2, lead_lag=False, temperature=0.1, iterations=100):
    u, v = edge
    src_map, dst_map = sheaf.get_restriction(edge)
    for _ in range(iterations):
        scores = []
        for i in range(src_map.shape[0]):
            for j in range(dst_map.shape[0]):
                new_src_map = src_map.copy()
                new_dst_map = dst_map.copy()
                new_src_map[i] = new_src_map[i] + np.random.normal(0, 0.1, size=new_src_map.shape[1])
                new_dst_map[j] = new_dst_map[j] + np.random.normal(0, 0.1, size=new_dst_map.shape[1])
                sheaf.set_restriction(edge, new_src_map, new_dst_map)
                scores.append(-signature_energy(sheaf, u, [(u, v)], M, beta, depth, lead_lag))
                sheaf.set_restriction(edge, src_map, dst_map)
        scores = np.array(scores)
        prob = np.exp(scores / temperature) / np.sum(np.exp(scores / temperature))
        idx = np.random.choice(len(scores), p=prob)
        i, j = divmod(idx, src_map.shape[0])
        new_src_map = src_map.copy()
        new_dst_map = dst_map.copy()
        new_src_map[i] = new_src_map[i] + np.random.normal(0, 0.1, size=new_src_map.shape[1])
        new_dst_map[j] = new_dst_map[j] + np.random.normal(0, 0.1, size=new_dst_map.shape[1])
        sheaf.set_restriction(edge, new_src_map, new_dst_map)

# Example usage:
sheaf = Sheaf([2, 2], [(0, 1)])
sheaf.set_restriction((0, 1), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]))
sheaf.set_section(0, np.array([1, 0]))
sheaf.set_section(1, np.array([0, 1]))

M = np.array([[1, 0], [0, 1]])

improved_update_restriction_via_signature(sheaf, (0, 1), M)