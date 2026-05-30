# DARWIN HAMMER — match 159, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s1.py (gen3)
# parent_b: hybrid_ternary_lens_router_hybrid_decision_hygi_m44_s1.py (gen2)
# born: 2026-05-29T23:27:19Z

import numpy as np
import math
import random
import sys
from collections import Counter

class Sheaf:
    def __init__(self, node_dims, edges):
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
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
        if node not in self.node_dims:
            raise KeyError(f"Node {node} not defined in sheaf")
        vec = np.asarray(value, dtype=float)
        if vec.shape[0] != self.node_dims[node]:
            raise ValueError("Section vector length must match node dimension")
        self._sections[node] = vec

    def get_section(self, node):
        return self._sections[node]

    def get_restriction(self, edge):
        return self._restrictions[edge]

def generate_ternary_vector(dim, seed=None):
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 0, 1], size=dim)

def shannon_entropy(vector):
    counts = Counter(vector.tolist())
    total = len(vector)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / total
        entropy -= p * math.log(p, 2)  
    return entropy

def _logsumexp(z):
    m = np.max(z)
    return m + np.log(np.exp(z - m).sum())

def dam_energy(xi, M, beta=1.0):
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    scores = beta * (M @ xi)
    lse = _logsumexp(scores) / beta
    quad = 0.5 * xi @ xi
    return -lse + quad

def hybrid_energy(sheaf, query_node, M, base_beta=1.0):
    vec = sheaf.get_section(query_node)
    if not np.all(np.isin(vec, [-1, 0, 1])):
        raise ValueError("Section vector must be ternary for entropy scaling")
    H = shannon_entropy(vec)
    beta = base_beta * (1.0 + H)   
    return dam_energy(vec, M, beta)

def update_restriction_entropy(sheaf, edge, M, base_beta=1.0, lr=0.01):
    u, v = edge
    src_map, dst_map = sheaf.get_restriction(edge)
    src_vec = sheaf.get_section(u)

    H = shannon_entropy(src_vec)
    beta = base_beta * (1.0 + H)

    scores = beta * (M @ src_vec)
    probs = np.exp(scores - np.max(scores))
    probs /= probs.sum()
    grad = src_vec - M.T @ probs

    delta_src = -lr * np.outer(grad, np.ones(src_map.shape[0]))
    delta_dst = -lr * np.outer(grad, np.ones(dst_map.shape[0]))

    sheaf._restrictions[(u, v)] = (src_map + delta_src, dst_map + delta_dst)

def route_and_update(sheaf, M, base_beta=1.0, lr=0.01):
    energies = {}
    for edge in sheaf.edges:
        u, _ = edge
        new_section = generate_ternary_vector(sheaf.node_dims[u])
        sheaf.set_section(u, new_section)

        E = hybrid_energy(sheaf, u, M, base_beta)
        energies[edge] = E

        update_restriction_entropy(sheaf, edge, M, base_beta, lr)

    return energies

def improved_hybrid_energy(sheaf, query_node, M, base_beta=1.0, alpha=0.5):
    vec = sheaf.get_section(query_node)
    if not np.all(np.isin(vec, [-1, 0, 1])):
        raise ValueError("Section vector must be ternary for entropy scaling")
    H = shannon_entropy(vec)
    beta = base_beta * (1.0 + H)   
    energy = dam_energy(vec, M, beta)
    return energy + alpha * H

def improved_update_restriction_entropy(sheaf, edge, M, base_beta=1.0, lr=0.01, alpha=0.5):
    u, v = edge
    src_map, dst_map = sheaf.get_restriction(edge)
    src_vec = sheaf.get_section(u)

    H = shannon_entropy(src_vec)
    beta = base_beta * (1.0 + H)

    scores = beta * (M @ src_vec)
    probs = np.exp(scores - np.max(scores))
    probs /= probs.sum()
    grad = src_vec - M.T @ probs

    delta_src = -lr * np.outer(grad, np.ones(src_map.shape[0]))
    delta_dst = -lr * np.outer(grad, np.ones(dst_map.shape[0]))

    sheaf._restrictions[(u, v)] = (src_map + delta_src, dst_map + delta_dst)

    return H

def improved_route_and_update(sheaf, M, base_beta=1.0, lr=0.01, alpha=0.5):
    energies = {}
    total_entropy = 0
    for edge in sheaf.edges:
        u, _ = edge
        new_section = generate_ternary_vector(sheaf.node_dims[u])
        sheaf.set_section(u, new_section)

        E = improved_hybrid_energy(sheaf, u, M, base_beta, alpha)
        energies[edge] = E

        H = improved_update_restriction_entropy(sheaf, edge, M, base_beta, lr, alpha)
        total_entropy += H

    return energies, total_entropy

if __name__ == "__main__":
    node_dims = {0: 5, 1: 5}
    edges = [(0, 1)]

    sheaf = Sheaf(node_dims, edges)

    rng = np.random.default_rng(42)
    src_map = rng.normal(size=(3, node_dims[0]))
    dst_map = rng.normal(size=(3, node_dims[1]))
    sheaf.set_restriction((0, 1), src_map, dst_map)

    sheaf.set_section(0, generate_ternary_vector(node_dims[0], seed=1))
    sheaf.set_section(1, np.zeros(node_dims[1]))

    M = rng.normal(size=(node_dims[0], node_dims[0]))

    energies, total_entropy = improved_route_and_update(sheaf, M)
    print(energies)
    print(total_entropy)