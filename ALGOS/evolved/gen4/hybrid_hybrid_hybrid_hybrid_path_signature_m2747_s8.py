# DARWIN HAMMER — match 2747, survivor 8
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s1.py (gen3)
# parent_b: path_signature.py (gen0)
# born: 2026-05-29T23:45:50Z

import numpy as np
import random
import math
import sys
import pathlib

# ----------------------------------------------------------------------
# Parent A – Sheaf and Dense Associative Memory utilities
# ----------------------------------------------------------------------
class Sheaf:
    def __init__(self, node_dims, edges):
        """
        node_dims: list of int, dimension of each node's vector space.
        edges: list of (u, v) tuples defining directed edges.
        """
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions = {}   # (u,v) -> (src_map, dst_map)
        self._sections = {}       # node -> vector

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
    """
    Dense Associative Memory energy.
    xi : (n,) query vector
    M : (m, n) pattern matrix
    """
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    scores = beta * (M @ xi)                 # (m,)
    lse_term = _lse(scores) / beta
    quadratic_term = 0.5 * xi @ xi
    return -lse_term + quadratic_term

# ----------------------------------------------------------------------
# Parent B – Path signature utilities
# ----------------------------------------------------------------------
def lead_lag_transform(path):
    """
    Lead‑lag transform: interleave (lead, lag) channels.
    path : (T, d)
    returns : (2T-1, 2d)
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     np.zeros(d)])
        out[2 * t + 1] = np.concatenate([np.zeros(d), path[t + 1]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], np.zeros(d)])
    return out

def signature_level1(path):
    """Level‑1 signature: total increment."""
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    """Level‑2 signature (tensor flattened)."""
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running    = path[:-1]                    # (T‑1, d)
    S2 = running.T @ increments                 # (d, d)
    return S2.reshape(-1)                       # flatten to (d*d,)

def signature(path, depth=2, lead_lag=False):
    """
    Compute signature up to `depth`. Returns list [S1, S2, ..., Sdepth]
    where each Sk is a flat numpy array.
    If lead_lag=True the path is first transformed.
    """
    path = np.asarray(path, dtype=float)
    if lead_lag:
        path = lead_lag_transform(path)

    if depth < 1:
        raise ValueError("depth must be >= 1")
    signatures = []

    # Level 1
    S1 = signature_level1(path)
    signatures.append(S1.reshape(-1))

    if depth == 1:
        return signatures

    # Level 2
    S2 = signature_level2(path)
    signatures.append(S2)

    # Higher levels – simple recursive Chen product (naïve, for demonstration)
    prev = signatures[-1]          # flat array of size d^k
    d = path.shape[1]
    for k in range(3, depth + 1):
        # Compute level k via outer product of level (k‑1) with increments and sum
        # This is O(T d^k) but acceptable for small d and depth.
        incr = np.diff(path, axis=0)          # (T‑1, d)
        accum = np.zeros(d ** k, dtype=float)
        for t in range(incr.shape[0]):
            # tensor product of prev (flattened) with incr[t] then add
            prod = np.tensordot(prev.reshape(d ** (k - 1)), incr[t], axes=0)
            accum += prod.reshape(-1)
        signatures.append(accum)
        prev = accum
    return signatures

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def sheaf_walk_path(sheaf, start_node, edge_sequence):
    """
    Follow a sequence of edges starting from `start_node` and collect the
    section vectors at each visited node, forming a discrete path.

    Parameters
    ----------
    sheaf : Sheaf
    start_node : int
    edge_sequence : list of (u, v) edges (must be compatible with the walk)

    Returns
    -------
    path : np.ndarray of shape (L, d) where d is the dimension of the start node.
    """
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
    return np.stack(path, axis=0)   # (L, d)

def sheaf_walk_signature(sheaf, start_node, edge_sequence, depth=2, lead_lag=False):
    """
    Compute the path signature of a sheaf walk.

    Returns a flat concatenation of signatures of all levels up to `depth`.
    """
    path = sheaf_walk_path(sheaf, start_node, edge_sequence)
    sig_levels = signature(path, depth=depth, lead_lag=lead_lag)
    return np.concatenate(sig_levels)

def signature_energy(sheaf, start_node, edge_sequence, M, beta=1.0,
                    depth=2, lead_lag=False):
    """
    Hybrid energy: compute the signature of the sheaf walk and evaluate the
    dense associative memory energy with pattern matrix M.
    """
    sig_vec = sheaf_walk_signature(sheaf, start_node, edge_sequence,
                                   depth=depth, lead_lag=lead_lag)
    return energy(sig_vec, M, beta=beta)

def update_restriction_via_signature(sheaf, edge, M, beta=1.0,
                                    depth=2, lead_lag=False, temperature=0.1):
    """
    Update the restriction maps of `edge` 
    """
    sig_vec = sheaf_walk_signature(sheaf, edge[0], [edge], 
                                   depth=depth, lead_lag=lead_lag)
    scores = -energy(sig_vec, M, beta=beta) / temperature
    softmax_scores = _softmax(scores)
    src_map, dst_map = sheaf.get_restriction(edge)
    updated_src_map = src_map * np.expand_dims(softmax_scores, axis=1)
    updated_dst_map = dst_map * np.expand_dims(softmax_scores, axis=1)
    sheaf.set_restriction(edge, updated_src_map, updated_dst_map)
    return updated_src_map, updated_dst_map

def improved_update_restriction_via_signature(sheaf, edge, M, beta=1.0,
                                             depth=2, lead_lag=False, temperature=0.1):
    """
    Improved update the restriction maps of `edge` 
    """
    sig_vec = sheaf_walk_signature(sheaf, edge[0], [edge], 
                                   depth=depth, lead_lag=lead_lag)
    scores = -energy(sig_vec, M, beta=beta) / temperature
    softmax_scores = _softmax(scores)
    src_map, dst_map = sheaf.get_restriction(edge)
    grad = np.gradient(src_map, axis=1) 
    updated_src_map = src_map + np.expand_dims(softmax_scores, axis=1) * grad
    updated_dst_map = dst_map + np.expand_dims(softmax_scores, axis=1) * grad
    sheaf.set_restriction(edge, updated_src_map, updated_dst_map)
    return updated_src_map, updated_dst_map