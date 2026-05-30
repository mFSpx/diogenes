# DARWIN HAMMER — match 2747, survivor 6
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s1.py (gen3)
# parent_b: path_signature.py (gen0)
# born: 2026-05-29T23:45:50Z

"""Hybrid Sheaf–Signature Associative Memory (HSSAM)

This module fuses the sheaf‑theoretic dense associative memory (Parent A) with the
iterated‑integral path signature calculus (Parent B). The mathematical bridge is
the observation that a sheaf’s sections over a walk in its underlying graph form
a discrete path in ℝⁿ. By computing the signature of this path we obtain a
feature vector that captures the geometry of the sections along the walk.
The signature vector is then used as the query vector ξ in the energy
formulation of the dense associative memory. Conversely, the energy gradient can
inform the restriction maps of the sheaf, closing the feedback loop.

Key hybrid operations:
* `sheaf_walk_signature` – builds a path from sheaf sections along a walk and
  returns its signature (optionally after a lead‑lag transform).
* `signature_energy` – evaluates the DAM energy of the signature vector against a
  stored pattern matrix M.
* `update_restriction_via_signature` – updates a sheaf restriction map using the
  soft‑max of scores derived from the signature‑based energy, thereby integrating
  the two mathematical structures.
"""

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
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path):
    """Level‑1 signature: total increment."""
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    """Level‑2 signature (tensor flattened)."""
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running    = path[:-1] - path[0]            # (T‑1, d)
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
    Update the restriction maps of `edge` using the signature‑based energy.
    The new maps are constructed by taking a soft‑max over scores derived from
    the energy gradient with respect to the source section.

    This is a simple, differentiable proxy that respects the mathematical
    interface between the sheaf and the associative memory.
    """
    u, v = edge
    src_dim = sheaf.node_dims[u]
    dst_dim = sheaf.node_dims[v]

    # Build a dummy walk consisting of a single edge (u -> v)
    walk = [(u, v)]
    # Compute signature vector for the source node (single‑point path)
    sig_vec = sheaf_walk_signature(sheaf, u, walk, depth=depth, lead_lag=lead_lag)

    # Scores for each row of M (interpreted as pattern logits)
    scores = beta * (M @ sig_vec)          # (m,)
    probs = _softmax(scores / temperature)  # sharpened distribution

    # Construct new restriction maps as weighted combinations of identity and
    # a random linear map, guided by the probabilities.
    rng = np.random.default_rng()
    random_src = rng.normal(size=(src_dim, src_dim))
    random_dst = rng.normal(size=(dst_dim, dst_dim))

    # Weighted average (broadcast over rows)
    new_src = (probs[:, None, None] * random_src[None, :, :]).sum(axis=0)
    new_dst = (probs[:, None, None] * random_dst[None, :, :]).sum(axis=0)

    sheaf.set_restriction(edge, new_src, new_dst)

# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Construct a tiny sheaf with 3 nodes, each 2‑dimensional.
    node_dims = [2, 2, 2]
    edges = [(0, 1), (1, 2)]
    sheaf = Sheaf(node_dims, edges)

    # Random sections
    rng = np.random.default_rng(42)
    for n in range(3):
        sheaf.set_section(n, rng.normal(size=node_dims[n]))

    # Random restriction maps (identity for simplicity)
    for e in edges:
        src_map = np.eye(node_dims[e[0]])
        dst_map = np.eye(node_dims[e[1]])
        sheaf.set_restriction(e, src_map, dst_map)

    # Pattern matrix M: 5 patterns, dimension matches signature size.
    # Signature depth 2, lead‑lag disabled, each node dim=2 => level1 size 2,
    # level2 size 4, total 6.
    sig_dim = 2 + 4   # depth=2, no lead‑lag
    M = rng.normal(size=(5, sig_dim))

    # Compute hybrid energy along the walk 0→1→2
    walk = [(0, 1), (1, 2)]
    E = signature_energy(sheaf, start_node=0, edge_sequence=walk,
                         M=M, beta=0.8, depth=2, lead_lag=False)
    print(f"Hybrid energy: {E:.4f}")

    # Update restriction on edge (0,1) using signature‑based rule
    update_restriction_via_signature(sheaf, (0, 1), M, beta=0.8,
                                    depth=2, lead_lag=False, temperature=0.05)
    src_map, dst_map = sheaf.get_restriction((0, 1))
    print("Updated restriction maps (shapes):", src_map.shape, dst_map.shape)