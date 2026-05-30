# DARWIN HAMMER — match 2747, survivor 7
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s1.py (gen3)
# parent_b: path_signature.py (gen0)
# born: 2026-05-29T23:45:50Z

"""Hybrid algorithm merging sheaf‑based dense associative memory with path‑signature algebra.

Parents:
- hybrid_hybrid_sketch_dense_associative_me_m32_s1.py (sheaf + DAM energy)
- path_signature.py (lead‑lag transform and iterated‑integral signatures)

Mathematical bridge:
Each sheaf node stores a *path* (T×d array) instead of a single vector.
The iterated‑integral signature of that path (level k) is a high‑order feature
vector 𝜙∈ℝ^{d^k}.  This feature vector is used as the query vector in the
Dense Associative Memory (DAM) energy  E(𝜙;M)=−log‑sum‑exp(βM𝜙)+½‖𝜙‖².
Thus restriction maps are updated based on gradients of the DAM energy with
respect to signature features, tightly coupling the sheaf topology and the
signature algebra."""

import numpy as np
import random
import math
import sys
import pathlib

# ----------------------------------------------------------------------
# Sheaf infrastructure (from Parent A)
# ----------------------------------------------------------------------
class Sheaf:
    def __init__(self, node_dims, edges):
        self.node_dims = node_dims            # list of dimensions per node
        self.edges = edges                    # list of (u,v) tuples
        self._restrictions = {}               # (u,v) -> (src_map, dst_map)
        self._sections = {}                   # node -> np.ndarray (path)

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
        """value is expected to be a (T, d) array – a time‑ordered path."""
        self._sections[node] = np.asarray(value, dtype=float)

# ----------------------------------------------------------------------
# Energy functions (from Parent A)
# ----------------------------------------------------------------------
def _softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()

def _lse(z):
    m = np.max(z)
    return m + np.log(np.exp(z - m).sum())

def energy(xi, M, beta=1.0):
    """Dense Associative Memory energy for vector xi and pattern matrix M."""
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    scores = beta * (M @ xi)
    lse_term = _lse(scores) / beta
    quadratic_term = 0.5 * xi @ xi
    return -lse_term + quadratic_term

def hybrid_energy(sheaf, query_node, M, beta=1.0):
    """DAM energy where the query is the raw section vector of a node."""
    query_vector = sheaf._sections[query_node]          # shape (T, d)
    # Collapse path to a single vector (e.g. last point) for compatibility
    query_vector = query_vector[-1]
    return energy(query_vector, M, beta)

# ----------------------------------------------------------------------
# Path‑signature utilities (from Parent B)
# ----------------------------------------------------------------------
def lead_lag_transform(path):
    """Lead‑lag transform: interleaves (lead, lag) channels for causality."""
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path):
    """Level‑1 signature – total increment."""
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    """Level‑2 signature – left‑point Riemann sum of iterated integrals."""
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running    = path[:-1] - path[0]            # (T‑1, d)
    return running.T @ increments               # (d, d)

def signature(path, depth=3):
    """Signature up to a given depth (iterated‑integral tensors flattened)."""
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    if T < 2:
        raise ValueError("Path must contain at least two points")
    inc = np.diff(path, axis=0)                 # (T‑1, d)

    # Initialize flat accumulators for each level
    accum = [np.zeros(d)]                       # level‑1 flat (size d)
    for k in range(2, depth + 1):
        accum.append(np.zeros(d ** k))

    for t in range(T - 1):
        inc_t = inc[t]                          # (d,)
        # Update higher levels in descending order to use previous level values
        for k in range(depth, 1, -1):
            prev = accum[k - 2]                 # level k‑1 flat
            accum[k - 1] += np.kron(prev, inc_t)
        # Level‑1 update
        accum[0] += inc_t

    # Reshape each accumulator to its tensor shape
    out = []
    for k, flat in enumerate(accum, start=1):
        out.append(flat.reshape((d,) * k))
    return out

# ----------------------------------------------------------------------
# Hybrid operations that intertwine signatures with DAM energy
# ----------------------------------------------------------------------
def hybrid_signature_energy(sheaf, query_node, M, beta=1.0, depth=2):
    """
    Compute DAM energy where the query vector is the flattened signature
    (level `depth`) of the node's path.
    """
    path = sheaf._sections[query_node]          # (T, d)
    sigs = signature(path, depth=depth)         # list of tensors
    vec = sigs[-1].reshape(-1)                  # flatten highest level
    return energy(vec, M, beta)

def update_restriction_by_signature(sheaf, edge, M, beta=1.0,
                                   depth=2, lr=1e-3):
    """
    Perform a single gradient‑descent step on the source restriction map
    using the DAM energy gradient with respect to the node's signature.
    """
    u, v = edge
    if (u, v) not in sheaf._restrictions:
        raise KeyError(f"Restriction for edge {(u, v)} not set")
    src_map, dst_map = sheaf._restrictions[(u, v)]

    # Obtain signature vector for source node
    path = sheaf._sections[u]                     # (T, d_u)
    sigs = signature(path, depth=depth)
    qvec = sigs[-1].reshape(-1)                   # shape (d_u**depth,)

    # Energy gradient w.r.t qvec: ∇_q E = -β Mᵀ softmax(β M q) + q
    scores = beta * (M @ qvec)
    probs = _softmax(scores)
    grad_q = -beta * (M.T @ probs) + qvec

    # Propagate gradient to src_map via a simple linear approximation:
    # Assume qvec ≈ src_map @ v0 where v0 is the first point of the path.
    # We update src_map in the direction of grad_q projected onto its rows.
    # This is a heuristic but demonstrates the hybrid coupling.
    row_update = lr * grad_q[:src_map.shape[0]]
    src_map += row_update[:, None] * np.ones((1, src_map.shape[1]))

    sheaf._restrictions[(u, v)] = (src_map, dst_map)

def hybrid_energy_with_lead_lag(sheaf, query_node, M, beta=1.0, depth=2):
    """
    Variant that first applies the lead‑lag transform to the node's path,
    then computes the signature‑based DAM energy.
    """
    raw_path = sheaf._sections[query_node]       # (T, d)
    ll_path = lead_lag_transform(raw_path)      # (2T‑1, 2d)
    sigs = signature(ll_path, depth=depth)
    vec = sigs[-1].reshape(-1)
    return energy(vec, M, beta)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny sheaf with two nodes
    dims = [3, 3]                       # each node lives in ℝ³
    edges = [(0, 1)]

    sheaf = Sheaf(node_dims=dims, edges=edges)

    # Random restriction maps (same row dimension for simplicity)
    src = np.random.randn(4, dims[0])
    dst = np.random.randn(4, dims[1])
    sheaf.set_restriction((0, 1), src, dst)

    # Random time‑series sections (paths) for each node
    T = 5
    path0 = np.cumsum(np.random.randn(T, dims[0]), axis=0)   # walk in ℝ³
    path1 = np.cumsum(np.random.randn(T, dims[1]), axis=0)
    sheaf.set_section(0, path0)
    sheaf.set_section(1, path1)

    # Pattern matrix M compatible with level‑2 signature (d² = 9)
    depth = 2
    sig_dim = dims[0] ** depth
    M = np.random.randn(8, sig_dim)   # 8 stored patterns

    # Compute hybrid energies
    e_raw = hybrid_energy(sheaf, 0, M, beta=0.8)
    e_sig = hybrid_signature_energy(sheaf, 0, M, beta=0.8, depth=depth)
    e_ll  = hybrid_energy_with_lead_lag(sheaf, 0, M, beta=0.8, depth=depth)

    print("Raw DAM energy :", e_raw)
    print("Signature‑based energy :", e_sig)
    print("Lead‑lag + signature energy :", e_ll)

    # Perform a restriction update
    before = sheaf._restrictions[(0, 1)][0].copy()
    update_restriction_by_signature(sheaf, (0, 1), M, beta=0.8,
                                    depth=depth, lr=1e-4)
    after = sheaf._restrictions[(0, 1)][0]
    print("Restriction map change (norm):", np.linalg.norm(after - before))