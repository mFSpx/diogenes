# DARWIN HAMMER — match 1693, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m536_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ternary_lens__m159_s2.py (gen4)
# born: 2026-05-29T23:38:28Z

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter

class Sheaf:
    """Validated sheaf supporting restriction maps and ternary sections."""
    def __init__(self, node_dims, edges):
        """
        node_dims: dict {node_id: dimension}
        edges: list of (u, v) tuples
        """
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
        self._restrictions = {}   # (u,v) -> (src_map, dst_map)
        self._sections = {}       # node -> vector

    def set_restriction(self, edge, src_map, dst_map):
        u, v = edge
        src_map = np.asarray(src_map, dtype=float)
        dst_map = np.asarray(dst_map, dtype=float)

        if u not in self.node_dims or v not in self.node_dims:
            raise KeyError(f"Edge nodes {u},{v} must be defined in node_dims")

        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")

        self._restrictions[(u, v)] = (src_map, dst_map)

    def get_restriction(self, edge):
        return self._restrictions[edge]

    def set_section(self, node, value):
        if node not in self.node_dims:
            raise KeyError(f"Node {node} not defined in sheaf")
        vec = np.asarray(value, dtype=float)
        if vec.shape[0] != self.node_dims[node]:
            raise ValueError("Section vector length must match node dimension")
        self._sections[node] = vec

    def get_section(self, node):
        return self._sections[node]

def generate_ternary_vector(dim, seed=None):
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 0, 1], size=dim)

def shannon_entropy(vector):
    counts = Counter(vector.tolist())
    total = len(vector)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / total
        entropy -= p * math.log(p, 2) if p > 0 else 0.0
    return entropy

def _logsumexp(z):
    m = np.max(z)
    return m + np.log(np.exp(z - m).sum())

def dam_energy(xi, M, beta=1.0):
    """Discrete-Analogue-Model (DAM) energy used in Algorithm B."""
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    scores = beta * (M @ xi)
    lse = _logsumexp(scores) / beta
    quad = 0.5 * xi @ xi
    return -lse + quad

def semantic_similarity(node_u, node_v, seed=None):
    """
    Semantic similarity based on mutual information of node sections.
    """
    # Assume sections are set for nodes u and v
    section_u = np.array([1, 0, 1])  # Placeholder
    section_v = np.array([0, 1, 1])  # Placeholder

    # Compute mutual information
    p_u = np.mean(section_u)
    p_v = np.mean(section_v)
    p_uv = np.mean(section_u * section_v)

    mi = math.log(p_uv / (p_u * p_v)) if p_uv > 0 and p_u > 0 and p_v > 0 else 0.0
    return mi

def bayesian_posterior(prior, likelihood):
    """
    Simple Bayesian update assuming a uniform evidence term.
    posterior ∝ prior * likelihood, normalized to sum to 1 across all edges.
    """
    unnorm = prior * likelihood
    total = unnorm.sum()
    return unnorm / total if total != 0 else prior

def build_global_matrix(sheaf, prior_edge_weights=None, seed=None):
    """
    Constructs the global interaction matrix M used in dam_energy.
    Steps:
    1. For each edge (u,v) retrieve restriction maps (S_uv, D_uv).
    2. Compute a semantic similarity s_uv based on mutual information.
    3. Scale both maps by s_uv (semantic weighting).
    4. Collect all scaled src maps into a block‑row matrix A and all scaled dst maps
       into a block‑column matrix B such that M = A @ B.T.
    5. Apply a Bayesian posterior scaling based on prior_edge_weights (optional).
    """
    if prior_edge_weights is None:
        prior_edge_weights = np.ones(len(sheaf.edges))

    # Collect scaled maps
    scaled_src = []
    scaled_dst = []
    similarities = []
    for idx, (u, v) in enumerate(sheaf.edges):
        src_map, dst_map = sheaf.get_restriction((u, v))
        s_uv = semantic_similarity(u, v)

        # Semantic scaling
        src_scaled = s_uv * src_map
        dst_scaled = s_uv * dst_map
        scaled_src.append(src_scaled)
        scaled_dst.append(dst_scaled)
        similarities.append(s_uv)

    # Stack into block matrices
    A = np.block([scaled_src])   # shape (r, sum dim_u)
    B = np.block([scaled_dst])   # shape (r, sum dim_v)
    # Initial global matrix
    M_raw = A @ B.T

    # Bayesian posterior scaling across edges
    likelihood = np.array(similarities)
    posterior = bayesian_posterior(prior_edge_weights, likelihood)

    # Apply posterior scaling as an outer product to M_raw
    # Reshape posterior to (r,1) where r is total rows of A/B
    posterior_vec = np.repeat(posterior, A.shape[0] // len(posterior))
    posterior_vec = posterior_vec[:, np.newaxis]
    M = M_raw * posterior_vec

    return M

def hybrid_energy(sheaf, query_node, beta=1.0, prior_edge_weights=None, seed=None):
    """
    Unified energy evaluation:
    - Retrieves the ternary section vector at `query_node`.
    - Builds the global matrix M with semantic and Bayesian weighting.
    - Computes DAM energy.
    """
    xi = sheaf.get_section(query_node)
    if not np.all(np.isin(xi, [-1, 0, 1])):
        raise ValueError("Section vector must be ternary (elements in {-1,0,1})")

    M = build_global_matrix(sheaf, prior_edge_weights=prior_edge_weights, seed=seed)
    return dam_energy(xi, M, beta=beta)

def demo_sheaf_construction():
    node_dims = {'A': 3, 'B': 2, 'C': 3}
    edges = [('A', 'B'), ('B', 'C')]
    sh = Sheaf(node_dims, edges)

    rng = np.random.default_rng(42)
    sh.set_restriction(('A', 'B'), rng.random((2, 3)), rng.random((2, 2)))
    sh.set_restriction(('B', 'C'), rng.random((3, 2)), rng.random((3, 3)))

    sh.set_section('A', generate_ternary_vector(3))
    sh.set_section('B', generate_ternary_vector(2))
    sh.set_section('C', generate_ternary_vector(3))

    print(hybrid_energy(sh, 'A'))

demo_sheaf_construction()