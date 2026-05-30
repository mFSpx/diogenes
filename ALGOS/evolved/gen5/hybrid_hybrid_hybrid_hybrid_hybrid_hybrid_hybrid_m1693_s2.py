# DARWIN HAMMER — match 1693, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m536_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ternary_lens__m159_s2.py (gen4)
# born: 2026-05-29T23:38:28Z

"""Hybrid Sheaf Cohomology with Ternary Energy Fusion

Parents:
- hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m536_s2.py (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_ternary_lens__m159_s2.py (Algorithm B)

Mathematical Bridge:
Algorithm A modifies sheaf edge restriction maps with semantic similarity scores and
Bayesian posterior updates, effectively re‑weighting the linear maps that connect node
sections. Algorithm B defines a ternary‑vector based energy functional (DAM energy)
that evaluates a sheaf configuration using the restriction maps assembled into a global
matrix *M*.

The hybrid algorithm therefore:
1. Constructs a sheaf with validated node dimensions (A + B).
2. Scales each restriction map by a semantic similarity factor *s₍u,v₎*.
3. Applies a Bayesian update to obtain a posterior scaling *p₍u,v₎*.
4. Builds the global interaction matrix *M* from the posterior‑scaled restrictions.
5. Evaluates the ternary DAM energy on a query node’s section vector.

The resulting system unifies sheaf cohomology (A) with ternary energy physics (B)."""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter

# ----------------------------------------------------------------------
# Core Sheaf class (merged validation and utility)
# ----------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Restriction handling (validation from Algorithm B, storage from A)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Section handling (validation from Algorithm B)
    # ------------------------------------------------------------------
    def set_section(self, node, value):
        if node not in self.node_dims:
            raise KeyError(f"Node {node} not defined in sheaf")
        vec = np.asarray(value, dtype=float)
        if vec.shape[0] != self.node_dims[node]:
            raise ValueError("Section vector length must match node dimension")
        self._sections[node] = vec

    def get_section(self, node):
        return self._sections[node]

# ----------------------------------------------------------------------
# Utility functions from Algorithm B
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Hybrid-specific mathematics
# ----------------------------------------------------------------------
def semantic_similarity(node_u, node_v, seed=None):
    """
    Placeholder semantic similarity: a random value in (0,1] seeded for reproducibility.
    In a real system this would be derived from language model embeddings.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 1.0)

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
    2. Compute a semantic similarity s_uv.
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
        s_uv = semantic_similarity(u, v, seed=seed)
        similarities.append(s_uv)

        # Semantic scaling
        src_scaled = s_uv * src_map
        dst_scaled = s_uv * dst_map
        scaled_src.append(src_scaled)
        scaled_dst.append(dst_scaled)

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

# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def demo_sheaf_construction():
    """Creates a small sheaf, sets restrictions and ternary sections."""
    node_dims = {'A': 3, 'B': 2, 'C': 3}
    edges = [('A', 'B'), ('B', 'C')]
    sh = Sheaf(node_dims, edges)

    # Random restriction maps respecting dimensions
    rng = np.random.default_rng(42)
    for e in edges:
        u, v = e
        rows = rng.integers(1, 4)   # random row count per edge
        src = rng.standard_normal((rows, node_dims[u]))
        dst = rng.standard_normal((rows, node_dims[v]))
        sh.set_restriction(e, src, dst)

    # Ternary sections
    sh.set_section('A', generate_ternary_vector(node_dims['A'], seed=1))
    sh.set_section('B', generate_ternary_vector(node_dims['B'], seed=2))
    sh.set_section('C', generate_ternary_vector(node_dims['C'], seed=3))

    return sh

def demo_hybrid_energy():
    """Runs hybrid_energy on the demo sheaf and prints the result."""
    sh = demo_sheaf_construction()
    energy = hybrid_energy(sh, query_node='A', beta=0.8, seed=99)
    print(f"Hybrid DAM energy for node 'A': {energy:.6f}")

def demo_entropy_of_section():
    """Computes Shannon entropy of a ternary section vector."""
    sh = demo_sheaf_construction()
    vec = sh.get_section('B')
    ent = shannon_entropy(vec)
    print(f"Shannon entropy of node 'B' section: {ent:.4f}")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Hybrid Sheaf Demo ===")
    demo_hybrid_energy()
    demo_entropy_of_section()
    print("Smoke test completed without errors.")