# DARWIN HAMMER — match 1693, survivor 4
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
from hashlib import sha256

# ----------------------------------------------------------------------
# Core Sheaf class (validated sections and restrictions)
# ----------------------------------------------------------------------
class Sheaf:
    """
    Validated sheaf supporting restriction maps, ternary sections,
    and construction of the coboundary operator used in cohomology.
    """
    def __init__(self, node_dims, edges):
        """
        node_dims: dict {node_id: dimension}
        edges: list of (u, v) tuples (directed)
        """
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
        self._restrictions = {}   # (u,v) -> (src_map, dst_map)
        self._sections = {}       # node -> vector

    # ------------------------------------------------------------------
    # Restriction handling
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
    # Section handling
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

    # ------------------------------------------------------------------
    # Coboundary operator (matrix representation of all restrictions)
    # ------------------------------------------------------------------
    def coboundary_matrix(self, edge_weights=None):
        """
        Returns a block matrix C such that for a global section vector s,
        C @ s yields the collection of restricted sections on each edge.
        Optional edge_weights (len(edges)) scale each edge's block.
        """
        if edge_weights is None:
            edge_weights = np.ones(len(self.edges), dtype=float)

        blocks = []
        for (u, v), w in zip(self.edges, edge_weights):
            src_map, dst_map = self._restrictions[(u, v)]
            # Build a block that maps concatenated node sections to the edge space.
            # Edge space dimension = rows of src_map (= rows of dst_map)
            rows = src_map.shape[0]
            cols_u = self.node_dims[u]
            cols_v = self.node_dims[v]

            block = np.zeros((rows, sum(self.node_dims.values())))
            # column offsets for u and v in the global concatenated vector
            offset_u = self._col_offset(u)
            offset_v = self._col_offset(v)

            block[:, offset_u:offset_u + cols_u] = w * src_map
            block[:, offset_v:offset_v + cols_v] = -w * dst_map
            blocks.append(block)

        return np.vstack(blocks)

    def _col_offset(self, node):
        """Helper: column offset of a node in the concatenated global vector."""
        offset = 0
        for n, dim in self.node_dims.items():
            if n == node:
                break
            offset += dim
        return offset

    def global_section_vector(self):
        """Concatenates all node sections in the order of node_dims keys."""
        vecs = []
        for node in self.node_dims:
            if node not in self._sections:
                raise KeyError(f"Section for node {node} not set")
            vecs.append(self._sections[node])
        return np.concatenate(vecs)


# ----------------------------------------------------------------------
# Utility functions (deterministic semantic similarity, entropy, etc.)
# ----------------------------------------------------------------------
def deterministic_similarity(node_u, node_v, seed=0):
    """
    Deterministic pseudo‑random similarity in (0.1, 1.0] derived from a hash of the node pair.
    Guarantees reproducibility across runs without storing a RNG state.
    """
    pair = f"{node_u}:{node_v}"
    h = sha256(pair.encode("utf-8")).digest()
    # Use first 8 bytes to create a 64‑bit integer, then map to (0,1)
    int_val = int.from_bytes(h[:8], "big")
    rng_val = (int_val ^ seed) & ((1 << 64) - 1)
    # Scale to (0.1, 1.0]
    return 0.1 + 0.9 * (rng_val / ((1 << 64) - 1))

def shannon_entropy(vector):
    counts = Counter(vector.tolist())
    total = len(vector)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / total
        if p > 0:
            entropy -= p * math.log(p, 2)
    return entropy

def _logsumexp(z):
    m = np.max(z)
    return m + np.log(np.exp(z - m).sum())

def dam_energy(xi, M, beta=1.0):
    """
    Discrete‑Analogue‑Model (DAM) energy.
    xi : ternary vector (shape (d,))
    M  : interaction matrix (shape (d, d))
    """
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    scores = beta * (M @ xi)
    lse = _logsumexp(scores) / beta
    quad = 0.5 * xi @ xi
    return -lse + quad

# ----------------------------------------------------------------------
# Bayesian posterior weighting (edge‑wise)
# ----------------------------------------------------------------------
def bayesian_posterior(prior, likelihood):
    """
    Returns normalized posterior weights.
    prior  : np.ndarray (len(edges))
    likelihood : np.ndarray (len(edges))
    """
    unnorm = prior * likelihood
    total = unnorm.sum()
    if total == 0:
        return prior / prior.sum()
    return unnorm / total

# ----------------------------------------------------------------------
# Global matrix construction with deeper integration of cohomology
# ----------------------------------------------------------------------
def build_global_matrix(sheaf, prior_edge_weights=None, seed=0):
    """
    Constructs a symmetric interaction matrix M that blends:
    1. The coboundary operator C (captures sheaf restriction structure).
    2. Edge‑wise semantic similarity s_uv.
    3. Bayesian posterior scaling p_uv.

    The final matrix is M = (C^T D C) where D is a diagonal matrix with
    entries d_i = p_i * s_i for each edge i.
    """
    if prior_edge_weights is None:
        prior_edge_weights = np.ones(len(sheaf.edges), dtype=float)

    # Compute raw similarities and posterior weights
    similarities = np.array([
        deterministic_similarity(u, v, seed=seed) for (u, v) in sheaf.edges
    ], dtype=float)
    posterior = bayesian_posterior(prior_edge_weights, similarities)

    # Edge scaling factors d_i = p_i * s_i
    edge_factors = posterior * similarities

    # Build coboundary matrix C (rows = sum edge dimensions)
    C = sheaf.coboundary_matrix(edge_weights=edge_factors)

    # Symmetrize via C^T D C where D is identity because scaling already applied
    M = C.T @ C
    return M

# ----------------------------------------------------------------------
# Unified energy evaluation (deeper mathematical coupling)
# ----------------------------------------------------------------------
def hybrid_energy(sheaf, query_node, beta=1.0, prior_edge_weights=None, seed=0):
    """
    Computes the DAM energy of the ternary section at `query_node`
    using a globally consistent interaction matrix that respects
    sheaf cohomology (via the coboundary operator) and the ternary
    physics model.
    """
    xi = sheaf.get_section(query_node)
    if not np.all(np.isin(xi, [-1, 0, 1])):
        raise ValueError("Section vector must be ternary (elements in {-1,0,1})")

    M = build_global_matrix(sheaf, prior_edge_weights=prior_edge_weights, seed=seed)
    return dam_energy(xi, M, beta=beta)

# ----------------------------------------------------------------------
# Demonstration utilities (three distinct demos)
# ----------------------------------------------------------------------
def demo_basic_construction():
    """Construct a tiny sheaf, set restrictions, sections, and compute energy."""
    node_dims = {'A': 3, 'B': 2, 'C': 3}
    edges = [('A', 'B'), ('B', 'C')]
    sh = Sheaf(node_dims, edges)

    rng = np.random.default_rng(42)
    for (u, v) in edges:
        rows = max(node_dims[u], node_dims[v])  # simple choice for demo
        src = rng.normal(size=(rows, node_dims[u]))
        dst = rng.normal(size=(rows, node_dims[v]))
        sh.set_restriction((u, v), src, dst)

    # Ternary sections
    sh.set_section('A', generate_ternary_vector(node_dims['A'], seed=1))
    sh.set_section('B', generate_ternary_vector(node_dims['B'], seed=2))
    sh.set_section('C', generate_ternary_vector(node_dims['C'], seed=3))

    energy = hybrid_energy(sh, query_node='A', beta=0.8, seed=7)
    print(f"Demo 1 – Energy at node A: {energy:.4f}")

def demo_cohomology_insight():
    """
    Show that the kernel of the coboundary matrix corresponds to global sections
    that are locally compatible. Verify that a compatible section yields near‑zero
    coboundary norm.
    """
    node_dims = {'X': 2, 'Y': 2}
    edges = [('X', 'Y')]
    sh = Sheaf(node_dims, edges)

    rng = np.random.default_rng(0)
    src = np.eye(2)          # identity restriction
    dst = np.eye(2)          # identity restriction
    sh.set_restriction(('X', 'Y'), src, dst)

    # Choose a compatible global section (same vector on both nodes)
    vec = generate_ternary_vector(2, seed=99)
    sh.set_section('X', vec)
    sh.set_section('Y', vec)

    C = sh.coboundary_matrix()
    global_vec = sh.global_section_vector()
    residual = C @ global_vec
    norm = np.linalg.norm(residual)
    print(f"Demo 2 – Coboundary residual norm (should be ~0): {norm:.6e}")

def demo_posterior_effect():
    """
    Compare energies with uniform priors vs. priors that heavily favor one edge.
    Demonstrates the impact of Bayesian weighting on the interaction matrix.
    """
    node_dims = {'P': 3, 'Q': 3, 'R': 3}
    edges = [('P', 'Q'), ('Q', 'R')]
    sh = Sheaf(node_dims, edges)

    rng = np.random.default_rng(123)
    for (u, v) in edges:
        rows = 4
        src = rng.normal(size=(rows, node_dims[u]))
        dst = rng.normal(size=(rows, node_dims[v]))
        sh.set_restriction((u, v), src, dst)

    sh.set_section('P', generate_ternary_vector(node_dims['P'], seed=11))
    sh.set_section('Q', generate_ternary_vector(node_dims['Q'], seed=22))
    sh.set_section('R', generate_ternary_vector(node_dims['R'], seed=33))

    uniform_prior = np.ones(len(edges))
    biased_prior = np.array([0.9, 0.1])  # favor first edge

    e_uniform = hybrid_energy(sh, 'P', prior_edge_weights=uniform_prior, seed=5)
    e_biased  = hybrid_energy(sh, 'P', prior_edge_weights=biased_prior,  seed=5)

    print(f"Demo 3 – Energy with uniform prior: {e_uniform:.4f}")
    print(f"Demo 3 – Energy with biased prior : {e_biased:.4f}")

# ----------------------------------------------------------------------
# Helper: ternary vector generator (kept from original code)
# ----------------------------------------------------------------------
def generate_ternary_vector(dim, seed=None):
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 0, 1], size=dim)

# ----------------------------------------------------------------------
# Run demos if executed as script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_basic_construction()
    demo_cohomology_insight()
    demo_posterior_effect()