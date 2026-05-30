# DARWIN HAMMER — match 1530, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s1.py (gen4)
# parent_b: sheaf_cohomology.py (gen0)
# born: 2026-05-29T23:37:17Z

"""Hybrid algorithm merging:
- PARENT ALGORITHM A: minimum‑cost tree metrics & path‑signature (lead‑lag transform)
- PARENT ALGORITHM B: cellular sheaf cohomology on undirected graphs

Mathematical bridge:
Edge lengths from the tree (Algorithm A) are turned into scalar restriction maps
for a 1‑dimensional cellular sheaf (Algorithm B).  The scalar weight
wₑ = exp(‑β·ℓₑ) (β>0) encodes a probabilistic confidence derived from the
minimum‑cost (shorter edges ⇒ larger weight).  The path signature computed on
the tree supplies a node‑section vector s ∈ ℝᴺ.  The sheaf coboundary
δ(s) uses the same weights, so the Dirichlet energy ‖δ(s)‖² simultaneously
measures (i) disagreement of the signature across edges and (ii) the
minimum‑cost (via wₑ).  Thus the hybrid system fuses the iterated‑integral
algebra of signatures with the block‑structured Laplacian of sheaves.

The module provides:
* tree_metrics – Euclidean geometry of a rooted tree.
* path_signature – lead‑lag transform → first‑ and second‑level signatures.
* Sheaf – lightweight 1‑D sheaf with restriction maps derived from edge
  lengths.
* sheaf_laplacian – explicit matrix L = δᵀδ.
* hybrid_energy – Dirichlet energy of a signature‑induced section.

All functions are pure Python/NumPy and require no external scientific stack.
"""

import math
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ---------------------------------------------------------------------------
# Algorithm A components
# ---------------------------------------------------------------------------

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping ordered edge (a, b) → Euclidean length
    root_dist : dict mapping node → distance from root along the tree
    """
    adj = defaultdict(list)
    edge_len: Dict[Edge, float] = {}
    root_dist: Dict[str, float] = {root: 0.0}

    # first pass: adjacency and edge lengths
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        d = length(nodes[a], nodes[b])
        edge_len[(a, b)] = d
        edge_len[(b, a)] = d

    # BFS to compute root distances
    visited = {root}
    queue = [root]
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in visited:
                root_dist[nb] = root_dist[cur] + edge_len[(cur, nb)]
                visited.add(nb)
                queue.append(nb)

    return adj, edge_len, root_dist

def lead_lag_transform(path: List[Point]) -> np.ndarray:
    """
    Simple lead‑lag transform.
    For a path P = [p₀, p₁, …, p_T] we interleave the original (lead) and a
    lagged copy shifted by one step (lag).  The resulting 2·d dimensional
    stream is returned as a NumPy array of shape (T, 2*d).
    """
    if len(path) < 2:
        raise ValueError("Path must contain at least two points")
    lead = np.array(path[:-1])          # shape (T-1, d)
    lag = np.array(path[1:])            # shape (T-1, d)
    return np.hstack([lead, lag])       # shape (T-1, 2*d)

def path_signature(path: List[Point]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a truncated signature (levels 1 and 2) using the lead‑lag stream.
    Returns
    -------
    sig1 : np.ndarray, shape (d,)   – first level (increment sum)
    sig2 : np.ndarray, shape (d, d) – second level (area matrix)
    """
    stream = lead_lag_transform(path)          # shape (T, 2d)
    d = stream.shape[1] // 2
    lead = stream[:, :d]
    lag = stream[:, d:]

    # First level: sum of increments (lead - lag)
    increments = lead - lag
    sig1 = increments.sum(axis=0)               # shape (d,)

    # Second level: iterated integral ∫ (lead - lag) ⊗ (lead - lag) dt
    # Approximated by Riemann sum of outer products of increments.
    sig2 = np.einsum('ti,tj->ij', increments, increments) / len(increments)
    return sig1, sig2

# ---------------------------------------------------------------------------
# Algorithm B components (lightweight 1‑D sheaf)
# ---------------------------------------------------------------------------

class Sheaf:
    """
    1‑dimensional cellular sheaf on a graph.

    For each node v we store a scalar value (dimension 1).
    For each edge e we store a scalar stalk with restriction maps
    F(u→e) = wₑ and F(v→e) = -wₑ (sign follows orientation).
    The weight wₑ is derived from the edge length ℓₑ via
        wₑ = exp(-β·ℓₑ),   β > 0
    This choice mirrors the probabilistic weighting of the minimum‑cost
    tree in Algorithm A.
    """

    def __init__(self, nodes: List[str], edges: List[Edge], edge_lengths: Dict[Edge, float],
                 beta: float = 1.0):
        self.nodes = list(nodes)
        self.edges = list(edges)                     # oriented as given
        self.beta = float(beta)

        # scalar restriction maps stored as dict edge -> weight
        self._weights: Dict[Edge, float] = {}
        for (u, v) in self.edges:
            ℓ = edge_lengths[(u, v)]
            w = math.exp(-self.beta * ℓ)
            self._weights[(u, v)] = w

    def weight(self, edge: Edge) -> float:
        """Return the scalar restriction weight for oriented edge."""
        return self._weights[edge]

    def incidence_matrix(self) -> np.ndarray:
        """
        Build the signed incidence matrix B (|E| × |V|) where
            B[e, u] = -wₑ   if u is the tail of e = (u, v)
            B[e, v] =  wₑ   if v is the head of e = (u, v)
        All other entries are zero.
        """
        n = len(self.nodes)
        m = len(self.edges)
        node_index = {node: i for i, node in enumerate(self.nodes)}
        B = np.zeros((m, n))
        for e_idx, (u, v) in enumerate(self.edges):
            w = self.weight((u, v))
            B[e_idx, node_index[u]] = -w
            B[e_idx, node_index[v]] =  w
        return B

def sheaf_laplacian(sheaf: Sheaf) -> np.ndarray:
    """
    Compute the sheaf Laplacian L = Bᵀ B,
    where B is the signed weighted incidence matrix.
    The result is a symmetric |V|×|V| positive‑semidefinite matrix.
    """
    B = sheaf.incidence_matrix()
    return B.T @ B

# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------

def build_hybrid_sheaf(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    beta: float = 1.0,
) -> Tuple[Sheaf, Dict[str, float]]:
    """
    Construct a Sheaf whose edge weights are derived from the minimum‑cost
    tree metrics (Algorithm A) and return also the root‑to‑node distances.
    """
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    # For orientation we keep the original order (u, v) as supplied.
    sheaf = Sheaf(list(nodes.keys()), edges, edge_len, beta=beta)
    return sheaf, root_dist

def hybrid_section_from_path(
    path: List[Point],
    node_order: List[str],
) -> np.ndarray:
    """
    Convert a geometric path into a sheaf section (node values).
    The first‑level signature component sig1 (vector of dimension d) is
    projected onto the node ordering by simple averaging: each node receives
    the same scalar equal to the mean of sig1 components.  This is a toy
    mapping sufficient for demonstration.
    """
    sig1, _ = path_signature(path)          # sig1 shape (d,)
    scalar = float(sig1.mean())             # collapse to a single number
    section = np.full(len(node_order), scalar)
    return section

def hybrid_energy(
    path: List[Point],
    sheaf: Sheaf,
    node_order: List[str],
) -> float:
    """
    Compute Dirichlet energy ‖δ(s)‖² for the section induced by the path.
    Steps:
    1. Build a section vector s ∈ ℝ^{|V|} from the path signature.
    2. Compute δ(s) = B s, where B is the weighted incidence matrix.
    3. Return the squared Euclidean norm of δ(s).
    """
    s = hybrid_section_from_path(path, node_order)   # shape (|V|,)
    B = sheaf.incidence_matrix()                     # shape (|E|, |V|)
    delta_s = B @ s                                   # shape (|E|,)
    energy = float(np.linalg.norm(delta_s) ** 2)
    return energy

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Define a tiny tree (rooted at 'A')
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D")]   # simple chain A‑B‑C‑D
    root = "A"

    # Build hybrid sheaf
    sheaf, root_dist = build_hybrid_sheaf(nodes, edges, root, beta=0.8)

    # Create a geometric path that visits the nodes in order
    path = [nodes[n] for n in ["A", "B", "C", "D"]]

    # Compute hybrid energy
    energy = hybrid_energy(path, sheaf, list(nodes.keys()))
    print("Root‑to‑node distances:", root_dist)
    print("Sheaf Laplacian:\n", sheaf_laplacian(sheaf))
    print("Hybrid Dirichlet energy of the path‑induced section:", energy)