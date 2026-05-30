# DARWIN HAMMER — match 1530, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s1.py (gen4)
# parent_b: sheaf_cohomology.py (gen0)
# born: 2026-05-29T23:37:17Z

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
    sig1, _ = path_signature(path)
    mean_value = np.mean(sig1)
    return np.full(len(node_order), mean_value)

def hybrid_energy(sheaf: Sheaf, section: np.ndarray) -> float:
    """
    Compute the Dirichlet energy of a sheaf section.

    Parameters
    ----------
    sheaf : Sheaf
    section : np.ndarray of shape (n,) where n is the number of nodes

    Returns
    -------
    energy : float
    """
    B = sheaf.incidence_matrix()
    return np.sum((B @ section) ** 2)

def improved_hybrid_section_from_path(
    path: List[Point],
    node_order: List[str],
    sheaf: Sheaf,
) -> np.ndarray:
    """
    Improved conversion of a geometric path into a sheaf section (node values).
    The first‑level signature component sig1 (vector of dimension d) is
    used to compute a section that minimizes the Dirichlet energy.

    Parameters
    ----------
    path : List[Point]
    node_order : List[str]
    sheaf : Sheaf

    Returns
    -------
    section : np.ndarray of shape (n,) where n is the number of nodes
    """
    sig1, _ = path_signature(path)
    L = sheaf_laplacian(sheaf)
    n = len(node_order)
    A = np.vstack((L, np.ones((1, n))))
    b = np.hstack((np.zeros(n), sig1[0]))
    section = np.linalg.lstsq(A, b, rcond=None)[0]
    return section