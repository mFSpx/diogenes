# DARWIN HAMMER — match 4327, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_sheaf_cohomology_m1530_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hard_t_ollivier_ricci_curva_m393_s1.py (gen4)
# born: 2026-05-29T23:55:04Z

"""Hybrid Sheaf‑Curvature Cohomology (HSCC)

This module fuses two ancestral algorithms:

* **Parent A** – a hybrid of minimum‑cost tree Bayes updates and cellular sheaf
  cohomology.  It supplies tree metrics, a `Sheaf` data structure and the idea of
  representing the coboundary operator as a matrix built from probabilistic
  statistics.

* **Parent B** – a stylometry‑driven weighted‑graph construction coupled with an
  Ollivier‑Ricci curvature estimator.  It supplies a method to turn textual
  data into a graph whose edge weights encode stylistic similarity and whose
  curvature measures local connectivity.

The mathematical bridge is the **edge‑curvature matrix**.  For each edge
`(u, v)` of the stylometric graph we compute an Ollivier‑Ricci curvature
`κ_uv ∈ ℝ`.  This scalar is used as a scaling factor for a restriction map
between the vector spaces attached to the incident nodes.  In sheaf language
the restriction `ρ_{uv}` is taken to be `κ_uv·I_k` where `I_k` is the identity on
the `k`‑dimensional feature space.  Stacking all such restrictions yields a
global coboundary matrix `δ`.  Simultaneously we build a minimum‑spanning tree
(MST) on the same graph (using `1‑κ_uv` as an edge length) to obtain the tree
metric required by the original hybrid algorithm.  The resulting system
provides a unified cohomological descriptor of the text corpus that depends
both on stylometric similarity and on the curvature‑driven geometry of the
graph.

The module supplies three high‑level functions demonstrating this hybrid
operation and a small smoke test.
"""

import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Stylometry utilities (Parent B)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> List[str]:
    """Return a list of lower‑case words stripped of punctuation."""
    cleaned = "".join(ch.lower() if ch.isalnum() or ch == "'" else " " for ch in text)
    return [w for w in cleaned.split() if w]


def stylometry_vector(text: str) -> np.ndarray:
    """Simple bag‑of‑words vector over the FUNCTION_CATS categories."""
    tokens = words(text)
    cat_counts = Counter()
    for token in tokens:
        for cat, vocab in FUNCTION_CATS.items():
            if token in vocab:
                cat_counts[cat] += 1
                break
    vec = np.array([cat_counts.get(cat, 0) for cat in sorted(FUNCTION_CATS)], dtype=float)
    # Normalise to unit length to obtain a direction in feature space
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def build_graph(texts: List[str]) -> Tuple[Dict[int, np.ndarray], List[Tuple[int, int]], np.ndarray]:
    """Create a fully connected weighted graph from stylometric vectors.

    Returns
    -------
    nodes : dict mapping integer node id → feature vector
    edges : list of (i, j) tuples with i < j
    weights : np.ndarray of shape (E,) containing cosine similarity for each edge
    """
    nodes = {i: stylometry_vector(t) for i, t in enumerate(texts)}
    edges = []
    sims = []
    ids = list(nodes.keys())
    for a_idx in range(len(ids)):
        for b_idx in range(a_idx + 1, len(ids)):
            i, j = ids[a_idx], ids[b_idx]
            vi, vj = nodes[i], nodes[j]
            # cosine similarity; safe for zero vectors
            denom = (np.linalg.norm(vi) * np.linalg.norm(vj))
            sim = float(np.dot(vi, vj) / denom) if denom > 0 else 0.0
            edges.append((i, j))
            sims.append(sim)
    return nodes, edges, np.array(sims, dtype=float)


# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature (simplified) (Parent B)
# ----------------------------------------------------------------------
def curvature_estimate(
    edges: List[Tuple[int, int]],
    sims: np.ndarray,
) -> Dict[Tuple[int, int], float]:
    """Very simple curvature proxy:
    κ_uv = 1 - d_uv / ⟨d⟩   where d_uv = 1 - similarity and ⟨d⟩ is the mean distance.
    """
    distances = 1.0 - sims
    avg_dist = float(np.mean(distances)) if len(distances) > 0 else 1.0
    curv = {}
    for (u, v), d in zip(edges, distances):
        curv[(u, v)] = 1.0 - d / avg_dist
    return curv


# ----------------------------------------------------------------------
# Minimum‑spanning tree utilities (Parent A)
# ----------------------------------------------------------------------
def edge_length_from_curvature(kappa: float) -> float:
    """Convert curvature to a length suitable for MST construction.
    Larger curvature → shorter effective length."""
    # Guard against division by zero; shift into (0, 2] interval.
    return 1.0 / (kappa + 1e-6)


def kruskal_mst(
    nodes: Dict[int, Any],
    edges: List[Tuple[int, int]],
    edge_lengths: List[float],
) -> List[Tuple[int, int]]:
    """Return a list of edges belonging to the MST using Kruskal's algorithm."""
    parent = {node: node for node in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            return True
        return False

    sorted_edges = sorted(zip(edges, edge_lengths), key=lambda e: e[1])
    mst = []
    for (u, v), _len in sorted_edges:
        if union(u, v):
            mst.append((u, v))
        if len(mst) == len(nodes) - 1:
            break
    return mst


# ----------------------------------------------------------------------
# Sheaf infrastructure (Parent A)
# ----------------------------------------------------------------------
class Sheaf:
    """Cellular sheaf on a simple undirected graph.

    Each node carries a vector space ℝ^k (k = feature dimension).
    Each edge (u, v) carries a restriction map ρ_uv : ℝ^k → ℝ^k,
    represented as a k×k matrix.
    """

    def __init__(self, node_dims: Dict[int, int], edges: List[Tuple[int, int]]):
        self.node_dims = dict(node_dims)          # node → dimension (all equal here)
        self.edges = list(edges)
        self._restrictions: Dict[Tuple[int, int], np.ndarray] = {}

    def set_restriction(self, edge: Tuple[int, int], matrix: np.ndarray):
        """Assign the restriction matrix for an undirected edge.
        The same matrix is used for both orientations.
        """
        u, v = edge
        if matrix.shape != (self.node_dims[u], self.node_dims[v]):
            raise ValueError("Restriction matrix size mismatch.")
        self._restrictions[(u, v)] = matrix
        self._restrictions[(v, u)] = matrix.T  # transpose for opposite orientation

    def coboundary_matrix(self) -> np.ndarray:
        """Construct the global coboundary matrix δ.

        The matrix maps the direct sum of node spaces (size Σ dim) to the
        direct sum of edge spaces (size Σ dim).  For each oriented edge (u→v)
        the block row contains -ρ_uv in the column of u and +I in the column
        of v.
        """
        node_list = sorted(self.node_dims.keys())
        node_index = {n: i for i, n in enumerate(node_list)}
        total_node_dim = sum(self.node_dims[n] for n in node_list)

        # One block per *oriented* edge
        oriented_edges = [(u, v) for (u, v) in self.edges] + [(v, u) for (u, v) in self.edges]
        rows = len(oriented_edges) * self.node_dims[node_list[0]]
        δ = np.zeros((rows, total_node_dim))

        row = 0
        block_dim = self.node_dims[node_list[0]]
        for (u, v) in oriented_edges:
            # -ρ_uv in column of u
            col_u = sum(self.node_dims[n] for n in node_list if n < u)
            ρ = self._restrictions.get((u, v))
            if ρ is None:
                raise ValueError(f"Missing restriction for edge {(u, v)}")
            δ[row:row + block_dim, col_u:col_u + block_dim] = -ρ
            # +I in column of v
            col_v = sum(self.node_dims[n] for n in node_list if n < v)
            δ[row:row + block_dim, col_v:col_v + block_dim] = np.eye(block_dim)
            row += block_dim
        return δ


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def construct_sheaf_from_texts(texts: List[str]) -> Tuple[Sheaf, Dict[Tuple[int, int], float]]:
    """Build a Sheaf whose restrictions are scaled by Ollivier‑Ricci curvature.

    Returns the Sheaf object and the curvature dictionary.
    """
    nodes, edges, sims = build_graph(texts)
    curvature = curvature_estimate(edges, sims)

    # All node vector spaces have the same dimension = feature length
    example_vec = next(iter(nodes.values()))
    k = int(example_vec.shape[0])
    node_dims = {nid: k for nid in nodes}

    sheaf = Sheaf(node_dims, edges)

    for (u, v) in edges:
        κ = curvature[(u, v)]
        # Restriction is κ * I_k
        mat = κ * np.eye(k)
        sheaf.set_restriction((u, v), mat)

    return sheaf, curvature


def compute_homology_dimensions(sheaf: Sheaf) -> Tuple[int, int]:
    """Return (dim H0, dim H1) using rank–nullity on the coboundary matrix."""
    δ = sheaf.coboundary_matrix()
    rank = np.linalg.matrix_rank(δ, tol=1e-9)
    total_node_dim = sum(sheaf.node_dims.values())
    total_edge_dim = len(sheaf.edges) * next(iter(sheaf.node_dims.values()))
    # H0 dimension = nullity of δ (sections that are globally consistent)
    h0 = total_node_dim - rank
    # H1 dimension = nullity of δ^T (cocycles modulo coboundaries)
    h1 = total_edge_dim - rank
    return int(round(h0)), int(round(h1))


def hybrid_analysis(texts: List[str]) -> Dict[str, Any]:
    """High‑level routine that ties together graph construction, MST, and sheaf cohomology.

    Returns a dictionary with keys:
        'curvature'   – edge → κ
        'mst_edges'   – list of edges in the minimum‑spanning tree
        'h0' , 'h1'   – Betti numbers of the constructed sheaf
    """
    # Build sheaf and curvature
    sheaf, curvature = construct_sheaf_from_texts(texts)

    # Build MST using curvature‑derived lengths
    edge_lengths = [edge_length_from_curvature(curvature[e]) for e in sheaf.edges]
    mst = kruskal_mst(sheaf.node_dims, sheaf.edges, edge_lengths)

    # Cohomology
    h0, h1 = compute_homology_dimensions(sheaf)

    return {
        "curvature": curvature,
        "mst_edges": mst,
        "h0": h0,
        "h1": h1,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "I think, therefore I am. The quick brown fox jumps over the lazy dog.",
        "She sells sea shells on the sea shore; the shells she sells are surely seashells.",
        "To be, or not to be, that is the question: whether 'tis nobler in the mind.",
        "All that glitters is not gold, but many things that sparkle are indeed valuable.",
    ]
    result = hybrid_analysis(sample_texts)

    print("Edge curvatures:")
    for e, k in result["curvature"].items():
        print(f"  {e}: {k:.4f}")

    print("\nMST edges (by node index):")
    for e in result["mst_edges"]:
        print(f"  {e}")

    print(f"\nBetti numbers: H0 = {result['h0']}, H1 = {result['h1']}")
    # Verify that the program terminates without exception.