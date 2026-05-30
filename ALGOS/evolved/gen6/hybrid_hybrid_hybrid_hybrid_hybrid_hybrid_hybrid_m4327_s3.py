# DARWIN HAMMER — match 4327, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_sheaf_cohomology_m1530_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hard_t_ollivier_ricci_curva_m393_s1.py (gen4)
# born: 2026-05-29T23:55:04Z

import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

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
        curv[(v, u)] = curv[(u, v)]  # Undirected graph
    return curv


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
        num_nodes = len(self.node_dims)
        num_edges = len(self.edges)
        matrix = np.zeros((num_edges, num_nodes), dtype=float)
        for e_idx, edge in enumerate(self.edges):
            u, v = edge
            restriction = self._restrictions[edge]
            matrix[e_idx, u] = -1
            matrix[e_idx, v] = 1
        return matrix


def hscc(texts: List[str]) -> Tuple[Sheaf, List[Tuple[int, int]]]:
    nodes, edges, sims = build_graph(texts)
    curv = curvature_estimate(edges, sims)
    edge_lengths = [edge_length_from_curvature(curv[edge]) for edge in edges]
    mst = kruskal_mst(nodes, edges, edge_lengths)

    # Create sheaf with curvature-driven restrictions
    k = nodes[0].shape[0]  # Feature dimension
    sheaf = Sheaf({node: k for node in nodes}, edges)
    for edge in edges:
        u, v = edge
        kappa = curv[edge]
        restriction = kappa * np.eye(k)
        sheaf.set_restriction(edge, restriction)

    return sheaf, mst


def main():
    texts = ["This is a test.", "This test is only a test."]
    sheaf, mst = hscc(texts)
    print(sheaf.coboundary_matrix())
    print(mst)


if __name__ == "__main__":
    main()