# DARWIN HAMMER — match 4172, survivor 3
# gen: 7
# parent_a: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_gliner_m1566_s1.py (gen6)
# born: 2026-05-29T23:54:02Z

import numpy as np
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Iterable


@dataclass(frozen=True)
class Span:
    """Immutable representation of a textual span."""
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str


class HybridSheaf:
    """
    A lightweight sheaf structure whose sections (vectors) live on nodes
    and whose restriction maps live on directed edges.

    Parameters
    ----------
    node_dims : Mapping[int, int]
        Mapping from node identifier to the dimension of its section vector.
    edge_list : Iterable[Tuple[int, int]]
        List of directed edges (src, dst).  Undirected edges are stored twice.
    width : int, optional
        Default width for newly created sections (unused but kept for API compatibility).
    depth : int, optional
        Default depth for future extensions (unused but kept for API compatibility).
    """

    def __init__(
        self,
        node_dims: Dict[int, int],
        edge_list: Iterable[Tuple[int, int]],
        width: int = 64,
        depth: int = 4,
    ) -> None:
        self.node_dims: Dict[int, int] = dict(node_dims)
        self.edges: List[Tuple[int, int]] = list(edge_list)
        self._sections: Dict[int, np.ndarray] = {}
        self._restrictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self.width = width
        self.depth = depth

    # --------------------------------------------------------------------- #
    # Section handling
    # --------------------------------------------------------------------- #
    def set_section(self, node: int, value: Iterable[float]) -> None:
        """Assign a section vector to *node*; the vector is cast to ``float64``."""
        dim = self.node_dims.get(node)
        if dim is None:
            raise KeyError(f"Node {node} not declared in node_dims.")
        arr = np.asarray(list(value), dtype=np.float64)
        if arr.shape[0] != dim:
            raise ValueError(
                f"Section dimension mismatch for node {node}: "
                f"expected {dim}, got {arr.shape[0]}"
            )
        self._sections[node] = arr

    def get_section(self, node: int) -> np.ndarray:
        """Return the section vector for *node* (read‑only)."""
        return self._sections[node].copy()

    # --------------------------------------------------------------------- #
    # Restriction handling
    # --------------------------------------------------------------------- #
    def set_restriction(
        self,
        edge: Tuple[int, int],
        src_map: Iterable[float],
        dst_map: Iterable[float],
    ) -> None:
        """
        Define the linear restriction maps for a directed edge.

        ``src_map`` maps the source node's section into the edge space,
        ``dst_map`` maps the edge space into the destination node's section.
        Both are stored as ``float64`` arrays.
        """
        src, dst = edge
        src_arr = np.asarray(list(src_map), dtype=np.float64)
        dst_arr = np.asarray(list(dst_map), dtype=np.float64)
        if src_arr.ndim != 1 or dst_arr.ndim != 1:
            raise ValueError("Restriction maps must be 1‑dimensional.")
        self._restrictions[(src, dst)] = (src_arr, dst_arr)

    def get_restriction(self, edge: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve the restriction maps for *edge*."""
        return self._restrictions[edge]

    # --------------------------------------------------------------------- #
    # Utility
    # --------------------------------------------------------------------- #
    def nodes(self) -> List[int]:
        """Return a list of node identifiers."""
        return list(self.node_dims.keys())

    def edges_iter(self) -> Iterable[Tuple[int, int]]:
        """Iterate over stored edges."""
        return iter(self.edges)


# --------------------------------------------------------------------- #
# Information‑theoretic utilities
# --------------------------------------------------------------------- #
def fisher_score(theta: np.ndarray) -> np.ndarray:
    """
    Compute a smooth, bounded Fisher‑like score.

    The classic Fisher information for a Bernoulli parameter ``θ`` is
    ``θ(1‑θ)``.  To keep the value in a convenient range we use a logistic
    transformation and then map it to ``[0, 1]``.
    """
    theta = np.clip(theta, 1e-12, 1 - 1e-12)  # avoid log(0)
    return 1.0 / (1.0 + np.exp(-12 * (theta - 0.5)))  # steep sigmoid


def shannon_entropy(p: np.ndarray) -> float:
    """Return the Shannon entropy of a probability vector ``p`` (base‑2)."""
    p = np.asarray(p, dtype=np.float64)
    if np.any(p < 0):
        raise ValueError("Probabilities must be non‑negative.")
    if not np.isclose(p.sum(), 1.0):
        p = p / p.sum()  # normalize silently
    mask = p > 0
    return -np.sum(p[mask] * np.log2(p[mask]))


# --------------------------------------------------------------------- #
# Graph utilities – Minimum‑Cost Spanning Tree (Kruskal)
# --------------------------------------------------------------------- #
def kruskal_mst(edge_weights: Dict[Tuple[int, int], float]) -> List[Tuple[int, int, float]]:
    """
    Compute a Minimum‑Cost Spanning Tree using Kruskal's algorithm.

    Parameters
    ----------
    edge_weights : dict
        Mapping ``(u, v)`` → weight.  The graph is assumed undirected; each
        unordered pair should appear exactly once.

    Returns
    -------
    List[Tuple[int, int, float]]
        List of edges in the MST together with their weights.
    """
    # Disjoint‑set data structure
    parent: Dict[int, int] = {}
    rank: Dict[int, int] = {}

    def make_set(v: int) -> None:
        parent[v] = v
        rank[v] = 0

    def find(v: int) -> int:
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # Initialise sets
    nodes = set()
    for (u, v) in edge_weights:
        nodes.update([u, v])
    for n in nodes:
        make_set(n)

    # Sort edges by weight
    sorted_edges = sorted(edge_weights.items(), key=lambda kv: kv[1])

    mst: List[Tuple[int, int, float]] = []
    for (u, v), w in sorted_edges:
        if find(u) != find(v):
            mst.append((u, v, w))
            union(u, v)

    return mst


# --------------------------------------------------------------------- #
# Core hybrid operation
# --------------------------------------------------------------------- #
def hybrid_operation(
    spans: List[Span],
    base_graph: Dict[int, Dict[int, float]],
    sheaf: HybridSheaf,
    *,
    temperature: float = 1.0,
) -> HybridSheaf:
    """
    Fuse information‑theoretic scores with a minimum‑cost spanning tree
    and propagate the result through the sheaf.

    The procedure is:

    1. Derive a per‑node confidence vector from the span scores.
    2. Compute Fisher scores and entropies.
    3. Blend these scores into edge weights.
    4. Build an MST on the blended weights.
    5. Update sheaf restriction maps using the MST weights.
    6. Modulate node sections by the Fisher scores.

    Parameters
    ----------
    spans : list[Span]
        The extracted spans; their ``score`` field is interpreted as a confidence.
    base_graph : dict[int, dict[int, float]]
        Undirected adjacency matrix (symmetric) with raw edge costs.
    sheaf : HybridSheaf
        The sheaf to be updated.
    temperature : float, optional
        Controls the softness of the softmax used to turn confidences into
        probabilities (default ``1.0``).

    Returns
    -------
    HybridSheaf
        The same ``sheaf`` instance, mutated in‑place and also returned for
        convenience.
    """
    if not spans:
        raise ValueError("The list of spans must not be empty.")

    # ----------------------------------------------------------------- #
    # 1. Per‑node confidence from span scores
    # ----------------------------------------------------------------- #
    # Map node id ↦ confidence; we assume node ids correspond to span indices.
    # If the sheaf contains extra nodes we assign a neutral confidence of 0.5.
    node_conf: Dict[int, float] = {}
    for i, span in enumerate(spans):
        node_conf[i] = float(np.clip(span.score, 0.0, 1.0))

    for n in sheaf.nodes():
        if n not in node_conf:
            node_conf[n] = 0.5

    # ----------------------------------------------------------------- #
    # 2. Fisher scores and entropies per node
    # ----------------------------------------------------------------- #
    theta = np.array([node_conf[n] for n in sheaf.nodes()], dtype=np.float64)
    fisher = fisher_score(theta)                     # shape (num_nodes,)
    # Build a probability distribution over nodes for entropy
    probs = np.exp(theta / temperature)
    probs /= probs.sum()
    entropy = shannon_entropy(probs)                 # scalar

    # ----------------------------------------------------------------- #
    # 3. Blend Fisher scores into edge weights
    # ----------------------------------------------------------------- #
    # Original edge weights are taken from ``base_graph``; we multiply them
    # by a factor that depends on the Fisher scores of the incident nodes.
    # The factor is (f_u + f_v) / 2, encouraging edges between high‑confidence
    # nodes to be cheaper (more likely to appear in the MST).
    blended_weights: Dict[Tuple[int, int], float] = {}
    for u, nbrs in base_graph.items():
        for v, w in nbrs.items():
            if u > v:  # ensure each undirected edge is processed once
                continue
            fu = fisher[sheaf.nodes().index(u)]
            fv = fisher[sheaf.nodes().index(v)]
            factor = (fu + fv) / 2.0
            blended_weights[(u, v)] = w * (1.0 - factor) + 1e-6  # avoid zero

    # ----------------------------------------------------------------- #
    # 4. Minimum‑Cost Spanning Tree on blended weights
    # ----------------------------------------------------------------- #
    mst_edges = kruskal_mst(blended_weights)

    # ----------------------------------------------------------------- #
    # 5. Update restriction maps using MST weights
    # ----------------------------------------------------------------- #
    for u, v, w in mst_edges:
        # Retrieve existing restriction maps; if missing, create identity maps.
        if (u, v) in sheaf._restrictions:
            src_map, dst_map = sheaf._restrictions[(u, v)]
        else:
            # Identity maps of appropriate size
            src_dim = sheaf.node_dims[u]
            dst_dim = sheaf.node_dims[v]
            src_map = np.eye(min(src_dim, dst_dim), dtype=np.float64).flatten()
            dst_map = np.eye(min(src_dim, dst_dim), dtype=np.float64).flatten()
            sheaf.set_restriction((u, v), src_map, dst_map)

        # Scale maps by the MST edge weight (higher weight → stronger restriction)
        scale = math.exp(-w)  # exponential decay makes large costs weak
        sheaf._restrictions[(u, v)] = (src_map * scale, dst_map * scale)

    # ----------------------------------------------------------------- #
    # 6. Modulate node sections by Fisher scores
    # ----------------------------------------------------------------- #
    for idx, node in enumerate(sheaf.nodes()):
        if node in sheaf._sections:
            sheaf._sections[node] = sheaf._sections[node] * fisher[idx]

    # Optional: incorporate global entropy as a uniform scaling of all sections
    entropy_scale = 1.0 + entropy / math.log2(len(sheaf.nodes()))
    for node in sheaf._sections:
        sheaf._sections[node] *= entropy_scale

    return sheaf


# --------------------------------------------------------------------- #
# Example usage (executed only when run as a script)
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    # Construct a tiny example graph (undirected, symmetric)
    base_graph = {
        0: {1: 0.5, 2: 0.3},
        1: {0: 0.5, 2: 0.2},
        2: {0: 0.3, 1: 0.2},
    }

    # Initialise a sheaf with 3 nodes of dimensions 4, 5, 6
    sheaf = HybridSheaf({0: 4, 1: 5, 2: 6}, [(0, 1), (0, 2), (1, 2)])

    # Populate sections with random vectors
    rng = np.random.default_rng(seed=42)
    for n, dim in sheaf.node_dims.items():
        sheaf.set_section(n, rng.normal(size=dim))

    # Dummy spans – scores are interpreted as confidences
    spans = [
        Span(0, 5, "alpha", "A", 0.9, "backendX"),
        Span(5, 10, "beta", "B", 0.4, "backendY"),
        Span(10, 15, "gamma", "C", 0.7, "backendZ"),
    ]

    updated_sheaf = hybrid_operation(spans, base_graph, sheaf, temperature=0.5)

    # Display the final sections for inspection
    for node, vec in updated_sheaf._sections.items():
        print(f"Node {node}: section norm = {np.linalg.norm(vec):.4f}")