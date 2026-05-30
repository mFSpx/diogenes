# DARWIN HAMMER — match 536, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_ternary_lens__hybrid_hybrid_sheaf__m72_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_minimu_semantic_neighbors_m298_s3.py (gen3)
# born: 2026-05-29T23:29:33Z

"""Hybrid Sheaf‑Semantic‑Bayesian Algorithm
Parents:
- hybrid_hybrid_ternary_lens__hybrid_hybrid_sheaf__m72_s1.py (Sheaf cohomology + pruning probability)
- hybrid_hybrid_hybrid_minimu_semantic_neighbors_m298_s3.py (Semantic similarity + Bayesian edge weighting)

Mathematical bridge:
Node sections of a sheaf are represented as feature vectors.  The semantic similarity
between the vectors of two incident nodes defines a *restriction map* weight.
That similarity is interpreted as a likelihood in a Bayesian update:
    posterior = prior * likelihood / marginal,
where the prior is the pruning probability supplied to the original sheaf
algorithm and the marginal is computed from the prior, likelihood and a
false‑positive rate.  The posterior probability decides whether a section
remains (kept) or is pruned.  The surviving sections define edge weights that
feed a minimum‑cost spanning‑tree (MST) computation, yielding a unified
structure that respects sheaf consistency, semantic relevance, and Bayesian
confidence.
"""

import math
import random
import sys
import pathlib
from typing import Dict, Tuple, List, Set

import numpy as np

# ----------------------------------------------------------------------
# Core Sheaf data structure (from Parent A, completed)
# ----------------------------------------------------------------------
Node = str
Edge = Tuple[Node, Node]

class Sheaf:
    """A simple sheaf over a finite graph.

    - node_dims: dimension of the vector space attached to each node.
    - edges: list of undirected edges.
    - _sections: concrete vectors stored at nodes.
    - _restrictions: linear maps (as NumPy arrays) attached to directed edges.
    """

    def __init__(self, node_dims: Dict[Node, int], edge_list: List[Edge]):
        self.node_dims = dict(node_dims)               # node -> dimension
        self.edges = [tuple(e) for e in edge_list]     # undirected
        self._sections: Dict[Node, np.ndarray] = {}
        # store restrictions for both orientations
        self._restrictions: Dict[Tuple[Node, Node], Tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Section handling
    # ------------------------------------------------------------------
    def set_section(self, node: Node, value: List[float]) -> None:
        dim = self.node_dims.get(node)
        if dim is None:
            raise KeyError(f"Node {node} not defined in node_dims")
        arr = np.array(value, dtype=float)
        if arr.shape[0] != dim:
            raise ValueError(f"Section dimension mismatch for node {node}")
        self._sections[node] = arr

    def get_section(self, node: Node) -> np.ndarray:
        return self._sections[node]

    # ------------------------------------------------------------------
    # Restriction maps (linear maps) between adjacent nodes
    # ------------------------------------------------------------------
    def set_restriction(self, edge: Edge, src_map: List[List[float]], dst_map: List[List[float]]) -> None:
        """Assign linear maps for both directions of an undirected edge.

        src_map : map from source node's space to the edge space.
        dst_map : map from destination node's space to the edge space.
        """
        u, v = edge
        src_mat = np.array(src_map, dtype=float)
        dst_mat = np.array(dst_map, dtype=float)
        # Consistency check: inner dimensions must match
        if src_mat.shape[1] != self.node_dims[u] or dst_mat.shape[1] != self.node_dims[v]:
            raise ValueError("Restriction matrix column size must match node dimension")
        self._restrictions[(u, v)] = (src_mat, dst_mat)
        self._restrictions[(v, u)] = (dst_mat, src_mat)  # reverse orientation

    def get_restriction(self, oriented_edge: Tuple[Node, Node]) -> Tuple[np.ndarray, np.ndarray]:
        return self._restrictions[oriented_edge]

    # ------------------------------------------------------------------
    # Consistency check (a very light‑weight cohomology analogue)
    # ------------------------------------------------------------------
    def is_consistent(self, edge: Edge, tol: float = 1e-6) -> bool:
        """Check whether the sections on the two ends agree after restriction."""
        u, v = edge
        if u not in self._sections or v not in self._sections:
            return False
        src_map, dst_map = self.get_restriction((u, v))
        left = src_map @ self._sections[u]
        right = dst_map @ self._sections[v]
        return np.allclose(left, right, atol=tol)

# ----------------------------------------------------------------------
# Semantic utilities (from Parent B)
# ----------------------------------------------------------------------
def semantic_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    den = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if den == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / den)


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability for Bayesian update."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, false_positive: float) -> float:
    """Posterior probability after observing evidence with given likelihood."""
    marginal = bayes_marginal(prior, likelihood, false_positive)
    if marginal == 0:
        return 0.0
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def edge_weight_from_sheaf(sheaf: Sheaf, edge: Edge) -> float:
    """Derive a weight for an edge using semantic similarity of node sections.

    The weight is defined as 1 - similarity so that higher similarity yields lower cost.
    """
    u, v = edge
    sec_u = sheaf.get_section(u).tolist()
    sec_v = sheaf.get_section(v).tolist()
    sim = semantic_similarity(sec_u, sec_v)          # in [0,1]
    return 1.0 - sim                                 # cost-like metric


def prune_sections_via_bayes(
    sheaf: Sheaf,
    prior_prune_prob: float = 0.3,
    false_positive: float = 0.1,
    keep_threshold: float = 0.5,
) -> Set[Node]:
    """Apply Bayesian pruning to each node's section.

    For each node, the likelihood is taken as the average semantic similarity
    to its neighbours.  Nodes whose posterior probability of being *useful*
    falls below ``keep_threshold`` are removed from the sheaf.

    Returns the set of nodes that survived the pruning.
    """
    survived: Set[Node] = set()
    for node in sheaf.node_dims:
        # gather neighbours
        neighbour_edges = [e for e in sheaf.edges if node in e]
        if not neighbour_edges:
            # isolated node – keep it if prior already good enough
            posterior = prior_prune_prob
        else:
            sims = []
            for e in neighbour_edges:
                other = e[1] if e[0] == node else e[0]
                if other not in sheaf._sections:
                    continue
                sims.append(semantic_similarity(
                    sheaf.get_section(node).tolist(),
                    sheaf.get_section(other).tolist()))
            likelihood = float(np.mean(sims)) if sims else 0.0
            posterior = bayes_update(prior_prune_prob, likelihood, false_positive)

        if posterior >= keep_threshold:
            survived.add(node)
        else:
            # prune: delete the section (but keep dimension info)
            sheaf._sections.pop(node, None)

    return survived


def minimum_cost_spanning_tree(
    nodes: List[Node],
    positions: Dict[Node, Tuple[float, float]],
    edge_costs: Dict[Edge, float],
) -> List[Edge]:
    """Prim's algorithm for an undirected MST using the provided edge costs."""
    if not nodes:
        return []

    # start from the first node
    visited: Set[Node] = {nodes[0]}
    mst_edges: List[Edge] = []

    # candidate edges (cost, u, v)
    import heapq
    candidates: List[Tuple[float, Node, Node]] = []
    for v in nodes[1:]:
        e = (nodes[0], v) if (nodes[0], v) in edge_costs else (v, nodes[0])
        heapq.heappush(candidates, (edge_costs[e], nodes[0], v))

    while len(visited) < len(nodes):
        cost, u, v = heapq.heappop(candidates)
        if v in visited:
            continue
        visited.add(v)
        mst_edges.append((u, v))
        # add new frontier edges
        for w in nodes:
            if w in visited:
                continue
            e = (v, w) if (v, w) in edge_costs else (w, v)
            heapq.heappush(candidates, (edge_costs[e], v, w))

    return mst_edges


def hybrid_process(
    sheaf: Sheaf,
    node_positions: Dict[Node, Tuple[float, float]],
    prior_prune: float = 0.3,
    false_pos: float = 0.1,
    keep_thr: float = 0.5,
) -> List[Edge]:
    """Run the full hybrid pipeline:

    1. Bayesian pruning of node sections.
    2. Compute semantic‑derived edge costs.
    3. Return a minimum‑cost spanning tree over the surviving nodes.
    """
    # Step 1: prune
    alive = prune_sections_via_bayes(
        sheaf, prior_prune_prob=prior_prune, false_positive=false_pos, keep_threshold=keep_thr
    )

    # Filter graph to alive vertices
    alive_edges = [e for e in sheaf.edges if e[0] in alive and e[1] in alive]

    # Step 2: compute costs
    edge_costs: Dict[Edge, float] = {}
    for e in alive_edges:
        cost = edge_weight_from_sheaf(sheaf, e)
        edge_costs[e] = cost

    # Step 3: MST
    mst = minimum_cost_spanning_tree(list(alive), node_positions, edge_costs)
    return mst

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph
    nodes = {"A", "B", "C"}
    dims = {n: 3 for n in nodes}
    edges = [("A", "B"), ("B", "C"), ("A", "C")]

    # Random but reproducible features
    random.seed(42)
    np.random.seed(42)

    sheaf = Sheaf(node_dims=dims, edge_list=edges)

    # Assign random sections (feature vectors)
    for n in nodes:
        vec = np.random.rand(dims[n]).tolist()
        sheaf.set_section(n, vec)

    # Simple identity restrictions (edge space dimension = node dimension)
    for u, v in edges:
        id_u = np.identity(dims[u]).tolist()
        id_v = np.identity(dims[v]).tolist()
        sheaf.set_restriction((u, v), id_u, id_v)

    # Arbitrary 2D positions for MST geometry (not used for costs, but required by API)
    positions = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, 1.0),
    }

    mst_edges = hybrid_process(sheaf, positions, prior_prune=0.4, false_pos=0.05, keep_thr=0.45)

    print("MST edges after hybrid processing:")
    for e in mst_edges:
        print(f"  {e[0]} – {e[1]}")
    # Verify that all edges refer to existing nodes
    assert all(u in nodes and v in nodes for u, v in mst_edges), "MST contains unknown nodes"
    print("Smoke test completed successfully.")