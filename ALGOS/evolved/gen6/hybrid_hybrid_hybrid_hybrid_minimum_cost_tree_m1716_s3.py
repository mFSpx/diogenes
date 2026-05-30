# DARWIN HAMMER — match 1716, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_hybrid_m1197_s0.py (gen5)
# parent_b: minimum_cost_tree.py (gen0)
# born: 2026-05-29T23:38:35Z

"""Hybrid Algorithm: RBF‑Similarity Weighted Minimum‑Cost Tree
Parents:
- hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_hybrid_m1197_s0.py (RBF similarity matrix)
- minimum_cost_tree.py (Euclidean tree cost)

Mathematical Bridge:
The RBF similarity matrix S∈ℝ^{N×N} (Gaussian kernel on Euclidean distances) is used
as a continuous weighting factor for the edges of the tree defined in the minimum‑cost
algorithm.  For an edge (i,j) the geometric length ℓ_{ij}=‖p_i−p_j‖ is modulated by
the similarity term (1−α·S_{ij}), where α∈[0,1] controls the influence of the kernel.
The total hybrid cost is  

    C_hybrid = Σ_{(i,j)∈E} ℓ_{ij}·(1−α·S_{ij})  +  β·Σ_{v∈V} d(v)

where d(v) is the root‑to‑v path length (as in the original tree_cost) and β is a
path‑weight hyper‑parameter.  This fuses the continuous kernel space of the RBF
approach with the discrete graph‑theoretic cost of the minimum‑cost tree.

The implementation below provides the core functions to compute the similarity
matrix, generate similarity‑driven edges, and evaluate the hybrid cost.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Sequence, List, Tuple, Dict, Set, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
FeatureVec = Sequence[float]
Point = Tuple[float, float]
Node = Hashable
Edge = Tuple[Node, Node]

# ----------------------------------------------------------------------
# Parent A – RBF similarity utilities
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two feature vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_rbf_similarity_matrix(features: List[FeatureVec]) -> np.ndarray:
    """Compute the full RBF similarity matrix for a list of feature vectors."""
    n = len(features)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean(features[i], features[j])
            similarity = gaussian(dist)
            sim[i, j] = similarity
            sim[j, i] = similarity
    np.fill_diagonal(sim, 1.0)
    return sim

# ----------------------------------------------------------------------
# Parent B – Euclidean tree cost utilities
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean length of an edge between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_distances(nodes: Dict[Node, Point], edges: List[Edge], root: Node) -> Dict[Node, float]:
    """Breadth‑first accumulation of distances from root along un‑weighted edges."""
    adjacency: Dict[Node, List[Node]] = {n: [] for n in nodes}
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    dist: Dict[Node, float] = {root: 0.0}
    stack: List[Node] = [root]
    while stack:
        cur = stack.pop()
        for nxt in adjacency[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + length(nodes[cur], nodes[nxt])
                stack.append(nxt)
    return dist

# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def generate_similarity_edges(
    features: List[FeatureVec],
    node_ids: List[Node],
    threshold: float = 0.5,
) -> Tuple[List[Edge], np.ndarray]:
    """
    Build a list of edges between nodes whose RBF similarity exceeds `threshold`.
    Returns the edge list and the full similarity matrix for later use.
    """
    sim_matrix = compute_rbf_similarity_matrix(features)
    edges: List[Edge] = []
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                edges.append((node_ids[i], node_ids[j]))
    return edges, sim_matrix

def hybrid_tree_cost(
    nodes: Dict[Node, Point],
    edges: List[Edge],
    root: Node,
    sim_matrix: np.ndarray,
    node_ids: List[Node],
    path_weight: float = 0.2,
    similarity_weight: float = 0.5,
) -> float:
    """
    Compute the hybrid cost:
        Σ edge_length·(1−α·similarity)  +  β·Σ root‑to‑node distances
    where α = similarity_weight and β = path_weight.
    """
    # Map node identifier -> index in the similarity matrix
    id_to_index = {nid: idx for idx, nid in enumerate(node_ids)}

    # Material cost with similarity modulation
    material = 0.0
    for u, v in edges:
        idx_u, idx_v = id_to_index[u], id_to_index[v]
        sim = sim_matrix[idx_u, idx_v]
        edge_len = length(nodes[u], nodes[v])
        material += edge_len * (1.0 - similarity_weight * sim)

    # Path‑distance component (identical to parent B)
    dists = tree_distances(nodes, edges, root)
    path_component = path_weight * sum(dists.values())

    return material + path_component

def hybrid_score(
    features: List[FeatureVec],
    nodes: Dict[Node, Point],
    root: Node,
    threshold: float = 0.5,
    path_weight: float = 0.2,
    similarity_weight: float = 0.5,
) -> float:
    """
    End‑to‑end hybrid evaluation:
    1. Generate similarity‑driven edges.
    2. Compute the hybrid tree cost.
    Returns the final scalar score.
    """
    if len(features) != len(nodes):
        raise ValueError("Number of feature vectors must equal number of nodes.")
    node_ids = list(nodes.keys())
    edges, sim_matrix = generate_similarity_edges(features, node_ids, threshold)
    if not edges:
        raise RuntimeError("No edges were generated; adjust the similarity threshold.")
    return hybrid_tree_cost(
        nodes,
        edges,
        root,
        sim_matrix,
        node_ids,
        path_weight,
        similarity_weight,
    )

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small synthetic example with 5 nodes
    random.seed(42)
    np.random.seed(42)

    # Generate random 2‑D coordinates for nodes
    nodes: Dict[str, Point] = {
        f"N{i}": (random.uniform(0, 10), random.uniform(0, 10)) for i in range(5)
    }

    # Use the same coordinates as feature vectors (the kernel works on any vector space)
    features: List[FeatureVec] = [list(coord) for coord in nodes.values()]

    root_node = list(nodes.keys())[0]

    try:
        score = hybrid_score(
            features,
            nodes,
            root_node,
            threshold=0.4,
            path_weight=0.2,
            similarity_weight=0.6,
        )
        print(f"Hybrid cost for the generated graph: {score:.4f}")
    except Exception as exc:
        print(f"Hybrid evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)