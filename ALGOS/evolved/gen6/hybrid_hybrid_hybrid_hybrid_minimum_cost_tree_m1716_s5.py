# DARWIN HAMMER — match 1716, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_hybrid_m1197_s0.py (gen5)
# parent_b: minimum_cost_tree.py (gen0)
# born: 2026-05-29T23:38:35Z

import math
import random
import sys
from dataclasses import dataclass
from typing import Sequence, List, Tuple, Dict, Set, Hashable, Iterable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Basic Types
# ----------------------------------------------------------------------
FeatureVec = Sequence[float]
Point = Tuple[float, float]
Node = Hashable
Edge = Tuple[Node, Node]

# ----------------------------------------------------------------------
# RBF Similarity Utilities (Parent A)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two feature vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_rbf_similarity_matrix(
    features: List[FeatureVec], epsilon: float = 1.0
) -> np.ndarray:
    """Full RBF similarity matrix for a list of feature vectors."""
    n = len(features)
    sim = np.empty((n, n), dtype=float)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            dist = euclidean(features[i], features[j])
            similarity = gaussian(dist, epsilon)
            sim[i, j] = similarity
            sim[j, i] = similarity
    return sim


# ----------------------------------------------------------------------
# Graph Utilities (Parent B)
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean length of an edge between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_adjacency(
    nodes: Dict[Node, Point], edges: Iterable[Edge]
) -> Dict[Node, List[Tuple[Node, float]]]:
    """Adjacency list with Euclidean edge lengths."""
    adj: Dict[Node, List[Tuple[Node, float]]] = {n: [] for n in nodes}
    for u, v in edges:
        w = length(nodes[u], nodes[v])
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def weighted_tree_distances(
    nodes: Dict[Node, Point],
    edges: List[Edge],
    root: Node,
    weight_factor: float,
    sim_matrix: np.ndarray,
    node_index: Dict[Node, int],
) -> Dict[Node, float]:
    """
    BFS accumulation of distances from root where each edge length is scaled
    by (1 - weight_factor * similarity).  Guarantees the same scaling used in
    the hybrid material cost.
    """
    adj = {n: [] for n in nodes}
    for u, v in edges:
        i, j = node_index[u], node_index[v]
        sim = sim_matrix[i, j]
        scaled = length(nodes[u], nodes[v]) * (1.0 - weight_factor * sim)
        adj[u].append((v, scaled))
        adj[v].append((u, scaled))

    dist: Dict[Node, float] = {root: 0.0}
    stack: List[Node] = [root]
    while stack:
        cur = stack.pop()
        for nxt, w in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + w
                stack.append(nxt)
    return dist


# ----------------------------------------------------------------------
# Union‑Find for Kruskal (ensures a spanning tree)
# ----------------------------------------------------------------------
class UnionFind:
    def __init__(self, elements: Iterable[Node]) -> None:
        self.parent: Dict[Node, Node] = {e: e for e in elements}
        self.rank: Dict[Node, int] = {e: 0 for e in elements}

    def find(self, x: Node) -> Node:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: Node, y: Node) -> bool:
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True


# ----------------------------------------------------------------------
# Hybrid Core – Deeper Integration
# ----------------------------------------------------------------------
def compute_combined_edge_weights(
    nodes: Dict[Node, Point],
    features: List[FeatureVec],
    node_ids: List[Node],
    sim_matrix: np.ndarray,
    similarity_weight: float,
) -> List[Tuple[float, Node, Node]]:
    """
    Returns a list of (combined_weight, u, v) for every unordered pair.
    Combined weight = Euclidean length * (1 - α·similarity).
    """
    n = len(node_ids)
    edges: List[Tuple[float, Node, Node]] = []
    for i in range(n):
        for j in range(i + 1, n):
            u, v = node_ids[i], node_ids[j]
            eucl = length(nodes[u], nodes[v])
            sim = sim_matrix[i, j]
            combined = eucl * (1.0 - similarity_weight * sim)
            edges.append((combined, u, v))
    return edges


def kruskal_mst(
    weighted_edges: List[Tuple[float, Node, Node]], nodes: Set[Node]
) -> List[Edge]:
    """
    Classic Kruskal algorithm returning a minimum spanning tree (as a list of edges)
    that connects all nodes with minimal summed combined weight.
    """
    uf = UnionFind(nodes)
    mst: List[Edge] = []
    for w, u, v in sorted(weighted_edges, key=lambda x: x[0]):
        if uf.union(u, v):
            mst.append((u, v))
            if len(mst) == len(nodes) - 1:
                break
    if len(mst) != len(nodes) - 1:
        raise RuntimeError("Failed to build a spanning tree; graph may be disconnected.")
    return mst


def hybrid_tree_cost(
    nodes: Dict[Node, Point],
    features: List[FeatureVec],
    root: Node,
    path_weight: float = 0.2,
    similarity_weight: float = 0.5,
    epsilon: float = 1.0,
) -> float:
    """
    Full hybrid cost:
        Σ_e ℓ_e·(1−α·S_e)          (material term)
      + β· Σ_v d_root→v          (path term, using the same scaling)
    where:
        α = similarity_weight
        β = path_weight
        ℓ_e = Euclidean length of edge e
        S_e = RBF similarity between the two incident nodes
    The tree is forced to be a minimum‑spanning tree under the combined weight,
    guaranteeing connectivity and a true tree structure.
    """
    if len(nodes) != len(features):
        raise ValueError("Number of nodes must equal number of feature vectors.")

    node_ids = list(nodes.keys())
    node_index = {nid: idx for idx, nid in enumerate(node_ids)}

    # 1️⃣ Compute similarity matrix once
    sim_matrix = compute_rbf_similarity_matrix(features, epsilon)

    # 2️⃣ Build combined edge weights and extract MST
    weighted_edges = compute_combined_edge_weights(
        nodes, features, node_ids, sim_matrix, similarity_weight
    )
    mst_edges = kruskal_mst(weighted_edges, set(node_ids))

    # 3️⃣ Material term (sum of combined weights)
    material = sum(
        length(nodes[u], nodes[v]) * (1.0 - similarity_weight * sim_matrix[node_index[u], node_index[v]])
        for u, v in mst_edges
    )

    # 4️⃣ Path term – distances from root using the same scaled edge lengths
    dists = weighted_tree_distances(
        nodes,
        mst_edges,
        root,
        similarity_weight,
        sim_matrix,
        node_index,
    )
    path_component = path_weight * sum(dists.values())

    return material + path_component


def hybrid_score(
    features: List[FeatureVec],
    nodes: Dict[Node, Point],
    root: Node,
    path_weight: float = 0.2,
    similarity_weight: float = 0.5,
    epsilon: float = 1.0,
) -> float:
    """
    Convenience wrapper that validates inputs and returns the hybrid cost.
    """
    return hybrid_tree_cost(
        nodes,
        features,
        root,
        path_weight=path_weight,
        similarity_weight=similarity_weight,
        epsilon=epsilon,
    )


# ----------------------------------------------------------------------
# Smoke test (executed when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Generate 7 random 2‑D points
    nodes: Dict[str, Point] = {
        f"N{i}": (random.uniform(0, 10), random.uniform(0, 10)) for i in range(7)
    }

    # Use the same coordinates as feature vectors (RBF operates on the same space)
    features: List[FeatureVec] = [list(coord) for coord in nodes.values()]

    root_node = next(iter(nodes))  # first key

    try:
        score = hybrid_score(
            features,
            nodes,
            root_node,
            path_weight=0.25,
            similarity_weight=0.6,
            epsilon=0.8,
        )
        print(f"Hybrid cost (MST‑based) = {score:.4f}")
    except Exception as exc:
        print(f"Hybrid evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)