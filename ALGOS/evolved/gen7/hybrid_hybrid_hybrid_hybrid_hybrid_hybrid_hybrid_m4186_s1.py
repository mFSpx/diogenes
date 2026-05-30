# DARWIN HAMMER — match 4186, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_minimum_cost_tree_m1716_s4.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_vorono_m2690_s5.py (gen6)
# born: 2026-05-29T23:54:05Z

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
    """Breadth-first accumulation of distances from root along unweighted edges."""
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
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_similarity_matrix(features: List[FeatureVec], points: List[Point], epsilon_f: float = 1.0, epsilon_g: float = 1.0) -> np.ndarray:
    """
    Compute the joint similarity matrix, which is the element-wise product of 
    the feature-based RBF similarity and the geometric RBF similarity to seed centroids.
    """
    n = len(features)
    sim_feature = compute_rbf_similarity_matrix(features)
    sim_geo = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dist = length(points[i], points[j])
            similarity = gaussian(dist, epsilon_g)
            sim_geo[i, j] = similarity
            sim_geo[j, i] = similarity
    np.fill_diagonal(sim_geo, 1.0)
    return sim_feature * sim_geo

def hybrid_voronoi_rbf_retrieval(features: List[FeatureVec], points: List[Point], seeds: List[Point], epsilon_f: float = 1.0, epsilon_g: float = 1.0) -> np.ndarray:
    """
    Perform Voronoi-aware, RBF-weighted memory readout.
    """
    n = len(features)
    sim = hybrid_similarity_matrix(features, points, epsilon_f, epsilon_g)
    retrieval = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            retrieval[i, j] = sim[i, j] / (1 + euclidean(features[i], features[j]))
    return retrieval

def hybrid_minimum_cost_tree(nodes: Dict[Node, Point], edges: List[Edge], root: Node, features: List[FeatureVec], points: List[Point], epsilon_f: float = 1.0, epsilon_g: float = 1.0) -> Dict[Node, float]:
    """
    Build a minimum cost tree using the joint similarity matrix.
    """
    sim = hybrid_similarity_matrix(features, points, epsilon_f, epsilon_g)
    n = len(features)
    distances = np.zeros(n, dtype=float)
    for i in range(n):
        distances[i] = np.min([sim[i, j] * length(points[i], points[j]) for j in range(n) if i != j])
    dist: Dict[Node, float] = {root: 0.0}
    stack: List[Node] = [root]
    while stack:
        cur = stack.pop()
        for nxt in [node for edge in edges for node in edge if node != cur and (cur, node) in edges or (node, cur) in edges]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + distances[list(nodes.keys()).index(nxt)]
                stack.append(nxt)
    return dist

if __name__ == "__main__":
    # Smoke test
    features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    seeds = [(1.5, 2.5), (3.5, 4.5), (5.5, 6.5)]
    nodes = {0: (1.0, 2.0), 1: (3.0, 4.0), 2: (5.0, 6.0)}
    edges = [(0, 1), (1, 2)]
    root = 0
    print(hybrid_similarity_matrix(features, points))
    print(hybrid_voronoi_rbf_retrieval(features, points, seeds))
    print(hybrid_minimum_cost_tree(nodes, edges, root, features, points))