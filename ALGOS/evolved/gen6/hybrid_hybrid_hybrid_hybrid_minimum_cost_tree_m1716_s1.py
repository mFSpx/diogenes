# DARWIN HAMMER — match 1716, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_hybrid_m1197_s0.py (gen5)
# parent_b: minimum_cost_tree.py (gen0)
# born: 2026-05-29T23:38:35Z

"""
Hybrid Algorithm: hybrid_rbf_minimum_cost_tree.py

Parents:
- hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_hybrid_m1197_s0.py (RBF similarity matrix & ternary decision-making)
- minimum_cost_tree.py (Minimum-cost tree scoring)

Mathematical Bridge:
The mathematical bridge between the two parent algorithms lies in the application of 
the RBF-derived similarity matrix to the edge weighting in the minimum-cost tree. 
By feeding each row of the RBF similarity matrix as the edge weights in the tree, 
we can effectively incorporate the kernel-based similarity into the tree's cost 
calculation. The tree's material cost is replaced by the RBF similarity between 
adjacent nodes.

"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Hashable, Sequence, List, Dict, Set, Tuple, Any

Node = Hashable
Graph = Dict[Node, Set[Node]]
FeatureVec = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two feature vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_rbf_similarity_matrix(features: List[FeatureVec]) -> np.ndarray:
    """Compute RBF similarity matrix."""
    num_nodes = len(features)
    similarity_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            dist = euclidean(features[i], features[j])
            similarity = gaussian(dist)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    return similarity_matrix

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def hybrid_tree_cost(nodes: Dict[str, Tuple[float, float]], 
                      edges: List[Tuple[str, str]], 
                      features: List[FeatureVec], 
                      root: str, 
                      path_weight: float = 0.2) -> float:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    rbf_similarities = compute_rbf_similarity_matrix(features)
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
        node_a_idx = list(nodes.keys()).index(a)
        node_b_idx = list(nodes.keys()).index(b)
        material += rbf_similarities[node_a_idx, node_b_idx]
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    return material + path_weight * sum(dist.values())

def get_hybrid_edge_weights(nodes: Dict[str, Tuple[float, float]], 
                            edges: List[Tuple[str, str]], 
                            features: List[FeatureVec]) -> Dict[Tuple[str, str], float]:
    rbf_similarities = compute_rbf_similarity_matrix(features)
    edge_weights = {}
    for a, b in edges:
        node_a_idx = list(nodes.keys()).index(a)
        node_b_idx = list(nodes.keys()).index(b)
        edge_weights[(a, b)] = rbf_similarities[node_a_idx, node_b_idx]
    return edge_weights

def hybrid_ternary_decision(edges: List[Tuple[str, str]], 
                             edge_weights: Dict[Tuple[str, str], float], 
                             threshold: float) -> List[Tuple[str, str]]:
    decisions = []
    for edge in edges:
        if edge_weights[edge] > threshold:
            decisions.append(edge)
    return decisions

if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (1, 0), 'C': (1, 1), 'D': (0, 1)}
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    features = [[0, 0], [1, 0], [1, 1], [0, 1]]
    root = 'A'
    cost = hybrid_tree_cost(nodes, edges, features, root)
    print(f"Hybrid tree cost: {cost}")
    edge_weights = get_hybrid_edge_weights(nodes, edges, features)
    decisions = hybrid_ternary_decision(edges, edge_weights, 0.5)
    print(f"Hybrid ternary decisions: {decisions}")