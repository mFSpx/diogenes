# DARWIN HAMMER — match 317, survivor 0
# gen: 3
# parent_a: hybrid_rbf_surrogate_hybrid_distributed_l_m58_s0.py (gen2)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s1.py (gen2)
# born: 2026-05-29T23:28:12Z

"""
This module integrates the radial basis functions from hybrid_rbf_surrogate_hybrid_distributed_l_m58_s0.py and 
the Gaussian distributions from hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s1.py. 
The mathematical bridge between the two structures is the use of Gaussian distributions to model uncertainty 
in the tree edges and nodes, similar to the uncertainty modeling in radial basis functions. 
In this hybrid algorithm, we use Gaussian distributions to model the uncertainty of the similarity weights 
in the hybrid maximal independent set algorithm.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

Node = int
Graph = dict[Node, set[Node]]
FeatureVec = tuple[float, float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: list[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def similarity_matrix(features: dict[Node, FeatureVec]) -> tuple[np.ndarray, list[Node]]:
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = compute_phash(list(features[ni]))
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
            else:
                hj = compute_phash(list(features[nj]))
                d = hamming_distance(hi, hj)
                S[i, j] = 1.0 - d / 64.0
    return S, nodes

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def hybrid_similarity_matrix(features: dict[Node, FeatureVec]) -> tuple[np.ndarray, list[Node]]:
    S, nodes = similarity_matrix(features)
    n = len(nodes)
    for i in range(n):
        for j in range(n):
            if i != j:
                S[i, j] *= gaussian_beam(S[i, j], 1.0, 0.1)
    return S, nodes

def modulated_probability(
    raw_p: float,
    node_idx: int,
    undecided_mask: np.ndarray,
    adjacency: np.ndarray,
    S: np.ndarray
) -> float:
    neighbor_prob = 0.0
    for neighbor in adjacency[node_idx]:
        neighbor_prob += raw_p * S[node_idx, neighbor]
    return raw_p * (1 - undecided_mask[node_idx]) + neighbor_prob

def tree_cost(nodes: dict[Node, FeatureVec], edges: list[tuple[Node, Node]], root: Node, path_weight: float = 0.2) -> float:
    adj: dict[Node, list[Node]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += euclidean(nodes[a], nodes[b])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + euclidean(nodes[a], nodes[b])
                stack.append(b)
    return material + path_weight * sum(dist.values())

def bayes_marginal(hybrid_p: float, prior: float, likelihood: float, false_positive: float) -> float:
    return hybrid_p * prior * likelihood / (hybrid_p * prior * likelihood + (1 - hybrid_p) * false_positive)

if __name__ == "__main__":
    nodes = {0: (0.0, 0.0), 1: (1.0, 1.0), 2: (2.0, 2.0)}
    edges = [(0, 1), (1, 2)]
    root = 0
    path_weight = 0.2
    features = {0: (0.1, 0.1), 1: (1.1, 1.1), 2: (2.1, 2.1)}
    S, nodes = hybrid_similarity_matrix(features)
    print("Hybrid Similarity Matrix:")
    print(S)
    print("Tree Cost:")
    print(tree_cost(nodes, edges, root, path_weight))
    print("Bayes Marginal:")
    print(bayes_marginal(0.5, 0.5, 0.5, 0.1))