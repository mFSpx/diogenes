# DARWIN HAMMER — match 58, survivor 0
# gen: 2
# parent_a: rbf_surrogate.py (gen0)
# parent_b: hybrid_distributed_leader_e_perceptual_dedupe_m16_s2.py (gen1)
# born: 2026-05-29T23:24:02Z

"""
This module provides a hybrid algorithm that fuses the governing equations of 
'rbf_surrogate.py' and 'hybrid_distributed_leader_e_perceptual_dedupe_m16_s2.py'. 
The mathematical bridge lies in the use of radial basis functions (RBFs) to model 
the perceptual similarity between nodes in the graph. The RBFs are used to compute 
the similarity weights in the hybrid maximal independent set algorithm.

The 'rbf_surrogate.py' algorithm uses RBFs to approximate a function, while the 
'hybrid_distributed_leader_e_perceptual_dedupe_m16_s2.py' algorithm uses 
perceptual hashing to cluster nodes. In this hybrid algorithm, we use the RBFs 
to model the similarity between nodes based on their feature vectors, and then 
use this similarity to modulate the broadcast probability in the hybrid maximal 
independent set algorithm.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from typing import Hashable, Sequence, List, Dict, Set, Tuple

Node = Hashable
Graph = Dict[Node, Set[Node]]
FeatureVec = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def similarity_matrix(features: Dict[Node, FeatureVec]) -> Tuple[np.ndarray, List[Node]]:
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

def modulated_probability(
    raw_p: float,
    node_idx: int,
    undecided_mask: np.ndarray,
    adjacency: np.ndarray,
    similarity: np.ndarray,
    epsilon: float = 1.0
) -> float:
    neigh_mask = adjacency[node_idx] & undecided_mask
    if not np.any(neigh_mask):
        return raw_p
    weights = []
    for idx in range(len(neigh_mask)):
        if neigh_mask[idx]:
            distance = euclidean(list(features[nodes[idx]]), list(features[nodes[node_idx]]))
            weights.append(gaussian(distance, epsilon))
    avg_weight = sum(weights) / len(weights)
    return raw_p * avg_weight

def hybrid_maximal_independent_set(
    graph: Graph,
    features: Dict[Node, FeatureVec],
    phases: int = 8,
    seed: int | str | None = None,
    epsilon: float = 1.0
) -> Set[Node]:
    global nodes
    hashes = {n: compute_phash(list(features[n])) for n in graph}
    sim_mat, nodes = similarity_matrix(hashes)

    idx_of = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    adjacency = np.zeros((n, n), dtype=bool)
    for u, nbrs in graph.items():
        iu = idx_of[u]
        for v in nbrs:
            if v in idx_of:
                iv = idx_of[v]
                adjacency[iu, iv] = True
                adjacency[iv, iu] = True  # undirected

    rng = random.Random(seed)
    undecided = np.ones(n, dtype=bool)   # boolean mask of nodes still in play
    leaders_mask = np.zeros(n, dtype=bool)

    for phase in range(1, phases + 1):
        if not undecided.any():
            break
        raw_p = 1.0 / (2 ** (phases - phase))

        broadcast_flags = np.zeros(n, dtype=bool)
        for i, u in enumerate(nodes):
            if undecided[i]:
                p = modulated_probability(raw_p, i, undecided, adjacency, sim_mat, epsilon)
                broadcast_flags[i] = rng.random() < p

        leaders_mask[broadcast_flags & undecided] = True
        undecided[~(leaders_mask | broadcast_flags)] = False

    return set(nodes[i] for i, leader in enumerate(leaders_mask) if leader)

def predict(rbf_surrogate: 'RBFSurrogate', x: FeatureVec) -> float:
    return sum(w * gaussian(euclidean(x, c), rbf_surrogate.epsilon) for w, c in zip(rbf_surrogate.weights, rbf_surrogate.centers))

@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

if __name__ == "__main__":
    graph = {
        'A': {'B', 'C'},
        'B': {'A', 'D'},
        'C': {'A', 'D'},
        'D': {'B', 'C'}
    }
    features = {
        'A': [1.0, 2.0, 3.0],
        'B': [4.0, 5.0, 6.0],
        'C': [7.0, 8.0, 9.0],
        'D': [10.0, 11.0, 12.0]
    }
    leaders = hybrid_maximal_independent_set(graph, features)
    print(leaders)