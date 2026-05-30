# DARWIN HAMMER — match 3917, survivor 7
# gen: 7
# parent_a: hybrid_hybrid_rbf_surrogate_hybrid_hybrid_endpoi_m423_s0.py (gen5)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m1708_s0.py (gen6)
# born: 2026-05-29T23:52:35Z

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple, Set
import numpy as np

Vector = Sequence[float]
Node = Hashable
Graph = Mapping[Node, Set[Node]]
FeatureVec = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
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

def similarity_matrix(hashes: Dict[Node, int]) -> Tuple[np.ndarray, List[Node]]:
    nodes = list(hashes.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = hashes[ni]
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
            else:
                hj = hashes[nj]
                d = hamming_distance(hi, hj)
                S[i, j] = 1.0 - d / 64.0
    return S, nodes

@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

def distance_1d(a: float, b: float) -> float:
    return abs(a - b)

def nearest_1d(value: float, seeds: List[float]) -> int:
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance_1d(value, seeds[i]), i))

def assign_1d(values: List[float], seeds: List[float]) -> Dict[int, List[int]]:
    regions: Dict[int, List[int]] = {i: [] for i in range(len(seeds))}
    for idx, v in enumerate(values):
        region = nearest_1d(v, seeds)
        regions[region].append(idx)
    return regions

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

class EndpointCircuitBreaker:
    def __init__(self, base_threshold: float = 0.5):
        self.base_threshold = base_threshold
        self.failures = 0
        self.threshold = base_threshold

    def update_threshold(self, similarity: float) -> None:
        factor = 0.5 + 1.5 * similarity
        self.threshold = self.base_threshold * factor

    def register(self, score: float) -> None:
        if score > self.threshold:
            self.failures += 1

    def is_tripped(self) -> bool:
        return self.failures > 0

def predict_node_scores(
    node_features: Dict[Node, FeatureVec],
    surrogate: RBFSurrogate,
) -> Dict[Node, float]:
    return {node: surrogate.predict(vec) for node, vec in node_features.items()}

def voronoi_partition_by_score(
    predictions: Dict[Node, float],
    seed_nodes: List[Node],
) -> Tuple[Dict[int, List[Node]], List[float]]:
    nodes = list(predictions.keys())
    values = [predictions[n] for n in nodes]
    seed_values = [predictions[s] for s in seed_nodes]
    region_indices = assign_1d(values, seed_values)
    regions: Dict[int, List[Node]] = {}
    for i, indices in region_indices.items():
        regions[i] = [nodes[j] for j in indices]
    return regions, seed_values

def fisher_score_for_partition(
    predictions: Dict[Node, float],
    regions: Dict[int, List[Node]],
    seed_values: List[float],
) -> Dict[Node, float]:
    scores = {}
    for i, nodes in regions.items():
        center = seed_values[i]
        for node in nodes:
            theta = predictions[node]
            scores[node] = fisher_score(theta, center, 1.0)
    return scores

def hybrid_rbf_voronoi_fisher(
    node_features: Dict[Node, FeatureVec],
    surrogate: RBFSurrogate,
    seed_nodes: List[Node],
) -> Tuple[Dict[Node, float], Dict[Node, float]]:
    predictions = predict_node_scores(node_features, surrogate)
    regions, seed_values = voronoi_partition_by_score(predictions, seed_nodes)
    scores = fisher_score_for_partition(predictions, regions, seed_values)
    return predictions, scores

# Example usage
if __name__ == "__main__":
    node_features = {
        'A': [1.0, 2.0, 3.0],
        'B': [4.0, 5.0, 6.0],
        'C': [7.0, 8.0, 9.0],
    }
    surrogate = RBFSurrogate(
        centers=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        weights=[1.0, 1.0],
    )
    seed_nodes = ['A', 'B']
    predictions, scores = hybrid_rbf_voronoi_fisher(node_features, surrogate, seed_nodes)
    print(predictions)
    print(scores)