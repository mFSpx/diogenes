# DARWIN HAMMER — match 4505, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_perceptual_de_hybrid_hdc_serpentin_m327_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_model_vram_sc_m61_s1.py (gen4)
# born: 2026-05-29T23:56:15Z

import numpy as np
import math
import random
from dataclasses import dataclass
from typing import Hashable, Sequence, List, Dict, Set, Tuple

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

Node = Hashable
Graph = Dict[Node, Set[Node]]
FeatureVec = Sequence[float]
Vector = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float], n_bits: int = 64) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:n_bits]:
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
            hj = compute_phash(list(features[nj]))
            S[i, j] = gaussian(hamming_distance(hi, hj) / (2 ** 10))
    return S, nodes

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (36 * math.pi * (length * width * height) ** 2) ** (1/3) / (length + width + height)

def create_bipolar_vector(feature_vec: FeatureVec, S: np.ndarray, nodes: List[Node]) -> Vector:
    n = len(nodes)
    bipolar_vector = [0.0] * n
    hi = compute_phash(list(feature_vec))
    for i, node in enumerate(nodes):
        hj = compute_phash(list(features[node]))
        bipolar_vector[i] = gaussian(hamming_distance(hi, hj) / (2 ** 10))
    return bipolar_vector

def compute_dhash(values: list[float]) -> int:
    bits = 0
    for i in range(len(values) - 1): 
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def cluster_by_phash(hashes: dict[str, int], max_distance: int = 4) -> list[list[str]]:
    clusters = []
    for k, h in hashes.items():
        clustered = False
        for c in clusters:
            if hamming_distance(h, hashes[c[0]]) <= max_distance: 
                c.append(k)
                clustered = True
                break
        if not clustered:
            clusters.append([k])
    return clusters

if __name__ == "__main__":
    features = {i: [random.random() for _ in range(10)] for i in range(10)}
    S, nodes = similarity_matrix(features)
    bipolar_vector = create_bipolar_vector(features[0], S, nodes)
    print(S.shape, bipolar_vector)
    hashes = {str(i): compute_phash(features[i]) for i in features}
    clusters = cluster_by_phash(hashes)
    print(clusters)