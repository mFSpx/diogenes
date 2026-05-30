# DARWIN HAMMER — match 4505, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_perceptual_de_hybrid_hdc_serpentin_m327_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_model_vram_sc_m61_s1.py (gen4)
# born: 2026-05-29T23:56:15Z

import math
import numpy as np
import random
import sys
from pathlib import Path
from typing import Hashable, Sequence, List, Dict, Tuple, Callable

Vector = Sequence[float]

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_dhash(values: List[float]) -> int:
    bits = 0
    for i in range(len(values) - 1): 
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

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

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (4/3) * math.pi * (length * width * height) ** (1/3)

def cluster_by_phash(hashes: Dict[str, int], max_distance: int = 4) -> List[List[str]]:
    clusters = []
    for k, h in hashes.items():
        for c in clusters:
            if hamming_distance(h, hashes[c[0]]) <= max_distance: 
                c.append(k)
                break
        else: 
            clusters.append([k])
    return clusters

def similarity_matrix(features: Dict[Hashable, Vector]) -> Tuple[np.ndarray, List[Hashable]]:
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = compute_phash(features[ni])
        for j, nj in enumerate(nodes):
            hj = compute_phash(features[nj])
            S[i, j] = gaussian(euclidean(features[ni], features[nj]))
    return S, nodes

def hybrid_hamming_distance(a: Vector, b: Vector, S: np.ndarray) -> float:
    d = euclidean(a, b)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return d + gaussian(euclidean(a, features[nodes[j]])) - S[i, j]

def hybrid_cluster_by_phash(hashes: Dict[str, int], max_distance: int = 4, S: np.ndarray = None) -> List[List[str]]:
    if S is None:
        S, _ = similarity_matrix(hashes)
    clusters = []
    for k, h in hashes.items():
        for c in clusters:
            if hamming_distance(h, hashes[c[0]]) <= max_distance: 
                c.append(k)
                break
        else: 
            clusters.append([k])
    return clusters

def hybrid_sphericity_index(length: float, width: float, height: float, S: np.ndarray) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    si = (4/3) * math.pi * (length * width * height) ** (1/3)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return si + gaussian(euclidean([length, width, height], features[nodes[j]])) - S[i, j]

if __name__ == "__main__":
    features = {1: [1.0, 2.0, 3.0], 2: [4.0, 5.0, 6.0], 3: [7.0, 8.0, 9.0]}
    S, nodes = similarity_matrix(features)
    hashes = {1: 1, 2: 2, 3: 3}
    clusters = hybrid_cluster_by_phash(hashes, 4, S)
    print(clusters)
    si = hybrid_sphericity_index(1.0, 2.0, 3.0, S)
    print(si)
    hd = hybrid_hamming_distance(features[1], features[2], S)
    print(hd)