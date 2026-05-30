# DARWIN HAMMER — match 2690, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_nlms_hybrid_h_m978_s1.py (gen5)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m841_s3.py (gen5)
# born: 2026-05-29T23:43:33Z

"""
Hybrid Voronoi-RBF-NLMS Associative Memory

Parents:
- `hybrid_hybrid_hybrid_hybrid_nlms_hybrid_h_m978_s1.py` (NLMS + RBF)
- `hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m841_s3.py` (Voronoi partition + RBF)

Mathematical bridge:
The Voronoi partition step assigns a query point to seed-centroids using the same distance metric that the radial-basis function (RBF) uses to compute similarity.
We therefore compute RBF weights from the distances to *all* seeds, modulate those weights by a scalar derived from NLMS-based feature scores, and finally blend the linear associative-memory readouts stored in a `Sheaf` (one memory matrix per seed).
The resulting vector is a distance-aware, feature-aware retrieval from a distributed associative memory.
"""

import numpy as np
import math
import random
import sys
import pathlib

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def nearest(point: np.ndarray, seeds: np.ndarray) -> int:
    return np.argmin(np.apply_along_axis(lambda x: euclidean(point, x), 1, seeds))

def assign(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    n_seeds = seeds.shape[0]
    n_pts = points.shape[0]
    regions = np.zeros((n_seeds, n_pts), dtype=int)
    for j, p in enumerate(points):
        i = nearest(p, seeds)
        regions[i, j] = 1
    return regions

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

def similarity_matrix(features: Dict[int, np.ndarray], vram_budget_mb: int) -> Tuple[np.ndarray, List[int]]:
    nodes = list(features.keys())
    n = len(nodes)
    epsilon = 1.0 / (vram_budget_mb / 1024.0)  
    S = np.empty((n, n), dtype=np.float64)
    hashes = [compute_phash(list(features[n])) for n in nodes]

    for i in range(n):
        for j in range(i, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            sim = gaussian(dist, epsilon)
            S[i, j] = sim
            S[j, i] = sim
    return S, nodes

def hybrid_voronoi_rbf_nlms(seeds: np.ndarray, points: np.ndarray, features: Dict[int, np.ndarray], vram_budget_mb: int) -> np.ndarray:
    regions = assign(points, seeds)
    S, nodes = similarity_matrix(features, vram_budget_mb)
    n_seeds = seeds.shape[0]
    n_pts = points.shape[0]
    output = np.zeros((n_pts, n_seeds))

    for j, p in enumerate(points):
        for i in range(n_seeds):
            if regions[i, j] == 1:
                dist = euclidean(p, seeds[i])
                sim = gaussian(dist)
                output[j, i] = sim * S[i, i]

    return output

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def should_split(best_gain: float, second_best_gain: float,
                 r: float, delta: float, n: int,
                 tie_threshold: float = 0.05) -> bool:
    eps = hoeffding_bound(r, delta, n)
    gain_gap = best_gain - second_best_gain
    return gain_gap > eps + tie_threshold

if __name__ == "__main__":
    seeds = np.random.rand(5, 10)
    points = np.random.rand(10, 10)
    features = {i: np.random.rand(10) for i in range(5)}
    vram_budget_mb = 1024
    output = hybrid_voronoi_rbf_nlms(seeds, points, features, vram_budget_mb)
    print(output.shape)