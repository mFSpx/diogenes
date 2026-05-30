# DARWIN HAMMER — match 10, survivor 7
# gen: 3
# parent_a: perceptual_dedupe.py (gen0)
# parent_b: hybrid_hybrid_rbf_surrogate_perceptual_dedupe_m57_s1.py (gen2)
# born: 2026-05-29T23:25:05Z

"""Hybrid Perceptual-RBF Deduplication Module.

This module fuses the perceptual hashing utilities from *perceptual_dedupe.py*
(parent A) with the radial‑basis‑function surrogate modeling from
*hybrid_hybrid_rbf_surrogate_perceptual_dedupe_m57_s1.py* (parent B).

Mathematical bridge
------------------
- A perceptual hash (phash) turns a numeric feature vector into a binary
  integer that captures its coarse visual/structural signature.
- The Hamming distance between two hashes provides a fast, discrete proxy for
  Euclidean similarity.
- In the hybrid system the hash is used **both** as a clustering key and as an
  *augmented feature* for the RBF surrogate: each cluster (set of points with
  identical or near‑identical hash) receives its own RBF model, and when a new
  query point arrives the surrogate of the closest hash (by Hamming distance)
  is invoked.

Thus the continuous kernel matrix of the RBF model and the discrete
Hamming‑based clustering are mathematically linked: the kernel matrix is built
from Euclidean distances of points **within** a hash cluster, while the choice
of cluster is driven by the Hamming metric on the hashes.

The module provides three high‑level hybrid operations:
1. `compute_combined_hash` – merges dhash and phash into a single integer.
2. `fit_surrogates_by_hash` – clusters data by perceptual hash and fits an
   `RBFSurrogate` per cluster.
3. `hybrid_predict` – selects the surrogate whose hash is closest (Hamming)
   to the query point and returns its prediction.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, List, Tuple, Dict

Vector = Sequence[float]

# ---------- Parent A: perceptual hashing utilities ----------
def compute_dhash(values: List[float]) -> int:
    """Difference hash: 1 bit per adjacent pair, 1 if decreasing."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: List[float]) -> int:
    """Average hash limited to first 64 values."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two integers."""
    return (a ^ b).bit_count()

def cluster_by_phash(hashes: Dict[str, int], max_distance: int = 4) -> List[List[str]]:
    """Greedy clustering of identifiers by phash Hamming distance."""
    clusters: List[List[str]] = []
    for k, h in hashes.items():
        for c in clusters:
            if hamming_distance(h, hashes[c[0]]) <= max_distance:
                c.append(k)
                break
        else:
            clusters.append([k])
    return clusters

# ---------- Parent B: RBF surrogate utilities ----------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b by Gauss‑Jordan elimination (no external libs)."""
    n = len(b)
    # Build augmented matrix
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        # Pivot selection
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        # Swap rows
        m[col], m[pivot] = m[pivot], m[col]
        # Normalize pivot row
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        # Eliminate other rows
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]

@dataclass(frozen=True)
class RBFSurrogate:
    """Radial‑basis‑function surrogate model."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Weighted sum of Gaussian kernels."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

def fit(points: Iterable[Vector],
        values: Iterable[float],
        epsilon: float = 1.0,
        ridge: float = 1e-9) -> RBFSurrogate:
    """Fit an RBF surrogate to (points, values)."""
    centers = [tuple(map(float, p)) for p in points]
    y = [float(v) for v in values]
    if not centers or len(centers) != len(y):
        raise ValueError("points and values must be non‑empty and of equal length")
    n = len(centers)
    # Build kernel matrix K_ij = φ(||c_i - c_j||)
    K: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            K[i][j] = gaussian(euclidean(centers[i], centers[j]), epsilon)
    # Ridge regularization (adds λI)
    for i in range(n):
        K[i][i] += ridge
    # Solve for weights
    weights = solve_linear(K, y)
    return RBFSurrogate(centers, weights, epsilon)

# ---------- Hybrid operations ----------
def compute_combined_hash(values: List[float]) -> int:
    """Combine dhash (high‑frequency) and phash (low‑frequency) into one int."""
    dh = compute_dhash(values)
    ph = compute_phash(values)
    # Shift dh to make room for up to 64 phash bits
    return (dh << 64) | ph

def fit_surrogates_by_hash(
    points: List[Vector],
    values: List[float],
    epsilon: float = 1.0,
    ridge: float = 1e-9,
    max_hamming: int = 2
) -> Dict[int, RBFSurrogate]:
    """
    Cluster points by their perceptual hash (phash) and fit an RBF surrogate
    per cluster. Returns a mapping hash → surrogate.
    """
    # Compute hash for each point (use phash only for clustering)
    point_hashes = [compute_phash(list(p)) for p in points]

    # Greedy clustering using Hamming distance threshold
    clusters: List[List[int]] = []
    for idx, h in enumerate(point_hashes):
        for cl in clusters:
            if hamming_distance(h, point_hashes[cl[0]]) <= max_hamming:
                cl.append(idx)
                break
        else:
            clusters.append([idx])

    # Fit surrogate per cluster
    surrogates: Dict[int, RBFSurrogate] = {}
    for cl in clusters:
        cluster_points = [points[i] for i in cl]
        cluster_values = [values[i] for i in cl]
        # Representative hash = phash of the first member
        rep_hash = point_hashes[cl[0]]
        surrogates[rep_hash] = fit(cluster_points, cluster_values, epsilon, ridge)
    return surrogates

def hybrid_predict(
    query: Vector,
    surrogates: Dict[int, RBFSurrogate],
    max_hamming: int = 4
) -> float:
    """
    Predict using the surrogate whose hash is closest (by Hamming distance) to
    the query point's hash. If no surrogate is within `max_hamming`, the nearest
    surrogate is still used.
    """
    q_hash = compute_phash(list(query))
    # Direct hit
    if q_hash in surrogates:
        return surrogates[q_hash].predict(query)

    # Find surrogate with minimal Hamming distance
    best_hash = min(surrogates.keys(),
                    key=lambda h: hamming_distance(q_hash, h))
    # Optional early‑exit if distance exceeds max_hamming (still use best)
    return surrogates[best_hash].predict(query)

# ---------- Smoke test ----------
if __name__ == "__main__":
    # Synthetic 2‑D dataset: f(x, y) = sin(x) + cos(y)
    random.seed(0)
    N = 120
    points: List[Vector] = [
        (random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi))
        for _ in range(N)
    ]
    values: List[float] = [math.sin(p[0]) + math.cos(p[1]) for p in points]

    # Fit hybrid model
    surrogates = fit_surrogates_by_hash(points, values, epsilon=1.2, ridge=1e-6)

    # Test prediction on a few random queries
    test_pts: List[Vector] = [
        (0.0, 0.0),
        (math.pi / 4, -math.pi / 3),
        (-1.2, 2.5)
    ]
    for pt in test_pts:
        pred = hybrid_predict(pt, surrogates)
        true = math.sin(pt[0]) + math.cos(pt[1])
        print(f"Query {pt}: predicted={pred:.6f}, true={true:.6f}, error={abs(pred-true):.6f}")

    # Verify that clustering function from parent A still works
    sample_hashes = {f"id_{i}": compute_phash(list(p)) for i, p in enumerate(points[:10])}
    clusters = cluster_by_phash(sample_hashes, max_distance=3)
    print(f"Parent‑A clustering produced {len(clusters)} clusters on a sample of 10 points.")