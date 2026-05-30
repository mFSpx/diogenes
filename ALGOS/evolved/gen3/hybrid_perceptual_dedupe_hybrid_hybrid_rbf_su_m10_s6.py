# DARWIN HAMMER — match 10, survivor 6
# gen: 3
# parent_a: perceptual_dedupe.py (gen0)
# parent_b: hybrid_hybrid_rbf_surrogate_perceptual_dedupe_m57_s1.py (gen2)
# born: 2026-05-29T23:25:05Z

"""hybrid_perceptual_rbf.py
Fusion of:
- perceptual_dedupe.py (hash‑based duplicate detection)
- hybrid_hybrid_rbf_surrogate_perceptual_dedupe_m57_s1.py (RBF surrogate)

Mathematical bridge:
The perceptual hash (phash) maps a high‑dimensional feature vector to a compact
binary integer.  Vectors whose hashes are within a small Hamming distance are
empirically “perceptually” similar.  By clustering data points via this hash we
obtain groups of points that live in a locally coherent sub‑space.  Within each
cluster we fit an independent Radial‑Basis‑Function (RBF) surrogate model.
Thus the global surrogate is a piece‑wise function

    f̂(x) = Σ_{c∈C(x)} w_c · exp(−ε·‖x−μ_c‖²),

where C(x) is the hash‑cluster containing x.  The hash clustering provides the
matrix‑partitioning that feeds the linear system solved for the RBF weights,
so the two parent topologies are mathematically fused rather than merely
concatenated.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, Tuple, Dict, List

import numpy as np

Vector = Sequence[float]


# ----------------------------------------------------------------------
# Perceptual hash utilities (Parent A)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Return a 64‑bit perceptual hash of a numeric sequence.

    The hash is built by thresholding each value against the mean of the
    sequence (or the first 64 values if longer).  A bit is set to 1 when the
    value is greater-or‑equal to the mean.
    """
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


def cluster_by_phash(
    hashes: Dict[str, int], max_distance: int = 4
) -> List[List[str]]:
    """Group keys whose hashes are within ``max_distance`` Hamming distance."""
    clusters: List[List[str]] = []
    for key, h in hashes.items():
        for cluster in clusters:
            if hamming_distance(h, hashes[cluster[0]]) <= max_distance:
                cluster.append(key)
                break
        else:
            clusters.append([key])
    return clusters


# ----------------------------------------------------------------------
# RBF surrogate utilities (Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b with simple Gaussian elimination (no pivoting beyond max)."""
    n = len(b)
    # Build augmented matrix
    M = [row[:] + [rhs] for row, rhs in zip(A, b)]

    for col in range(n):
        # Partial pivoting
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        M[col], M[pivot] = M[pivot], M[col]

        # Normalize pivot row
        div = M[col][col]
        M[col] = [v / div for v in M[col]]

        # Eliminate column in other rows
        for row in range(n):
            if row == col:
                continue
            factor = M[row][col]
            M[row] = [v - factor * p for v, p in zip(M[row], M[col])]

    return [row[-1] for row in M]


@dataclass(frozen=True)
class RBFSurrogate:
    """Weighted sum of isotropic Gaussians centred on training points."""
    centers: Tuple[Tuple[float, ...], ...]
    weights: Tuple[float, ...]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Evaluate the surrogate at point ``x``."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )


def fit_rbf(
    points: Iterable[Vector],
    values: Iterable[float],
    epsilon: float = 1.0,
    ridge: float = 1e-9,
) -> RBFSurrogate:
    """Fit an RBF surrogate to (points, values) by solving the linear system.

    The kernel matrix K_ij = φ(||p_i - p_j||) is regularised with ``ridge`` on the
    diagonal before solving K·w = y.
    """
    centers = [tuple(map(float, p)) for p in points]
    y = np.asarray(list(values), dtype=float)

    if not centers or len(centers) != len(y):
        raise ValueError("points and values must be non‑empty and of equal length")

    n = len(centers)
    K: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i, ci in enumerate(centers):
        for j, cj in enumerate(centers):
            K[i][j] = gaussian(euclidean(ci, cj), epsilon)

    # Ridge regularisation
    for i in range(n):
        K[i][i] += ridge

    weights = solve_linear(K, y.tolist())
    return RBFSurrogate(centers=tuple(centers), weights=tuple(weights), epsilon=epsilon)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_cluster_surrogate(
    points: List[Vector],
    values: List[float],
    max_hash_distance: int = 4,
    epsilon: float = 1.0,
    ridge: float = 1e-9,
) -> Dict[int, RBFSurrogate]:
    """Cluster data by perceptual hash and fit an RBF surrogate per cluster.

    Returns a mapping ``hash_key → RBFSurrogate`` where ``hash_key`` is the
    representative hash of the cluster (the first member's hash).
    """
    # Step 1 – compute hashes for each point
    hashes = {idx: compute_phash(list(p)) for idx, p in enumerate(points)}

    # Step 2 – cluster indices by hash proximity
    idx_to_hash = {idx: h for idx, h in hashes.items()}
    clusters = cluster_by_phash(idx_to_hash, max_hash_distance)

    # Step 3 – fit a surrogate per cluster
    surrogate_map: Dict[int, RBFSurrogate] = {}
    for cluster in clusters:
        idxs = [int(k) for k in cluster]  # keys are stringified ints from cluster_by_phash
        cluster_pts = [points[i] for i in idxs]
        cluster_vals = [values[i] for i in idxs]
        rep_hash = hashes[idxs[0]]
        surrogate_map[rep_hash] = fit_rbf(cluster_pts, cluster_vals, epsilon, ridge)

    return surrogate_map


def hybrid_predict(
    x: Vector,
    surrogate_map: Dict[int, RBFSurrogate],
    max_hash_distance: int = 4,
) -> float:
    """Predict the target for ``x`` using the surrogate of the nearest hash cluster.

    The function searches for a surrogate whose representative hash lies within
    ``max_hash_distance`` Hamming distance from ``x``'s hash.  If none is found,
    the nearest surrogate in Euclidean space is used as a fallback.
    """
    x_hash = compute_phash(list(x))

    # Direct hash match
    for rep_hash, surrogate in surrogate_map.items():
        if hamming_distance(x_hash, rep_hash) <= max_hash_distance:
            return surrogate.predict(x)

    # Fallback: choose surrogate whose centre is closest to x
    best_val = None
    best_dist = float("inf")
    for surrogate in surrogate_map.values():
        # Use the first centre as a proxy for the cluster location
        centre = surrogate.centers[0]
        d = euclidean(x, centre)
        if d < best_dist:
            best_dist = d
            best_val = surrogate.predict(x)
    if best_val is None:
        raise RuntimeError("No surrogate available for prediction")
    return best_val


def hybrid_mse(
    test_points: List[Vector],
    test_values: List[float],
    surrogate_map: Dict[int, RBFSurrogate],
    max_hash_distance: int = 4,
) -> float:
    """Mean‑squared error of the hybrid model on a test set."""
    preds = [hybrid_predict(p, surrogate_map, max_hash_distance) for p in test_points]
    err = np.mean((np.array(preds) - np.array(test_values)) ** 2)
    return float(err)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic 2‑D dataset: a mixture of two Gaussian bumps
    random.seed(0)
    np.random.seed(0)

    def sample_blob(center: Tuple[float, float], n: int, scale: float = 0.5):
        return (np.random.randn(n, 2) * scale + center).tolist()

    points_a = sample_blob((0.0, 0.0), 50)
    points_b = sample_blob((3.0, 3.0), 50)
    points = points_a + points_b

    # Target function: sum of coordinates plus a small sinusoidal term
    values = [p[0] + p[1] + 0.2 * math.sin(3 * p[0]) for p in points]

    # Build hybrid model
    surrogates = hybrid_cluster_surrogate(
        points, values, max_hash_distance=2, epsilon=1.5, ridge=1e-6
    )

    # Evaluate on a held‑out set
    test_pts = sample_blob((1.5, 1.5), 30, scale=0.7)
    test_vals = [p[0] + p[1] + 0.2 * math.sin(3 * p[0]) for p in test_pts]

    mse = hybrid_mse(test_pts, test_vals, surrogates, max_hash_distance=2)
    print(f"Hybrid model MSE on test set: {mse:.6f}")

    # Quick sanity check: predict a point known to belong to cluster A
    sample = points[0]
    pred = hybrid_predict(sample, surrogates, max_hash_distance=2)
    print(f"Sample point {sample} → true {values[0]:.4f}, pred {pred:.4f}")