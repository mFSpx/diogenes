# DARWIN HAMMER — match 2101, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m1009_s4.py (gen4)
# born: 2026-05-29T23:40:59Z

import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]
FeatureVec = List[float]

def euclidean_distance(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the seed closest to *point* (break ties by index)."""
    if not seeds:
        raise ValueError("At least one seed point is required")
    return min(range(len(seeds)), key=lambda i: (euclidean_distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Voronoi‑like assignment of *points* to the nearest *seed*."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Multivector algebra (grade‑0 and grade‑1 only, sufficient for similarity)
# ----------------------------------------------------------------------
class Multivector:
    """
    Very small geometric algebra implementation supporting:

    * grade‑0 (scalar) and grade‑1 (vector) components,
    * geometric product,
    * inner product,
    * magnitude,
    * cosine similarity.
    """
    def __init__(self, components: List[float]):
        # store as a NumPy array for speed and broadcasting
        self.vec = np.asarray(components, dtype=np.float64)

    def __repr__(self) -> str:
        return f"Multivector({self.vec.tolist()})"

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------
    def geometric_product(self, other: "Multivector") -> "Multivector":
        """Geometric product = inner + outer (outer is dropped for grade‑1)."""
        # For pure vectors a·b = aᵀb (scalar), a∧b is bivector (ignored here)
        scalar = float(np.dot(self.vec, other.vec))
        return Multivector([scalar])

    def inner_product(self, other: "Multivector") -> float:
        """Scalar inner product (dot product)."""
        return float(np.dot(self.vec, other.vec))

    def magnitude(self) -> float:
        """Euclidean norm of the vector part."""
        return float(np.linalg.norm(self.vec))

    def cosine_similarity(self, other: "Multivector") -> float:
        """Cosine of the angle between two vectors; safe for zero vectors."""
        norm_self = self.magnitude()
        norm_other = other.magnitude()
        if norm_self == 0.0 or norm_other == 0.0:
            return 0.0
        return self.inner_product(other) / (norm_self * norm_other)

# ----------------------------------------------------------------------
# Structural similarity (SSIM) for feature vectors
# ----------------------------------------------------------------------
def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Compute a 1‑D SSIM index between two equal‑length vectors.
    If *dynamic_range* is None it is inferred from the data (max‑min).
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if dynamic_range is None:
        dynamic_range = max(x_arr.max(), y_arr.max()) - min(x_arr.min(), y_arr.min())
        if dynamic_range == 0:
            dynamic_range = 1.0  # avoid division by zero for constant signals

    mean_x = x_arr.mean()
    mean_y = y_arr.mean()
    var_x = x_arr.var()
    var_y = y_arr.var()
    cov_xy = ((x_arr - mean_x) * (y_arr - mean_y)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)
    denominator = (mean_x ** 2 + mean_y ** 2 + c1) * (var_x + var_y + c2)

    return float(numerator / denominator)

# ----------------------------------------------------------------------
# Radial Basis Function surrogate with deep hybrid integration
# ----------------------------------------------------------------------
def _rbf_kernel(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-(epsilon * r) ** 2)

def hybrid_rbf_surrogate(
    points: List[Point],
    seeds: List[Point],
    feature_vectors: List[FeatureVec],
    epsilon: float = 1.0,
) -> Dict[int, Dict[str, Any]]:
    """
    Assign points to seeds, then compute a surrogate for each region that
    combines:

    * RBF interpolation of the spatial coordinates,
    * Multivector cosine similarity of the region's feature vector with its neighbours,
    * SSIM between consecutive feature vectors (cyclic).

    The return value is a mapping ``region_id → {region_points, surrogate_center,
    similarity_weights, ssim}``.
    """
    if len(seeds) != len(feature_vectors):
        raise ValueError("Number of seeds must match number of feature vectors")

    # 1️⃣ Voronoi assignment
    regions = assign(points, seeds)

    # 2️⃣ Build Multivectors for each seed's feature vector
    mvectors = [Multivector(fv) for fv in feature_vectors]

    # 3️⃣ Pairwise cosine similarities (used as RBF amplitude modifiers)
    cosine_mat = np.zeros((len(seeds), len(seeds)), dtype=np.float64)
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            cosine_mat[i, j] = mvectors[i].cosine_similarity(mvectors[j])

    # 4️⃣ SSIM values for a cyclic neighbour chain (i ↔ i+1)
    ssim_vals = np.zeros(len(seeds), dtype=np.float64)
    for i in range(len(seeds)):
        ssim_vals[i] = compute_ssim(
            feature_vectors[i],
            feature_vectors[(i + 1) % len(seeds)],
        )

    # 5️⃣ Build surrogate per region
    result: Dict[int, Dict[str, Any]] = {}
    for idx, pts in regions.items():
        if not pts:
            # Empty region – fall back to the seed location
            surrogate_center = np.array(seeds[idx], dtype=np.float64)
        else:
            # Compute weighted RBF centre
            # Distance from each point to the seed
            dists = np.array([euclidean_distance(p, seeds[idx]) for p in pts], dtype=np.float64)
            # RBF weights modulated by cosine similarity to *all* other seeds
            # (the more similar a seed is to its neighbours, the larger its influence)
            similarity_factor = cosine_mat[idx].mean()
            rbf_weights = np.array([_rbf_kernel(r, epsilon) for r in dists]) * similarity_factor
            # Normalise
            if rbf_weights.sum() == 0:
                rbf_weights = np.ones_like(rbf_weights) / len(rbf_weights)
            else:
                rbf_weights /= rbf_weights.sum()
            # Weighted centroid
            pts_arr = np.asarray(pts, dtype=np.float64)
            surrogate_center = (rbf_weights[:, None] * pts_arr).sum(axis=0)

        result[idx] = {
            "region_points": pts,
            "surrogate_center": surrogate_center,
            "similarity_weights": cosine_mat[idx].tolist(),
            "ssim_with_next": float(ssim_vals[idx]),
        }

    return result

# ----------------------------------------------------------------------
# Weekday‑based weight vector (unchanged but with better validation)
# ----------------------------------------------------------------------
def weekday_weight_vector(groups: List[str], dow: int) -> np.ndarray:
    """
    Produce a smooth cyclic weight vector for *groups* based on day‑of‑week *dow*.
    *dow* follows Python's ``datetime.weekday()`` convention (Mon=0 … Sun=6).
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")

    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)

# ----------------------------------------------------------------------
# Simple Doomsday helper (kept for backward compatibility)
# ----------------------------------------------------------------------
def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index (Mon=0 … Sun=6) for the given Gregorian date."""
    import datetime as dt

    return dt.date(year, month, day).weekday()

# ----------------------------------------------------------------------
# Demo / sanity‑check block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample geometry
    points = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 1.0)]
    seeds = [(0.0, 0.0), (5.0, 5.0)]
    feature_vectors = [
        [0.2, 0.8, 0.5],
        [0.9, 0.1, 0.4],
    ]

    surrogate = hybrid_rbf_surrogate(points, seeds, feature_vectors, epsilon=0.8)
    for region_id, data in surrogate.items():
        print(f"Region {region_id}:")
        print(f"  points           -> {data['region_points']}")
        print(f"  surrogate_center -> {data['surrogate_center']}")
        print(f"  similarity_weights -> {data['similarity_weights']}")
        print(f"  ssim_with_next   -> {data['ssim_with_next']:.4f}")

    groups = ["codex", "groq", "cohere", "local_models"]
    dow = doomsday(2026, 5, 29)  # Monday -> 0
    print("Weekday weight vector:", weekday_weight_vector(groups, dow))