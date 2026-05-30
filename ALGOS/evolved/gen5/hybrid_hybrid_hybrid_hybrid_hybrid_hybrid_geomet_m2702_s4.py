# DARWIN HAMMER — match 2702, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_sheaf__m1068_s2.py (gen4)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s2.py (gen2)
# born: 2026-05-29T23:43:55Z

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set, FrozenSet, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = int
FeatureVec = Tuple[float, ...]
Point = Tuple[float, float]

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two feature vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def compute_phash(values: List[float]) -> int:
    """64‑bit perceptual hash based on median threshold."""
    if not values:
        return 0
    median = np.median(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= median)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Parent A – Enhanced similarity utilities
# ----------------------------------------------------------------------
def gaussian_similarity_matrix(
    features: Dict[Node, FeatureVec],
    epsilon: float = 1.0,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, List[Node]]:
    """
    Build a symmetric similarity matrix S where

        S[i, j] = (1 - Hamming(phash_i, phash_j)/64) ** alpha *
                  exp(-epsilon**2 * ||x_i - x_j||^2)

    The exponent ``alpha`` controls the influence of the binary hash similarity.
    Returns the matrix and the ordered list of nodes.
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)

    # Pre‑compute phashes once
    phashes = {node: compute_phash(list(features[node])) for node in nodes}

    for i, ni in enumerate(nodes):
        hi = phashes[ni]
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
                continue

            hj = phashes[nj]
            ham = hamming_distance(hi, hj) / 64.0
            hash_sim = (1.0 - ham) ** alpha

            d_euc = euclidean(features[ni], features[nj])
            rbf = gaussian(d_euc, epsilon)

            S[i, j] = hash_sim * rbf
    return S, nodes


# ----------------------------------------------------------------------
# Parent B – Voronoi partitioning utilities
# ----------------------------------------------------------------------
def distance2d(a: Point, b: Point) -> float:
    """2‑D Euclidean distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the nearest seed (ties broken by index)."""
    if not seeds:
        raise ValueError("At least one seed required")
    return min(range(len(seeds)), key=lambda i: (distance2d(point, seeds[i]), i))


def voronoi_partition(points: List[Point], seeds: List[Point]) -> Dict[int, List[int]]:
    """
    Assign each point to the nearest seed.
    Returns a mapping seed_index → list of point indices.
    """
    regions: Dict[int, List[int]] = {i: [] for i in range(len(seeds))}
    for idx, p in enumerate(points):
        regions[nearest(p, seeds)].append(idx)
    return regions


# ----------------------------------------------------------------------
# Geometric Algebra core (enhanced)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """
    Sort ``indices`` into canonical order and return the sign of the permutation.
    Duplicate indices cancel the blade (they are removed).
    """
    lst = list(indices)
    sign = 1
    i = 0
    while i < len(lst) - 1:
        j = 0
        while j < len(lst) - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel duplicate basis vectors
                del lst[j : j + 2]
                i = -1  # restart outer loop
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(
    blade_a: FrozenSet[int], blade_b: FrozenSet[int]
) -> Tuple[FrozenSet[int], int]:
    """
    Geometric product of two basis blades *ignoring* the metric.
    The metric will be injected later when multiplying grade‑1 vectors.
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """
    Simple exterior algebra (wedge) implementation with optional metric support.
    ``components`` maps a frozen set of basis indices to a scalar coefficient.
    """

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # prune near‑zero entries
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-12
        }

    # ------------------------------------------------------------------
    # Basic algebraic operations
    # ------------------------------------------------------------------
    def grade(self, k: int) -> "Multivector":
        """Return the grade‑k part of the multivector."""
        return Multivector(
            {b: c for b, c in self.components.items() if len(b) == k}, self.n
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(
            self.components.items(),
            key=lambda x: (len(x[0]), sorted(x[0])),
        ):
            label = "1" if not blade else "e" + "".join(str(i) for i in sorted(blade))
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for b, c in other.components.items():
            result[b] = result.get(b, 0.0) + c
        return Multivector(result, self.n)

    def __xor__(self, other: "Multivector") -> "Multivector":
        """Exterior (wedge) product."""
        result: Dict[FrozenSet[int], float] = {}
        for a_blade, a_coef in self.components.items():
            for b_blade, b_coef in other.components.items():
                new_blade, sign = _multiply_blades(a_blade, b_blade)
                result[new_blade] = result.get(new_blade, 0.0) + sign * a_coef * b_coef
        return Multivector(result, self.n)

    # ------------------------------------------------------------------
    # Metric‑aware geometric product (only for vectors)
    # ------------------------------------------------------------------
    def geometric_product(
        self, other: "Multivector", metric: np.ndarray
    ) -> "Multivector":
        """
        Compute the geometric product using a symmetric positive‑definite metric
        matrix ``metric`` of shape (n, n).  The implementation is exact for
        grade‑1 elements (vectors) and falls back to wedge‑only for higher grades.
        """
        if self.n != metric.shape[0] or other.n != metric.shape[0]:
            raise ValueError("Metric dimension does not match multivector space")

        # Fast path for pure vectors
        if all(len(b) == 1 for b in self.components) and all(
            len(b) == 1 for b in other.components
        ):
            # a·b = Σ_i Σ_j a_i b_j g_ij   (scalar part)
            # a∧b = Σ_i Σ_j a_i b_j (e_i ^ e_j)   (bivector part)
            a_vec = np.zeros(self.n)
            b_vec = np.zeros(self.n)
            for blade, coef in self.components.items():
                i = next(iter(blade))
                a_vec[i] = coef
            for blade, coef in other.components.items():
                j = next(iter(blade))
                b_vec[j] = coef

            scalar = float(a_vec @ metric @ b_vec)
            bivector = {}
            for i in range(self.n):
                for j in range(self.n):
                    if i == j:
                        continue
                    coef = a_vec[i] * b_vec[j] - a_vec[j] * b_vec[i]
                    if abs(coef) > 1e-12:
                        blade = frozenset({i, j})
                        bivector[blade] = bivector.get(blade, 0.0) + coef
            # combine scalar and bivector parts
            components = {}
            if abs(scalar) > 1e-12:
                components[frozenset()] = scalar
            components.update(bivector)
            return Multivector(components, self.n)

        # General case: fallback to exterior product (no metric contraction)
        return self ^ other


# ----------------------------------------------------------------------
# Hybrid operation: region‑wise multivector aggregation
# ----------------------------------------------------------------------
def region_multivector_aggregation(
    features: Dict[Node, FeatureVec],
    points: List[Point],
    seeds: List[Point],
    epsilon: float = 1.0,
    alpha: float = 0.5,
) -> Dict[int, Multivector]:
    """
    1. Compute the hybrid similarity matrix K using both hash‑based and Gaussian terms.
    2. Partition the points into Voronoi regions.
    3. For each region R, build a grade‑1 multivector

           M_R = Σ_{i∈R} w_i e_i

       where

           w_i = Σ_{j} K[i, j] * 𝟙_{j∈R} / Σ_{p,q∈R} K[p, q]

       The denominator normalises the total influence inside the region.
    Returns a mapping ``region_index → Multivector``.
    """
    # ------------------------------------------------------------------
    # Step 1 – similarity matrix
    # ------------------------------------------------------------------
    K, ordered_nodes = gaussian_similarity_matrix(
        features, epsilon=epsilon, alpha=alpha
    )
    node_to_index = {node: idx for idx, node in enumerate(ordered_nodes)}

    # ------------------------------------------------------------------
    # Step 2 – Voronoi partition
    # ------------------------------------------------------------------
    regions = voronoi_partition(points, seeds)

    # ------------------------------------------------------------------
    # Step 3 – build multivectors
    # ------------------------------------------------------------------
    region_mv: Dict[int, Multivector] = {}
    n = len(ordered_nodes)

    # Pre‑compute region masks for fast indexing
    region_masks: Dict[int, np.ndarray] = {}
    for r_idx, point_indices in regions.items():
        mask = np.zeros(n, dtype=bool)
        for pt_idx in point_indices:
            # map point index to node index (assume one‑to‑one correspondence)
            if pt_idx < n:
                mask[pt_idx] = True
        region_masks[r_idx] = mask

    for r_idx, mask in region_masks.items():
        if not mask.any():
            # empty region → zero multivector
            region_mv[r_idx] = Multivector({}, n)
            continue

        # Numerator: weighted influence of each node inside the region
        numerator = K[:, mask].sum(axis=1)  # shape (n,)

        # Denominator: total similarity inside the region
        denom = float(K[np.ix_(mask, mask)].sum())
        if denom == 0.0:
            weights = np.zeros(n)
        else:
            weights = numerator / denom

        # Build grade‑1 multivector
        components: Dict[FrozenSet[int], float] = {}
        for idx, w in enumerate(weights):
            if abs(w) > 1e-12:
                components[frozenset({idx})] = w
        region_mv[r_idx] = Multivector(components, n)

    return region_mv


# ----------------------------------------------------------------------
# Helper: metric‑aware norm of a vector‑grade multivector
# ----------------------------------------------------------------------
def vector_norm(mv: Multivector, metric: np.ndarray) -> float:
    """
    Compute the norm ‖v‖ = sqrt(v · v) for a pure vector multivector ``mv``.
    The inner product uses the supplied metric matrix.
    """
    if not all(len(b) == 1 for b in mv.components):
        raise ValueError("vector_norm expects a pure grade‑1 multivector")
    # Build coefficient vector
    vec = np.zeros(mv.n)
    for blade, coef in mv.components.items():
        i = next(iter(blade))
        vec[i] = coef
    return float(math.sqrt(vec @ metric @ vec))


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data: 10 nodes, 5‑dimensional features, random 2‑D points
    random.seed(42)
    np.random.seed(42)

    num_nodes = 10
    dim_feat = 5
    features = {
        i: tuple(np.random.rand(dim_feat).tolist()) for i in range(num_nodes)
    }
    points = [tuple(np.random.rand(2).tolist()) for _ in range(num_nodes)]
    seeds = [tuple(np.random.rand(2).tolist()) for _ in range(3)]

    region_mvs = region_multivector_aggregation(
        features, points, seeds, epsilon=1.2, alpha=0.7
    )

    # Build metric from similarity matrix (used for norm calculations)
    K, _ = gaussian_similarity_matrix(features, epsilon=1.2, alpha=0.7)

    print("Region multivectors:")
    for r, mv in region_mvs.items():
        print(f"Region {r}: {mv}")
        if any(len(b) == 1 for b in mv.components):
            norm = vector_norm(mv, K)
            print(f"  → norm (metric‑aware) = {norm:.4f}")
        else:
            print("  → not a pure vector, norm undefined")