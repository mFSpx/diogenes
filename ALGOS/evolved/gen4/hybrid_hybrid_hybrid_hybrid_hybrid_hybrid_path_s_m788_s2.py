# DARWIN HAMMER — match 788, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_pherom_m30_s2.py (gen3)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s2.py (gen3)
# born: 2026-05-29T23:31:02Z

import math
import random
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]


def euclidean_distance(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_seed(point: Point, seeds: List[Point]) -> int:
    """Index of the seed closest to *point* (break ties by index)."""
    if not seeds:
        raise ValueError("At least one seed is required.")
    return min(range(len(seeds)), key=lambda i: (euclidean_distance(point, seeds[i]), i))


def voronoi_assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to the nearest seed, returning a dict seed→points."""
    regions = defaultdict(list)
    for p in points:
        regions[nearest_seed(p, seeds)].append(p)
    # Ensure every seed appears in the dict (even if empty)
    for i in range(len(seeds)):
        regions[i] = regions.get(i, [])
    return dict(regions)


# ----------------------------------------------------------------------
# Lead‑lag transform (path → 2T‑1 × 2d matrix)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Perform the lead‑lag (or signature) transform on a 2‑D path.

    Parameters
    ----------
    path : (T, d) ndarray
        Original path.

    Returns
    -------
    out : (2T‑1, 2d) ndarray
        Interleaved lead‑lag representation.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("Path must be a 2‑D array (T, d).")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)

    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])

    out[-1] = np.concatenate([path[-1], path[-1]])
    return out


# ----------------------------------------------------------------------
# Cox–de Boor recursion for B‑spline basis evaluation
# ----------------------------------------------------------------------
def _cox_de_boor(x: np.ndarray, knots: np.ndarray, i: int, k: int) -> np.ndarray:
    """
    Recursive evaluation of a single B‑spline basis function N_{i,k} at points x.

    Parameters
    ----------
    x : (n,) ndarray
        Evaluation points.
    knots : (m,) ndarray
        Non‑decreasing knot vector.
    i : int
        Basis index.
    k : int
        Degree (order = k+1).

    Returns
    -------
    values : (n,) ndarray
        N_{i,k}(x) for each x.
    """
    if k == 0:
        left = knots[i]
        right = knots[i + 1]
        return np.where((x >= left) & (x < right), 1.0, 0.0)
    else:
        denom_left = knots[i + k] - knots[i]
        denom_right = knots[i + k + 1] - knots[i + 1]

        term_left = np.zeros_like(x)
        term_right = np.zeros_like(x)

        if denom_left > 0:
            term_left = ((x - knots[i]) / denom_left) * _cox_de_boor(x, knots, i, k - 1)
        if denom_right > 0:
            term_right = ((knots[i + k + 1] - x) / denom_right) * _cox_de_boor(x, knots, i + 1, k - 1)

        return term_left + term_right


def bspline_basis_matrix(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Evaluate the full B‑spline basis matrix B(x) where B_{j}(x_i) = N_{j,degree}(x_i).

    Parameters
    ----------
    x : (n,) ndarray
        Points at which to evaluate the basis.
    knots : (m,) ndarray
        Knot vector (must satisfy m ≥ n_basis + degree + 1).
    degree : int, default 3
        Polynomial degree (cubic = 3).

    Returns
    -------
    B : (n, n_basis) ndarray
        Basis matrix.
    """
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)

    if degree < 0:
        raise ValueError("Degree must be non‑negative.")
    if len(knots) < degree + 2:
        raise ValueError("Knot vector too short for the requested degree.")

    n_basis = len(knots) - degree - 1
    B = np.empty((x.size, n_basis), dtype=float)

    for j in range(n_basis):
        B[:, j] = _cox_de_boor(x, knots, j, degree)

    return B


# ----------------------------------------------------------------------
# Core hybrid computations
# ----------------------------------------------------------------------
def _region_bspline_sum(region_points: List[Point], degree: int = 3) -> float:
    """
    Compute a scalar summarising the B‑spline representation of a region.

    The region is first transformed by the lead‑lag map, then the first
    coordinate of the transformed points is used as the evaluation grid.
    A uniform knot vector spanning the range of that coordinate is built.
    The sum of all basis values is returned (acts as a proxy for “mass”).
    """
    if not region_points:
        return 0.0

    region_arr = np.asarray(region_points, dtype=float)          # (n, 2)
    ll = lead_lag_transform(region_arr)                         # (2n‑1, 4)
    x_vals = ll[:, 0]                                           # use first coordinate

    # Uniform knots with clamping at the ends
    min_x, max_x = x_vals.min(), x_vals.max()
    if np.isclose(min_x, max_x):
        # Degenerate case – create a tiny interval to avoid zero division
        max_x = min_x + 1e-6
    n_knots = max( degree + 2, len(x_vals) + degree + 1 )
    knots = np.linspace(min_x, max_x, n_knots)

    B = bspline_basis_matrix(x_vals, knots, degree=degree)      # (n, n_basis)
    return float(np.sum(B))


def hybrid_voronoi_pheromone(points: List[Point],
                             seeds: List[Point],
                             path: np.ndarray,
                             degree: int = 3) -> float:
    """
    Voronoi‑weighted pheromone estimate.

    For each Voronoi cell we compute a B‑spline “mass” from the points
    belonging to that cell and weight it by the cell size.
    """
    regions = voronoi_assign(points, seeds)
    total = 0.0
    for pts in regions.values():
        total += len(pts) * _region_bspline_sum(pts, degree=degree)
    return total


def hybrid_pheromone_voronoi(path: np.ndarray,
                             seeds: List[Point],
                             points: List[Point],
                             degree: int = 3) -> float:
    """
    Global pheromone signal (from the whole path) multiplied by Voronoi cell sizes.
    """
    ll = lead_lag_transform(path)
    x_vals = ll[:, 0]

    # Build a global knot vector spanning the whole transformed path
    min_x, max_x = x_vals.min(), x_vals.max()
    if np.isclose(min_x, max_x):
        max_x = min_x + 1e-6
    n_knots = max(degree + 2, len(x_vals) + degree + 1)
    knots = np.linspace(min_x, max_x, n_knots)

    global_bspline_sum = float(np.sum(bspline_basis_matrix(x_vals, knots, degree=degree)))

    regions = voronoi_assign(points, seeds)
    total = 0.0
    for pts in regions.values():
        total += len(pts) * global_bspline_sum
    return total


def hybrid_shannon_entropy(points: List[Point],
                           seeds: List[Point],
                           path: np.ndarray,
                           degree: int = 3,
                           eps: float = 1e-12) -> float:
    """
    Shannon entropy of the normalized B‑spline coefficients per Voronoi region.

    The entropy is computed on the distribution of basis‑function values
    within each region, then summed over all regions.
    """
    regions = voronoi_assign(points, seeds)
    total_entropy = 0.0

    for pts in regions.values():
        if not pts:
            continue

        ll = lead_lag_transform(np.asarray(pts, dtype=float))
        x_vals = ll[:, 0]

        min_x, max_x = x_vals.min(), x_vals.max()
        if np.isclose(min_x, max_x):
            max_x = min_x + 1e-6
        n_knots = max(degree + 2, len(x_vals) + degree + 1)
        knots = np.linspace(min_x, max_x, n_knots)

        B = bspline_basis_matrix(x_vals, knots, degree=degree)
        coeffs = B.sum(axis=0)                     # aggregate per basis function
        prob = coeffs / (coeffs.sum() + eps)       # normalise to a probability vector
        region_entropy = -np.sum(prob * np.log2(prob + eps))
        total_entropy += region_entropy

    return total_entropy


# ----------------------------------------------------------------------
# Simple demo when executed as a script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    # Generate synthetic data
    points = [(random.random(), random.random()) for _ in range(200)]
    seeds = [(random.random(), random.random()) for _ in range(15)]
    path = np.random.rand(25, 2)                     # a random walk‑like path

    vp = hybrid_voronoi_pheromone(points, seeds, path)
    pv = hybrid_pheromone_voronoi(path, seeds, points)
    he = hybrid_shannon_entropy(points, seeds, path)

    print(f"Voronoi‑Pheromone   : {vp:.6f}")
    print(f"Pheromone‑Voronoi   : {pv:.6f}")
    print(f"Shannon Entropy    : {he:.6f}")