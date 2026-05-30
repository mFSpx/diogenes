# DARWIN HAMMER — match 4431, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m788_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_ternar_m1199_s0.py (gen4)
# born: 2026-05-29T23:55:47Z

import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]

# ----------------------------------------------------------------------
# Core geometric utilities
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_seed(point: Point, seeds: Sequence[Point]) -> int:
    """Index of the seed closest to *point* (ties broken by index)."""
    if not seeds:
        raise ValueError("At least one seed is required.")
    distances = [euclidean_distance(point, s) for s in seeds]
    return int(np.argmin(distances))


def voronoi_assign(points: Sequence[Point], seeds: Sequence[Point]) -> Dict[int, List[Point]]:
    """
    Partition *points* into Voronoi regions defined by *seeds*.
    Returns a mapping ``region_index -> list_of_points``.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest_seed(p, seeds)].append(p)
    return regions


def region_centroids(regions: Dict[int, List[Point]]) -> np.ndarray:
    """
    Compute the centroid of each region.
    Returns an ``(n_regions, 2)`` array where each row is (x, y).
    Empty regions are assigned the origin (0, 0) to keep shape stable.
    """
    centroids = []
    for pts in regions.values():
        if pts:
            arr = np.asarray(pts, dtype=float)
            centroids.append(arr.mean(axis=0))
        else:
            centroids.append(np.zeros(2, dtype=float))
    return np.vstack(centroids)


# ----------------------------------------------------------------------
# Lead‑lag transform (now used in the pipeline)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Given a path ``(T, d)`` produce the lead‑lag representation
    ``(2*T‑1, 2*d)`` as described in the original hybrid paper.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time, dim).")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    out[0::2] = np.repeat(path, repeats=1, axis=0).repeat(2, axis=1)[: out.shape[0] : 2]
    out[1::2] = np.concatenate([path[1:], path[:-1]], axis=1)[: out.shape[0] - 1]
    # The last row (t = T‑1) is duplicated as in the reference implementation
    out[-1] = np.concatenate([path[-1], path[-1]])
    return out


# ----------------------------------------------------------------------
# B‑spline basis (Cox‑de Boor recursion)
# ----------------------------------------------------------------------
def _augmented_knots(grid: np.ndarray, degree: int) -> np.ndarray:
    """
    Build a clamped knot vector from a non‑uniform grid.
    The first and last knots are repeated ``degree`` times.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError("grid must be one‑dimensional.")
    start = np.full(degree, grid[0])
    end = np.full(degree, grid[-1])
    return np.concatenate([start, grid, end])


def _cox_de_boor(x: np.ndarray, knots: np.ndarray, i: int, k: int) -> np.ndarray:
    """
    Recursive evaluation of the i‑th B‑spline basis function of degree k
    at points *x* using the Cox‑de Boor formula.
    """
    if k == 0:
        # Zero‑degree (piecewise constant) basis
        left = knots[i]
        right = knots[i + 1]
        return np.where((x >= left) & (x < right), 1.0, 0.0)
    else:
        denom1 = knots[i + k] - knots[i]
        term1 = np.zeros_like(x, dtype=float)
        if denom1 > 0:
            term1 = ((x - knots[i]) / denom1) * _cox_de_boor(x, knots, i, k - 1)

        denom2 = knots[i + k + 1] - knots[i + 1]
        term2 = np.zeros_like(x, dtype=float)
        if denom2 > 0:
            term2 = ((knots[i + k + 1] - x) / denom2) * _cox_de_boor(x, knots, i + 1, k - 1)

        return term1 + term2


def bspline_basis(x: np.ndarray, grid: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Evaluate all B‑spline basis functions of a given *degree* on the
    points *x* using the knot vector derived from *grid*.
    Returns an ``(len(x), n_basis)`` matrix.
    """
    x = np.asarray(x, dtype=float)
    grid = np.asarray(grid, dtype=float)

    if degree < 1:
        raise ValueError("degree must be >= 1")
    if grid.size < 2:
        raise ValueError("grid must contain at least two points")

    knots = _augmented_knots(grid, degree)
    n_basis = len(grid) + degree - 1
    B = np.empty((x.size, n_basis), dtype=float)

    for i in range(n_basis):
        B[:, i] = _cox_de_boor(x, knots, i, degree)

    return B


# ----------------------------------------------------------------------
# Information‑theoretic utilities
# ----------------------------------------------------------------------
def shannon_entropy(counts: Sequence[int]) -> float:
    """Compute Shannon entropy (base e) of a discrete distribution."""
    total = float(sum(counts))
    if total == 0:
        return 0.0
    probs = np.array(counts, dtype=float) / total
    # Guard against log(0) by filtering zero probabilities
    nonzero = probs[probs > 0]
    return -np.sum(nonzero * np.log(nonzero))


def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) for 1‑D signals.
    This implementation follows the classic formulation with
    small stabilising constants.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.std()
    sigma_y = y.std()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Deep hybrid allocation
# ----------------------------------------------------------------------
def deep_hybrid_allocate(
    points: Sequence[Point],
    seeds: Sequence[Point],
    x: np.ndarray,
    y: np.ndarray,
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    groups: Sequence[str] = ("codex", "groq", "cohere", "local_models"),
) -> Dict[str, object]:
    """
    Allocate work units using a *deep* fusion of the constituent
    mathematical components:

    1. Voronoi partitioning → region sizes → entropy.
    2. Region centroids → lead‑lag transform → path length.
    3. Signal similarity → SSIM.
    4. All three scalars are combined (geometric mean) to weight the
       LLM‑derived share per group.

    The deterministic part of the budget is kept untouched.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Voronoi + entropy
    # ------------------------------------------------------------------
    regions = voronoi_assign(list(points), list(seeds))
    region_sizes = [len(v) for v in regions.values()]
    entropy = shannon_entropy(region_sizes)  # ≥ 0

    # ------------------------------------------------------------------
    # 2️⃣ Lead‑lag path length
    # ------------------------------------------------------------------
    centroids = region_centroids(regions)               # (n_regions, 2)
    ll_path = lead_lag_transform(centroids)             # (2*n-1, 4)
    path_length = float(np.linalg.norm(np.diff(ll_path, axis=0), axis=1).sum())

    # Normalise entropy and path length to [0, 1] using simple heuristics
    # (these constants are chosen to be reasonable for typical inputs)
    max_entropy = math.log(len(regions)) if len(regions) > 1 else 1.0
    norm_entropy = min(entropy / max_entropy, 1.0)

    # Path length is scaled by the diagonal of the bounding box of seeds
    seed_arr = np.asarray(seeds, dtype=float)
    bbox_diag = float(np.linalg.norm(seed_arr.max(axis=0) - seed_arr.min(axis=0)))
    norm_path = min(path_length / (bbox_diag * (len(seeds) * 2)), 1.0)

    # ------------------------------------------------------------------
    # 3️⃣ SSIM
    # ------------------------------------------------------------------
    ssim = compute_ssim(x, y)  # already in [0, 1]

    # ------------------------------------------------------------------
    # 4️⃣ Combine scalars (geometric mean gives balanced influence)
    # ------------------------------------------------------------------
    combined_factor = (ssim * norm_entropy * norm_path) ** (1 / 3)

    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units

    # Distribute LLM units proportionally to the combined factor,
    # but keep a minimal baseline so no group receives zero.
    baseline = 0.05  # 5 % of the per‑group share is guaranteed
    n_groups = len(groups)
    per_group_base = llm_units / n_groups * baseline
    remaining = llm_units - per_group_base * n_groups

    # Allocate the remaining units weighted by the combined factor
    weighted_share = remaining * combined_factor / n_groups

    lanes = []
    for grp in groups:
        grp_units = per_group_base + weighted_share
        lanes.append(
            {
                "group": grp,
                "llm_units": round(grp_units, 6),
                "llm_share_pct": round(100.0 * grp_units / llm_units, 6)
                if llm_units > 0
                else 0.0,
                "proof_required": True,
                "metadata": {
                    "ssim": round(ssim, 6),
                    "entropy": round(entropy, 6),
                    "norm_entropy": round(norm_entropy, 6),
                    "path_length": round(path_length, 6),
                    "norm_path": round(norm_path, 6),
                    "combined_factor": round(combined_factor, 6),
                },
            }
        )

    return {
        "total_units": round(total_units, 6),
        "deterministic_units": round(deterministic_units, 6),
        "llm_units": round(llm_units, 6),
        "lanes": lanes,
        "regions": regions,
        "entropy": round(entropy, 6),
        "path_length": round(path_length, 6),
        "ssim": round(ssim, 6),
    }


# ----------------------------------------------------------------------
# Convenience wrapper for B‑spline evaluation on the same grid used for
# region centroids (demonstrates deeper integration)
# ----------------------------------------------------------------------
def hybrid_bspline_on_centroids(
    points: Sequence[Point],
    seeds: Sequence[Point],
    degree: int = 3,
) -> np.ndarray:
    """
    Build a B‑spline basis matrix evaluated at the centroids of the
    Voronoi regions.  The knot grid is the sorted unique x‑coordinates of
    the centroids, guaranteeing that the basis respects the spatial
    distribution of the data.
    """
    regions = voronoi_assign(list(points), list(seeds))
    centroids = region_centroids(regions)[:, 0]  # use x‑coordinate only
    if centroids.size < 2:
        raise ValueError("Not enough distinct centroids for a spline.")
    grid = np.unique(np.sort(centroids))
    return bspline_basis(centroids, grid, degree)


# ----------------------------------------------------------------------
# Simple demo when run as a script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    # Synthetic geometry
    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(200)]
    seeds = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(7)]

    # Synthetic signals
    x = np.random.rand(150)
    y = np.random.rand(150)

    allocation = deep_hybrid_allocate(
        points,
        seeds,
        x,
        y,
        total_units=120.0,
        deterministic_target_pct=85.0,
    )
    print("=== Allocation Summary ===")
    for key, val in allocation.items():
        if key != "lanes":
            print(f"{key}: {val}")

    print("\n=== Lane Details ===")
    for lane in allocation["lanes"]:
        print(lane)

    # Demonstrate the deeper B‑spline integration
    B = hybrid_bspline_on_centroids(points, seeds, degree=3)
    print("\nB‑spline basis shape (centroids × basis):", B.shape)