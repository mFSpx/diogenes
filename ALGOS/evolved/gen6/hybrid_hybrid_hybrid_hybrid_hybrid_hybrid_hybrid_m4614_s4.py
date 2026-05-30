# DARWIN HAMMER — match 4614, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m784_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_ternar_m1215_s4.py (gen4)
# born: 2026-05-29T23:56:55Z

"""Hybrid Voronoi‑Fisher‑SSIM Algorithm
===================================

Parent algorithms
-----------------
* **A** – *hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m784_s0.py*  
  Provides a Voronoi partitioning of 2‑D points and a geometric‑algebra
  ``Multivector`` implementation.  The key mathematical object is the
  **epistemic certainty** of each Voronoi cell, obtained from a Fisher
  information calculation.

* **B** – *hybrid_hybrid_hybrid_fisher_hybrid_hybrid_ternar_m1215_s4.py*  
  Supplies a Gaussian‑beam based Fisher information score, a structural
  similarity index (SSIM) for comparing two one‑dimensional signals and a
  bandit‑style decision loop.

Mathematical bridge
-------------------
Both parents rely on a *Gaussian* description of a quantity (intensity,
beam, probability density).  In **A** the Gaussian beam is used to weight
the dimensionality reduction of multivectors; in **B** the same Gaussian
produces the Fisher information, i.e. a measure of certainty.  The SSIM
in **B** is a similarity metric that can be applied to the *histograms*
of points belonging to two Voronoi cells.  By using the Fisher
information of a cell as a *confidence weight* and the SSIM between two
cells as a *structural similarity weight*, we obtain a unified objective
that can drive a multi‑armed bandit which selects seed updates for the
Voronoi diagram.

The resulting hybrid system therefore consists of:

1. **Voronoi partition** of a point cloud (`assign`).
2. **Fisher certainty** for each cell (`cell_fisher_certainty`).
3. **SSIM similarity** between pairs of cells (`cell_ssim_matrix`).
4. **Bandit‑guided seed relocation** that maximises a weighted sum of
   certainty and similarity (`bandit_optimize_seeds`).

The three public functions below illustrate this fused pipeline.
"""

import math
import random
import sys
import pathlib
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Iterable, FrozenSet

# ----------------------------------------------------------------------
# Basic geometric utilities (from Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the seed closest to *point* (break ties by index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)),
               key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """
    Voronoi assignment of *points* to the nearest *seeds*.
    Returns a dict ``cell_index -> list_of_points``.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Gaussian‑beam, Fisher information and SSIM (from Parent B)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def _ssim_statistics(x: np.ndarray, y: np.ndarray, k1: float, k2: float, C1: float, C2: float):
    """Helper that returns mean, variance and covariance needed for SSIM."""
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x2 = x.var()
    sigma_y2 = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    return mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy

def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Structural Similarity Index (SSIM) for two 1‑D signals.
    The implementation follows the original SSIM formulation.
    """
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy = _ssim_statistics(
        x_arr, y_arr, k1, k2, C1, C2
    )
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return numerator / denominator

# ----------------------------------------------------------------------
# Geometric algebra utilities (trimmed version of Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices: Iterable[int]) -> Tuple[List[int], int]:
    """
    Sorts *indices* into canonical order, returns the sorted list and the
    sign (+1 / -1) of the permutation.  Duplicate indices cancel out
    (exterior product rule).
    """
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel duplicate basis vectors
                lst.pop(j)
                lst.pop(j)  # remove the element that shifted left
                n -= 2
                i = -1  # restart outer loop because length changed
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    combined = list(blade_a) + list(blade_b)
    sorted_indices, sign = _blade_sign(combined)
    return frozenset(sorted_indices), sign

class Multivector:
    """
    Very small subset of a full geometric algebra implementation.
    ``components`` maps a frozenset of basis indices to a scalar coefficient.
    """
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = n
        self.components: Dict[FrozenSet[int], float] = {
            k: float(v) for k, v in components.items()
        }

    def __add__(self, other: 'Multivector') -> 'Multivector':
        result = self.components.copy()
        for k, v in other.components.items():
            result[k] = result.get(k, 0.0) + v
        return Multivector(result, self.n)

    def __mul__(self, other: 'Multivector') -> 'Multivector':
        """Geometric product."""
        result: Dict[FrozenSet[int], float] = {}
        for a_key, a_val in self.components.items():
            for b_key, b_val in other.components.items():
                blade, sign = _multiply_blades(a_key, b_key)
                result[blade] = result.get(blade, 0.0) + sign * a_val * b_val
        return Multivector(result, self.n)

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) component."""
        return self.components.get(frozenset(), 0.0)

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def cell_fisher_certainty(
    cells: Dict[int, List[Point]],
    width: float = 0.1,
) -> Dict[int, float]:
    """
    Compute a Fisher‑information based certainty for each Voronoi cell.
    The cell centre (mean of its points) acts as ``theta`` in the Gaussian.
    The cell index itself is used as the Gaussian centre to keep the
    expression dimension‑less.
    """
    certainty: Dict[int, float] = {}
    for idx, pts in cells.items():
        if not pts:
            certainty[idx] = 0.0
            continue
        xs, ys = zip(*pts)
        theta = (np.mean(xs) + np.mean(ys)) / 2.0
        centre = float(idx)  # deterministic centre per cell
        certainty[idx] = fisher_score(theta, centre, width)
    return certainty

def cell_ssim_matrix(
    cells: Dict[int, List[Point]],
    bins: int = 16,
    range_min: float = -1.0,
    range_max: float = 1.0,
) -> np.ndarray:
    """
    Build an ``N x N`` SSIM matrix where ``N`` is the number of cells.
    For each cell we create a 1‑D histogram of distances of its points
    from the cell centroid and compare histograms with SSIM.
    """
    n = len(cells)
    histograms: List[np.ndarray] = []
    for idx in range(n):
        pts = cells[idx]
        if not pts:
            histograms.append(np.zeros(bins))
            continue
        xs, ys = zip(*pts)
        centroid = (np.mean(xs), np.mean(ys))
        dists = [distance(p, centroid) for p in pts]
        hist, _ = np.histogram(dists, bins=bins, range=(range_min, range_max), density=True)
        histograms.append(hist.astype(np.float64))

    ssim_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            s = compute_ssim(histograms[i].tolist(), histograms[j].tolist(),
                             dynamic_range=1.0)
            ssim_mat[i, j] = s
            ssim_mat[j, i] = s
    return ssim_mat

def bandit_optimize_seeds(
    points: List[Point],
    seeds: List[Point],
    iterations: int = 50,
    epsilon: float = 0.2,
    width: float = 0.1,
) -> List[Point]:
    """
    Multi‑armed bandit that treats each seed as an arm.
    At each iteration:

    1. Partition the points (Voronoi) using the current seeds.
    2. Compute Fisher certainty per cell.
    3. Compute SSIM similarity matrix between cells.
    4. Define the *reward* of an arm as the weighted sum
       ``certainty + mean_similarity`` of its cell.
    5. Choose an arm with epsilon‑greedy policy and move its seed
       towards the centroid of its cell (small step).

    Returns the final list of seeds.
    """
    n = len(seeds)
    # Estimated value of each arm (seed)
    values = np.zeros(n)
    counts = np.zeros(n, dtype=int)

    for it in range(iterations):
        # 1. Voronoi partition
        cells = assign(points, seeds)

        # 2. Certainty per cell
        certainty = cell_fisher_certainty(cells, width=width)

        # 3. Similarity matrix
        ssim_mat = cell_ssim_matrix(cells)

        # 4. Reward per arm
        rewards = np.empty(n)
        for i in range(n):
            mean_sim = (ssim_mat[i].sum() - 1.0) / (n - 1) if n > 1 else 0.0
            rewards[i] = certainty[i] + mean_sim

        # 5. Epsilon‑greedy selection
        if random.random() < epsilon:
            arm = random.randrange(n)
        else:
            arm = int(np.argmax(values))

        # Update statistics
        counts[arm] += 1
        values[arm] += (rewards[arm] - values[arm]) / counts[arm]   # incremental mean

        # Move the selected seed slightly towards its cell centroid
        cell_pts = cells[arm]
        if cell_pts:
            xs, ys = zip(*cell_pts)
            centroid = (np.mean(xs), np.mean(ys))
            step = 0.1  # learning rate
            old = seeds[arm]
            new_seed = (old[0] + step * (centroid[0] - old[0]),
                        old[1] + step * (centroid[1] - old[1]))
            seeds[arm] = new_seed

    return seeds

# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. generate a random cloud of points inside [-1, 1]²
    random.seed(42)
    num_points = 500
    points = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(num_points)]

    # 2. initialise 5 random seeds
    seeds = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(5)]

    # 3. run the hybrid bandit optimiser
    final_seeds = bandit_optimize_seeds(points, seeds, iterations=30, epsilon=0.15)

    # 4. final partition and diagnostics
    final_cells = assign(points, final_seeds)
    cert = cell_fisher_certainty(final_cells)
    ssim = cell_ssim_matrix(final_cells)

    print("Final seed positions:")
    for i, s in enumerate(final_seeds):
        print(f"  Seed {i}: {s}")

    print("\nFisher certainty per cell:")
    for i, v in cert.items():
        print(f"  Cell {i}: {v:.4f}")

    print("\nSSIM matrix (rounded):")
    print(np.round(ssim, 3))