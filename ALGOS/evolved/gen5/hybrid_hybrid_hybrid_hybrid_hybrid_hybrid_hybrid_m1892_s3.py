# DARWIN HAMMER — match 1892, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_physarum_netw_m334_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m788_s1.py (gen4)
# born: 2026-05-29T23:39:33Z

"""Hybrid Voronoi‑Physarum Multivector Model
========================================

Parent algorithms
-----------------
* **hybrid_hybrid_hybrid_geomet_hybrid_physarum_netw_m334_s1.py** – defines a
  `Multivector` class (Clifford algebra) and a flux‑based conductance update
  used in physarum‑network simulations.
* **hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m788_s1.py** – provides
  geometric utilities (distance, Voronoi assignment, lead‑lag transform,
  B‑spline basis) for handling point clouds and paths.

Mathematical bridge
-------------------
The bridge is built by interpreting the *region conductances* of a Voronoi
decomposition as a multivector whose basis vectors correspond to the seed
indices.  A path travelling through the point cloud generates a *flux*
multivector via a B‑spline weighting of the lead‑lag transformed path.
Both multivectors live in the same Clifford algebra, therefore they can be
combined with the geometric product to obtain a unified update rule that
simultaneously respects the physarum conductance dynamics and the geometric
structure of the Voronoi‑path system.

The core hybrid update is


g_{new} = g + α·(g * f) – β·g


where `g` is the conductance multivector, `f` the flux multivector, `*` the
geometric product, `α` a learning rate and `β` a decay coefficient.
"""

import math
import random
import sys
import pathlib
import numpy as np
from typing import Dict, FrozenSet, Tuple, List

# ----------------------------------------------------------------------
# 1. Clifford algebra – Multivector
# ----------------------------------------------------------------------
class Multivector:
    """Sparse multivector in an n‑dimensional Euclidean Clifford algebra."""
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # drop near‑zero components
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    # ------------------------------------------------------------------
    # basic selectors
    # ------------------------------------------------------------------
    def grade(self, k: int) -> "Multivector":
        return Multivector({b: c for b, c in self.components.items() if len(b) == k}, self.n)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    # ------------------------------------------------------------------
    # representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), list(x[0]))):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    # ------------------------------------------------------------------
    # addition / subtraction
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        neg = Multivector({b: -c for b, c in other.components.items()}, other.n)
        return self + neg

    # ------------------------------------------------------------------
    # geometric product
    # ------------------------------------------------------------------
    @staticmethod
    def _blade_product(a: FrozenSet[int], b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
        """Geometric product of two basis blades.
        Returns (resulting blade, sign) where sign = ±1.
        """
        # work with ordered lists
        a_list = sorted(a)
        b_list = sorted(b)
        result = a_list + b_list
        sign = 1

        # cancel duplicate indices (e_i * e_i = 1) while tracking swaps
        i = 0
        while i < len(result):
            # count how many times result[i] appears later
            dup_idx = None
            for j in range(i + 1, len(result)):
                if result[j] == result[i]:
                    dup_idx = j
                    break
            if dup_idx is not None:
                # remove both occurrences
                del result[dup_idx]
                del result[i]
                # each removal of a duplicate corresponds to an even number of swaps,
                # therefore sign does not change.
                i = 0  # restart scan because list changed
                continue
            i += 1

        # bring the list to sorted order, counting swaps (sign flips)
        # simple bubble‑sort counting
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if result[i] > result[j]:
                    result[i], result[j] = result[j], result[i]
                    sign = -sign

        return frozenset(result), sign

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                blade_res, sign = self._blade_product(blade_a, blade_b)
                result[blade_res] = result.get(blade_res, 0.0) + sign * coef_a * coef_b
        return Multivector(result, self.n)

    # ------------------------------------------------------------------
    # scalar multiplication
    # ------------------------------------------------------------------
    def __rmul__(self, scalar: float) -> "Multivector":
        return Multivector({b: scalar * c for b, c in self.components.items()}, self.n)

# ----------------------------------------------------------------------
# 2. Geometric utilities (from Parent B)
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to the nearest seed (Voronoi regions)."""
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def lead_lag_transform(path: List[Point]) -> np.ndarray:
    """Convert a path into its lead‑lag representation."""
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Cubic B‑spline basis matrix evaluated at points x over knot vector grid."""
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    N = len(x)
    B = np.zeros((N, len(t) - 1), dtype=np.float64)

    # order‑1 (piecewise constant) basis
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    # recursive Cox‑de Boor
    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]

            term_l = ((x - t[i]) / denom_l * B[:, i]) if denom_l > 0 else np.zeros(N)
            term_r = ((t[i + order] - x) / denom_r * B[:, i + 1]) if denom_r > 0 else np.zeros(N)

            B_new[:, i] = (term_l + term_r) / (order - 1)
        B = B_new
    return B

# ----------------------------------------------------------------------
# 3. Hybrid core functions
# ----------------------------------------------------------------------
def region_conductance_multivector(points: List[Point],
                                   seeds: List[Point]) -> Multivector:
    """
    Compute a multivector whose scalar part is zero and whose grade‑1
    components encode the total Euclidean distance of points belonging to
    each Voronoi region.
    """
    regions = assign(points, seeds)
    comps: Dict[FrozenSet[int], float] = {}
    for idx, pts in regions.items():
        total = sum(distance(p, seeds[idx]) for p in pts)
        if total != 0.0:
            comps[frozenset({idx})] = total
    return Multivector(comps, n=len(seeds))

def path_flux_multivector(path: List[Point],
                          seeds: List[Point],
                          alpha: float = 1.0) -> Multivector:
    """
    Produce a flux multivector from a path.
    The lead‑lag transform is B‑spline‑smoothed; the resulting scalar field
    is projected onto seed directions by weighting with inverse distance.
    """
    if not path:
        raise ValueError("path must contain at least one point")
    # lead‑lag representation (2D → 4D vectors)
    ll = lead_lag_transform(path)                     # shape (2T‑1, 4)
    # parameterise by cumulative Euclidean length
    seg_len = np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1)
    t = np.concatenate(([0.0], np.cumsum(seg_len)))
    t_norm = t / t[-1] if t[-1] != 0 else t

    # B‑spline basis on normalized parameter
    grid = np.linspace(0.0, 1.0, num= max(4, len(t_norm)))
    B = bspline_basis(t_norm, grid, k=3)              # (2T‑1) × nbasis
    # smear the lead‑lag vectors through the basis
    smooth = B.T @ ll                                    # nbasis × 4

    # collapse the 4‑dimensional vectors to a scalar field (norm)
    field = np.linalg.norm(smooth, axis=1)               # nbasis

    # project onto seeds using inverse distance weighting
    comps: Dict[FrozenSet[int], float] = {}
    for i, seed in enumerate(seeds):
        dists = np.array([distance(pt, seed) for pt in path])
        weight = np.sum(field / (dists + 1e-9))          # avoid division by zero
        if weight != 0.0:
            comps[frozenset({i})] = alpha * weight
    return Multivector(comps, n=len(seeds))

def hybrid_update(conductance: Multivector,
                  flux: Multivector,
                  lr: float = 0.1,
                  decay: float = 0.01) -> Multivector:
    """
    Perform the physarum‑style update inside the Clifford algebra:
        g_new = g + lr·(g * f) – decay·g
    """
    return conductance + lr * (conductance * flux) - decay * conductance

def hybrid_voronoi_pheromone(points: List[Point],
                             seeds: List[Point],
                             path: List[Point],
                             lr: float = 0.1,
                             decay: float = 0.01) -> Multivector:
    """
    Full hybrid pipeline:
      1. Build conductance multivector from Voronoi regions.
      2. Build flux multivector from the supplied path.
      3. Update conductances using the geometric product.
    Returns the updated conductance multivector.
    """
    g = region_conductance_multivector(points, seeds)
    f = path_flux_multivector(path, seeds, alpha=1.0)
    g_new = hybrid_update(g, f, lr=lr, decay=decay)
    return g_new

# ----------------------------------------------------------------------
# 4. Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    # generate random seeds and points
    seeds = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(5)]
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(200)]

    # simple sinusoidal path through the domain
    t_vals = np.linspace(0, 2 * math.pi, 30)
    path = [(5 + 4 * math.cos(t), 5 + 4 * math.sin(t)) for t in t_vals]

    updated = hybrid_voronoi_pheromone(points, seeds, path, lr=0.05, decay=0.02)
    print("Updated conductance multivector:")
    print(updated)