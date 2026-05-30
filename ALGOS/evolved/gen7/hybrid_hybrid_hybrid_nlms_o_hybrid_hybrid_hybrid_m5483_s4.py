# DARWIN HAMMER — match 5483, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1209_s2.py (gen6)
# born: 2026-05-30T00:02:24Z

import numpy as np
import math
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Iterable, Generator

# ----------------------------------------------------------------------
# Basic geometric primitives
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def euclidean_distance(a: Point, b: Point) -> float:
    """Standard L2 distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# NLMS adaptive filter
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction using current weight vector."""
    return float(np.dot(weights, x))

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Normalized Least‑Mean‑Squares update.

    Returns the updated weight vector and the instantaneous error.
    """
    y = nlms_predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)          # normalisation term
    new_weights = weights + mu * error * x / power
    return new_weights, error

# ----------------------------------------------------------------------
# Data structures for textual spans
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float

# ----------------------------------------------------------------------
# Multivector implementation (geometric algebra)
# ----------------------------------------------------------------------
class Multivector:
    """
    Very small geometric‑algebra engine supporting:
      * addition,
      * scalar multiplication,
      * geometric product (with weight‑scaled basis vectors).
    Blades are represented by frozensets of basis indices.
    Coefficients are stored in a dict ``{blade: value}``.
    """

    def __init__(self, components: Dict[frozenset, float] | None = None, dim: int = 2):
        self.dim = int(dim)
        self.components: Dict[frozenset, float] = {}
        if components:
            for blade, val in components.items():
                if abs(val) > 1e-12:
                    self.components[frozenset(blade)] = float(val)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _blade_sign(indices: List[int]) -> Tuple[frozenset, int]:
        """
        Sorts a list of basis indices and returns the resulting blade
        together with the sign produced by the required swaps.
        """
        sign = 1
        # bubble‑sort while counting swaps
        for i in range(len(indices)):
            for j in range(len(indices) - 1 - i):
                if indices[j] > indices[j + 1]:
                    indices[j], indices[j + 1] = indices[j + 1], indices[j]
                    sign = -sign
        # remove duplicate indices (e_i ^ e_i = 0)
        cleaned = []
        i = 0
        while i < len(indices):
            if i + 1 < len(indices) and indices[i] == indices[i + 1]:
                i += 2                     # cancel the pair
            else:
                cleaned.append(indices[i])
                i += 1
        return frozenset(cleaned), sign

    @staticmethod
    def _geometric_product_blades(a: frozenset, b: frozenset, metric: np.ndarray) -> Tuple[frozenset, int, float]:
        """
        Returns (result_blade, sign, metric_factor) where ``metric_factor`` is the
        product of metric entries for any duplicated basis vectors that survive
        the anti‑symmetrisation.
        """
        # concatenate basis lists
        combined = list(a) + list(b)
        # sort while tracking sign
        blade, sign = Multivector._blade_sign(combined)

        # metric contribution: each time a basis index appears twice it contributes g_ii
        metric_factor = 1.0
        for idx in set(combined):
            count = combined.count(idx)
            if count % 2 == 0:               # even multiplicity cancels, but contributes metric^ (count/2)
                metric_factor *= metric[idx, idx] ** (count // 2)
        return blade, sign, metric_factor

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = Counter(self.components)
        result.update(other.components)
        return Multivector(dict(result), self.dim)

    def __rmul__(self, scalar: float) -> "Multivector":
        return Multivector({b: scalar * v for b, v in self.components.items()}, self.dim)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """
        Geometric product with a *metric* derived from the current NLMS weights.
        The metric is a diagonal matrix ``diag(weights)`` – this ties the two
        mathematical worlds together.
        """
        if not isinstance(other, Multivector):
            raise TypeError("Geometric product only defined between Multivectors.")
        # Build a diagonal metric from the weight vector (length must equal dim)
        metric = np.diag(self._current_weights[: self.dim]) if hasattr(self, "_current_weights") else np.eye(self.dim)

        result = Counter()
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                blade, sign, metric_factor = self._geometric_product_blades(blade_a, blade_b, metric)
                result[blade] += coeff_a * coeff_b * sign * metric_factor
        return Multivector(dict(result), self.dim)

    # ------------------------------------------------------------------
    # Grade extraction
    # ------------------------------------------------------------------
    def grade(self, k: int) -> "Multivector":
        """Return the part of the multivector of grade *k*."""
        return Multivector(
            {b: v for b, v in self.components.items() if len(b) == k},
            self.dim,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        if not self.components:
            return f"Multivector(0, dim={self.dim})"
        terms = [f"{v:.3g}{''.join(str(i) for i in sorted(b)) or '1'}" for b, v in self.components.items()]
        return f"Multivector({', '.join(terms)}, dim={self.dim})"

# ----------------------------------------------------------------------
# Voronoi / region handling
# ----------------------------------------------------------------------
def nearest_seed(point: Point, seeds: List[Point]) -> int:
    """Index of the seed closest to *point* (break ties by index)."""
    if not seeds:
        raise ValueError("At least one seed required.")
    return min(range(len(seeds)), key=lambda i: (euclidean_distance(point, seeds[i]), i))

def assign_regions(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to the Voronoi region of its nearest seed."""
    regions: Dict[int, List[Point]] = defaultdict(list)
    for p in points:
        regions[nearest_seed(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature (simplified)
# ----------------------------------------------------------------------
def ollivier_ricci(
    p: Point,
    q: Point,
    weights: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """
    Very lightweight Ollivier‑Ricci estimate for a pair of points.
    The NLMS weights act as a *transport cost* modifier:
        cost(e_i) = 1 / (|w_i| + ε)
    """
    # Build a diagonal transport‑cost matrix from the absolute weights
    cost_diag = 1.0 / (np.abs(weights) + epsilon)

    # Vector from p to q
    vec = np.array([q[0] - p[0], q[1] - p[1]])

    # Effective distance under the weighted metric
    weighted_dist = math.sqrt(float(vec @ np.diag(cost_diag) @ vec))

    # Ollivier‑Ricci curvature approximation (1 - weighted_dist / euclidean)
    euclid = euclidean_distance(p, q)
    if euclid == 0:
        return 0.0
    return 1.0 - weighted_dist / euclid

# ----------------------------------------------------------------------
# Core hybrid routine
# ----------------------------------------------------------------------
def hybrid_operation(
    spans: List[Span],
    points: List[Point],
    seeds: List[Point],
    initial_weights: np.ndarray,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Generator[Multivector, None, None]:
    """
    For each Voronoi region:
      1. Adapt NLMS weights using the region's points.
      2. Build a multivector whose basis vectors are weighted by the
         *final* NLMS weights (deep integration of the two theories).
      3. Compute a curvature‑adjusted scalar and embed it as the scalar
         (grade‑0) part of the multivector.
      4. Yield the region‑specific multivector.
    """
    if initial_weights.shape[0] != 2:
        raise ValueError("Initial weight vector must have length 2 for 2‑D points.")

    # ------------------------------------------------------------------
    # 1. Voronoi partition
    # ------------------------------------------------------------------
    regions = assign_regions(points, seeds)

    # ------------------------------------------------------------------
    # 2. Process each region independently
    # ------------------------------------------------------------------
    for region_idx, region_pts in regions.items():
        # start from the global weight vector for each region
        w = initial_weights.astype(float).copy()

        # adapt weights with NLMS on every point in the region
        for pt in region_pts:
            x = np.array(pt, dtype=float)
            w, _ = nlms_update(w, x, target=1.0, mu=mu, eps=eps)

        # ------------------------------------------------------------------
        # 3. Build a multivector that encodes the region geometry
        # ------------------------------------------------------------------
        #   • Basis e0 and e1 are scaled by the final weights.
        #   • The scalar part stores the average Ollivier‑Ricci curvature
        #     between all unordered point pairs in the region.
        # ------------------------------------------------------------------
        # a) geometric‑product basis
        basis_vectors = {
            frozenset({0}): w[0],
            frozenset({1}): w[1],
        }
        region_mv = Multivector(basis_vectors, dim=2)

        # b) curvature scalar
        if len(region_pts) < 2:
            curvature_scalar = 0.0
        else:
            curv_sum = 0.0
            count = 0
            for i in range(len(region_pts)):
                for j in range(i + 1, len(region_pts)):
                    curv_sum += ollivier_ricci(region_pts[i], region_pts[j], w)
                    count += 1
            curvature_scalar = curv_sum / count

        # embed scalar as grade‑0 blade (empty frozenset)
        region_mv = region_mv + Multivector({frozenset(): curvature_scalar}, dim=2)

        # expose the adapted weights for the geometric product (used internally)
        region_mv._current_weights = w

        yield region_mv

# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    spans = [
        Span(0, 10, "alpha", "A", 1.0),
        Span(10, 20, "beta", "B", 2.0),
    ]
    points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 0.5)]
    seeds = [(0.0, 0.0), (3.0, 0.5)]
    init_w = np.array([1.0, 1.0])

    for mv in hybrid_operation(spans, points, seeds, init_w):
        print(mv)