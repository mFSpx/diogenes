# DARWIN HAMMER — match 1360, survivor 2
# gen: 4
# parent_a: hybrid_serpentina_self_righ_xgboost_objective_m78_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_pherom_m30_s0.py (gen3)
# born: 2026-05-29T23:35:47Z

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, FrozenSet, Iterable, Optional

import numpy as np


# ----------------------------------------------------------------------
# Parent A – Morphology & Logistic Gradient/Hessian
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Four physical descriptors of an object."""
    length: float
    width: float
    height: float
    mass: float


def _positive(*values: float) -> None:
    """Validate that all supplied values are strictly positive."""
    for v in values:
        if v <= 0.0:
            raise ValueError("all morphological dimensions must be > 0")


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity – ratio of the geometric mean edge to the longest edge."""
    _positive(length, width, height)
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness – how spread the footprint is relative to the thickness."""
    _positive(length, width, height)
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology,
    b: float = 1.0 / 3.0,
    k: float = 0.35,
    neck_lever: float = 1.0,
) -> float:
    """A simple physics‑inspired index for the time needed to self‑right."""
    _positive(m.mass, neck_lever)
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """
    Normalised base recovery probability r ∈ [0, 1].
    ``max_index`` is a domain‑specific scaling constant.
    """
    if max_index <= 0.0:
        raise ValueError("max_index must be > 0")
    r = righting_time_index(m) / max_index
    return max(0.0, min(1.0, r))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    # use np.where to avoid overflow
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out = np.empty_like(x, dtype=np.float64)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)
    return out if out.shape else out.item()


def binary_logistic_grad_hess(
    y_true: np.ndarray,
    margin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gradient and Hessian of the binary logistic loss.
    ``margin`` is the raw model output (log‑odds).
    """
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h


# ----------------------------------------------------------------------
# Parent B – Geometry, Voronoi, Entropy, Curvature
# ----------------------------------------------------------------------


Point = Tuple[float, float]


def euclidean_distance(a: Point, b: Point) -> float:
    """2‑D Euclidean distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the seed nearest to ``point`` (break ties by index)."""
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean_distance(point, seeds[i]), i))


def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """
    Voronoi assignment of each point to its nearest seed.
    Returns a mapping seed_index → list of points belonging to that cell.
    """
    if not seeds:
        raise ValueError("seed list cannot be empty")
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


def shannon_entropy(probs: List[float]) -> float:
    """Standard Shannon entropy (natural logarithm)."""
    eps = 1e-12
    return -sum(p * math.log(p + eps) for p in probs if p > 0.0)


def ollivier_ricci_curvature(seeds: List[Point], probs: List[float]) -> float:
    """
    Simple average Ollivier‑Ricci curvature approximation.
    For each unordered pair (i, j) we use
        κ_{i,j} = 1 - |p_i - p_j| / (d_{i,j} + ε)
    The final curvature is the mean over all pairs.
    """
    n = len(seeds)
    if n < 2:
        return 0.0
    eps = 1e-12
    curv_sum = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(seeds[i], seeds[j])
            curv_sum += 1.0 - abs(probs[i] - probs[j]) / (d + eps)
            count += 1
    return curv_sum / count


# ----------------------------------------------------------------------
# Minimal Geometric Algebra (Multivector) – deeper integration
# ----------------------------------------------------------------------


def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """
    Sort ``indices`` while tracking the sign of the permutation.
    Duplicate indices cancel (wedge product rule).
    """
    lst = list(indices)
    sign = 1
    i = 0
    while i < len(lst) - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign *= -1
            i = max(i - 1, 0)
        elif lst[i] == lst[i + 1]:
            # cancel equal indices
            lst.pop(i)
            lst.pop(i)
            i = max(i - 1, 0)
        else:
            i += 1
    return lst, sign


def _multiply_blades(
    blade_a: FrozenSet[int],
    blade_b: FrozenSet[int],
) -> Tuple[FrozenSet[int], int]:
    combined = list(blade_a) + list(blade_b)
    sorted_indices, sign = _blade_sign(combined)
    return frozenset(sorted_indices), sign


class Multivector:
    """
    Very small subset of geometric algebra sufficient for the hybrid.
    Internally a sparse mapping blade → coefficient.
    """

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # prune near‑zero components
        self.components: Dict[FrozenSet[int], float] = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------

    def __add__(self, other: "Multivector") -> "Multivector":
        result = self.components.copy()
        for blade, val in other.components.items():
            result[blade] = result.get(blade, 0.0) + val
        return Multivector(result, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product (bilinear, not optimised)."""
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                result[blade_res] = result.get(blade_res, 0.0) + sign * coeff_a * coeff_b
        return Multivector(result, self.n)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) component."""
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        return f"Multivector({self.components})"


def morphology_to_multivector(m: Morphology) -> Multivector:
    """
    Encode the four morphology parameters as a grade‑1 multivector.
    Basis vectors are labelled 1‑4.
    """
    comps = {
        frozenset({1}): m.length,
        frozenset({2}): m.width,
        frozenset({3}): m.height,
        frozenset({4}): m.mass,
    }
    return Multivector(comps, n=4)


def curvature_entropy_to_multivector(curvature: float, entropy: float) -> Multivector:
    """
    Encode curvature (grade‑1) and entropy (grade‑2) into a multivector.
    This allows a geometric‑algebraic interaction with the morphology vector.
    """
    comps = {
        frozenset({5}): curvature,          # basis‑5 for curvature
        frozenset({6, 7}): entropy,          # a bivector for entropy
    }
    return Multivector(comps, n=7)


def fuse_multivectors(a: Multivector, b: Multivector) -> float:
    """
    Fuse two multivectors via geometric product and extract the scalar part.
    The scalar part is guaranteed to be a real number in ℝ.
    """
    return (a * b).scalar_part()


# ----------------------------------------------------------------------
# Hybrid Functions – robust, deeper integration
# ----------------------------------------------------------------------


def pheromone_region_entropy(
    points: List[Point],
    seeds: List[Point],
) -> Tuple[float, List[float]]:
    """
    Compute Voronoi region probabilities and their Shannon entropy.
    Returns ``(entropy, probabilities)``.
    """
    if not seeds:
        raise ValueError("seed list cannot be empty")
    regions = assign(points, seeds)
    total = sum(len(v) for v in regions.values())
    if total == 0:
        # uniform distribution fallback
        probs = [1.0 / len(seeds)] * len(seeds)
        return 0.0, probs
    probs = [len(regions[i]) / total for i in range(len(seeds))]
    return shannon_entropy(probs), probs


def _normalise_entropy(H: float, H_max: float) -> float:
    """Return a factor in [0,1] that down‑weights high entropy."""
    if H_max <= 0.0:
        return 1.0  # no entropy variation possible (single seed)
    return max(0.0, min(1.0, 1.0 - H / H_max))


def _normalise_curvature(kappa: float) -> float:
    """Map curvature from (−∞,1] to [0,1] safely."""
    # The theoretical upper bound of κ in our approximation is 1.
    # Values may dip below −1 due to pathological inputs; clamp.
    return max(0.0, min(1.0, (1.0 + kappa) / 2.0))


def hybrid_recovery_score(
    m: Morphology,
    points: List[Point],
    seeds: List[Point],
) -> float:
    """
    Compute the fused recovery probability ``p̂``.
    The pipeline is:
        1. Base recovery priority ``r`` from morphology.
        2. Voronoi entropy ``H`` and normalised factor ``e = 1 - H/H_max``.
        3. Ollivier‑Ricci curvature ``κ`` and normalised factor ``c = (1+κ)/2``.
        4. Geometric‑algebraic interaction between morphology and (κ,H).
    The final scalar is clipped to [0,1].
    """
    # 1️⃣ Morphology → base probability
    r = recovery_priority(m)

    # 2️⃣ Entropy
    H, probs = pheromone_region_entropy(points, seeds)
    H_max = math.log(len(seeds)) if len(seeds) > 1 else 0.0
    e_factor = _normalise_entropy(H, H_max)

    # 3️⃣ Curvature
    kappa = ollivier_ricci_curvature(seeds, probs)
    c_factor = _normalise_curvature(kappa)

    # 4️⃣ Geometric‑algebraic coupling (deeper integration)
    mv_morph = morphology_to_multivector(m)
    mv_geo = curvature_entropy_to_multivector(kappa, H)
    algebraic_coupling = fuse_multivectors(mv_morph, mv_geo)  # scalar in ℝ

    # Map coupling to a [0,1] modifier – sigmoid ensures smoothness
    coupling_factor = sigmoid(algebraic_coupling)

    # Combine all modifiers multiplicatively
    p_hat = r * e_factor * c_factor * coupling_factor

    # Final safety clipping
    return float(np.clip(p_hat, 0.0, 1.0))


class HREC:
    """
    High‑level wrapper exposing the hybrid probability together with
    the logistic gradient/Hessian required by XGBoost‑style learners.
    """

    def __init__(
        self,
        morphology: Morphology,
        points: List[Point],
        seeds: List[Point],
        max_index: float = 10.0,
    ):
        self.morphology = morphology
        self.points = points
        self.seeds = seeds
        self.max_index = max_index

    def probability(self) -> float:
        """Compute the fused probability ``p̂``."""
        return hybrid_recovery_score(
            self.morphology,
            self.points,
            self.seeds,
        )

    def margin(self) -> float:
        """
        Convert probability to log‑odds (margin) required by the logistic loss.
        ``margin = log(p / (1-p))``; safe‑guard against p∈{0,1}.
        """
        p = self.probability()
        eps = 1e-12
        p = max(eps, min(1.0 - eps, p))
        return math.log(p / (1.0 - p))

    def grad_hess(self, y_true: float) -> Tuple[float, float]:
        """
        Return gradient and Hessian for a single scalar observation.
        ``y_true`` must be 0 or 1.
        """
        if y_true not in (0.0, 1.0):
            raise ValueError("y_true must be 0 or 1 for binary logistic loss")
        margin = np.array([self.margin()], dtype=np.float64)
        y = np.array([y_true], dtype=np.float64)
        g, h = binary_logistic_grad_hess(y, margin)
        return float(g[0]), float(h[0])


# ----------------------------------------------------------------------
# Example usage (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data for sanity‑check
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)
    pts = [(random.random(), random.random()) for _ in range(200)]
    seed_pts = [(0.2, 0.2), (0.8, 0.8), (0.2, 0.8), (0.8, 0.2)]

    hrec = HREC(morph, pts, seed_pts)
    prob = hrec.probability()
    grad, hess = hrec.grad_hess(y_true=1.0)

    print(f"Hybrid probability: {prob:.5f}")
    print(f"Gradient: {grad:.5f}, Hessian: {hess:.5f}")