# DARWIN HAMMER — match 1360, survivor 0
# gen: 4
# parent_a: hybrid_serpentina_self_righ_xgboost_objective_m78_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_pherom_m30_s0.py (gen3)
# born: 2026-05-29T23:35:47Z

"""Hybrid Recovery-Entropy-Curvature (HREC) algorithm.
Parents:
    - hybrid_serpentina_self_righ_xgboost_objective_m78_s0 (morphology & logistic gradient/hessian)
    - hybrid_hybrid_hybrid_geomet_hybrid_hybrid_pherom_m30_s0 (geometric algebra, Voronoi partition, Shannon entropy, Ollivier‑Ricci curvature)

Mathematical bridge:
Both parents produce scalar descriptors that can be interpreted as probabilities.
The serpentina side yields a recovery priority 𝑟∈[0,1]; the geometric‑pheromone side yields
a Shannon entropy 𝐻 and an Ollivier‑Ricci curvature κ (both scalar).  By treating
r as a base probability and modulating it with (1‑H/𝐻ₘₐₓ) and (1+κ)/2 we obtain a unified
probability p̂.  This probability can be fed to the binary‑logistic gradient/hessian
formulation of the XGBoost objective, thus mathematically fusing the two topologies
into a single differentiable system.
"""

import math
import random
import sys
import pathlib
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, FrozenSet

# ---------- Parent A: Morphology & Logistic Gradient/Hessian ----------

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Base recovery probability in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))

def binary_logistic_grad_hess(y_true: np.ndarray,
                              margin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gradient and Hessian of binary logistic loss."""
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h

# ---------- Parent B: Geometry, Voronoi, Entropy, Curvature ----------

Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)),
               key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Voronoi assignment of each point to its nearest seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def shannon_entropy(probs: List[float]) -> float:
    """Standard Shannon entropy (base e)."""
    eps = 1e-12
    return -sum(p * math.log(p + eps) for p in probs if p > 0)

def ollivier_ricci_curvature(seeds: List[Point],
                             probs: List[float]) -> float:
    """Simple average Ollivier‑Ricci curvature approximation.
    For each edge (i,j) we use κ₍i,j₎ = 1 - |p_i-p_j| / (d_ij + ε).
    The final curvature is the mean over all unordered pairs.
    """
    if len(seeds) < 2:
        return 0.0
    eps = 1e-12
    curvatures = []
    n = len(seeds)
    for i in range(n):
        for j in range(i + 1, n):
            d = distance(seeds[i], seeds[j])
            prob_diff = abs(probs[i] - probs[j])
            curv = 1.0 - prob_diff / (d + eps)
            curvatures.append(curv)
    return sum(curvatures) / len(curvatures)

# ---------- Minimal Geometric Algebra support (Multivector) ----------

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return sorted list of indices and the sign of the permutation.
    Duplicate indices cancel (wedge product rule)."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign *= -1
            i = max(i - 1, 0)
        elif lst[i] == lst[i + 1]:
            # cancel equal indices
            lst.pop(i)
            lst.pop(i)
            n -= 2
            i = max(i - 1, 0)
        else:
            i += 1
    return lst, sign

def _multiply_blades(blade_a: FrozenSet[int],
                     blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    combined = list(blade_a) + list(blade_b)
    sorted_indices, sign = _blade_sign(combined)
    return frozenset(sorted_indices), sign

class Multivector:
    """Very small subset of geometric algebra needed for the hybrid."""
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # store only non‑zero components
        self.components: Dict[FrozenSet[int], float] = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }

    def __add__(self, other: 'Multivector') -> 'Multivector':
        result = self.components.copy()
        for blade, val in other.components.items():
            result[blade] = result.get(blade, 0.0) + val
        return Multivector(result, self.n)

    def __mul__(self, other: 'Multivector') -> 'Multivector':
        """Geometric product (bilinear, not optimized)."""
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                result[blade_res] = result.get(blade_res, 0.0) + sign * coeff_a * coeff_b
        return Multivector(result, self.n)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        return f"Multivector({self.components})"

# ---------- Hybrid Functions ----------

def morphology_to_multivector(m: Morphology) -> Multivector:
    """Encode the four morphology parameters as a grade‑1 multivector."""
    comps = {
        frozenset({1}): m.length,
        frozenset({2}): m.width,
        frozenset({3}): m.height,
        frozenset({4}): m.mass,
    }
    return Multivector(comps, n=4)

def pheromone_region_entropy(points: List[Point],
                             seeds: List[Point]) -> Tuple[float, List[float]]:
    """Assign points to Voronoi regions, compute region probabilities and entropy."""
    regions = assign(points, seeds)
    total = sum(len(v) for v in regions.values())
    if total == 0:
        probs = [1.0 / len(seeds)] * len(seeds)
        return 0.0, probs
    probs = [len(regions[i]) / total for i in range(len(seeds))]
    entropy = shannon_entropy(probs)
    return entropy, probs

def hybrid_recovery_score(m: Morphology,
                          points: List[Point],
                          seeds: List[Point]) -> float:
    """
    Combine:
        * Base recovery priority r from morphology.
        * Entropy H of pheromone region distribution.
        * Ollivier‑Ricci curvature κ derived from region probabilities.
    The formula is
        p̂ = r * (1 - H / H_max) * (1 + κ) / 2
    where H_max = ln(#seeds) (maximum entropy for a uniform distribution).
    The result is clipped to [0,1].
    """
    r = recovery_priority(m)

    H, probs = pheromone_region_entropy(points, seeds)
    H_max = math.log(len(seeds)) if len(seeds) > 1 else 0.0
    norm_entropy = H / H_max if H_max > 0 else 0.0

    κ = ollivier_ricci_curvature(seeds, probs)  # lies roughly in [-inf, 1]
    κ_norm = (1.0 + κ) / 2.0  # map to [0,1] (may exceed bounds, will be clipped later)

    combined = r * (1.0 - norm_entropy) * κ_norm
    return max(0.0, min(1.0, combined))

def hybrid_logistic_grad_hess(y_true: np.ndarray,
                              m: Morphology,
                              points: List[Point],
                              seeds: List[Point]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient and Hessian of the binary logistic loss where the model
    prediction is the hybrid recovery score transformed to a log‑odds margin.
    """
    p_hat = hybrid_recovery_score(m, points, seeds)
    # protect against exact 0/1 to keep logit finite
    eps = 1e-12
    p_hat = max(eps, min(1.0 - eps, p_hat))
    margin = np.log(p_hat / (1.0 - p_hat))
    return binary_logistic_grad_hess(y_true, np.array([margin]))

# ---------- Smoke test ----------

if __name__ == "__main__":
    # Random morphology
    morph = Morphology(
        length=random.uniform(0.5, 2.0),
        width=random.uniform(0.5, 2.0),
        height=random.uniform(0.5, 2.0),
        mass=random.uniform(1.0, 5.0)
    )

    # Random points and seeds for pheromone field
    seed_count = 5
    point_count = 200
    seeds = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(seed_count)]
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(point_count)]

    # Compute hybrid score
    score = hybrid_recovery_score(morph, points, seeds)
    print(f"Hybrid recovery score: {score:.4f}")

    # Gradient/Hessian test (binary label 1)
    y = np.array([1.0])
    g, h = hybrid_logistic_grad_hess(y, morph, points, seeds)
    print(f"Gradient: {g}, Hessian: {h}")

    # Demonstrate multivector conversion
    mv = morphology_to_multivector(morph)
    print(f"Multivector representation: {mv}")
    print(f"Scalar part (should be 0): {mv.scalar_part():.4f}")