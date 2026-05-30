# DARWIN HAMMER — match 4859, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1209_s4.py (gen6)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_geomet_m412_s0.py (gen5)
# born: 2026-05-29T23:58:27Z

"""
Hybrid Module: Fusion of Ollivier‑Ricci curvature & Shannon entropy (Parent A)
with Fisher information weighted geometric product (Parent B).

Mathematical Bridge
------------------
Both parent algorithms employ a *weighting* mechanism:
* Parent A applies reconstruction‑risk scores to weight causal‑effect estimates,
  then regularises the result with Shannon entropy.
* Parent B uses the Fisher information metric as a scalar weight for the
  geometric (Clifford) product of multivectors.

The hybrid therefore:
1. Computes a curvature‑based multivector representation of a point cloud
   (using the Ollivier‑Ricci idea of transport between seed regions).
2. Evaluates a Fisher information weight from a Gaussian‑beam model for each
   region.
3. Forms a weighted geometric product of the curvature multivectors.
4. Regularises the scalar outcome with Shannon entropy of the region‑wise
   reconstruction‑risk distribution.

The resulting scalar can be interpreted as a *risk‑aware, entropy‑regularised
causal effect* that respects the underlying geometric structure of the data.
"""

import math
import random
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric utilities (from Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the nearest seed (ties broken by smallest index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Voronoi‑like assignment of points to the nearest seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Multivector and geometric product (from Parent B)
# ----------------------------------------------------------------------
def _blade_sign(indices: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """
    Sort indices, cancel duplicate pairs (which square to +1) and return the
    sorted tuple together with the sign (+1 / -1) induced by the swaps.
    """
    counts: Dict[int, int] = {}
    for i in indices:
        counts[i] = counts.get(i, 0) + 1

    # keep only indices with odd multiplicity
    remaining = [i for i, c in counts.items() if c % 2 == 1]

    # bubble‑sort while tracking sign
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
    # remove cancelled pairs
    cleaned = [i for i in lst if i in remaining]
    return tuple(sorted(set(cleaned))), sign

def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    combined = tuple(list(blade_a) + list(blade_b))
    sorted_idxs, sign = _blade_sign(combined)
    return frozenset(sorted_idxs), sign

class Multivector:
    """Simple multivector supporting addition and geometric product."""
    def __init__(self, components: Dict[FrozenSet[int], float] | None = None):
        self.components: Dict[FrozenSet[int], float] = {}
        if components:
            for k, v in components.items():
                if abs(v) > 1e-15:
                    self.components[frozenset(k)] = float(v)

    def __add__(self, other: 'Multivector') -> 'Multivector':
        result = Multivector(self.components)
        for k, v in other.components.items():
            result.components[k] = result.components.get(k, 0.0) + v
            if abs(result.components[k]) < 1e-15:
                del result.components[k]
        return result

    def geometric_product(self, other: 'Multivector') -> 'Multivector':
        """Geometric product a ∘ b."""
        result: Dict[FrozenSet[int], float] = {}
        for a_blade, a_val in self.components.items():
            for b_blade, b_val in other.components.items():
                blade, sign = _multiply_blades(a_blade, b_blade)
                val = a_val * b_val * sign
                result[blade] = result.get(blade, 0.0) + val
                if abs(result[blade]) < 1e-15:
                    del result[blade]
        return Multivector(result)

    def scale(self, scalar: float) -> 'Multivector':
        return Multivector({k: v * scalar for k, v in self.components.items()})

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) component, 0 if absent."""
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        return f"Multivector({self.components})"

# ----------------------------------------------------------------------
# Fisher information weight (from Parent B)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ)."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_information(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature approximation (simplified)
# ----------------------------------------------------------------------
def ollivier_ricci_curvature(p: Point, q: Point, sigma: float = 1.0) -> float:
    """
    Approximate Ollivier‑Ricci curvature between two points using a Gaussian
    transport plan with standard deviation *sigma*.
    κ(p,q) = 1 - W1(μ_p, μ_q) / d(p,q)
    where W1 is the 1‑Wasserstein distance between two isotropic Gaussians.
    For isotropic Gaussians the closed form is:
        W1 = sqrt( (d(p,q))^2 + 2σ^2 )
    """
    d = distance(p, q)
    if d == 0.0:
        return 0.0
    w1 = math.sqrt(d * d + 2.0 * sigma * sigma)
    return 1.0 - w1 / d

def region_curvature(seeds: List[Point], points: List[Point]) -> Multivector:
    """
    Build a multivector whose scalar part is the average curvature over all
    seed‑to‑seed pairs, and higher‑grade blades encode pairwise curvature signs.
    """
    if len(seeds) < 2:
        return Multivector()
    # assign points to seeds (Voronoi regions)
    regions = assign(points, seeds)

    # compute curvature for each unordered seed pair
    components: Dict[FrozenSet[int], float] = {}
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            # pick a representative point from each region (fallback to seed)
            pi = regions[i][0] if regions[i] else seeds[i]
            pj = regions[j][0] if regions[j] else seeds[j]
            kappa = ollivier_ricci_curvature(pi, pj)
            # scalar contribution (grade‑0)
            components[frozenset()] = components.get(frozenset(), 0.0) + kappa
            # bivector (grade‑2) encoding the pair (i,j)
            blade = frozenset({i, j})
            components[blade] = components.get(blade, 0.0) + kappa
    return Multivector(components)

# ----------------------------------------------------------------------
# Shannon entropy regularisation (from Parent A)
# ----------------------------------------------------------------------
def shannon_entropy(probs: List[float]) -> float:
    """Compute -Σ p log p for a probability vector."""
    eps = 1e-12
    return -sum(p * math.log(max(p, eps)) for p in probs)

def label_distribution(points: List[Point], bins: int = 10) -> List[float]:
    """Create a simple histogram of distances from origin as a probability vector."""
    dists = [math.hypot(x, y) for x, y in points]
    hist, _ = np.histogram(dists, bins=bins, range=(0.0, max(dists) or 1.0))
    total = hist.sum()
    if total == 0:
        return [1.0 / bins] * bins
    return (hist / total).tolist()

# ----------------------------------------------------------------------
# Reconstruction risk (placeholder – uses variance of intra‑region distances)
# ----------------------------------------------------------------------
def reconstruction_risk(region_points: List[Point]) -> float:
    """Higher variance of distances indicates higher reconstruction risk."""
    if len(region_points) < 2:
        return 0.0
    dists = [distance(p, region_points[0]) for p in region_points[1:]]
    return np.var(dists)

# ----------------------------------------------------------------------
# Hybrid operations ----------------------------------------------------
# ----------------------------------------------------------------------
def hybrid_weighted_product(
    seeds: List[Point],
    points: List[Point],
    theta: float,
    center: float,
    width: float
) -> float:
    """
    1. Build curvature multivector for the point cloud.
    2. Compute Fisher information weight from the Gaussian‑beam parameters.
    3. Perform weighted geometric product of the multivector with itself.
    4. Extract scalar part and regularise with Shannon entropy of a simple label
       distribution.
    Returns the final hybrid scalar score.
    """
    # Step 1: curvature multivector
    cur_mv = region_curvature(seeds, points)

    # Step 2: Fisher weight
    weight = fisher_information(theta, center, width)

    # Step 3: weighted geometric product (self‑product for demonstration)
    prod_mv = cur_mv.geometric_product(cur_mv).scale(weight)

    # Step 4: entropy regularisation
    probs = label_distribution(points, bins=8)
    entropy = shannon_entropy(probs)

    # Combine scalar part with entropy (entropy acts as a dampening factor)
    scalar = prod_mv.scalar_part()
    return scalar * math.exp(-entropy)  # exp(-H) reduces magnitude for high entropy

def regionwise_hybrid_score(
    seeds: List[Point],
    points: List[Point],
    theta_params: Tuple[float, float, float]
) -> Dict[int, float]:
    """
    Compute a hybrid score per Voronoi region:
        score_i = (average curvature in region i) *
                  (Fisher weight) *
                  exp(-entropy_i) *
                  (1 / (1 + risk_i))
    Returns a dict mapping region index to its score.
    """
    theta, center, width = theta_params
    weight = fisher_information(theta, center, width)

    regions = assign(points, seeds)
    scores: Dict[int, float] = {}
    for idx, pts in regions.items():
        # curvature within region (average pairwise curvature to seed)
        if not pts:
            scores[idx] = 0.0
            continue
        curvs = [ollivier_ricci_curvature(p, seeds[idx]) for p in pts]
        avg_curv = sum(curvs) / len(curvs)

        # entropy of distances inside the region
        probs = label_distribution(pts, bins=5)
        ent = shannon_entropy(probs)

        # reconstruction risk
        risk = reconstruction_risk(pts)

        scores[idx] = avg_curv * weight * math.exp(-ent) / (1.0 + risk)
    return scores

def global_hybrid_metric(
    seeds: List[Point],
    points: List[Point],
    theta_params: Tuple[float, float, float]
) -> float:
    """
    Aggregate regionwise scores into a single global metric using a weighted
    geometric product across regions.
    """
    region_scores = regionwise_hybrid_score(seeds, points, theta_params)
    # Build a multivector where each region contributes a basis blade with its score
    components: Dict[FrozenSet[int], float] = {}
    for idx, score in region_scores.items():
        # scalar part accumulates all scores
        components[frozenset()] = components.get(frozenset(), 0.0) + score
        # higher‑grade blade encodes region identity
        components[frozenset({idx})] = components.get(frozenset({idx}), 0.0) + score

    mv = Multivector(components)
    # Weighted self‑product with Fisher weight
    weight = fisher_information(*theta_params)
    final_mv = mv.geometric_product(mv).scale(weight)

    # Return regularised scalar part
    entropy = shannon_entropy(list(region_scores.values()))
    return final_mv.scalar_part() * math.exp(-entropy)

# ----------------------------------------------------------------------
# Smoke test -----------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # generate synthetic point cloud
    num_points = 200
    points = [(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(num_points)]

    # choose 5 random seeds
    seeds = random.sample(points, 5)

    # Gaussian‑beam parameters (theta, center, width)
    theta_params = (0.3, 0.0, 1.5)

    # Execute hybrid functions
    scalar_score = hybrid_weighted_product(seeds, points, *theta_params)
    region_scores = regionwise_hybrid_score(seeds, points, theta_params)
    global_score = global_hybrid_metric(seeds, points, theta_params)

    print(f"Hybrid weighted product scalar: {scalar_score:.6f}")
    print(f"Regionwise scores: { {k: round(v,4) for k,v in region_scores.items()} }")
    print(f"Global hybrid metric: {global_score:.6f}")