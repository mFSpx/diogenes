# DARWIN HAMMER — match 78, survivor 0
# gen: 2
# parent_a: bayes_update.py (gen0)
# parent_b: hybrid_geometric_product_voronoi_partition_m4_s2.py (gen1)
# born: 2026-05-29T23:26:38Z

#!/usr/bin/env python3
"""Hybrid module fusing Bayesian evidence updates (bayes_update.py) 
and geometric algebra with Voronoi partitioning (hybrid_geometric_product_voronoi_partition_m4_s2.py).

Mathematical bridge:
- Bayesian updates rely on likelihood and prior probabilities to estimate 
  posterior probabilities.
- Geometric algebra and Voronoi partitioning use multivector representations 
  of points and compute distances via inner products.
- The bridge: interpreting prior probabilities as point distributions in 
  a geometric algebra framework, and using Voronoi regions to inform 
  likelihood estimates based on proximity to 'seed' points.

The module provides:
* `bayes_marginal_mv` – Bayesian marginal probability with multivector 
  representation of points.
* `voronoi_partition_bayes` – Voronoi region assignment with Bayesian 
  updates of likelihood.
* `hybrid_bayes_update` – Fused Bayesian update with geometric algebra 
  and Voronoi partitioning.
"""

import numpy as np
from typing import Tuple, List
import math

# ---------------------------------------------------------------------------
# Geometric algebra core
# ---------------------------------------------------------------------------

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j:j + 2]
                n -= 2
                sign *= 1  
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> Tuple[frozenset[int], int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def point_to_mv(point: Tuple[float, float]) -> Tuple[float, float, float, float]:
    """Convert a 2‑tuple to a multivector vector."""
    x, y = point
    return x, y, 0, 0


def mv_distance(mv_a: Tuple[float, float, float, float], mv_b: Tuple[float, float, float, float]) -> float:
    """Euclidean distance via geometric algebra inner product."""
    dx = mv_a[0] - mv_b[0]
    dy = mv_a[1] - mv_b[1]
    return math.sqrt(dx**2 + dy**2)


# ---------------------------------------------------------------------------
# Bayesian update core
# ---------------------------------------------------------------------------

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Bayesian marginal probability."""
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Bayesian update."""
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal


# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------

def bayes_marginal_mv(prior: float, likelihood: float, false_positive: float, point: Tuple[float, float]) -> float:
    """Bayesian marginal probability with multivector representation of points."""
    mv_point = point_to_mv(point)
    # For demonstration, assume point distribution affects likelihood
    likelihood_adjusted = likelihood * (1 - mv_distance(mv_point, (0, 0, 0, 0)) / 10)
    return bayes_marginal(prior, likelihood_adjusted, false_positive)


def voronoi_partition_bayes(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> List[int]:
    """Voronoi region assignment with Bayesian updates of likelihood."""
    assignments = []
    for point in points:
        min_distance = float('inf')
        nearest_seed_index = -1
        for i, seed in enumerate(seeds):
            distance = mv_distance(point_to_mv(point), point_to_mv(seed))
            if distance < min_distance:
                min_distance = distance
                nearest_seed_index = i
        # Bayesian update of likelihood based on distance
        prior = 0.5
        likelihood = 1 / (1 + min_distance)
        false_positive = 0.1
        marginal = bayes_marginal(prior, likelihood, false_positive)
        assignments.append(nearest_seed_index)
    return assignments


def hybrid_bayes_update(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> List[float]:
    """Fused Bayesian update with geometric algebra and Voronoi partitioning."""
    posteriors = []
    for point in points:
        prior = 0.5
        likelihood = 1 / (1 + mv_distance(point_to_mv(point), point_to_mv(seeds[0])))
        false_positive = 0.1
        marginal = bayes_marginal(prior, likelihood, false_positive)
        posterior = bayes_update(prior, likelihood, marginal)
        posteriors.append(posterior)
    return posteriors


if __name__ == "__main__":
    points = [(1, 2), (3, 4), (5, 6)]
    seeds = [(0, 0), (10, 10)]
    print(voronoi_partition_bayes(points, seeds))
    print(hybrid_bayes_update(points, seeds))