# DARWIN HAMMER — match 4393, survivor 0
# gen: 5
# parent_a: hybrid_sparse_wta_hybrid_privacy_model_m29_s0.py (gen2)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s4.py (gen4)
# born: 2026-05-29T23:55:16Z

"""
Module for fusing the sparse winner-take-all (WTA) mechanism with the hybrid Voronoi-circuit-breaker 
and Clifford geometric-product resource allocation. The mathematical bridge lies in applying the WTA 
mechanism to the resource allocation process, where the site with the highest resource utilization 
score (derived from the geometric product) is prioritized for resource updates.

Parents:
- hybrid_sparse_wta_hybrid_privacy_model_m29_s0.py (Sparse WTA with hybrid privacy model)
- hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s4.py (Hybrid Voronoi-circuit-breaker with 
  Clifford geometric-product resource allocation)
"""

from __future__ import annotations
from typing import Any, Iterable, Dict
import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

@dataclass
class Site:
    point: tuple[float, float]
    circuit_breaker: bool
    resource: dict[frozenset[int], float]

def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_voronoi_regions(points: list[tuple[float, float]],
                            sites: list[Site]) -> dict[int, list[tuple[float, float]]]:
    """
    Assign each point to the index of the nearest site.
    Returns a dict ``site_index → list[points]``.
    """
    regions: dict[int, list[tuple[float, float]]] = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s.point) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions

def geometric_product(multivector1: dict[frozenset[int], float], 
                      multivector2: dict[frozenset[int], float]) -> dict[frozenset[int], float]:
    result: dict[frozenset[int], float] = {}
    for blade1, coeff1 in multivector1.items():
        for blade2, coeff2 in multivector2.items():
            blade = frozenset(blade1 | blade2)
            result[blade] = result.get(blade, 0) + coeff1 * coeff2
    return result

def wta_resource_allocation(sites: list[Site], 
                            demand: dict[frozenset[int], float]) -> Site:
    scores = []
    for site in sites:
        score = site.resource.get(frozenset(), 0)  # default to scalar part
        scores.append((site, score))
    winner = max(scores, key=lambda x: x[1])[0]
    winner.resource = geometric_product(winner.resource, demand)
    return winner

def top_k_mask(values: list[float], k: int) -> list[int]:
    k = max(0, min(k, len(values)))
    winners = {i for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]}
    return list(winners)

def load_with_eviction(model: ModelTier, 
                       loaded: dict[str, ModelTier], 
                       ram_ceiling_mb: int) -> None:
    while loaded and model.ram_mb + sum(m.ram_mb for m in loaded.values()) > ram_ceiling_mb:
        loaded.pop(next(iter(loaded)))
    loaded[model.name] = model

if __name__ == "__main__":
    sites = [
        Site((0, 0), True, {frozenset(): 1.0}),
        Site((1, 1), True, {frozenset(): 2.0}),
    ]
    demand = {frozenset(): 3.0}
    winner = wta_resource_allocation(sites, demand)
    print(winner.resource)

    model_tiers = [
        ModelTier("model1", 1000, "T1"),
        ModelTier("model2", 2000, "T2"),
    ]
    loaded_models = {}
    load_with_eviction(model_tiers[0], loaded_models, 6000)
    load_with_eviction(model_tiers[1], loaded_models, 6000)
    print({m.name: m.ram_mb for m in loaded_models.values()})