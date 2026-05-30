# DARWIN HAMMER — match 4457, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m1513_s0.py (gen6)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_vorono_m1224_s3.py (gen5)
# born: 2026-05-29T23:55:59Z

"""
This module fuses the hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m1513_s0 and 
hybrid_hybrid_fisher_locali_hybrid_hybrid_vorono_m1224_s3 algorithms.

The mathematical bridge between these two algorithms lies in representing the bandit router's 
action space as a probability distribution over the Voronoi regions, allowing for a novel 
exploration-exploitation trade-off in a label-aware and spatially-structured manner.

The bandit router's action space is used to modulate the pheromone signal values, 
which in turn are used to update the Fisher-information scoring and Voronoi partitioning process.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Tuple

R_CAL = 1.987  
K25 = 298.15  

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.delta_h_activation * (1 / K25 - 1 / temp_k)
    return math.exp(numerator)

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_voronoi_regions(points: list[tuple[float, float]], 
                            sites: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
    """
    Assign each point to the index of the nearest site.
    Returns a dict ``site_index → list[points]``.
    """
    regions: dict[int, list[tuple[float, float]]] = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions

def pheromone_modulation(label: str, action_space: List[BanditAction]) -> float:
    modulation_factor = 0.0
    for action in action_space:
        modulation_factor += action.propensity * developmental_rate(303.15)
    return modulation_factor

def hybrid_operation(points: list[tuple[float, float]], 
                     sites: list[tuple[float, float]], 
                     action_space: List[BanditAction]) -> dict[int, list[tuple[float, float]]]:
    modulation_factor = pheromone_modulation("voronoi", action_space)
    voronoi_regions = compute_voronoi_regions(points, sites)
    fisher_scores = {}
    for region, points in voronoi_regions.items():
        center = sites[region]
        width = 1.0
        theta = np.mean([point[0] for point in points])
        fisher_scores[region] = fisher_score(theta, center[0], width) * modulation_factor
    return {region: points for region, points in voronoi_regions.items() if fisher_scores[region] > 0.0}

def smoke_test() -> None:
    points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    sites = [(0.0, 0.0), (10.0, 10.0)]
    action_space = [BanditAction("action1", 0.5, 1.0, 0.1, "algorithm1"), 
                    BanditAction("action2", 0.3, 2.0, 0.2, "algorithm2")]
    voronoi_regions = hybrid_operation(points, sites, action_space)
    print(voronoi_regions)

if __name__ == "__main__":
    smoke_test()