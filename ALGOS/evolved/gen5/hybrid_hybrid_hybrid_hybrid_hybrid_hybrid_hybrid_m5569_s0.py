# DARWIN HAMMER — match 5569, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m67_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_pherom_m30_s2.py (gen3)
# born: 2026-05-30T00:02:47Z

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# Constants from parent A
_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]
_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)

# Constants from parent B
GEOMETRIC_PRODUCT_WEIGHTS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float64)

# Data structures (from Parent B)
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

# Simple in-memory policy store (Parent B)
_POLICY: dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def shannon_entropy(probabilities: List[float]) -> float:
    return -sum([p * math.log2(p) for p in probabilities if p != 0.0])

def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: tuple[float, float], seeds: list[tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[tuple[float, float]], seeds: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)  
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    def __init__(self, components, n):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k):
        return Multivector(self.components.get(k, 0.0), self.n)

def hybrid_geometric_shannon_entropy(points: list[tuple[float, float]], seeds: list[tuple[float, float]], pheromone_distribution: List[float]) -> float:
    regions = assign(points, seeds)
    weighted_regions = {}
    for region, points in regions.items():
        weighted_regions[region] = sum([pheromone_distribution[region] * pheromone_distribution[region] for _ in points])
    return shannon_entropy([weighted_regions[i] / sum(weighted_regions.values()) for i in weighted_regions])

def hybrid_multivector_shannon_entropy(components: Dict[int, float], n: int, pheromone_distribution: List[float]) -> float:
    multivector = Multivector(components, n)
    weighted_components = {}
    for k, v in multivector.components.items():
        weighted_components[k] = v * pheromone_distribution[k]
    return shannon_entropy([weighted_components[i] / sum(weighted_components.values()) for i in weighted_components])

def hybrid_bandit_pheromone_update(updates: List[BanditUpdate], pheromone_distribution: List[float]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward) * pheromone_distribution[u.action_id]
        s[1] += 1.0 * pheromone_distribution[u.action_id]

if __name__ == "__main__":
    # Smoke test
    points = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    seeds = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    pheromone_distribution = [0.2, 0.3, 0.5]
    print(hybrid_geometric_shannon_entropy(points, seeds, pheromone_distribution))