# DARWIN HAMMER — match 1529, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s6.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m1120_s2.py (gen5)
# born: 2026-05-29T23:37:15Z

# hybrid_voronoi_bandit_hybrid_s11.py:
"""
Hybrid Algorithm: Voronoi Partition + Bandit Router + Honeybee Store + Schoolfield Temperature

Parents:
- hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s3.py (Voronoi partition and endpoint circuit-breaker)
- hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m1120_s2.py (Bandit router, honeybee store, Schoolfield temperature model, and graph curvature)

Mathematical Bridge:
1. Compute a Voronoi region for each point in the dataset using the Voronoi partition algorithm.
2. For each Voronoi region, compute the node-wise Ollivier-Ricci curvature as the raw expected reward for the bandit.
3. The honeybee store value is transformed by σ_S = 1/(1+exp(-S_i)) and multiplies the curvature, yielding a temperature-aware expected reward ȓ_i = ϰ_i·σ_S.
4. The Schoolfield developmental rate λ_T(T) (temperature T in °C) scales the learning-rate η = η₀·λ_T·σ_S.
5. After an action (node) a is chosen, the store is updated S_a ← S_a + r_a – baseline and the adjacency edges incident to a are linearly updated:
   A[a, j] ← A[a, j]·(1 + η·ΔS_a)  for all neighbours j.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, FrozenSet
import numpy as np

# ----------------------------------------------------------------------
# Core Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = FrozenSet[int]          # basis blade represented by a set of indices
Multivector = Dict[Blade, float]  # mapping blade → coefficient

# ----------------------------------------------------------------------
# Shared data structures (identical to parents)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Voronoi partition helpers (Parent A)
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_voronoi_regions(points: List[Point],
                            sites: List[Point]) -> Dict[int, List[Point]]:
    """
    Assign each point to the index of the nearest site.
    Returns a dict ``site_index → list[points]``.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions


# ----------------------------------------------------------------------
# Bandit selection (Parent B)
# ----------------------------------------------------------------------
def compute_curvature(adjacency_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the Ollivier-Ricci curvature for each node in the graph.
    """
    # Note: Simplified implementation for demonstration purposes
    return np.linalg.inv(adjacency_matrix) @ np.ones_like(adjacency_matrix)


def transform_honeybee_store(store_values: np.ndarray) -> np.ndarray:
    """
    Transform the honeybee store values using the sigmoid function.
    """
    return 1 / (1 + np.exp(-store_values))


def schoolfield_temperature_model(temperature: float) -> float:
    """
    Compute the Schoolfield developmental rate λ_T(T) for a given temperature T.
    """
    # Note: Simplified implementation for demonstration purposes
    return 1.0 + (temperature - 25) / 50


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_bandit_selection(points: List[Point],
                            adjacency_matrix: np.ndarray,
                            store_values: np.ndarray,
                            temperature: float) -> List[BanditAction]:
    """
    Compute the expected rewards and confidence bounds for each node in the graph.
    """
    voronoi_regions = compute_voronoi_regions(points, points)
    curvatures = compute_curvature(adjacency_matrix)
    transformed_store_values = transform_honeybee_store(store_values)
    schoolfield_rate = schoolfield_temperature_model(temperature)
    expected_rewards = curvatures * transformed_store_values * schoolfield_rate
    confidence_bounds = np.sqrt(np.diag(adjacency_matrix)) * transformed_store_values
    return [BanditAction(str(i), 1.0, expected_reward, confidence_bound, "Hybrid") for i, (expected_reward, confidence_bound) in enumerate(zip(expected_rewards, confidence_bounds))]


def hybrid_update(adjacency_matrix: np.ndarray,
                  store_values: np.ndarray,
                  action_id: int,
                  reward: float) -> np.ndarray:
    """
    Update the adjacency matrix and store values after an action is chosen.
    """
    store_values[action_id] += reward - np.mean(store_values)
    adjacency_matrix[action_id, :] *= (1 + 0.1 * store_values[action_id])
    return adjacency_matrix, store_values


def hybrid_test() -> None:
    """
    Smoke test the hybrid algorithm.
    """
    points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    adjacency_matrix = np.array([[0, 1, 1, 1],
                                 [1, 0, 1, 1],
                                 [1, 1, 0, 1],
                                 [1, 1, 1, 0]])
    store_values = np.array([0.5, 0.5, 0.5, 0.5])
    temperature = 25
    bandit_actions = hybrid_bandit_selection(points, adjacency_matrix, store_values, temperature)
    print(bandit_actions)
    adjacency_matrix, store_values = hybrid_update(adjacency_matrix, store_values, 0, 1.0)
    print(adjacency_matrix)
    print(store_values)


if __name__ == "__main__":
    hybrid_test()