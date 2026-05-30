# DARWIN HAMMER — match 5398, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1208_s7.py (gen6)
# parent_b: hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s4.py (gen3)
# born: 2026-05-30T00:01:40Z

"""Hybrid Physarum‑Voronoi Network

This module fuses the two parent algorithms:

* **Parent A** – Physarum‑inspired flux and conductance dynamics together with a
  sinusoidal weekday weight vector.
* **Parent B** – Planar Voronoi partitioning, Bayesian failure estimation and a
  simple endpoint circuit‑breaker.

Mathematical bridge
-------------------
Both parents operate on a *network* of entities:

* In A the network is an abstract graph whose edges have a length, a conductance
  `c_ij` and a flux `q_ij = c_ij / L_ij * (p_i - p_j)`.
* In B the network is a set of planar points (seeds) that induce a Voronoi
  diagram; each seed can be equipped with a Bayesian failure estimator
  `β_i = α_i/(α_i+β_i)`.

The fusion treats the Voronoi seeds as graph nodes.  Edge lengths are Euclidean
distances, conductances evolve with the Physarum ODE, and node pressures are
derived from a weighted combination of the weekday weight vector and the
current Bayesian failure estimate of each node.  Consequently the same
mathematical objects (pressures, fluxes, conductances) are driven by spatial
partitioning and epistemic uncertainty.

The core functions below implement this unified system.
"""

from __future__ import annotations

import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]

# ----------------------------------------------------------------------
# Parent A primitives
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on a single edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 0.1,
                       gain: float = 1.0,
                       decay: float = 0.01,
                       eps: float = 1e-12) -> float:
    """Conductance ODE step based on absolute flux."""
    delta = dt * (gain * abs(q) - decay * conductance)
    new_c = max(0.0, conductance + delta)
    return new_c


GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """Return a normalized weight vector that varies sinusoidally with day‑of‑week."""
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec


# ----------------------------------------------------------------------
# Parent B primitives
# ----------------------------------------------------------------------
def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean(point, seeds[i]), i))


def assign_voronoi(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to the nearest seed, returning region lists."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


class EndpointCircuitBreaker:
    """Simple circuit‑breaker that opens after a configurable number of failures."""
    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.is_open = False

    def record_success(self) -> None:
        if self.is_open:
            self.reset()

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.is_open = True

    def reset(self) -> None:
        self.failures = 0
        self.is_open = False

    def status(self) -> Dict[str, Any]:
        return {"failures": self.failures, "is_open": self.is_open}


class BayesianFailureEstimator:
    """Beta‑Bernoulli estimator of the probability of failure."""
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.alpha = alpha
        self.beta = beta

    def update(self, success: bool) -> None:
        if success:
            self.beta += 1.0
        else:
            self.alpha += 1.0

    @property
    def mean(self) -> float:
        """Mean failure probability."""
        return self.alpha / (self.alpha + self.beta)


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def compute_node_pressures(
    estimators: List[BayesianFailureEstimator],
    dow: int,
    groups: Sequence[str] = GROUPS,
) -> np.ndarray:
    """
    Compute a pressure for each node.

    Pressure_i = w_i * (1 - f_i)

    where:
        w_i – weekday weight for the node's group,
        f_i – current mean failure probability from the Bayesian estimator.
    """
    n = len(estimators)
    if n == 0:
        raise ValueError("At least one estimator required")
    # Align groups to nodes (repeat if fewer groups than nodes)
    groups_cycle = (list(groups) * ((n // len(groups)) + 1))[:n]
    weight_vec = weekday_weight_vector(groups_cycle, dow)
    pressures = np.empty(n, dtype=float)
    for i, est in enumerate(estimators):
        pressures[i] = weight_vec[i] * (1.0 - est.mean)
    return pressures


def hybrid_update_conductances(
    conductance: np.ndarray,
    seeds: List[Point],
    pressures: np.ndarray,
    dt: float = 0.1,
    gain: float = 1.0,
    decay: float = 0.01,
) -> np.ndarray:
    """
    Perform one Physarum conductance update over the fully connected graph
    induced by the Voronoi seeds.

    The conductance matrix is symmetric; diagonal entries are ignored.
    """
    n = len(seeds)
    if conductance.shape != (n, n):
        raise ValueError("Conductance matrix shape mismatch")
    new_c = conductance.copy()
    for i in range(n):
        for j in range(i + 1, n):
            L_ij = euclidean(seeds[i], seeds[j])
            q_ij = flux(conductance[i, j], L_ij, pressures[i], pressures[j])
            updated = update_conductance(conductance[i, j], q_ij, dt, gain, decay)
            new_c[i, j] = new_c[j, i] = updated
    return new_c


def hybrid_assign_and_evolve(
    points: List[Point],
    seeds: List[Point],
    conductance: np.ndarray,
    estimators: List[BayesianFailureEstimator],
    dow: int,
    dt: float = 0.1,
    distance_threshold: float = 5.0,
) -> Tuple[np.ndarray, List[BayesianFailureEstimator], Dict[int, List[Point]]]:
    """
    1. Partition `points` into Voronoi regions defined by `seeds`.
    2. For each region, compute the mean distance to its seed.
       * If the mean distance ≤ `distance_threshold` → record a success,
         otherwise a failure, updating the node's Bayesian estimator.
    3. Compute node pressures using weekday weights and the updated estimators.
    4. Update the conductance matrix via Physarum dynamics.

    Returns the new conductance matrix, the (mutated) estimators list,
    and the Voronoi region mapping.
    """
    # 1. Voronoi partition
    regions = assign_voronoi(points, seeds)

    # 2. Update Bayesian estimators per region
    for idx, region_pts in regions.items():
        if not region_pts:
            # No data → treat as neutral success to avoid dead‑lock
            estimators[idx].update(True)
            continue
        dists = [euclidean(p, seeds[idx]) for p in region_pts]
        mean_dist = sum(dists) / len(dists)
        success = mean_dist <= distance_threshold
        estimators[idx].update(success)

    # 3. Compute pressures
    pressures = compute_node_pressures(estimators, dow)

    # 4. Conductance update
    new_conductance = hybrid_update_conductances(conductance, seeds, pressures, dt)

    return new_conductance, estimators, regions


def hybrid_edge_cost(
    point: Point,
    seed: Point,
    estimator: BayesianFailureEstimator,
    dow: int,
    lambda_d: float = 1.0,
    lambda_f: float = 2.0,
) -> float:
    """
    A cost metric that blends geometric distance with epistemic failure risk
    and weekday modulation.

    cost = λ_d * distance + λ_f * (failure_prob) / w_dow

    where w_dow is the weekday weight for the seed's group (default group order).
    """
    distance = euclidean(point, seed)
    failure = estimator.mean
    # Use a generic group index based on seed order
    weight = weekday_weight_vector(GROUPS, dow)[0]  # same weight for illustration
    return lambda_d * distance + lambda_f * (failure / max(weight, 1e-12))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed points (simple square)
    seeds: List[Point] = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]

    # Random points in the bounding box
    random.seed(42)
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(200)]

    n = len(seeds)
    # Initialise a fully connected conductance matrix with small positive values
    conductance = np.full((n, n), 0.1, dtype=float)
    np.fill_diagonal(conductance, 0.0)

    # One Bayesian estimator per seed
    estimators = [BayesianFailureEstimator() for _ in range(n)]

    # Day of week (0 = Monday)
    dow = datetime.now(timezone.utc).weekday()

    # Run a single hybrid iteration
    new_c, new_estimators, regions = hybrid_assign_and_evolve(
        points,
        seeds,
        conductance,
        estimators,
        dow,
        dt=0.05,
        distance_threshold=4.0,
    )

    # Simple sanity prints
    print("Updated conductance matrix:")
    print(new_c)
    print("\nEstimator means per node:")
    for i, est in enumerate(new_estimators):
        print(f"Node {i}: failure mean = {est.mean:.3f}")
    print("\nVoronoi region sizes:")
    for idx, pts in regions.items():
        print(f"Region {idx}: {len(pts)} points")