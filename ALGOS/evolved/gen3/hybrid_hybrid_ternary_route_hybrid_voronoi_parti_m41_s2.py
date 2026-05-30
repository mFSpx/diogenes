# DARWIN HAMMER — match 41, survivor 2
# gen: 3
# parent_a: hybrid_ternary_router_hybrid_minimum_cost__m36_s3.py (gen2)
# parent_b: hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s1.py (gen2)
# born: 2026-05-29T23:26:37Z

"""Hybrid Voronoi‑Ternary Minimum‑Cost Router with Circuit‑Breaker

Parents
-------
* **Algorithm A** – *ternary_router* + *hybrid_minimum_cost_tree_bayes_update*.
  It builds a 3‑ary routing tree whose edge weights are continuously
  refined by a Bayesian update of observed successes/failures.

* **Algorithm B** – *voronoi_partition* + *endpoint_circuit_breaker*.
  It partitions a set of 2‑D points into Voronoi cells defined by a set of
  seed points and guards each seed (endpoint) with a simple failure‑counter
  circuit‑breaker.

Mathematical Bridge
-------------------
The hybrid algorithm first **partitions** the spatial domain using the
Voronoi construction of Algorithm B.  Within each Voronoi cell we construct a
**ternary minimum‑cost routing tree** (Algorithm A).  The cost of an edge
between a point *p* and a seed *s* is defined as


c(p, s) = λ·‖p‑s‖₂  +  μ·ĥ(s)


where `‖·‖₂` is the Euclidean distance, `ĥ(s)` is the Bayesian posterior mean
failure probability of seed *s* (updated by the circuit‑breaker statistics),
and `λ, μ ≥ 0` are weighting hyper‑parameters.  The ternary router selects,
for each point, the three seeds with the smallest `c(p, s)` that are not
currently “open” in their circuit‑breaker.  This fuses the spatial
partitioning, the 3‑ary routing topology, and the Bayesian cost adaptation
into a single unified system.
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def emit_json(obj: Any) -> None:
    """Print a JSON object in a deterministic order."""
    print(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))


# ----------------------------------------------------------------------
# Voronoi utilities (from parent B)
# ----------------------------------------------------------------------


Point = Tuple[float, float]


def euclidean(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the closest seed (ties broken by smallest index)."""
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean(point, seeds[i]), i))


def assign_voronoi(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """
    Partition ``points`` into Voronoi regions defined by ``seeds``.

    Returns
    -------
    dict[int, list[Point]]
        Mapping from seed index to the list of points belonging to its cell.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


# ----------------------------------------------------------------------
# Circuit‑breaker (from parent B, completed)
# ----------------------------------------------------------------------


class EndpointCircuitBreaker:
    """
    Simple failure counter that opens after a configurable threshold.
    When opened the endpoint is excluded from routing decisions until a
    manual reset (simulated here by ``reset``).
    """

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.is_open = False

    def record_success(self) -> None:
        """A successful call reduces the perceived failure pressure."""
        if self.is_open:
            # successful call while open may indicate recovery; we reset.
            self.reset()
        # successes do not directly affect the counter but can be used
        # for Bayesian updates elsewhere.

    def record_failure(self) -> None:
        """Increment failure counter and open if threshold is crossed."""
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.is_open = True

    def reset(self) -> None:
        """Close the breaker and clear failure history."""
        self.failures = 0
        self.is_open = False

    def status(self) -> Dict[str, Any]:
        return {"failures": self.failures, "is_open": self.is_open}


# ----------------------------------------------------------------------
# Bayesian cost estimator (core of parent A)
# ----------------------------------------------------------------------


class BayesianFailureEstimator:
    """
    Maintains a Beta(α, β) posterior for the failure probability of an endpoint.
    α and β are initialized to 1 (uniform prior).  Each success increments β,
    each failure increments α.
    The posterior mean ``α/(α+β)`` is used as the failure cost component.
    """

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
        """Posterior mean of the failure probability."""
        return self.alpha / (self.alpha + self.beta)


# ----------------------------------------------------------------------
# Hybrid routing logic
# ----------------------------------------------------------------------


def compute_edge_cost(
    point: Point,
    seed: Point,
    estimator: BayesianFailureEstimator,
    lambda_dist: float = 1.0,
    mu_failure: float = 5.0,
) -> float:
    """
    Composite cost used by the ternary router.

    Parameters
    ----------
    point, seed
        2‑D coordinates.
    estimator
        Bayesian estimator for the seed's failure probability.
    lambda_dist, mu_failure
        Weighting hyper‑parameters (λ and μ in the docstring).

    Returns
    -------
    float
        Weighted sum of Euclidean distance and estimated failure cost.
    """
    distance_cost = lambda_dist * euclidean(point, seed)
    failure_cost = mu_failure * estimator.mean
    return distance_cost + failure_cost


def ternary_route(
    point: Point,
    seeds: List[Point],
    estimators: List[BayesianFailureEstimator],
    breakers: List[EndpointCircuitBreaker],
    lambda_dist: float = 1.0,
    mu_failure: float = 5.0,
) -> List[int]:
    """
    Select up to three viable seeds for ``point`` using the composite cost.
    Seeds whose circuit‑breaker is open are ignored.

    Returns
    -------
    list[int]
        Indices of the chosen seeds ordered by increasing cost.
    """
    viable = [
        (i, compute_edge_cost(point, seeds[i], estimators[i], lambda_dist, mu_failure))
        for i in range(len(seeds))
        if not breakers[i].is_open
    ]
    viable.sort(key=lambda x: x[1])
    return [idx for idx, _ in viable[:3]]


def hybrid_route(
    points: List[Point],
    seeds: List[Point],
    breakers: List[EndpointCircuitBreaker],
    lambda_dist: float = 1.0,
    mu_failure: float = 5.0,
) -> Dict[int, List[int]]:
    """
    Full hybrid operation:
    1. Partition points into Voronoi cells.
    2. Within each cell, route each point to its three best seeds using the
       ternary minimum‑cost rule.
    3. After routing, simulate a random success/failure per chosen seed and
       update both the Bayesian estimator and the circuit‑breaker.

    Returns
    -------
    dict[int, list[int]]
        Mapping from point index to the list of selected seed indices.
    """
    # Initialise Bayesian estimators (one per seed)
    estimators = [BayesianFailureEstimator() for _ in seeds]

    # 1. Voronoi partition
    regions = assign_voronoi(points, seeds)

    # Result container
    routing: Dict[int, List[int]] = {}

    # 2. Process each region independently
    for seed_idx, region_points in regions.items():
        for p in region_points:
            # Global point index (needed for deterministic output)
            point_idx = points.index(p)

            selected = ternary_route(
                p,
                seeds,
                estimators,
                breakers,
                lambda_dist=lambda_dist,
                mu_failure=mu_failure,
            )
            routing[point_idx] = selected

            # 3. Simulate outcome for each selected seed
            for s_idx in selected:
                # Randomly decide success/failure (10 % failure rate)
                success = random.random() > 0.10
                estimators[s_idx].update(success)
                if success:
                    breakers[s_idx].record_success()
                else:
                    breakers[s_idx].record_failure()

    return routing


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(0)

    # Generate synthetic data
    NUM_SEEDS = 7
    NUM_POINTS = 50

    seeds = [tuple(coord) for coord in np.random.uniform(0, 100, size=(NUM_SEEDS, 2))]
    points = [tuple(coord) for coord in np.random.uniform(0, 100, size=(NUM_POINTS, 2))]

    # One circuit‑breaker per seed
    breakers = [EndpointCircuitBreaker(failure_threshold=3) for _ in range(NUM_SEEDS)]

    # Execute hybrid routing
    routing_map = hybrid_route(
        points,
        seeds,
        breakers,
        lambda_dist=0.8,
        mu_failure=4.0,
    )

    # Emit a concise JSON summary
    summary = {
        "timestamp": now_z(),
        "num_seeds": NUM_SEEDS,
        "num_points": NUM_POINTS,
        "routing": routing_map,
        "breakers": [br.status() for br in breakers],
    }
    emit_json(summary)