# DARWIN HAMMER — match 41, survivor 4
# gen: 3
# parent_a: hybrid_ternary_router_hybrid_minimum_cost__m36_s3.py (gen2)
# parent_b: hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s1.py (gen2)
# born: 2026-05-29T23:26:37Z

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def emit_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))

Point = Tuple[float, float]

def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean(point, seeds[i]), i))

def assign_voronoi(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

class EndpointCircuitBreaker:
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
        return self.alpha / (self.alpha + self.beta)

def compute_edge_cost(
    point: Point,
    seed: Point,
    estimator: BayesianFailureEstimator,
    lambda_dist: float = 1.0,
    mu_failure: float = 5.0,
    mu_variance: float = 0.1
) -> float:
    distance_cost = lambda_dist * euclidean(point, seed)
    failure_cost = mu_failure * estimator.mean
    variance_cost = mu_variance * (estimator.alpha / (estimator.alpha + estimator.beta) * (1 - estimator.alpha / (estimator.alpha + estimator.beta)))
    return distance_cost + failure_cost + variance_cost

def ternary_route(
    point: Point,
    seeds: List[Point],
    estimators: List[BayesianFailureEstimator],
    breakers: List[EndpointCircuitBreaker],
    lambda_dist: float = 1.0,
    mu_failure: float = 5.0,
    mu_variance: float = 0.1
) -> List[int]:
    viable = [
        (i, compute_edge_cost(point, seeds[i], estimators[i], lambda_dist, mu_failure, mu_variance))
        for i in range(len(seeds))
        if not breakers[i].is_open
    ]
    viable.sort(key=lambda x: x[1])
    return [idx for idx, _ in viable[:3]]

def hybrid_route(
    points: List[Point],
    seeds: List[Point],
    breakers: List[EndpointCircuitBreaker],
    estimators: List[BayesianFailureEstimator],
    lambda_dist: float = 1.0,
    mu_failure: float = 5.0,
    mu_variance: float = 0.1
) -> Dict[int, List[int]]:
    voronoi = assign_voronoi(points, seeds)
    routes = {}
    for i, point in enumerate(points):
        for seed_idx in voronoi:
            if point in voronoi[seed_idx]:
                routes[i] = ternary_route(point, seeds, estimators, breakers, lambda_dist, mu_failure, mu_variance)
                for seed in routes[i]:
                    success = random.random() > estimators[seed].mean
                    if success:
                        estimators[seed].update(True)
                        breakers[seed].record_success()
                    else:
                        estimators[seed].update(False)
                        breakers[seed].record_failure()
    return routes

def main():
    points = [(random.random(), random.random()) for _ in range(100)]
    seeds = [(random.random(), random.random()) for _ in range(10)]
    breakers = [EndpointCircuitBreaker() for _ in range(10)]
    estimators = [BayesianFailureEstimator() for _ in range(10)]
    routes = hybrid_route(points, seeds, breakers, estimators)
    emit_json(routes)

if __name__ == "__main__":
    main()