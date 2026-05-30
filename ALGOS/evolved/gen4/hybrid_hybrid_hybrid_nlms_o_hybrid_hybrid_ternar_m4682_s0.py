# DARWIN HAMMER — match 4682, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s5.py (gen2)
# parent_b: hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s3.py (gen3)
# born: 2026-05-29T23:57:20Z

"""
Hybrid Ternary Voronoi NLMS Fusion

This module fuses two parent algorithms:

* **Parent A** – hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s5.py: 
  NLMS adaptive filter and graph fusion.
* **Parent B** – hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s3.py: 
  Ternary Voronoi partition and endpoint circuit breaker.

The mathematical bridge between the two parents lies in the use of 
weighted similarity features in the NLMS algorithm and the 
assignment of points to seeds in the Voronoi partition.

In the hybrid algorithm, we use the NLMS update to adaptively 
weight the similarity features between points and seeds, and then 
use these weighted features to assign points to seeds in the 
Voronoi partition.
"""

import json
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS adaptation step.

    Args:
        weights: Current weight vector (shape (n,)).
        x: Input feature vector (shape (n,)).
        target: Desired scalar output.
        mu: Step‑size (0 < mu ≤ 1).
        eps: Small constant to avoid division by zero.

    Returns:
        (new_weights, error) where error = target - y.
    """
    y = predict(weights, x)
    error = target - y
    new_weights = weights + mu * error * x / (eps + np.dot(x, x))
    return new_weights, error

def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]], weights: np.ndarray) -> int:
    if not seeds:
        raise ValueError("seed list cannot be empty")
    weighted_distances = [euclidean(point, seed) * weights[i] for i, seed in enumerate(seeds)]
    return np.argmin(weighted_distances)

def assign_voronoi(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]], weights: np.ndarray) -> dict:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds, weights)].append(p)
    return regions

def hybrid_voronoi_nlms(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]], 
                         initial_weights: np.ndarray, targets: List[float], 
                         mu: float = 0.5, eps: float = 1e-9) -> Tuple[dict, np.ndarray]:
    weights = initial_weights
    for point, target in zip(points, targets):
        weights, _ = nlms_update(weights, np.array(point), target, mu, eps)
    return assign_voronoi(points, seeds, weights), weights

def main():
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(100)]
    seeds = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(5)]
    initial_weights = np.ones(2)
    targets = [euclidean(point, seeds[0]) for point in points]
    regions, final_weights = hybrid_voronoi_nlms(points, seeds, initial_weights, targets)
    print(json.dumps(regions, ensure_ascii=False, sort_keys=True, default=str))

if __name__ == "__main__":
    main()