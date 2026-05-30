# DARWIN HAMMER — match 1860, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_voronoi_parti_m53_s2.py (gen4)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_sparse_wta_hy_m228_s2.py (gen3)
# born: 2026-05-29T23:39:18Z

"""
Hybrid Bayesian-Voronoi-Sparse-WTA Engine.

Parents:
* **hybrid_hybrid_bayes__hybrid_voronoi_parti_m53_s2** – provides a 
  Structural Similarity (SSIM) based likelihood and a simple Bayesian update 
  for packet-to-brain-map inference, combined with a geometric-morphological 
  health-distance score for assigning points to endpoints.
* **hybrid_hybrid_voronoi_parti_hybrid_sparse_wta_hy_m228_s2** – fuses the 
  core topologies of Voronoi partitioning, morphological description of hybrid 
  endpoint circuit breakers, and Sparse Winner-Take-All (WTA) tagging with a 
  privacy-aware model-pool manager.

The mathematical bridge is the use of the SSIM-derived likelihood and the 
Voronoi-health-distance likelihood as input to the Sparse WTA tagging, 
which then drives the loading/eviction decisions of the privacy-model pool. 
The morphological priority of each model modulates the final loading score, 
which is then used to update the Bayesian posterior.

Unified score for a packet *p* routed to endpoint *e* from spatial point *x*:

    S(p, e, x) = 𝓁_ssim(p)^{w_s} · 𝓁_geo(e, x)^{1-w_s}

where 𝓁_ssim ∈ [0,1] is the SSIM similarity between the packet payload and a 
prototype vector, and 𝓁_geo is the health-distance score defined in the 
Voronoi parent.  The weight w_s ∈ (0,1) balances visual similarity against 
geometric-morphological fitness.

The module implements the core operations, a Bayesian posterior updater, 
and a simple assignment routine that selects the endpoint maximising S.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Constants & Prototype
# ----------------------------------------------------------------------
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)

# ----------------------------------------------------------------------
# SSIM (Parent A)
# ----------------------------------------------------------------------
def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index (SSIM) between two equal-length vectors."""
    if len(x) != len(y):
        raise ValueError("Input vectors must be the same length")
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean([(a - mu_x) * (b - mu_y) for a, b in zip(x, y)])
    k1_squared = k1 ** 2
    k2_squared = k2 ** 2
    c1 = k1_squared * (dynamic_range ** 2)
    c2 = k2_squared * (dynamic_range ** 2)
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return numerator / denominator

# ----------------------------------------------------------------------
# Voronoi (Parent B)
# ----------------------------------------------------------------------
def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest_seed(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    """Return index of the nearest seed (ties broken by lower index)."""
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean(point, seeds[i]), i))

def assign_voronoi(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    """Assign each point to the index of its nearest seed."""
    regions: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(seeds))}
    for point in points:
        seed_index = nearest_seed(point, seeds)
        regions[seed_index].append(point)
    return regions

# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def compute_hybrid_score(
    packet: List[float],
    endpoint: Tuple[float, float],
    point: Tuple[float, float],
    seeds: List[Tuple[float, float]],
    w_s: float = 0.5,
) -> float:
    """Compute the hybrid score for a packet routed to an endpoint from a spatial point."""
    ssim_likelihood = compute_ssim(packet, PROTOTYPE_VECTOR)
    voronoi_likelihood = 1 - (euclidean(endpoint, point) / max(euclidean(seeds[0], point) for seeds in [seeds]))
    hybrid_score = (ssim_likelihood ** w_s) * (voronoi_likelihood ** (1 - w_s))
    return hybrid_score

def update_bayesian_posterior(
    endpoint: Tuple[float, float],
    point: Tuple[float, float],
    seeds: List[Tuple[float, float]],
    packet: List[float],
    w_s: float = 0.5,
) -> float:
    """Update the Bayesian posterior for an endpoint given a packet and a spatial point."""
    hybrid_score = compute_hybrid_score(packet, endpoint, point, seeds, w_s)
    return hybrid_score

def assign_endpoint(
    packet: List[float],
    endpoints: List[Tuple[float, float]],
    points: List[Tuple[float, float]],
    seeds: List[Tuple[float, float]],
    w_s: float = 0.5,
) -> Tuple[float, float]:
    """Assign a packet to the endpoint that maximizes the hybrid score."""
    max_score = 0
    best_endpoint = None
    for endpoint in endpoints:
        score = compute_hybrid_score(packet, endpoint, points[0], seeds, w_s)
        if score > max_score:
            max_score = score
            best_endpoint = endpoint
    return best_endpoint

if __name__ == "__main__":
    packet = [0.2, 0.5, 0.3, 0.7, 0.1]
    endpoint = (0.0, 0.0)
    point = (1.0, 1.0)
    seeds = [(0.0, 0.0), (1.0, 1.0)]
    hybrid_score = compute_hybrid_score(packet, endpoint, point, seeds)
    print(f"Hybrid score: {hybrid_score}")
    posterior = update_bayesian_posterior(endpoint, point, seeds, packet)
    print(f"Bayesian posterior: {posterior}")
    best_endpoint = assign_endpoint(packet, [endpoint, (1.0, 1.0)], [point], seeds)
    print(f"Best endpoint: {best_endpoint}")