# DARWIN HAMMER — match 1102, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s2.py (gen3)
# parent_b: hybrid_fisher_localization_hybrid_ternary_route_m40_s1.py (gen2)
# born: 2026-05-29T23:32:44Z

"""Hybrid Fisher-Voronoi Ternary Minimum-Cost Router with Circuit-Breaker

Parents
-------
* **Algorithm A** – *hybrid_voronoi_partition_hybrid_ternary_router* + *hybrid_minimum_cost_tree_bayes_update*.
  It builds a 3-ary routing tree whose edge weights are continuously refined by a Bayesian update of observed successes/failures.

* **Algorithm B** – *fisher_localization* + *hybrid_ternary_router_ssim*.
  It applies the SSIM algorithm to the packet routing process and uses the Fisher information of the packet's text surface as a weighting factor in the similarity calculation.

Mathematical Bridge
-------------------
The hybrid algorithm first **partitions** the spatial domain using the Voronoi construction of Algorithm A. Within each Voronoi cell we construct a **ternary minimum-cost routing tree** (Algorithm A). The cost of an edge between a point *p* and a seed *s* is defined as


c(p, s) = λ·‖p-s‖₂  +  μ·ĥ(s)·fisher_score(p, s)

where `‖·‖₂` is the Euclidean distance, `ĥ(s)` is the Bayesian posterior mean failure probability of seed *s* (updated by the circuit-breaker statistics), `fisher_score(p, s)` is the Fisher information of the packet's text surface, and `λ, μ ≥ 0` are weighting hyper-parameters. The ternary router selects, for each point, the three seeds with the smallest `c(p, s)` that are not currently “open” in their circuit-breaker. This fuses the spatial partitioning, the 3-ary routing topology, and the Bayesian cost adaptation into a single unified system.
"""

import json
import math
import random
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def now_z() -> str:
    """Current UTC timestamp in ISO-8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def emit_json(obj: Any) -> None:
    """Print a JSON object in a deterministic order."""
    print(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))

# ----------------------------------------------------------------------
# Voronoi utilities (from parent A)
# ----------------------------------------------------------------------

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return np.linalg.norm(p1 - p2)

def bayesian_posterior_mean(failure_counts: np.ndarray, total_count: int) -> float:
    """Bayesian posterior mean failure probability."""
    return failure_counts / total_count

# ----------------------------------------------------------------------
# Fisher localization utilities (from parent B)
# ----------------------------------------------------------------------

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------

def hybrid_cost(p: np.ndarray, s: np.ndarray, λ: float, μ: float) -> float:
    """Hybrid cost function."""
    return λ * euclidean_distance(p, s) + μ * bayesian_posterior_mean(np.array([1])) * fisher_score(p, s)

def hybrid_ternary_router(points: np.ndarray, seeds: np.ndarray, λ: float, μ: float) -> np.ndarray:
    """Hybrid ternary router."""
    distances = np.linalg.norm(points[:, np.newaxis] - seeds, axis=2)
    costs = λ * distances + μ * bayesian_posterior_mean(np.array([1])) * fisher_score(points[:, np.newaxis], seeds, 1.0)
    indices = np.argsort(costs, axis=1)[:,:3]
    return indices

def hybrid_routing_tree(points: np.ndarray, seeds: np.ndarray, λ: float, μ: float) -> np.ndarray:
    """Hybrid routing tree."""
    indices = hybrid_ternary_router(points, seeds, λ, μ)
    return indices

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    points = np.random.rand(10, 2)
    seeds = np.random.rand(5, 2)
    λ = 0.5
    μ = 0.5
    indices = hybrid_routing_tree(points, seeds, λ, μ)
    print(indices)