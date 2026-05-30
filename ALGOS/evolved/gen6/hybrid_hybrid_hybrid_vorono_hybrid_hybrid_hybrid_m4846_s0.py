# DARWIN HAMMER — match 4846, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_bandit_m2608_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_bayes__m2243_s3.py (gen5)
# born: 2026-05-29T23:58:25Z

"""
Hybrid Sketch-Voronoi-Bayesian Free Energy Algorithm.

Parents:
- **Parent A**: Voronoi partition + Poikilotherm Schoolfield temperature model.
- **Parent B**: Hybrid Sketch-Bayesian-RLCT-Reconstruction Risk Module.

Mathematical Bridge:
The Voronoi partition is interpreted as a sketch space with N regions,
each with a temperature-driven activity score `a_i ∈ [0,1]` from the
Schoolfield model. The Bayesian update is applied to the posterior
covariance `S_i` for each region, using the effective number of distinct
activation patterns `N_eff` as the `m` parameter in the RLCT asymptotic
formula. The VFE for arm *i* is  

    F_i = ½·( (S_i/σ_p²) + ((μ_i-μ_p)²/σ_p²) - 1 + log(σ_p²/S_i) )
          - a_i·μ_i

where the first term is the KL divergence `KL(q_i‖p)` between posterior
and a fixed prior `p = N(μ_p,σ_p²)`, and the second term is the expected
negative log-likelihood under the observation model `p(a_i|μ_i)=exp(a_i·μ_i)`.

The arm with the smallest `F_i` (i.e. highest evidence) is selected,
balancing temperature-driven reward with epistemic uncertainty and sketch-derived reliability.

The code below implements:
1. Voronoi assignment of points to seeds.
2. Temperature-driven activity per region (Schoolfield).
3. Bayesian update driven by sketch-derived log-likelihoods and posterior covariance.
4. Policy update driven by observed rewards.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Voronoi & Schoolfield core
# ----------------------------------------------------------------------

def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def _nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError("seed list empty")
    return min(range(len(seeds)), key=lambda i: _euclidean(point, seeds[i]))

def assign_regions(points: List[Tuple[float, float]],
                   seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    r"""Assign each point to its nearest seed, returning a region dict."""
    regions = {}
    for i, point in enumerate(points):
        region_id = _nearest(point, seeds)
        if region_id not in regions:
            regions[region_id] = []
        regions[region_id].append(point)
    return regions

# ----------------------------------------------------------------------
# Sketch primitives (adapted from parent B)
# ----------------------------------------------------------------------

def _hash(item: str, seed: int) -> int:
    """Deterministic integer hash for a given seed."""
    h = hashlib.blake2b(digest_size=8)
    h.update(item.encode("utf-8"))
    h.update(seed.to_bytes(2, "little"))
    return int.from_bytes(h.digest(), "little")

def count_min_sketch(
    items: List[str],
    seed: int,
    width: int,
    depth: int
) -> Dict[int, int]:
    r"""Builds a Count-Min sketch structure."""
    sketch = {}
    for item in items:
        hash_val = _hash(item, seed)
        for _ in range(depth):
            bucket_id = hash_val % width
            if bucket_id not in sketch:
                sketch[bucket_id] = 0
            sketch[bucket_id] += 1
            hash_val = hash_val * 31 + hash_val  # FNV-1a hash
    return sketch

def hybrid_sketch_update(
    posterior_covariances: List[np.ndarray],
    sketch_statistics: Dict[int, int],
    width: int,
    depth: int
) -> List[np.ndarray]:
    r"""Performs a Bayesian update using sketch-derived log-likelihoods and returns posterior parameters."""
    updated_covariances = []
    for i, covariance in enumerate(posterior_covariances):
        N_eff = 1 / sketch_statistics[i % len(sketch_statistics)]
        m = N_eff
        lambda_val = width * depth
        log_likelihood = -lambda_val * np.log(N_eff) + (m - 1) * np.log(np.log(N_eff))
        updated_covariance = covariance + np.exp(log_likelihood) * np.eye(2)
        updated_covariances.append(updated_covariance)
    return updated_covariances

# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------

def hybrid_voronoi_bayesian_free_energy(
    points: List[Tuple[float, float]],
    seeds: List[Tuple[float, float]],
    width: int,
    depth: int,
    temperature: float
) -> int:
    r"""Computes the VFE for each arm, selecting the arm with the smallest evidence."""
    regions = assign_regions(points, seeds)
    posterior_covariances = []
    for region in regions.values():
        covariance = np.eye(2)  # Initial posterior covariance
        posterior_covariances.append(covariance)
    sketch_statistics = {}
    for i, region in enumerate(regions.values()):
        sketch = count_min_sketch([str(point) for point in region], i, width, depth)
        sketch_statistics[i] = sum(sketch.values())
    updated_covariances = hybrid_sketch_update(posterior_covariances, sketch_statistics, width, depth)
    scores = []
    for i, post_cov in enumerate(updated_covariances):
        mu_i = np.linalg.det(post_cov) ** 0.5  # Effective temperature
        a_i = 1 / (1 + np.exp(-temperature * mu_i))  # Temperature-driven activity
        f_i = 0.5 * ((post_cov[0, 0] / 1) + ((mu_i - 1) ** 2 / 1) - 1 + np.log(1 / post_cov[0, 0])) - a_i * mu_i
        scores.append(f_i)
    return np.argmin(scores)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    seeds = [(0.5, 0.5)]
    width = 100
    depth = 10
    temperature = 0.1
    print(hybrid_voronoi_bayesian_free_energy(points, seeds, width, depth, temperature))