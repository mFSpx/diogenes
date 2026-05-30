# DARWIN HAMMER — match 5198, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_perceptual_de_hybrid_hybrid_ternar_m1907_s1.py (gen4)
# parent_b: hybrid_variational_free_ene_hybrid_hybrid_label__m1556_s6.py (gen4)
# born: 2026-05-30T00:00:33Z

"""Hybrid Perceptual‑RBF Voronoi‑Free‑Energy Router.

Parents
-------
* hybrid_hybrid_perceptual_de_hybrid_hybrid_ternar_m1907_s1.py (perceptual hashing + Voronoi‑ternary routing)
* hybrid_variational_free_ene_hybrid_hybrid_label__m1556_s6.py (Gaussian KL & free‑energy)

Mathematical Bridge
-------------------
For every seed *s* we keep a Gaussian belief 𝒩(μ_s,σ_s²) describing the
distribution of its latent “failure‑probability”.  The variational free energy

    F_s = KL[𝒩(μ_q,σ_q)‖𝒩(μ_s,σ_s)] – ℓ_s

is used as the surrogate posterior mean failure probability ĥ(s) that appears
in the edge cost of the original hybrid router.  The edge cost between a point
p (with Euclidean coordinate *p.pos* and feature vector *p.vec*) and a seed s
becomes

    c(p,s) = λ·‖p.pos – s.pos‖₂  +  μ·F_s

where λ,μ ≥ 0 are user‑defined weights.  The perceptual hash (phash) of the
feature vector clusters points and seeds; only seeds whose hash equals the
point’s hash are considered for routing.  The algorithm finally selects, for
each point, the three seeds with smallest c(p,s) that are not “open” (the
open‑circuit flag is omitted in this minimal implementation).

The code below fuses the two parent modules by re‑using the hashing utilities,
the Euclidean distance, the KL‑divergence / free‑energy formulas, and by
wrapping them in a coherent hybrid routing pipeline.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Parent‑A utilities (hashing & geometry)
# ---------------------------------------------------------------------------

Vector = List[float]
Point2D = Tuple[float, float]


def compute_dhash(values: List[float]) -> int:
    """Difference hash: 1 bit per adjacent pair, 1 if decreasing."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits


def compute_phash(values: List[float]) -> int:
    """Average hash limited to first 64 values."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers."""
    return bin(a ^ b).count('1')


def euclidean_distance(p: Point2D, q: Point2D) -> float:
    """Euclidean distance between two points."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


# ---------------------------------------------------------------------------
# Parent‑B utilities (Gaussian KL & free energy)
# ---------------------------------------------------------------------------


def kl_gaussian(
    mu_q: np.ndarray | float,
    sigma_q: np.ndarray | float,
    mu_p: np.ndarray | float,
    sigma_p: np.ndarray | float,
) -> float:
    """KL divergence KL[𝒩(μ_q,σ_q)‖𝒩(μ_p,σ_p)] for diagonal Gaussians."""
    mu_q = np.asarray(mu_q, dtype=float)
    sigma_q = np.asarray(sigma_q, dtype=float)
    mu_p = np.asarray(mu_p, dtype=float)
    sigma_p = np.asarray(sigma_p, dtype=float)

    if np.any(sigma_q <= 0) or np.any(sigma_p <= 0):
        raise ValueError("Standard deviations must be strictly positive.")

    term1 = np.log(sigma_p / sigma_q)
    term2 = (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p ** 2)
    kl = term1 + term2 - 0.5
    return float(np.sum(kl))


def free_energy_gaussian(
    mu_q: np.ndarray | float,
    sigma_q: np.ndarray | float,
    mu_p: np.ndarray | float,
    sigma_p: np.ndarray | float,
    log_likelihood: float,
) -> float:
    """Variational free energy = KL – log‑likelihood."""
    return kl_gaussian(mu_q, sigma_q, mu_p, sigma_p) - log_likelihood


# ---------------------------------------------------------------------------
# Data structures for the hybrid system
# ---------------------------------------------------------------------------


@dataclass
class Seed:
    """A routing seed with spatial position and a Gaussian belief."""
    id: str
    pos: Point2D
    mu: np.ndarray          # mean of latent failure probability (vector)
    sigma: np.ndarray       # std‑dev (vector, >0)
    features: List[float]  # raw feature vector for hashing
    open_circuit: bool = False  # placeholder flag


@dataclass
class Point:
    """A point to be routed."""
    id: str
    pos: Point2D
    vec: List[float]       # feature vector used for perceptual hashing


# ---------------------------------------------------------------------------
# Hybrid core functions
# ---------------------------------------------------------------------------


def seed_free_energy(seed: Seed, point: Point) -> float:
    """
    Compute a surrogate posterior mean failure probability for a seed.

    The point vector is interpreted as a noisy observation of the seed's latent
    variable.  We treat the observation as a Gaussian with mean = point.vec and
    unit variance, then compute the free energy of the seed's belief w.r.t. this
    observation.
    """
    # Observation model: μ_q = point.vec, σ_q = 1 (isotropic)
    mu_q = np.asarray(point.vec, dtype=float)
    sigma_q = np.ones_like(mu_q)

    mu_p = seed.mu
    sigma_p = seed.sigma

    # Simple Gaussian log‑likelihood: -0.5 * ||point.vec - seed.mu||²
    # (ignoring constants)
    diff = mu_q - mu_p
    log_likelihood = -0.5 * np.sum(diff ** 2)

    return free_energy_gaussian(mu_q, sigma_q, mu_p, sigma_p, log_likelihood)


def hybrid_edge_cost(point: Point, seed: Seed, lam: float, mu: float) -> float:
    """
    Edge cost defined in the hybrid paper:
        c(p,s) = λ·‖p.pos – s.pos‖₂ + μ·F_s
    where F_s is the free‑energy based surrogate failure probability.
    """
    spatial = euclidean_distance(point.pos, seed.pos)
    fe = seed_free_energy(seed, point)
    return lam * spatial + mu * fe


def top_k_seeds_for_point(
    point: Point,
    seeds: List[Seed],
    lam: float,
    mu: float,
    k: int = 3,
) -> List[Tuple[Seed, float]]:
    """
    Return the k seeds with smallest hybrid cost that share the same perceptual
    hash cluster as the point and are not marked as open_circuit.
    """
    point_hash = compute_phash(point.vec)

    # Filter by hash equality and circuit state
    eligible = [
        (seed, hybrid_edge_cost(point, seed, lam, mu))
        for seed in seeds
        if not seed.open_circuit and compute_phash(seed.features) == point_hash
    ]

    # Sort by cost
    eligible.sort(key=lambda tup: tup[1])
    return eligible[:k]


def hybrid_route(
    points: List[Point],
    seeds: List[Seed],
    lam: float = 1.0,
    mu: float = 1.0,
    k: int = 3,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Perform the full hybrid routing pass.

    Returns a dict mapping point.id → list of (seed.id, cost) tuples ordered by
    increasing cost.  The function demonstrates the mathematical fusion of the
    two parents: hashing & Voronoi‑style spatial cost from Parent A and the
    free‑energy surrogate from Parent B.
    """
    routing: Dict[str, List[Tuple[str, float]]] = {}
    for pt in points:
        top = top_k_seeds_for_point(pt, seeds, lam, mu, k)
        routing[pt.id] = [(seed.id, cost) for seed, cost in top]
    return routing


# ---------------------------------------------------------------------------
# Auxiliary demonstration utilities
# ---------------------------------------------------------------------------


def generate_random_seed(seed_id: str, dim: int = 5) -> Seed:
    """Create a random seed with Gaussian belief and a random feature vector."""
    pos = (random.uniform(0, 100), random.uniform(0, 100))
    mu = np.random.randn(dim)
    sigma = np.abs(np.random.randn(dim)) + 0.1  # ensure >0
    features = np.random.randn(dim).tolist()
    return Seed(id=seed_id, pos=pos, mu=mu, sigma=sigma, features=features)


def generate_random_point(point_id: str, dim: int = 5) -> Point:
    """Create a random point with a feature vector."""
    pos = (random.uniform(0, 100), random.uniform(0, 100))
    vec = np.random.randn(dim).tolist()
    return Point(id=point_id, pos=pos, vec=vec)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Create a modest dataset
    seeds = [generate_random_seed(f"seed_{i}") for i in range(10)]
    points = [generate_random_point(f"pt_{i}") for i in range(20)]

    # Run the hybrid router
    result = hybrid_route(points, seeds, lam=0.8, mu=0.5, k=3)

    # Print a few entries to verify that the code executed
    for pt_id, assignments in list(result.items())[:5]:
        print(f"Point {pt_id} →")
        for seed_id, cost in assignments:
            print(f"    Seed {seed_id} with cost {cost:.4f}")
    print("\nRouting completed without errors.")