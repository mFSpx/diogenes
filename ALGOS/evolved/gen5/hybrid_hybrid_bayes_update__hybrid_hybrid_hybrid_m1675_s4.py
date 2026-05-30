# DARWIN HAMMER — match 1675, survivor 4
# gen: 5
# parent_a: hybrid_bayes_update_hybrid_geometric_pro_m78_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_sheaf__m1068_s1.py (gen4)
# born: 2026-05-29T23:38:10Z

"""Hybrid Bayesian‑Geometric‑RBF Module
====================================

Parents
-------
* **Parent A** – `hybrid_bayes_update_hybrid_geometric_pro_m78_s0.py`  
  Provides Bayesian updates expressed with multivector (geometric algebra) points and
  Voronoi region assignments.

* **Parent B** – `hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_sheaf__m1068_s1.py`  
  Supplies radial‑basis‑function (Gaussian) interpolation, hash‑based similarity
  matrices and graph‑oriented feature handling.

Mathematical Bridge
-------------------
The bridge is built on the observation that a Bayesian posterior can be interpreted
as a **confidence weight** for each geometric seed. Those weights naturally feed
into a Gaussian radial basis function (RBF) – the same Gaussian that appears in
the RBF surrogate of Parent B.  

Thus:

1. Points are lifted to multivectors (Parent A) → distances → likelihoods.  
2. Priors are multiplied by those likelihoods → posterior **weights**.  
3. The posterior weights scale Gaussian RBF kernels centred at the same points,
   yielding a *Bayes‑weighted RBF* estimator (Parent B).  

Voronoi partitions (Parent A) are used to restrict the influence of each seed
to its neighbourhood, while the hash‑based similarity matrix (Parent B) can be
employed to modulate the RBF bandwidth according to feature similarity.

The three exported functions demonstrate this fused pipeline.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Geometric algebra utilities (from Parent A)
# ---------------------------------------------------------------------------

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j : j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> Tuple[frozenset[int], int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def point_to_mv(point: Tuple[float, float]) -> Tuple[float, float, float, float]:
    """Convert a 2‑tuple to a multivector (scalar+vector+bi‑vector)."""
    x, y = point
    return (x, y, 0.0, 0.0)


def mv_distance(mv_a: Tuple[float, float, float, float],
                mv_b: Tuple[float, float, float, float]) -> float:
    """Euclidean distance between two multivectors (treated as 4‑D vectors)."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(mv_a, mv_b)))


# ---------------------------------------------------------------------------
# Bayesian core (adapted from Parent A)
# ---------------------------------------------------------------------------

def bayes_marginal_mv(
    points: List[Tuple[float, float]],
    priors: List[float],
    observation: Tuple[float, float],
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Compute the Bayesian posterior for a set of points represented as multivectors.

    Parameters
    ----------
    points : list of (x, y)
        Seed points (the “hypotheses”).
    priors : list of float
        Prior probability for each point (must sum to 1).
    observation : (x, y)
        The measured datum.
    sigma : float
        Scale of the Gaussian likelihood (larger → flatter).

    Returns
    -------
    posterior : np.ndarray
        Normalised posterior probabilities, one per point.
    """
    if len(points) != len(priors):
        raise ValueError("points and priors must have the same length")
    obs_mv = point_to_mv(observation)

    # Convert to multivectors once
    mv_points = [point_to_mv(p) for p in points]

    # Gaussian likelihood based on multivector distance
    likelihoods = np.array(
        [math.exp(-mv_distance(p_mv, obs_mv) ** 2 / (2 * sigma ** 2)) for p_mv in mv_points],
        dtype=np.float64,
    )

    prior_arr = np.array(priors, dtype=np.float64)
    unnorm = prior_arr * likelihoods
    total = unnorm.sum()
    if total == 0.0:
        # avoid division by zero – fall back to uniform posterior
        return np.full_like(unnorm, 1.0 / len(unnorm))
    return unnorm / total


def voronoi_partition_bayes(
    points: List[Tuple[float, float]],
    priors: List[float],
    observation: Tuple[float, float],
    sigma: float = 1.0,
) -> Dict[int, List[int]]:
    """
    Assign each point to the Voronoi cell of the observation and return posterior
    probabilities restricted to that cell.

    Returns a dict mapping cell id (0 for the observation’s cell, 1… for others)
    to a list of point indices that belong to the cell.
    """
    # Compute posterior weights first
    posterior = bayes_marginal_mv(points, priors, observation, sigma)

    # Voronoi assignment: nearest seed to each point (including the observation as a seed)
    all_seeds = points + [observation]  # last seed is the observation itself
    mv_seeds = [point_to_mv(s) for s in all_seeds]

    assignments: Dict[int, List[int]] = {}
    for idx, p in enumerate(points):
        mv_p = point_to_mv(p)
        # find nearest seed index
        dists = [mv_distance(mv_p, s_mv) for s_mv in mv_seeds]
        nearest = int(np.argmin(dists))  # 0..len(points) (last index = observation)
        assignments.setdefault(nearest, []).append(idx)

    # Attach posterior weights to the assignment dict for possible downstream use
    # (stored as a side‑channel attribute)
    assignments["posterior"] = posterior.tolist()  # type: ignore

    return assignments


# ---------------------------------------------------------------------------
# Radial basis functions and similarity utilities (from Parent B)
# ---------------------------------------------------------------------------

Node = int
Graph = dict[Node, set[Node]]
FeatureVec = Tuple[float, ...]


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Standard Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: 1‑bit per value relative to the mean (max 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def similarity_matrix_mv(
    points: List[Tuple[float, float]],
) -> Tuple[np.ndarray, List[int]]:
    """
    Build a similarity matrix between multivector points using perceptual hashing.
    Mirrors `similarity_matrix` from Parent B but operates on the 4‑D multivector.
    """
    n = len(points)
    S = np.empty((n, n), dtype=np.float64)
    hashes = [compute_phash(list(point_to_mv(p))) for p in points]

    for i in range(n):
        for j in range(i, n):
            d = hamming_distance(hashes[i], hashes[j])
            sim = 1.0 - d / 64.0
            S[i, j] = sim
            S[j, i] = sim
    return S, list(range(n))


def bayes_weighted_rbf(
    query_points: List[Tuple[float, float]],
    seed_points: List[Tuple[float, float]],
    priors: List[float],
    observation: Tuple[float, float],
    sigma_likelihood: float = 1.0,
    sigma_rbf: float = 1.0,
) -> np.ndarray:
    """
    Perform RBF interpolation where each seed's amplitude is its Bayesian posterior
    (computed with multivector distances). The RBF uses a Gaussian kernel.

    Parameters
    ----------
    query_points : list of (x, y)
        Locations where the interpolated value is requested.
    seed_points : list of (x, y)
        Centers of the Gaussian basis functions.
    priors : list of float
        Prior probability for each seed (must sum to 1).
    observation : (x, y)
        Evidence that updates the priors.
    sigma_likelihood : float
        Bandwidth of the Bayesian Gaussian likelihood.
    sigma_rbf : float
        Bandwidth of the RBF Gaussian.

    Returns
    -------
    values : np.ndarray
        Interpolated scalar values at each query point.
    """
    # 1. Bayesian posterior weights for the seeds
    posterior = bayes_marginal_mv(seed_points, priors, observation, sigma_likelihood)

    # 2. Optional similarity‑based modulation of sigma_rbf per seed
    #    (more similar seeds get a slightly tighter kernel)
    S, _ = similarity_matrix_mv(seed_points)
    # Scale sigma per seed: sigma_i = sigma_rbf * (1 - mean_similarity_i/2)
    sigma_per_seed = sigma_rbf * (1.0 - np.mean(S, axis=1) / 2.0)

    # 3. Evaluate weighted RBF sum at each query point
    values = np.zeros(len(query_points), dtype=np.float64)
    mv_queries = [point_to_mv(q) for q in query_points]
    mv_seeds = [point_to_mv(p) for p in seed_points]

    for q_idx, q_mv in enumerate(mv_queries):
        acc = 0.0
        for s_idx, s_mv in enumerate(mv_seeds):
            r = mv_distance(q_mv, s_mv)
            kernel = math.exp(- (r ** 2) / (2 * sigma_per_seed[s_idx] ** 2))
            acc += posterior[s_idx] * kernel
        values[q_idx] = acc

    return values


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple synthetic scenario
    seed_pts = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)]
    priors = [0.25, 0.25, 0.25, 0.25]
    observation = (1.0, 1.0)

    # 1. Bayesian posterior check
    post = bayes_marginal_mv(seed_pts, priors, observation, sigma=0.8)
    print("Posterior:", post)

    # 2. Voronoi partition (should assign all seeds to the observation cell = last index)
    partitions = voronoi_partition_bayes(seed_pts, priors, observation, sigma=0.8)
    print("Voronoi partitions (seed index -> point indices):", {k: v for k, v in partitions.items() if k != "posterior"})

    # 3. RBF interpolation on a grid
    grid = [(x, y) for x in np.linspace(-1, 3, 5) for y in np.linspace(-1, 3, 5)]
    interpolated = bayes_weighted_rbf(
        query_points=grid,
        seed_points=seed_pts,
        priors=priors,
        observation=observation,
        sigma_likelihood=0.8,
        sigma_rbf=0.5,
    )
    print("Interpolated values on grid:", interpolated.reshape(5, 5))
    sys.exit(0)