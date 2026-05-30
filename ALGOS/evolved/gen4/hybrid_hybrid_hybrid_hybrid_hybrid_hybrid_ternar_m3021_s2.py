# DARWIN HAMMER — match 3021, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s6.py (gen3)
# parent_b: hybrid_hybrid_ternary_lens__hybrid_hybrid_model__m716_s2.py (gen3)
# born: 2026-05-29T23:47:26Z

"""Hybrid Voronoi‑Geometric‑Algebra / RBF‑Similarity & Audit‑Pruning Algorithm
================================================================================

Parents
-------
* **Parent A** – `hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s6.py`  
  Builds a Voronoi partition, represents each cell as a (scalar, centroid) multivector,
  and creates a similarity matrix **S** by multiplying a Gaussian RBF of Euclidean
  centroid distances with a hash‑derived similarity of the scalar parts.

* **Parent B** – `hybrid_hybrid_ternary_lens__hybrid_hybrid_model__m716_s2.py`  
  From an audit manifest it derives a classification count vector **c**, normalises it
  to a weight vector **w** (interpreted as a curvature field), and forms a prune‑
  probability matrix **P** = p(t)·w, where p(t) is a scalar schedule.

Mathematical Bridge
-------------------
Both parents rely on a *weight* matrix that modulates a scalar base probability.
We therefore fuse them by letting the similarity matrix **S** from Parent A play the
role of the curvature field **w** in Parent B.  Concretely


w_ij = S_ij                         # similarity between seed i and seed j
p(t)  = base_prune                  # user‑defined scalar schedule
P_ij  = p(t) * w_ij                 # prune probability for point‑seed pair


During stochastic re‑assignment a point is kept in its current Voronoi cell with
probability 1‑P_ij and may jump to a more similar neighbour with probability P_ij.
The result is a *similarity‑aware, audit‑pruned* Voronoi partition.

The implementation below follows this bridge, exposing three core functions that
realise the hybrid pipeline.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric utilities (from Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the nearest seed (ties broken by lower index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Deterministic Voronoi assignment of points to the nearest seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for pt in points:
        idx = nearest(pt, seeds)
        regions[idx].append(pt)
    return regions

# ----------------------------------------------------------------------
# Multivector representation (scalar = region size, grade‑1 = centroid)
# ----------------------------------------------------------------------
def multivector_from_region(region: List[Point]) -> Tuple[float, np.ndarray]:
    """
    Returns a tuple (scalar, centroid) for a Voronoi region.
    * scalar   – number of points in the region (size)
    * centroid – 2‑D numpy array of the region centroid (grade‑1 part)
    """
    scalar = float(len(region))
    if scalar == 0:
        centroid = np.zeros(2)
    else:
        pts = np.array(region)
        centroid = pts.mean(axis=0)
    return scalar, centroid

# ----------------------------------------------------------------------
# Similarity matrix construction (Parent A core)
# ----------------------------------------------------------------------
def rbf_similarity(centroids: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian RBF similarity between all pairs of centroids.
    S_ij = exp( -||c_i - c_j||² / (2·σ²) )
    """
    diff = centroids[:, None, :] - centroids[None, :, :]          # (n, n, 2)
    sqdist = np.sum(diff ** 2, axis=2)                           # (n, n)
    return np.exp(-sqdist / (2.0 * sigma ** 2))

def hash_similarity(scalars: np.ndarray) -> np.ndarray:
    """
    Simple deterministic similarity derived from scalar parts.
    We map each scalar to an integer hash and compute a similarity that decays
    with absolute hash distance.
    """
    # deterministic integer hash (mod large prime)
    prime = 2_147_483_647
    hashes = (scalars.astype(np.int64) * 31 + 7) % prime
    diff = np.abs(hashes[:, None] - hashes[None, :])
    return 1.0 / (1.0 + diff.astype(np.float64))

def combined_similarity(centroids: np.ndarray, scalars: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Element‑wise product of the RBF similarity and the hash‑based similarity.
    This mirrors Parent A's fusion of geometric and perceptual information.
    """
    return rbf_similarity(centroids, sigma) * hash_similarity(scalars)

# ----------------------------------------------------------------------
# Audit‑pruning side (Parent B core)
# ----------------------------------------------------------------------
def classification_weights(class_counts: Dict[str, int]) -> np.ndarray:
    """
    Convert a classification count dictionary into a normalised weight vector.
    The vector length equals the number of distinct classes; weights sum to 1.
    """
    total = sum(class_counts.values())
    if total == 0:
        raise ValueError("At least one classification count must be non‑zero")
    weights = np.array([class_counts.get(cls, 0) for cls in sorted(class_counts)], dtype=np.float64)
    return weights / total

def prune_probability_matrix(base_p: float, similarity: np.ndarray) -> np.ndarray:
    """
    Build the prune probability matrix P_ij = base_p * similarity_ij.
    The matrix is symmetric and bounded in [0, base_p].
    """
    if not (0.0 <= base_p <= 1.0):
        raise ValueError("base_p must be in [0, 1]")
    return base_p * similarity

# ----------------------------------------------------------------------
# Hybrid stochastic reassignment
# ----------------------------------------------------------------------
def stochastic_reassign(
    points: List[Point],
    seeds: List[Point],
    similarity: np.ndarray,
    base_p: float = 0.2,
    rng: random.Random | None = None,
) -> Dict[int, List[Point]]:
    """
    Perform one stochastic re‑assignment step:
    1. Compute prune probabilities P = base_p * similarity.
    2. For each point, decide whether it stays in its current Voronoi cell
       or jumps to a more similar neighbour, using P as transition likelihood.
    Returns a new region dictionary.
    """
    if rng is None:
        rng = random.Random()
    # Current deterministic assignment
    regions = assign(points, seeds)
    # Pre‑compute prune matrix
    P = prune_probability_matrix(base_p, similarity)

    # Prepare new empty regions
    new_regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}

    for pt in points:
        cur_idx = nearest(pt, seeds)
        # Gather probabilities of moving to any other seed
        move_probs = P[cur_idx].copy()
        move_probs[cur_idx] = 0.0                         # cannot move to self via prune
        stay_prob = 1.0 - move_probs.sum()                # probability to stay
        # Normalise to a proper distribution
        probs = np.append(move_probs, stay_prob)
        choices = list(range(len(seeds))) + [cur_idx]     # last entry = stay
        chosen = rng.choices(choices, weights=probs, k=1)[0]
        if chosen == cur_idx:
            new_regions[cur_idx].append(pt)
        else:
            new_regions[chosen].append(pt)

    return new_regions

# ----------------------------------------------------------------------
# End‑to‑end hybrid pipeline (three public functions)
# ----------------------------------------------------------------------
def build_hybrid_regions(points: List[Point], seeds: List[Point], sigma: float = 1.0) -> Tuple[Dict[int, List[Point]], np.ndarray]:
    """
    1. Deterministic Voronoi assignment.
    2. Multivector extraction.
    3. Combined similarity matrix.
    Returns the region dict and the similarity matrix.
    """
    regions = assign(points, seeds)
    scalars = []
    centroids = []
    for i in range(len(seeds)):
        s, c = multivector_from_region(regions[i])
        scalars.append(s)
        centroids.append(c)
    scalars_arr = np.array(scalars, dtype=np.float64)
    centroids_arr = np.vstack(centroids)                # shape (n, 2)
    sim = combined_similarity(centroids_arr, scalars_arr, sigma)
    return regions, sim

def hybrid_prune_and_reassign(
    points: List[Point],
    seeds: List[Point],
    class_counts: Dict[str, int],
    base_p: float = 0.2,
    sigma: float = 1.0,
    rng: random.Random | None = None,
) -> Dict[int, List[Point]]:
    """
    Full hybrid operation:
    * Build similarity from Voronoi geometry.
    * Convert audit classification counts to a weight vector w.
    * Modulate the base prune probability by w (broadcast over rows).
    * Perform stochastic reassignment.
    """
    _, sim_geo = build_hybrid_regions(points, seeds, sigma)

    # Classification weights act as a global curvature factor.
    w = classification_weights(class_counts)          # shape (k,)
    # Broadcast w over the similarity matrix: each seed gets a curvature factor.
    # For simplicity we average w to a single scalar curvature.
    curvature = w.mean()
    sim = sim_geo * curvature                           # incorporate audit curvature

    # Stochastic reassignment using the curvature‑modulated similarity.
    return stochastic_reassign(points, seeds, sim, base_p, rng)

def evaluate_hybrid(points: List[Point], seeds: List[Point], class_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Utility that runs the hybrid pipeline once and returns simple statistics:
    * average region size,
    * entropy of region size distribution,
    * overall prune probability (mean of the matrix).
    """
    regions, sim = build_hybrid_regions(points, seeds)
    sizes = np.array([len(regions[i]) for i in range(len(seeds))], dtype=np.float64)
    avg_size = sizes.mean()
    prob_matrix = prune_probability_matrix(0.2, sim)
    mean_prune = prob_matrix.mean()
    # Entropy of region size distribution (normalized)
    prob_sizes = sizes / sizes.sum() if sizes.sum() > 0 else np.zeros_like(sizes)
    entropy = -np.sum(prob_sizes * np.log2(prob_sizes + 1e-12))
    return {
        "avg_region_size": float(avg_size),
        "size_entropy": float(entropy),
        "mean_prune_probability": float(mean_prune),
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic data
    rng = random.Random(42)
    np.random.seed(0)

    num_points = 500
    num_seeds = 8
    points = [(float(rng.random() * 10), float(rng.random() * 10)) for _ in range(num_points)]
    seeds = [(float(rng.random() * 10), float(rng.random() * 10)) for _ in range(num_seeds)]

    # Fake audit classification counts
    class_counts = {
        "usable_now": 120,
        "research_only": 80,
        "needs_conversion": 50,
        "unsafe_for_fastpath": 30,
        "unsupported": 20,
    }

    # Run the hybrid pipeline
    final_regions = hybrid_prune_and_reassign(points, seeds, class_counts, base_p=0.15, sigma=2.0, rng=rng)

    stats = evaluate_hybrid(points, seeds, class_counts)

    print("Hybrid evaluation statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    total_assigned = sum(len(v) for v in final_regions.values())
    print(f"Total points after stochastic reassignment: {total_assigned} (expected {num_points})")
    sys.exit(0)