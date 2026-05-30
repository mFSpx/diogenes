# DARWIN HAMMER — match 3021, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s6.py (gen3)
# parent_b: hybrid_hybrid_ternary_lens__hybrid_hybrid_model__m716_s2.py (gen3)
# born: 2026-05-29T23:47:26Z

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric utilities
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
# Multivector representation 
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
# Similarity matrix construction 
# ----------------------------------------------------------------------
def rbf_similarity(centroids: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian RBF similarity between all pairs of centroids.
    S_ij = exp( -||c_i - c_j||² / (2·σ²) )
    """
    diff = centroids[:, None, :] - centroids[None, :, :]          
    sqdist = np.sum(diff ** 2, axis=2)                           
    return np.exp(-sqdist / (2.0 * sigma ** 2))

def hash_similarity(scalars: np.ndarray) -> np.ndarray:
    """
    Simple deterministic similarity derived from scalar parts.
    We map each scalar to an integer hash and compute a similarity that decays
    with absolute hash distance.
    """
    prime = 2_147_483_647
    hashes = (scalars.astype(np.int64) * 31 + 7) % prime
    diff = np.abs(hashes[:, None] - hashes[None, :])
    return 1.0 / (1.0 + diff.astype(np.float64))

def combined_similarity(centroids: np.ndarray, scalars: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Element‑wise product of the RBF similarity and the hash‑based similarity.
    """
    return rbf_similarity(centroids, sigma) * hash_similarity(scalars)

# ----------------------------------------------------------------------
# Audit‑pruning side 
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
# Hybrid stochastic reassignment with KL divergence regularisation
# ----------------------------------------------------------------------
def stochastic_reassign_kl(
    points: List[Point],
    seeds: List[Point],
    similarity: np.ndarray,
    base_p: float = 0.2,
    kl_beta: float = 0.1,
    rng: random.Random | None = None,
) -> Dict[int, List[Point]]:
    """
    Perform one stochastic re‑assignment step with KL divergence regularisation:
    1. Compute prune probabilities P = base_p * similarity.
    2. For each point, decide whether it stays in its current Voronoi cell
       or jumps to a more similar neighbour, using P as transition likelihood.
    3. Regularise the transition probabilities using KL divergence.
    Returns a new region dictionary.
    """
    if rng is None:
        rng = random.Random()
    regions = assign(points, seeds)
    P = prune_probability_matrix(base_p, similarity)

    new_regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for pt in points:
        cur_idx = nearest(pt, seeds)
        move_probs = P[cur_idx].copy()
        move_probs[cur_idx] = 0.0 
        stay_prob = 1.0 - move_probs.sum() 
        probs = np.append(move_probs, stay_prob)

        # KL divergence regularisation
        kl_div = np.zeros_like(probs)
        for i in range(len(probs)):
            if probs[i] > 0:
                kl_div[i] = probs[i] * np.log(probs[i] / (1.0 / len(probs)))
        kl_penalty = kl_beta * np.sum(kl_div)

        # Annealed softmax
        probs = np.exp((probs + kl_penalty) / (1.0 + kl_beta))
        probs /= np.sum(probs)

        choices = list(range(len(seeds))) + [cur_idx]     
        chosen = rng.choices(choices, weights=probs, k=1)[0]
        if chosen == cur_idx:
            new_regions[cur_idx].append(pt)
        else:
            new_regions[chosen].append(pt)

    return new_regions

# ----------------------------------------------------------------------
# End‑to‑end hybrid pipeline 
# ----------------------------------------------------------------------
def build_hybrid_regions_kl(points: List[Point], seeds: List[Point], sigma: float = 1.0, kl_beta: float = 0.1) -> Tuple[Dict[int, List[Point]], np.ndarray]:
    """
    1. Deterministic Voronoi assignment.
    2. Compute multivector representation.
    3. Compute similarity matrix.
    4. Stochastic re‑assignment with KL divergence regularisation.

    Returns:
    * regions – stochastic Voronoi regions
    * similarity – similarity matrix
    """
    regions = assign(points, seeds)
    scalars = np.array([len(region) for region in regions.values()])
    centroids = np.array([np.mean(np.array(region), axis=0) for region in regions.values()])
    similarity = combined_similarity(centroids, scalars, sigma)

    regions_kl = stochastic_reassign_kl(points, seeds, similarity, kl_beta=kl_beta)
    return regions_kl, similarity