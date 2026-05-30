# DARWIN HAMMER — match 3055, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_perceptual_de_m828_s0.py (gen4)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_vorono_m1224_s5.py (gen5)
# born: 2026-05-29T23:47:32Z

"""Hybrid Pheromone‑Fisher‑Voronoi Algorithm
==========================================

This module fuses the two parent algorithms:

* **Parent A** – ``hybrid_hybrid_pherom_hybrid_perceptual_de_m828_s0.py``  
  Provides a pheromone system with decay, an expected‑entropy evaluation of
  candidate “surfaces”, a pruning probability and a sheaf‑cohomology matrix
  transform applied to feature vectors.  Perceptual hashes are used to group
  candidates and a lightweight RBF surrogate is fitted inside each group.

* **Parent B** – ``hybrid_hybrid_fisher_locali_hybrid_hybrid_vorono_m1224_s5.py``  
  Supplies a Gaussian‑beam Fisher‑information score, a lead‑lag transform for
  time‑series paths and a weighted Euclidean Voronoi partition where the
  weight of a Voronoi site is the inverse Fisher score.  A circuit‑breaker
  aborts the partition when too many points yield non‑finite scores.

**Mathematical bridge**

The bridge is the *weight* that governs the Voronoi partition.  
For a candidate surface *s* we compute


I_s   = fisher_information_score(s)                     # Parent B
H_s   = expected_entropy(pheromone_signal(s))          # Parent A
w_s   = I_s / (H_s + ε)                                 # hybrid weight


`w_s` is used as the inverse weight in the weighted Euclidean distance
`d_w(p,s)=‖p−s‖₂ / w_s`.  Thus high Fisher information (informative direction)
and low pheromone entropy (high confidence) both shrink the effective distance,
making the Voronoi region favour such sites.

The sheaf‑cohomology matrix `S` from Parent A is applied **before** the
lead‑lag transform, turning raw feature vectors `x` into `x' = S·x`.  The
resulting cloud is then clustered by the weighted Voronoi construction and
each cluster receives an RBF surrogate fitted on its members.

The three public functions below illustrate the hybrid workflow:
`hybrid_weighted_voronoi`, `update_pheromones_and_entropy`, and
`fit_cluster_rbfs`.
"""

import sys
import math
import random
import pathlib
import numpy as np
from typing import List, Tuple, Dict, Any, FrozenSet

# ----------------------------------------------------------------------
# Type aliases (shared with Parent B)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = FrozenSet[int]          # basis blade represented by a set of indices
Multivector = Dict[Blade, float]  # mapping blade → coefficient

# ----------------------------------------------------------------------
# Parent A – Pheromone system and sheaf matrix utilities
# ----------------------------------------------------------------------
class PheromoneSystem:
    """Tracks pheromone signals per surface and provides decay & entropy."""
    def __init__(self):
        # surface_key → dict(signal_kind, signal_value, half_life_seconds, created_time)
        self.pheromones: Dict[str, Dict[str, Any]] = {}

    def add_signal(self, surface_key: str, signal_kind: str,
                   signal_value: float, half_life_seconds: float,
                   now: Any) -> None:
        """Create or replace a pheromone entry."""
        self.pheromones[surface_key] = {
            'signal_kind': signal_kind,
            'signal_value': signal_value,
            'half_life_seconds': half_life_seconds,
            'created_time': now
        }

    def decay(self, now: Any) -> None:
        """Apply exponential decay to all stored signals."""
        for key, rec in self.pheromones.items():
            elapsed = (now - rec['created_time']).total_seconds()
            decay_factor = 0.5 ** (elapsed / rec['half_life_seconds'])
            rec['signal_value'] *= decay_factor
            rec['created_time'] = now

    def values(self) -> List[float]:
        """Return the current signal values for entropy calculation."""
        return [rec['signal_value'] for rec in self.pheromones.values()]

def expected_entropy(values: List[float]) -> float:
    """Shannon entropy of normalized pheromone strengths."""
    if not values:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log(p, 2) for p in probs)

def sheaf_transform(matrix: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Apply a sheaf‑cohomology matrix to a set of column vectors."""
    return matrix @ vectors

# ----------------------------------------------------------------------
# Parent B – Fisher information and Voronoi utilities
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile evaluated at angle ``theta``."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_information_score(point: np.ndarray,
                             center: float = 0.0,
                             width: float = 1.0) -> float:
    """
    Compute a scalar Fisher information score for a 2‑D point.
    Uses the polar angle of the first two coordinates.
    """
    if point.shape[0] < 2:
        raise ValueError('point must have at least two dimensions')
    x, y = point[0], point[1]
    theta = math.atan2(y, x)  # polar angle in [-π, π]
    return gaussian_beam(theta, center, width) + 1e-8  # avoid zero

def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Given a (T, d) array representing a time‑series path,
    produce a (2T‑1, 2d) lead‑lag embedding.
    """
    if path.ndim != 2:
        raise ValueError('path must be 2‑dimensional (time × dim)')
    T, d = path.shape
    lead = np.repeat(path, 2, axis=0)[:2 * T - 1]
    lag = np.concatenate([np.zeros((1, d)), path])[:2 * T - 1]
    return np.hstack([lead, lag])

def weighted_euclidean(p: np.ndarray, s: np.ndarray, weight: float) -> float:
    """Weighted Euclidean distance d_w(p,s)=‖p−s‖₂ / weight."""
    if weight == 0:
        return float('inf')
    return np.linalg.norm(p - s) / weight

def circuit_breaker(non_finite_count: int, threshold: int = 10) -> bool:
    """Return True if the partition should abort."""
    return non_finite_count > threshold

# ----------------------------------------------------------------------
# Hybrid core – combine pheromone entropy with Fisher information
# ----------------------------------------------------------------------
def compute_hybrid_weights(sites: np.ndarray,
                           pheromone_sys: PheromoneSystem,
                           now: Any,
                           epsilon: float = 1e-6) -> np.ndarray:
    """
    For each Voronoi site compute the hybrid weight w_s = I_s / (H_s + ε).

    Parameters
    ----------
    sites : (N, d) array of site coordinates.
    pheromone_sys : PheromoneSystem instance (provides entropy).
    now : datetime‑like current time (used for decay).
    epsilon : small constant to avoid division by zero.

    Returns
    -------
    weights : (N,) array of positive floats.
    """
    # Update decay before using values
    pheromone_sys.decay(now)

    # Global entropy based on all pheromone values
    H_global = expected_entropy(pheromone_sys.values())

    # Compute Fisher information per site
    I_vals = np.array([fisher_information_score(site) for site in sites])

    # Hybrid weight: high Fisher & low entropy → larger weight
    weights = I_vals / (H_global + epsilon)
    return weights

def hybrid_weighted_voronoi(points: np.ndarray,
                           initial_sites: np.ndarray,
                           pheromone_sys: PheromoneSystem,
                           now: Any,
                           max_iters: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform an iterative weighted Voronoi partition where the weight of each
    site is the hybrid quantity w_s = I_s / (H_s + ε).

    Returns
    -------
    assignments : (M,) integer array assigning each point to a site index.
    final_sites : (K, d) array of site coordinates after iteration.
    """
    sites = initial_sites.copy()
    for _ in range(max_iters):
        weights = compute_hybrid_weights(sites, pheromone_sys, now)

        # Assign each point to the nearest site under weighted distance
        assignments = np.empty(points.shape[0], dtype=int)
        non_finite = 0
        for i, p in enumerate(points):
            dists = np.array([weighted_euclidean(p, s, w) for s, w in zip(sites, weights)])
            if np.all(~np.isfinite(dists)):
                non_finite += 1
                assignments[i] = -1
            else:
                assignments[i] = int(np.argmin(dists))

        if circuit_breaker(non_finite):
            break

        # Re‑compute sites as centroids of assigned points
        new_sites = []
        for k in range(sites.shape[0]):
            mask = assignments == k
            if np.any(mask):
                new_sites.append(points[mask].mean(axis=0))
            else:
                # If a site lost all points, keep its old location
                new_sites.append(sites[k])
        sites = np.vstack(new_sites)

    return assignments, sites

def fit_cluster_rbfs(points: np.ndarray,
                    assignments: np.ndarray,
                    gamma: float = 1.0) -> List[Dict[str, Any]]:
    """
    Fit a simple Radial‑Basis‑Function surrogate for each Voronoi cluster.
    The surrogate is stored as a dict with 'center', 'coeff', and 'gamma'.

    Returns a list of surrogates (one per unique cluster index).
    """
    surrogates = []
    unique_labels = np.unique(assignments)
    for label in unique_labels:
        if label == -1:
            continue  # skip unassigned points
        mask = assignments == label
        cluster_pts = points[mask]
        if cluster_pts.shape[0] == 0:
            continue
        center = cluster_pts.mean(axis=0)

        # Coefficients are simply the distance from the center (placeholder)
        dists = np.linalg.norm(cluster_pts - center, axis=1)
        coeff = np.exp(-gamma * dists ** 2).mean()

        surrogates.append({
            'label': int(label),
            'center': center,
            'coeff': float(coeff),
            'gamma': float(gamma)
        })
    return surrogates

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import datetime
    from datetime import timezone

    # Synthetic data generation
    rng = np.random.default_rng(42)
    num_points = 200
    dim = 3

    # Random point cloud
    points = rng.normal(size=(num_points, dim))

    # Random initial Voronoi sites (5 sites)
    init_sites = rng.normal(size=(5, dim))

    # Initialise pheromone system with dummy signals
    pheromone = PheromoneSystem()
    now = datetime.datetime.now(tz=timezone.utc)
    for i in range(5):
        key = f"site_{i}"
        pheromone.add_signal(
            surface_key=key,
            signal_kind="candidate",
            signal_value=rng.random(),
            half_life_seconds=30.0,
            now=now
        )

    # Apply sheaf matrix (random invertible matrix)
    while True:
        S = rng.normal(size=(dim, dim))
        if np.linalg.cond(S) < 1e3:
            break
    transformed_points = sheaf_transform(S, points.T).T
    transformed_sites = sheaf_transform(S, init_sites.T).T

    # Lead‑lag transform on the point cloud (treat each point as a length‑1 path)
    # For demonstration we just reshape to (T, d) where T=num_points
    lead_lag_pts = lead_lag_transform(transformed_points)

    # Hybrid weighted Voronoi partition
    assignments, final_sites = hybrid_weighted_voronoi(
        points=lead_lag_pts,
        initial_sites=transformed_sites,
        pheromone_sys=pheromone,
        now=now,
        max_iters=3
    )

    # Fit RBF surrogates per cluster
    surrogates = fit_cluster_rbfs(lead_lag_pts, assignments)

    # Simple sanity prints
    print(f"Number of clusters formed: {len(np.unique(assignments)) - (1 if -1 in assignments else 0)}")
    print(f"RBF surrogates fitted: {len(surrogates)}")
    for s in surrogates[:3]:
        print(f"Cluster {s['label']} – center norm {np.linalg.norm(s['center']):.3f}, coeff {s['coeff']:.3f}")

    # Ensure no unhandled exceptions – end of smoke test.