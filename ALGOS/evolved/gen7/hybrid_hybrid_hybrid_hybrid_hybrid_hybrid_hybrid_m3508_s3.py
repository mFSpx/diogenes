# DARWIN HAMMER — match 3508, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1207_s5.py (gen6)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_krampus_brain_m340_s0.py (gen4)
# born: 2026-05-29T23:50:26Z

"""Hybrid Algorithm combining:
- Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1207_s5.py
- Parent B: hybrid_hybrid_hybrid_vorono_hybrid_krampus_brain_m340_s0.py

Mathematical Bridge:
Parent A provides Euclidean edge lengths and Bayesian regret weighting.
Parent B supplies an *operator‑geometric ratio* derived from endpoint morphology
and extracted features.  The bridge is the modulation of every Euclidean
distance (and consequently every edge weight) by the factor (1 + operator_ratio).
Thus edge costs become:

    w_ij = ‖p_i − p_j‖₂ · (1 + operator_ratio(morph_i, features))

Voronoi regions are built using these modulated distances, and within each
region the Bayesian regret (bayes_marginal) is applied.  The resulting system
fuses pruning, curvature‑like modulation, and region‑wise Bayesian updates
into a single coherent pipeline.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import numpy as np
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[int, int]  # indices into a point list


@dataclass(frozen=True)
class Morphology:
    """Simple geometric description of an endpoint."""
    length: float
    width: float
    height: float


# ----------------------------------------------------------------------
# Parent‑A utilities (pruning, Euclidean geometry, Bayesian marginal)
# ----------------------------------------------------------------------
def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Probability that an edge is *removed* after time t."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non‑negative')
    return min(1.0, lam * math.exp(-alpha * t))


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Bayesian marginal used as a regret‑weighted update."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive


# ----------------------------------------------------------------------
# Parent‑B utilities (operator‑geometric ratio, morphology‑aware distance)
# ----------------------------------------------------------------------
def operator_geometric_ratio(morph: Morphology, features: Dict[str, float]) -> float:
    """
    Compute the ratio that couples morphology with extracted features.
    The denominator expects two feature keys; a tiny epsilon avoids division by zero.
    """
    denom = (features.get("operator_visceral_ratio", 0.0) +
             features.get("operator_tech_ratio", 0.0) + 1e-9)
    return (morph.length + morph.width + morph.height) / denom


def morph_distance(a: Point, b: Point, morph_a: Morphology,
                   morph_b: Morphology, features: Dict[str, float]) -> float:
    """
    Distance between points a and b modulated by the operator‑geometric ratios
    of both endpoints.
    """
    base = length(a, b)
    ratio_a = operator_geometric_ratio(morph_a, features)
    ratio_b = operator_geometric_ratio(morph_b, features)
    # Symmetric modulation: average of the two ratios
    return base * (1.0 + 0.5 * (ratio_a + ratio_b))


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def weighted_edge_lengths(points: List[Point],
                          edges: List[Edge],
                          morphs: List[Morphology],
                          features: Dict[str, float]) -> List[float]:
    """
    Compute edge weights using the hybrid distance metric.
    """
    weights = []
    for i, j in edges:
        w = morph_distance(points[i], points[j], morphs[i], morphs[j], features)
        weights.append(w)
    return weights


def prune_edges_hybrid(points: List[Point],
                       edges: List[Edge],
                       t: float,
                       lam: float = 1.0,
                       alpha: float = 0.2,
                       morphs: List[Morphology] = None,
                       features: Dict[str, float] = None,
                       seed: int | str | None = None) -> List[Edge]:
    """
    Prune edges based on a time‑dependent probability.  The probability
    is applied *after* the hybrid weighting, i.e. longer (more costly) edges
    are more likely to survive because they have larger weights that are
    *not* used in the probability calculation – we keep the original
    prune_probability semantics but expose the hybrid weights for downstream
    use.
    """
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    # Decide per edge independently
    kept = [e for e in edges if rng.random() >= p]
    return kept


def voronoi_partition(points: List[Point],
                      seeds: List[Point],
                      morphs: List[Morphology],
                      features: Dict[str, float]) -> List[int]:
    """
    Assign each point to the index of its nearest seed using the hybrid distance.
    Returns a list `region[i] = seed_index`.
    """
    if not seeds:
        raise ValueError("At least one seed required for Voronoi partitioning")
    # Pre‑compute morphology for seeds (if fewer seeds than points, reuse list)
    seed_morphs = [morphs[i] if i < len(morphs) else morphs[-1] for i in range(len(seeds))]
    regions = []
    for idx, pt in enumerate(points):
        dists = [morph_distance(pt, s, morphs[idx], seed_morphs[s_idx], features)
                 for s_idx, s in enumerate(seeds)]
        nearest = int(np.argmin(dists))
        regions.append(nearest)
    return regions


def regional_bayesian_update(points: List[Point],
                             regions: List[int],
                             prior: float,
                             likelihood: float,
                             false_positive: float) -> Dict[int, float]:
    """
    For each Voronoi region compute a Bayesian marginal based on the number
    of points it contains.  The result is a mapping `region_id -> updated probability`.
    """
    region_counts: Dict[int, int] = {}
    for r in regions:
        region_counts[r] = region_counts.get(r, 0) + 1

    updates: Dict[int, float] = {}
    for r, count in region_counts.items():
        # Simple heuristic: treat count as a scaling factor for the likelihood
        scaled_likelihood = min(1.0, likelihood * (count / max(1, len(points))))
        updates[r] = bayes_marginal(prior, scaled_likelihood, false_positive)
    return updates


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic dataset
    pts: List[Point] = [(0.0, 0.0), (1.0, 0.5), (2.0, 1.0), (3.0, 0.0), (0.5, 2.0)]
    # Create a fully connected graph (undirected) for demonstration
    edges: List[Edge] = [(i, j) for i in range(len(pts)) for j in range(i + 1, len(pts))]

    # Morphology for each point (random but deterministic for test)
    morphs = [
        Morphology(length=1.0, width=0.5, height=0.2),
        Morphology(length=0.9, width=0.4, height=0.3),
        Morphology(length=1.2, width=0.6, height=0.25),
        Morphology(length=1.1, width=0.5, height=0.22),
        Morphology(length=0.8, width=0.45, height=0.18),
    ]

    # Feature dictionary required by operator_geometric_ratio
    features = {
        "operator_visceral_ratio": 0.7,
        "operator_tech_ratio": 0.3,
    }

    # 1. Compute hybrid edge weights
    weights = weighted_edge_lengths(pts, edges, morphs, features)
    print("Hybrid edge weights (first 5):", weights[:5])

    # 2. Prune edges with a synthetic time parameter
    kept_edges = prune_edges_hybrid(pts, edges, t=2.0, seed=42,
                                    morphs=morphs, features=features)
    print("Edges kept after pruning:", kept_edges)

    # 3. Voronoi partition using a subset of points as seeds
    seed_pts = [pts[0], pts[2], pts[4]]
    regions = voronoi_partition(pts, seed_pts, morphs, features)
    print("Voronoi region assignment:", regions)

    # 4. Regional Bayesian update
    updates = regional_bayesian_update(pts, regions,
                                       prior=0.4,
                                       likelihood=0.6,
                                       false_positive=0.1)
    print("Regional Bayesian updates:", updates)

    # Verify that the script runs without raising exceptions
    sys.exit(0)