# DARWIN HAMMER — match 3917, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_rbf_surrogate_hybrid_hybrid_endpoi_m423_s0.py (gen5)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m1708_s0.py (gen6)
# born: 2026-05-29T23:52:35Z

"""Hybrid RBF‑Voronoi‑Fisher Algorithm

Parents:
    * hybrid_hybrid_rbf_surrogate_hybrid_hybrid_endpoi_m423_s0.py
    * hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m1708_s0.py

Mathematical bridge:
    The RBF surrogate provides a continuous similarity estimate ϕ(x) for any
    feature vector x.  This scalar field is used as the “intensity” in the
    Fisher‑score formulation of the Voronoi‑partitioned space.  Concretely,
    each node is projected to a 2‑D point p, assigned to the nearest Voronoi
    seed s, and a Fisher score is computed from the distance d(p, s) while
    being weighted by the RBF prediction ϕ(x).  The resulting hybrid score

        ψ = ϕ(x) · Fisher(d(p,s); centre, width)

    fuses the two parent topologies into a single unified system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Mapping, Hashable, List, Dict, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Vector = Sequence[float]
Node = Hashable
FeatureVec = Sequence[float]
Point = Tuple[float, float]

# ----------------------------------------------------------------------
# Parent A – RBF surrogate utilities
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass(frozen=True)
class RBFSurrogate:
    """Thin wrapper around a set of RBF centres and corresponding weights."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """RBF interpolation at point x."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )


def rbf_predict_all(features: List[FeatureVec], surrogate: RBFSurrogate) -> List[float]:
    """Predict similarity for a collection of feature vectors."""
    return [surrogate.predict(f) for f in features]

# ----------------------------------------------------------------------
# Parent B – Voronoi partitioning & Fisher score utilities
# ----------------------------------------------------------------------
def distance(a: Point, b: Point) -> float:
    """2‑D Euclidean distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the seed closest to *point* (break ties by index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[int]]:
    """
    Voronoi assignment.

    Returns a mapping ``region_index -> list_of_point_indices``.
    """
    regions: Dict[int, List[int]] = {i: [] for i in range(len(seeds))}
    for idx, p in enumerate(points):
        region = nearest(p, seeds)
        regions[region].append(idx)
    return regions


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity used inside the Fisher formulation."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Classical Fisher information score for a Gaussian beam.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def region_fisher_scores(
    point_indices: List[int],
    points: List[Point],
    seed: Point,
    centre: float,
    width: float,
) -> List[float]:
    """
    Compute Fisher scores for every point in a Voronoi region.
    The angle ``theta`` is taken as the Euclidean distance to the seed.
    """
    scores: List[float] = []
    for idx in point_indices:
        d = distance(points[idx], seed)
        scores.append(fisher_score(d, centre, width))
    return scores

# ----------------------------------------------------------------------
# Hybrid core – marrying RBF predictions with Voronoi‑Fisher scoring
# ----------------------------------------------------------------------
def project_to_2d(features: List[FeatureVec]) -> List[Point]:
    """
    Very simple linear projection: use the first two components.
    If a feature vector has fewer than two dimensions, pad with zeros.
    """
    proj: List[Point] = []
    for f in features:
        x = f[0] if len(f) > 0 else 0.0
        y = f[1] if len(f) > 1 else 0.0
        proj.append((float(x), float(y)))
    return proj


def hybrid_algorithm(
    node_features: Dict[Node, FeatureVec],
    surrogate: RBFSurrogate,
    seeds_2d: List[Point],
    fisher_centre: float,
    fisher_width: float,
) -> Dict[int, List[Tuple[Node, float, float]]]:
    """
    Full hybrid pipeline.

    1. Compute RBF similarity ϕ(x) for each node.
    2. Project feature vectors to 2‑D points.
    3. Perform Voronoi partitioning of the projected points.
    4. For each region compute the Fisher score of every point.
    5. Return a mapping ``region -> [(node, φ, ψ)]`` where
       ψ = φ * FisherScore.
    """
    # Preserve deterministic order
    nodes = list(node_features.keys())
    features = [list(node_features[n]) for n in nodes]

    # 1. RBF predictions
    phi_vals = rbf_predict_all(features, surrogate)

    # 2. 2‑D projection
    points_2d = project_to_2d(features)

    # 3. Voronoi assignment
    assignment = assign(points_2d, seeds_2d)

    # 4. Fisher scores per region
    result: Dict[int, List[Tuple[Node, float, float]]] = {}
    for region_idx, point_idxs in assignment.items():
        seed = seeds_2d[region_idx]
        fisher_vals = region_fisher_scores(
            point_idxs, points_2d, seed, fisher_centre, fisher_width
        )
        combined: List[Tuple[Node, float, float]] = []
        for local_i, node_idx in enumerate(point_idxs):
            node = nodes[node_idx]
            phi = phi_vals[node_idx]
            fisher = fisher_vals[local_i]
            psi = phi * fisher
            combined.append((node, phi, psi))
        result[region_idx] = combined

    return result

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic graph of 8 nodes with random 5‑D features
    random.seed(42)
    np.random.seed(42)

    node_features: Dict[Node, FeatureVec] = {
        f"n{i}": np.random.rand(5).tolist() for i in range(8)
    }

    # Random RBF surrogate (3 centres, random weights)
    centres = [tuple(np.random.rand(5)) for _ in range(3)]
    weights = list(np.random.rand(3))
    surrogate = RBFSurrogate(centers=centres, weights=weights, epsilon=1.5)

    # Random Voronoi seeds in the same 2‑D projection space
    seeds_2d = [(random.random(), random.random()) for _ in range(3)]

    # Fisher parameters
    fisher_centre = 0.0   # centre at zero distance
    fisher_width = 0.2   # relatively narrow beam

    # Run the hybrid algorithm
    hybrid_result = hybrid_algorithm(
        node_features,
        surrogate,
        seeds_2d,
        fisher_centre,
        fisher_width,
    )

    # Simple sanity printout
    for region, items in hybrid_result.items():
        print(f"Region {region} (seed {seeds_2d[region]}):")
        for node, phi, psi in items:
            print(f"  Node {node}: φ={phi:.4f}, ψ={psi:.6f}")
    sys.exit(0)