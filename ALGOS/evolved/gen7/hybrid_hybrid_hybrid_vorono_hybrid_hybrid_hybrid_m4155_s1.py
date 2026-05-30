# DARWIN HAMMER — match 4155, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s5.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m1388_s6.py (gen6)
# born: 2026-05-29T23:53:45Z

"""
Hybrid algorithm combining Voronoi partitioning (Parent A) with binary hyperdimensional routing
(Parent B) and resource-aware model selection.

The mathematical bridge:
- Parent A provides high-dimensional Voronoi diagrams and a Euclidean distance metric.
- Parent B defines a low-dimensional model resource vector and evaluates compatibility via a bilinear form.
The fusion maps the Voronoi diagram onto the model-space using the Euclidean distance metric, then scores each model with the bilinear form.

This implementation integrates the Voronoi partitioning from Parent A with the binary hyperdimensional routing
and resource-aware model selection from Parent B.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Core Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = FrozenSet[int]          # basis blade represented by a set of indices
Multivector = Dict[Blade, float]  # mapping blade → coefficient
Vector = List[int]

# ----------------------------------------------------------------------
# Parent A – Voronoi helpers
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_voronoi_regions(points: List[Point],
                            sites: List[Point]) -> Dict[int, List[Point]]:
    """
    Assign each point to the index of the nearest site.
    Returns a dict ``site_index → list[points]``.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions

# ----------------------------------------------------------------------
# Parent B – hyperdimensional primitives
# ----------------------------------------------------------------------
def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: List[Vector]) -> Vector:
    if not vectors:
        raise ValueError("bundle requires at least one vector")
    dim = len(vectors[0])
    for v in vectors:
        if len(v) != dim:
            raise ValueError("all vectors must have same dimension")
    summed = [sum(comp) for comp in zip(*vectors)]
    return [1 if s >= 0 else -1 for s in summed]

def similarity(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in b)))

# ----------------------------------------------------------------------
# Hybrid Algorithm
# ----------------------------------------------------------------------
def project_voronoi_to_model_space(points: List[Point],
                                  sites: List[Point],
                                  model_resources: List[Tuple[float, float]]) -> List[float]:
    """
    Project the Voronoi diagram onto the model-space using the Euclidean distance metric.
    """
    regions = compute_voronoi_regions(points, sites)
    projected = []
    for region in regions.values():
        model_distances = [euclidean_distance(region[0], resource) for resource in model_resources]
        nearest = int(np.argmin(model_distances))
        projected.append(model_distances[nearest])
    return projected

def score_models(model_resources: List[Tuple[float, float]],
                 model_vectors: List[Vector]) -> List[float]:
    """
    Score each model with the bilinear form.
    """
    projected = project_voronoi_to_model_space([p for r in model_resources for p in r], [p for r in model_resources for p in r], model_resources)
    scores = []
    for i, vector in enumerate(model_vectors):
        score = 0
        for j, v in enumerate(model_vectors):
            score += similarity(vector, v) * projected[j]
        scores.append(score)
    return scores

def select_best_model(scores: List[float],
                      model_resources: List[Tuple[float, float]]) -> int:
    """
    Select the model with the highest score.
    """
    return int(np.argmax(scores))

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    points = [(0, 0), (1, 0), (0, 1)]
    sites = [(0, 0), (1, 1), (1, 0)]
    model_resources = [(0.5, 0.5), (0.2, 0.8), (0.7, 0.3)]
    model_vectors = [random_vector(10000) for _ in range(len(model_resources))]
    scores = score_models(model_resources, model_vectors)
    best_model = select_best_model(scores, model_resources)
    print("Best model:", best_model)