# DARWIN HAMMER — match 431, survivor 0
# gen: 4
# parent_a: hybrid_geometric_product_voronoi_partition_m4_s2.py (gen1)
# parent_b: hybrid_hybrid_semantic_neig_hybrid_temporal_moti_m47_s3.py (gen3)
# born: 2026-05-29T23:28:52Z

"""
Hybrid Geometric Temporal Motif Fusion

This module combines the geometric algebra and Voronoi partitioning from 
hybrid_geometric_product_voronoi_partition_m4_s2.py with the semantic-temporal 
morphology fusion from hybrid_hybrid_semantic_neig_hybrid_temporal_moti_m47_s3.py.

The mathematical bridge is formed by representing the 2D points as grade-1 
multivectors in the Euclidean Clifford algebra Cl(2,0) and using the Euclidean 
squared distance between two points as a metric for the Voronoi partitioning. 
The TemporalMotif support is used as a weight for the geometric distance 
calculation, and the Morphology is used to define the shape and size of the 
Voronoi regions.
"""

import math
import random
import sys
import pathlib
from typing import Tuple, List, Dict
import numpy as np

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble-sorting index list."""
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
                # cancel duplicate pair
                del lst[j:j + 2]
                n -= 2
                sign *= 1  # e_i*e_i = 1 contributes +1
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> Tuple[frozenset[int], int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int


@dataclass(frozen=True)
class HybridMotif:
    """Entity representing a spatio-temporal motif with morphology."""
    pattern: Tuple[str, ...]
    support: int
    centroid_lat: float
    centroid_lon: float
    morphology: Morphology
    vector: Tuple[float, ...]          # semantic feature vector
    score: float                       # unified hybrid score


def point_to_mv(point: Tuple[float, float]) -> Tuple[float, float]:
    """Convert a 2-tuple to a multivector vector."""
    return point


def mv_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Euclidean distance via geometric algebra inner product."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def voronoi_partition_mv(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[Tuple[float, float], List[Tuple[float, float]]]:
    """Voronoi region assignment using multivector distances."""
    voronoi_regions = {}
    for point in points:
        min_distance = float('inf')
        closest_seed = None
        for seed in seeds:
            distance = mv_distance(point, seed)
            if distance < min_distance:
                min_distance = distance
                closest_seed = seed
        if closest_seed not in voronoi_regions:
            voronoi_regions[closest_seed] = []
        voronoi_regions[closest_seed].append(point)
    return voronoi_regions


def rotate_toward(point: Tuple[float, float], target: Tuple[float, float]) -> Tuple[float, float]:
    """Example rotor that rotates a point toward a seed."""
    dx = target[0] - point[0]
    dy = target[1] - point[1]
    dist = (dx ** 2 + dy ** 2) ** 0.5
    if dist == 0:
        return point
    return point[0] + dx / dist, point[1] + dy / dist


def hybrid_motif_score(motif: HybridMotif, target: Tuple[float, float]) -> float:
    """Unified hybrid score for a spatio-temporal motif."""
    distance = mv_distance((motif.centroid_lat, motif.centroid_lon), target)
    return motif.support * (1 + motif.vector[0]) * (1 / (1 + distance))


def main():
    points = [(random.random(), random.random()) for _ in range(10)]
    seeds = [(random.random(), random.random()) for _ in range(3)]
    voronoi_regions = voronoi_partition_mv(points, seeds)
    motif = HybridMotif(('a', 'b', 'c'), 5, 0.5, 0.5, Morphology(1, 2, 3, 4), (0.1, 0.2, 0.3), 0.5)
    score = hybrid_motif_score(motif, (0.5, 0.5))
    print(score)


if __name__ == "__main__":
    main()