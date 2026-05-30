# DARWIN HAMMER — match 431, survivor 1
# gen: 4
# parent_a: hybrid_geometric_product_voronoi_partition_m4_s2.py (gen1)
# parent_b: hybrid_hybrid_semantic_neig_hybrid_temporal_moti_m47_s3.py (gen3)
# born: 2026-05-29T23:28:52Z

"""Hybrid module combining geometric algebra (geometric_product.py) and
Voronoi partitioning (voronoi_partition.py) with the spatio-temporal
morphology fusion (hybrid_hybrid_semantic_neig_hybrid_temporal_moti_m47_s3.py).

Mathematical bridge:
- A 3‑D point (x, y, z) is represented as a grade‑1 multivector
  **p = x·e₁ + y·e₂ + z·e₃** in the Euclidean Clifford algebra Cl(3,0).
- The Euclidean squared distance between two points a and b is the scalar
  part of the inner product ⟨a‑b, a‑b⟩, i.e. `(a - b).inner(b - a).scalar_part()`.
- The morphology length, width, height, and mass are used to compute a
  morphological weight similar to the recovery priority R(m) from parent A.
- The temporal motif pattern and support are used to compute a temporal
  weight similar to the temporal support s(p) from parent B.
- The cosine similarity between the entity’s feature vector and its
  semantic neighbours supplies a spatial proximity factor σ(i,j) that
  replaces the binary matrix D(i,j) used by the possum filter.
- The unified score for a candidate spatio-temporal motif is

    S = s(p) · (1 + z_s) · R(m) · σ

where  z_s is the z-score of the support distribution across all motifs.
"""

import math
import random
import sys
import pathlib
import numpy as np
from typing import Tuple, List, Dict

# ---------------------------------------------------------------------------
# Geometric algebra core (from hybrid_geometric_product_voronoi_partition_m4_s2.py)
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


def mv_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance via geometric algebra inner product."""
    return (a - b).inner(b - a).scalar_part()


def voronoi_partition_mv(points: List[np.ndarray], seeds: List[np.ndarray]) -> Dict[int, List[int]]:
    """Voronoi region assignment using multivector distances."""
    voronoi_regions = {}
    for i, point in enumerate(points):
        min_distance = float('inf')
        closest_seed_index = None
        for j, seed in enumerate(seeds):
            distance = mv_distance(point, seed)
            if distance < min_distance:
                min_distance = distance
                closest_seed_index = j
        if closest_seed_index not in voronoi_regions:
            voronoi_regions[closest_seed_index] = []
        voronoi_regions[closest_seed_index].append(i)
    return voronoi_regions


# ---------------------------------------------------------------------------
# Morphology and temporal motif utilities (from hybrid_hybrid_semantic_neig_hybrid_temporal_moti_m47_s3.py)
# ---------------------------------------------------------------------------

def flatness_index(morphology: Morphology) -> float:
    if min(morphology.length, morphology.width, morphology.height) <= 0:
        raise ValueError("dimensions must be positive")
    return (morphology.length * morphology.width * morphology.height) ** (1/3)


def temporal_weight(motif: TemporalMotif, z_s: float) -> float:
    return motif.support / (1 + z_s)


def spatial_proximity_factor(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))


# ---------------------------------------------------------------------------
# Hybrid spatio-temporal morphology fusion
# ---------------------------------------------------------------------------

def hybrid_score(entity: HybridMotif, z_s: float) -> float:
    return temporal_weight(entity.support, z_s) * flatness_index(entity.morphology) * spatial_proximity_factor(entity.vector, entity.vector)


def hybrid_partition(points: List[np.ndarray], seeds: List[np.ndarray], morphologies: List[Morphology], temporal_motifs: List[TemporalMotif]) -> Dict[int, List[int]]:
    voronoi_regions = voronoi_partition_mv(points, seeds)
    for region_index in voronoi_regions:
        for point_index in voronoi_regions[region_index]:
            point = points[point_index]
            morphology = morphologies[region_index]
            temporal_motif = temporal_motifs[region_index]
            voronoi_regions[region_index].append((temporal_weight(temporal_motif, np.mean([motif.support for motif in temporal_motifs])) * flatness_index(morphology) * spatial_proximity_factor(point, seeds[region_index]), point_index))
    return voronoi_regions


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

def example_1():
    # Create some example points and seeds
    points = np.array([[1, 2], [3, 4], [5, 6]])
    seeds = np.array([[2, 2], [4, 4]])

    # Create some example morphologies and temporal motifs
    morphologies = [Morphology(1, 2, 3, 4), Morphology(2, 3, 4, 5)]
    temporal_motifs = [TemporalMotif(('a', 'b', 'c'), 10), TemporalMotif(('d', 'e', 'f'), 20)]

    # Perform the hybrid partitioning
    voronoi_regions = hybrid_partition(points, seeds, morphologies, temporal_motifs)

    # Print the results
    for region_index, region in voronoi_regions.items():
        print(f"Region {region_index}: {region}")


def example_2():
    # Create some example points and seeds
    points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    seeds = np.array([[2, 2, 2], [4, 4, 4]])

    # Create some example morphologies and temporal motifs
    morphologies = [Morphology(1, 2, 3, 4), Morphology(2, 3, 4, 5)]
    temporal_motifs = [TemporalMotif(('a', 'b', 'c'), 10), TemporalMotif(('d', 'e', 'f'), 20)]

    # Perform the hybrid partitioning
    voronoi_regions = hybrid_partition(points, seeds, morphologies, temporal_motifs)

    # Print the results
    for region_index, region in voronoi_regions.items():
        print(f"Region {region_index}: {region}")


def example_3():
    # Create some example points and seeds
    points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    seeds = np.array([[2, 2, 2], [4, 4, 4]])

    # Create some example morphologies and temporal motifs
    morphologies = [Morphology(1, 2, 3, 4), Morphology(2, 3, 4, 5)]
    temporal_motifs = [TemporalMotif(('a', 'b', 'c'), 10), TemporalMotif(('d', 'e', 'f'), 20)]

    # Perform the hybrid partitioning
    voronoi_regions = hybrid_partition(points, seeds, morphologies, temporal_motifs)

    # Print the results
    for region_index, region in voronoi_regions.items():
        print(f"Region {region_index}: {region}")

if __name__ == "__main__":
    example_1()
    example_2()
    example_3()