# DARWIN HAMMER — match 2702, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_sheaf__m1068_s2.py (gen4)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s2.py (gen2)
# born: 2026-05-29T23:43:55Z

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = int
FeatureVec = Tuple[float, ...]
Point = Tuple[float, float]

# ----------------------------------------------------------------------
# Parent A – Gaussian similarity utilities
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two feature vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: 64‑bit based on median threshold."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()

def gaussian_similarity_matrix(features: Dict[Node, FeatureVec],
                               epsilon: float = 1.0) -> Tuple[np.ndarray, List[Node]]:
    """
    Build a symmetric similarity matrix S where
        S[i, j] = 1 - (Hamming(phash_i, phash_j) / 64)
    and then modulate it with a Gaussian kernel based on Euclidean distance.
    Returns the matrix and the ordered list of nodes.
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)

    # Pre‑compute phashes
    phashes = {node: compute_phash(list(features[node])) for node in nodes}

    for i, ni in enumerate(nodes):
        hi = phashes[ni]
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
                continue
            hj = phashes[nj]
            d_ham = hamming_distance(hi, hj)
            base_sim = 1.0 - d_ham / 64.0

            # Gaussian modulation using Euclidean distance of feature vectors
            d_euc = euclidean(features[ni], features[nj])
            gauss = gaussian(d_euc, epsilon)

            S[i, j] = base_sim * gauss
    return S, nodes

# ----------------------------------------------------------------------
# Parent B – Voronoi partitioning utilities
# ----------------------------------------------------------------------
def distance(a: Point, b: Point) -> float:
    """2‑D Euclidean distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the nearest seed (ties broken by index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def voronoi_partition(points: List[Point], seeds: List[Point]) -> Dict[int, List[int]]:
    """
    Assign each point to the nearest seed.
    Returns a mapping seed_index → list of point indices.
    """
    regions: Dict[int, List[int]] = {i: [] for i in range(len(seeds))}
    for idx, p in enumerate(points):
        regions[nearest(p, seeds)].append(idx)
    return regions

# ----------------------------------------------------------------------
# Geometric Algebra core (Parent B)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Sort indices and compute sign of the permutation.
    Duplicate indices cancel the blade (return empty list, sign unchanged)."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel duplicate basis vectors
                del lst[j:j + 2]
                n -= 2
                i = -1  # restart outer loop
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Geometric product of two basis blades (ignoring metric)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    """Simple exterior algebra (wedge product) implementation."""
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-12}
        self.n = int(n)

    def grade(self, k: int) -> 'Multivector':
        """Extract grade‑k part."""
        return Multivector({blade: coef for blade, coef in self.components.items()
                            if len(blade) == k}, self.n)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(),
                                  key=lambda x: (len(x[0]), sorted(x[0]))):
            label = "1" if not blade else "e" + "".join(str(i) for i in sorted(blade))
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def __add__(self, other: 'Multivector') -> 'Multivector':
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __xor__(self, other: 'Multivector') -> 'Multivector':
        """Exterior (wedge) product."""
        result: Dict[FrozenSet[int], float] = {}
        for a_blade, a_coef in self.components.items():
            for b_blade, b_coef in other.components.items():
                new_blade, sign = _multiply_blades(a_blade, b_blade)
                result[new_blade] = result.get(new_blade, 0.0) + sign * a_coef * b_coef
        return Multivector(result, self.n)

    def __mul__(self, other: 'Multivector') -> 'Multivector':
        """Geometric product."""
        result = Multivector({}, self.n)
        for grade_a in range(self.n + 1):
            for grade_b in range(self.n + 1):
                grade_result = grade_a + grade_b
                if grade_result > self.n:
                    continue
                a_part = self.grade(grade_a)
                b_part = other.grade(grade_b)
                result_part = a_part ^ b_part
                result += result_part
        return result

def region_multivector_aggregation(S: np.ndarray, nodes: List[Node], 
                                    regions: Dict[int, List[int]], 
                                    epsilon: float = 1.0) -> Dict[int, Multivector]:
    """
    Aggregate weighted blades into a single Multivector for each region.

    Args:
    S (np.ndarray): The RBF similarity matrix.
    nodes (List[Node]): The ordered list of nodes.
    regions (Dict[int, List[int]]): The Voronoi partition.
    epsilon (float): The epsilon value for the Gaussian kernel.

    Returns:
    Dict[int, Multivector]: A dictionary mapping region indices to Multivectors.
    """
    multivectors = {}
    n = len(nodes)
    for region_idx, node_indices in regions.items():
        M = Multivector({}, len(nodes))
        for node_idx in node_indices:
            w = 0.0
            for other_node_idx in range(n):
                K = S[node_idx, other_node_idx]
                w += K
            w /= sum(S[other_node_idx, other_node_idx] for other_node_idx in node_indices)
            blade = frozenset([node_idx])
            M.components[blade] = M.components.get(blade, 0.0) + w
        multivectors[region_idx] = M
    return multivectors

def main():
    # Example usage
    features = {0: (1.0, 2.0), 1: (3.0, 4.0), 2: (5.0, 6.0)}
    S, nodes = gaussian_similarity_matrix(features)
    points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    seeds = [(0.0, 0.0), (10.0, 10.0)]
    regions = voronoi_partition(points, seeds)
    multivectors = region_multivector_aggregation(S, nodes, regions)
    for region_idx, M in multivectors.items():
        print(f"Region {region_idx}: {M}")

if __name__ == "__main__":
    main()