# DARWIN HAMMER — match 2702, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_sheaf__m1068_s2.py (gen4)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s2.py (gen2)
# born: 2026-05-29T23:43:55Z

"""
Hybrid Algorithm: RBF‑Sheaf × Geometric‑Algebra Fusion

Parent A (hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_sheaf__m1068_s2.py) provides:
    • Gaussian‑based similarity between node feature vectors via phash & Hamming distance.
    • A similarity matrix S ∈ ℝ^{n×n}.

Parent B (hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s2.py) provides:
    • Euclidean Voronoi partitioning of points (assign/nearest).
    • A lightweight geometric‑algebra implementation (Multivector, blade multiplication).

Mathematical Bridge
------------------
We treat each graph node as a point in feature space.  The Gaussian similarity from A
produces a kernel K_{ij}=exp(−ε²‖x_i−x_j‖²) that weights the interaction of blades
indexed by node identifiers.  The Voronoi partition from B groups nodes into regions.
For each region we aggregate the weighted blades into a single Multivector:

    M_R = Σ_{i∈R} w_i  e_{i}

where w_i = Σ_j K_{ij} / Σ_{p,q∈R} K_{pq} is the normalized Gaussian influence of node i
within its region.  This yields a hybrid representation that couples the RBF similarity
with geometric‑algebraic aggregation.

The module implements three core hybrid operations:
    1. `gaussian_similarity_matrix` – builds the RBF similarity matrix from node features.
    2. `voronoi_partition` – assigns nodes (as points) to the nearest seed.
    3. `region_multivector_aggregation` – builds a Multivector for each region using the
       Gaussian weights as coefficients.

All functionality is self‑contained, uses only the allowed standard library and numpy,
and runs a smoke test when executed as a script.
"""

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

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def build_hybrid_structure(features: Dict[Node, FeatureVec],
                           seed_points: List[Point],
                           epsilon: float = 1.0) -> Tuple[Dict[int, Multivector], np.ndarray, List[Node]]:
    """
    End‑to‑end hybrid pipeline:
        1. Compute Gaussian‑modulated similarity matrix S.
        2. Treat each feature vector as a 2‑D point (first two components) for Voronoi partition.
        3. For each region, aggregate a Multivector whose blades are node indices,
           weighted by normalized Gaussian influence within the region.
    Returns:
        region_multivectors – mapping seed_index → Multivector,
        similarity_matrix   – S from step 1,
        ordered_nodes       – list of nodes corresponding to rows/cols of S.
    """
    # 1. Similarity matrix
    S, ordered_nodes = gaussian_similarity_matrix(features, epsilon)

    # 2. Points for partitioning (fallback to first two dims)
    points: List[Point] = []
    for node in ordered_nodes:
        vec = features[node]
        if len(vec) < 2:
            raise ValueError(f"Feature vector of node {node} has fewer than 2 dimensions")
        points.append((float(vec[0]), float(vec[1])))

    # 3. Voronoi regions
    regions = voronoi_partition(points, seed_points)

    # 4. Region‑wise Multivector aggregation
    region_mvs: Dict[int, Multivector] = {}
    node_index_map = {node: idx for idx, node in enumerate(ordered_nodes)}

    for seed_idx, node_idxs in regions.items():
        if not node_idxs:
            region_mvs[seed_idx] = Multivector({}, len(ordered_nodes))
            continue

        # Compute intra‑region Gaussian weights
        # w_i = Σ_j S[i,j]  for j in same region, then normalize
        raw_weights = {}
        for i in node_idxs:
            row = S[i, node_idxs]  # slice of similarity row for region members
            raw_weights[i] = float(row.sum())
        total = sum(raw_weights.values())
        if total == 0:
            # fallback to uniform weights
            normalized = {i: 1.0 / len(node_idxs) for i in node_idxs}
        else:
            normalized = {i: w / total for i, w in raw_weights.items()}

        # Build multivector: each node contributes blade e_{node_id}
        components: Dict[FrozenSet[int], float] = {}
        for i in node_idxs:
            node_id = ordered_nodes[i]
            blade = frozenset({node_id})
            components[blade] = normalized[i]

        region_mvs[seed_idx] = Multivector(components, len(ordered_nodes))

    return region_mvs, S, ordered_nodes

def region_similarity_matrix(region_mvs: Dict[int, Multivector]) -> np.ndarray:
    """
    Compute a simple similarity matrix between region multivectors using
    the inner product defined as the sum of products of matching blade coefficients.
    """
    seeds = sorted(region_mvs.keys())
    k = len(seeds)
    R = np.zeros((k, k), dtype=np.float64)
    for i, si in enumerate(seeds):
        mi = region_mvs[si].components
        for j, sj in enumerate(seeds):
            if j < i:
                R[i, j] = R[j, i]
                continue
            mj = region_mvs[sj].components
            # inner product over matching blades
            common = set(mi.keys()) & set(mj.keys())
            val = sum(mi[b] * mj[b] for b in common)
            R[i, j] = val
    return R

def blend_regions(region_mvs: Dict[int, Multivector],
                 similarity: np.ndarray,
                 alpha: float = 0.5) -> Dict[int, Multivector]:
    """
    Produce blended multivectors for each region by mixing its own multivector
    with a weighted sum of its neighbours' multivectors.
    The weights are derived from the region similarity matrix (row‑stochastic).
    """
    seeds = sorted(region_mvs.keys())
    k = len(seeds)
    # Row‑normalize similarity to obtain mixing weights
    row_sums = similarity.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    weights = similarity / row_sums

    blended: Dict[int, Multivector] = {}
    for idx, seed in enumerate(seeds):
        self_mv = region_mvs[seed]
        mix_mv = Multivector({}, self_mv.n)
        for jdx, other_seed in enumerate(seeds):
            mix_mv = mix_mv + region_mvs[other_seed] * weights[idx, jdx]
        # Linear blend
        blended_mv = self_mv * (1 - alpha) + mix_mv * alpha
        blended[seed] = blended_mv
    return blended

# Extend Multivector with scalar multiplication for convenience
def _multivector_scalar_mul(self, scalar: float) -> 'Multivector':
    return Multivector({blade: coef * scalar for blade, coef in self.components.items()}, self.n)

Multivector.__mul__ = _multivector_scalar_mul  # type: ignore

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic graph with 6 nodes, each having a 3‑dimensional feature vector
    rng = random.Random(42)
    features: Dict[Node, FeatureVec] = {
        i: (rng.random(), rng.random(), rng.random())
        for i in range(6)
    }

    # Two seed points for Voronoi partition (use first two dimensions)
    seed_points: List[Point] = [(0.2, 0.2), (0.8, 0.8)]

    # Build hybrid structure
    region_mvs, S, nodes = build_hybrid_structure(features, seed_points, epsilon=1.5)

    print("Similarity matrix S:")
    print(S)

    print("\nRegion Multivectors:")
    for seed, mv in region_mvs.items():
        print(f"Seed {seed}: {mv}")

    # Region‑to‑region similarity
    R = region_similarity_matrix(region_mvs)
    print("\nRegion similarity matrix R:")
    print(R)

    # Blend regions
    blended = blend_regions(region_mvs, R, alpha=0.3)
    print("\nBlended Region Multivectors:")
    for seed, mv in blended.items():
        print(f"Seed {seed}: {mv}")

    sys.exit(0)