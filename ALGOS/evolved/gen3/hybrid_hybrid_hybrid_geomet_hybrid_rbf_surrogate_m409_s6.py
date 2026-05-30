# DARWIN HAMMER — match 409, survivor 6
# gen: 3
# parent_a: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s3.py (gen2)
# parent_b: hybrid_rbf_surrogate_hybrid_distributed_l_m58_s0.py (gen2)
# born: 2026-05-29T23:28:56Z

"""Hybrid Voronoi‑Geometric‑Algebra & RBF‑Perceptual Algorithm

Parents:
- hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s3.py (geometric algebra,
  Voronoi partition)
- hybrid_rbf_surrogate_hybrid_distributed_l_m58_s0.py (radial‑basis‑function
  surrogate + perceptual hashing for similarity)

Mathematical bridge:
Both parents rely on a notion of distance between objects.
Parent A uses Euclidean distance to build Voronoi regions; Parent B feeds a
Euclidean distance into a Gaussian RBF to obtain a similarity matrix and
modulates it with a perceptual hash derived from feature vectors.

The hybrid therefore:
1. Builds Voronoi regions of a set of points around seed points (Parent A).
2. Represents each region as a multivector (geometric algebra) – the scalar
   part encodes the region size, the grade‑1 part encodes the centroid.
3. Computes a similarity matrix between seeds by:
   • Gaussian RBF of Euclidean distances between centroids.
   • Multiplicative modulation with a hash‑based similarity derived from the
     scalar parts of the region multivectors.
4. Uses the resulting matrix to probabilistically re‑assign points to seeds,
   yielding a stochastic, similarity‑aware Voronoi partition.

The code below implements this hybrid pipeline in pure Python with only the
standard library and NumPy.
"""

import math
import random
import sys
import pathlib
from typing import List, Tuple, Dict, Set, Hashable, Sequence
import numpy as np

# ----------------------------------------------------------------------
# Basic geometric utilities (from Parent A)
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
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Minimal geometric‑algebra implementation (excerpt from Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate basis vector → cancels (grade‑2+ becomes 0)
                lst.pop(j)
                lst.pop(j)
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    """Very small GA container supporting only addition and outer product."""
    def __init__(self, components: Dict[frozenset, float], n: int):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(),
                                  key=lambda x: (len(x[0]), sorted(x[0]))):
            label = "1" if not blade else "e" + "".join(str(i) for i in sorted(blade))
            terms.append(f"{coef:+.3g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def __add__(self, other: 'Multivector') -> 'Multivector':
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if v != 0.0}, self.n)

    def outer(self, other: 'Multivector') -> 'Multivector':
        """Exterior (wedge) product."""
        result: Dict[frozenset, float] = {}
        for ba, ca in self.components.items():
            for bb, cb in other.components.items():
                blade, sign = _multiply_blades(ba, bb)
                if blade:  # zero blade disappears
                    result[blade] = result.get(blade, 0.0) + ca * cb * sign
        return Multivector(result, self.n)

# ----------------------------------------------------------------------
# Perceptual hashing utilities (from Parent B)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """64‑bit perceptual hash of a list of floats."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def region_multivector(region: List[Point]) -> Multivector:
    """
    Build a multivector from a Voronoi region:
    - scalar part = number of points (region size)
    - grade‑1 part = centroid encoded as e0, e1
    """
    size = len(region)
    if size == 0:
        return Multivector({}, 2)
    cx = sum(p[0] for p in region) / size
    cy = sum(p[1] for p in region) / size
    comps = {
        frozenset(): float(size),                # scalar
        frozenset({0}): cx,                      # e0
        frozenset({1}): cy,                      # e1
    }
    return Multivector(comps, 2)

def rbf_similarity(centroids: List[Point], epsilon: float = 1.0) -> np.ndarray:
    """
    Gaussian RBF similarity matrix built from Euclidean distances between centroids.
    S_ij = exp(- (epsilon * ||c_i - c_j||)^2 )
    """
    n = len(centroids)
    S = np.empty((n, n), dtype=np.float64)
    for i, ci in enumerate(centroids):
        for j, cj in enumerate(centroids):
            d = distance(ci, cj)
            S[i, j] = math.exp(-((epsilon * d) ** 2))
    return S

def hybrid_similarity_matrix(
    regions: Dict[int, List[Point]],
    epsilon: float = 1.0
) -> Tuple[np.ndarray, List[int]]:
    """
    Combine two similarity notions:
    1. RBF on centroids (continuous geometry)
    2. Perceptual‑hash similarity on the scalar part of region multivectors
       (discrete, combinatorial)

    The final matrix is:
        H_ij = S_ij * (1 - hamdist(hi, hj)/64)
    where S_ij is the RBF similarity and hi is the hash of region i.
    """
    # 1. Build multivectors and extract centroids / hashes
    multivectors = {i: region_multivector(pts) for i, pts in regions.items()}
    centroids = [
        (mv.components.get(frozenset({0}), 0.0),
         mv.components.get(frozenset({1}), 0.0))
        for i, mv in sorted(multivectors.items())
    ]
    hashes = [
        compute_phash([mv.scalar_part()]) for i, mv in sorted(multivectors.items())
    ]

    # 2. RBF similarity on centroids
    S = rbf_similarity(centroids, epsilon)

    # 3. Hash‑based modulation
    n = len(centroids)
    for i in range(n):
        for j in range(i, n):
            ham = hamming_distance(hashes[i], hashes[j])
            mod = 1.0 - ham / 64.0
            H = S[i, j] * mod
            S[i, j] = H
            S[j, i] = H
    return S, list(sorted(multivectors.keys()))

def stochastic_reassignment(
    points: List[Point],
    seeds: List[Point],
    similarity: np.ndarray,
    temperature: float = 0.5
) -> Dict[int, List[Point]]:
    """
    Re‑assign points to seeds using a softmax over similarity‑weighted distances.
    For a point p and seed i:
        score_i = exp( -d(p, seed_i) / temperature ) * similarity_i
    The point is assigned to the seed with the highest score.
    """
    n_seeds = len(seeds)
    regions: Dict[int, List[Point]] = {i: [] for i in range(n_seeds)}
    # pre‑compute a per‑seed similarity weight (row‑sum of similarity matrix)
    sim_weights = similarity.sum(axis=1)  # shape (n_seeds,)

    for p in points:
        scores = np.empty(n_seeds, dtype=np.float64)
        for i, s in enumerate(seeds):
            d = distance(p, s)
            geo = math.exp(-d / temperature)
            scores[i] = geo * sim_weights[i]
        chosen = int(np.argmax(scores))
        regions[chosen].append(p)
    return regions

# ----------------------------------------------------------------------
# Public hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_voronoi_rbf(
    points: List[Point],
    seed_count: int = 5,
    epsilon: float = 1.0,
    temperature: float = 0.5,
    rng: random.Random = random.Random()
) -> Dict[int, List[Point]]:
    """
    End‑to‑end hybrid algorithm:
    1. Randomly pick `seed_count` seed points from the dataset.
    2. Deterministic Voronoi assignment (Parent A).
    3. Build similarity matrix that mixes RBF (continuous) and
       perceptual‑hash (discrete) information.
    4. Stochastically re‑assign points using the similarity matrix.
    Returns the final regions.
    """
    if seed_count <= 0 or seed_count > len(points):
        raise ValueError("seed_count must be between 1 and len(points)")

    # Step 1: initialise seeds
    seeds = rng.sample(points, seed_count)

    # Step 2: deterministic Voronoi partition
    init_regions = assign(points, seeds)

    # Step 3: similarity matrix blending RBF and hash
    sim_matrix, _ = hybrid_similarity_matrix(init_regions, epsilon)

    # Step 4: stochastic reassignment using the blended similarity
    final_regions = stochastic_reassignment(points, seeds, sim_matrix, temperature)

    return final_regions

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # generate a modest random point cloud
    rng = random.Random(42)
    pts = [(rng.uniform(-10, 10), rng.uniform(-10, 10)) for _ in range(200)]

    # run the hybrid algorithm
    result = hybrid_voronoi_rbf(
        points=pts,
        seed_count=7,
        epsilon=0.8,
        temperature=0.3,
        rng=rng
    )

    # simple verification: every point appears exactly once
    total = sum(len(v) for v in result.values())
    assert total == len(pts), "some points lost during processing"

    # print a concise summary
    for idx, region in result.items():
        print(f"Seed {idx}: {len(region)} points")
    print("Hybrid Voronoi‑RBF execution completed successfully.")