# DARWIN HAMMER — match 2387, survivor 2
# gen: 5
# parent_a: hybrid_voronoi_partition_hybrid_hybrid_hybrid_m325_s4.py (gen4)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_hybrid_regret_m890_s0.py (gen4)
# born: 2026-05-29T23:42:04Z

"""Hybrid Voronoi‑Sheaf / Doomsday‑Gini‑Ternary Lens (VSG‑DG‑TL)

This module fuses the two parent algorithms:

* **Parent A** – Voronoi partitioning + sheaf‑based restriction maps.
* **Parent B** – Doomsday‑Gini coefficient + ternary‑lens similarity.

Mathematical bridge
------------------
Both parents operate on *distributions* over a discrete index set:

* The Voronoi diagram assigns each data point to a seed, producing a
  region size vector `n_i = |R_i|`.
* The Gini coefficient quantifies inequality of that same size vector.
* The sheaf uses the Voronoi seeds as node identifiers; restriction maps
  are linear operators between node vector spaces.
* The ternary lens creates a compact signature per region (hash‑based)
  which is then projected to a ternary vector `t_i ∈ {‑1,0,1}^k`.
* We weight each restriction map by a factor derived from the Gini
  coefficient, thereby letting the inequality of the Voronoi partition
  modulate the strength of information flow in the sheaf.

The resulting system can be used wherever a spatial partition (Voronoi)
must be coupled to a context‑sensitive algebraic structure (sheaf) while
respecting global distributional imbalance (Gini) and providing fast
similarity checks (ternary lens)."""

import math
import random
import hashlib
from typing import List, Tuple, Dict, Iterable
import numpy as np
import sys
import pathlib

# ----------------------------------------------------------------------
# Core geometric utilities (from Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the closest seed (break ties by index)."""
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Voronoi assignment – returns a mapping seed_index → list of points."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Signature / ternary utilities (from Parent B)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Min‑hash style signature of a token set (length k)."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def ternary_vector(sig: List[int]) -> List[int]:
    """Project a signature to a ternary vector using modulo‑3 mapping."""
    # Map 0 → -1, 1 → 0, 2 → +1
    mapping = {0: -1, 1: 0, 2: 1}
    return [mapping[h % 3] for h in sig]

def ternary_similarity(v1: List[int], v2: List[int]) -> float:
    """Fraction of equal components between two ternary vectors."""
    if len(v1) != len(v2):
        raise ValueError("vectors must have equal length")
    return sum(1 for a, b in zip(v1, v2) if a == b) / len(v1)

# ----------------------------------------------------------------------
# Gini coefficient (from Parent B)
# ----------------------------------------------------------------------
def gini_coefficient(values: List[float]) -> float:
    """Standard Gini coefficient for a non‑negative list."""
    if not values:
        return 0.0
    if any(v < 0 for v in values):
        raise ValueError("Gini undefined for negative values")
    sorted_vals = sorted(values)
    n = len(values)
    cumvals = np.cumsum(sorted_vals, dtype=float)
    numerator = (2 * np.arange(1, n + 1) - n - 1) * sorted_vals
    return max(0.0, numerator.sum() / (n * cumvals[-1])) if cumvals[-1] != 0 else 0.0

# ----------------------------------------------------------------------
# Sheaf data structure (from Parent A) with Gini‑scaled restrictions
# ----------------------------------------------------------------------
class Sheaf:
    """Simple sheaf where each node carries a vector space of given dimension."""

    def __init__(self, node_dims: Dict[int, int], edges: List[Tuple[int, int]]):
        self.node_dims: Dict[int, int] = dict(node_dims)
        self.edges: List[Tuple[int, int]] = list(edges)
        self._restrictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[int, np.ndarray] = {}

    def set_restriction(
        self,
        edge: Tuple[int, int],
        src_map: np.ndarray,
        dst_map: np.ndarray,
        weight: float = 1.0,
    ) -> None:
        """Assign restriction maps; optionally scale them by a weight (e.g. 1‑Gini)."""
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have equal row count")
        # Apply scalar weight to both maps
        self._restrictions[edge] = (src_map * weight, dst_map * weight)

    def get_restriction(self, edge: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        return self._restrictions[edge]

    def set_section(self, node: int, vector: np.ndarray) -> None:
        if vector.shape[0] != self.node_dims[node]:
            raise ValueError("section dimension mismatch")
        self._sections[node] = vector

    def get_section(self, node: int) -> np.ndarray:
        return self._sections[node]

# ----------------------------------------------------------------------
# Hybrid operations ----------------------------------------------------
# ----------------------------------------------------------------------
def build_region_signatures(regions: Dict[int, List[Point]], k: int = 64) -> Dict[int, List[int]]:
    """
    For each Voronoi region produce a min‑hash signature.
    Tokens are simple stringified coordinates of points belonging to the region.
    """
    sigs: Dict[int, List[int]] = {}
    for idx, pts in regions.items():
        tokens = (f"{x:.5f},{y:.5f}" for x, y in pts)
        sigs[idx] = signature(tokens, k=k)
    return sigs

def compute_region_ternary_vectors(signatures: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """Project every region signature to a ternary vector."""
    return {idx: ternary_vector(sig) for idx, sig in signatures.items()}

def construct_hybrid_sheaf(
    seeds: List[Point],
    regions: Dict[int, List[Point]],
    ternary_vecs: Dict[int, List[int]],
    gini_weight: float = None,
) -> Sheaf:
    """
    Create a Sheaf whose nodes correspond to Voronoi seeds.
    Edge set is a complete graph (for demonstration).  Each restriction map
    is a random linear operator whose magnitude is scaled by (1‑Gini) to
    encode the inequality of the underlying partition.
    """
    n = len(seeds)
    node_dims = {i: len(ternary_vecs[i]) for i in range(n)}
    edges = [(i, j) for i in range(n) for j in range(n) if i != j]

    sheaf = Sheaf(node_dims, edges)

    # Global Gini if not supplied
    if gini_weight is None:
        region_sizes = [len(regions[i]) for i in range(n)]
        gini = gini_coefficient(region_sizes)
        gini_weight = 1.0 - gini  # larger inequality → smaller weight

    rng = np.random.default_rng(42)

    for (u, v) in edges:
        dim_u, dim_v = node_dims[u], node_dims[v]
        rows = max(dim_u, dim_v)  # square-ish maps for simplicity
        src_map = rng.normal(size=(rows, dim_u))
        dst_map = rng.normal(size=(rows, dim_v))
        sheaf.set_restriction((u, v), src_map, dst_map, weight=gini_weight)

    # Initialise each node's section with its ternary vector (as a column)
    for i in range(n):
        vec = np.array(ternary_vecs[i], dtype=float).reshape(-1, 1)
        sheaf.set_section(i, vec)

    return sheaf

def hybrid_similarity(
    sheaf: Sheaf,
    node_a: int,
    node_b: int,
    ternary_vectors: Dict[int, List[int]],
) -> float:
    """
    Combine sheaf‑based restriction similarity with ternary vector similarity.
    The restriction similarity is measured by the Frobenius inner product of
    the two restriction maps associated to the directed edge (a→b).
    """
    src_map, dst_map = sheaf.get_restriction((node_a, node_b))
    # Frobenius inner product normalised
    restr_sim = np.sum(src_map * dst_map) / (np.linalg.norm(src_map) * np.linalg.norm(dst_map) + 1e-12)

    tern_sim = ternary_similarity(ternary_vectors[node_a], ternary_vectors[node_b])

    # Weighted blend: give 70% weight to restriction similarity, 30% to ternary similarity
    return 0.7 * restr_sim + 0.3 * tern_sim

# ----------------------------------------------------------------------
# Smoke test -----------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate random points and seeds
    random.seed(0)
    np.random.seed(0)
    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(500)]
    seeds = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(5)]

    # Voronoi partition
    regions = assign(points, seeds)

    # Signatures & ternary vectors
    sigs = build_region_signatures(regions, k=64)
    tern_vecs = compute_region_ternary_vectors(sigs)

    # Build the hybrid sheaf (Gini‑scaled)
    sheaf = construct_hybrid_sheaf(seeds, regions, tern_vecs)

    # Demonstrate hybrid similarity on a couple of node pairs
    for i in range(len(seeds) - 1):
        sim = hybrid_similarity(sheaf, i, i + 1, tern_vecs)
        print(f"Hybrid similarity between node {i} and {i+1}: {sim:.4f}")

    print("Smoke test completed without error.")