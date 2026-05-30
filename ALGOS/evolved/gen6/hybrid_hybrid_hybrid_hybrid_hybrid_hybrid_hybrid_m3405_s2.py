# DARWIN HAMMER — match 3405, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_geometric_pro_m1459_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s0.py (gen3)
# born: 2026-05-29T23:49:51Z

"""Hybrid Algorithm: Topological‑Geometric Fusion

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_geometric_pro_m1459_s1.py (Algorithm A)
- hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s0.py (Algorithm B)

Mathematical Bridge
-------------------
Algorithm A supplies *topological similarity* via perceptual hashing,
morphological descriptors and a scalar Gini‑inequality term.
Algorithm B supplies *geometric algebra* (multivectors) and *radial basis
functions* (RBF) that model perceptual similarity between points.

The fusion treats the RBF similarity `σ(i,j)` as a *modulation weight* for the
geometric product of multivectors that encode the morphological state of
graph nodes.  The Gini‑derived scalar `γ(i)` adjusts the recovery priority of
each node.  The three core functions below illustrate:

1. `similarity_matrix(points, morphologies)` – builds an RBF‑weighted
   topological similarity matrix using both Euclidean distance and a
   perceptual‑hash distance derived from the nodes’ morphology vectors.

2. `geometric_product_modulated(a, b, weight)` – computes the Clifford
   geometric product of two multivectors and scales every blade by the
   supplied similarity weight.

3. `node_recovery_priority(node, graph, morphologies)` – combines the
   Gini‑inequality of a node’s neighbourhood with its morphological hash to
   produce a scalar priority used for failure‑recovery scheduling.

The module is self‑contained and runs a smoke test when executed as a script.
"""

import math
import random
import sys
import pathlib
import numpy as np
from collections.abc import Mapping, Hashable
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Set

# ----------------------------------------------------------------------
# Basic Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]

Vector = List[int]                     # bipolar hypervector
Point = Tuple[float, float]            # 2‑D geometric point

# ----------------------------------------------------------------------
# Morphology & Hypervector utilities (from Algorithm A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def random_vector(dim: int = 1024, seed: str | int | None = None) -> Vector:
    """Generate a bipolar hypervector of ±1."""
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def morphology_to_vector(m: Morphology, dim: int = 1024) -> Vector:
    """Encode a Morphology into a deterministic hypervector."""
    # simple deterministic hash → seed
    seed = hash((round(m.length, 4), round(m.width, 4),
                 round(m.height, 4), round(m.mass, 4)))
    return random_vector(dim, seed)

def compute_phash(values: List[float]) -> int:
    """Perceptual hash of a list of floats (64‑bit)."""
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
# Geometric Algebra (simplified) – from Algorithm B
# ----------------------------------------------------------------------
class Multivector:
    """Very small Clifford algebra implementation for 3‑D space.

    Internally stored as a dict mapping blade index tuple to scalar.
    e.g. {(1,): 2.0, (2,3): -1.5, (): 0.7} where () is the scalar part.
    """
    def __init__(self, components: Dict[Tuple[int, ...], float] | None = None):
        self.comp: Dict[Tuple[int, ...], float] = {}
        if components:
            for blade, val in components.items():
                if abs(val) > 1e-12:
                    self.comp[tuple(sorted(blade))] = float(val)

    def __add__(self, other: "Multivector") -> "Multivector":
        res = Multivector(self.comp)
        for blade, val in other.comp.items():
            res.comp[blade] = res.comp.get(blade, 0.0) + val
            if abs(res.comp[blade]) < 1e-12:
                del res.comp[blade]
        return res

    def __repr__(self) -> str:
        return f"Multivector({self.comp})"

def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """Standard geometric product (no scaling)."""
    res = Multivector()
    for blade_a, val_a in a.comp.items():
        for blade_b, val_b in b.comp.items():
            # concatenate blades and sort, counting swaps for sign
            merged = list(blade_a) + list(blade_b)
            sign = 1
            # bubble‑sort to canonical order counting swaps
            for i in range(len(merged)):
                for j in range(i + 1, len(merged)):
                    if merged[i] > merged[j]:
                        merged[i], merged[j] = merged[j], merged[i]
                        sign *= -1
            # remove duplicate indices (e_i * e_i = 1)
            i = 0
            while i < len(merged) - 1:
                if merged[i] == merged[i + 1]:
                    # e_i^2 = 1 removes the pair
                    del merged[i:i + 2]
                    # no sign change for removal of a pair
                else:
                    i += 1
            blade = tuple(merged)
            res.comp[blade] = res.comp.get(blade, 0.0) + sign * val_a * val_b
    return res

def geometric_product_modulated(a: Multivector, b: Multivector, weight: float) -> Multivector:
    """Geometric product where each resulting blade is scaled by `weight`."""
    base = geometric_product(a, b)
    scaled = Multivector({blade: val * weight for blade, val in base.comp.items()})
    return scaled

# ----------------------------------------------------------------------
# Radial Basis Function similarity (Algorithm B) combined with
# topological (hash) similarity (Algorithm A)
# ----------------------------------------------------------------------
def euclidean(p: Point, q: Point) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])

def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def similarity_matrix(
    points: List[Point],
    morphologies: List[Morphology],
    epsilon: float = 1.0,
    hash_weight: float = 0.3,
) -> np.ndarray:
    """Hybrid similarity matrix.

    - Euclidean distance is turned into an RBF similarity.
    - Perceptual hash distance of morphology vectors is turned into a
      similarity term `h = 1 - (hamming / bits)`.
    - Final entry:  σ = (1‑hash_weight) * rbf + hash_weight * h
    """
    n = len(points)
    if n != len(morphologies):
        raise ValueError("points and morphologies must have same length")
    S = np.empty((n, n), dtype=np.float64)

    # pre‑compute hashes for speed
    morph_hashes = [
        compute_phash(morphology_to_vector(m)[:64]) for m in morphologies
    ]
    bits = 64

    for i in range(n):
        for j in range(i, n):
            # geometric part
            d = euclidean(points[i], points[j])
            rbf = gaussian_rbf(d, epsilon)

            # topological part
            ham = hamming_distance(morph_hashes[i], morph_hashes[j])
            h_sim = 1.0 - ham / bits

            sigma = (1.0 - hash_weight) * rbf + hash_weight * h_sim
            S[i, j] = sigma
            S[j, i] = sigma
    return S

# ----------------------------------------------------------------------
# Gini‑based inequality and node priority (Algorithm A)
# ----------------------------------------------------------------------
def gini_coefficient(values: List[float]) -> float:
    """Standard Gini coefficient for a list of non‑negative numbers."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(values)
    cumulative = 0.0
    for i, val in enumerate(sorted_vals, 1):
        cumulative += i * val
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    gini = (2 * cumulative) / (n * total) - (n + 1) / n
    return gini

def node_recovery_priority(
    node: Node,
    graph: Graph,
    morphologies: Mapping[Node, Morphology],
    alpha: float = 0.6,
) -> float:
    """Combine neighbourhood Gini inequality with morphological hash similarity.

    Returns a scalar where larger values indicate higher recovery priority.
    """
    neighbors = graph.get(node, set())
    if not neighbors:
        return 0.0

    # Gini of neighbor degrees
    neighbor_degrees = [len(graph.get(nb, set())) for nb in neighbors]
    gini = gini_coefficient(neighbor_degrees)

    # Morphological hash similarity to the average neighbour morphology
    node_vec = morphology_to_vector(morphologies[node])[:64]
    node_hash = compute_phash(node_vec)

    neigh_hashes = [
        compute_phash(morphology_to_vector(morphologies[nb])[:64]) for nb in neighbors
    ]
    avg_hamming = sum(
        hamming_distance(node_hash, nh) for nh in neigh_hashes
    ) / len(neigh_hashes)
    hash_sim = 1.0 - avg_hamming / 64

    # Weighted blend
    priority = alpha * gini + (1.0 - alpha) * hash_sim
    return priority

# ----------------------------------------------------------------------
# Example hybrid operation: build multivectors from morphology,
# combine via modulated geometric product using similarity matrix.
# ----------------------------------------------------------------------
def morphology_to_multivector(m: Morphology) -> Multivector:
    """Map a 4‑component morphology to a simple 3‑D multivector.

    - scalar part = mass
    - vector part (e1, e2, e3) = length, width, height
    """
    components = {
        (): m.mass,                     # scalar
        (1,): m.length,
        (2,): m.width,
        (3,): m.height,
    }
    return Multivector(components)

def hybrid_product_between_nodes(
    i: int,
    j: int,
    points: List[Point],
    morphologies: List[Morphology],
    epsilon: float = 1.0,
) -> Multivector:
    """Compute a similarity‑weighted geometric product for node i and j."""
    S = similarity_matrix(points, morphologies, epsilon=epsilon, hash_weight=0.3)
    weight = float(S[i, j])
    a = morphology_to_multivector(morphologies[i])
    b = morphology_to_multivector(morphologies[j])
    return geometric_product_modulated(a, b, weight)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic data
    points = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.866)]
    morphs = [
        Morphology(1.0, 0.5, 0.2, 2.0),
        Morphology(0.9, 0.55, 0.25, 1.8),
        Morphology(1.1, 0.48, 0.22, 2.1),
    ]

    # Build similarity matrix and display
    S = similarity_matrix(points, morphs, epsilon=2.0, hash_weight=0.4)
    print("Hybrid similarity matrix:\n", S)

    # Compute a modulated product between node 0 and 1
    prod = hybrid_product_between_nodes(0, 1, points, morphs, epsilon=2.0)
    print("\nModulated geometric product (0 ↔ 1):", prod)

    # Create a tiny graph for priority computation
    graph = {
        "A": {"B", "C"},
        "B": {"A"},
        "C": {"A"},
    }
    node_morph_map = {"A": morphs[0], "B": morphs[1], "C": morphs[2]}

    for n in graph:
        prio = node_recovery_priority(n, graph, node_morph_map, alpha=0.7)
        print(f"\nRecovery priority for node {n}: {prio:.4f}")

    print("\nSmoke test completed successfully.")