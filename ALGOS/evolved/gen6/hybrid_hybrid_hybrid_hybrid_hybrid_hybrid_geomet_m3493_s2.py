# DARWIN HAMMER — match 3493, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m602_s2.py (gen5)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s5.py (gen4)
# born: 2026-05-29T23:50:25Z

"""Hybrid Algorithm combining perceptual hashing (Parent A) with geometric algebra
multivector similarity (Parent B).

Mathematical bridge:
- Parent A produces a 64‑bit perceptual hash for a numeric sequence.
- Parent B provides a Clifford‑geometric algebra where each basis blade `e_i`
  can represent a single bit of that hash.
- By mapping a hash to a multivector whose non‑zero blades correspond to the
  positions of 1‑bits, the scalar part of the geometric product `Mv_a * Mv_b`
  equals the dot‑product of the two bit‑vectors (i.e. the number of equal
  1‑bits).  The Hamming distance follows from the identity  

      d_H = popcount(a) + popcount(b) - 2·⟨a·b⟩

  where `⟨a·b⟩` is the scalar part of the product.
- This scalar similarity is then combined with the ecological weighting
  functions from Parent A (`schoolfield_rate`, `gini_coefficient`) to build a
  weighted similarity graph.

The module therefore fuses both topologies into a single unified system.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities (hashing, distance, ecological weights)
# ----------------------------------------------------------------------
def compute_phash(values: Sequence[float]) -> int:
    """64‑bit perceptual hash based on mean comparison."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Classic bitwise Hamming distance."""
    return (a ^ b).bit_count()


def schoolfield_rate(temperature: float) -> float:
    """Logistic temperature‑performance model in (0,1)."""
    return 1.0 / (1.0 + math.exp(-(temperature - 20.0) / 5.0))


def gini_coefficient(rewards: Sequence[float]) -> float:
    """Gini coefficient of a reward batch."""
    rewards_arr = np.asarray(rewards, dtype=float)
    if rewards_arr.size == 0:
        return 0.0
    mean = rewards_arr.mean()
    if mean == 0.0:
        return 0.0
    sorted_rewards = np.sort(rewards_arr)
    n = rewards_arr.size
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * sorted_rewards)) / (n * np.sum(sorted_rewards)) - (n + 1) / n
    return float(gini)


# ----------------------------------------------------------------------
# Parent B geometric algebra core
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
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
                del lst[j : j + 2]
                n -= 2
                i = -1
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Sparse representation of a multivector in a Clifford algebra."""

    def __init__(self, components: Dict[frozenset, float] = None):
        self.components: Dict[frozenset, float] = dict(components or {})

    def __add__(self, other: "Multivector") -> "Multivector":
        res = self.components.copy()
        for k, v in other.components.items():
            res[k] = res.get(k, 0.0) + v
            if abs(res[k]) < 1e-15:
                del res[k]
        return Multivector(res)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({k: -v for k, v in self.components.items()})

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result: Dict[frozenset, float] = {}
        for ba, ca in self.components.items():
            for bb, cb in other.components.items():
                blade, sign = _multiply_blades(ba, bb)
                coeff = ca * cb * sign
                result[blade] = result.get(blade, 0.0) + coeff
        result = {k: v for k, v in result.items() if abs(v) > 1e-15}
        return Multivector(result)

    def scalar_part(self) -> float:
        """Return the scalar (grade‑0) component."""
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        return f"Multivector({self.components})"


def vector_to_mv(x: float, y: float) -> Multivector:
    """Utility converting a 2‑D vector into a multivector."""
    return Multivector({frozenset({0}): x, frozenset({1}): y})


# ----------------------------------------------------------------------
# Hybrid utilities – linking hash bits to geometric algebra
# ----------------------------------------------------------------------
def phash_to_multivector(hash_int: int, bits: int = 64) -> Multivector:
    """
    Convert a perceptual hash into a multivector.
    Each set bit creates a basis blade e_i with unit coefficient.
    """
    components: Dict[frozenset, float] = {}
    for i in range(bits):
        if (hash_int >> (bits - 1 - i)) & 1:
            components[frozenset({i})] = 1.0
    return Multivector(components)


def multivector_hamming(mv_a: Multivector, mv_b: Multivector, bits: int = 64) -> int:
    """
    Compute Hamming distance between the underlying bit‑vectors using
    the scalar part of the geometric product.
    """
    # popcounts of the original hashes
    pop_a = sum(1 for blade in mv_a.components if len(blade) == 1)
    pop_b = sum(1 for blade in mv_b.components if len(blade) == 1)
    # scalar part equals dot‑product of bit vectors
    dot = (mv_a * mv_b).scalar_part()
    return pop_a + pop_b - 2 * int(round(dot))


def hybrid_edge_weight(
    seq_a: Sequence[float],
    seq_b: Sequence[float],
    vram_a: float,
    vram_b: float,
    max_hamming: int = 4,
    bits: int = 64,
) -> float:
    """
    Produce a unified edge weight between two nodes:
      * similarity derived from hash → multivector → Hamming distance
      * ecological scaling via schoolfield_rate (temperature proxy)
      * inequality penalty via Gini of a synthetic reward batch
    The weight is in [0, 1]; zero means no edge.
    """
    # 1. perceptual hashes
    h_a = compute_phash(seq_a)
    h_b = compute_phash(seq_b)

    # 2. multivector conversion
    mv_a = phash_to_multivector(h_a, bits)
    mv_b = phash_to_multivector(h_b, bits)

    # 3. Hamming distance via GA
    ham = multivector_hamming(mv_a, mv_b, bits)

    if ham > max_hamming:
        return 0.0

    # 4. similarity factor (the closer the Hamming, the higher)
    sim = 1.0 - ham / max_hamming

    # 5. temperature factor (average of the two VRAM‑like values)
    temp_factor = schoolfield_rate((vram_a + vram_b) / 2.0)

    # 6. Gini penalty – use rewards drawn from the two sequences
    rewards = list(seq_a) + list(seq_b)
    gini = gini_coefficient(rewards)
    gini_factor = 1.0 - gini  # high inequality -> lower weight

    weight = sim * temp_factor * gini_factor
    return max(0.0, min(1.0, weight))


def hybrid_build_graph(
    elements: List[List[float]],
    vram_weights: List[float],
    max_hamming: int = 4,
    bits: int = 64,
) -> Dict[str, Dict[str, float]]:
    """
    Build an undirected similarity graph where each node corresponds to an
    element (identified by its index). Edge weights are computed by
    `hybrid_edge_weight`. The graph is represented as a nested dict:
        { "0": {"1": 0.73, "2": 0.41}, "1": {"0": 0.73}, ... }
    """
    n = len(elements)
    graph: Dict[str, Dict[str, float]] = {str(i): {} for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            w = hybrid_edge_weight(
                elements[i],
                elements[j],
                vram_weights[i],
                vram_weights[j],
                max_hamming=max_hamming,
                bits=bits,
            )
            if w > 0.0:
                graph[str(i)][str(j)] = w
                graph[str(j)][str(i)] = w
    return graph


# ----------------------------------------------------------------------
# Demonstration functions (three distinct hybrids)
# ----------------------------------------------------------------------
def demo_hash_to_mv():
    """Show conversion of a random float sequence to hash and multivector."""
    seq = [random.random() for _ in range(20)]
    h = compute_phash(seq)
    mv = phash_to_multivector(h)
    print(f"Sequence: {seq[:5]}...")
    print(f"Hash (bin): {h:064b}")
    print(f"Multivector components (first 5): {list(mv.components.items())[:5]}")


def demo_pairwise_similarity():
    """Compute and display hybrid similarity between two random vectors."""
    a = [random.uniform(-5, 5) for _ in range(15)]
    b = [random.uniform(-5, 5) for _ in range(15)]
    w = hybrid_edge_weight(a, b, vram_a=30.0, vram_b=45.0)
    print(f"Hybrid edge weight between a and b: {w:.4f}")


def demo_graph_construction():
    """Create a small graph from synthetic data and print its adjacency."""
    elems = [[random.random() for _ in range(10)] for _ in range(6)]
    vram = [random.uniform(10, 60) for _ in range(6)]
    g = hybrid_build_graph(elems, vram, max_hamming=6)
    print("Constructed graph adjacency:")
    for node, neigh in g.items():
        print(f"  {node}: {neigh}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo: Hash → Multivector ===")
    demo_hash_to_mv()
    print("\n=== Demo: Pairwise Hybrid Similarity ===")
    demo_pairwise_similarity()
    print("\n=== Demo: Graph Construction ===")
    demo_graph_construction()