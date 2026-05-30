# DARWIN HAMMER — match 1375, survivor 4
# gen: 5
# parent_a: hybrid_geometric_product_voronoi_partition_m4_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hdc_se_m172_s3.py (gen4)
# born: 2026-05-29T23:35:41Z

"""Hybrid Geometric‑Voronoi ↔ Fisher‑JEPA Hyperdimensional Algorithm.

Parents:
- PARENT A: ``geometric_product_voronoi_partition`` – provides a Clifford‑algebra
  geometric product, multivector arithmetic and a Voronoi assignment based on the
  geometric distance between multivectors.
- PARENT B: ``hybrid_fisher_jepa_hdc`` – supplies a scalar Fisher score → bipolar
  hypervector mapping, hyperdimensional binding/bundling primitives and an
  energy formulation in hyperdimensional space.

Mathematical bridge:
Every multivector 𝑀 = Σ c_B B (B a basis blade) is projected onto a high‑dimensional
bipolar hypervector **h**(𝑀) ∈ {±1}^D by a deterministic hash of the blade identifier.
The sign of each hypervector component encodes the sign of the corresponding
coefficient c_B.  This projection is linear and invertible only up to the chosen
hash, but it allows us to feed Clifford‑derived quantities (geometric distances,
products) into hyperdimensional operations (binding, bundling, similarity) and
vice‑versa.  The hybrid pipeline therefore consists of:

1. Compute geometric distances between multivectors → Voronoi partition.
2. Convert the selected seed (multivector) to a hypervector.
3. Turn a scalar Fisher score F(θ) into a bipolar hypervector **ẑ**.
4. Predict the next representation by hyperdimensional binding **ẑ ⊗ h(prev)**
   and bundling with **h(prev)**.
5. Measure the JEPA‑style energy as the squared Euclidean distance between the
   true hypervector of the current multivector and the prediction.

The three public functions below illustrate these steps.
"""

import math
import random
import sys
import pathlib
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Clifford algebra core (excerpt from parent A)
# ---------------------------------------------------------------------------

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list.

    Each transposition of adjacent indices that are out of order flips the sign
    (anti‑commutativity). Duplicate indices cancel because e_i*e_i = 1.
    """
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
                # cancel the pair
                lst.pop(j)
                lst.pop(j)  # second element now occupies position j
                n -= 2
                i = -1  # restart outer loop because length changed
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    """Multiply two basis blades.

    Returns (result_blade_frozenset, sign).
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Clifford algebra element Cl(n,0) as a sparse dict of blades → coeff."""

    def __init__(self, components: Dict[frozenset, float] = None):
        self.components: Dict[frozenset, float] = {}
        if components:
            # prune zeroes
            for b, c in components.items():
                if abs(c) > 1e-12:
                    self.components[frozenset(b)] = float(c)

    @staticmethod
    def scalar(value: float) -> "Multivector":
        return Multivector({frozenset(): float(value)})

    @staticmethod
    def basis_vector(idx: int) -> "Multivector":
        return Multivector({frozenset([idx]): 1.0})

    # -----------------------------------------------------------------------
    # Algebraic operators
    # -----------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        res = self.components.copy()
        for b, c in other.components.items():
            res[b] = res.get(b, 0.0) + c
            if abs(res[b]) < 1e-12:
                del res[b]
        return Multivector(res)

    def __sub__(self, other: "Multivector") -> "Multivector":
        res = self.components.copy()
        for b, c in other.components.items():
            res[b] = res.get(b, 0.0) - c
            if abs(res[b]) < 1e-12:
                del res[b]
        return Multivector(res)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self.components.items()})

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result: Dict[frozenset, float] = {}
        for b1, c1 in self.components.items():
            for b2, c2 in other.components.items():
                b_res, sign = _multiply_blades(b1, b2)
                coeff = c1 * c2 * sign
                result[b_res] = result.get(b_res, 0.0) + coeff
        # prune
        result = {b: c for b, c in result.items() if abs(c) > 1e-12}
        return Multivector(result)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------
    def norm(self) -> float:
        """Euclidean norm of the coefficient vector."""
        return math.sqrt(sum(c * c for c in self.components.values()))

    def __repr__(self) -> str:
        if not self.components:
            return "0"
        terms = []
        for blade, coeff in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if not blade:
                term = f"{coeff:.3g}"
            else:
                indices = "*".join(str(i) for i in sorted(blade))
                term = f"{coeff:.3g}{indices}"
            terms.append(term)
        return " + ".join(terms)


# ---------------------------------------------------------------------------
# Hyperdimensional primitives (excerpt from parent B)
# ---------------------------------------------------------------------------

Vector = List[int]  # bipolar hypervector (elements are +1 or -1)


def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    """Generate a random bipolar hypervector."""
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    """Deterministic hypervector for an arbitrary symbol."""
    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Component‑wise binding (multiplication) of two hypervectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension for binding")
    return [ai * bi for ai, bi in zip(a, b)]


def bundle(vectors: Iterable[Vector]) -> Vector:
    """Bundling (majority vote) of bipolar hypervectors."""
    vectors = list(vectors)
    if not vectors:
        raise ValueError("bundle requires at least one vector")
    dim = len(vectors[0])
    sums = np.sum(np.array(vectors, dtype=np.int8), axis=0)
    # Tie → +1
    return [1 if s >= 0 else -1 for s in sums]


def hamming_distance(a: Vector, b: Vector) -> int:
    """Half the Hamming distance (since components are ±1)."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")
    return sum(1 for ai, bi in zip(a, b) if ai != bi)


# ---------------------------------------------------------------------------
# Bridge: multivector → hypervector
# ---------------------------------------------------------------------------

def mv_to_hv(mv: Multivector, dim: int = 10000) -> Vector:
    """Project a multivector onto a bipolar hypervector.

    Each blade is hashed to an index; the sign of the component encodes the sign
    of the blade coefficient.  All other components remain +1.
    """
    hv = [1] * dim
    for blade, coeff in mv.components.items():
        # deterministic hash from the sorted tuple of indices
        blade_bytes = str(sorted(blade)).encode("utf-8")
        idx = int.from_bytes(hashlib.sha256(blade_bytes).digest()[:8], "big") % dim
        hv[idx] = 1 if coeff >= 0 else -1
    return hv


# ---------------------------------------------------------------------------
# Hybrid core functions
# ---------------------------------------------------------------------------

def geometric_voronoi_assign(seeds: List[Multivector], query: Multivector) -> int:
    """Assign *query* to the nearest seed using geometric distance.

    Returns the index of the closest seed.
    """
    if not seeds:
        raise ValueError("seed list cannot be empty")
    distances = [ (seed - query).norm() for seed in seeds ]
    return int(np.argmin(distances))


def hybrid_voronoi_fisher(
    seeds: List[Multivector],
    points: List[Multivector],
    fisher_score: float,
    dim: int = 10000
) -> List[int]:
    """Hybrid operation combining Voronoi partitioning with a Fisher‑derived hypervector.

    For each *point*:
        1. Find the nearest seed via geometric Voronoi.
        2. Convert that seed to a hypervector **h_seed**.
        3. Convert the scalar *fisher_score* to a bipolar hypervector **ẑ**.
        4. Bind **ẑ** with **h_seed** and bundle the result with **h_seed**,
           producing a prediction hypervector **ĥ**.
        5. Return the index of the selected seed (the Voronoi assignment).

    The returned list has the same length as *points*.
    """
    # Step 3 – Fisher hypervector (deterministic from the score)
    fisher_hv = random_vector(dim, seed=int(fisher_score * 1e9) if fisher_score != 0 else 0)

    assignments = []
    for pt in points:
        idx = geometric_voronoi_assign(seeds, pt)
        seed_mv = seeds[idx]
        seed_hv = mv_to_hv(seed_mv, dim)
        bound = bind(fisher_hv, seed_hv)
        pred = bundle([seed_hv, bound])
        # (pred is not used further here, but the operation demonstrates the hybrid flow)
        assignments.append(idx)
    return assignments


def hybrid_predict_energy(
    prev_mv: Multivector,
    curr_mv: Multivector,
    fisher_score: float,
    dim: int = 10000
) -> float:
    """Compute JEPA‑style energy in hyperdimensional space.

    1. Encode *prev_mv* and *curr_mv* as hypervectors.
    2. Generate Fisher hypervector **ẑ** from *fisher_score*.
    3. Predict **ĥ** = bundle( h(prev), bind(ẑ, h(prev)) ).
    4. Energy = squared Euclidean distance between **ĥ** and h(curr).

    Returns the scalar energy.
    """
    h_prev = np.array(mv_to_hv(prev_mv, dim), dtype=np.int8)
    h_curr = np.array(mv_to_hv(curr_mv, dim), dtype=np.int8)
    fisher_hv = np.array(random_vector(dim, seed=int(fisher_score * 1e9) if fisher_score != 0 else 0), dtype=np.int8)

    bound = h_prev * fisher_hv
    pred = bundle([h_prev.tolist(), bound.tolist()])
    pred_arr = np.array(pred, dtype=np.int8)

    diff = pred_arr - h_curr
    return float(np.sum(diff * diff))


def hybrid_similarity(mv_a: Multivector, mv_b: Multivector, dim: int = 10000) -> float:
    """Similarity between two multivectors via their hypervector embeddings.

    Returns the normalized cosine similarity in [-1, 1].
    """
    hv_a = np.array(mv_to_hv(mv_a, dim), dtype=np.float32)
    hv_b = np.array(mv_to_hv(mv_b, dim), dtype=np.float32)
    dot = float(np.dot(hv_a, hv_b))
    norm_a = float(np.linalg.norm(hv_a))
    norm_b = float(np.linalg.norm(hv_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create a small Clifford algebra with basis vectors e0, e1, e2
    e0 = Multivector.basis_vector(0)
    e1 = Multivector.basis_vector(1)
    e2 = Multivector.basis_vector(2)

    # Seeds: random linear combinations of basis vectors
    seeds = [
        e0 + 2 * e1,
        3 * e1 - e2,
        -e0 + e2,
    ]

    # Query points
    points = [
        e0 + e1,
        2 * e1 + e2,
        -e0 - e2,
    ]

    fisher_score = 0.732  # arbitrary scalar

    assignments = hybrid_voronoi_fisher(seeds, points, fisher_score)
    print("Voronoi assignments:", assignments)

    # Energy between consecutive points
    energy = hybrid_predict_energy(seeds[assignments[0]], seeds[assignments[1]], fisher_score)
    print("Hybrid JEPA energy (first two assignments):", energy)

    # Similarity between two seeds
    sim = hybrid_similarity(seeds[0], seeds[1])
    print("Hybrid similarity between seed 0 and seed 1:", sim)