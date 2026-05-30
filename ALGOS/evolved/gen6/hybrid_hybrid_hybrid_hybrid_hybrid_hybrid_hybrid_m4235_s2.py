# DARWIN HAMMER — match 4235, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_endpoi_m1071_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hard_t_m1426_s1.py (gen5)
# born: 2026-05-29T23:54:33Z

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Physical description used by the original algorithms."""
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Clifford algebra utilities (3‑D + scalar)
# ----------------------------------------------------------------------


def _blade_mul_one_vector(blade: Tuple[int, ...], i: int) -> Tuple[Tuple[int, ...], int]:
    """
    Multiply a basis blade (sorted tuple of indices) by a single basis vector e_i.

    Returns the resulting blade (still sorted) and the sign (+1 / -1) that
    arises from anti‑commutation.  If the vector already appears in the blade,
    it squares to +1 and is removed (cancellation).
    """
    if i in blade:
        # e_i * e_i = 1  → cancel the index, sign unchanged
        new_blade = tuple(x for x in blade if x != i)
        return new_blade, 1

    # Count how many existing indices are greater than i; each such swap flips the sign
    swaps = sum(1 for x in blade if x > i)
    sign = -1 if swaps % 2 else 1
    # Insert i while preserving order
    new_blade = tuple(sorted(blade + (i,)))
    return new_blade, sign


def multiply_blades(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """
    Geometric product of two basis blades.
    The result is a blade (tuple of indices) and an overall sign.
    """
    result_blade: Tuple[int, ...] = a
    sign = 1
    for idx in b:
        result_blade, s = _blade_mul_one_vector(result_blade, idx)
        sign *= s
    return result_blade, sign


# ----------------------------------------------------------------------
# Multivector class
# ----------------------------------------------------------------------


class Multivector:
    """
    Sparse representation of a multivector in a Clifford algebra of
    dimension 4 (e0 = scalar, e1‑e4 = basis vectors).  Internally a dict
    maps a blade (sorted tuple of indices) to its scalar coefficient.
    The empty tuple () denotes the scalar part.
    """

    __slots__ = ("_terms",)

    def __init__(self, terms: Dict[Tuple[int, ...], float] | None = None):
        self._terms: Dict[Tuple[int, ...], float] = {}
        if terms:
            for blade, coeff in terms.items():
                if abs(coeff) > 1e-15:
                    self._terms[blade] = float(coeff)

    @staticmethod
    def scalar(value: float) -> "Multivector":
        return Multivector({(): value})

    @staticmethod
    def vector(index: int, value: float = 1.0) -> "Multivector":
        return Multivector({(index,): value})

    def __add__(self, other: "Multivector") -> "Multivector":
        result = Multivector(self._terms)
        for b, c in other._terms.items():
            result._terms[b] = result._terms.get(b, 0.0) + c
            if abs(result._terms[b]) < 1e-15:
                del result._terms[b]
        return result

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self._terms.items()})

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result_terms: Dict[Tuple[int, ...], float] = {}
        for ba, ca in self._terms.items():
            for bb, cb in other._terms.items():
                blade, sign = multiply_blades(ba, bb)
                coeff = ca * cb * sign
                result_terms[blade] = result_terms.get(blade, 0.0) + coeff
        # prune near‑zero terms
        result_terms = {b: c for b, c in result_terms.items() if abs(c) > 1e-15}
        return Multivector(result_terms)

    def reverse(self) -> "Multivector":
        """
        Reversion (grade involution) flips the sign of blades with grade 2 or 3.
        For a blade of grade g, the sign is (-1)^{g(g‑1)/2}.
        """
        rev_terms: Dict[Tuple[int, ...], float] = {}
        for blade, coeff in self._terms.items():
            g = len(blade)
            sign = -1 if (g * (g - 1) // 2) % 2 else 1
            rev_terms[blade] = coeff * sign
        return Multivector(rev_terms)

    def scalar_part(self) -> float:
        return self._terms.get((), 0.0)

    def norm(self) -> float:
        """
        Euclidean norm derived from the scalar part of mv * mv.reverse().
        """
        prod = self * self.reverse()
        return math.sqrt(abs(prod.scalar_part()))

    def __repr__(self) -> str:
        if not self._terms:
            return "0"
        parts = []
        for blade, coeff in sorted(self._terms.items()):
            if blade == ():
                parts.append(f"{coeff:.3g}")
            else:
                basis = "e" + "".join(str(i) for i in blade)
                parts.append(f"{coeff:.3g}{basis}")
        return " + ".join(parts)


# ----------------------------------------------------------------------
# Morphology ↔ Multivector embedding
# ----------------------------------------------------------------------


def embed_morphology(m: Morphology) -> Multivector:
    """
    Encode a Morphology as a multivector:
        length → e1, width → e2, height → e3, mass → scalar (e0)
    """
    mv = (
        Multivector.scalar(m.mass)
        + Multivector.vector(1, m.length)
        + Multivector.vector(2, m.width)
        + Multivector.vector(3, m.height)
    )
    return mv


# ----------------------------------------------------------------------
# Original domain‑specific helpers (unchanged but typed)
# ----------------------------------------------------------------------


def compute_dhash(values: List[float]) -> int:
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits


def compute_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Deeply fused operations
# ----------------------------------------------------------------------


def weighted_clifford_distance(a: Morphology, b: Morphology) -> float:
    """
    Euclidean distance in the Clifford embedding, weighted by the
    average recovery priority of the two objects.
    """
    mv_a = embed_morphology(a)
    mv_b = embed_morphology(b)
    diff = mv_a - mv_b
    base_dist = diff.norm()
    weight = 1.0 - (recovery_priority(a) + recovery_priority(b)) / 2.0
    return base_dist * weight


def nearest_hard_truth(
    query: Morphology, models: List[Morphology]
) -> Tuple[Morphology, float]:
    """
    Find the model with minimal weighted Clifford distance to *query*.
    Returns the model and the distance.
    """
    best = None
    best_dist = math.inf
    for m in models:
        d = weighted_clifford_distance(query, m)
        if d < best_dist:
            best_dist, best = d, m
    return best, best_dist


def cluster_by_perceptual_hash(
    items: List[Morphology], threshold: int = 5
) -> List[List[Morphology]]:
    """
    Simple agglomerative clustering using perceptual hashes of the
    4‑dimensional feature vector.  Two items belong to the same cluster
    if their hash Hamming distance is ≤ *threshold*.
    """
    hashes = [compute_phash([i.length, i.width, i.height, i.mass]) for i in items]
    clusters: List[List[Morphology]] = []
    assigned = [False] * len(items)

    for i, (item, h) in enumerate(zip(items, hashes)):
        if assigned[i]:
            continue
        cluster = [item]
        assigned[i] = True
        for j in range(i + 1, len(items)):
            if not assigned[j] and hamming_distance(h, hashes[j]) <= threshold:
                cluster.append(items[j])
                assigned[j] = True
        clusters.append(cluster)
    return clusters


# ----------------------------------------------------------------------
# Demonstration / test harness
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Sample morphologies (could be read from a dataset)
    samples = [
        Morphology(length=1.0, width=2.0, height=3.0, mass=10.0),
        Morphology(length=4.0, width=5.0, height=6.0, mass=20.0),
        Morphology(length=7.0, width=8.0, height=9.0, mass=30.0),
        Morphology(length=1.5, width=2.5, height=3.5, mass=12.0),
    ]

    # Hard‑truth models (e.g., pre‑computed reference objects)
    models = [
        Morphology(length=1.0, width=2.0, height=3.0, mass=10.0),
        Morphology(length=5.0, width=5.0, height=5.0, mass=25.0),
    ]

    # 1️⃣ Weighted Clifford distances
    print("Weighted Clifford distances to each model:")
    for s in samples:
        m, d = nearest_hard_truth(s, models)
        print(f"  Sample {s} → Model {m}  distance = {d:.4f}")

    # 2️⃣ Perceptual‑hash clustering
    clusters = cluster_by_perceptual_hash(samples, threshold=2)
    print("\nPerceptual‑hash clusters (threshold=2):")
    for idx, cl in enumerate(clusters, 1):
        print(f"  Cluster {idx}: {cl}")

    # 3️⃣ Direct multivector inspection (shows deeper integration)
    print("\nMultivector embeddings:")
    for s in samples:
        print(f"  {s} → {embed_morphology(s)}")